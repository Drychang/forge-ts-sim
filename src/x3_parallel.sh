#!/usr/bin/env bash
# x3: finish the matrix. Three tasks x two arms x three seeds x eight noise levels.
#
# What is missing after x2b: gear and nut have one seed per arm, so their high-noise
# advantage (+7.42 gear, +16.41 nut at 20mm) cannot be separated from seed variance --
# and peg's own full-model spread at 15mm was 6.90 points, which is larger than several
# of the effects being claimed. Peg is done with three seeds on both arms; this brings
# gear and nut to the same standing.
#
# All three GPUs are free for the first time in this study, so this runs three workers
# instead of the strictly serial loop the earlier scripts used. Workers claim jobs with
# mkdir, which is atomic on a local filesystem -- two workers cannot take the same job.
#
# Phases are barriers: every phase-1 eval finishes before any phase-2 training starts,
# because phase 3 evaluates what phase 2 produces.
#   1  gear+nut FULL seeds 0,1,2 at 7.5/10/15/20        24 evals, no training
#   2  gear+nut -state seeds 1,2                         4 trainings
#   3  those four models at all eight noise levels      32 evals
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x3_master.log
QDIR=$LOGD/x3_queue
CLAIMS=$QDIR/claims
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"
GPUS="${GPUS:-0 1 2}"

mkdir -p "$LOGD" "$QDIR"
# Claims are per-run, not per-job-completion. A job that was claimed and then failed must be
# retryable on the next launch; the do_eval / do_train skip checks are what make already
# finished work free, so clearing this costs nothing and un-strands failures.
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
declare -A ENVNAME=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0
                     [gear]=Isaac-Forge-GearMesh-TBCamera-v0
                     [nut]=Isaac-Forge-NutThread-TBCamera-v0 )

log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }

free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
             | tr -d ' ' | awk -F, '{print $2-$1}'; }

# Hold until this GPU has room. Only ever reads memory -- other users' processes are never
# signalled, so a busy card means waiting, not competing.
wait_for_gpu(){
    local G="$1" f
    while true; do
        f=$(free_on "$G")
        [ "${f:-0}" -ge "$MIN_FREE_MIB" ] && return 0
        log "gpu $G: ${f} MiB free < $MIN_FREE_MIB, waiting"
        sleep 300
    done
}

# $1 gpu  $2 task  $3 ckptdir  $4 ablate  $5 noise  $6 tag
do_eval(){
    local G="$1" T="$2" CK="$3" ABL="$4" NM="$5" TAG="$6"
    local LG="$LOGD/${TAG}.log"
    grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; return 0; }
    [ -f "$CK/best.pt" ] || { log "SKIP $TAG -- no best.pt in $CK"; return 1; }
    wait_for_gpu "$G"
    log "EVAL $TAG gpu=$G"
    CUDA_VISIBLE_DEVICES=$G "$PY" "$SRC/student/eval_ablate_student.py" --headless \
        --task "${ENVNAME[$T]}" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm "$NM" --dyn_rand on \
        --ablate "$ABL" --tag "$TAG" > "$LG" 2>&1
    log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
}

# $1 gpu  $2 task  $3 seed  $4 outdir  $5 logtag
do_train(){
    local G="$1" T="$2" S="$3" OUT="$4" LT="$5" attempt rc f
    [ -f "$OUT/best.pt" ] && { log "SKIP TRAIN $LT (done)"; return 0; }
    mkdir -p "$OUT"
    # Inherit the canonical norm_stats. compute_norm_stats() is a 2000-sample estimate whose
    # index->sample mapping depends on shard load order, so a fresh one differs by ~0.14 on
    # some channels -- enough to z-score the inputs differently from every cell it is
    # compared against.
    if [ ! -f "$OUT/norm_stats.npz" ]; then
        cp "$CKPT_ROOT/$T/norm_stats.npz" "$OUT/norm_stats.npz" \
          || { log "ABORT $LT: no canonical norm_stats"; return 1; }
    fi
    for attempt in 1 2 3; do
        wait_for_gpu "$G"
        f=$(free_on "$G")
        log "TRAIN $LT attempt $attempt gpu=$G free=${f}MiB"
        "$PY" -u "$SRC/student/run_seed_earlystop.py" \
            --task "$T" --seed "$S" --data_root "$DATA" --out_dir "$OUT" --gpu "$G" \
            --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
            --ablate state > "$LOGD/x3_train_${LT}.log" 2>&1
        rc=$?
        # run_seed_earlystop.py returned rc=0 for the OOM that produced no checkpoint on the
        # nut run, so best.pt is the only trustworthy signal.
        [ -f "$OUT/best.pt" ] && { log "TRAIN $LT OK attempt $attempt (rc=$rc)"; return 0; }
        log "TRAIN $LT attempt $attempt FAILED (rc=$rc, no best.pt)"
        grep -oE "OutOfMemoryError.*" "$LOGD/x3_train_${LT}.log" | head -1 >> "$MASTER"
        sleep 300
    done
    log "ABORT $LT after three attempts"; return 1
}

# Run one phase file with one worker per GPU. mkdir is the claim: it fails for everyone
# after the first, so a job runs exactly once even though nothing is coordinating.
run_phase(){
    local PF="$1" name="$2" G pid pids=()
    log "======== phase $name: $(wc -l < "$PF") jobs, workers on: $GPUS ========"
    for G in $GPUS; do
        (
            n=0
            while IFS= read -r line; do
                n=$((n+1))
                mkdir "$CLAIMS/$(basename "$PF").$n" 2>/dev/null || continue
                set -- $line
                case "$1" in
                    EVAL)  do_eval  "$G" "$2" "$3" "$4" "$5" "$6" ;;
                    TRAIN) do_train "$G" "$2" "$3" "$4" "$5" ;;
                esac
            done < "$PF"
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    log "======== phase $name done ========"
}

# ------------------------------------------------------------------ phase 1
P1=$QDIR/phase1
: > "$P1"
for T in gear nut; do
  for S in 0 1 2; do
    for NM in 7.5 10 15 20; do
      echo "EVAL $T $CKPT_ROOT/$T/gate2_seed$S none $NM x3_full_${T}_s${S}_n${NM}" >> "$P1"
    done
  done
done

# ------------------------------------------------------------------ phase 2
P2=$QDIR/phase2
: > "$P2"
for T in gear nut; do
  for S in 1 2; do
    echo "TRAIN $T $S $CKPT_ROOT/$T/ablate_state_s$S ${T}_state_s$S" >> "$P2"
  done
done

# ------------------------------------------------------------------ phase 3
P3=$QDIR/phase3
: > "$P3"
for T in gear nut; do
  for S in 1 2; do
    for NM in 0 1 2.5 5 7.5 10 15 20; do
      echo "EVAL $T $CKPT_ROOT/$T/ablate_state_s$S state $NM x3_state_${T}_s${S}_n${NM}" >> "$P3"
    done
  done
done

log "==================== x3 start ===================="
run_phase "$P1" 1
run_phase "$P2" 2
run_phase "$P3" 3
log "==================== X3_ALL_DONE ===================="
