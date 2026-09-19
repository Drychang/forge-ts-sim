#!/usr/bin/env bash
# x4: recover from the OOM kill, and settle the question the modality table cannot answer
# on its own -- whether the state input is genuinely harmful, or merely fed values it was
# never trained on.
#
# Established before writing this:
#   training  TB_RANDOMIZE_NOISE=1 -> per-episode std ~ U[0, 5mm], offset ~ N(0, std)
#   eval      TB_RANDOMIZE_NOISE=0 -> fixed std = --fixed_pos_noise_mm
# so an eval at 20mm runs the state channel at four times the widest std ever trained, and
# pushes obs[0:3] 5.4-5.9 sigma off its training mean. Vision never leaves distribution --
# a displaced fixture is just a fixture somewhere else -- which is exactly the asymmetry
# that would make dropping state look like an improvement without state being useless.
#
# PHASE A settles it cheaply. The SAME full model, trained with state, evaluated with the
# state vector zeroed at test time. No retraining, so any difference is attributable to the
# input alone:
#   A ~= full          -> the model barely uses state anywhere
#   A >  full at 20mm  -> the state input is actively harmful once out of distribution
#   A << full at 0-5mm -> the model does rely on state in distribution, and the crossover
#                         is an out-of-distribution effect rather than state being useless
#
# What went wrong in x3: three camera trainings at once exhausted host RAM (62 GB total)
# and the OOM killer took two of them. run_seed_earlystop.py still wrote a first-epoch
# best.pt, so an existence check called it a success. Fixed here by gating on the training
# actually reaching early stop, and by capping concurrent trainings on free RAM -- the
# guard run_bp2_student.sh already had and this script's predecessor dropped.
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x4_master.log
QDIR=$LOGD/x4_queue
CLAIMS=$QDIR/claims
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-25}"
GPUS="${GPUS:-0 1 2}"

mkdir -p "$LOGD" "$QDIR"
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
declare -A ENVNAME=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0
                     [gear]=Isaac-Forge-GearMesh-TBCamera-v0
                     [nut]=Isaac-Forge-NutThread-TBCamera-v0 )

log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }

free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
             | tr -d ' ' | awk -F, '{print $2-$1}'; }
free_ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }

wait_for_gpu(){
    local G="$1" f
    while true; do
        f=$(free_on "$G")
        [ "${f:-0}" -ge "$MIN_FREE_MIB" ] && return 0
        log "gpu $G: ${f} MiB free < $MIN_FREE_MIB, waiting"; sleep 300
    done
}

# Host RAM, not GPU memory, is what the OOM killer acted on. Camera training holds the
# decoded image batches in host memory, so two concurrent runs are near the limit on a
# 62 GB box and three are over it.
wait_for_ram(){
    local r
    while true; do
        r=$(free_ram_gb)
        [ "${r:-0}" -ge "$MIN_FREE_RAM_GB" ] && return 0
        log "host RAM ${r} GB available < ${MIN_FREE_RAM_GB}, waiting"; sleep 300
    done
}

# $1 gpu  $2 task  $3 ckptdir  $4 ablate  $5 noise  $6 tag
do_eval(){
    local G="$1" T="$2" CK="$3" ABL="$4" NM="$5" TAG="$6" rc
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
    rc=$?
    log "EVAL $TAG rc=$rc $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
}

# A finished training is one that reached early stop or ran out of epochs. The OOM kills
# left "reason=process_ended rc=-9" behind together with a perfectly loadable first-epoch
# checkpoint, which is why existence alone is not a completion signal.
train_ok(){
    local OUT="$1"
    [ -f "$OUT/best.pt" ] || return 1
    grep -q "RUN_SEED_EARLYSTOP_DONE reason=\(early_stop\|max_epochs\)" "$OUT/train.log" 2>/dev/null
}

# $1 gpu  $2 task  $3 seed  $4 outdir  $5 logtag
do_train(){
    local G="$1" T="$2" S="$3" OUT="$4" LT="$5" attempt rc ep
    train_ok "$OUT" && { log "SKIP TRAIN $LT (already complete)"; return 0; }
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
        wait_for_ram
        wait_for_gpu "$G"
        log "TRAIN $LT attempt $attempt gpu=$G ram=$(free_ram_gb)GB"
        "$PY" -u "$SRC/student/run_seed_earlystop.py" \
            --task "$T" --seed "$S" --data_root "$DATA" --out_dir "$OUT" --gpu "$G" \
            --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
            --ablate state > "$LOGD/x4_train_${LT}.log" 2>&1
        rc=$?
        ep=$(grep -cE "^epoch " "$OUT/train.log" 2>/dev/null || echo 0)
        if train_ok "$OUT"; then
            log "TRAIN $LT OK attempt $attempt (${ep} epochs, rc=$rc)"; return 0
        fi
        log "TRAIN $LT attempt $attempt FAILED (${ep} epochs, rc=$rc): $(grep -oE 'reason=\S+ ?\S*' "$OUT/train.log" 2>/dev/null | tail -1)"
        # A first-epoch checkpoint from a killed run must not be mistaken for a result.
        mv "$OUT/best.pt" "$OUT/best.pt.failed_attempt$attempt" 2>/dev/null
        sleep 300
    done
    log "ABORT $LT after three attempts"; return 1
}

run_phase(){
    local PF="$1" name="$2" nw="$3" G pid pids=() used=0
    log "======== phase $name: $(wc -l < "$PF") jobs, $nw workers ========"
    for G in $GPUS; do
        used=$((used+1)); [ "$used" -gt "$nw" ] && break
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

log "==================== x4 start ===================="
log "waiting for the gear trainings x3 left running"
while pgrep -u "$(whoami)" -f "train_student_b[c].py" > /dev/null; do sleep 180; done
log "gear trainings finished:"
for S in 1 2; do
    d=$CKPT_ROOT/gear/ablate_state_s$S
    log "  gear s$S: $(grep -cE '^epoch ' "$d/train.log" 2>/dev/null) epochs, $(grep -oE 'reason=\S+' "$d/train.log" 2>/dev/null | tail -1), best.pt=$([ -f "$d/best.pt" ] && echo yes || echo NO)"
done

# ---------------------------------------------------------------- phase A
# Trained with state, evaluated without it. Same weights as the full model.
PA=$QDIR/phaseA; : > "$PA"
for S in 0 1 2; do
  for NM in 0 1 2.5 5 7.5 10 15 20; do
    echo "EVAL peg $CKPT_ROOT/peg/gate2_seed$S state $NM x4_dropstate_s${S}_n${NM}" >> "$PA"
  done
done
run_phase "$PA" "A (state dropped at test time only)" 3

# ---------------------------------------------------------------- phase B
PB=$QDIR/phaseB; : > "$PB"
for S in 1 2; do
  echo "TRAIN nut $S $CKPT_ROOT/nut/ablate_state_s$S nut_state_s$S" >> "$PB"
done
run_phase "$PB" "B (nut -state seeds 1,2 retrain)" 2      # 2 workers: host RAM, not GPU

# ---------------------------------------------------------------- phase C
PC=$QDIR/phaseC; : > "$PC"
for T in gear nut; do
  for S in 1 2; do
    for NM in 0 1 2.5 5 7.5 10 15 20; do
      echo "EVAL $T $CKPT_ROOT/$T/ablate_state_s$S state $NM x3_state_${T}_s${S}_n${NM}" >> "$PC"
    done
  done
done
run_phase "$PC" "C (gear/nut -state seeds 1,2 eval)" 3

log "==================== X4_ALL_DONE ===================="
