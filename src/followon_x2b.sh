#!/usr/bin/env bash
# x2 follow-on, revision b. Same plan as followon_x2.sh plus the nut recovery that the
# first run needed and could not do.
#
# What went wrong on the first pass: nut's state-ablated training was placed on a GPU that
# passed the free-memory check, then another process allocated 8.71 GiB while our run was
# starting, and it OOMed three minutes in. run_seed_earlystop.py returned rc=0 anyway, so
# only the best.pt gate caught it -- the nut row of the modality table is empty.
#
# Fixes here:
#   * re-verify headroom immediately before launch, not just when picking
#   * retry a failed training up to three times instead of abandoning the task
#   * expandable_segments, so fragmentation is not what pushes a marginal run over
#   * nut recovery runs BEFORE the peg seed sweep: a missing task is a hole in the main
#     claim, while peg seeds 1-2 only tighten an interval that already exists
#
# Every completed cell is skipped on sight, so relaunching costs nothing but the eval that
# was in flight when the previous revision was stopped.
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x2_master.log
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"

mkdir -p "$LOGD"
declare -A ENVNAME=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0
                     [gear]=Isaac-Forge-GearMesh-TBCamera-v0
                     [nut]=Isaac-Forge-NutThread-TBCamera-v0 )

log(){ echo "[$(date -Is)] $*" | tee -a "$MASTER"; }

free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
             | tr -d ' ' | awk -F, '{print $2-$1}'; }

# Counts free memory only. Other users' processes are never signalled: frank's VLLM holds
# GPU0/1 and this waits rather than competing for them.
pick_gpu(){
    while true; do
        local best=-1 bestfree=-1 idx used total free
        while IFS=, read -r idx used total; do
            free=$(( total - used ))
            if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$idx; fi
        done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
                            --format=csv,noheader,nounits | tr -d ' ')
        if [ "$bestfree" -ge "$MIN_FREE_MIB" ]; then echo "$best"; return 0; fi
        log "waiting for a GPU: best free ${bestfree} MiB < ${MIN_FREE_MIB}" >&2
        sleep 600
    done
}

# Bracketed so the pattern cannot match this script's own pgrep.
wait_for_my_jobs(){
    while [ "$(pgrep -fc 'train_student_b[c].py|eval_ablate_studen[t].py|eval_frozen_studen[t].py' || true)" -gt 0 ]; do
        sleep 120
    done
}

# $1 task  $2 ckpt dir  $3 ablate  $4 noise  $5 tag
run_eval(){
    local T="$1" CK="$2" ABL="$3" NM="$4" TAG="$5"
    local LG="$LOGD/${TAG}.log"
    if grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null; then
        log "SKIP  $TAG (already done)"; return 0
    fi
    [ -f "$CK/best.pt" ] || { log "SKIP  $TAG -- no best.pt in $CK"; return 1; }
    wait_for_my_jobs
    local G; G=$(pick_gpu)
    log "EVAL  $TAG  gpu=$G"
    CUDA_VISIBLE_DEVICES=$G "$PY" "$SRC/student/eval_ablate_student.py" --headless \
        --task "${ENVNAME[$T]}" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm "$NM" --dyn_rand on \
        --ablate "$ABL" --tag "$TAG" > "$LG" 2>&1
    local rc=$?
    local sr; sr=$(grep -oE "sr=[0-9.]+" "$LG" | tail -1)
    log "EVAL  $TAG  rc=$rc  $sr"
    [ "$rc" -eq 0 ]
}

# $1 task  $2 seed  $3 out dir  $4 log tag
train_with_retry(){
    local T="$1" S="$2" OUT="$3" LT="$4" attempt G freenow rc
    [ -f "$OUT/best.pt" ] && { log "TRAIN $LT already done"; return 0; }
    mkdir -p "$OUT"
    # Inherit the canonical norm_stats. compute_norm_stats() is a 2000-sample estimate whose
    # index->sample mapping depends on shard load order, so a fresh one differs by ~0.14 on
    # some channels -- enough to z-score the inputs differently from every cell it is
    # compared against.
    if [ ! -f "$OUT/norm_stats.npz" ]; then
        cp "$CKPT_ROOT/$T/norm_stats.npz" "$OUT/norm_stats.npz" \
          || { log "ABORT $LT: no canonical norm_stats at $CKPT_ROOT/$T/norm_stats.npz"; return 1; }
        log "$LT: inherited canonical norm_stats"
    fi
    for attempt in 1 2 3; do
        wait_for_my_jobs
        G=$(pick_gpu)
        # The previous run died here: pick_gpu saw headroom, another process took it during
        # startup, and the OOM arrived three minutes in. Re-reading immediately before launch
        # does not close the window, but it does catch the case where it has already closed.
        freenow=$(free_on "$G")
        if [ "${freenow:-0}" -lt "$MIN_FREE_MIB" ]; then
            log "TRAIN $LT attempt $attempt: gpu $G lost headroom (${freenow} MiB) between pick and launch; waiting"
            sleep 600; continue
        fi
        log "TRAIN $LT attempt $attempt  gpu=$G  free=${freenow} MiB"
        "$PY" -u "$SRC/student/run_seed_earlystop.py" \
            --task "$T" --seed "$S" --data_root "$DATA" --out_dir "$OUT" --gpu "$G" \
            --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
            --ablate state > "$LOGD/x2_train_${LT}.log" 2>&1
        rc=$?
        # run_seed_earlystop.py returned rc=0 for the OOM that produced no checkpoint, so the
        # only trustworthy signal is whether best.pt exists.
        if [ -f "$OUT/best.pt" ]; then
            log "TRAIN $LT OK on attempt $attempt (wrapper rc=$rc)"; return 0
        fi
        log "TRAIN $LT attempt $attempt FAILED (wrapper rc=$rc, no best.pt)"
        grep -oE "OutOfMemoryError.*" "$LOGD/x2_train_${LT}.log" | head -1 | sed 's/^/    /' | tee -a "$MASTER"
        sleep 300
    done
    log "ABORT $LT: three attempts produced no best.pt"
    tail -20 "$LOGD/x2_train_${LT}.log" | sed 's/^/    /' | tee -a "$MASTER"
    return 1
}

log "==================== x2b follow-on start ===================="

# The previous revision's parent was stopped while one eval was still running, and that eval
# was left alive to finish rather than throwing away the time it had already spent. Wait for
# it here: the skip check reads the log file, and the log only says EVAL_SUMMARY once the
# process exits. Without this the cell looks unfinished and gets run a second time.
log "waiting for any eval left over from the previous revision"
wait_for_my_jobs
log "clear; proceeding"

# --------------------------------------------------------------- stage 0/1: already done
log "--- stage 0: code-path equivalence ---"
run_eval peg "$CKPT_ROOT/peg/gate2_seed0" none 15 "x2_full_s0_n15"
log "--- stage 1: full-model seed spread ---"
for S in 1 2; do for NM in 7.5 10 15 20; do
    run_eval peg "$CKPT_ROOT/peg/gate2_seed$S" none "$NM" "x2_full_s${S}_n${NM}"
done; done
for NM in 7.5 10 20; do
    run_eval peg "$CKPT_ROOT/peg/gate2_seed0" none "$NM" "x2_full_s0_n${NM}"
done

# --------------------------------------------------------------- stage 2: gear
log "--- stage 2: gear state-ablated breaking point ---"
for NM in 7.5 10 15 20; do
    run_eval gear "$CKPT_ROOT/gear/ablate_state" state "$NM" "x2_state_gear_s0_n${NM}"
done

# --------------------------------------------------------------- stage 2b: nut recovery
log "--- stage 2b: nut state-ablated, RECOVERY from the OOM ---"
if train_with_retry nut 0 "$CKPT_ROOT/nut/ablate_state" "nut_state_s0"; then
    for NM in 0 1 2.5 5 7.5 10 15 20; do
        run_eval nut "$CKPT_ROOT/nut/ablate_state" state "$NM" "x2_state_nut_s0_n${NM}"
    done
else
    log "nut recovery failed; the nut row of the modality table stays empty"
fi

# --------------------------------------------------------------- stage 3: peg seeds 1,2
log "--- stage 3: peg state-ablated seeds 1,2 ---"
for S in 1 2; do
    OUT="$CKPT_ROOT/peg/ablate_state_s$S"
    if train_with_retry peg "$S" "$OUT" "peg_state_s$S"; then
        for NM in 0 1 2.5 5 7.5 10 15 20; do
            run_eval peg "$OUT" state "$NM" "x2_state_s${S}_n${NM}"
        done
    fi
done

log "==================== X2B_ALL_DONE ===================="
