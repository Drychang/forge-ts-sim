#!/usr/bin/env bash
# T3 modality study, the missing cell: train and evaluate with the STATE input removed.
#
# The existing table already has state+vision (ablate_force), state+force (ablate_vision)
# and state-only (ablate_both). This adds vision+force, the cell that answers whether the
# 24-dim state is needed at all -- and, because the injected pose noise reaches the
# observation ONLY through state[0:3], whether removing it helps at high noise.
#
# Removing state does NOT remove the noise: fixed_pos_action_frame is still displaced by
# init_fixed_pos_obs_noise, so the policy has to infer the displacement visually.
#
#   TASKS=peg ./run_t3_state.sh state    # vision + force, peg only
#   ./run_t3_state.sh state              # all three tasks
#   ./run_t3_state.sh state_force        # vision only
#
# v2: the first version invoked isaaclab.sh, which cannot find python in a
# non-interactive ssh shell. The working scripts (run_aug_seed_one.sh) use the conda
# python directly for BOTH training and eval -- isaacsim is a pip package in that env.
# It also had no failure gate, so three tasks "completed" in zero seconds and printed
# ALL_DONE after every single step had failed. Both fixed.
set -uo pipefail

ABL="${1:-state}"
TASKS="${TASKS:-peg gear nut}"
NOISES="${NOISES:-0 1 2.5 5}"

# Isaac Sim prompts for the Omniverse EULA on first run and reads stdin for the answer.
# In a non-interactive ssh shell that is EOF, and every eval dies with
#   "Unable to bootstrap inner kit kernel: EOF when reading a line".
# Every working script in this repo exports this; the first version of this one did not.
export OMNI_KIT_ACCEPT_EULA=Y

PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
# Measured, not guessed: batch_size 256 with dual 256x256 backbones took 20.3 GiB on the
# peg run (GPU2 went 470 -> 20758 MiB). The first version guessed 9800 and only survived
# because GPU2 happened to be empty; on a half-used card it would have OOMed.
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"

mkdir -p "$LOGD"
declare -A ENVNAME=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0
                     [gear]=Isaac-Forge-GearMesh-TBCamera-v0
                     [nut]=Isaac-Forge-NutThread-TBCamera-v0 )

log () { echo "[$(date +'%F %T')] $*"; }

# --- shared machine: take a GPU only when one is genuinely free -------------------------
# frank / yanhong / oyea are on this box. Never evict them; wait instead.
pick_gpu () {
    while true; do
        local best=-1 bestfree=0 idx used total free
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

# Bracketed so the pattern cannot match this script's own pgrep -- that self-match once
# phase-locked three guards against each other for hours.
wait_for_my_jobs () {
    while [ "$(pgrep -fc 'train_student_b[c].py|eval_ablate_studen[t].py' || true)" -gt 0 ]; do
        sleep 120
    done
}

FAILED=0

for T in $TASKS; do
    OUT="$CKPT_ROOT/$T/ablate_$ABL"
    mkdir -p "$OUT"

    # Inherit the canonical norm_stats rather than recomputing. compute_norm_stats() is a
    # 2000-sample estimate whose index->sample mapping depends on shard load order, so a
    # fresh run differs by ~0.14 on some channels -- enough to z-score the inputs
    # differently from the 27 cells this variant is compared against. Every existing
    # ablate_* dir holds byte-identical stats, so they inherited it too, and
    # train_student_bc.py reuses out_dir/norm_stats.npz when it is already there.
    if [ ! -f "$OUT/norm_stats.npz" ]; then
        cp "$CKPT_ROOT/$T/norm_stats.npz" "$OUT/norm_stats.npz" || {
            log "ABORT $T: no canonical norm_stats at $CKPT_ROOT/$T/norm_stats.npz"; FAILED=1; continue; }
        log "$T: inherited canonical norm_stats"
    fi

    # ---------------------------------------------------------------- train
    if [ -f "$OUT/best.pt" ]; then
        log "$T/$ABL already trained, skipping to eval"
    else
        wait_for_my_jobs
        G=$(pick_gpu)
        log "TRAIN $T ablate=$ABL gpu=$G -> $OUT"
        "$PY" -u "$SRC/student/run_seed_earlystop.py" \
            --task "$T" --seed 0 --data_root "$DATA" --out_dir "$OUT" --gpu "$G" \
            --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
            --ablate "$ABL" > "$LOGD/t3_${ABL}_${T}_train.log" 2>&1
        rc=$?
        log "TRAIN $T rc=$rc best.pt=$([ -f "$OUT/best.pt" ] && echo yes || echo NO)"
        # Hard gate: without this the previous version marched on to eval and to the next
        # task after every step had failed, then printed ALL_DONE.
        if [ ! -f "$OUT/best.pt" ]; then
            log "ABORT $T: training produced no best.pt -- see $LOGD/t3_${ABL}_${T}_train.log"
            tail -20 "$LOGD/t3_${ABL}_${T}_train.log" | sed 's/^/    /'
            FAILED=1
            continue
        fi
    fi

    # ---------------------------------------------------------------- eval
    for NM in $NOISES; do
        TAG="ablate_${ABL}_n${NM}"
        LG="$LOGD/t3_${ABL}_${T}_eval_n${NM}.log"
        if grep -q "EVAL_DONE" "$LG" 2>/dev/null; then
            log "EVAL  $T n=${NM}mm already done, skipping"; continue
        fi
        wait_for_my_jobs
        G=$(pick_gpu)
        log "EVAL  $T noise=${NM}mm gpu=$G"
        CUDA_VISIBLE_DEVICES=$G "$PY" "$SRC/student/eval_ablate_student.py" --headless \
            --task "${ENVNAME[$T]}" \
            --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
            --episodes 256 --num_envs 32 --protocol_seed 42 \
            --fixed_pos_noise_mm "$NM" --dyn_rand on \
            --ablate "$ABL" --tag "$TAG" > "$LG" 2>&1
        rc=$?
        if grep -q "EVAL_DONE" "$LG" 2>/dev/null; then
            grep -hE "^\[.*task=" "$LG" | tail -1 | sed 's/^/    /'
        else
            log "EVAL  $T n=${NM}mm FAILED rc=$rc"
            tail -15 "$LG" | sed 's/^/    /'
            FAILED=1
        fi
    done
done

if [ "$FAILED" -eq 0 ]; then
    log "T3_${ABL}_ALL_DONE"
else
    log "T3_${ABL}_FINISHED_WITH_FAILURES"
    exit 1
fi
