#!/usr/bin/env bash
# x6: can a policy be trained to survive whichever modality goes bad?
#
# The two failure modes found so far are mirror images, and no single fixed fusion handles
# both. In simulation at 20mm pose noise, STATE runs 5.4-5.9 sigma outside its training
# range and the model that uses it loses ~9 points to the model that does not. On the real
# rig, VISION is the one out of distribution -- the wrist mount sits at roughly twice the
# simulator's 198.5mm stand-off -- and there the model that has state still collapses to
# 2.34% (peg) / 0.00% (gear) under only 20mm/4deg of jitter, below what a state-only model
# reaches at 5mm pose noise. Having the intact modality available is not the same as being
# able to use it.
#
# Modality dropout is the standard remedy: zero one modality per sample during training so
# the network never gets to depend on any single channel. Patch verified before this ran --
# at most one modality per sample, rate 0.299-0.314 against a 0.30 target, uniform over the
# three, both cameras dropped together, dtypes preserved.
#
# peg only, one seed, p=0.30 as a proof of concept. Widening to gear/nut and sweeping p is
# only worth the GPU time if this first cell shows the effect.
#
# Four questions, four eval blocks:
#   1  noise sweep, all modalities present -- did robustness cost in-distribution accuracy?
#      (compare: full 98.31/96.61 at 0/5mm, -state 97.40/96.48)
#   2  wrist-only jitter -- the deployment question this was built for
#      (compare: full 2.34% peg at 20mm/4deg with BOTH cameras jittered)
#   3  state removed at test time -- did it learn to run without state?
#   4  vision removed at test time -- did it learn to run without vision?
#      (compare: full ablate_vision 60.55% at 0mm, 32.42% at 5mm)
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x6_master.log
QDIR=$LOGD/x6_queue; CLAIMS=$QDIR/claims
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-25}"
GPUS="${GPUS:-0 1 2}"
P_DROP="${P_DROP:-0.30}"
OUT=$CKPT_ROOT/peg/moddrop_p30
ENVN=Isaac-Forge-PegInsert-TBCamera-v0

mkdir -p "$LOGD" "$QDIR"; rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
             | tr -d ' ' | awk -F, '{print $2-$1}'; }
free_ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }
wait_for_gpu(){ local G="$1" f; while true; do f=$(free_on "$G")
    [ "${f:-0}" -ge "$MIN_FREE_MIB" ] && return 0; log "gpu $G ${f}MiB, waiting"; sleep 300; done; }

# $1 gpu $2 ablate $3 noise $4 jitter_pos $5 jitter_rot $6 tag
do_eval(){
    local G="$1" ABL="$2" NM="$3" PM="$4" RD="$5" TAG="$6" rc
    local LG="$LOGD/${TAG}.log"
    grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; return 0; }
    [ -f "$OUT/best.pt" ] || { log "SKIP $TAG -- no best.pt"; return 1; }
    wait_for_gpu "$G"
    log "EVAL $TAG gpu=$G ablate=$ABL noise=${NM} jitter=${PM}/${RD}"
    CUDA_VISIBLE_DEVICES=$G \
    TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
    TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$ENVN" \
        --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm "$NM" --dyn_rand on \
        --ablate "$ABL" --tag "$TAG" > "$LG" 2>&1
    rc=$?
    log "EVAL $TAG rc=$rc $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
}

train_ok(){ [ -f "$OUT/best.pt" ] && \
    grep -q "RUN_SEED_EARLYSTOP_DONE reason=\(early_stop\|max_epochs\)" "$OUT/train.log" 2>/dev/null; }

log "==================== x6 start ===================="
log "waiting for x5"
while ! grep -q "X5_ALL_DONE" "$LOGD/x5_master.log" 2>/dev/null; do sleep 300; done

# ------------------------------------------------------------------ train
if train_ok; then
    log "moddrop already trained"
else
    mkdir -p "$OUT"
    # Inherit the canonical norm_stats: compute_norm_stats() is a 2000-sample estimate whose
    # index->sample mapping depends on shard load order, so a fresh one would z-score the
    # inputs differently from every cell this is compared against.
    [ -f "$OUT/norm_stats.npz" ] || cp "$CKPT_ROOT/peg/norm_stats.npz" "$OUT/norm_stats.npz"
    for attempt in 1 2 3; do
        while [ "$(free_ram_gb)" -lt "$MIN_FREE_RAM_GB" ]; do
            log "host RAM $(free_ram_gb)GB < $MIN_FREE_RAM_GB, waiting"; sleep 300; done
        G=0; for g in $GPUS; do [ "$(free_on $g)" -ge "$MIN_FREE_MIB" ] && { G=$g; break; }; done
        wait_for_gpu "$G"
        log "TRAIN moddrop p=$P_DROP attempt $attempt gpu=$G ram=$(free_ram_gb)GB"
        "$PY" -u "$SRC/student/run_seed_earlystop.py" \
            --task peg --seed 0 --data_root "$DATA" --out_dir "$OUT" --gpu "$G" \
            --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
            --extra_args "--modality_dropout $P_DROP" > "$LOGD/x6_train.log" 2>&1
        # An OOM-killed run still writes a first-epoch best.pt, so existence is not completion.
        train_ok && { log "TRAIN OK attempt $attempt ($(grep -cE '^epoch ' "$OUT/train.log") epochs)"; break; }
        log "TRAIN attempt $attempt FAILED: $(grep -oE 'reason=\S+ ?\S*' "$OUT/train.log" 2>/dev/null | tail -1)"
        mv "$OUT/best.pt" "$OUT/best.pt.failed_attempt$attempt" 2>/dev/null
        sleep 300
    done
fi
train_ok || { log "ABORT: no usable moddrop checkpoint"; exit 1; }
log "modality_dropout recorded in run header: $(grep -o 'modality_dropout=[0-9.]*' "$OUT/train.log" | head -1)"

# ------------------------------------------------------------------ eval
Q=$QDIR/jobs; : > "$Q"
for NM in 0 1 2.5 5 7.5 10 15 20; do echo "none $NM 0 0 x6_md_clean_n${NM}" >> "$Q"; done
for J in "10 2" "20 4" "40 8"; do set -- $J; echo "none 2.5 $1 $2 x6_md_wj_p$1r$2" >> "$Q"; done
for NM in 0 5 20;  do echo "state  $NM 0 0 x6_md_nostate_n${NM}" >> "$Q"; done
for NM in 0 5;     do echo "vision $NM 0 0 x6_md_novision_n${NM}" >> "$Q"; done
log "$(wc -l < "$Q") evals queued"

pids=()
for G in $GPUS; do
    ( n=0
      while IFS= read -r line; do
          n=$((n+1)); mkdir "$CLAIMS/j$n" 2>/dev/null || continue
          set -- $line; do_eval "$G" "$1" "$2" "$3" "$4" "$5"
      done < "$Q" ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X6_ALL_DONE ===================="
