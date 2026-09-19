#!/usr/bin/env bash
# ============================================================================
# P0-1, one arm of the matched pair. Usage: run_p0_arm.sh <ta|tb> <gpu> [task]
#
#   ta  student distilled from the NON-privileged T-A teacher
#   tb  student distilled from the privileged T-B teacher, BC-ONLY
#
# Both arms: same env, same per-episode noise law (std ~ U[0,5mm]), same
# success-only BC data budget (~2000 demos, shards 0-15), same trainer / early
# stopping / seed, each computing its own norm_stats from its own training
# split, same frozen eval protocol. The ONLY difference is which teacher
# produced the actions -- which is exactly the claim under test.
#
# The tb arm's dataset is a symlink view of shards 0-15 of the existing T-B
# dataset (the BC stage: success-filtered, hence 126-128 episodes/shard, vs the
# DAgger shards 16-23 which are all exactly 128 because they are unfiltered).
# Zero copy, and it doubles as the BC-only vs +DAgger contrast for the T-B side.
#
# NOTE run_seed_earlystop.py sets CUDA_VISIBLE_DEVICES=<--gpu> itself, so the
# training stage takes a REAL gpu index and must NOT be wrapped in an outer
# CUDA_VISIBLE_DEVICES (that was a bug in the first driver: it would have
# trained on physical GPU0, colliding with the breaking-point camera evals).
# ============================================================================
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
ARM=${1:?arm: ta|tb}
GPU=${2:?gpu index}
TASK=${3:-peg}
MASTER=$LOGD/p0_${ARM}_master.log

declare -A GID=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0 [gear]=Isaac-Forge-GearMesh-TBCamera-v0 [nut]=Isaac-Forge-NutThread-TBCamera-v0 )
case "$ARM" in
  ta) DATA=/media/data/forge_ts_data_ta;     OUT=~/forge_ts/student_ckpts/$TASK/ta_teacher_seed0 ;;
  tb) DATA=/media/data/forge_ts_data_bconly; OUT=~/forge_ts/student_ckpts/$TASK/tb_bconly_seed0 ;;
  *)  echo "arm must be ta|tb"; exit 2 ;;
esac

log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_ram(){
  # Relaxed guard for the second arm. The strict version counted the sibling
  # arm's 4 dataloader workers as camera jobs, so this arm could never start
  # while the other trained -- serialising the matched pair by ~6h for no reason.
  # Measured footprints: one training group 21.6G, one camera eval 8.7G, box has
  # 62G. Two trainings coexist; a training plus a camera eval also coexists. So
  # gate on (no camera EVAL running) and (>=30G available), not on job count.
  local need=${1:-30}
  while true; do
    evals=$(pgrep -fc "eval_frozen_studen[t]|eval_ablate_studen[t]|eval_arch_studen[t]|eval_latency_studen[t]|collect_ta_rollout[s]|collect_camera_rollout[s]" || true)
    avail=$(free -g | awk '/^Mem:/{print $7}')
    if [ "${evals:-0}" -eq 0 ] && [ "$avail" -ge "$need" ]; then return; fi
    log "wait_ram: camera_evals=${evals:-0} avail=${avail}G (need 0 evals, >=${need}G)"
    sleep 240
  done
}

log "P0 arm=$ARM task=$TASK gpu=$GPU data=$DATA"

# ---- stage 0 (ta arm only): the collection may still be running ------------
if [ "$ARM" = ta ]; then
  while pgrep -f "collect_ta_rollout[s].py" > /dev/null 2>&1; do
    log "waiting for T-A collection: $(ls $DATA/${TASK^} 2>/dev/null | wc -l) files, $(grep -c 'wrote' $LOGD/p0_ta_collect_${TASK}.log 2>/dev/null || echo 0) shards"
    sleep 300
  done
  log "collection finished: $(grep 'DONE' $LOGD/p0_ta_collect_${TASK}.log 2>/dev/null | tail -1)"
fi

DDIR=$DATA/$(case $TASK in peg) echo PegInsert;; gear) echo GearMesh;; nut) echo NutThread;; esac)
n=$(ls "$DDIR"/shard_*.npz 2>/dev/null | wc -l)
[ "$n" -ge 10 ] || { log "FATAL only $n shards in $DDIR"; exit 1; }
log "dataset ready: $n shards in $DDIR"

# ---- stage 1: train --------------------------------------------------------
if [ -f "$OUT/best.pt" ]; then
  log "SKIP train: $OUT/best.pt exists"
else
  wait_ram 22
  mkdir -p "$OUT"
  log "train -> $OUT (real gpu $GPU, own norm_stats)"
  $PY -u "$SRC/student/run_seed_earlystop.py" \
    --task "$TASK" --seed 0 --data_root "$DATA" --out_dir "$OUT" --gpu "$GPU" \
    --num_workers 4 > $LOGD/p0_${ARM}_train_${TASK}.log 2>&1
  log "train rc=$? best=$([ -f $OUT/best.pt ] && echo yes || echo NO)"
fi
[ -f "$OUT/best.pt" ] || { log "FATAL no best.pt, cannot eval"; exit 1; }

# ---- stage 2: frozen protocol eval ----------------------------------------
for nm in 0 1 2.5 5; do
  lg=$LOGD/p0_${ARM}_eval_${TASK}_n${nm}.log
  grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP eval n=$nm"; continue; }
  wait_ram 25
  log "eval n=${nm}mm"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/student/eval_frozen_student.py" --headless \
    --task "${GID[$TASK]}" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag p0_${ARM}_${TASK}_n${nm} >> "$lg" 2>&1
  log "eval n=${nm} rc=$?"
done

log "P0_ARM_DONE arm=$ARM task=$TASK"
