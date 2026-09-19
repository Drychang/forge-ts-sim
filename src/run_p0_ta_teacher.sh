#!/usr/bin/env bash
# ============================================================================
# P0-1 CONTROL: distil from the NON-privileged teacher (T-A).
#
# Question the paper cannot currently answer: is the *privileged noise teacher*
# actually load-bearing, or would vision+force distilled from an ordinary
# state-only teacher do just as well? P1 showed the fusion architecture is not
# the source of performance, which sharpens exactly this doubt.
#
# Design: byte-identical to the T-B student pipeline EXCEPT the teacher.
#   same env (TBCamera), same per-episode noise law (std ~ U[0,5mm]),
#   same success-only filtering, same BC target (2000 successes),
#   same trainer / early stopping / seed, same frozen eval protocol.
# The comparison point is the EXISTING T-B BC-only student (peg 5mm = 76.2%),
# so this stage needs no DAgger to be decisive.
#
# Expected (favourable) outcome: T-A only succeeds ~32% at 5mm, so success-only
# filtering starves the dataset of high-noise demonstrations -> the student never
# sees the compensation behaviour -> it degrades on the noise axis. That IS the
# necessity argument for the privileged teacher.
#
# num_envs=64 (not the original 128) purely to stay RAM-safe next to the running
# breaking-point camera evals. Rollouts are independent and noise is per-episode,
# so batch width changes shard boundaries, not the data distribution.
# ============================================================================
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
MASTER=$LOGD/p0_ta_teacher_master.log
GPU=${1:-1}
TASK=${2:-peg}

declare -A GID=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0 [gear]=Isaac-Forge-GearMesh-TBCamera-v0 [nut]=Isaac-Forge-NutThread-TBCamera-v0 )
declare -A DSET=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )
declare -A CKN=( [peg]=repro_peg_s0 [gear]=repro_gear_s0 [nut]=repro_nut_s0 )

FORGE=/extra_home3/user/force_vla_research/IsaacLab/logs/rl_games/Forge
DATA_TA=/media/data/forge_ts_data_ta
OUT=~/forge_ts/student_ckpts/$TASK/ta_teacher_seed0

log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_ram(){   # camera work is RAM-bound; never fight the other camera jobs
  local need=${1:-22}
  while true; do
    busy=$(pgrep -fc "train_student_bc.py|train_student_arch.py|eval_frozen_student|eval_ablate_student|eval_arch_student|collect_camera_rollouts" || true)
    ram=$(free -g | awk '/^Mem:/{print $7}')
    if [ "${busy:-0}" -le 1 ] && [ "$ram" -ge "$need" ]; then return; fi
    log "wait_ram: camera_jobs=${busy:-0} free=${ram}G (need ${need}G)"
    sleep 300
  done
}

log "P0-1 START task=$TASK gpu=$GPU"

# ---- stage 1: BC demonstrations from the T-A teacher ------------------------
CK=$(ls -t $FORGE/${CKN[$TASK]}/nn/*.pth 2>/dev/null | head -1)
[ -n "$CK" ] || { log "FATAL no T-A checkpoint under $FORGE/${CKN[$TASK]}/nn"; exit 1; }
DDIR=$DATA_TA/${DSET[$TASK]}
mkdir -p "$DDIR"
have=$(ls "$DDIR"/shard_*.npz 2>/dev/null | wc -l)
if [ "$have" -ge 15 ]; then
  log "SKIP stage1: $have shards already in $DDIR"
else
  wait_ram 22
  log "stage1 collect: ckpt=$(basename $CK) -> $DDIR"
  cd "$SRC"
  CUDA_VISIBLE_DEVICES=$GPU $PY collect_ta_rollouts.py --headless \
    --task "${GID[$TASK]}" --checkpoint "$CK" \
    --num_envs 64 --target_success 2000 --shard_size 100 --jpeg_quality 90 \
    --seed 0 --out_dir "$DDIR" > $LOGD/p0_ta_collect_${TASK}.log 2>&1
  log "stage1 rc=$? shards=$(ls $DDIR/shard_*.npz 2>/dev/null | wc -l)"
fi

# ---- stage 2: train the student on T-A demos (identical recipe) -------------
if [ -f "$OUT/best.pt" ]; then
  log "SKIP stage2: $OUT/best.pt exists"
else
  wait_ram 25
  log "stage2 train -> $OUT"
  cd "$SRC/student"
  CUDA_VISIBLE_DEVICES=$GPU $PY run_seed_earlystop.py \
    --task "$TASK" --seed 0 --data_root "$DATA_TA" --out_dir "$OUT" --gpu 0 \
    > $LOGD/p0_ta_train_${TASK}.log 2>&1
  log "stage2 rc=$?"
fi

# ---- stage 3: frozen protocol eval -----------------------------------------
for nm in 0 1 2.5 5; do
  lg=$LOGD/p0_ta_eval_${TASK}_n${nm}.log
  grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP eval n=$nm"; continue; }
  wait_ram 25
  log "stage3 eval n=${nm}mm"
  cd "$SRC/student"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval_frozen_student.py --headless \
    --task "${GID[$TASK]}" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag p0_ta_${TASK}_n${nm} > "$lg" 2>&1
  log "stage3 n=${nm} rc=$?"
done

log "P0_TA_TEACHER_DONE task=$TASK"
