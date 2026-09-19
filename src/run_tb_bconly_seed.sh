#!/usr/bin/env bash
# BC-only vs +DAgger, seeds 1 and 2 (usage: run_tb_bconly_seed.sh <seed> <gpu>).
#
# Why now: tonight's matched seed-0 pair showed DAgger is worth only +0.8 points
# on peg @5mm (BC-only 95.3 vs BC+DAgger 96.1), which contradicts the previously
# recorded "+6~24 points". That old number is currently the weakest claim in the
# report, and a single seed is not enough to replace it with. The BC+DAgger side
# already has 3 seeds (gate2_seed0/1/2), so two more BC-only seeds make the
# contrast a proper 3-vs-3.
#
# Identical recipe to the seed-0 arm: same BC-only shard view (shards 0-15 of the
# T-B dataset, 2010 demos), same trainer, same early stopping, own norm_stats,
# same frozen eval protocol. Only the seed differs.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
SEED=${1:?seed}
GPU=${2:?gpu}
TASK=peg
GYM=Isaac-Forge-PegInsert-TBCamera-v0
DATA=/media/data/forge_ts_data_bconly
OUT=~/forge_ts/student_ckpts/$TASK/tb_bconly_seed${SEED}
MASTER=$LOGD/tb_bconly_s${SEED}_master.log
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_mem(){ # anonymous memory is the real constraint; page cache is reclaimable
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "${1:-20}" ]; do
    log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 240
  done
}

log "START seed=$SEED gpu=$GPU data=$DATA"

if [ -f "$OUT/best.pt" ]; then
  log "SKIP train: best.pt exists"
else
  wait_mem 20
  mkdir -p "$OUT"
  log "train -> $OUT"
  $PY -u "$SRC/student/run_seed_earlystop.py" \
    --task "$TASK" --seed "$SEED" --data_root "$DATA" --out_dir "$OUT" --gpu "$GPU" \
    --num_workers 4 > $LOGD/tb_bconly_s${SEED}_train.log 2>&1
  log "train rc=$? best=$([ -f $OUT/best.pt ] && echo yes || echo NO)"
fi
[ -f "$OUT/best.pt" ] || { log "FATAL no best.pt"; exit 1; }

for nm in 0 1 2.5 5; do
  lg=$LOGD/tb_bconly_s${SEED}_eval_${TASK}_n${nm}.log
  grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP eval n=$nm"; continue; }
  wait_mem 20
  log "eval n=${nm}mm"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/student/eval_frozen_student.py" --headless \
    --task "$GYM" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag tb_bconly_s${SEED}_${TASK}_n${nm} >> "$lg" 2>&1
  log "eval n=${nm} rc=$?"
done
log "TB_BCONLY_SEED_DONE seed=$SEED"
