#!/usr/bin/env bash
# P0-1 tb arm: frozen-protocol eval, RAM-only guard.
#
# The tb driver's eval guard demanded ZERO camera evals running, which meant it
# would queue behind the ta arm's own 4 cells AND the wrench-delay sweep -- ~2h
# of an idle GPU2 for a constraint that isn't real: one camera eval is ~8.7G and
# 45G is available, so three coexist comfortably (already demonstrated tonight by
# running the last breaking-point cell alongside two trainings).
#
# Training is already done (EARLY_STOP epoch=13, best_epoch=7, best_val=0.08477),
# so this only runs the eval stage. It appends P0_ARM_DONE to the tb master log
# on completion so the queued extrapolation follow-up (run_p0_extrap.sh) unblocks.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/p0_tb_master.log
GPU=${1:-2}
TASK=peg
GYM=Isaac-Forge-PegInsert-TBCamera-v0
OUT=~/forge_ts/student_ckpts/$TASK/tb_bconly_seed0
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

[ -f "$OUT/best.pt" ] || { log "FATAL no best.pt"; exit 1; }

for nm in 0 1 2.5 5; do
  lg=$LOGD/p0_tb_eval_${TASK}_n${nm}.log
  grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP eval n=$nm"; continue; }
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
    log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 180
  done
  log "eval n=${nm}mm (permissive guard, gpu $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_frozen_student.py" --headless \
    --task "$GYM" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag p0_tb_${TASK}_n${nm} >> "$lg" 2>&1
  log "eval n=${nm} rc=$?"
done

log "P0_ARM_DONE arm=tb task=$TASK"
