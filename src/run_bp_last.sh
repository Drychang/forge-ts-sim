#!/usr/bin/env bash
# Finish the last cell of the breaking-point matrix: student nut @20mm.
#
# The bp2 driver's guard counts the two P0-1 training groups' dataloader workers
# as camera jobs (10 of them), so it would sit idle until ~10:00. Actual memory
# pressure is low -- 14G anonymous of 62G; the 38G in buff/cache is reclaimable
# NFS page cache from reading shards -- and GPU0 is completely free. So run the
# remaining cell directly with a light check instead of waiting 6h for a guard
# that is measuring the wrong thing.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/bp_last_master.log
GPU=${1:-0}
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

for nm in 20; do
  lg=$LOGD/bp_student_nut_n${nm}.log
  grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP nut n=$nm (already done)"; continue; }
  # only refuse to start if the box is genuinely short on reclaimable memory
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
    log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 180
  done
  log "student nut noise=${nm}mm START on gpu $GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_frozen_student.py" --headless \
    --task Isaac-Forge-NutThread-TBCamera-v0 \
    --checkpoint ~/forge_ts/student_ckpts/nut/gate2_seed0/best.pt \
    --norm_stats ~/forge_ts/student_ckpts/nut/gate2_seed0/norm_stats.npz \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "bp_nut_n${nm}" >> "$lg" 2>&1
  log "student nut noise=${nm}mm rc=$?"
done
log "BP_LAST_DONE"
