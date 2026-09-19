#!/usr/bin/env bash
# Breaking-point ORACLE CEILING: the privileged T-B teacher at the same
# beyond-training noise levels (7.5/10/15/20mm).
#
# Why this matters: without it, a student drop at 15/20mm is ambiguous between
#   (a) distillation/perception failing to extrapolate, and
#   (b) the task itself becoming unsolvable at that displacement
# T-B reads the true noise vector, so it is the information-theoretic ceiling.
# If T-B also collapses, the axis has run out of task, not out of method.
# State-only eval -> fast and light; runs after the T-A chain releases the GPU.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
MASTER=$LOGD/bp_tb_master.log
GPU=${1:-2}
FORGE=/extra_home3/user/force_vla_research/IsaacLab/logs/rl_games/Forge
declare -A GID=( [peg]=Isaac-Forge-PegInsert-TB-v0 [gear]=Isaac-Forge-GearMesh-TB-v0 [nut]=Isaac-Forge-NutThread-TB-v0 )
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

log "waiting for the T-A breaking-point chain to release GPU $GPU"
while ! grep -q "BP2_TA_ALL_DONE" $LOGD/bp2_ta_master.log 2>/dev/null; do sleep 180; done

for t in peg gear nut; do
  ck=$(ls -t $FORGE/tb_${t}_s0/nn/*.pth 2>/dev/null | head -1)
  [ -n "$ck" ] || { log "SKIP $t: no tb checkpoint"; continue; }
  for nm in 7.5 10 15 20; do
    lg=$LOGD/bp_tb_${t}_n${nm}.log
    grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP $t n=$nm"; continue; }
    log "BP T-B $t noise=${nm}mm START (ckpt=$(basename $ck))"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_frozen_tb.py" --headless \
      --task "${GID[$t]}" --checkpoint "$ck" \
      --num_envs 128 --episodes 256 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "bp_tb_${t}_n${nm}" >> "$lg" 2>&1
    log "BP T-B $t noise=${nm}mm rc=$?"
  done
done
log "BP_TB_ALL_DONE"
