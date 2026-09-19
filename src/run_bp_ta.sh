#!/usr/bin/env bash
# BREAKING-POINT experiment (T-A official baseline, state-only -> light on RAM):
# same extended noise levels for a reference curve.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
MASTER=$LOGD/bp_ta_master.log
FORGE=/extra_home3/user/force_vla_research/IsaacLab/logs/rl_games/Forge
GPU=${1:-2}
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }
declare -A GID=( [peg]=Isaac-Forge-PegInsert-Direct-v0 [gear]=Isaac-Forge-GearMesh-Direct-v0 [nut]=Isaac-Forge-NutThread-Direct-v0 )
for t in peg gear nut; do
  ck=$(ls -t $FORGE/repro_${t}_s0/nn/*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "no T-A ckpt for $t"; continue; }
  for nm in 7.5 10; do
    lg=$LOGD/bp_ta_${t}_n${nm}.log
    grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null && { log "SKIP $t n=$nm"; continue; }
    log "BP T-A $t noise=${nm}mm START (ckpt=$(basename $ck))"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_frozen.py" --headless \
      --task "${GID[$t]}" --checkpoint "$ck" \
      --episodes 256 --num_envs 128 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "bp_ta_${t}_n${nm}" >> "$lg" 2>&1
    log "BP T-A $t noise=${nm}mm rc=$?"
  done
done
log "BP_TA_ALL_DONE"
