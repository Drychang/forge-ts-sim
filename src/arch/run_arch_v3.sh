#!/usr/bin/env bash
# ARCH-INSERT v3 = FINAL fair recipe: EE velocity obs (ARCH_EE_VEL=1) AND
# rotation scale matched to T-A's reachable set (ARCH_ANGVEL_SCALE=15.0).
# v2 had a 5x rotation handicap that crippled NutThread (0% despite contact).
# Usage: run_arch_v3.sh <task> <gpu>
export OMNI_KIT_ACCEPT_EULA=Y
export ARCH_EE_VEL=1
export ARCH_ANGVEL_SCALE=15.0
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/arch
LOGD=~/forge_ts/logs
MASTER=$LOGD/arch_v3_master.log
NN=$SRC/logs/rl_games/Forge
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }
TASK=$1; GPU=$2
case $TASK in
  peg)  gid=Isaac-Forge-PegInsert-ARCH-v0 ;;
  gear) gid=Isaac-Forge-GearMesh-ARCH-v0 ;;
  nut)  gid=Isaac-Forge-NutThread-ARCH-v0 ;;
esac
if ls "$NN/arch_${TASK}_v3_s0/nn"/last_Forge_ep_200*.pth >/dev/null 2>&1; then
  log "SKIP train ${TASK}_v3"
else
  log "v3 train $TASK on GPU$GPU START"
  CUDA_VISIBLE_DEVICES=$GPU $PY train_arch.py --task "$gid" --headless \
    --num_envs 128 --seed 0 --max_iterations 200 \
    agent.params.config.full_experiment_name=arch_${TASK}_v3_s0 \
    >> "$LOGD/arch_train_${TASK}_v3.log" 2>&1
  log "v3 train $TASK rc=$?"
fi
ck=$(ls "$NN/arch_${TASK}_v3_s0/nn"/last_Forge_ep_200_rew_[0-9]*.pth 2>/dev/null | head -1)
[ -z "$ck" ] && ck=$(ls "$NN/arch_${TASK}_v3_s0/nn"/last_Forge_ep_200*.pth 2>/dev/null | head -1)
[ -z "$ck" ] && { log "v3 $TASK NO CKPT"; exit 1; }
log "v3 $TASK ckpt: $(basename "$ck")"
for nm in 0 1 2.5 5; do
  lg=$LOGD/arch_v3_eval_${TASK}_n${nm}.log
  grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null && { log "SKIP eval $TASK n=$nm"; continue; }
  log "v3 eval $TASK noise=$nm START"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval_arch.py --headless --task "$gid" \
    --checkpoint "$ck" --num_envs 32 --episodes 256 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "archv3_${TASK}_n${nm}" >> "$lg" 2>&1
  log "v3 eval $TASK noise=$nm rc=$?"
done
log "ARCH_V3_${TASK}_DONE"
