#!/usr/bin/env bash
# ARCH-INSERT v2 (CORRECT recipe: obs includes EE velocity, ARCH_EE_VEL=1).
# v1 (pose-only) is INVALID -- it handicapped the policy (peg 0% / nut 0%),
# proven by the peg fairness retest: v1 0% vs v2 98.4% @0mm.
# GPU arg: 1 -> gear chain, 2 -> nut chain, 0 -> peg remaining evals.
export OMNI_KIT_ACCEPT_EULA=Y
export ARCH_EE_VEL=1
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/arch
LOGD=~/forge_ts/logs
MASTER=$LOGD/arch_v2_full_master.log
NN=$SRC/logs/rl_games/Forge
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

TASK=$1; GPU=$2
case $TASK in
  gear) gid=Isaac-Forge-GearMesh-ARCH-v0 ;;
  nut)  gid=Isaac-Forge-NutThread-ARCH-v0 ;;
  peg)  gid=Isaac-Forge-PegInsert-ARCH-v0 ;;
esac

# --- train (peg_v2 already trained; skip) ---
if [ "$TASK" != "peg" ]; then
  if ls "$NN/arch_${TASK}_v2_s0/nn"/last_Forge_ep_200*.pth >/dev/null 2>&1; then
    log "SKIP train ${TASK}_v2 (ckpt exists)"
  else
    log "v2 train $TASK on GPU$GPU START"
    CUDA_VISIBLE_DEVICES=$GPU $PY train_arch.py --task "$gid" --headless \
      --num_envs 128 --seed 0 --max_iterations 200 \
      agent.params.config.full_experiment_name=arch_${TASK}_v2_s0 \
      >> "$LOGD/arch_train_${TASK}_v2.log" 2>&1
    log "v2 train $TASK rc=$?"
  fi
fi

# --- eval all 4 noise levels ---
ck=$(ls "$NN/arch_${TASK}_v2_s0/nn"/last_Forge_ep_200_rew_[0-9]*.pth 2>/dev/null | head -1)
[ -z "$ck" ] && ck=$(ls "$NN/arch_${TASK}_v2_s0/nn"/last_Forge_ep_200*.pth 2>/dev/null | head -1)
if [ -z "$ck" ]; then log "v2 $TASK NO CKPT -- abort evals"; exit 1; fi
log "v2 $TASK ckpt: $(basename "$ck")"
for nm in 0 1 2.5 5; do
  lg=$LOGD/arch_v2_eval_${TASK}_n${nm}.log
  if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then log "SKIP eval $TASK n=$nm"; continue; fi
  log "v2 eval $TASK noise=$nm START"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval_arch.py --headless --task "$gid" \
    --checkpoint "$ck" --num_envs 32 --episodes 256 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "archv2_${TASK}_n${nm}" \
    >> "$lg" 2>&1
  log "v2 eval $TASK noise=$nm rc=$?"
done
log "ARCH_V2_${TASK}_DONE"
