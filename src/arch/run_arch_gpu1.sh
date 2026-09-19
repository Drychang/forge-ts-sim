#!/usr/bin/env bash
# ARCH-INSERT baseline: train peg then gear on GPU1 (200 epochs, == T-A budget).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/arch
LOGD=~/forge_ts/logs
MASTER=$LOGD/arch_master.log
cd "$SRC"
for t in peg gear; do
  case $t in
    peg)  gid=Isaac-Forge-PegInsert-ARCH-v0 ;;
    gear) gid=Isaac-Forge-GearMesh-ARCH-v0 ;;
  esac
  if ls "$SRC/logs/rl_games/Forge/arch_${t}_s0/nn"/*.pth >/dev/null 2>&1; then
    echo "=== [$(date -Is)] SKIP $t (ckpt exists)" >> "$MASTER"; continue
  fi
  echo "=== [$(date -Is)] ARCH train $t on GPU1 START" >> "$MASTER"
  CUDA_VISIBLE_DEVICES=1 $PY train_arch.py --task "$gid" --headless \
    --num_envs 128 --seed 0 --max_iterations 200 \
    agent.params.config.full_experiment_name=arch_${t}_s0 \
    >> "$LOGD/arch_train_${t}.log" 2>&1
  echo "=== [$(date -Is)] ARCH train $t rc=$?" >> "$MASTER"
done
echo "=== [$(date -Is)] ARCH_GPU1_CHAIN_DONE" >> "$MASTER"
