#!/usr/bin/env bash
# ARCH-INSERT baseline: train nut on GPU2 (longest task, own GPU).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/arch
LOGD=~/forge_ts/logs
MASTER=$LOGD/arch_master.log
cd "$SRC"
if ls "$SRC/logs/rl_games/Forge/arch_nut_s0/nn"/*.pth >/dev/null 2>&1; then
  echo "=== [$(date -Is)] SKIP nut (ckpt exists)" >> "$MASTER"
else
  echo "=== [$(date -Is)] ARCH train nut on GPU2 START" >> "$MASTER"
  CUDA_VISIBLE_DEVICES=2 $PY train_arch.py --task Isaac-Forge-NutThread-ARCH-v0 --headless \
    --num_envs 128 --seed 0 --max_iterations 200 \
    agent.params.config.full_experiment_name=arch_nut_s0 \
    >> "$LOGD/arch_train_nut.log" 2>&1
  echo "=== [$(date -Is)] ARCH train nut rc=$?" >> "$MASTER"
fi
echo "=== [$(date -Is)] ARCH_GPU2_CHAIN_DONE" >> "$MASTER"
