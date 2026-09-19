#!/usr/bin/env bash
# TC teacher on GPU2: nut (longest task, gets its own GPU).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/corma
LOGD=~/forge_ts/logs
MASTER=$LOGD/tc_teachers_master.log
cd "$SRC"
nn="$SRC/logs/rl_games/Forge/tc_nut_s0/nn"
if ls "$nn"/*.pth >/dev/null 2>&1; then
  echo "=== [$(date -Is)] SKIP nut (ckpt exists)" >> "$MASTER"
else
  echo "=== [$(date -Is)] TC train nut on GPU2 START" >> "$MASTER"
  CUDA_VISIBLE_DEVICES=2 $PY train_tc.py --task Isaac-Forge-NutThread-TC-v0 --headless \
    --num_envs 128 --seed 0 --max_iterations 200 \
    agent.params.config.full_experiment_name=tc_nut_s0 \
    >> "$LOGD/tc_train_nut.log" 2>&1
  echo "=== [$(date -Is)] TC train nut rc=$?" >> "$MASTER"
fi
echo "=== [$(date -Is)] TC_GPU2_CHAIN_DONE" >> "$MASTER"
