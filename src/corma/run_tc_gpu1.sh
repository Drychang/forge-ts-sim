#!/usr/bin/env bash
# TC teachers on GPU1: peg then gear (sequential). Simple + robust (no set -u,
# no associative arrays, no subshells).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/corma
LOGD=~/forge_ts/logs
MASTER=$LOGD/tc_teachers_master.log
cd "$SRC"

for t in peg gear; do
  case $t in
    peg)  gid=Isaac-Forge-PegInsert-TC-v0 ;;
    gear) gid=Isaac-Forge-GearMesh-TC-v0 ;;
  esac
  nn="$SRC/logs/rl_games/Forge/tc_${t}_s0/nn"
  if ls "$nn"/*.pth >/dev/null 2>&1; then
    echo "=== [$(date -Is)] SKIP $t (ckpt exists)" >> "$MASTER"; continue
  fi
  echo "=== [$(date -Is)] TC train $t on GPU1 START" >> "$MASTER"
  CUDA_VISIBLE_DEVICES=1 $PY train_tc.py --task "$gid" --headless \
    --num_envs 128 --seed 0 --max_iterations 200 \
    agent.params.config.full_experiment_name=tc_${t}_s0 \
    >> "$LOGD/tc_train_${t}.log" 2>&1
  echo "=== [$(date -Is)] TC train $t rc=$?" >> "$MASTER"
done
echo "=== [$(date -Is)] TC_GPU1_CHAIN_DONE" >> "$MASTER"
