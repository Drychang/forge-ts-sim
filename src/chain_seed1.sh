#!/usr/bin/env bash
set -u
TASK="${1:?}"; SWEEP_LOG="${2:?}"; GPU="${3:?}"; SHORT="${4:?}"
SRC=/home/user/force_vla_research/IsaacLab
until grep -q "SWEEP_DONE" "$SWEEP_LOG" 2>/dev/null; do sleep 15; done
echo "$(date -Is) sweep done for $TASK on GPU$GPU, launching seed1 training"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd "$SRC" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=Y
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task "$TASK" --headless --num_envs 128 --seed 1 \
  agent.params.config.full_experiment_name=repro_${SHORT}_s1 \
  > /home/user/forge_ts/logs/repro_${SHORT}_s1.log 2>&1
echo "TRAIN_S1_DONE $TASK"
