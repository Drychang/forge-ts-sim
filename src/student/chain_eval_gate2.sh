#!/bin/bash
# Serially run the 8-condition frozen eval sweep for one Gate-2 student checkpoint.
# Same as chain_eval_student.sh but takes an explicit --tag so 3 seeds don't collide.
# usage: chain_eval_gate2.sh <task_gym_id> <gpu_idx> <checkpoint_path> <norm_stats_path> <task_short_name> <tag>
set -e
TASK_GYM=$1
GPU=$2
CKPT=$3
NORM_STATS=$4
TASK_SHORT=$5
TAG=$6

export OMNI_KIT_ACCEPT_EULA=Y
cd ~/forge_ts/src/student

for noise in 0 1 2.5 5; do
  for dr in on off; do
    echo "=== [$(date '+%H:%M:%S')] task=${TASK_SHORT} tag=${TAG} noise=${noise}mm dyn_rand=${dr} ===" >> ~/forge_ts/logs/eval_gate2_${TASK_SHORT}_${TAG}.log
    CUDA_VISIBLE_DEVICES=$GPU ~/miniconda3/envs/isaaclab/bin/python eval_frozen_student.py --headless \
      --task "$TASK_GYM" \
      --checkpoint "$CKPT" \
      --norm_stats "$NORM_STATS" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm $noise --dyn_rand $dr --tag "$TAG" \
      >> ~/forge_ts/logs/eval_gate2_${TASK_SHORT}_${TAG}.log 2>&1
  done
done
echo "CHAIN_EVAL_GATE2_DONE task=${TASK_SHORT} tag=${TAG}" >> ~/forge_ts/logs/eval_gate2_${TASK_SHORT}_${TAG}.log
