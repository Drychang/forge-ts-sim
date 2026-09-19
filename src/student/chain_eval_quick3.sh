#!/bin/bash
# Quick 3-condition eval sweep (0/2.5/5mm x dyn_rand=on) for a DAgger checkpoint,
# per spec: full 8-condition sweep only on the FINAL DAgger round.
# usage: chain_eval_quick3.sh <task_gym_id> <gpu_idx> <checkpoint_path> <norm_stats_path> <task_short_name> <tag>
set -e
TASK_GYM=$1
GPU=$2
CKPT=$3
NORM_STATS=$4
TASK_SHORT=$5
TAG=$6

export OMNI_KIT_ACCEPT_EULA=Y
cd ~/forge_ts/src/student

for noise in 0 2.5 5; do
  echo "=== [$(date '+%H:%M:%S')] task=${TASK_SHORT} noise=${noise}mm dyn_rand=on tag=${TAG} ===" >> /tmp/eval_quick3_${TASK_SHORT}.log
  CUDA_VISIBLE_DEVICES=$GPU python eval_frozen_student.py --headless \
    --task "$TASK_GYM" \
    --checkpoint "$CKPT" \
    --norm_stats "$NORM_STATS" \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm $noise --dyn_rand on --tag "$TAG" \
    >> /tmp/eval_quick3_${TASK_SHORT}.log 2>&1
done
echo "CHAIN_EVAL_QUICK3_DONE" >> /tmp/eval_quick3_${TASK_SHORT}.log
