#!/bin/bash
# Serially run the 8-condition frozen eval sweep for one student checkpoint.
# usage: chain_eval_student.sh <task_gym_id> <gpu_idx> <checkpoint_path> <norm_stats_path> <task_short_name>
set -e
TASK_GYM=$1
GPU=$2
CKPT=$3
NORM_STATS=$4
TASK_SHORT=$5

export OMNI_KIT_ACCEPT_EULA=Y
cd ~/forge_ts/src/student

for noise in 0 1 2.5 5; do
  for dr in on off; do
    echo "=== [$(date '+%H:%M:%S')] task=${TASK_SHORT} noise=${noise}mm dyn_rand=${dr} ===" >> /tmp/eval_chain_${TASK_SHORT}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval_frozen_student.py --headless \
      --task "$TASK_GYM" \
      --checkpoint "$CKPT" \
      --norm_stats "$NORM_STATS" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm $noise --dyn_rand $dr --tag student_bc_s0 \
      >> /tmp/eval_chain_${TASK_SHORT}.log 2>&1
  done
done
echo "CHAIN_EVAL_DONE" >> /tmp/eval_chain_${TASK_SHORT}.log
