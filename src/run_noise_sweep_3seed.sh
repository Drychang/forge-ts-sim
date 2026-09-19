#!/usr/bin/env bash
# Noise-spectrum sweep across all 3 seeds for one task.
# usage: TASK=Isaac-Forge-PegInsert-Direct-v0 SHORT=peg GPU=0 bash run_noise_sweep_3seed.sh
set -u

TASK="${TASK:?set TASK}"
SHORT="${SHORT:?set SHORT}"
GPU="${GPU:-0}"
SRC_DIR="$HOME/forge_ts/src"
LOG_DIR="$HOME/forge_ts/logs"
FORGE_DIR="$HOME/force_vla_research/IsaacLab/logs/rl_games/Forge"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=Y
cd "$HOME/force_vla_research/IsaacLab" || exit 1

for seed in 0 1 2; do
  ckpt=$(ls "$FORGE_DIR/repro_${SHORT}_s${seed}/nn/"*ep_200*.pth 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then
    echo "=== SKIP seed=$seed: no ep_200 checkpoint found for repro_${SHORT}_s${seed} ==="
    continue
  fi
  tag="noisespec_s${seed}"
  for noise in 0 1 2.5 5; do
    for dr in on off; do
      logf="$LOG_DIR/${tag}_${SHORT}_noise${noise}mm_dr${dr}.log"
      echo "=== $(date -Is) task=$TASK seed=$seed noise=${noise}mm dyn_rand=$dr ckpt=$ckpt ===" | tee -a "$logf"
      ./isaaclab.sh -p "$SRC_DIR/eval_frozen.py" \
        --task "$TASK" --checkpoint "$ckpt" \
        --num_envs 128 --episodes 256 --protocol_seed 42 \
        --fixed_pos_noise_mm "$noise" --dyn_rand "$dr" \
        --tag "$tag" --headless >> "$logf" 2>&1
      rc=$?
      echo "=== exit=$rc ===" | tee -a "$logf"
    done
  done
done
echo "SWEEP3_DONE task=$TASK"
