#!/usr/bin/env bash
# Noise-spectrum sweep for the frozen eval protocol.
# usage: TASK=Isaac-Forge-PegInsert-Direct-v0 CKPT=/path/to.pth TAG=noisespec_s0 GPU=0 bash run_noise_sweep.sh
set -u

TASK="${TASK:?set TASK}"
CKPT="${CKPT:?set CKPT}"
TAG="${TAG:-noisespec_s0}"
GPU="${GPU:-0}"
SRC_DIR="$HOME/forge_ts/src"
LOG_DIR="$HOME/forge_ts/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=Y
cd "$HOME/force_vla_research/IsaacLab" || exit 1

for noise in 0 1 2.5 5; do
  for dr in on off; do
    logf="$LOG_DIR/${TAG}_$(basename "$TASK")_noise${noise}mm_dr${dr}.log"
    echo "=== $(date -Is) task=$TASK noise=${noise}mm dyn_rand=$dr ===" | tee -a "$logf"
    ./isaaclab.sh -p "$SRC_DIR/eval_frozen.py" \
      --task "$TASK" --checkpoint "$CKPT" \
      --num_envs 128 --episodes 256 --protocol_seed 42 \
      --fixed_pos_noise_mm "$noise" --dyn_rand "$dr" \
      --tag "$TAG" --headless >> "$logf" 2>&1
    rc=$?
    echo "=== exit=$rc ===" | tee -a "$logf"
  done
done
echo "SWEEP_DONE task=$TASK tag=$TAG"
