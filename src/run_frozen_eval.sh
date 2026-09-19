#!/usr/bin/env bash
# Frozen eval wrapper.
# usage: run_frozen_eval.sh TASK CKPT [NOISE_MM] [DYN_RAND] [GPU]
#   TASK      e.g. Isaac-Forge-PegInsert-Direct-v0
#   CKPT      path to rl_games .pth
#   NOISE_MM  fixed-asset pos obs noise std in mm (default 1.0)
#   DYN_RAND  on|off (default on)
#   GPU       CUDA device index (default 0)
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 TASK CKPT [NOISE_MM] [DYN_RAND] [GPU]" >&2
    exit 1
fi

TASK="$1"
CKPT="$2"
NOISE_MM="${3:-1.0}"
DYN_RAND="${4:-on}"
GPU="${5:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=Y

cd "$HOME/force_vla_research/IsaacLab"

echo "[run_frozen_eval] task=$TASK ckpt=$CKPT noise_mm=$NOISE_MM dyn_rand=$DYN_RAND gpu=$GPU"
./isaaclab.sh -p "$HOME/forge_ts/src/eval_frozen.py" \
    --task "$TASK" \
    --checkpoint "$CKPT" \
    --fixed_pos_noise_mm "$NOISE_MM" \
    --dyn_rand "$DYN_RAND" \
    --headless
