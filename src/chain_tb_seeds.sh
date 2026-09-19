#!/usr/bin/env bash
# Train T-B seed 0 -> 1 -> 2 sequentially on one GPU, checking for completion
# via checkpoint-file existence (NOT a log string marker -- avoids the
# earlier bug where a "done" marker was written to the wrong file).
# usage: TASK=Isaac-Forge-PegInsert-TB-v0 SHORT=peg GPU=0 bash chain_tb_seeds.sh
set -u

TASK="${TASK:?set TASK}"
SHORT="${SHORT:?set SHORT}"
GPU="${GPU:-0}"
FORGE_DIR="$HOME/force_vla_research/IsaacLab/logs/rl_games/Forge"
LOG_DIR="$HOME/forge_ts/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=Y
cd "$HOME/force_vla_research/IsaacLab" || exit 1

for seed in 0 1 2; do
  exp="tb_${SHORT}_s${seed}"
  logf="$LOG_DIR/${exp}.log"
  ckpt_glob="$FORGE_DIR/${exp}/nn/"'*ep_200*.pth'
  echo "=== $(date -Is) starting $exp on GPU$GPU ===" | tee -a "$logf"
  ./isaaclab.sh -p "$HOME/forge_ts/src/train_tb.py" \
    --task "$TASK" --headless --num_envs 128 --seed "$seed" \
    "agent.params.config.full_experiment_name=${exp}" \
    >> "$logf" 2>&1
  rc=$?
  # shellcheck disable=SC2086
  n_ckpt=$(ls $ckpt_glob 2>/dev/null | wc -l)
  echo "=== $(date -Is) $exp exit=$rc ep200_ckpt_count=$n_ckpt ===" | tee -a "$logf"
  if [ "$n_ckpt" -lt 1 ]; then
    echo "=== ABORTING CHAIN: $exp did not produce an ep_200 checkpoint ===" | tee -a "$logf"
    exit 1
  fi
done
echo "TB_CHAIN_DONE task=$TASK short=$SHORT"
