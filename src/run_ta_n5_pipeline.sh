#!/usr/bin/env bash
# Noise-augmented T-A baseline: train 3 seeds at 5mm obs noise (vs official 1mm),
# then run the same 8-condition x 3-seed frozen eval sweep as the original T-A/T-B.
# Purpose: answer reviewer R1 -- "baseline wasn't trained under high noise, of
# course it degrades at test time." If this ALSO holds up at 5mm, the whole
# vision+force thesis needs rethinking; if it stays low like official T-A, the
# contribution is solid.
# usage: TASK=Isaac-Forge-PegInsert-Direct-v0 SHORT=peg GPU=0 bash run_ta_n5_pipeline.sh
set -u
TASK="${TASK:?set TASK}"
SHORT="${SHORT:?set SHORT}"
GPU="${GPU:-0}"
SRC=/home/user/force_vla_research/IsaacLab
LOG_DIR=/home/user/forge_ts/logs
FORGE_DIR=/home/user/force_vla_research/IsaacLab/logs/rl_games/Forge
SRC_DIR="$HOME/forge_ts/src"
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd "$SRC" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=Y

MASTER_LOG="$LOG_DIR/ta_n5_${SHORT}_master.log"

for seed in 0 1 2; do
  echo "=== [$(date -Is)] START train ta_n5 ${SHORT} seed=${seed} ===" >> "$MASTER_LOG"
  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task "$TASK" --headless --num_envs 128 --seed "$seed" \
    agent.params.config.full_experiment_name=ta_n5_${SHORT}_s${seed} \
    env.obs_rand.fixed_asset_pos="[0.005,0.005,0.005]" \
    > "$LOG_DIR/ta_n5_${SHORT}_s${seed}.log" 2>&1
  rc=$?
  echo "=== [$(date -Is)] END train ta_n5 ${SHORT} seed=${seed} rc=${rc} ===" >> "$MASTER_LOG"
done

echo "=== [$(date -Is)] START eval sweep ta_n5 ${SHORT} ===" >> "$MASTER_LOG"
for seed in 0 1 2; do
  ckpt=$(ls "$FORGE_DIR/ta_n5_${SHORT}_s${seed}/nn/"*ep_200*.pth 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then
    echo "=== [$(date -Is)] SKIP eval seed=$seed: no ep_200 checkpoint for ta_n5_${SHORT}_s${seed} ===" >> "$MASTER_LOG"
    continue
  fi
  tag="ta_n5_noisespec_s${seed}"
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
echo "TA_N5_PIPELINE_DONE task=$TASK short=$SHORT [$(date -Is)]" >> "$MASTER_LOG"
