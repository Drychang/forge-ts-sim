#!/usr/bin/env bash
# Record 3 success + 3 failure labeled clips for every (task x noise) condition,
# for one model type (ta or noise-aug-ta covered via MODEL_TYPE/CKPT_ROOT args).
# usage: MODEL_TYPE=ta GPU=0 bash run_record_all.sh
#        MODEL_TYPE=student GPU=1 bash run_record_all.sh
set -u
MODEL_TYPE="${MODEL_TYPE:?set MODEL_TYPE (ta|student)}"
GPU="${GPU:-0}"
SRC=/home/user/force_vla_research/IsaacLab
LOG_DIR=/home/user/forge_ts/logs
OUT_DIR=/home/user/forge_ts/videos
FORGE_DIR=/home/user/force_vla_research/IsaacLab/logs/rl_games/Forge
mkdir -p "$LOG_DIR" "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd "$SRC" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
export OMNI_KIT_ACCEPT_EULA=Y

MASTER_LOG="$LOG_DIR/record_${MODEL_TYPE}_master.log"

declare -A GYM=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )
declare -A LABEL=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )

for task in peg gear nut; do
  gym=${GYM[$task]}
  label_task=${LABEL[$task]}
  for noise in 0 1 2.5 5; do
    echo "=== [$(date -Is)] START ${MODEL_TYPE} ${task} noise=${noise} ===" >> "$MASTER_LOG"
    if [ "$MODEL_TYPE" = "ta" ]; then
      ckpt=$(ls "$FORGE_DIR/ta_n5_${task}_s0/nn/"*ep_200*.pth 2>/dev/null | head -1)
      ./isaaclab.sh -p ~/forge_ts/src/record_labeled_clips.py --headless \
        --task "Isaac-Forge-${gym}-Direct-v0" --model_type ta \
        --checkpoint "$ckpt" \
        --fixed_pos_noise_mm "$noise" --dyn_rand on --n_success 3 --n_fail 3 --max_episodes 80 \
        --label_model NoiseAugTA --label_task "$label_task" \
        --out_dir "$OUT_DIR" >> "$LOG_DIR/record_ta_${task}_n${noise}.log" 2>&1
    else
      ./isaaclab.sh -p ~/forge_ts/src/record_labeled_clips.py --headless \
        --task "Isaac-Forge-${gym}-TBCamera-v0" --model_type student \
        --checkpoint "$HOME/forge_ts/student_ckpts/${task}/gate2_seed0/best.pt" \
        --norm_stats "$HOME/forge_ts/student_ckpts/${task}/gate2_seed0/norm_stats.npz" \
        --fixed_pos_noise_mm "$noise" --dyn_rand on --n_success 3 --n_fail 3 --max_episodes 80 \
        --label_model Student --label_task "$label_task" \
        --out_dir "$OUT_DIR" >> "$LOG_DIR/record_student_${task}_n${noise}.log" 2>&1
    fi
    echo "=== [$(date -Is)] END ${MODEL_TYPE} ${task} noise=${noise} rc=$? ===" >> "$MASTER_LOG"
  done
done
echo "RECORD_ALL_DONE model_type=${MODEL_TYPE} [$(date -Is)]" >> "$MASTER_LOG"
