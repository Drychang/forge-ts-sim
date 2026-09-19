#!/usr/bin/env bash
# G1: photometric-augmentation retraining, 3 tasks fully sequential (camera
# training + camera eval mutually exclusive -- RAM). Each task: train with
# --img_aug v1 into a NEW aug_v1/ dir (never touches gate2_seed*), then
# immediately run a quick3 (0/2.5/5mm x dr_on) clean-sim eval to check for
# regression vs the existing gate2_seed0 checkpoint before moving to the
# next task.
set -u
SRC=~/forge_ts/src/student
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGS=~/forge_ts/logs
GPU="${GPU:-1}"
PATIENCE=6
MAX_EPOCHS=30
export OMNI_KIT_ACCEPT_EULA=Y
mkdir -p "$LOGS"
MASTER_LOG="$LOGS/g1_aug_master.log"

declare -A GYM=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )

for task in peg gear nut; do
  gym=${GYM[$task]}
  outdir="$CKPT_ROOT/${task}/aug_v1"
  mkdir -p "$outdir"
  cp -n "$CKPT_ROOT/${task}/norm_stats.npz" "$outdir/norm_stats.npz"

  echo "=== [$(date -Is)] START train ${task} img_aug=v1 ===" >> "$MASTER_LOG"
  ~/miniconda3/envs/isaaclab/bin/python -u "$SRC/run_seed_earlystop.py" \
    --task "$task" --seed 0 --data_root "$DATA" \
    --out_dir "$outdir" --gpu "$GPU" --patience $PATIENCE --max_epochs $MAX_EPOCHS \
    --img_aug v1 \
    >> "$MASTER_LOG" 2>&1
  echo "=== [$(date -Is)] END train ${task} img_aug=v1 rc=$? ===" >> "$MASTER_LOG"

  if [ ! -f "$outdir/best.pt" ]; then
    echo "=== [$(date -Is)] ERROR no best.pt for ${task} aug_v1, skipping its verification eval ===" >> "$MASTER_LOG"
    continue
  fi

  echo "=== [$(date -Is)] START quick3-verify ${task} aug_v1 (clean sim) ===" >> "$MASTER_LOG"
  for noise in 0 2.5 5; do
    ~/miniconda3/envs/isaaclab/bin/python "$SRC/eval_frozen_student.py" --headless \
      --task "Isaac-Forge-${gym}-TBCamera-v0" \
      --checkpoint "$outdir/best.pt" --norm_stats "$outdir/norm_stats.npz" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm $noise --dyn_rand on --tag aug_v1 \
      >> "$LOGS/g1_verify_${task}.log" 2>&1
  done
  echo "=== [$(date -Is)] END quick3-verify ${task} aug_v1 rc=$? ===" >> "$MASTER_LOG"
done

echo "G1_AUG_PIPELINE_ALL_DONE [$(date -Is)]" >> "$MASTER_LOG"
