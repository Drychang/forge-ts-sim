#!/usr/bin/env bash
# Nut-only Gate 2 retry, FULLY serialized (train -> its own eval -> next seed's
# train). The original run_gate2_3seed.sh let eval overlap with the next
# seed's training on the assumption eval is RAM-light; that assumption broke
# down at Nut's data scale (1.24M train transitions) -- all 3 seeds got
# OOM-killed (rc=-9) before finishing epoch 1. No overlap this time.
set -u

SRC=~/forge_ts/src/student
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGS=~/forge_ts/logs
TRAIN_GPU=1
EVAL_GPU=0
PATIENCE=6
MAX_EPOCHS=30

mkdir -p "$LOGS"
MASTER_LOG="$LOGS/gate2_master.log"

for seed in 0 1 2; do
  outdir="$CKPT_ROOT/nut/gate2_seed${seed}"
  mkdir -p "$outdir"
  cp -n "$CKPT_ROOT/nut/norm_stats.npz" "$outdir/norm_stats.npz"

  echo "=== [$(date -Is)] RETRY START train task=nut seed=$seed ===" >> "$MASTER_LOG"
  ~/miniconda3/envs/isaaclab/bin/python -u "$SRC/run_seed_earlystop.py" \
    --task nut --seed "$seed" --data_root "$DATA" \
    --out_dir "$outdir" --gpu "$TRAIN_GPU" \
    --patience $PATIENCE --max_epochs $MAX_EPOCHS \
    >> "$MASTER_LOG" 2>&1
  echo "=== [$(date -Is)] RETRY END train task=nut seed=$seed ===" >> "$MASTER_LOG"

  if [ ! -f "$outdir/best.pt" ]; then
    echo "=== [$(date -Is)] RETRY ERROR no best.pt for task=nut seed=$seed, skipping its eval ===" >> "$MASTER_LOG"
    continue
  fi

  echo "=== [$(date -Is)] RETRY START eval task=nut seed=$seed ===" >> "$MASTER_LOG"
  bash "$SRC/chain_eval_gate2.sh" "Isaac-Forge-NutThread-TBCamera-v0" "$EVAL_GPU" \
    "$outdir/best.pt" "$outdir/norm_stats.npz" "nut" "gate2_seed${seed}"
  echo "=== [$(date -Is)] RETRY END eval task=nut seed=$seed ===" >> "$MASTER_LOG"
done

echo "GATE2_NUT_RETRY_ALL_DONE [$(date -Is)]" >> "$MASTER_LOG"
