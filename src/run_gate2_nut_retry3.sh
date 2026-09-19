#!/usr/bin/env bash
# Second-line Nut retry for whichever seeds still lack a best.pt after the
# first (already-serialized) retry. RAM/swap got pushed to the brink even
# fully serialized (58GB used, swap 31/31GB full) -- num_workers=8 forks 8
# copies of dataloader worker state on top of an already-huge eager-loaded
# dataset. Dropping num_workers reduces fork/COW pressure at some data-
# loading throughput cost. Only (re)trains seeds passed as args; does not
# touch seeds that already have a good best.pt.
set -u

SRC=~/forge_ts/src/student
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGS=~/forge_ts/logs
TRAIN_GPU=1
EVAL_GPU=0
PATIENCE=6
MAX_EPOCHS=30
NUM_WORKERS=8

mkdir -p "$LOGS"
MASTER_LOG="$LOGS/gate2_master.log"

for seed in "$@"; do
  outdir="$CKPT_ROOT/nut/gate2_seed${seed}"
  mkdir -p "$outdir"
  rm -f "$outdir/best.pt" "$outdir/last.pt" "$outdir/train.log"
  cp -n "$CKPT_ROOT/nut/norm_stats.npz" "$outdir/norm_stats.npz"

  echo "=== [$(date -Is)] RETRY3(nw=$NUM_WORKERS) START train task=nut seed=$seed ===" >> "$MASTER_LOG"
  ~/miniconda3/envs/isaaclab/bin/python -u "$SRC/run_seed_earlystop.py" \
    --task nut --seed "$seed" --data_root "$DATA" \
    --out_dir "$outdir" --gpu "$TRAIN_GPU" \
    --patience $PATIENCE --max_epochs $MAX_EPOCHS --num_workers $NUM_WORKERS \
    >> "$MASTER_LOG" 2>&1
  echo "=== [$(date -Is)] RETRY3 END train task=nut seed=$seed ===" >> "$MASTER_LOG"

  if [ ! -f "$outdir/best.pt" ]; then
    echo "=== [$(date -Is)] RETRY3 ERROR no best.pt for task=nut seed=$seed, skipping its eval ===" >> "$MASTER_LOG"
    continue
  fi

  echo "=== [$(date -Is)] RETRY3 START eval task=nut seed=$seed ===" >> "$MASTER_LOG"
  bash "$SRC/chain_eval_gate2.sh" "Isaac-Forge-NutThread-TBCamera-v0" "$EVAL_GPU" \
    "$outdir/best.pt" "$outdir/norm_stats.npz" "nut" "gate2_seed${seed}"
  echo "=== [$(date -Is)] RETRY3 END eval task=nut seed=$seed ===" >> "$MASTER_LOG"
done

echo "GATE2_NUT_RETRY3_DONE seeds=$* [$(date -Is)]" >> "$MASTER_LOG"
