#!/usr/bin/env bash
# Gate 2 full 3-seed verification driver.
# For each task x seed(0,1,2): train with external early-stopping (RAM-bound,
# fully serialized), then background-launch the 8-condition eval chain on a
# separate GPU so it overlaps with the NEXT seed's training. Evals themselves
# are serialized against each other via flock (cheap on RAM, but keep GPU0
# tidy rather than piling up 3 Isaac Sim instances at once).
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

declare -A GYM=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )

for task in peg gear nut; do
  gym=${GYM[$task]}
  for seed in 0 1 2; do
    outdir="$CKPT_ROOT/$task/gate2_seed${seed}"
    mkdir -p "$outdir"
    cp -n "$CKPT_ROOT/$task/norm_stats.npz" "$outdir/norm_stats.npz"

    echo "=== [$(date -Is)] START train task=$task seed=$seed ===" >> "$MASTER_LOG"
    ~/miniconda3/envs/isaaclab/bin/python -u "$SRC/run_seed_earlystop.py" \
      --task "$task" --seed "$seed" --data_root "$DATA" \
      --out_dir "$outdir" --gpu "$TRAIN_GPU" \
      --patience $PATIENCE --max_epochs $MAX_EPOCHS \
      >> "$MASTER_LOG" 2>&1
    echo "=== [$(date -Is)] END train task=$task seed=$seed ===" >> "$MASTER_LOG"

    if [ ! -f "$outdir/best.pt" ]; then
      echo "=== [$(date -Is)] ERROR no best.pt for task=$task seed=$seed, skipping its eval ===" >> "$MASTER_LOG"
      continue
    fi

    (
      flock -w 10800 200
      echo "=== [$(date -Is)] START eval task=$task seed=$seed ===" >> "$MASTER_LOG"
      bash "$SRC/chain_eval_gate2.sh" "Isaac-Forge-${gym}-TBCamera-v0" "$EVAL_GPU" \
        "$outdir/best.pt" "$outdir/norm_stats.npz" "${task}" "gate2_seed${seed}"
      echo "=== [$(date -Is)] END eval task=$task seed=$seed ===" >> "$MASTER_LOG"
    ) 200>"$LOGS/eval_gate2.lock" &
  done
done

wait
echo "GATE2_3SEED_ALL_DONE [$(date -Is)]" >> "$MASTER_LOG"
