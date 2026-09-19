#!/usr/bin/env bash
# Render-throughput benchmark, ONE env count per process — the safe path around
# kit's flaky in-process teardown of RTX sensors between env builds.
#
#   TASK=peg COUNTS="32 48 64" GPU=0 ISAACLAB_DIR=~/IsaacLab bash run_bench.sh
set -u

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK="${TASK:-peg}"
STEPS="${STEPS:-120}"
COUNTS="${COUNTS:-32 48 64}"
GPU="${GPU:-0}"
LOG="$SRC_DIR/bench_render_${TASK}.log"

for n in $COUNTS; do
    echo "=== bench task=$TASK num_envs=$n steps=$STEPS $(date -Is) ===" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" OMNI_KIT_ACCEPT_EULA=Y \
        "$ISAACLAB_DIR/isaaclab.sh" -p "$SRC_DIR/bench_render.py" \
        --headless --task_variant "$TASK" --env_counts "$n" --steps "$STEPS" \
        2>&1 | tee -a "$LOG"
    # non-zero exit (e.g. OOM at high counts) should not stop the sweep
done

echo "=== sweep done; grep '^|' $LOG for the per-count markdown rows ==="
