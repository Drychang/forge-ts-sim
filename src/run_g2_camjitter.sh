#!/usr/bin/env bash
# G2: camera-extrinsics sensitivity scan for the gate2_seed0 student checkpoint.
# Step 0 = self-validation (jitter=0 must reproduce the known baseline SR).
# Then a 3-level jitter matrix at a fixed condition (2.5mm, dr_on), 128 eps.
set -u
SRC=~/forge_ts/src/student
LOGS=~/forge_ts/logs
GPU="${GPU:-0}"
mkdir -p "$LOGS"
MASTER_LOG="$LOGS/g2_camjitter_master.log"

source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
export OMNI_KIT_ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SRC"

declare -A GYM=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )

run_one() {
  local task=$1 posmm=$2 rotdeg=$3 tag=$4
  local gym=${GYM[$task]}
  echo "=== [$(date -Is)] START $task pos=${posmm}mm rot=${rotdeg}deg tag=$tag ===" >> "$MASTER_LOG"
  TB_CAM_JITTER_POS_MM=$posmm TB_CAM_JITTER_ROT_DEG=$rotdeg TB_CAM_JITTER_SEED=1 \
    ~/miniconda3/envs/isaaclab/bin/python eval_frozen_student.py --headless \
    --task "Isaac-Forge-${gym}-TBCamera-v0" \
    --checkpoint ~/forge_ts/student_ckpts/${task}/gate2_seed0/best.pt \
    --norm_stats ~/forge_ts/student_ckpts/${task}/gate2_seed0/norm_stats.npz \
    --episodes 128 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm 2.5 --dyn_rand on --tag "$tag" \
    >> "$LOGS/g2_${task}_${tag}.log" 2>&1
  echo "=== [$(date -Is)] END $task pos=${posmm}mm rot=${rotdeg}deg tag=$tag rc=$? ===" >> "$MASTER_LOG"
}

# Step 0: self-validation, one task is enough (mechanism is task-agnostic)
run_one peg 0 0 camjit_validate_zero

# Full matrix
for task in peg gear nut; do
  run_one "$task" 5  1 camjit_p5r1_s1
  run_one "$task" 10 2 camjit_p10r2_s1
  run_one "$task" 20 4 camjit_p20r4_s1
done

echo "G2_CAMJITTER_ALL_DONE [$(date -Is)]" >> "$MASTER_LOG"
