#!/usr/bin/env bash
# Unattended GPU queue after G1 training finished. Fully serialized (camera
# training / camera eval are mutually exclusive on this 62G-RAM box).
# Stages:
#   A) aug_v1 clean-sim verify quick3 x3 tasks (256 eps)   -- the EULA-bug redo
#   B) G1b color-shift comparison: {gate2_seed0, aug_v1} x 3 tasks x 3 color
#      seeds @ 2.5mm/dr_on, 128 eps
#   C) G3 latency: gate2_seed0 x 3 tasks x img_delay {0,1,2} @ 2.5mm/dr_on, 128 eps
#   D) T3 modality ablation: 3 tasks x {vision,force,both}: train seed0 (GPU1)
#      then quick3 ablate-eval (GPU0), serialized per variant
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
CK=~/forge_ts/student_ckpts
LOGS=~/forge_ts/logs
DATA=/media/data/forge_ts_data
ML="$LOGS/post_g1_queue_master.log"
mkdir -p "$LOGS"
cd "$SRC"

declare -A GYM=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )
log() { echo "=== [$(date -Is)] $*" >> "$ML"; }

# ---------- Stage A: aug_v1 clean-sim verify ----------
for task in peg gear nut; do
  gym=${GYM[$task]}
  log "A START verify $task aug_v1"
  for noise in 0 2.5 5; do
    CUDA_VISIBLE_DEVICES=0 $PY "$SRC/eval_frozen_student.py" --headless \
      --task "Isaac-Forge-${gym}-TBCamera-v0" \
      --checkpoint "$CK/$task/aug_v1/best.pt" --norm_stats "$CK/$task/aug_v1/norm_stats.npz" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm $noise --dyn_rand on --tag aug_v1 \
      >> "$LOGS/postg1_A_verify_${task}.log" 2>&1
    log "A $task noise=$noise rc=$?"
  done
done
log "A_DONE"

# ---------- Stage B: G1b color-shift comparison ----------
for task in peg gear nut; do
  gym=${GYM[$task]}
  for model in gate2_seed0 aug_v1; do
    for cs in 1 2 3; do
      log "B START $task $model cs=$cs"
      CUDA_VISIBLE_DEVICES=0 $PY "$SRC/eval_colorshift_student.py" --headless \
        --task "Isaac-Forge-${gym}-TBCamera-v0" \
        --checkpoint "$CK/$task/$model/best.pt" --norm_stats "$CK/$task/$model/norm_stats.npz" \
        --episodes 128 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm 2.5 --dyn_rand on --color_seed $cs --tag "cshift_${model}_cs${cs}" \
        >> "$LOGS/postg1_B_cshift_${task}_${model}_cs${cs}.log" 2>&1
      log "B $task $model cs=$cs rc=$?"
    done
  done
done
log "B_DONE"

# ---------- Stage C: G3 latency ----------
for task in peg gear nut; do
  gym=${GYM[$task]}
  for d in 0 1 2; do
    log "C START $task img_delay=$d"
    CUDA_VISIBLE_DEVICES=0 $PY "$SRC/eval_latency_student.py" --headless \
      --task "Isaac-Forge-${gym}-TBCamera-v0" \
      --checkpoint "$CK/$task/gate2_seed0/best.pt" --norm_stats "$CK/$task/gate2_seed0/norm_stats.npz" \
      --episodes 128 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm 2.5 --dyn_rand on \
      --img_delay_steps $d --wrench_delay_steps 0 --tag "latency_d${d}" \
      >> "$LOGS/postg1_C_latency_${task}_d${d}.log" 2>&1
    log "C $task d=$d rc=$?"
  done
done
log "C_DONE"

# ---------- Stage D: T3 modality ablation ----------
for task in peg gear nut; do
  gym=${GYM[$task]}
  for ab in vision force both; do
    outdir="$CK/$task/ablate_${ab}"
    mkdir -p "$outdir"
    cp -n "$CK/$task/norm_stats.npz" "$outdir/norm_stats.npz"
    log "D START train $task ablate=$ab"
    $PY -u "$SRC/run_seed_earlystop.py" \
      --task "$task" --seed 0 --data_root "$DATA" \
      --out_dir "$outdir" --gpu 1 --patience 6 --max_epochs 30 --ablate "$ab" \
      >> "$LOGS/postg1_D_train_${task}_${ab}.log" 2>&1
    log "D END train $task ablate=$ab rc=$?"
    if [ ! -f "$outdir/best.pt" ]; then
      log "D ERROR no best.pt $task $ab -- skipping its eval"
      continue
    fi
    for noise in 0 2.5 5; do
      CUDA_VISIBLE_DEVICES=0 $PY "$SRC/eval_ablate_student.py" --headless \
        --task "Isaac-Forge-${gym}-TBCamera-v0" \
        --checkpoint "$outdir/best.pt" --norm_stats "$outdir/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm $noise --dyn_rand on --ablate "$ab" --tag "ablate_${ab}" \
        >> "$LOGS/postg1_D_eval_${task}_${ab}.log" 2>&1
      log "D eval $task $ab noise=$noise rc=$?"
    done
  done
done
log "D_DONE"
log "POST_G1_QUEUE_ALL_DONE"
