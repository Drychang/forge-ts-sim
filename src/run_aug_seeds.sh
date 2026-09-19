#!/usr/bin/env bash
# aug_v1 seeds 1 and 2 for one task.  usage: run_aug_seeds.sh <peg|gear|nut> <gpu>
#
# Why this one first: aug_v1 is the model that actually goes on the robot -- the
# deployment decision (photometric augmentation, dark table) rests on it, and
# without augmentation gear collapses to 6-10% under the real 3D-printed part
# colours. It is also the ONLY arm still at a single seed. Tonight's BC-only run
# showed a single seed can span 14 points, so shipping a one-seed model with no
# variance estimate is the largest remaining risk in the deployment package.
#
# Exactly the g1 recipe, only the seed differs: same full BC+DAgger dataset,
# --img_aug v1, patience 6, max_epochs 30. Evaluated on the SAME three conditions
# the deployment decision uses -- clean sim, real part colours (dark table), and
# real part colours (bright table) -- not just clean sim.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
DATA=/media/data/forge_ts_data
TASK=${1:?task: peg|gear|nut}
GPU=${2:?gpu}
MASTER=$LOGD/aug_seeds_${TASK}_master.log
declare -A GID=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0 [gear]=Isaac-Forge-GearMesh-TBCamera-v0 [nut]=Isaac-Forge-NutThread-TBCamera-v0 )
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }
wait_mem(){ while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
  log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 240; done; }

for seed in 1 2; do
  OUT=~/forge_ts/student_ckpts/$TASK/aug_v1_s${seed}
  if [ -f "$OUT/best.pt" ]; then
    log "SKIP train seed=$seed"
  else
    wait_mem
    mkdir -p "$OUT"
    log "train $TASK aug_v1 seed=$seed gpu=$GPU"
    $PY -u "$SRC/student/run_seed_earlystop.py" \
      --task "$TASK" --seed "$seed" --data_root "$DATA" \
      --out_dir "$OUT" --gpu "$GPU" --patience 6 --max_epochs 30 \
      --img_aug v1 --num_workers 4 > $LOGD/aug_${TASK}_s${seed}_train.log 2>&1
    log "train seed=$seed rc=$? best=$([ -f $OUT/best.pt ] && echo yes || echo NO)"
  fi
  [ -f "$OUT/best.pt" ] || { log "no best.pt seed=$seed"; continue; }

  # clean sim
  for nm in 0 2.5 5; do
    lg=$LOGD/aug_${TASK}_s${seed}_clean_n${nm}.log
    grep -q EVAL_SUMMARY "$lg" 2>/dev/null && continue
    wait_mem
    log "clean eval seed=$seed n=${nm}mm"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/student/eval_frozen_student.py" --headless \
      --task "${GID[$TASK]}" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag aug_${TASK}_s${seed}_n${nm} >> "$lg" 2>&1
    log "clean n=${nm} rc=$?"
  done

  # real part colours, dark table -- the actual deployment condition
  for nm in 0 2.5 5; do
    lg=$LOGD/aug_${TASK}_s${seed}_rc_n${nm}.log
    grep -q EVAL_SUMMARY "$lg" 2>/dev/null && continue
    wait_mem
    log "realcolor eval seed=$seed n=${nm}mm"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/student/eval_realcolor_student.py" --headless \
      --task "${GID[$TASK]}" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag aug_rc_${TASK}_s${seed}_n${nm} >> "$lg" 2>&1
    log "realcolor n=${nm} rc=$?"
  done
done
log "AUG_SEEDS_DONE task=$TASK"
