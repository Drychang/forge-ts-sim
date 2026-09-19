#!/usr/bin/env bash
# P0-1 follow-up: evaluate BOTH matched arms at the EXTRAPOLATION levels too.
#
# Rationale: if the privileged teacher's contribution is real, it should show up
# most strongly where the policy must compensate for offsets it was never given
# clean information about -- i.e. beyond the 5mm training ceiling. The 0-5mm
# comparison alone could look flat even when the arms differ, exactly as the main
# student sits at 96-99% across that whole range.
#
# Waits for both arms to finish their own 4-level evals first (P0_ARM_DONE), so
# it never competes with them for RAM. Same frozen protocol throughout.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/p0_extrap_master.log
TASK=peg
GYM=Isaac-Forge-PegInsert-TBCamera-v0
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

log "waiting for both P0-1 arms to finish their 0-5mm evals"
while true; do
  a=$(grep -c "P0_ARM_DONE" $LOGD/p0_ta_master.log 2>/dev/null || echo 0)
  b=$(grep -c "P0_ARM_DONE" $LOGD/p0_tb_master.log 2>/dev/null || echo 0)
  [ "$a" -ge 1 ] && [ "$b" -ge 1 ] && break
  log "waiting: ta_done=$a tb_done=$b"
  sleep 600
done

for arm in ta tb; do
  case $arm in
    ta) OUT=~/forge_ts/student_ckpts/$TASK/ta_teacher_seed0; GPU=1 ;;
    tb) OUT=~/forge_ts/student_ckpts/$TASK/tb_bconly_seed0;  GPU=2 ;;
  esac
  [ -f "$OUT/best.pt" ] || { log "SKIP $arm: no best.pt"; continue; }
  for nm in 7.5 10; do
    lg=$LOGD/p0_${arm}_eval_${TASK}_n${nm}.log
    grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP $arm n=$nm"; continue; }
    while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 22 ]; do
      log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 180
    done
    log "$arm eval n=${nm}mm START"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_frozen_student.py" --headless \
      --task "$GYM" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag p0_${arm}_${TASK}_n${nm} >> "$lg" 2>&1
    log "$arm eval n=${nm} rc=$?"
  done
done
log "P0_EXTRAP_DONE"
