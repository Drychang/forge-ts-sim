#!/usr/bin/env bash
# Real-color screenshot pass. Waits for stage D to fully finish (RAM mutual
# exclusion: never launch a camera Kit while another camera train/eval runs),
# then captures sim-color + real-color frames for all 3 tasks on GPU2.
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOG_DIR=~/forge_ts/logs
OUT=~/forge_ts/screenshots_realcolor
MASTER=$LOG_DIR/realcolor_screenshots_master.log
mkdir -p "$OUT" "$LOG_DIR"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

log "waiting for STAGE_D_REMAINDER_ALL_DONE + no camera procs + RAM>=25G"
while true; do
  if grep -q "STAGE_D_REMAINDER_ALL_DONE" "$LOG_DIR/stage_d_remainder_master.log" 2>/dev/null; then
    if ! pgrep -f "train_student_bc.py|eval_ablate_student.py" >/dev/null 2>&1; then
      free_gb=$(free -g | awk '/^Mem:/{print $7}')
      if [ "$free_gb" -ge 25 ]; then
        break
      fi
      log "stage D done but only ${free_gb}G RAM available; waiting"
    fi
  fi
  sleep 300
done
log "clear to go; starting screenshots on GPU2"

for task_pair in "Isaac-Forge-PegInsert-TBCamera-v0 peg" "Isaac-Forge-GearMesh-TBCamera-v0 gear" "Isaac-Forge-NutThread-TBCamera-v0 nut"; do
  set -- $task_pair
  gymid=$1; short=$2
  for mode in off on; do
    log "screenshot $short realcolor=$mode START"
    CUDA_VISIBLE_DEVICES=2 $PY "$SRC/student/screenshot_realcolor.py" --headless \
      --task "$gymid" --realcolor "$mode" --num_envs 2 --steps 12 \
      --out_dir "$OUT" >> "$LOG_DIR/realcolor_shot_${short}_${mode}.log" 2>&1
    log "screenshot $short realcolor=$mode rc=$?"
  done
done
log "REALCOLOR_SCREENSHOTS_ALL_DONE"
