#!/usr/bin/env bash
# Real-color rehearsal v2: same as run_realcolor_eval.sh but with the table
# flat-recolored to light cream (matching the author's real desk mat), testing the
# hypothesis that peg's v1 drop was a black-on-dark contrast problem.
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOG_DIR=~/forge_ts/logs
MASTER=$LOG_DIR/realcolor_lt_eval_master.log
TABLE="0.92,0.90,0.86"
mkdir -p "$LOG_DIR"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_ready(){  # RAM mutual exclusion + VRAM headroom on GPU1
  while true; do
    busy=$(pgrep -f "train_student_bc.py|eval_ablate_student.py|screenshot_realcolor.py" | wc -l)
    vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
    ram=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$busy" -eq 0 ] && [ "$vram" -ge 15000 ] && [ "$ram" -ge 25 ]; then return; fi
    log "waiting: busy_procs=$busy gpu1_free=${vram}MiB ram_avail=${ram}G"
    sleep 300
  done
}

run_one(){  # task_short gym_id noise
  local t=$1 gymid=$2 noise=$3
  local lg=$LOG_DIR/realcolor_lt_eval_${t}_n${noise}.log
  if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then
    log "SKIP $t noise=$noise (EVAL_SUMMARY already in log)"
    return 0
  fi
  wait_ready
  log "realcolor_lt eval $t noise=$noise START"
  CUDA_VISIBLE_DEVICES=1 $PY "$SRC/student/eval_realcolor_student.py" --headless \
    --task "$gymid" \
    --checkpoint ~/forge_ts/student_ckpts/$t/aug_v1/best.pt \
    --norm_stats ~/forge_ts/student_ckpts/$t/aug_v1/norm_stats.npz \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$noise" --dyn_rand on --tag realcolor_lt_aug_v1 \
    --table_color "$TABLE" \
    >> "$lg" 2>&1
  log "realcolor_lt eval $t noise=$noise rc=$?"
}

for pair in "peg Isaac-Forge-PegInsert-TBCamera-v0" "gear Isaac-Forge-GearMesh-TBCamera-v0" "nut Isaac-Forge-NutThread-TBCamera-v0"; do
  set -- $pair
  for noise in 0 2.5 5; do
    run_one "$1" "$2" "$noise"
  done
done
log "REALCOLOR_LT_EVAL_ALL_DONE"
