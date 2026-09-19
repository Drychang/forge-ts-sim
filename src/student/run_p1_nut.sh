#!/usr/bin/env bash
# P1 architecture ablation on NUT (hardest task: 449-step episodes, 15GB data): train concat + act with the SAME recipe as
# the reference FMT student (gate2), SEQUENTIALLY (camera training is RAM-bound
# -- never run two camera jobs at once, per the project's standing rule).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/p1_nut_master.log
DATA=/media/data/forge_ts_data
GPU=${1:-0}
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_ram(){  # camera training needs headroom; never start if another camera job is up
  while true; do
    busy=$(pgrep -f "train_student_bc.py|train_student_arch.py|eval_frozen_student|eval_ablate_student" | grep -v $$ | wc -l)
    ram=$(free -g | awk '/^Mem:/{print $7}')
    [ "$busy" -le 1 ] && [ "$ram" -ge 35 ] && return
    log "waiting: camera_jobs=$busy ram_avail=${ram}G"
    sleep 300
  done
}

for arch in concat act; do
  out=~/forge_ts/student_ckpts/nut/p1_${arch}
  if [ -f "$out/best.pt" ]; then log "SKIP $arch (best.pt exists)"; continue; fi
  wait_ram
  log "P1 train nut arch=$arch on GPU$GPU START"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/run_seed_earlystop.py" \
    --data_root "$DATA" --task nut --out_dir "$out" --gpu "$GPU" \
    --patience 6 --max_epochs 30 --seed 0 \
    --trainer "$SRC/train_student_arch.py" --extra_args "--arch $arch" \
    >> "$LOGD/p1_train_nut_${arch}.log" 2>&1
  log "P1 train nut arch=$arch rc=$?"
done
log "P1_NUT_TRAIN_ALL_DONE"
