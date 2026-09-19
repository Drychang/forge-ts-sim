#!/usr/bin/env bash
# P1 arch-ablation frozen-protocol eval: concat + act on peg x {0,1,2.5,5}mm.
# Camera eval is RAM-bound -> wait for headroom and never run two at once.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/p1_eval_master.log
GPU=${1:-0}
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }
wait_ram(){
  while true; do
    busy=$(pgrep -f "train_student_bc.py|train_student_arch.py|eval_frozen_student|eval_ablate_student|eval_arch_student" | grep -v $$ | wc -l)
    ram=$(free -g | awk '/^Mem:/{print $7}')
    [ "$busy" -le 1 ] && [ "$ram" -ge 20 ] && return
    log "waiting: camera_jobs=$busy ram=${ram}G"; sleep 300
  done
}
for arch in concat act; do
  ck=~/forge_ts/student_ckpts/peg/p1_${arch}/best.pt
  ns=~/forge_ts/student_ckpts/peg/p1_${arch}/norm_stats.npz
  [ -f "$ck" ] || { log "no ckpt for $arch"; continue; }
  for nm in 0 1 2.5 5; do
    lg=$LOGD/p1_eval_peg_${arch}_n${nm}.log
    grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null && { log "SKIP $arch n=$nm"; continue; }
    wait_ram
    log "P1 eval peg arch=$arch noise=$nm START"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_arch_student.py" --headless \
      --task Isaac-Forge-PegInsert-TBCamera-v0 --checkpoint "$ck" --norm_stats "$ns" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "p1_${arch}_n${nm}" >> "$lg" 2>&1
    log "P1 eval peg arch=$arch noise=$nm rc=$?"
  done
done
log "P1_EVAL_ALL_DONE"
echo "--- P1 peg matrix (SR) ---" >> "$MASTER"
for arch in concat act; do for nm in 0 1 2.5 5; do
  sr=$(grep -oE "\"sr\": [0-9.]+" $LOGD/p1_eval_peg_${arch}_n${nm}.log 2>/dev/null|head -1)
  echo "$arch n=$nm $sr" >> "$MASTER"
done; done
