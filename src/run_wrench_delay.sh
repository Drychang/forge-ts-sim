#!/usr/bin/env bash
# WRENCH-DELAY SWEEP -- the missing half of the G3 latency study.
#
# G3 swept IMAGE delay only (0/1/2 ticks at 2.5mm noise): 1 tick free, 2 ticks
# costs 4-6 points. But on a real robot the F/T bus has its own latency, and the
# force branch is the modality we claim buys contact safety -- so its delay
# tolerance is a deployment number we currently do not have.
#
# Mirrors the G3 protocol EXACTLY so the two axes are directly comparable:
#   gate2_seed0, noise 2.5mm, dyn_rand on, n=128, num_envs 32, protocol_seed 42,
#   img_delay 0. Only --wrench_delay_steps varies (1/2/4 ticks = 67/133/267 ms).
# d=0 is already published as the G3 baseline, so it is not re-run.
#
# Runs LAST: the RAM guard keeps it behind the P0-1 training arms, which matter
# more. It will simply wait for hours if it has to.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/wrench_delay_master.log
GPU=${1:-2}
declare -A GID=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0 [gear]=Isaac-Forge-GearMesh-TBCamera-v0 [nut]=Isaac-Forge-NutThread-TBCamera-v0 )
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_ram(){
  while true; do
    busy=$(pgrep -fc "train_student_b[c].py|train_student_arc[h].py|eval_frozen_studen[t]|eval_ablate_studen[t]|eval_arch_studen[t]|collect_ta_rollout[s]|collect_camera_rollout[s]|eval_latency_studen[t]" || true)
    ram=$(free -g | awk '/^Mem:/{print $7}')
    if [ "${busy:-0}" -le 1 ] && [ "$ram" -ge 25 ]; then return; fi
    log "wait_ram: camera_jobs=${busy:-0} free=${ram}G"
    sleep 300
  done
}

for t in peg gear nut; do
  ck=~/forge_ts/student_ckpts/$t/gate2_seed0/best.pt
  ns=~/forge_ts/student_ckpts/$t/gate2_seed0/norm_stats.npz
  [ -f "$ck" ] || { log "SKIP $t: no checkpoint"; continue; }
  for d in 1 2 4; do
    lg=$LOGD/wrd_${t}_d${d}.log
    grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP $t d=$d"; continue; }
    wait_ram
    log "wrench_delay $t d=$d START"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_latency_student.py" --headless \
      --task "${GID[$t]}" --checkpoint "$ck" --norm_stats "$ns" \
      --episodes 128 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm 2.5 --dyn_rand on \
      --img_delay_steps 0 --wrench_delay_steps "$d" --tag "wrd_${t}_d${d}" >> "$lg" 2>&1
    log "wrench_delay $t d=$d rc=$?"
  done
done
log "WRENCH_DELAY_ALL_DONE"
