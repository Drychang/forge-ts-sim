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
  # RAM-only guard. The job-count version counted the two P0-1 trainings' 8
  # dataloader workers as camera jobs, so this sweep would have sat on an IDLE
  # GPU0 for ~3h. Measured: anonymous memory 19G of 62G (the rest is reclaimable
  # NFS page cache), and one camera eval adds ~8.7G -- verified safe by running
  # the last breaking-point cell alongside both trainings.
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
    log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 180
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
