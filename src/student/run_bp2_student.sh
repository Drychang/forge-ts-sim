#!/usr/bin/env bash
# BREAKING-POINT experiment (student): eval the EXISTING gate2_seed0 student at
# noise levels BEYOND the 5mm training/eval ceiling (7.5, 10mm) to find where
# our method degrades. Eval-only, no retraining. NOTE: these are OUT of the
# teacher's training noise range (tb_noise_std_max = 5mm), so they are an
# extrapolation / breaking-point probe -- report as such.
# Camera eval is RAM-bound -> strictly sequential.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/bp2_student_master.log
GPU=${1:-0}
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }
log "waiting for the 7.5/10mm chain to finish first"
while ! grep -q "BP_STUDENT_ALL_DONE" ~/forge_ts/logs/bp_student_master.log 2>/dev/null; do sleep 180; done
wait_ram(){
  while true; do
    busy=$(pgrep -f "train_student_bc.py|train_student_arch.py|eval_frozen_student|eval_ablate_student|eval_arch_student" | grep -v $$ | wc -l)
    ram=$(free -g | awk '/^Mem:/{print $7}')
    [ "$busy" -le 1 ] && [ "$ram" -ge 25 ] && return
    log "waiting: camera_jobs=$busy ram=${ram}G"; sleep 300
  done
}
declare -A GID=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0 [gear]=Isaac-Forge-GearMesh-TBCamera-v0 [nut]=Isaac-Forge-NutThread-TBCamera-v0 )
for t in peg gear nut; do
  ck=~/forge_ts/student_ckpts/$t/gate2_seed0/best.pt
  ns=~/forge_ts/student_ckpts/$t/gate2_seed0/norm_stats.npz
  for nm in 15 20; do
    lg=$LOGD/bp_student_${t}_n${nm}.log
    grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null && { log "SKIP $t n=$nm"; continue; }
    wait_ram
    log "BP student $t noise=${nm}mm START"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_frozen_student.py" --headless \
      --task "${GID[$t]}" --checkpoint "$ck" --norm_stats "$ns" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "bp_${t}_n${nm}" >> "$lg" 2>&1
    log "BP student $t noise=${nm}mm rc=$?"
  done
done
log "BP2_STUDENT_ALL_DONE"
