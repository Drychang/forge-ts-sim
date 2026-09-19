#!/usr/bin/env bash
# Missing control for the deployment decision: what does the photometric
# augmentation actually buy IN THE DEPLOYMENT CONDITION?
#
# The real-colour rehearsal only ever evaluated aug_v1. Its colour robustness was
# measured against SYNTHETIC shifts (cs1/cs2/cs3), where gate2 collapses to
# 0-79.7%. But nobody ran gate2 under the ACTUAL 3D-printed part colours, so the
# gap in the condition that will really be deployed is unmeasured -- and that is
# exactly the number needed to justify "ship aug_v1, not gate2".
#
# Same recolored assets, same dark table, same frozen protocol. Eval-only, one
# cell at a time on a dedicated GPU (an Isaac eval sharing a GPU with another
# heavy CUDA process dies with create_articulation_view on None).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
MASTER=$LOGD/realcolor_gate2_master.log
GPU=${1:-1}
declare -A GID=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0 [gear]=Isaac-Forge-GearMesh-TBCamera-v0 [nut]=Isaac-Forge-NutThread-TBCamera-v0 )
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

for t in peg gear nut; do
  for nm in 0 2.5 5; do
    lg=$LOGD/realcolor_gate2_${t}_n${nm}.log
    grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP $t n=$nm"; continue; }
    while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
      log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 240
    done
    log "gate2 realcolor $t n=${nm}mm START"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/student/eval_realcolor_student.py" --headless \
      --task "${GID[$t]}" \
      --checkpoint ~/forge_ts/student_ckpts/$t/gate2_seed0/best.pt \
      --norm_stats ~/forge_ts/student_ckpts/$t/gate2_seed0/norm_stats.npz \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag rc_gate2_${t}_n${nm} >> "$lg" 2>&1
    log "gate2 realcolor $t n=${nm} rc=$?"
  done
done
log "REALCOLOR_GATE2_DONE"
