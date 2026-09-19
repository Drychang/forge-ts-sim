#!/usr/bin/env bash
# x12: evaluate the two gear models, and cross them.
#
# The interesting table is not each model at its own camera pose -- both should look fine
# there -- but each model at the OTHER model's pose. That cross is what says whether
# centring the training distribution on the measured pose actually buys anything, and the
# nom-model-at-real-pose cell is the current state of the rig: a policy trained at the
# nominal viewpoint, run on a camera 142.8mm away from it.
#
#   real @ real   the number that decides whether the physical experiment can start now
#   nom  @ nom    the same thing for the reprint-the-mount path
#   nom  @ real   what running today's model on today's mount would give
#   real @ nom    the mirror; confirms the gain is centring and not something incidental
#
# Chained on X11_TRAIN_DONE rather than launched by hand so the GPUs are not idle between
# the training finishing and someone noticing.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
MASTER=$LOGD/x12_master.log; CLAIMS=$LOGD/x12_claims
TASK=Isaac-Forge-GearMesh-TBCamera-v0
MIN_FREE_MIB=21500
NOM_POS="0.130000,0.000000,-0.150000";  NOM_ROT="-0.706140,0.037010,0.037010,-0.706140"
REAL_POS="0.036666,-0.030747,-0.046328"; REAL_ROT="0.702792,0.013969,0.004503,0.711244"

mkdir -p "$LOGD"; rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }

log "==== x12 waiting for X11_TRAIN_DONE ===="
while ! grep -q "X11_TRAIN_DONE" "$LOGD/x11_master.log" 2>/dev/null; do sleep 300; done
log "==== x11 finished, evaluating ===="

Q=$LOGD/x12_jobs; : > "$Q"
for M in real nom; do
  # own centre: pose-noise sweep, then mount error around that centre
  for NM in 0 1 2.5 5; do echo "$M $M $NM 0 0 x12_${M}_at_${M}_n${NM}" >> "$Q"; done
  for PM in 10 20 40;    do echo "$M $M 2.5 $PM $(awk "BEGIN{printf \"%.1f\",$PM/5}") x12_${M}_at_${M}_wj$PM" >> "$Q"; done
done
# the cross
echo "nom  real 2.5 0 0 x12_nom_at_real"  >> "$Q"
echo "real nom  2.5 0 0 x12_real_at_nom"  >> "$Q"
log "$(wc -l < "$Q") evals queued"

order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1)
pids=()
for G in $order; do
 ( n=0
   while IFS= read -r line; do
     n=$((n+1)); mkdir "$CLAIMS/j$n" 2>/dev/null || continue
     set -- $line; M=$1; C=$2; NM=$3; PM=$4; RD=$5; TAG=$6
     LG=$LOGD/$TAG.log
     grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
     MD=$CKPT/jitcam_$M
     [ -f "$MD/best.pt" ] || { log "SKIP $TAG -- no best.pt in $MD"; continue; }
     if [ "$C" = real ]; then CP=$REAL_POS; CR=$REAL_ROT; else CP=$NOM_POS; CR=$NOM_ROT; fi
     while [ "$(free_on $G)" -lt "$MIN_FREE_MIB" ]; do log "gpu $G busy"; sleep 240; done
     log "EVAL $TAG gpu=$G model=$M centre=$C noise=$NM jitter=${PM}mm"
     CUDA_VISIBLE_DEVICES=$G TB_CAM_WRIST_POS="$CP" TB_CAM_WRIST_ROT="$CR" \
     TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
     TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
     "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
       --checkpoint "$MD/best.pt" --norm_stats "$MD/norm_stats.npz" \
       --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm "$NM" \
       --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1
     log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
   done < "$Q" ) &
 pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X12_ALL_DONE ===================="
