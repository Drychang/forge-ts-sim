#!/usr/bin/env bash
# x13: pin down gear's mount tolerance between 20mm (88.67) and 40mm (2.73).
#
# The spec that goes on the drawing is "how far can the camera be from where the training
# distribution is centred", and right now that is only bracketed to within a factor of two.
# peg's cliff sat at 45-55mm; gear's is clearly earlier, and gear is the task the robot
# actually runs, so the gap matters more here.
#
# jitcam_real evaluated around its own centre -- the real camera pose -- because that is the
# configuration being deployed.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; M=~/forge_ts/student_ckpts/gear/jitcam_real; LOGD=~/forge_ts/logs
MASTER=$LOGD/x13_master.log; CLAIMS=$LOGD/x13_claims
TASK=Isaac-Forge-GearMesh-TBCamera-v0; MIN_FREE_MIB=21500
REAL_POS="0.036666,-0.030747,-0.046328"; REAL_ROT="0.702792,0.013969,0.004503,0.711244"
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS" "$LOGD"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }

log "==== x13 waiting for X12_ALL_DONE ===="
while ! grep -q "X12_ALL_DONE" "$LOGD/x12_master.log" 2>/dev/null; do sleep 240; done
log "==== x12 done, locating gear cliff ===="

Q=$LOGD/x13_jobs; : > "$Q"
for J in "25 5" "30 6" "35 7"; do set -- $J; echo "$1 $2 x13_gear_real_wj$1" >> "$Q"; done
order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1)
pids=()
for G in $order; do
 ( n=0
   while IFS= read -r line; do
     n=$((n+1)); mkdir "$CLAIMS/j$n" 2>/dev/null || continue
     set -- $line; PM=$1; RD=$2; TAG=$3; LG=$LOGD/$TAG.log
     grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
     while [ "$(free_on $G)" -lt "$MIN_FREE_MIB" ]; do log "gpu $G busy"; sleep 240; done
     log "EVAL $TAG gpu=$G wrist=+${PM}mm/${RD}deg from the real pose"
     CUDA_VISIBLE_DEVICES=$G TB_CAM_WRIST_POS="$REAL_POS" TB_CAM_WRIST_ROT="$REAL_ROT" \
     TB_CAM_JITTER_POS_MM=$PM TB_CAM_JITTER_ROT_DEG=$RD \
     TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
     "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
       --checkpoint "$M/best.pt" --norm_stats "$M/norm_stats.npz" \
       --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm 2.5 \
       --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1
     log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
   done < "$Q" ) &
 pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X13_ALL_DONE ===================="
