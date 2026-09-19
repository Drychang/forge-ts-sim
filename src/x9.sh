#!/usr/bin/env bash
# x9: does the viewpoint-augmented model hold at the error the real rig actually has?
#
# x8 trained on wrist displacements of 0-25mm and the model stayed flat out to 40mm
# (93.36 vs the control's 23.05), so it generalises past its training range. The real mount
# is about 100mm off -- the camera sits at roughly half the simulator's 198.5mm stand-off --
# which is 4x the widest displacement in the training data and 2.5x the largest tested.
# Whether the invariance reaches that far decides whether the rig can be deployed as-is or
# the mount has to be reprinted first, so it is worth 25 minutes to find out.
#
# Both arms at every level: the control's numbers are what "deploy the existing model"
# looks like, and they are the comparison that makes the augmented ones meaningful.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/peg; LOGD=~/forge_ts/logs
MASTER=$LOGD/x9_master.log; CLAIMS=$LOGD/x9_claims
ENVN=Isaac-Forge-PegInsert-TBCamera-v0
MIN_FREE_MIB=21500
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS" "$LOGD"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }
wait_gpu(){ while [ "$(free_on $1)" -lt "$MIN_FREE_MIB" ]; do log "gpu $1 busy, waiting"; sleep 180; done; }

Q=$LOGD/x9_jobs; : > "$Q"
for A in ctrl jit; do for J in "60 12" "80 16" "100 20"; do
  set -- $J; echo "$A $1 $2 x9_${A}_wj_p$1r$2" >> "$Q"
done; done
log "==== x9 start: $(wc -l < $Q) jobs ===="

order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1)
pids=()
for G in $order; do
 ( n=0
   while IFS= read -r line; do
     n=$((n+1)); mkdir "$CLAIMS/j$n" 2>/dev/null || continue
     set -- $line; A=$1; PM=$2; RD=$3; TAG=$4
     LG=$LOGD/$TAG.log
     grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
     wait_gpu "$G"
     log "EVAL $TAG gpu=$G wrist=${PM}mm/${RD}deg"
     CUDA_VISIBLE_DEVICES=$G TB_CAM_JITTER_POS_MM=$PM TB_CAM_JITTER_ROT_DEG=$RD \
     TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
     "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$ENVN" \
       --checkpoint "$CKPT/jitdata_$A/best.pt" --norm_stats "$CKPT/jitdata_$A/norm_stats.npz" \
       --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm 2.5 \
       --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1
     log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
   done < "$Q" ) &
 pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X9_ALL_DONE ===================="
