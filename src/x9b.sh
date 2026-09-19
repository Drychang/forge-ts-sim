#!/usr/bin/env bash
# x9b: locate the tolerance cliff of the viewpoint-augmented model.
#
# It holds to 40mm (93.36) and is exactly 0.00 by 60mm, so the usable band ends somewhere
# between. That boundary is the build spec for the reprinted wrist mount -- "within 45mm"
# and "within 55mm" are different manufacturing problems -- so it is worth four evals.
#
# jit+full only: it is the deployment candidate. x10 showed the -state variant loses ~34
# points at baseline on this data, and ctrl+full is already at 23.05 by 40mm.
#
# Written as a new script rather than by editing x9's queue file: x9.sh regenerates that
# queue on every launch, so a hand-edited queue is silently overwritten and every job is
# skipped as already-done.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; M=~/forge_ts/student_ckpts/peg/jitdata_jit; LOGD=~/forge_ts/logs
MASTER=$LOGD/x9b_master.log; CLAIMS=$LOGD/x9b_claims
ENVN=Isaac-Forge-PegInsert-TBCamera-v0; MIN_FREE_MIB=21500
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS" "$LOGD"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }

Q=$LOGD/x9b_jobs; : > "$Q"
for J in "45 9" "50 10" "55 11" "48 9.6"; do set -- $J; echo "$1 $2 x9b_jit_wj_p$1r$2" >> "$Q"; done
log "==== x9b start: $(wc -l < $Q) jobs ===="
order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1)
pids=()
for G in $order; do
 ( n=0
   while IFS= read -r line; do
     n=$((n+1)); mkdir "$CLAIMS/j$n" 2>/dev/null || continue
     set -- $line; PM=$1; RD=$2; TAG=$3; LG=$LOGD/$TAG.log
     grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
     while [ "$(free_on $G)" -lt "$MIN_FREE_MIB" ]; do log "gpu $G busy"; sleep 180; done
     log "EVAL $TAG gpu=$G wrist=${PM}mm/${RD}deg"
     CUDA_VISIBLE_DEVICES=$G TB_CAM_JITTER_POS_MM=$PM TB_CAM_JITTER_ROT_DEG=$RD \
     TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
     "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$ENVN" \
       --checkpoint "$M/best.pt" --norm_stats "$M/norm_stats.npz" \
       --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm 2.5 \
       --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1
     log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
   done < "$Q" ) &
 pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X9B_ALL_DONE ===================="
