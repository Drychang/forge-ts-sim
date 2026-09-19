#!/usr/bin/env bash
# Re-run the one x12 cell that OOMed, pinned to GPU 1 (empty) rather than letting a picker
# choose: the failure was another user's process taking 16.28 GiB mid-run on GPU 0, and the
# eval path had no watchdog, so Isaac hung on the exception for 12 hours and stalled the
# whole chain behind it.
#
# Watchdog added here: an eval is finished when EVAL_SUMMARY appears. If it appears, close
# the app rather than waiting for a shutdown that does not come. If the log shows an
# exception instead, stop immediately -- no point holding a GPU for a dead run.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
LOGD=~/forge_ts/logs; M=~/forge_ts/student_ckpts/gear/jitcam_real
LG=$LOGD/x12_real_at_nom.log; MASTER=$LOGD/x12_redo.log
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
log "==== redo x12_real_at_nom on gpu 1 ===="
CUDA_VISIBLE_DEVICES=1 \
TB_CAM_WRIST_POS="0.130000,0.000000,-0.150000" \
TB_CAM_WRIST_ROT="-0.706140,0.037010,0.037010,-0.706140" \
TB_CAM_JITTER_POS_MM=0 TB_CAM_JITTER_ROT_DEG=0 TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
"$PY" ~/forge_ts/src/student/eval_ablate_student.py --headless \
  --task Isaac-Forge-GearMesh-TBCamera-v0 \
  --checkpoint "$M/best.pt" --norm_stats "$M/norm_stats.npz" \
  --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm 2.5 \
  --dyn_rand on --ablate none --blank_cam none --tag x12_real_at_nom > "$LG" 2>&1 &
pid=$!; w=0
while kill -0 $pid 2>/dev/null; do
  if grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null; then
    sleep 15; kill $pid 2>/dev/null; sleep 8; kill -9 $pid 2>/dev/null
    log "reached EVAL_SUMMARY; closed the hung app"; break
  fi
  if grep -qE "OutOfMemoryError|Traceback" "$LG" 2>/dev/null; then
    sleep 5; kill -9 $pid 2>/dev/null; log "FAILED with an exception -- not waiting"; break
  fi
  sleep 20; w=$((w+20))
  [ $w -gt 5400 ] && { kill -9 $pid 2>/dev/null; log "TIMEOUT"; break; }
done
wait $pid 2>/dev/null
log "result: $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
log "==================== REDO_DONE ===================="
