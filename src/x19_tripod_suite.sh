#!/usr/bin/env bash
# x19: WHERE DO I PUT THE TRIPOD? -- measured, not reasoned.
#
# Context. x17 measured jitcam_real_s2 with the TP camera at its real calibrated pose
# and the real 55.7 deg lens: sr=0.7461, against 0.8711 at sim nominal. -12.5 points.
#
# Projecting the fixture through the real intrinsics shows why: subject DISTANCE is
# already right (0.704 m real vs 0.700 m sim) and horizontal aim is right (13 px off
# center), but the camera is pitched 8.6 deg too high, dropping the fixture to 70% down
# the frame. That is CLAUDE.md 8's "third-person composition FAIL".
#
# Two competing fixes, and CLAUDE.md 9 picked one by REASONING, never by measurement:
#
#   A  RE-AIM ONLY   tripod stays put, tilt down until the fixture centers.
#                    -> viewing geometry matches training exactly
#                    -> but the 55.7 deg lens makes the subject 0.827x smaller than trained
#
#   B  MOVE CLOSER   CLAUDE.md 9's advice: scale standoff by 0.827 to restore apparent
#                    size (its x=+330mm / z=+290mm).
#                    -> subject size matches training
#                    -> but perspective/parallax now differ from training
#
# Both keep the real 55.7 deg HFOV, because the lens cannot be changed.
# Wrist stays at its real calibrated pose throughout -- it cannot move without a reprint.
#
# Geometry for B: sim nominal cam (1.0,0,0.4) aims along (-0.866,0,-0.5), meeting the
# task plane at (0.3938,0,0.05), 0.700 m away. Scaling that standoff by 0.827 gives
# 0.579 m, i.e. cam at (0.8952,0,0.3395), orientation unchanged.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
TASK=Isaac-Forge-GearMesh-TBCamera-v0
CK=$CKPT/jitcam_real_s2
MASTER=$LOGD/x19_master.log; LOCK=$LOGD/x16_gpulocks
MIN_FREE_MIB=15000; MIN_RAM_GB=12; EVAL_TIMEOUT=5400
REAL_HFOV=55.7
WRIST_POS="0.036666,-0.030747,-0.046328"
WRIST_ROT="0.702792,0.013969,0.004503,0.711244"
TP_ROT="0.35355,-0.61237,-0.61237,0.35355"     # nominal orientation, both strategies
A_POS="1.0,0.0,0.4"                             # A: re-aim only, nominal standoff
B_POS="0.8952,0.0,0.3395"                       # B: 0.827x standoff, size-matched

mkdir -p "$LOGD" "$LOCK"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }
ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }

claim(){
  local who="$1" G order
  while :; do
    if [ "$(ram_gb)" -ge "$MIN_RAM_GB" ]; then
      order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1)
      for G in $order; do
        [ "$(free_on "$G")" -ge "$MIN_FREE_MIB" ] || continue
        mkdir "$LOCK/g$G" 2>/dev/null || continue
        if [ "$(free_on "$G")" -lt "$MIN_FREE_MIB" ]; then rmdir "$LOCK/g$G" 2>/dev/null; continue; fi
        echo "$G"; return 0
      done
    fi
    log "$who waiting: ram=$(ram_gb)GB free=[$(free_on 0),$(free_on 1),$(free_on 2)]MiB"
    sleep 120
  done
}
release(){ rmdir "$LOCK/g$1" 2>/dev/null; }

# $1=gpu  $2=tp_pos  $3=tag  $4=human label
run_eval(){
  local G="$1" TPP="$2" TAG="$3" WHAT="$4"
  local LG="$LOGD/$TAG.log"
  [ -f "$LG" ] && grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG (done)"; return 0; }
  log "EVAL $TAG gpu=$G : $WHAT  tp_pos=$TPP hfov=$REAL_HFOV"
  CUDA_VISIBLE_DEVICES="$G" \
  TB_CAM_WRIST_POS="$WRIST_POS" TB_CAM_WRIST_ROT="$WRIST_ROT" \
  TB_CAM_TP_POS="$TPP" TB_CAM_TP_ROT="$TP_ROT" \
  TB_CAM_HFOV_DEG="$REAL_HFOV" \
  "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
    --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
    --episodes 256 --num_envs 16 --protocol_seed 42 --fixed_pos_noise_mm 2.5 \
    --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null && { sleep 15; kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null; break; }
    if grep -qE "Traceback|OutOfMemoryError|CUDA out of memory" "$LG" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null; log "EVAL $TAG DIED"; break
    fi
    sleep 20; waited=$((waited+20))
    [ "$waited" -gt "$EVAL_TIMEOUT" ] && { kill -9 "$pid" 2>/dev/null; log "EVAL $TAG TIMEOUT"; break; }
  done
  wait "$pid" 2>/dev/null
  log "EVAL $TAG done $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
}

log "==================== x19 start: tripod placement, measured ===================="
G=$(claim "A-reaim");  run_eval "$G" "$A_POS" "x18_jitcam_s2_nomtp_realhfov_n2.5" "A: re-aim only, nominal standoff"; release "$G"
G=$(claim "B-closer"); run_eval "$G" "$B_POS" "x19_jitcam_s2_closer_realhfov_n2.5" "B: 0.827x standoff, size-matched"; release "$G"

log "==================== X19_DONE ===================="
log "TRIPOD DECISION TABLE (jitcam_real_s2, gear, 2.5mm noise, n=256):"
log "  do nothing (real pose)     : $(grep -oE 'sr=[0-9.]+' "$LOGD/x17_jitcam_s2_realtp_n2.5.log" 2>/dev/null | tail -1)"
log "  A re-aim only              : $(grep -oE 'sr=[0-9.]+' "$LOGD/x18_jitcam_s2_nomtp_realhfov_n2.5.log" 2>/dev/null | tail -1)"
log "  B move closer 0.827x       : $(grep -oE 'sr=[0-9.]+' "$LOGD/x19_jitcam_s2_closer_realhfov_n2.5.log" 2>/dev/null | tail -1)"
log "  reference: sim pose+sim lens (x12) = 0.8711"
