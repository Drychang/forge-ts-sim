#!/usr/bin/env bash
# x18: "should I re-aim the tripod tomorrow morning?"
#
# x17 measured jitcam_real_s2 at the REAL TP pose + real 55.7 deg HFOV: sr=0.7461.
# The sim-nominal reference (x12 real_at_real n2.5) was 0.8711. So the real TP costs
# ~12.5 points. That gap has TWO possible causes and they need different fixes:
#
#   (a) POSE  - real TP is 60.2mm / 12.50 deg off nominal. FIXABLE IN 10 MINUTES
#               by re-aiming the tripod. x15 says 12.5 deg of TP rotation alone lands
#               around 70-75%, which would make aim the dominant term.
#   (b) HFOV  - real lens is 55.7 deg vs sim 47.2 deg. NOT fixable (it is the lens).
#
# This isolates them: sim-nominal TP POSE while KEEPING the real 55.7 deg HFOV.
#   -> if sr recovers to ~0.86, the gap is aim; tell the user to re-aim the tripod.
#   -> if sr stays ~0.76, the gap is the lens; re-aiming is wasted effort.
#
# Wrist stays at its real calibrated pose throughout (it cannot move without a reprint).
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
TASK=Isaac-Forge-GearMesh-TBCamera-v0
CK=$CKPT/jitcam_real_s2
MASTER=$LOGD/x18_master.log; LOCK=$LOGD/x16_gpulocks
MIN_FREE_MIB=15000; MIN_RAM_GB=12; EVAL_TIMEOUT=5400
REAL_HFOV=55.7
WRIST_POS="0.036666,-0.030747,-0.046328"
WRIST_ROT="0.702792,0.013969,0.004503,0.711244"
# sim nominal TP, straight from tb_camera_env._tp_cfg()
NOM_TP_POS="1.0,0.0,0.4"
NOM_TP_ROT="0.35355,-0.61237,-0.61237,0.35355"

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

run_eval(){
  local G="$1" TAG="$2"
  local LG="$LOGD/$TAG.log"
  [ -f "$LG" ] && grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG (done)"; return 0; }
  log "EVAL $TAG gpu=$G : TP at SIM NOMINAL pose, HFOV kept at real ${REAL_HFOV}deg"
  CUDA_VISIBLE_DEVICES="$G" \
  TB_CAM_WRIST_POS="$WRIST_POS" TB_CAM_WRIST_ROT="$WRIST_ROT" \
  TB_CAM_TP_POS="$NOM_TP_POS" TB_CAM_TP_ROT="$NOM_TP_ROT" \
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

log "==================== x18 start: tripod re-aim decision ===================="
G=$(claim "x18-nomtp-realhfov"); run_eval "$G" "x18_jitcam_s2_nomtp_realhfov_n2.5"; release "$G"
log "==================== X18_DONE ===================="
log "  real TP + real HFOV (x17): $(grep -oE 'sr=[0-9.]+' "$LOGD/x17_jitcam_s2_realtp_n2.5.log" 2>/dev/null | tail -1)"
log "  nom  TP + real HFOV (x18): $(grep -oE 'sr=[0-9.]+' "$LOGD/x18_jitcam_s2_nomtp_realhfov_n2.5.log" 2>/dev/null | tail -1)"
