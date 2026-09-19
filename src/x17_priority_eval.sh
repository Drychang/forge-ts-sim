#!/usr/bin/env bash
# x17: PRIORITY eval — what does the ALREADY-TRAINED jitcam_real_s2 score at the
# REAL TP camera pose + real HFOV?
#
# WHY: user needs a deployable model tomorrow 10:00. x16 (forreal, trained with both
# cameras at real poses) will NOT finish by then. jitcam_real_s2 exists NOW and has the
# wrist camera at the real pose, but its TP was trained at sim nominal — 60.2mm / 12.50 deg
# off from the real TP, plus an 8.5 deg HFOV mismatch.
#
# x15 predicts ~70-85% for that offset, but that was an extrapolation from a model trained
# at sim nominal TP. This measures it directly. It is the single most decision-relevant
# number for tomorrow morning.
#
# Runs 3 conditions on jitcam_real_s2:
#   A) real TP pose + real HFOV, 2.5mm state noise   <- the deployment condition
#   B) real TP pose + real HFOV, 0mm  state noise    <- clean upper bound
#   C) sim nominal TP (baseline it was trained on)   <- reference / regression check
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
TASK=Isaac-Forge-GearMesh-TBCamera-v0
CK=$CKPT/jitcam_real_s2
MASTER=$LOGD/x17_master.log; LOCK=$LOGD/x16_gpulocks
MIN_FREE_MIB=15000; MIN_RAM_GB=12; EVAL_TIMEOUT=5400
REAL_HFOV=55.7
WRIST_POS="0.036666,-0.030747,-0.046328"
WRIST_ROT="0.702792,0.013969,0.004503,0.711244"
TP_POS="1.023540,-0.050991,0.378229"
TP_ROT="0.436309,-0.597628,-0.550977,0.385874"

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

# $1=gpu $2=noise_mm $3=tag $4=use_real_tp(1/0)
run_eval(){
  local G="$1" NS="$2" TAG="$3" REALTP="$4"
  local LG="$LOGD/$TAG.log"
  [ -f "$LG" ] && grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG (done)"; return 0; }
  log "EVAL $TAG gpu=$G noise=${NS}mm realtp=$REALTP"
  if [ "$REALTP" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$G" \
    TB_CAM_WRIST_POS="$WRIST_POS" TB_CAM_WRIST_ROT="$WRIST_ROT" \
    TB_CAM_TP_POS="$TP_POS" TB_CAM_TP_ROT="$TP_ROT" \
    TB_CAM_HFOV_DEG="$REAL_HFOV" \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
      --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
      --episodes 256 --num_envs 16 --protocol_seed 42 --fixed_pos_noise_mm "$NS" \
      --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$G" \
    TB_CAM_WRIST_POS="$WRIST_POS" TB_CAM_WRIST_ROT="$WRIST_ROT" \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
      --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
      --episodes 256 --num_envs 16 --protocol_seed 42 --fixed_pos_noise_mm "$NS" \
      --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1 &
  fi
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
  log "EVAL $TAG done $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1) | hfov=$(grep -c 'HFOV override' "$LG") tp_abs=$(grep -c 'CAM ABS tp' "$LG")"
}

log "==================== x17 start: jitcam_real_s2 @ REAL TP ===================="
log "ckpt=$CK"
[ -f "$CK/best.pt" ] || { log "FATAL: no best.pt at $CK"; exit 1; }

# A) THE deployment condition — real TP + real HFOV, 2.5mm noise
G=$(claim "eval-realtp-n2.5"); run_eval "$G" "2.5" "x17_jitcam_s2_realtp_n2.5" 1; release "$G"

# B) clean upper bound at real TP
G=$(claim "eval-realtp-n0"); run_eval "$G" "0" "x17_jitcam_s2_realtp_n0" 1; release "$G"

# C) sim nominal TP reference (what it was trained on)
G=$(claim "eval-simtp-n2.5"); run_eval "$G" "2.5" "x17_jitcam_s2_simtp_n2.5" 0; release "$G"

log "==================== X17_DONE ===================="
log "SUMMARY:"
for t in x17_jitcam_s2_realtp_n2.5 x17_jitcam_s2_realtp_n0 x17_jitcam_s2_simtp_n2.5; do
  log "  $t: $(grep -oE 'sr=[0-9.]+' "$LOGD/$t.log" 2>/dev/null | tail -1)"
done
