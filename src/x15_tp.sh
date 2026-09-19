#!/usr/bin/env bash
# x15: how much third-person camera pose error can the deployed model absorb?
#
# Everything measured so far perturbs the WRIST. The tp camera was never touched -- x11
# collected with TB_CAM_JITTER_CAMS=wrist, and the pixel audit confirmed tp moved 0.16-0.39
# (the render noise floor) across all 24 runs. So jitcam_real has seen exactly ONE tp
# viewpoint, and the real rig's tp has just been repositioned to a new spot.
#
# WHY NOT REUSE THE WRIST'S mm/5deg COUPLING. That coupling encodes one mount's geometry.
# tp sits 531.5 mm from the fixture, the wrist 66.6 mm -- a factor of 7.98. A translation of
# d mm swings the viewing angle by atan(d/L), so the same d costs tp EIGHT TIMES LESS than
# the wrist. Rotation carries no such factor: 4 deg is 4 deg at any stand-off. Translation
# and rotation therefore decouple differently here than they did for the wrist, and pairing
# them at a fixed ratio would confound the two. This sweep varies ONE at a time so any
# combination can be predicted afterwards.
#
# THREE ARMS
#   pos-only   20/40/80/160/240 mm, random direction (seed 1)   -- generic mount slip
#   rot-only   1/2/4/8/16 deg, random axis                      -- aim error
#   radial     absolute override along the fixture->camera line -- the case that is actually
#              planned: COMMISSIONING 6.1 tells the operator to place the real tp at
#              +330/+290 mm instead of the sim's +400/+350, a deliberate 92.2 mm pull-in to
#              compensate for the wider real lens (55.7 vs 47.2 deg HFOV). That displacement
#              is NOT a mistake to be tolerated, it is the recommended placement, and its
#              effect on a policy trained at the sim pose has never been measured.
#              (Caveat for the writeup: sim keeps its 47.2 deg lens here, so a pull-in
#              zooms in where the real wide lens would not. This arm therefore UPPER-BOUNDS
#              the cost of the recommended placement.)
#
# TWO SANITY GATES, both cheap, both catch whole classes of plumbing bugs:
#   base     no override, no jitter          -> must reproduce x12_real_at_real_n2.5 = 0.8711
#   nullovr  TB_CAM_TP_POS set to nominal    -> must also reproduce it; proves the override
#            path itself is inert when handed the value it is replacing
#
# Wrist is pinned to the measured real pose in every cell, because that is what gets deployed.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
TASK=Isaac-Forge-GearMesh-TBCamera-v0
REAL_POS="0.036666,-0.030747,-0.046328"; REAL_ROT="0.702792,0.013969,0.004503,0.711244"
MASTER=$LOGD/x15_master.log; LOCK=$LOGD/x15_gpulocks; QDIR=$LOGD/x15_claims
MIN_FREE_MIB=21500; MIN_RAM_GB=12
EVAL_TIMEOUT=5400          # 90 min; a gear eval took 31 min in x14

mkdir -p "$LOGD" "$LOCK" "$QDIR"; rmdir "$LOCK"/g* 2>/dev/null
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }
ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }

# Take the emptiest card that clears the gate, by lock dir so two workers cannot share one.
# Re-checks free VRAM after taking the lock: another tenant can arrive between the two reads.
# Never preempts -- GPU0 is currently another user's and this simply waits it out.
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
    sleep 240
  done
}
release(){ rmdir "$LOCK/g$1" 2>/dev/null; }

# Isaac Sim does not always exit: after the OOM in x12 it sat on the exception for 12 hours
# and stalled the queue behind it. Poll for the terminal line instead of trusting the process
# to end, and treat a traceback as terminal too.
run_eval(){
  local G="$1" CK="$2" PM="$3" RD="$4" ABS="$5" TAG="$6"
  local LG="$LOGD/$TAG.log" pid waited=0 extra=()
  [ "$ABS" = "-" ] || extra+=("TB_CAM_TP_POS=$ABS")
  log "EVAL $TAG gpu=$G tp_pos=${PM}mm tp_rot=${RD}deg tp_abs=$ABS"
  env CUDA_VISIBLE_DEVICES="$G" \
      TB_CAM_WRIST_POS="$REAL_POS" TB_CAM_WRIST_ROT="$REAL_ROT" \
      TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
      TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=tp \
      "${extra[@]}" \
      "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm 2.5 \
        --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null; then
      sleep 15; kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null
      log "EVAL $TAG reached EVAL_SUMMARY; closed the app"; break
    fi
    if grep -qE "Traceback|OutOfMemoryError|CUDA out of memory" "$LG" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null
      log "EVAL $TAG DIED: $(grep -oE 'OutOfMemoryError|CUDA out of memory|Traceback' "$LG" | head -1)"; break
    fi
    sleep 20; waited=$((waited+20))
    if [ "$waited" -gt "$EVAL_TIMEOUT" ]; then
      kill -9 "$pid" 2>/dev/null; log "EVAL $TAG TIMEOUT after ${waited}s"; break
    fi
  done
  wait "$pid" 2>/dev/null
  # Confirm the perturbation actually reached the camera cfg. Without this line the env vars
  # were silently ignored and the cell is a duplicate of the baseline wearing another name.
  log "EVAL $TAG done $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1) | wrist=$(grep -cE 'CAM ABS wrist' "$LG") tp_jit=$(grep -cE 'CAM JITTER tp' "$LG") tp_abs=$(grep -cE 'CAM ABS tp' "$LG") wrist_skip=$(grep -cE 'CAM JITTER wrist: SKIPPED' "$LG")"
}

# seed  pos_mm  rot_deg  abs_pos                      name
QUEUE=$LOGD/x15_queue; cat > "$QUEUE" <<'JOBS'
0 0   0  -                          base
0 0   0  1.0,0.0,0.4                nullovr
0 20  0  -                          p20
0 40  0  -                          p40
0 80  0  -                          p80
0 160 0  -                          p160
0 240 0  -                          p240
0 0   1  -                          r1
0 0   2  -                          r2
0 0   4  -                          r4
0 0   8  -                          r8
0 0   16 -                          r16
0 0   0  0.965306,0.0,0.369643      rad_in46
0 0   0  0.930612,0.0,0.339286      rad_in92
0 0   0  0.895919,0.0,0.308929      rad_in138
0 0   0  1.069388,0.0,0.460714      rad_out92
1 0   0  -                          base
1 80  0  -                          p80
1 0   4  -                          r4
1 0   0  0.930612,0.0,0.339286      rad_in92
2 0   0  -                          base
2 80  0  -                          p80
2 0   4  -                          r4
2 0   0  0.930612,0.0,0.339286      rad_in92
JOBS

worker(){
  local W="$1" n=0 line SEED PM RD ABS NAME CK TAG G
  while IFS= read -r line; do
    n=$((n+1)); [ -n "$line" ] || continue
    mkdir "$QDIR/job$n" 2>/dev/null || continue      # atomic claim; also makes reruns resume
    set -- $line; SEED=$1; PM=$2; RD=$3; ABS=$4; NAME=$5
    [ "$SEED" = "0" ] && CK="$CKPT/jitcam_real" || CK="$CKPT/jitcam_real_s$SEED"
    TAG="x15_gear_s${SEED}_$NAME"
    if grep -q "EVAL_SUMMARY" "$LOGD/$TAG.log" 2>/dev/null; then log "SKIP $TAG (done)"; continue; fi
    [ -f "$CK/best.pt" ] || { log "SKIP $TAG -- no best.pt in $CK"; continue; }
    G=$(claim "w$W")
    run_eval "$G" "$CK" "$PM" "$RD" "$ABS" "$TAG"
    release "$G"
  done < "$QUEUE"
  log "worker $W drained"
}

log "==================== x15 start: gear tp viewpoint sensitivity ===================="
log "tenancy at launch: ram=$(ram_gb)GB free=[$(free_on 0),$(free_on 1),$(free_on 2)]MiB"
log "$(wc -l < "$QUEUE") jobs queued"
for W in 1 2 3; do worker "$W" & done      # 3 shells, they self-limit to the free cards
wait
log "==================== X15_ALL_DONE ===================="
{ echo "== x15 summary =="; grep -hE "EVAL x15_\S+ done" "$MASTER"; } >> "$MASTER"
