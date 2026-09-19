#!/usr/bin/env bash
# Mechanism probe: WHY is the vision+force student worse than the state-based baseline under fixture tilt?
#
# Observed at sigma=0 (frozen protocol, existing checkpoints, no retraining):
#   peg  5 deg:  T-A 69.9%  vs  student 44.5%      <- student LOSES to the vision-blind baseline
#   peg  8 deg:  T-A 32.4%  vs  student 17.2%
# Same inversion as the paper's own AutoMate cell (student 4.6% vs noise-aug state baseline 18.4%),
# but here the GEOMETRY IS UNCHANGED - only the pose of a familiar part differs. That isolates
# "unseen pose" from "unseen shape" as the cause.
#
# Hypothesis: the tilted fixture is an out-of-distribution VISUAL input, and the vision-dominant
# student is hurt more by a corrupted view than the baseline is by having no view at all
# (the "bad input is worse than no input" effect already documented for colour shift in G1b).
#
# Test: blank one camera at a time, at 0 and 5 deg, on the SAME checkpoint. If blanking the
# third-person camera IMPROVES success at 5 deg relative to the full student, that camera is
# actively misleading the policy under tilt.
#
# Waits until the sigma=0 and sigma=5mm chains are finished (6 CHAIN_DONE) so only one sim per GPU.
# Usage: setsid nohup bash run_tilt_mech.sh > ~/logs/tilt/mech_driver.log 2>&1 < /dev/null &
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y TMPDIR=$HOME/tmp_isaac PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TILT_MODE=fixed TILT_AXIS=random TILT_ABOUT=root
unset TB_CAM_WRIST_POS TB_CAM_WRIST_ROT TB_CAM_TP_POS TB_CAM_TP_ROT TB_CAM_HFOV_DEG TB_COLLECT_REALCOLOR
PY=$HOME/miniconda3/envs/isaaclab/bin/python
SRC=$HOME/forge_ts/src
SCK=/extra_home3/user/toolcam_ckpt
OUT=$HOME/forge_ts/eval_tilt
LOGD=$HOME/logs/tilt
M=$LOGD/master.log
mkdir -p "$OUT" "$LOGD" "$TMPDIR"
log(){ echo "[$(date -Is)] mech: $*" >> "$M"; }

while :; do n=$(grep -c 'CHAIN_DONE' "$M" 2>/dev/null); [ "$n" -ge 6 ] && break; sleep 180; done
log "==== mechanism probe start (sigma=0 and sigma=5mm chains finished) ===="

# task tilt blank_cam  -> one cell each; peg first (largest observed inversion), then gear
QUEUE="peg:0:tp peg:5:tp peg:0:wrist peg:5:wrist gear:0:tp gear:5:tp gear:0:wrist gear:5:wrist"
gym_of(){ case "$1" in peg) echo PegInsert;; gear) echo GearMesh;; nut) echo NutThread;; esac; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ' | awk -F, 'NF==2{print $2-$1; f=1} END{if(!f) print 0}'; }

G=0
for item in $QUEUE; do
  T="${item%%:*}"; rest="${item#*:}"; D="${rest%%:*}"; CAM="${rest##*:}"
  TAG="mech_${T}_t${D}_blank${CAM}"; LG="$LOGD/$TAG.log"
  grep -q 'EVAL_SUMMARY' "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
  # pick the emptiest GPU, require 8 GB so other users keep headroom
  G=$(for g in 0 1 2; do echo "$g,$(free_on $g)"; done | sort -t, -k2 -nr | head -1 | cut -d, -f1)
  [ "$(free_on "$G")" -ge 8000 ] || { log "waiting for VRAM: [$(free_on 0),$(free_on 1),$(free_on 2)]"; sleep 300; continue; }
  log "RUN $TAG gpu=$G"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$G TILT_DEG=$D timeout -k 60 14400 "$PY" "$SRC/student/eval_ablate_student_tilt.py" \
    --headless --enable_cameras --task "Isaac-Forge-$(gym_of $T)-TBCamera-v0" \
    --checkpoint "$SCK/relabel_dagger_${T}_s0/best.pt" --norm_stats "$SCK/relabel_dagger_${T}_s0/norm_stats.npz" \
    --episodes 128 --num_envs 16 --protocol_seed 42 --fixed_pos_noise_mm 0 --dyn_rand on \
    --ablate none --blank_cam "$CAM" --tag "$TAG" --out_dir "$OUT" > "$LG" 2>&1 < /dev/null
  log "DONE $TAG rc=$? $(( $(date +%s) - t0 ))s $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
  pkill -9 -f -- "--tag $TAG " 2>/dev/null; sleep 5
done
log "==== MECH_PROBE_DONE ===="
