#!/usr/bin/env bash
# Decide which reading of the robot side's R,t is correct by rendering the simulator's wrist
# view at each candidate pose and measuring the foreground-area fraction.
#
# GEAR, not peg: the real photo being matched (real_wrist_policy256.png) is a gear scene, and
# the simulator's wrist reference differs by 38% between tasks (gear 0.0791, peg 0.0570), so
# rendering the wrong task would compare two different subjects.
#
# The nominal pose is the control. It must reproduce ~0.0791 -- the value gear_ref_wrist.png
# scores under this same formula, verified by reproducing the recorded 0.24080 and 0.07912
# exactly -- before any candidate is believed. Target for the real camera is 0.3952, measured
# on the 256x256 crop the policy actually receives. (The 2026-08-12 comparison put a 480x640
# real frame beside a 256x256 simulator frame and reported 0.2408 vs 0.0791; like for like at
# 256x256 the gap is 5.0x, not 3x.)
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOGD=~/forge_ts/logs
OUT=/media/data/forge_ts_data_jit/_verify
TEACHER=~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_gear_s0/nn/last_Forge_ep_200_rew_746.5314.pth
TASK=Isaac-Forge-GearMesh-TBCamera-v0
M=$LOGD/verify_master.log
mkdir -p "$OUT"
: > "$M"
log(){ echo "[$(date -Is)] $*" >> "$M"; }

render(){
  # Split across two `local` statements on purpose: bash expands every argument of a single
  # `local` before assigning any of them, so "$OUT/$N" on the same line reads an N that does
  # not exist yet and dies under set -u.
  local N="$1" P="$2" R="$3"
  local D="$OUT/$N" LG="$LOGD/verify_$N.log"
  if [ -n "$(ls "$D"/shard_*.npz 2>/dev/null)" ]; then log "SKIP $N"; return 0; fi
  mkdir -p "$D"
  log "RENDER $N pos=$P rot=$R"
  CUDA_VISIBLE_DEVICES=2 TB_CAM_WRIST_POS="$P" TB_CAM_WRIST_ROT="$R" \
  "$PY" "$SRC/collect_camera_rollouts.py" --headless \
    --task "$TASK" --checkpoint "$TEACHER" \
    --num_envs 32 --target_success 8 --shard_size 100 --seed 4242 \
    --out_dir "$D" > "$LG" 2>&1 &
  local pid=$!
  local w=0
  # Isaac does not exit after the collection loop returns; the data is complete at DONE.
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "COLLECT\] DONE:" "$LG" 2>/dev/null; then
      sleep 25; kill "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; break
    fi
    sleep 20; w=$((w+20))
    if [ "$w" -gt 1800 ]; then kill -9 "$pid" 2>/dev/null; log "TIMEOUT $N"; break; fi
  done
  wait "$pid" 2>/dev/null
  log "RENDER $N done | $(grep -oE 'CAM ABS wrist: pos -> .*' "$LG" | head -1)"
}

render nominal "0.130000,0.000000,-0.150000"  "-0.706140,0.037010,0.037010,-0.706140"
render caseA   "0.036666,-0.030747,-0.046328" "0.702792,0.013969,0.004503,0.711244"
render caseB   "0.031801,0.037484,0.044940"   "0.702792,-0.013969,-0.004503,-0.711244"

log "==== foreground fraction ===="
"$PY" - >> "$M" 2>&1 <<'PYEOF'
import numpy as np, glob, cv2
def fg(img):
    a = np.asarray(img, np.float32)
    L = 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
    bg = float(np.median(L[L <= np.median(L)]))
    return float((L > (bg+60)).mean())
print("%-9s %9s %8s %6s" % ("pose", "fg_mean", "fg_sd", "n"))
for n in ("nominal", "caseA", "caseB"):
    fs = sorted(glob.glob("/media/data/forge_ts_data_jit/_verify/%s/shard_*.npz" % n))
    if not fs:
        print("%-9s (no data)" % n); continue
    d = np.load(fs[0], allow_pickle=True); w = d["wrist_jpeg"]
    vals = []
    for e in range(min(16, len(w))):
        for t in (30, 70, 110):
            b = w[e][t] if w.dtype == object else w[e, t]
            im = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            vals.append(fg(im))
    v = np.array(vals)
    print("%-9s %9.4f %8.4f %6d" % (n, v.mean(), v.std(), v.size))
print()
print("targets:  sim gear_ref_wrist.png = 0.0791   real_wrist_policy256.png = 0.3952")
PYEOF
log "==================== VERIFY_DONE ===================="
