#!/usr/bin/env bash
# Zero-shot FIXTURE-TILT sweep on Mercury (2026-09-19). One sim per GPU, sequential chain, idempotent.
# Usage: bash run_tilt_chain.sh <gpu> <mode: ta|student> "<tasks: peg gear nut>" "<tilts deg: 0 0.5 1 2 3 5 8>" [noise_mm=0] [episodes] [num_envs]
#   ta      -> official T-A (repro_<task>_s0, ep 200, md5 == QAQ ~/forge_orig/repro_<task>_s0.pth) via eval_frozen_tilt.py
#   student -> Table I student relabel_dagger_<task>_s0 (md5 == QAQ ~/student_ckpts/...) via student/eval_ablate_student_tilt.py
# Logs: ~/logs/tilt/<mode>_<task>_t<tilt>_n<noise>.log ; summaries: ~/forge_ts/eval_tilt/*.json ; master: ~/logs/tilt/master.log
# Skips a cell whose log already contains EVAL_SUMMARY. Never touches other users' processes.
set -uo pipefail
G=${1:?gpu}; MODE=${2:?ta|student}; TASKS=${3:?tasks}; TILTS=${4:?tilts}; NOISE=${5:-0}; EPS=${6:-}; NENV=${7:-}
export OMNI_KIT_ACCEPT_EULA=Y TMPDIR=$HOME/tmp_isaac PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TILT_MODE=${TILT_MODE:-fixed} TILT_AXIS=${TILT_AXIS:-random} TILT_ABOUT=${TILT_ABOUT:-root}
unset TB_CAM_WRIST_POS TB_CAM_WRIST_ROT TB_CAM_TP_POS TB_CAM_TP_ROT TB_CAM_HFOV_DEG TB_COLLECT_REALCOLOR
PY=$HOME/miniconda3/envs/isaaclab/bin/python
SRC=$HOME/forge_ts/src
F=$HOME/force_vla_research/IsaacLab/logs/rl_games/Forge
SCK=/extra_home3/user/toolcam_ckpt
OUT=$HOME/forge_ts/eval_tilt
LOGD=$HOME/logs/tilt
M=$LOGD/master.log
mkdir -p "$OUT" "$LOGD" "$TMPDIR"
log(){ echo "[$(date -Is)] gpu$G $MODE: $*" >> "$M"; }
gym_of(){ case "$1" in peg) echo PegInsert;; gear) echo GearMesh;; nut) echo NutThread;; esac; }
ta_ckpt(){ case "$1" in
  peg)  echo "$F/repro_peg_s0/nn/last_Forge_ep_200_rew__326.7232_.pth";;
  gear) echo "$F/repro_gear_s0/nn/last_Forge_ep_200_rew__714.4058_.pth";;
  nut)  echo "$F/repro_nut_s0/nn/last_Forge_ep_200_rew__1045.4905_.pth";; esac; }

if [ "$MODE" = ta ]; then EPS=${EPS:-256}; NENV=${NENV:-128}; TMO=5400; else EPS=${EPS:-128}; NENV=${NENV:-8}; TMO=14400; fi
log "==== chain start tasks=[$TASKS] tilts=[$TILTS] noise=${NOISE}mm eps=$EPS nenv=$NENV mode=$TILT_MODE axis=$TILT_AXIS about=$TILT_ABOUT ===="
for T in $TASKS; do for D in $TILTS; do
  TAG="${MODE}_${T}_t${D}_n${NOISE}"; LG="$LOGD/$TAG.log"
  if grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null; then log "SKIP $TAG (done)"; continue; fi
  log "RUN $TAG"
  t0=$(date +%s)
  if [ "$MODE" = ta ]; then
    CUDA_VISIBLE_DEVICES=$G TILT_DEG=$D timeout -k 60 $TMO "$PY" "$SRC/eval_frozen_tilt.py" --headless \
      --task "Isaac-Forge-$(gym_of $T)-Direct-v0" --checkpoint "$(ta_ckpt $T)" \
      --episodes "$EPS" --num_envs "$NENV" --protocol_seed 42 --fixed_pos_noise_mm "$NOISE" --dyn_rand on \
      --tag "$TAG" --out_dir "$OUT" > "$LG" 2>&1 < /dev/null
  else
    CUDA_VISIBLE_DEVICES=$G TILT_DEG=$D timeout -k 60 $TMO "$PY" "$SRC/student/eval_ablate_student_tilt.py" --headless --enable_cameras \
      --task "Isaac-Forge-$(gym_of $T)-TBCamera-v0" --checkpoint "$SCK/relabel_dagger_${T}_s0/best.pt" \
      --norm_stats "$SCK/relabel_dagger_${T}_s0/norm_stats.npz" \
      --episodes "$EPS" --num_envs "$NENV" --protocol_seed 42 --fixed_pos_noise_mm "$NOISE" --dyn_rand on \
      --ablate none --blank_cam none --tag "$TAG" --out_dir "$OUT" > "$LG" 2>&1 < /dev/null
  fi
  rc=$?
  log "DONE $TAG rc=$rc $(( $(date +%s) - t0 ))s $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1) $(grep -oE 'measured fixture tilt[^(]*' "$LG" | tail -1)"
  # a hung Kit teardown holds the GPU: make sure nothing of this cell is left
  pkill -9 -f -- "--tag $TAG " 2>/dev/null; sleep 5
done; done
log "==== CHAIN_DONE tasks=[$TASKS] tilts=[$TILTS] noise=${NOISE}mm ===="
