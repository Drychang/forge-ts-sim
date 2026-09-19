#!/usr/bin/env bash
# x16: train a "for real" gear model with both cameras at measured real poses + matched HFOV.
#
# WHY: jitcam_real was trained with TP at sim nominal. The real TP is 60.2mm / 12.5 deg off,
# which by x15's curve puts it in the danger zone (~70% expected). Meanwhile we now have
# calibrated extrinsics for BOTH cameras. Training with matched poses + matched HFOV (55.7 deg
# real vs 47.2 deg sim) should give a model that works out-of-the-box on the real rig.
#
# DESIGN:
#   Collection: 12 runs, both cameras at real extrinsics, HFOV=55.7 deg, jitter on BOTH cameras
#     run 0,1:  0mm/0deg (clean, two different collection seeds)
#     run 2-5:  5mm/1deg (very small mount noise)
#     run 6-9:  10mm/2deg (target margin)
#     run 10-11: 15mm/3deg (edge safety)
#   Each run: 200 successful episodes, gear task
#
#   Training: 3 seeds, same data, canonical norm_stats
#   Eval: noise sweep (0/1/2.5/5mm) + tp sensitivity check at real pose
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
NEW=/media/data/forge_ts_gear_forreal
TEACHER=~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_gear_s0/nn/last_Forge_ep_200_rew_746.5314.pth
TASK=Isaac-Forge-GearMesh-TBCamera-v0
MASTER=$LOGD/x16_master.log; LOCK=$LOGD/x16_gpulocks; QDIR=$LOGD/x16_claims
MIN_FREE_MIB=15000; MIN_RAM_GB=12; MIN_RAM_TRAIN=25
EVAL_TIMEOUT=5400; COLLECT_TIMEOUT=5400
REAL_HFOV=55.7

# Real camera poses (from calibration npz files, verified against x11 REAL_POS/ROT for wrist)
WRIST_POS="0.036666,-0.030747,-0.046328"
WRIST_ROT="0.702792,0.013969,0.004503,0.711244"
TP_POS="1.023540,-0.050991,0.378229"
TP_ROT="0.436309,-0.597628,-0.550977,0.385874"

mkdir -p "$LOGD" "$NEW" "$LOCK" "$QDIR"; rmdir "$LOCK"/g* 2>/dev/null
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d " " | awk -F, 'NF==2{print $2-$1; f=1} END{if(!f) print 0}'; }
ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }

claim(){
  local MINRAM="$1" who="$2" G order
  while :; do
    if [ "$(ram_gb)" -ge "$MINRAM" ]; then
      order=$(for g in 0 1 2; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d " "); case "$u" in ""|*[!0-9]*) continue;; esac; echo "$g,$u"; done | sort -t, -k2 -n | cut -d, -f1)
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

# ---- COLLECTION ----
do_collect(){
  local G="$1" I="$2" PM="$3" RD="$4" JS="$5" CS="$6"
  local D="$NEW/_run$I" LG="$LOGD/x16_collect_$I.log"
  grep -q "COLLECT\] DONE:" "$LG" 2>/dev/null && { log "SKIP collect/$I (done)"; return 0; }
  rm -f "$D"/shard_*.npz 2>/dev/null; mkdir -p "$D"
  log "COLLECT run=$I gpu=$G jitter=${PM}mm/${RD}deg jseed=$JS cseed=$CS"
  CUDA_VISIBLE_DEVICES=$G \
  TB_CAM_WRIST_POS="$WRIST_POS" TB_CAM_WRIST_ROT="$WRIST_ROT" \
  TB_CAM_TP_POS="$TP_POS" TB_CAM_TP_ROT="$TP_ROT" \
  TB_CAM_HFOV_DEG="$REAL_HFOV" \
  TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
  TB_CAM_JITTER_SEED="$JS" \
  "$PY" "$SRC/collect_camera_rollouts.py" --headless --task "$TASK" \
    --checkpoint "$TEACHER" --num_envs 16 --target_success 200 --shard_size 100 \
    --seed "$CS" --out_dir "$D" > "$LG" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "COLLECT\] DONE:" "$LG" 2>/dev/null; then
      sleep 30; kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null; break
    fi
    if grep -qE "Traceback|OutOfMemoryError|CUDA out of memory" "$LG" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null
      log "COLLECT run=$I DIED: $(grep -oE 'OutOfMemoryError|CUDA out of memory|Traceback' "$LG" | head -1)"; break
    fi
    sleep 30; waited=$((waited+30))
    if [ "$waited" -gt "$COLLECT_TIMEOUT" ]; then
      kill -9 "$pid" 2>/dev/null; log "COLLECT run=$I TIMEOUT after ${waited}s"; break
    fi
  done
  wait "$pid" 2>/dev/null
  local shards=$(ls "$D"/shard_*.npz 2>/dev/null | wc -l)
  log "COLLECT run=$I done shards=$shards | $(grep -oE 'DONE: [0-9]+ successful' "$LG" | head -1) | hfov=$(grep -c 'HFOV override' "$LG") tp_abs=$(grep -c 'CAM ABS tp' "$LG") wrist_abs=$(grep -c 'CAM ABS wrist' "$LG")"
}

# Collection plan: run_idx jitter_pos_mm jitter_rot_deg jitter_seed collect_seed
COLLECT_PLAN="$LOGD/x16_collect_plan"
cat > "$COLLECT_PLAN" <<'JOBS'
0   0   0   0   42
1   0   0   0   123
2   5   1   1   200
3   5   1   2   201
4   5   1   3   202
5   5   1   4   203
6   10  2   1   300
7   10  2   2   301
8   10  2   3   302
9   10  2   4   303
10  15  3   1   400
11  15  3   2   401
JOBS

# ---- TRAINING ----
train_ok(){ [ -f "$1/best.pt" ] && grep -qE "RUN_SEED_EARLYSTOP_DONE reason=(early_stop|natural_finish)" "$1/train.log" 2>/dev/null; }

train_one(){
  local SEED="$1"
  local OUT="$CKPT/forreal_s$SEED" G att
  train_ok "$OUT" && { log "SKIP TRAIN s$SEED (already complete)"; return 0; }
  mkdir -p "$OUT"
  [ -f "$OUT/norm_stats.npz" ] || cp "$CKPT/norm_stats.npz" "$OUT/norm_stats.npz"
  for att in 1 2 3; do
    G=$(claim "$MIN_RAM_TRAIN" "train-s$SEED")
    log "TRAIN s$SEED attempt $att gpu=$G ram=$(ram_gb)GB"
    "$PY" -u "$SRC/student/run_seed_earlystop.py" --task gear --seed "$SEED" \
      --data_root "$NEW" --out_dir "$OUT" --gpu "$G" --patience 6 --max_epochs 30 \
      --batch_size 256 --num_workers 4 > "$LOGD/x16_train_s$SEED.log" 2>&1
    release "$G"
    train_ok "$OUT" && { log "TRAIN s$SEED OK epochs=$(grep -cE '^epoch ' "$OUT/train.log") $(grep -oE 'best_epoch=[0-9]+ best_val=[0-9.]+' "$OUT/train.log" | tail -1)"; return 0; }
    log "TRAIN s$SEED attempt $att FAILED: $(grep -oE 'reason=\S+( rc=\S+)?' "$OUT/train.log" 2>/dev/null | tail -1)"
    mv "$OUT/best.pt" "$OUT/best.pt.failed$att" 2>/dev/null; sleep 300
  done
  log "ABORT TRAIN s$SEED"; return 1
}

# ---- EVAL ----
run_eval(){
  local G="$1" CK="$2" NS="$3" TAG="$4" EXTRA_ENV="$5"
  local LG="$LOGD/$TAG.log" pid waited=0
  [ -f "$LG" ] && grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP EVAL $TAG (done)"; return 0; }
  log "EVAL $TAG gpu=$G noise=${NS}mm"
  env CUDA_VISIBLE_DEVICES="$G" \
      TB_CAM_WRIST_POS="$WRIST_POS" TB_CAM_WRIST_ROT="$WRIST_ROT" \
      TB_CAM_TP_POS="$TP_POS" TB_CAM_TP_ROT="$TP_ROT" \
      TB_CAM_HFOV_DEG="$REAL_HFOV" \
      $EXTRA_ENV \
      "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 16 --protocol_seed 42 --fixed_pos_noise_mm "$NS" \
        --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null; then
      sleep 15; kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null
      log "EVAL $TAG reached EVAL_SUMMARY; closed"; break
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
  log "EVAL $TAG done $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1) | hfov=$(grep -c 'HFOV override' "$LG") tp_abs=$(grep -c 'CAM ABS tp' "$LG")"
}

eval_seed(){
  local SEED="$1"
  local CK="$CKPT/forreal_s$SEED" G
  [ -f "$CK/best.pt" ] || { log "EVAL s$SEED skipped: no best.pt"; return 1; }

  # Noise sweep at real camera poses + real HFOV
  for NS in 0 1 2.5 5; do
    G=$(claim "$MIN_RAM_GB" "eval-s${SEED}-n${NS}")
    run_eval "$G" "$CK" "$NS" "x16_forreal_s${SEED}_n${NS}" ""
    release "$G"
  done

  # Also eval at sim nominal TP (no TP override, no HFOV override) to see backward compat
  G=$(claim "$MIN_RAM_GB" "eval-s${SEED}-simtp")
  local LG="$LOGD/x16_forreal_s${SEED}_simtp_n2.5.log"
  [ -f "$LG" ] && grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP EVAL simtp s$SEED"; release "$G"; } || {
    log "EVAL x16_forreal_s${SEED}_simtp_n2.5 gpu=$G (TP at sim nominal, sim HFOV)"
    CUDA_VISIBLE_DEVICES="$G" \
    TB_CAM_WRIST_POS="$WRIST_POS" TB_CAM_WRIST_ROT="$WRIST_ROT" \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
      --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
      --episodes 256 --num_envs 16 --protocol_seed 42 --fixed_pos_noise_mm 2.5 \
      --dyn_rand on --ablate none --blank_cam none --tag "x16_forreal_s${SEED}_simtp_n2.5" > "$LG" 2>&1 &
    local pid=$! waited=0
    while kill -0 "$pid" 2>/dev/null; do
      grep -q "EVAL_SUMMARY" "$LG" 2>/dev/null && { sleep 15; kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null; break; }
      grep -qE "Traceback|OutOfMemoryError" "$LG" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; break; }
      sleep 20; waited=$((waited+20)); [ "$waited" -gt "$EVAL_TIMEOUT" ] && { kill -9 "$pid" 2>/dev/null; break; }
    done
    wait "$pid" 2>/dev/null
    log "EVAL x16_forreal_s${SEED}_simtp_n2.5 done $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
    release "$G"
  }

  # Also eval jitcam_real at real TP pose + real HFOV for head-to-head comparison
  local JITCK="$CKPT/jitcam_real_s2"
  if [ "$SEED" = "0" ] && [ -f "$JITCK/best.pt" ]; then
    G=$(claim "$MIN_RAM_GB" "eval-jitcam-baseline")
    run_eval "$G" "$JITCK" "2.5" "x16_jitcam_s2_realtp_n2.5" ""
    release "$G"
  fi
}

# ====================  MAIN  ====================
log "==================== x16 start: forreal gear model ===================="
log "tenancy: ram=$(ram_gb)GB free=[$(free_on 0),$(free_on 1),$(free_on 2)]MiB"
log "cameras: wrist=$WRIST_POS tp=$TP_POS hfov=$REAL_HFOV"

# Phase 1: Collect data (2 workers, serialized per worker)
log "===== PHASE 1: DATA COLLECTION ====="
collect_worker(){
  local W="$1" n=0 line I PM RD JS CS G
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    n=$((n+1))
    mkdir "$QDIR/collect_$n" 2>/dev/null || continue
    set -- $line; I=$1; PM=$2; RD=$3; JS=$4; CS=$5
    # skip already-completed runs WITHOUT burning a GPU claim cycle
    if grep -q "COLLECT\] DONE:" "$LOGD/x16_collect_$I.log" 2>/dev/null; then
      log "SKIP collect/$I (done, pre-claim)"; continue
    fi
    G=$(claim "$MIN_RAM_GB" "collect-w${W}-run${I}")
    do_collect "$G" "$I" "$PM" "$RD" "$JS" "$CS"
    release "$G"
  done < "$COLLECT_PLAN"
  log "collect worker $W drained"
}

for W in 1 2 3; do collect_worker "$W" & done
wait
log "===== PHASE 1 DONE: $(ls "$NEW"/_run*/shard_*.npz 2>/dev/null | wc -l) total shards ====="

# Verify we have enough data
TOTAL_SHARDS=$(ls "$NEW"/_run*/shard_*.npz 2>/dev/null | wc -l)
if [ "$TOTAL_SHARDS" -lt 20 ]; then
  log "ABORT: only $TOTAL_SHARDS shards collected (expected ~24). Check logs."
  exit 1
fi

# Phase 2: Train (serialized — one at a time to avoid OOM)
log "===== PHASE 2: TRAINING ====="
# DEADLINE MODE: only seed 2 (historically dominant across x14/x15)
train_one 2

# Phase 3: Eval (parallel where possible)
log "===== PHASE 3: EVALUATION ====="
eval_seed 2

log "==================== X16_ALL_DONE ===================="
{
  echo "== x16 forreal summary =="
  grep -hE "EVAL x16_\S+ done|TRAIN s[0-9] OK|PHASE|ABORT" "$MASTER"
} >> "$MASTER"
