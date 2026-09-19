#!/usr/bin/env bash
# x11: bring the viewpoint augmentation to gear, centred where the real camera actually is.
#
# Everything measured so far is peg, but every real-robot session has been GEAR -- the teach
# points, the compare_view frames, the gripper analysis. A peg model is not what gets
# deployed.
#
# Two arms, same recipe as x8 (12 runs x 224 episodes, wrist displaced 0/5/10/15/20/25 mm,
# rot = mm/5 deg, two directions per level), differing only in where the distribution is
# centred:
#
#   nom  centred on the simulator's nominal (0.13, 0, -0.15) -- the model to use IF the
#        mount is reprinted to spec. Tolerance +-40mm around nominal (x9b), a spec a printed
#        part can hit; without augmentation it was ~10mm, which it could not.
#   real centred on the pose the robot side measured -- the model to use if the mount is
#        NOT touched. This is the faster path to a physical experiment and the reason the
#        extrinsics mattered.
#
# The absolute-pose override runs before the jitter in _maybe_jitter_offset, so the two
# compose: POS/ROT set the centre, JITTER perturbs around it.
#
# Gear costs about twice peg per episode (299 steps vs 149) and its epochs ran ~1610s
# against peg's ~800s, so budget ~2h collection and ~5h training per arm.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
NEW=/media/data/forge_ts_gear_jit
TEACHER=~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_gear_s0/nn/last_Forge_ep_200_rew_746.5314.pth
TASK=Isaac-Forge-GearMesh-TBCamera-v0
MASTER=$LOGD/x11_master.log; CLAIMS=$LOGD/x11_claims
MIN_FREE_MIB=21500; MIN_RAM_GB=25

# Centres. REAL_* is filled in from the measured extrinsics once the render check confirms
# the frame convention; leaving it equal to nominal would silently collect two identical
# datasets, so the script refuses to run until it is set.
NOM_POS="0.130000,0.000000,-0.150000"
NOM_ROT="-0.706140,0.037010,0.037010,-0.706140"
REAL_POS="0.036666,-0.030747,-0.046328"
REAL_ROT="0.702792,0.013969,0.004503,0.711244"
case "$REAL_POS" in __FILL*) echo "REAL_POS not set -- refusing to run"; exit 1;; esac

mkdir -p "$LOGD" "$NEW"; rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }
ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }
wait_res(){ while [ "$(free_on $1)" -lt "$MIN_FREE_MIB" ] || [ "$(ram_gb)" -lt "$MIN_RAM_GB" ]; do
    log "gpu $1 $(free_on $1)MiB / ram $(ram_gb)GB, waiting"; sleep 240; done; }

do_collect(){
  local G="$1" ARM="$2" I="$3" PM="$4" RD="$5" JS="$6" CS="$7" CP="$8" CR="$9"
  local D="$NEW/$ARM/_run$I" LG="$LOGD/x11_collect_${ARM}_$I.log"
  # Completion is the DONE line: a run killed between flushes leaves valid-looking shards.
  grep -q "COLLECT\] DONE:" "$LG" 2>/dev/null && { log "SKIP $ARM/$I"; return 0; }
  rm -f "$D"/shard_*.npz 2>/dev/null; mkdir -p "$D"
  wait_res "$G"
  log "COLLECT $ARM/$I gpu=$G centre=$CP jitter=${PM}mm/${RD}deg"
  CUDA_VISIBLE_DEVICES=$G TB_CAM_WRIST_POS="$CP" TB_CAM_WRIST_ROT="$CR" \
  TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
  TB_CAM_JITTER_SEED="$JS" TB_CAM_JITTER_CAMS=wrist \
  "$PY" "$SRC/collect_camera_rollouts.py" --headless --task "$TASK" \
    --checkpoint "$TEACHER" --num_envs 32 --target_success 200 --shard_size 100 \
    --seed "$CS" --out_dir "$D" > "$LG" 2>&1 &
  local pid=$! w=0
  # Isaac never exits after the collection loop; the data is complete at DONE.
  while kill -0 "$pid" 2>/dev/null; do
    grep -q "COLLECT\] DONE:" "$LG" 2>/dev/null && { sleep 30; kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null; break; }
    sleep 30; w=$((w+30)); [ "$w" -gt 5400 ] && { kill -9 "$pid" 2>/dev/null; log "TIMEOUT $ARM/$I"; break; }
  done
  wait "$pid" 2>/dev/null
  log "COLLECT $ARM/$I shards=$(ls "$D"/shard_*.npz 2>/dev/null | wc -l) | $(grep -oE 'CAM ABS wrist: pos -> .*' "$LG" | head -1) | $(grep -oE 'DONE: [0-9]+ successful' "$LG" | head -1)"
}

run_phase(){
  local PF="$1" name="$2" nw="$3" G pids=() used=0 order
  order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1 | tr '\n' ' ')
  log "======== $name: $(wc -l < "$PF") jobs, $nw workers, gpus: $order ========"
  for G in $order; do
    used=$((used+1)); [ "$used" -gt "$nw" ] && break
    ( n=0
      while IFS= read -r line; do
        n=$((n+1)); mkdir "$CLAIMS/$(basename "$PF").$n" 2>/dev/null || continue
        set -- $line
        do_collect "$G" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
      done < "$PF" ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "======== $name done ========"
}

log "==================== x11 start ===================="
P=$LOGD/x11_collect_jobs; : > "$P"
i=0
for PM in 0 5 10 15 20 25; do
  RD=$(awk "BEGIN{printf \"%.1f\", $PM/5}")
  for K in 0 1; do
    echo "real $i $PM $RD $((300+i)) $((3000+i)) $REAL_POS $REAL_ROT" >> "$P"
    echo "nom  $i $PM $RD $((300+i)) $((3000+i)) $NOM_POS $NOM_ROT"   >> "$P"
    i=$((i+1))
  done
done
run_phase "$P" "collect (12 real + 12 nom)" 2

for ARM in real nom; do
  D="$NEW/$ARM/GearMesh"; mkdir -p "$D"; k=0
  for f in $(ls "$NEW/$ARM"/_run*/shard_*.npz 2>/dev/null | sort -V); do
    mv "$f" "$D/$(printf 'shard_%04d.npz' $k)"; k=$((k+1))
  done
  log "MERGE $ARM: $k shards ($(du -sh "$D" 2>/dev/null | cut -f1))"
  [ "$k" -ge 8 ] || { log "ABORT: $ARM only $k shards"; exit 1; }
done

train_ok(){ [ -f "$1/best.pt" ] && grep -q "RUN_SEED_EARLYSTOP_DONE reason=\(early_stop\|max_epochs\)" "$1/train.log" 2>/dev/null; }
order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1 | head -2)
i=0; pids=()
for ARM in real nom; do
  G=$(echo $order | cut -d' ' -f$((i+1))); i=$((i+1))
  ( OUT=$CKPT/jitcam_$ARM
    train_ok "$OUT" && { log "SKIP TRAIN $ARM"; exit 0; }
    mkdir -p "$OUT"
    # Canonical norm_stats: a fresh estimate is a 2000-sample subsample whose value depends
    # on shard load order, which would z-score these differently from every other cell.
    [ -f "$OUT/norm_stats.npz" ] || cp "$CKPT/norm_stats.npz" "$OUT/norm_stats.npz"
    for att in 1 2 3; do
      wait_res "$G"
      log "TRAIN $ARM attempt $att gpu=$G ram=$(ram_gb)GB"
      "$PY" -u "$SRC/student/run_seed_earlystop.py" --task gear --seed 0 \
        --data_root "$NEW/$ARM" --out_dir "$OUT" --gpu "$G" --patience 6 --max_epochs 30 \
        --batch_size 256 --num_workers 4 > "$LOGD/x11_train_$ARM.log" 2>&1
      # An OOM-killed run still writes a first-epoch best.pt, so existence is not completion.
      train_ok "$OUT" && { log "TRAIN $ARM OK ($(grep -cE '^epoch ' "$OUT/train.log") epochs)"; exit 0; }
      log "TRAIN $ARM attempt $att FAILED: $(grep -oE 'reason=\S+ ?\S*' "$OUT/train.log" 2>/dev/null | tail -1)"
      mv "$OUT/best.pt" "$OUT/best.pt.failed$att" 2>/dev/null; sleep 300
    done
    log "ABORT TRAIN $ARM" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X11_TRAIN_DONE ===================="
