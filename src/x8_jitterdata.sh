#!/usr/bin/env bash
# x8: teach viewpoint tolerance from the data side, because the training side could not.
#
# Both dropout variants failed, and x7 showed why: zeroing a modality teaches "this input is
# MISSING", while the real rig's wrist camera is "this input is WRONG". Those are different
# distributions, and the vision-only dropout model proved it -- it got better at no-cameras
# (54.30 vs 27.73) and worse at a jittered camera (1.56 vs 74.61). Blanking the wrist at
# eval was worse still (peg 4.30), and worse than blanking BOTH cameras (50.00), because one
# black camera beside one working camera is a combination training never contained.
#
# The only way to train for "wrong viewpoint" is to render wrong viewpoints. The teacher is
# state-based and never looks at the cameras -- verified: success rate 1.000 with the wrist
# jittered 10mm -- so perturbing the camera changes the recorded images and nothing else.
# Same trajectories, same actions, different viewpoint: a clean augmentation.
#
# TWO datasets, identical in every respect except the wrist camera, because the original
# dataset mixed teacher-driven and DAgger collection and the exact commands are not
# recoverable from the logs. Comparing a new jittered set against the old mixed set would
# confound viewpoint augmentation with the collection recipe.
#
#   ctrl : 12 runs, wrist at the nominal (0.13, 0, -0.15)
#   jit  : 12 runs, wrist displaced 0/5/10/15/20/25 mm (rot = mm/5 deg), two directions each
#
# The jitter is centred on the NOMINAL pose, not on a measured real pose, because the real
# extrinsics have never been measured -- the three TF trees are disconnected and hand-eye
# calibration is unassigned. Nominal is also the correct centre if the mount is reprinted to
# spec, which is the recommended path anyway.
#
# Measured before writing this: 32 episodes / 80 s, 56 MB per 32 episodes.
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x8_master.log
QDIR=$LOGD/x8_queue; CLAIMS=$QDIR/claims
NEW=/media/data/forge_ts_data_jit
TEACHER=~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_peg_s2/nn/last_Forge_ep_200_rew_380.91483.pth
ENVN=Isaac-Forge-PegInsert-TBCamera-v0
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-25}"
GPUS="${GPUS:-0 1 2}"

mkdir -p "$LOGD" "$QDIR" "$NEW"; rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
             | tr -d ' ' | awk -F, '{print $2-$1}'; }
free_ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }
wait_res(){ local G="$1" f r
    while true; do f=$(free_on "$G"); r=$(free_ram_gb)
        [ "${f:-0}" -ge "$MIN_FREE_MIB" ] && [ "${r:-0}" -ge "$MIN_FREE_RAM_GB" ] && return 0
        log "gpu $G ${f}MiB / ram ${r}GB, waiting"; sleep 300; done; }

# COLLECT <arm> <idx> <pos_mm> <rot_deg> <jitter_seed> <collect_seed>
do_collect(){
    local G="$1" ARM="$2" I="$3" PM="$4" RD="$5" JS="$6" CS="$7"
    local OUT="$NEW/$ARM/_run$I" LG="$LOGD/x8_collect_${ARM}_$I.log" pid waited=0
    # Completion is the DONE line, not the presence of shards: a run killed between two
    # flushes leaves valid-looking shards behind and would be skipped as finished.
    grep -q "COLLECT\] DONE:" "$LG" 2>/dev/null && { log "SKIP collect $ARM/$I (done)"; return 0; }
    rm -f "$OUT"/shard_*.npz 2>/dev/null
    mkdir -p "$OUT"
    wait_res "$G"
    log "COLLECT $ARM/$I gpu=$G wrist=${PM}mm/${RD}deg jseed=$JS cseed=$CS"
    CUDA_VISIBLE_DEVICES=$G \
    TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
    TB_CAM_JITTER_SEED="$JS" TB_CAM_JITTER_CAMS=wrist \
    "$PY" "$SRC/collect_camera_rollouts.py" --headless --task "$ENVN" \
        --checkpoint "$TEACHER" --num_envs 32 --target_success 200 --shard_size 100 \
        --seed "$CS" --out_dir "$OUT" > "$LG" 2>&1 &
    pid=$!
    # Isaac Sim does not exit after the collection loop returns. The first two runs finished
    # their work in ~9 minutes, printed DONE, wrote every shard, and then sat spinning for
    # 8.5 hours emitting OgnSdOnNewFrame warnings -- which stalled the whole queue behind
    # them. The data is complete at DONE, so stop waiting for a shutdown that never comes.
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q "COLLECT\] DONE:" "$LG" 2>/dev/null; then
            sleep 30                      # let the final shard flush to the network mount
            kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null
            log "COLLECT $ARM/$I reached DONE; closed the hung app"
            break
        fi
        sleep 30; waited=$((waited + 30))
        if [ "$waited" -gt 3600 ]; then
            kill -9 "$pid" 2>/dev/null
            log "COLLECT $ARM/$I TIMED OUT after ${waited}s with no DONE line"
            break
        fi
    done
    wait "$pid" 2>/dev/null
    # The applied perturbation is echoed by tb_camera_env; without that line the env vars
    # never reached the camera cfg and the run is a nominal-viewpoint duplicate.
    log "COLLECT $ARM/$I shards=$(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l) | $(grep -oE 'CAM JITTER wrist: pos_mm=[0-9.]+' "$LG" | head -1) | $(grep -oE 'DONE: [0-9]+ successful' "$LG" | head -1)"
}

do_eval(){
    local G="$1" CK="$2" ABL="$3" NM="$4" PM="$5" RD="$6" BC="$7" TAG="$8" rc
    local LG="$LOGD/${TAG}.log"
    grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; return 0; }
    [ -f "$CK/best.pt" ] || { log "SKIP $TAG -- no best.pt"; return 1; }
    wait_res "$G"
    log "EVAL $TAG gpu=$G"
    CUDA_VISIBLE_DEVICES=$G \
    TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
    TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$ENVN" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm "$NM" --dyn_rand on \
        --ablate "$ABL" --blank_cam "$BC" --tag "$TAG" > "$LG" 2>&1
    rc=$?
    log "EVAL $TAG rc=$rc $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
}

train_ok(){ [ -f "$1/best.pt" ] && \
    grep -q "RUN_SEED_EARLYSTOP_DONE reason=\(early_stop\|max_epochs\)" "$1/train.log" 2>/dev/null; }

do_train(){
    local G="$1" ROOT="$2" OUT="$3" LT="$4" attempt rc
    train_ok "$OUT" && { log "SKIP TRAIN $LT"; return 0; }
    mkdir -p "$OUT"
    # Both new models inherit the canonical norm_stats so they are z-scored identically to
    # each other and to every existing cell. A fresh estimate is a 2000-sample subsample
    # whose value depends on shard load order.
    [ -f "$OUT/norm_stats.npz" ] || cp "$CKPT_ROOT/peg/norm_stats.npz" "$OUT/norm_stats.npz"
    for attempt in 1 2 3; do
        wait_res "$G"
        log "TRAIN $LT attempt $attempt gpu=$G root=$ROOT ram=$(free_ram_gb)GB"
        "$PY" -u "$SRC/student/run_seed_earlystop.py" \
            --task peg --seed 0 --data_root "$ROOT" --out_dir "$OUT" --gpu "$G" \
            --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
            > "$LOGD/x8_train_${LT}.log" 2>&1
        rc=$?
        train_ok "$OUT" && { log "TRAIN $LT OK ($(grep -cE '^epoch ' "$OUT/train.log") epochs)"; return 0; }
        log "TRAIN $LT attempt $attempt FAILED: $(grep -oE 'reason=\S+ ?\S*' "$OUT/train.log" 2>/dev/null | tail -1)"
        mv "$OUT/best.pt" "$OUT/best.pt.failed_attempt$attempt" 2>/dev/null
        sleep 300
    done
    log "ABORT $LT"; return 1
}

run_phase(){
    local PF="$1" name="$2" nw="$3" G pids=() used=0 order
    # Take the emptiest cards, not the lowest-numbered ones. Another user's job was sitting
    # on GPU1 while GPU2 was idle, and a worker pinned to GPU1 by position would have waited
    # on it indefinitely with a free card next to it.
    order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1 | tr '\n' ' ')
    log "======== phase $name: $(wc -l < "$PF") jobs, $nw workers, gpu order: $order ========"
    for G in $order; do
        used=$((used+1)); [ "$used" -gt "$nw" ] && break
        ( n=0
          while IFS= read -r line; do
              n=$((n+1)); mkdir "$CLAIMS/$(basename "$PF").$n" 2>/dev/null || continue
              set -- $line
              case "$1" in
                COLLECT) do_collect "$G" "$2" "$3" "$4" "$5" "$6" "$7" ;;
                EVAL)    do_eval    "$G" "$2" "$3" "$4" "$5" "$6" "$7" "$8" ;;
                TRAIN)   do_train   "$G" "$2" "$3" "$4" ;;
              esac
          done < "$PF" ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    log "======== phase $name done ========"
}

log "==================== x8 start ===================="
rm -rf "$NEW/_smoke"

# ---------------------------------------------------------------- phase 1: collect
P1=$QDIR/p1; : > "$P1"
i=0
for PM in 0 5 10 15 20 25; do
  RD=$(awk "BEGIN{printf \"%.1f\", $PM/5}")
  for K in 0 1; do
    JS=$((100 + i)); CS=$((1000 + i))
    echo "COLLECT jit  $i $PM $RD $JS $CS" >> "$P1"
    echo "COLLECT ctrl $i 0 0 $JS $CS"     >> "$P1"
    i=$((i+1))
  done
done
run_phase "$P1" "1 (collect 12+12 runs)" 2      # 2 workers: camera collection is RAM-bound

# ---------------------------------------------------------------- phase 2: merge
for ARM in ctrl jit; do
    D="$NEW/$ARM/PegInsert"; mkdir -p "$D"
    k=0
    for f in $(ls "$NEW/$ARM"/_run*/shard_*.npz 2>/dev/null | sort -V); do
        mv "$f" "$D/$(printf 'shard_%04d.npz' $k)"; k=$((k+1))
    done
    log "MERGE $ARM: $k shards -> $D ($(du -sh "$D" 2>/dev/null | cut -f1))"
done
for ARM in ctrl jit; do
    n=$(ls "$NEW/$ARM/PegInsert"/shard_*.npz 2>/dev/null | wc -l)
    [ "$n" -ge 8 ] || { log "ABORT: $ARM has only $n shards"; exit 1; }
done

# ---------------------------------------------------------------- phase 3: train
P3=$QDIR/p3; : > "$P3"
echo "TRAIN $NEW/ctrl $CKPT_ROOT/peg/jitdata_ctrl ctrl" >> "$P3"
echo "TRAIN $NEW/jit  $CKPT_ROOT/peg/jitdata_jit  jit"  >> "$P3"
run_phase "$P3" "3 (train ctrl + jit)" 2

# ---------------------------------------------------------------- phase 4: eval
P4=$QDIR/p4; : > "$P4"
for ARM in ctrl jit; do
  M=$CKPT_ROOT/peg/jitdata_$ARM
  for NM in 0 1 2.5 5 7.5 10 15 20; do echo "EVAL $M none $NM 0 0 none jd_${ARM}_clean_n${NM}" >> "$P4"; done
  for J in "10 2" "20 4" "40 8"; do set -- $J; echo "EVAL $M none 2.5 $1 $2 none jd_${ARM}_wj_p$1r$2" >> "$P4"; done
  echo "EVAL $M none 2.5 0 0 wrist jd_${ARM}_blankwrist" >> "$P4"
done
run_phase "$P4" "4 (eval both)" 3

log "==================== X8_ALL_DONE ===================="
