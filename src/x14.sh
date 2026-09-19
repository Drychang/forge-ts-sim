#!/usr/bin/env bash
# x14: seed spread for the model that actually gets deployed (gear, jitcam_real).
#
# The 87.11% handed to the robot side is one seed. Every other headline in this project is a
# 3-seed number, and the whole point of the figure is to say "this is what to expect tomorrow",
# so a single draw with no spread is the weakest part of the claim. Seeds 1 and 2, same data
# (forge_ts_gear_jit/real), same recipe, same canonical norm_stats.
#
# Serialised on purpose: both seeds read the same 12G dataset, and dataset.py eager-loads it to
# 2-3x on disk size (~30G each). Two trainings at once is how nut/ablate_state_s{1,2} got
# OOM-killed on 08-15. One training at a time, evals allowed to overlap on a different GPU.
#
# GPU tenancy: all three cards are currently held by another user's SAM2 jobs (15-18G each).
# Nothing here starts until a card has 21500 MiB free -- the same gate x11-x13 used. It waits,
# it never preempts.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
DATA=/media/data/forge_ts_gear_jit/real
TASK=Isaac-Forge-GearMesh-TBCamera-v0
REAL_POS="0.036666,-0.030747,-0.046328"; REAL_ROT="0.702792,0.013969,0.004503,0.711244"
MASTER=$LOGD/x14_master.log; LOCK=$LOGD/x14_gpulocks
MIN_FREE_MIB=21500; MIN_RAM_TRAIN=30; MIN_RAM_EVAL=12

mkdir -p "$LOGD" "$LOCK"; rmdir "$LOCK"/g* 2>/dev/null
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }
ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }

# Claim a card by lock dir so the trainer and the eval worker cannot pick the same one.
# Re-checks free VRAM after taking the lock: another tenant can grab it between the two.
claim(){
  local minram="$1" who="$2" G order
  while :; do
    if [ "$(ram_gb)" -ge "$minram" ]; then
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

# An OOM-killed run still writes a first-epoch best.pt, so existence is not completion.
# x11 grepped for reason=early_stop|max_epochs, but the wrapper writes natural_finish when it
# runs the full 30 -- that spelling would fail a perfectly good run three times over.
train_ok(){ [ -f "$1/best.pt" ] && grep -qE "RUN_SEED_EARLYSTOP_DONE reason=(early_stop|natural_finish)" "$1/train.log" 2>/dev/null; }

train_one(){
  local SEED="$1" OUT="$CKPT/jitcam_real_s$1" G att
  train_ok "$OUT" && { log "SKIP TRAIN s$SEED (already complete)"; return 0; }
  mkdir -p "$OUT"
  # Canonical norm_stats, never a fresh estimate: a refit is a 2000-sample subsample whose
  # value depends on shard load order, which would z-score this seed differently from s0.
  [ -f "$OUT/norm_stats.npz" ] || cp "$CKPT/norm_stats.npz" "$OUT/norm_stats.npz"
  for att in 1 2 3; do
    G=$(claim "$MIN_RAM_TRAIN" "train-s$SEED")
    log "TRAIN s$SEED attempt $att gpu=$G ram=$(ram_gb)GB"
    "$PY" -u "$SRC/student/run_seed_earlystop.py" --task gear --seed "$SEED" \
      --data_root "$DATA" --out_dir "$OUT" --gpu "$G" --patience 6 --max_epochs 30 \
      --batch_size 256 --num_workers 4 > "$LOGD/x14_train_s$SEED.log" 2>&1
    release "$G"
    train_ok "$OUT" && { log "TRAIN s$SEED OK epochs=$(grep -cE '^epoch ' "$OUT/train.log") $(grep -oE 'best_epoch=[0-9]+ best_val=[0-9.]+' "$OUT/train.log" | tail -1)"; return 0; }
    log "TRAIN s$SEED attempt $att FAILED: $(grep -oE 'reason=\S+( rc=\S+)?' "$OUT/train.log" 2>/dev/null | tail -1)"
    mv "$OUT/best.pt" "$OUT/best.pt.failed$att" 2>/dev/null; sleep 300
  done
  log "ABORT TRAIN s$SEED"; return 1
}

# Evaluated around the real camera pose, because that is the configuration being deployed.
# n2.5 goes first: it is the number in the handoff table, so it should exist earliest.
eval_seed(){
  local SEED="$1" OUT="$CKPT/jitcam_real_s$1" G LG N PM RD NAME TAG line
  train_ok "$OUT" || { log "EVAL s$SEED skipped: training never completed"; return 1; }
  while IFS= read -r line; do
    set -- $line; N=$1; PM=$2; RD=$3; NAME=$4
    TAG="x14_gear_real_s${SEED}_$NAME"; LG="$LOGD/$TAG.log"
    grep -qE "sr=[0-9.]+" "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
    G=$(claim "$MIN_RAM_EVAL" "eval-s$SEED")
    log "EVAL $TAG gpu=$G noise=${N}mm wrist=+${PM}mm/${RD}deg from the real pose"
    CUDA_VISIBLE_DEVICES=$G TB_CAM_WRIST_POS="$REAL_POS" TB_CAM_WRIST_ROT="$REAL_ROT" \
    TB_CAM_JITTER_POS_MM=$PM TB_CAM_JITTER_ROT_DEG=$RD \
    TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
      --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
      --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm "$N" \
      --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1
    log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
    release "$G"
    sleep 60   # let the trainer have a shot at the card instead of re-claiming immediately
  done <<'CASES'
2.5 0 0 n2.5
0 0 0 n0
1 0 0 n1
5 0 0 n5
2.5 25 5 wj25
2.5 30 6 wj30
CASES
  log "EVAL s$SEED done"
}

log "==================== x14 start: gear jitcam_real seeds 1,2 ===================="
log "tenancy at launch: ram=$(ram_gb)GB free=[$(free_on 0),$(free_on 1),$(free_on 2)]MiB"
train_one 1 && { eval_seed 1 & E1=$!; } || E1=""
train_one 2 && { eval_seed 2 & E2=$!; } || E2=""
[ -n "$E1" ] && wait "$E1"
[ -n "$E2" ] && wait "$E2"
log "==================== X14_ALL_DONE ===================="
{ echo "== x14 summary =="; grep -hE "^\[.*(TRAIN s|EVAL x14)" "$MASTER" | grep -vE "waiting|SKIP"; } >> "$MASTER"
