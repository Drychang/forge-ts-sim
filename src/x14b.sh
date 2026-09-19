#!/usr/bin/env bash
# x14b: redo seed 1, which x14 aborted after three CUDA OOMs.
#
# Why it failed: the 21500 MiB gate only checks the instant before launch. The other tenant's
# jobs arrive in bursts and pick whichever card looks emptiest -- which, while our training is
# still ramping from 6GB toward its 20.3GB peak, is the card we are already on. All three
# attempts died the same way: "this process has 5.94 GiB in use, Process <theirs> has 17.64 GiB".
#
# Two changes:
#   1. Exclusive start -- the card must be genuinely empty (<1000 MiB), not merely 21500 free.
#   2. A 2GB holder process on the same card for the duration. It closes the window: from the
#      first second the card reads ~22.8GB used, so a newcomer scanning for the emptiest GPU
#      goes elsewhere instead of landing on top of us. This is one card out of three -- our
#      share -- and it preempts nobody: no process is signalled, nothing of theirs is killed.
#
# Serialised behind x14's s2 training: both seeds eager-load the same 12G dataset to ~30G of
# RAM, and 2x30G does not fit in 62G. Waits for x14 to finish with s2 before starting.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/gear; LOGD=~/forge_ts/logs
DATA=/media/data/forge_ts_gear_jit/real
TASK=Isaac-Forge-GearMesh-TBCamera-v0
REAL_POS="0.036666,-0.030747,-0.046328"; REAL_ROT="0.702792,0.013969,0.004503,0.711244"
MASTER=$LOGD/x14b_master.log; LOCK=$LOGD/x14_gpulocks   # same locks as x14: they coexist
SEED=1; OUT=$CKPT/jitcam_real_s$SEED
MAX_USED_MIB=1000; MIN_RAM_TRAIN=34; MIN_RAM_EVAL=12; HOLD_MIB=2048; ATTEMPTS=12

mkdir -p "$LOGD" "$LOCK"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
used_on(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }
ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }

# Empty card, not just a roomy one.
claim_exclusive(){
  local minram="$1" who="$2" G
  while :; do
    if [ "$(ram_gb)" -ge "$minram" ]; then
      for G in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1); do
        [ "$(used_on "$G")" -le "$MAX_USED_MIB" ] || continue
        mkdir "$LOCK/g$G" 2>/dev/null || continue
        if [ "$(used_on "$G")" -gt "$MAX_USED_MIB" ]; then rmdir "$LOCK/g$G" 2>/dev/null; continue; fi
        echo "$G"; return 0
      done
    fi
    log "$who waiting: ram=$(ram_gb)GB used=[$(used_on 0),$(used_on 1),$(used_on 2)]MiB"
    sleep 180
  done
}
release(){ rmdir "$LOCK/g$1" 2>/dev/null; }

# Reserve the rest of the card so we do not look like the emptiest one while ramping.
hold_start(){
  local G="$1" F="$LOGD/x14b_hold_$1.release"
  rm -f "$F"
  CUDA_VISIBLE_DEVICES=$G nohup "$PY" -u -c "
import torch,time,os,sys
n=int(os.environ['HOLD_MIB'])
x=torch.empty(n*1024*1024//2, dtype=torch.float16, device='cuda')  # touch it so it is real
x.fill_(0); torch.cuda.synchronize()
f=os.environ['REL']
while not os.path.exists(f): time.sleep(20)
" > "$LOGD/x14b_hold_$1.log" 2>&1 &
  HOLD_PID=$!
  sleep 45
  kill -0 "$HOLD_PID" 2>/dev/null && log "HOLD $HOLD_MIB MiB on gpu $G pid=$HOLD_PID (card now $(used_on "$G")MiB)" \
    || { log "HOLD failed on gpu $G, continuing without it"; HOLD_PID=""; }
}
hold_stop(){ [ -n "${HOLD_PID:-}" ] && { touch "$LOGD/x14b_hold_$1.release"; sleep 25; kill "$HOLD_PID" 2>/dev/null; }; HOLD_PID=""; }

train_ok(){ [ -f "$OUT/best.pt" ] && grep -qE "RUN_SEED_EARLYSTOP_DONE reason=(early_stop|natural_finish)" "$OUT/train.log" 2>/dev/null; }

log "==================== x14b start: redo gear jitcam_real seed 1 ===================="
log "waiting for x14 to finish with s2 (RAM cannot hold two trainings)"
while ! grep -qE "TRAIN s2 OK|ABORT TRAIN s2" "$LOGD/x14_master.log" 2>/dev/null; do sleep 180; done
log "x14 s2 done: $(grep -E 'TRAIN s2 OK|ABORT TRAIN s2' "$LOGD/x14_master.log" | tail -1)"

mkdir -p "$OUT"
[ -f "$OUT/norm_stats.npz" ] || cp "$CKPT/norm_stats.npz" "$OUT/norm_stats.npz"
for att in $(seq 1 $ATTEMPTS); do
  train_ok && break
  G=$(claim_exclusive "$MIN_RAM_TRAIN" "train-s1")
  export HOLD_MIB REL="$LOGD/x14b_hold_$G.release"
  hold_start "$G"
  log "TRAIN s1 attempt $att gpu=$G ram=$(ram_gb)GB card=$(used_on "$G")MiB"
  "$PY" -u "$SRC/student/run_seed_earlystop.py" --task gear --seed "$SEED" \
    --data_root "$DATA" --out_dir "$OUT" --gpu "$G" --patience 6 --max_epochs 30 \
    --batch_size 256 --num_workers 4 > "$LOGD/x14b_train_s1_att$att.log" 2>&1
  hold_stop "$G"; release "$G"
  if train_ok; then
    log "TRAIN s1 OK epochs=$(grep -cE '^epoch ' "$OUT/train.log") $(grep -oE 'best_epoch=[0-9]+ best_val=[0-9.]+' "$OUT/train.log" | tail -1)"
    break
  fi
  log "TRAIN s1 attempt $att FAILED: $(grep -oE 'reason=\S+( rc=\S+)?' "$OUT/train.log" | tail -1) | $(grep -oE 'torch.OutOfMemoryError.{0,60}' "$OUT/train.log" | tail -1)"
  mv "$OUT/best.pt" "$OUT/best.pt.failed_b$att" 2>/dev/null; sleep 240
done
train_ok || { log "==================== X14B_TRAIN_FAILED after $ATTEMPTS attempts ===================="; exit 1; }

while IFS= read -r line; do
  set -- $line; N=$1; PM=$2; RD=$3; NAME=$4
  TAG="x14_gear_real_s1_$NAME"; LG="$LOGD/$TAG.log"
  grep -qE "sr=[0-9.]+" "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
  G=$(claim_exclusive "$MIN_RAM_EVAL" "eval-s1")
  log "EVAL $TAG gpu=$G noise=${N}mm wrist=+${PM}mm/${RD}deg from the real pose"
  CUDA_VISIBLE_DEVICES=$G TB_CAM_WRIST_POS="$REAL_POS" TB_CAM_WRIST_ROT="$REAL_ROT" \
  TB_CAM_JITTER_POS_MM=$PM TB_CAM_JITTER_ROT_DEG=$RD \
  TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
  "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$TASK" \
    --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
    --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm "$N" \
    --dyn_rand on --ablate none --blank_cam none --tag "$TAG" > "$LG" 2>&1
  log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
  release "$G"; sleep 60
done <<'CASES'
2.5 0 0 n2.5
0 0 0 n0
1 0 0 n1
5 0 0 n5
2.5 25 5 wj25
2.5 30 6 wj30
CASES
log "==================== X14B_ALL_DONE ===================="
