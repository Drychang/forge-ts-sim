#!/usr/bin/env bash
# x10: combine the two things that worked, completing a 2x2.
#
# The session produced two independent robustness levers, each aimed at a different input
# going out of distribution:
#   -state           wins when the POSE ESTIMATE is bad (62.50 vs 53.00 at 20mm noise),
#                    because training without state forces a stronger visual pathway
#   jitter-augmented wins when the CAMERA is bad (96.48 vs 12.89 at 20mm mount error),
#                    because it is the only intervention that trains on wrong viewpoints
#                    rather than absent ones
#
# They address different failure modes and nothing says they conflict -- but nothing says
# they compose either. -state leans harder on vision, and jitter makes vision less reliable,
# so the combination could equally cancel. The data for both arms already exists, so the
# question costs two trainings.
#
#   {ctrl, jit} x {full, -state}, and the full cells are already done in x8.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src; CKPT=~/forge_ts/student_ckpts/peg; LOGD=~/forge_ts/logs
NEW=/media/data/forge_ts_data_jit
MASTER=$LOGD/x10_master.log; CLAIMS=$LOGD/x10_claims
ENVN=Isaac-Forge-PegInsert-TBCamera-v0
MIN_FREE_MIB=21500; MIN_RAM_GB=25
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS" "$LOGD"
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" | tr -d ' ' | awk -F, '{print $2-$1}'; }
ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }
wait_res(){ while [ "$(free_on $1)" -lt "$MIN_FREE_MIB" ] || [ "$(ram_gb)" -lt "$MIN_RAM_GB" ]; do
    log "gpu $1 $(free_on $1)MiB / ram $(ram_gb)GB, waiting"; sleep 240; done; }
train_ok(){ [ -f "$1/best.pt" ] && grep -q "RUN_SEED_EARLYSTOP_DONE reason=\(early_stop\|max_epochs\)" "$1/train.log" 2>/dev/null; }

log "==== x10 start: waiting for x9 ===="
while ! grep -q "X9_ALL_DONE" "$LOGD/x9_master.log" 2>/dev/null; do sleep 120; done

# ---------------------------------------------------------------- train
order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1 | head -2)
i=0; pids=()
for A in ctrl jit; do
  G=$(echo $order | cut -d' ' -f$((i+1))); i=$((i+1))
  ( OUT=$CKPT/jitdata_${A}_nostate
    train_ok "$OUT" && { log "SKIP TRAIN $A"; exit 0; }
    mkdir -p "$OUT"
    # Same canonical norm_stats as every other cell; a fresh estimate is a 2000-sample
    # subsample whose value depends on shard load order.
    [ -f "$OUT/norm_stats.npz" ] || cp "$CKPT/norm_stats.npz" "$OUT/norm_stats.npz"
    for att in 1 2 3; do
      wait_res "$G"
      log "TRAIN ${A}_nostate attempt $att gpu=$G ram=$(ram_gb)GB"
      "$PY" -u "$SRC/student/run_seed_earlystop.py" --task peg --seed 0 \
        --data_root "$NEW/$A" --out_dir "$OUT" --gpu "$G" --patience 6 --max_epochs 30 \
        --batch_size 256 --num_workers 4 --ablate state > "$LOGD/x10_train_$A.log" 2>&1
      # An OOM-killed run still writes a first-epoch best.pt, so existence is not completion.
      train_ok "$OUT" && { log "TRAIN ${A}_nostate OK ($(grep -cE '^epoch ' "$OUT/train.log") epochs)"; exit 0; }
      log "TRAIN ${A}_nostate attempt $att FAILED: $(grep -oE 'reason=\S+ ?\S*' "$OUT/train.log" 2>/dev/null | tail -1)"
      mv "$OUT/best.pt" "$OUT/best.pt.failed$att" 2>/dev/null; sleep 300
    done
    log "ABORT ${A}_nostate" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==== training done ===="

# ---------------------------------------------------------------- eval
Q=$LOGD/x10_jobs; : > "$Q"
for A in ctrl jit; do
  for NM in 0 1 2.5 5 7.5 10 15 20; do echo "$A $NM 0 0 x10_${A}_ns_clean_n${NM}" >> "$Q"; done
  for J in "10 2" "20 4" "40 8" "100 20"; do set -- $J; echo "$A 2.5 $1 $2 x10_${A}_ns_wj_p$1r$2" >> "$Q"; done
done
log "==== eval: $(wc -l < $Q) jobs ===="
order=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -n | cut -d, -f1)
pids=()
for G in $order; do
 ( n=0
   while IFS= read -r line; do
     n=$((n+1)); mkdir "$CLAIMS/j$n" 2>/dev/null || continue
     set -- $line; A=$1; NM=$2; PM=$3; RD=$4; TAG=$5; LG=$LOGD/$TAG.log
     grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; continue; }
     M=$CKPT/jitdata_${A}_nostate
     [ -f "$M/best.pt" ] || { log "SKIP $TAG -- no best.pt"; continue; }
     wait_res "$G"
     log "EVAL $TAG gpu=$G"
     CUDA_VISIBLE_DEVICES=$G TB_CAM_JITTER_POS_MM=$PM TB_CAM_JITTER_ROT_DEG=$RD \
     TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
     "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "$ENVN" \
       --checkpoint "$M/best.pt" --norm_stats "$M/norm_stats.npz" \
       --episodes 256 --num_envs 32 --protocol_seed 42 --fixed_pos_noise_mm "$NM" \
       --dyn_rand on --ablate state --blank_cam none --tag "$TAG" > "$LG" 2>&1
     log "EVAL $TAG rc=$? $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
   done < "$Q" ) &
 pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X10_ALL_DONE ===================="
