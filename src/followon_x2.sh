#!/usr/bin/env bash
# Follow-on to launch_bp.sh. Everything here answers one question: is the +8.98 point
# advantage the state-ablated peg student showed at 20mm real, or is it seed noise?
#
# The full model's own seed spread at 10mm is already 90.23 / 85.16 / 91.02 -- a range of
# 5.86 points from three seeds of the SAME configuration. Spread grows with noise, so at
# 20mm it could plausibly exceed the 8.98 gap entirely. One seed per arm cannot tell the
# difference between an effect and a draw.
#
# Stages, cheapest-decisive first:
#   0  code-path check: re-eval gate2_seed0 @15mm through eval_ablate_student --ablate none.
#      The existing 0.7422 came from eval_frozen_student.py. If the two paths disagree the
#      whole cross-script table is invalid, and that is worth 12 minutes to rule out.
#   1  full model seeds 1,2 at 7.5/10/15/20 -- the seed spread at high noise
#   2  gear + nut state-ablated at 7.5/10/15/20 -- completes the breaking-point row
#   3  peg state-ablated seeds 1,2 -- 3-seed statistics on the headline claim
#
# Written to a file and scp'd rather than passed inline: nested quoting of a space-
# separated list did not survive an ssh round trip before and the loop silently ran zero
# iterations.
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x2_master.log
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"
CHAIN_PID="${CHAIN_PID:-936882}"

mkdir -p "$LOGD"
declare -A ENVNAME=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0
                     [gear]=Isaac-Forge-GearMesh-TBCamera-v0
                     [nut]=Isaac-Forge-NutThread-TBCamera-v0 )

log(){ echo "[$(date -Is)] $*" | tee -a "$MASTER"; }

# Only ever counts GPUs by free memory. Other users' processes are never touched: frank's
# VLLM workers hold GPU0/1 and this simply waits rather than competing for them.
pick_gpu(){
    while true; do
        local best=-1 bestfree=-1 idx used total free
        while IFS=, read -r idx used total; do
            free=$(( total - used ))
            if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$idx; fi
        done < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
                            --format=csv,noheader,nounits | tr -d ' ')
        if [ "$bestfree" -ge "$MIN_FREE_MIB" ]; then echo "$best"; return 0; fi
        log "waiting for a GPU: best free ${bestfree} MiB < ${MIN_FREE_MIB}" >&2
        sleep 600
    done
}

# Bracketed so the pattern cannot match this script's own pgrep.
wait_for_my_jobs(){
    while [ "$(pgrep -fc 'train_student_b[c].py|eval_ablate_studen[t].py|eval_frozen_studen[t].py' || true)" -gt 0 ]; do
        sleep 120
    done
}

# $1 task  $2 ckpt dir  $3 ablate  $4 noise  $5 tag
run_eval(){
    local T="$1" CK="$2" ABL="$3" NM="$4" TAG="$5"
    local LG="$LOGD/${TAG}.log"
    if grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null; then
        log "SKIP  $TAG (already done)"; return 0
    fi
    [ -f "$CK/best.pt" ] || { log "SKIP  $TAG -- no best.pt in $CK"; return 1; }
    wait_for_my_jobs
    local G; G=$(pick_gpu)
    log "EVAL  $TAG  gpu=$G"
    CUDA_VISIBLE_DEVICES=$G "$PY" "$SRC/student/eval_ablate_student.py" --headless \
        --task "${ENVNAME[$T]}" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm "$NM" --dyn_rand on \
        --ablate "$ABL" --tag "$TAG" > "$LG" 2>&1
    local rc=$?
    local sr; sr=$(grep -oE "sr=[0-9.]+" "$LG" | tail -1)
    log "EVAL  $TAG  rc=$rc  $sr"
    [ "$rc" -eq 0 ]
}

log "==================== x2 follow-on start ===================="
log "waiting for the launch_bp chain (pid $CHAIN_PID) to finish"
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 300; done
log "chain finished; proceeding"

# --------------------------------------------------------------- provenance check
# Every cell in the comparison table must be z-scored by the same stats. Cheap to verify,
# expensive to discover later.
log "norm_stats md5 (all peg variants must match the canonical one):"
md5sum "$CKPT_ROOT/peg/norm_stats.npz" \
       "$CKPT_ROOT/peg/gate2_seed"{0,1,2}"/norm_stats.npz" \
       "$CKPT_ROOT/peg/ablate_state/norm_stats.npz" 2>&1 | sed 's/^/    /' | tee -a "$MASTER"

# --------------------------------------------------------------- stage 0
log "--- stage 0: code-path equivalence, expect sr=0.7422 ---"
run_eval peg "$CKPT_ROOT/peg/gate2_seed0" none 15 "x2_full_s0_n15"
S0=$(grep -oE "sr=[0-9.]+" "$LOGD/x2_full_s0_n15.log" 2>/dev/null | tail -1)
log "stage 0 result: $S0  (eval_frozen_student gave sr=0.7422)"

# --------------------------------------------------------------- stage 1
log "--- stage 1: full-model seed spread at high noise ---"
for S in 1 2; do
  for NM in 7.5 10 15 20; do
    run_eval peg "$CKPT_ROOT/peg/gate2_seed$S" none "$NM" "x2_full_s${S}_n${NM}"
  done
done
# seed 0 at the remaining levels, same code path as the other two
for NM in 7.5 10 20; do
  run_eval peg "$CKPT_ROOT/peg/gate2_seed0" none "$NM" "x2_full_s0_n${NM}"
done

# --------------------------------------------------------------- stage 2
log "--- stage 2: gear + nut state-ablated, breaking point ---"
for T in gear nut; do
  for NM in 7.5 10 15 20; do
    run_eval "$T" "$CKPT_ROOT/$T/ablate_state" state "$NM" "x2_state_${T}_s0_n${NM}"
  done
done

# --------------------------------------------------------------- stage 3
log "--- stage 3: peg state-ablated seeds 1,2 (train + eval) ---"
for S in 1 2; do
  OUT="$CKPT_ROOT/peg/ablate_state_s$S"
  mkdir -p "$OUT"
  # Inherit the canonical norm_stats. compute_norm_stats() is a 2000-sample estimate whose
  # index->sample mapping depends on shard load order, so a fresh one differs by ~0.14 on
  # some channels -- enough to z-score the inputs differently from every cell it is
  # compared against.
  if [ ! -f "$OUT/norm_stats.npz" ]; then
      cp "$CKPT_ROOT/peg/norm_stats.npz" "$OUT/norm_stats.npz" \
        || { log "ABORT seed $S: no canonical norm_stats"; continue; }
      log "peg/ablate_state_s$S: inherited canonical norm_stats"
  fi
  if [ -f "$OUT/best.pt" ]; then
      log "peg ablate_state seed $S already trained"
  else
      wait_for_my_jobs
      G=$(pick_gpu)
      log "TRAIN peg ablate=state seed=$S gpu=$G -> $OUT"
      "$PY" -u "$SRC/student/run_seed_earlystop.py" \
          --task peg --seed "$S" --data_root "$DATA" --out_dir "$OUT" --gpu "$G" \
          --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
          --ablate state > "$LOGD/x2_train_peg_state_s$S.log" 2>&1
      rc=$?
      log "TRAIN peg seed=$S rc=$rc best.pt=$([ -f "$OUT/best.pt" ] && echo yes || echo NO)"
      # Hard gate. Without it an earlier version marched through every downstream step
      # after each one had failed, then printed ALL_DONE.
      if [ ! -f "$OUT/best.pt" ]; then
          log "ABORT peg seed $S: training produced no best.pt"
          tail -20 "$LOGD/x2_train_peg_state_s$S.log" | sed 's/^/    /' | tee -a "$MASTER"
          continue
      fi
  fi
  for NM in 0 1 2.5 5 7.5 10 15 20; do
      run_eval peg "$OUT" state "$NM" "x2_state_s${S}_n${NM}"
  done
done

log "==================== X2_ALL_DONE ===================="
