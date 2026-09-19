#!/usr/bin/env bash
# Stage-D remainder: finish the T3 modality-ablation matrix after the original
# queue parent died. Idempotent (skips any (task, ablate) that already has
# best.pt) and GPU-defensive (checks free VRAM before every training/eval and
# falls back to another GPU or waits -- the original stage D failed because a
# transient 17GB job from another user squatted GPU1; never assume a GPU is
# free just because it was an hour ago).
# NEW FILE on purpose -- never overwrite a running script (bash lazy-read).
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
CK=~/forge_ts/student_ckpts
LOGS=~/forge_ts/logs
DATA=/media/data/forge_ts_data
ML="$LOGS/stage_d_remainder_master.log"
mkdir -p "$LOGS"

declare -A GYM=( [peg]=PegInsert [gear]=GearMesh [nut]=NutThread )
log() { echo "=== [$(date -Is)] $*" >> "$ML"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }

wait_gpu() {  # $1=min_free_mib, rest=preference-ordered gpu indices; echoes chosen index
  local need=$1; shift
  while true; do
    for g in "$@"; do
      if [ "$(free_mib "$g")" -ge "$need" ]; then echo "$g"; return 0; fi
    done
    log "WAIT no GPU with ${need}MiB free among [$*] -- sleep 600s"
    sleep 600
  done
}

log "stage-D remainder starting; waiting for any live camera training to finish first"
until ! pgrep -f "train_student_bc.py" >/dev/null 2>&1; do sleep 180; done
log "no camera training running -- proceeding"

for task in peg gear nut; do
  gym=${GYM[$task]}
  for ab in vision force both; do
    outdir="$CK/$task/ablate_${ab}"
    mkdir -p "$outdir"
    cp -n "$CK/$task/norm_stats.npz" "$outdir/norm_stats.npz"

    if [ -f "$outdir/best.pt" ]; then
      log "SKIP train $task ablate=$ab (best.pt exists)"
    else
      tg=$(wait_gpu 19000 1 2)
      log "D START train $task ablate=$ab on GPU$tg"
      $PY -u "$SRC/run_seed_earlystop.py" \
        --task "$task" --seed 0 --data_root "$DATA" \
        --out_dir "$outdir" --gpu "$tg" --patience 6 --max_epochs 30 --ablate "$ab" \
        >> "$LOGS/postg1_D_train_${task}_${ab}.log" 2>&1
      log "D END train $task ablate=$ab rc=$?"
    fi

    if [ ! -f "$outdir/best.pt" ]; then
      log "D ERROR no best.pt $task $ab -- skipping its eval"
      continue
    fi
    eg=$(wait_gpu 14000 0 2 1)
    log "D START eval $task ablate=$ab on GPU$eg"
    for noise in 0 2.5 5; do
      CUDA_VISIBLE_DEVICES=$eg $PY "$SRC/eval_ablate_student.py" --headless \
        --task "Isaac-Forge-${gym}-TBCamera-v0" \
        --checkpoint "$outdir/best.pt" --norm_stats "$outdir/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm $noise --dyn_rand on --ablate "$ab" --tag "ablate_${ab}" \
        >> "$LOGS/postg1_D_eval_${task}_${ab}.log" 2>&1
      log "D eval $task $ab noise=$noise rc=$?"
    done
  done
done
log "STAGE_D_REMAINDER_ALL_DONE"
