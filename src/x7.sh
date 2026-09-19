#!/usr/bin/env bash
# x7: three follow-ups from x5/x6, cheapest and most immediately usable first.
#
# P1  BLANK THE WRIST CAMERA. nut's full model scores 51.17 with the wrist jittered 20mm and
#     28.52 at 40mm, but 68.36 with vision removed altogether -- a wrong viewpoint costs more
#     than no viewpoint. The real mount is ~100mm off, far past 40mm, so if blanking beats
#     jitter then taping over the wrist camera is a usable stopgap until it is reprinted.
#     --ablate vision could not test this: it zeroes both cameras, and the third-person one
#     is in tolerance. --blank_cam wrist was added for exactly this.
#
# P2  DROP ONLY VISION DURING TRAINING. p=0.30 over all three modalities made peg MORE
#     vision-dependent: no-vision 60.55 -> 27.73, wrist jitter at 10mm 74.61 -> 23.83.
#     Vision is the only channel that can carry the task alone, so dropping the others just
#     pushes the optimiser onto it. Dropping only vision adds the one signal that is missing
#     -- operate without cameras -- and applies no pressure in the other direction.
#
# P3  SEEDS FOR THE EXISTING DROPOUT MODEL. Its 42.19 at 20mm rests on one seed against a
#     seed spread that reaches +-6.9 at 15mm. The wrist-jitter collapse is far too large to
#     be seed noise, but the high-noise number is not, and it should not be quoted until it
#     has company.
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
DATA=/media/data/forge_ts_data
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x7_master.log
QDIR=$LOGD/x7_queue; CLAIMS=$QDIR/claims
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-25}"
GPUS="${GPUS:-0 1 2}"

mkdir -p "$LOGD" "$QDIR"; rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
declare -A ENVNAME=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0
                     [gear]=Isaac-Forge-GearMesh-TBCamera-v0
                     [nut]=Isaac-Forge-NutThread-TBCamera-v0 )
log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
             | tr -d ' ' | awk -F, '{print $2-$1}'; }
free_ram_gb(){ free -g | awk '/^Mem:/{print $7}'; }
wait_for_gpu(){ local G="$1" f; while true; do f=$(free_on "$G")
    [ "${f:-0}" -ge "$MIN_FREE_MIB" ] && return 0; log "gpu $G ${f}MiB, waiting"; sleep 300; done; }

# EVAL <task> <ckptdir> <ablate> <noise> <jit_pos> <jit_rot> <blank_cam> <tag>
do_eval(){
    local G="$1" T="$2" CK="$3" ABL="$4" NM="$5" PM="$6" RD="$7" BC="$8" TAG="$9" rc
    local LG="$LOGD/${TAG}.log"
    grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; return 0; }
    [ -f "$CK/best.pt" ] || { log "SKIP $TAG -- no best.pt in $CK"; return 1; }
    wait_for_gpu "$G"
    log "EVAL $TAG gpu=$G ablate=$ABL noise=$NM jitter=$PM/$RD blank=$BC"
    CUDA_VISIBLE_DEVICES=$G \
    TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
    TB_CAM_JITTER_SEED=1 TB_CAM_JITTER_CAMS=wrist \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless --task "${ENVNAME[$T]}" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm "$NM" --dyn_rand on \
        --ablate "$ABL" --blank_cam "$BC" --tag "$TAG" > "$LG" 2>&1
    rc=$?
    log "EVAL $TAG rc=$rc $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1)"
}

train_ok(){ [ -f "$1/best.pt" ] && \
    grep -q "RUN_SEED_EARLYSTOP_DONE reason=\(early_stop\|max_epochs\)" "$1/train.log" 2>/dev/null; }

# TRAIN <task> <seed> <outdir> <dropout_mods> <logtag>
do_train(){
    local G="$1" T="$2" S="$3" OUT="$4" MODS="$5" LT="$6" attempt rc
    train_ok "$OUT" && { log "SKIP TRAIN $LT (complete)"; return 0; }
    mkdir -p "$OUT"
    # Inherit the canonical norm_stats: compute_norm_stats() is a 2000-sample estimate whose
    # index->sample mapping depends on shard load order, so a fresh one would z-score the
    # inputs differently from every cell this is compared against.
    [ -f "$OUT/norm_stats.npz" ] || cp "$CKPT_ROOT/$T/norm_stats.npz" "$OUT/norm_stats.npz" \
      || { log "ABORT $LT: no canonical norm_stats"; return 1; }
    for attempt in 1 2 3; do
        while [ "$(free_ram_gb)" -lt "$MIN_FREE_RAM_GB" ]; do
            log "host RAM $(free_ram_gb)GB < $MIN_FREE_RAM_GB, waiting"; sleep 300; done
        wait_for_gpu "$G"
        log "TRAIN $LT attempt $attempt gpu=$G ram=$(free_ram_gb)GB mods=$MODS"
        "$PY" -u "$SRC/student/run_seed_earlystop.py" \
            --task "$T" --seed "$S" --data_root "$DATA" --out_dir "$OUT" --gpu "$G" \
            --patience 6 --max_epochs 30 --batch_size 256 --num_workers 4 \
            --extra_args "--modality_dropout 0.30 --dropout_modalities $MODS" \
            > "$LOGD/x7_train_${LT}.log" 2>&1
        rc=$?
        # An OOM-killed run still writes a first-epoch best.pt, so existence is not completion.
        train_ok "$OUT" && { log "TRAIN $LT OK attempt $attempt ($(grep -cE '^epoch ' "$OUT/train.log") epochs)"; return 0; }
        log "TRAIN $LT attempt $attempt FAILED: $(grep -oE 'reason=\S+ ?\S*' "$OUT/train.log" 2>/dev/null | tail -1)"
        mv "$OUT/best.pt" "$OUT/best.pt.failed_attempt$attempt" 2>/dev/null
        sleep 300
    done
    log "ABORT $LT"; return 1
}

run_phase(){
    local PF="$1" name="$2" nw="$3" G pids=() used=0
    log "======== phase $name: $(wc -l < "$PF") jobs, $nw workers ========"
    for G in $GPUS; do
        used=$((used+1)); [ "$used" -gt "$nw" ] && break
        ( n=0
          while IFS= read -r line; do
              n=$((n+1)); mkdir "$CLAIMS/$(basename "$PF").$n" 2>/dev/null || continue
              set -- $line
              case "$1" in
                EVAL)  do_eval  "$G" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" ;;
                TRAIN) do_train "$G" "$2" "$3" "$4" "$5" "$6" ;;
              esac
          done < "$PF" ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    log "======== phase $name done ========"
}

log "==================== x7 start ===================="

# ---------------------------------------------------------------- P1 blank wrist
P1=$QDIR/p1; : > "$P1"
for T in peg gear nut; do
  for NM in 0 2.5 5; do
    echo "EVAL $T $CKPT_ROOT/$T/gate2_seed0  none  $NM 0 0 wrist bw_${T}_full_n${NM}"    >> "$P1"
  done
  echo   "EVAL $T $CKPT_ROOT/$T/ablate_state state 2.5 0 0 wrist bw_${T}_nostate_n2.5"   >> "$P1"
  # Which camera carries the task: same intervention on the one that is already in tolerance.
  echo   "EVAL $T $CKPT_ROOT/$T/gate2_seed0  none  2.5 0 0 tp    bt_${T}_full_n2.5"      >> "$P1"
done
run_phase "$P1" "1 (blank one camera)" 3

# ---------------------------------------------------------------- P2 vision-only dropout
P2=$QDIR/p2; : > "$P2"
echo "TRAIN peg 0 $CKPT_ROOT/peg/moddrop_vis_p30 vision peg_moddrop_vis" >> "$P2"
run_phase "$P2" "2a (train: drop vision only)" 1

P2b=$QDIR/p2b; : > "$P2b"
MV=$CKPT_ROOT/peg/moddrop_vis_p30
for NM in 0 1 2.5 5 7.5 10 15 20; do echo "EVAL peg $MV none $NM 0 0 none mv_clean_n${NM}" >> "$P2b"; done
for J in "10 2" "20 4" "40 8"; do set -- $J; echo "EVAL peg $MV none 2.5 $1 $2 none mv_wj_p$1r$2" >> "$P2b"; done
echo "EVAL peg $MV none   2.5 0 0 wrist mv_blankwrist_n2.5" >> "$P2b"
for NM in 0 5; do echo "EVAL peg $MV vision $NM 0 0 none mv_novision_n${NM}" >> "$P2b"; done
run_phase "$P2b" "2b (eval: drop vision only)" 3

# ---------------------------------------------------------------- P3 seeds for x6's model
P3=$QDIR/p3; : > "$P3"
for S in 1 2; do
  echo "TRAIN peg $S $CKPT_ROOT/peg/moddrop_p30_s$S state,vision,force peg_moddrop_s$S" >> "$P3"
done
run_phase "$P3" "3a (moddrop seeds 1,2)" 2

P3b=$QDIR/p3b; : > "$P3b"
for S in 1 2; do
  for NM in 0 1 2.5 5 7.5 10 15 20; do
    echo "EVAL peg $CKPT_ROOT/peg/moddrop_p30_s$S none $NM 0 0 none x6_md_s${S}_clean_n${NM}" >> "$P3b"
  done
  echo "EVAL peg $CKPT_ROOT/peg/moddrop_p30_s$S none 2.5 10 2 none x6_md_s${S}_wj_p10r2" >> "$P3b"
done
run_phase "$P3b" "3b (moddrop seeds eval)" 3

log "==================== X7_ALL_DONE ===================="
