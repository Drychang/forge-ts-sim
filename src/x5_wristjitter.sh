#!/usr/bin/env bash
# x5: does keeping state in the observation buy any protection against a miscalibrated
# camera? This is the deployment question, and it is the mirror image of the noise study.
#
#   in simulation at high pose noise : state leaves distribution, vision does not
#   on the real robot                : vision leaves distribution, state does not
#                                      (the teach error is 1.5mm, inside the 0-5mm
#                                       range the student was trained on)
#
# So the two ablations answer opposite deployment risks, and the sim result "drop state"
# must not be carried over to a rig whose wrist camera is wrong.
#
# What the existing data already says: the FULL model -- which has state -- falls to 2.34%
# (peg) and 0.00% (gear) under 20mm/4 deg jitter, below what a state-only model manages at
# 5mm pose noise. A fixed BC fusion does not fall back on the intact modality; the corrupted
# visual features propagate into the fused representation and drag the output with them.
#
# What it does NOT say: those runs perturbed BOTH cameras. The real rig has the third-person
# camera in tolerance (fg 0.442 vs 0.434) and only the wrist mount wrong. That is the case
# TB_CAM_JITTER_CAMS was added for and it has never been run.
#
# Both arms see the identical perturbation: the jitter is drawn from (seed, camera_name) and
# applied once per process, so a shared seed means a shared camera pose. Anything else would
# compare two different cameras rather than two different observation spaces.
#
# Scale note for reading the result: the simulator's wrist camera sits 198.5 mm from
# panda_hand ((0.13, 0, -0.15)), and the real mount is at roughly twice that stand-off --
# a ~200 mm error, an order of magnitude beyond the largest level swept here. If the full
# model has no advantage even at 40 mm, no amount of state input rescues the real mount.
set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
CKPT_ROOT=~/forge_ts/student_ckpts
LOGD=~/forge_ts/logs
MASTER=$LOGD/x5_master.log
QDIR=$LOGD/x5_queue
CLAIMS=$QDIR/claims
MIN_FREE_MIB="${MIN_FREE_MIB:-21500}"
GPUS="${GPUS:-0 1 2}"
NOISE_MM="${NOISE_MM:-2.5}"      # the realistic deployment regime, not the breaking point
JITTER_SEED="${JITTER_SEED:-1}"  # matches the existing camjit_*_s1 runs

mkdir -p "$LOGD" "$QDIR"; rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
declare -A ENVNAME=( [peg]=Isaac-Forge-PegInsert-TBCamera-v0
                     [gear]=Isaac-Forge-GearMesh-TBCamera-v0
                     [nut]=Isaac-Forge-NutThread-TBCamera-v0 )

log(){ echo "[$(date -Is)] $*" >> "$MASTER"; }
free_on(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
             | tr -d ' ' | awk -F, '{print $2-$1}'; }
wait_for_gpu(){ local G="$1" f; while true; do f=$(free_on "$G")
    [ "${f:-0}" -ge "$MIN_FREE_MIB" ] && return 0
    log "gpu $G ${f}MiB free, waiting"; sleep 300; done; }

# $1 gpu  $2 task  $3 ckptdir  $4 ablate  $5 pos_mm  $6 rot_deg  $7 tag
do_eval(){
    local G="$1" T="$2" CK="$3" ABL="$4" PM="$5" RD="$6" TAG="$7" rc
    local LG="$LOGD/${TAG}.log"
    grep -q "EVAL_DONE\|EVAL_SUMMARY" "$LG" 2>/dev/null && { log "SKIP $TAG"; return 0; }
    [ -f "$CK/best.pt" ] || { log "SKIP $TAG -- no best.pt"; return 1; }
    wait_for_gpu "$G"
    log "EVAL $TAG gpu=$G  wrist jitter ${PM}mm/${RD}deg"
    CUDA_VISIBLE_DEVICES=$G \
    TB_CAM_JITTER_POS_MM="$PM" TB_CAM_JITTER_ROT_DEG="$RD" \
    TB_CAM_JITTER_SEED="$JITTER_SEED" TB_CAM_JITTER_CAMS=wrist \
    "$PY" "$SRC/student/eval_ablate_student.py" --headless \
        --task "${ENVNAME[$T]}" \
        --checkpoint "$CK/best.pt" --norm_stats "$CK/norm_stats.npz" \
        --episodes 256 --num_envs 32 --protocol_seed 42 \
        --fixed_pos_noise_mm "$NOISE_MM" --dyn_rand on \
        --ablate "$ABL" --tag "$TAG" > "$LG" 2>&1
    rc=$?
    # The env var only takes effect if the camera cfg actually saw it. Record the proof.
    log "EVAL $TAG rc=$rc $(grep -oE 'sr=[0-9.]+' "$LG" | tail -1) | $(grep -oE 'CAM JITTER \w+: (SKIPPED|pos_mm=[0-9.]+)' "$LG" | tr '\n' ' ')"
}

log "==================== x5 start ===================="
log "waiting for x4 to finish"
while ! grep -q "X4_ALL_DONE" "$LOGD/x4_master.log" 2>/dev/null; do sleep 300; done
log "x4 done, proceeding"

Q=$QDIR/jobs; : > "$Q"
for T in peg gear nut; do
  for J in "10 2" "20 4" "40 8"; do
    set -- $J
    echo "EVAL $T $CKPT_ROOT/$T/gate2_seed0   none  $1 $2 wj_${T}_full_p$1r$2"   >> "$Q"
    echo "EVAL $T $CKPT_ROOT/$T/ablate_state  state $1 $2 wj_${T}_nostate_p$1r$2" >> "$Q"
  done
done
log "$(wc -l < "$Q") jobs queued"

pids=()
for G in $GPUS; do
    (
        n=0
        while IFS= read -r line; do
            n=$((n+1))
            mkdir "$CLAIMS/j$n" 2>/dev/null || continue
            set -- $line
            do_eval "$G" "$2" "$3" "$4" "$5" "$6" "$7"
        done < "$Q"
    ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "==================== X5_ALL_DONE ===================="
