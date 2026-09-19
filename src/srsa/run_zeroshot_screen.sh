#!/usr/bin/env bash
# SRSA Phase 2: zero-shot screening -- 5 library checkpoints x 3 FORGE tasks,
# 64 eps each, noise 0mm, dyn_rand on, frozen-protocol seed. GPU1.
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/srsa
LOG_DIR=~/forge_ts/logs
MASTER=$LOG_DIR/srsa_zeroshot_master.log
CKPT_DIR=/extra_home3/user/SRSA/checkpoints
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

for pair in "peg Isaac-SRSA-Forge-PegInsert-v0" "gear Isaac-SRSA-Forge-GearMesh-v0" "nut Isaac-SRSA-Forge-NutThread-v0"; do
  set -- $pair
  t=$1; gymid=$2
  for ck in 00141 00211 00426 00638 00783; do
    lg=$LOG_DIR/srsa_zs_${t}_${ck}.log
    if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then
      log "SKIP $t $ck (done)"
      continue
    fi
    log "zeroshot $t ckpt=$ck START"
    CUDA_VISIBLE_DEVICES=1 $PY "$SRC/screen_zeroshot.py" --headless \
      --task "$gymid" --checkpoint "$CKPT_DIR/$ck.pth" \
      --num_envs 32 --episodes 64 --protocol_seed 42 \
      --fixed_pos_noise_mm 0 --dyn_rand on --tag "zs_${ck}" \
      >> "$lg" 2>&1
    log "zeroshot $t ckpt=$ck rc=$?"
  done
done
log "SRSA_ZEROSHOT_ALL_DONE"
echo "--- matrix ---" >> "$MASTER"
for t in peg gear nut; do
  for ck in 00141 00211 00426 00638 00783; do
    sr=$(grep -oE "\"sr\": [0-9.]+" $LOG_DIR/srsa_zs_${t}_${ck}.log 2>/dev/null | head -1)
    echo "$t $ck $sr" >> "$MASTER"
  done
done
