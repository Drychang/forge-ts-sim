#!/usr/bin/env bash
# SRSA Stage 3a: short-budget SIL fine-tune screening.
# 3 tasks x 5 source ckpts x 30 epochs (sparse+SIL, seed 0, 1mm training noise
# = env default), each followed by a 64-ep frozen-protocol eval at 1mm.
# Winner per task feeds Stage 3b (full 200-epoch run).
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/srsa
LOG_DIR=~/forge_ts/logs
MASTER=$LOG_DIR/srsa_silscreen_master.log
CKPT_DIR=/extra_home3/user/SRSA/checkpoints
SHADOW=/extra_home3/user/srsa_pylibs:/extra_home3/user/SRSA/source/SRSA
cd "$SRC"   # rl_games artifacts land under CWD/logs/rl_games/Assembly/<exp>/nn/
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

for pair in "peg Isaac-SRSA-Forge-PegInsert-v0" "gear Isaac-SRSA-Forge-GearMesh-v0" "nut Isaac-SRSA-Forge-NutThread-v0"; do
  set -- $pair
  t=$1; gymid=$2
  for ck in 00141 00211 00426 00638 00783; do
    exp="silscr_${t}_${ck}"
    evlog=$LOG_DIR/srsa_silscr_eval_${t}_${ck}.log
    if grep -q "EVAL_SUMMARY" "$evlog" 2>/dev/null; then
      log "SKIP $t $ck (eval done)"
      continue
    fi
    log "3a train $t from=$ck START"
    CUDA_VISIBLE_DEVICES=1 SRSA_SPARSE_REWARD=1 PYTHONPATH=$SHADOW \
      $PY "$SRC/train_srsa_forge.py" --headless \
      --task "$gymid" --num_envs 128 --seed 0 --max_iterations 30 \
      --checkpoint "$CKPT_DIR/$ck.pth" --load_mode actor \
      agent.params.config.full_experiment_name="$exp" \
      >> $LOG_DIR/srsa_silscr_train_${t}_${ck}.log 2>&1
    rc=$?
    log "3a train $t from=$ck rc=$rc"
    p=$(ls -t "$SRC/logs/rl_games/Assembly/$exp/nn/"*.pth 2>/dev/null | head -1)
    if [ -z "$p" ]; then
      log "3a $t $ck NO_CKPT_PRODUCED -- skipping eval"
      continue
    fi
    log "3a eval $t from=$ck ckpt=$(basename $p) START"
    CUDA_VISIBLE_DEVICES=1 $PY "$SRC/screen_zeroshot.py" --headless \
      --task "$gymid" --checkpoint "$p" \
      --num_envs 32 --episodes 64 --protocol_seed 42 \
      --fixed_pos_noise_mm 1 --dyn_rand on --tag "silscr_${ck}" \
      >> "$evlog" 2>&1
    log "3a eval $t from=$ck rc=$?"
  done
done
log "SRSA_SILSCREEN_ALL_DONE"
echo "--- 3a matrix (SR after 30-epoch SIL fine-tune, 1mm) ---" >> "$MASTER"
for t in peg gear nut; do
  for ck in 00141 00211 00426 00638 00783; do
    sr=$(grep -oE "\"sr\": [0-9.]+" $LOG_DIR/srsa_silscr_eval_${t}_${ck}.log 2>/dev/null | head -1)
    echo "$t $ck $sr" >> "$MASTER"
  done
done
