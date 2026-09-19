#!/usr/bin/env bash
# SRSA Stage 3b: full-budget (200 epoch, matches T-A) shaped+SIL commit runs.
# Screening (3a sparse, 3a-B shaped) gave no clean winner -- both rungs mostly
# 0% at 30 epochs (insufficient budget to differentiate, not a reward-choice
# issue). Winners picked by frequency of ANY non-zero signal across all three
# 64-ep screening probes (phase-2 zero-shot, 3a sparse+SIL, 3a-B shaped+SIL):
#   peg  <- 00638 (only source with any signal at all: 3a sparse, 1/3 probes)
#   gear <- 00638 (zero-shot + 3a sparse: 2/3 probes; 00426 only 1/3)
#   nut  <- 00141 (3a sparse + 3a-B shaped: 2/3 probes; 00211 only 1/3)
# Reward = shaped (FORGE standard recipe, used everywhere else in this
# project; sparse totally failed both screening rungs).
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/srsa
LOG_DIR=~/forge_ts/logs
MASTER=$LOG_DIR/srsa_3b_master.log
CKPT_DIR=/extra_home3/user/SRSA/checkpoints
SHADOW=/extra_home3/user/srsa_pylibs:/extra_home3/user/SRSA/source/SRSA
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

train_one(){  # gpu task gymid source
  local gpu=$1 t=$2 gymid=$3 src=$4
  local exp="3b_${t}_seed0"
  local nn="$SRC/logs/rl_games/Assembly/$exp/nn"
  if ls "$nn"/last_*.pth >/dev/null 2>&1; then
    log "SKIP train $t (3b ckpt exists)"
    return 0
  fi
  log "3b train $t from=$src on GPU$gpu START"
  CUDA_VISIBLE_DEVICES=$gpu SRSA_SPARSE_REWARD=0 PYTHONPATH=$SHADOW \
    $PY "$SRC/train_srsa_forge.py" --headless \
    --task "$gymid" --num_envs 128 --seed 0 --max_iterations 200 \
    --checkpoint "$CKPT_DIR/$src.pth" --load_mode actor \
    agent.params.config.full_experiment_name="$exp" \
    >> "$LOG_DIR/srsa_3b_train_${t}.log" 2>&1
  log "3b train $t rc=$?"
}

eval_one(){  # gpu task gymid noise
  local gpu=$1 t=$2 gymid=$3 noise=$4
  local nn="$SRC/logs/rl_games/Assembly/3b_${t}_seed0/nn"
  local p; p=$(ls -t "$nn"/last_*.pth 2>/dev/null | head -1)
  local lg=$LOG_DIR/srsa_3b_eval_${t}_n${noise}.log
  if [ -z "$p" ]; then log "3b eval $t noise=$noise SKIP (no ckpt)"; return 0; fi
  if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then log "SKIP eval $t noise=$noise (done)"; return 0; fi
  log "3b eval $t noise=$noise START"
  CUDA_VISIBLE_DEVICES=$gpu $PY "$SRC/screen_zeroshot.py" --headless \
    --task "$gymid" --checkpoint "$p" \
    --num_envs 32 --episodes 256 --protocol_seed 42 \
    --fixed_pos_noise_mm "$noise" --dyn_rand on --tag "3b_seed0" \
    >> "$lg" 2>&1
  log "3b eval $t noise=$noise rc=$?"
}

# Chain A on GPU1: peg then nut (train+eval each)
(
  train_one 1 peg Isaac-SRSA-Forge-PegInsert-v0 00638
  for n in 0 2.5 5; do eval_one 1 peg Isaac-SRSA-Forge-PegInsert-v0 "$n"; done
  train_one 1 nut Isaac-SRSA-Forge-NutThread-v0 00141
  for n in 0 2.5 5; do eval_one 1 nut Isaac-SRSA-Forge-NutThread-v0 "$n"; done
  log "CHAIN_A_DONE"
) &

# Chain B on GPU2: gear (train+eval)
(
  train_one 2 gear Isaac-SRSA-Forge-GearMesh-v0 00638
  for n in 0 2.5 5; do eval_one 2 gear Isaac-SRSA-Forge-GearMesh-v0 "$n"; done
  log "CHAIN_B_DONE"
) &

wait
log "SRSA_3B_ALL_DONE"
echo "--- 3b matrix (seed0, full 200-epoch shaped+SIL) ---" >> "$MASTER"
for t in peg gear nut; do
  for n in 0 2.5 5; do
    sr=$(grep -oE "\"sr\": [0-9.]+" $LOG_DIR/srsa_3b_eval_${t}_n${n}.log 2>/dev/null | head -1)
    echo "$t noise=$n $sr" >> "$MASTER"
  done
done
