#!/usr/bin/env bash
# SRSA Stage 3c: seeds 1 and 2 (seed0 = 3b-v2, already done), same recipe
# (source ckpts, shaped+SIL, 200 epochs, buffer=256 fix) -- only RL seed
# changes, per project standing rule: any claim needs >=3 seeds.
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/srsa
LOG_DIR=~/forge_ts/logs
MASTER=$LOG_DIR/srsa_3c_master.log
CKPT_DIR=/extra_home3/user/SRSA/checkpoints
SHADOW=/extra_home3/user/srsa_pylibs:/extra_home3/user/SRSA/source/SRSA
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

train_one(){  # gpu task gymid source seed
  local gpu=$1 t=$2 gymid=$3 src=$4 seed=$5
  local exp="3c_${t}_seed${seed}"
  local nn="$SRC/logs/rl_games/Assembly/$exp/nn"
  if ls "$nn"/last_*.pth >/dev/null 2>&1; then log "SKIP train $t seed$seed (ckpt exists)"; return 0; fi
  log "3c train $t seed=$seed from=$src on GPU$gpu START"
  CUDA_VISIBLE_DEVICES=$gpu SRSA_SPARSE_REWARD=0 PYTHONPATH=$SHADOW \
    $PY "$SRC/train_srsa_forge.py" --headless \
    --task "$gymid" --num_envs 128 --seed "$seed" --max_iterations 200 \
    --checkpoint "$CKPT_DIR/$src.pth" --load_mode actor \
    agent.params.config.full_experiment_name="$exp" \
    >> "$LOG_DIR/srsa_3c_train_${t}_seed${seed}.log" 2>&1
  log "3c train $t seed=$seed rc=$?"
}

eval_one(){  # gpu task gymid noise seed
  local gpu=$1 t=$2 gymid=$3 noise=$4 seed=$5
  local nn="$SRC/logs/rl_games/Assembly/3c_${t}_seed${seed}/nn"
  local p; p=$(ls -t "$nn"/last_*.pth 2>/dev/null | head -1)
  local lg=$LOG_DIR/srsa_3c_eval_${t}_seed${seed}_n${noise}.log
  if [ -z "$p" ]; then log "3c eval $t seed$seed noise=$noise SKIP (no ckpt)"; return 0; fi
  if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then log "SKIP eval $t seed$seed noise=$noise (done)"; return 0; fi
  log "3c eval $t seed=$seed noise=$noise START"
  CUDA_VISIBLE_DEVICES=$gpu $PY "$SRC/screen_zeroshot.py" --headless \
    --task "$gymid" --checkpoint "$p" \
    --num_envs 32 --episodes 256 --protocol_seed 42 \
    --fixed_pos_noise_mm "$noise" --dyn_rand on --tag "3c_seed${seed}" \
    >> "$lg" 2>&1
  log "3c eval $t seed=$seed noise=$noise rc=$?"
}

run_seed(){  # seed
  local seed=$1
  # GPU1: nut then peg; GPU2: gear (same split that gave best wall-clock balance in 3b-v2)
  (
    train_one 1 nut Isaac-SRSA-Forge-NutThread-v0 00141 "$seed"
    for n in 0 2.5 5; do eval_one 1 nut Isaac-SRSA-Forge-NutThread-v0 "$n" "$seed"; done
    train_one 1 peg Isaac-SRSA-Forge-PegInsert-v0 00638 "$seed"
    for n in 0 2.5 5; do eval_one 1 peg Isaac-SRSA-Forge-PegInsert-v0 "$n" "$seed"; done
  ) &
  (
    train_one 2 gear Isaac-SRSA-Forge-GearMesh-v0 00638 "$seed"
    for n in 0 2.5 5; do eval_one 2 gear Isaac-SRSA-Forge-GearMesh-v0 "$n" "$seed"; done
  ) &
  wait
  log "SEED_${seed}_DONE"
}

run_seed 1
run_seed 2
log "SRSA_3C_ALL_DONE"
echo "--- 3c matrix (seeds 1,2) ---" >> "$MASTER"
for seed in 1 2; do
  for t in peg gear nut; do
    for n in 0 2.5 5; do
      sr=$(grep -oE "\"sr\": [0-9.]+" $LOG_DIR/srsa_3c_eval_${t}_seed${seed}_n${n}.log 2>/dev/null | head -1)
      echo "seed=$seed $t noise=$n $sr" >> "$MASTER"
    done
  done
done
