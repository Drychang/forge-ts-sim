#!/usr/bin/env bash
# SRSA Stage 3b v2: rerun all 3 tasks with the SIL buffer fix (max_trajs
# 1000->256, self_imitation.py patched 2026-07-21) for a UNIFORM, consistent
# recipe across tasks. Root cause of the nut OOM at epoch 33/200: default
# buffer held 1000 full trajectories sized for AutoMate's 5s episodes;
# FORGE's nut episodes are 30s/449 steps -> the crashing process itself held
# 23.31/23.68 GiB (confirmed self-inflicted, not another user).
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/srsa
LOG_DIR=~/forge_ts/logs
MASTER=$LOG_DIR/srsa_3b_v2_master.log
CKPT_DIR=/extra_home3/user/SRSA/checkpoints
SHADOW=/extra_home3/user/srsa_pylibs:/extra_home3/user/SRSA/source/SRSA
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

train_one(){  # gpu task gymid source
  local gpu=$1 t=$2 gymid=$3 src=$4
  local exp="3b_${t}_v2_seed0"
  local nn="$SRC/logs/rl_games/Assembly/$exp/nn"
  if ls "$nn"/last_*.pth >/dev/null 2>&1; then log "SKIP train $t (v2 ckpt exists)"; return 0; fi
  log "3b-v2 train $t from=$src on GPU$gpu START"
  CUDA_VISIBLE_DEVICES=$gpu SRSA_SPARSE_REWARD=0 PYTHONPATH=$SHADOW \
    $PY "$SRC/train_srsa_forge.py" --headless \
    --task "$gymid" --num_envs 128 --seed 0 --max_iterations 200 \
    --checkpoint "$CKPT_DIR/$src.pth" --load_mode actor \
    agent.params.config.full_experiment_name="$exp" \
    >> "$LOG_DIR/srsa_3b_v2_train_${t}.log" 2>&1
  log "3b-v2 train $t rc=$?"
}

eval_one(){  # gpu task gymid noise
  local gpu=$1 t=$2 gymid=$3 noise=$4
  local nn="$SRC/logs/rl_games/Assembly/3b_${t}_v2_seed0/nn"
  local p; p=$(ls -t "$nn"/last_*.pth 2>/dev/null | head -1)
  local lg=$LOG_DIR/srsa_3b_v2_eval_${t}_n${noise}.log
  if [ -z "$p" ]; then log "3b-v2 eval $t noise=$noise SKIP (no ckpt)"; return 0; fi
  if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then log "SKIP eval $t noise=$noise (done)"; return 0; fi
  log "3b-v2 eval $t noise=$noise START"
  CUDA_VISIBLE_DEVICES=$gpu $PY "$SRC/screen_zeroshot.py" --headless \
    --task "$gymid" --checkpoint "$p" \
    --num_envs 32 --episodes 256 --protocol_seed 42 \
    --fixed_pos_noise_mm "$noise" --dyn_rand on --tag "3b_v2_seed0" \
    >> "$lg" 2>&1
  log "3b-v2 eval $t noise=$noise rc=$?"
}

wait_gpu_free(){  # gpu
  local gpu=$1
  while pgrep -f "train_srsa_forge.*CUDA_VISIBLE_DEVICES=$gpu" >/dev/null 2>&1; do sleep 60; done
  while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    [ "$free_mib" -ge 18000 ] && return
    log "waiting for GPU$gpu memory (free=${free_mib}MiB)"
    sleep 60
  done
}

# GPU1: nut (urgent -- crashed, needs immediate fixed rerun) then peg (rerun
# for cross-task buffer-size consistency; original completed but on old buffer).
(
  train_one 1 nut Isaac-SRSA-Forge-NutThread-v0 00141
  for n in 0 2.5 5; do eval_one 1 nut Isaac-SRSA-Forge-NutThread-v0 "$n"; done
  train_one 1 peg Isaac-SRSA-Forge-PegInsert-v0 00638
  for n in 0 2.5 5; do eval_one 1 peg Isaac-SRSA-Forge-PegInsert-v0 "$n"; done
  log "CHAIN_A_V2_DONE"
) &

# GPU2: wait for the ORIGINAL (old-buffer) gear run to reach a terminal state
# (finish or crash) -- do not touch a live process -- then rerun gear on the
# fixed buffer for consistency.
(
  log "waiting for original gear run (GPU2, old buffer) to finish or crash"
  while pgrep -f "3b_gear_seed0" >/dev/null 2>&1; do sleep 120; done
  log "original gear run reached terminal state; starting v2 rerun"
  train_one 2 gear Isaac-SRSA-Forge-GearMesh-v0 00638
  for n in 0 2.5 5; do eval_one 2 gear Isaac-SRSA-Forge-GearMesh-v0 "$n"; done
  log "CHAIN_B_V2_DONE"
) &

wait
log "SRSA_3B_V2_ALL_DONE"
echo "--- 3b-v2 matrix (seed0, full 200-epoch shaped+SIL, buffer=256) ---" >> "$MASTER"
for t in peg gear nut; do
  for n in 0 2.5 5; do
    sr=$(grep -oE "\"sr\": [0-9.]+" $LOG_DIR/srsa_3b_v2_eval_${t}_n${n}.log 2>/dev/null | head -1)
    echo "$t noise=$n $sr" >> "$MASTER"
  done
done
