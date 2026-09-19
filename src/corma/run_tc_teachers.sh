#!/usr/bin/env bash
# CoRMA/C3 (option A): train the 27-dim minimal-privileged T-C teachers.
# 3 tasks x 200 epochs (== T-A/T-B budget) x seed 0. GPU1: peg->gear; GPU2: nut.
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/corma
LOGD=~/forge_ts/logs
MASTER=$LOGD/tc_teachers_master.log
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

declare -A GID=( [peg]=Isaac-Forge-PegInsert-TC-v0 [gear]=Isaac-Forge-GearMesh-TC-v0 [nut]=Isaac-Forge-NutThread-TC-v0 )

train_one(){  # gpu task
  local gpu=$1 t=$2 exp="tc_${t}_s0"
  local nn="$SRC/logs/rl_games/Forge/$exp/nn"
  if ls "$nn"/*.pth >/dev/null 2>&1; then log "SKIP $t (ckpt exists)"; return 0; fi
  log "TC train $t on GPU$gpu START"
  CUDA_VISIBLE_DEVICES=$gpu $PY train_tc.py --task "${GID[$t]}" --headless \
    --num_envs 128 --seed 0 --max_iterations 200 \
    agent.params.config.full_experiment_name="$exp" \
    >> "$LOGD/tc_train_${t}.log" 2>&1
  log "TC train $t rc=$?"
}

( train_one 1 peg; train_one 1 gear; log "CHAIN_GPU1_DONE" ) &
( train_one 2 nut; log "CHAIN_GPU2_DONE" ) &
wait
log "TC_TEACHERS_ALL_DONE"
echo "--- final rewards ---" >> "$MASTER"
for t in peg gear nut; do
  r=$(grep -oE "saving next best rewards:  \[[-0-9.]+\]" "$LOGD/tc_train_${t}.log" 2>/dev/null | tail -1)
  echo "$t $r" >> "$MASTER"
done
