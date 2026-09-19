#!/usr/bin/env bash
# Full C3-TC frozen-protocol eval (n=256). Per task (waits for its teacher ckpt):
#   adapter mode x {0,1,2.5,5}mm  (the deployable RMA baseline -> paper table)
#   true + zero  x 5mm            (oracle upper / no-info lower reference bounds)
# GPU1 (clean, NO orphan -- never GPU0 with Isaac PhysX). dyn_rand on.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/corma
LOGD=~/forge_ts/logs
MASTER=$LOGD/tc_c3_eval_master.log
NN=$SRC/logs/rl_games/Forge
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

ev(){  # task gymid ckpt noise mode
  local t=$1 gid=$2 ck=$3 nm=$4 mode=$5
  local lg=$LOGD/tc_c3_${t}_${mode}_n${nm}.log
  if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then log "SKIP $t $mode n=$nm"; return 0; fi
  log "eval $t mode=$mode noise=$nm START"
  CUDA_VISIBLE_DEVICES=1 $PY eval_corma_c3_tc.py --headless \
    --task "$gid" --tb_checkpoint "$ck" \
    --adapter ~/forge_ts/adapter_ckpts/$t/adapter_best.pt \
    --adapter_norm ~/forge_ts/adapter_ckpts/$t/adapter_norm.npz \
    --num_envs 128 --episodes 256 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --inject_mode "$mode" \
    --tag c3_${t}_${mode}_n${nm} >> "$lg" 2>&1
  log "eval $t mode=$mode noise=$nm rc=$?"
}

for pair in "peg Isaac-Forge-PegInsert-TC-v0" "gear Isaac-Forge-GearMesh-TC-v0" "nut Isaac-Forge-NutThread-TC-v0"; do
  set -- $pair; t=$1; gid=$2
  log "waiting for $t teacher ckpt"
  while ! ls "$NN/tc_${t}_s0/nn"/last_Forge_ep_200*.pth >/dev/null 2>&1; do sleep 120; done
  ck=$(ls "$NN/tc_${t}_s0/nn"/last_Forge_ep_200_rew_[0-9]*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && ck=$(ls "$NN/tc_${t}_s0/nn"/last_Forge_ep_200*.pth 2>/dev/null | head -1)
  log "$t teacher ready: $(basename "$ck")"
  for nm in 0 1 2.5 5; do ev "$t" "$gid" "$ck" "$nm" adapter; done   # baseline sweep
  ev "$t" "$gid" "$ck" 5 true; ev "$t" "$gid" "$ck" 5 zero          # reference bounds @5mm
done
log "TC_C3_EVAL_ALL_DONE"
echo "--- C3-TC adapter matrix (SR, dyn_rand on) ---" >> "$MASTER"
for t in peg gear nut; do for nm in 0 1 2.5 5; do
  sr=$(grep -oE "\"sr\": [0-9.]+" $LOGD/tc_c3_${t}_adapter_n${nm}.log 2>/dev/null | head -1)
  echo "$t adapter n=$nm $sr" >> "$MASTER"
done; done
for t in peg gear nut; do for m in true zero; do
  sr=$(grep -oE "\"sr\": [0-9.]+" $LOGD/tc_c3_${t}_${m}_n5.log 2>/dev/null | head -1)
  echo "$t $m n=5 $sr" >> "$MASTER"
done; done
