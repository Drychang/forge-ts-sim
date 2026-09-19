#!/usr/bin/env bash
# ARCH-INSERT frozen-protocol eval: per task x noise {0,1,2.5,5}mm, n=256,
# dyn_rand on. Waits for each task's teacher ckpt (nut may still be training).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/arch
LOGD=~/forge_ts/logs
MASTER=$LOGD/arch_eval_master.log
GPU=${1:-1}
cd "$SRC"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

for t in peg gear nut; do
  case $t in
    peg)  gid=Isaac-Forge-PegInsert-ARCH-v0 ;;
    gear) gid=Isaac-Forge-GearMesh-ARCH-v0 ;;
    nut)  gid=Isaac-Forge-NutThread-ARCH-v0 ;;
  esac
  log "waiting for $t final ckpt"
  while ! ls "$SRC/logs/rl_games/Forge/arch_${t}_s0/nn"/last_Forge_ep_200*.pth >/dev/null 2>&1; do sleep 180; done
  ck=$(ls "$SRC/logs/rl_games/Forge/arch_${t}_s0/nn"/last_Forge_ep_200_rew_[0-9]*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && ck=$(ls "$SRC/logs/rl_games/Forge/arch_${t}_s0/nn"/last_Forge_ep_200*.pth 2>/dev/null | head -1)
  log "$t ckpt: $(basename "$ck")"
  for nm in 0 1 2.5 5; do
    lg=$LOGD/arch_eval_${t}_n${nm}.log
    if grep -q "EVAL_SUMMARY" "$lg" 2>/dev/null; then log "SKIP $t n=$nm"; continue; fi
    log "eval $t noise=$nm START"
    CUDA_VISIBLE_DEVICES=$GPU $PY eval_arch.py --headless --task "$gid" \
      --checkpoint "$ck" --num_envs 32 --episodes 256 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "arch_${t}_n${nm}" \
      >> "$lg" 2>&1
    log "eval $t noise=$nm rc=$?"
  done
done
log "ARCH_EVAL_ALL_DONE"
echo "--- ARCH-INSERT matrix (SR, dyn_rand on, n=256) ---" >> "$MASTER"
for t in peg gear nut; do for nm in 0 1 2.5 5; do
  sr=$(grep -oE "\"sr\": [0-9.]+" $LOGD/arch_eval_${t}_n${nm}.log 2>/dev/null | head -1)
  echo "$t n=$nm $sr" >> "$MASTER"
done; done
