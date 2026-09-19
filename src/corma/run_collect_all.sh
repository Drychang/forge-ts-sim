#!/usr/bin/env bash
# CoRMA/C3 Phase 1 full data collection: T-B rollouts across the noise spectrum
# for all 3 tasks, to train the noise-regression adapter. State-only (no
# cameras) so fast. 256 eps x {0,1,2.5,5}mm x 3 tasks = 3072 episodes.
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src
LOG=~/forge_ts/logs/corma_collect_master.log
FORGE=/extra_home3/user/force_vla_research/IsaacLab/logs/rl_games/Forge
OUT=~/forge_ts/adapter_data
mkdir -p "$OUT"
log(){ echo "=== [$(date -Is)] $*" >> "$LOG"; }

declare -A CK=( [peg]="$FORGE/tb_peg_s0/nn/Forge.pth" [gear]="$FORGE/tb_gear_s0/nn/Forge.pth" [nut]="$FORGE/tb_nut_s0/nn/Forge.pth" )
declare -A GID=( [peg]=Isaac-Forge-PegInsert-TB-v0 [gear]=Isaac-Forge-GearMesh-TB-v0 [nut]=Isaac-Forge-NutThread-TB-v0 )

cd "$SRC"
for t in peg gear nut; do
  for nm in 0 1 2.5 5; do
    out="$OUT/${t}_n${nm}.npz"
    if [ -f "$out" ]; then log "SKIP $t n=$nm (exists)"; continue; fi
    log "collect $t noise=$nm START"
    CUDA_VISIBLE_DEVICES=1 $PY corma/collect_adapter_data.py --headless \
      --task "${GID[$t]}" --checkpoint "${CK[$t]}" \
      --num_envs 128 --episodes 256 --fixed_pos_noise_mm "$nm" --seed 0 \
      --out "$out" >> "$HOME/forge_ts/logs/corma_collect_${t}_n${nm}.log" 2>&1
    log "collect $t noise=$nm rc=$?"
  done
done
log "CORMA_COLLECT_ALL_DONE"
