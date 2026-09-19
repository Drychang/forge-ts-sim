#!/usr/bin/env bash
# CoRMA/C3 Phase 2+3 pipeline: train noise adapters (seed0) then frozen-protocol
# eval with adapter-injected noise, all 3 tasks x {0,1,2.5,5}mm (dyn_rand on).
# Waits for Phase-1 collection to finish first. Idempotent (skips done work).
set -u
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/corma
LOGD=~/forge_ts/logs
DATA=~/forge_ts/adapter_data
CKD=~/forge_ts/adapter_ckpts
FORGE=/extra_home3/user/force_vla_research/IsaacLab/logs/rl_games/Forge
MASTER=$LOGD/corma_c3_master.log
mkdir -p "$CKD"
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

declare -A TBCK=( [peg]="$FORGE/tb_peg_s0/nn/Forge.pth" [gear]="$FORGE/tb_gear_s0/nn/Forge.pth" [nut]="$FORGE/tb_nut_s0/nn/Forge.pth" )
declare -A GID=( [peg]=Isaac-Forge-PegInsert-TB-v0 [gear]=Isaac-Forge-GearMesh-TB-v0 [nut]=Isaac-Forge-NutThread-TB-v0 )

# wait for collection
log "waiting for CORMA_COLLECT_ALL_DONE"
while ! grep -q "CORMA_COLLECT_ALL_DONE" "$LOGD/corma_collect_master.log" 2>/dev/null; do sleep 120; done
log "collection done; starting Phase 2 (adapter training)"

# ---- Phase 2: train adapters (GPU1), state-only fast ----
for t in peg gear nut; do
  if [ -f "$CKD/$t/adapter_best.pt" ]; then log "SKIP train $t (adapter exists)"; continue; fi
  log "P2 train adapter $t START"
  CUDA_VISIBLE_DEVICES=1 $PY "$SRC/train_adapter.py" --task "$t" \
    --data_dir "$DATA" --out_dir "$CKD/$t" --epochs 40 --seed 0 \
    >> "$LOGD/corma_c3_train_${t}.log" 2>&1
  log "P2 train adapter $t rc=$?"
done

# ---- Phase 3: frozen-protocol eval with adapter injection (GPU1) ----
for t in peg gear nut; do
  for nm in 0 1 2.5 5; do
    ev="$LOGD/corma_c3_eval_${t}_n${nm}.log"
    if grep -q "EVAL_SUMMARY" "$ev" 2>/dev/null; then log "SKIP eval $t n=$nm (done)"; continue; fi
    if [ ! -f "$CKD/$t/adapter_best.pt" ]; then log "eval $t n=$nm SKIP (no adapter)"; continue; fi
    log "P3 eval $t noise=$nm START"
    CUDA_VISIBLE_DEVICES=1 $PY "$SRC/eval_corma_c3.py" --headless \
      --task "${GID[$t]}" --tb_checkpoint "${TBCK[$t]}" \
      --adapter "$CKD/$t/adapter_best.pt" --adapter_norm "$CKD/$t/adapter_norm.npz" \
      --num_envs 128 --episodes 256 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "c3_${t}_n${nm}" \
      >> "$ev" 2>&1
    log "P3 eval $t noise=$nm rc=$?"
  done
done

log "CORMA_C3_ALL_DONE"
echo "--- C3 matrix (seed0, adapter-injected noise, dyn_rand on) ---" >> "$MASTER"
for t in peg gear nut; do
  for nm in 0 1 2.5 5; do
    sr=$(grep -oE "\"sr\": [0-9.]+" "$LOGD/corma_c3_eval_${t}_n${nm}.log" 2>/dev/null | head -1)
    echo "$t noise=$nm $sr" >> "$MASTER"
  done
done
