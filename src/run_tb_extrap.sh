#!/usr/bin/env bash
# tb arm extrapolation cells, on GPU1 (which is free) instead of GPU2 (now busy
# with the BC-only seed2 training -- an Isaac eval sharing a GPU with another
# heavy CUDA process dies with create_articulation_view on None, which is exactly
# how the first attempt failed).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
OUT=~/forge_ts/student_ckpts/peg/tb_bconly_seed0
MASTER=$LOGD/p0_extrap_master.log
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }
for nm in 7.5 10; do
  lg=$LOGD/p0_tb_eval_peg_n${nm}.log
  grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP tb n=$nm"; continue; }
  log "tb eval n=${nm}mm START on gpu 1 (retry after gpu conflict)"
  CUDA_VISIBLE_DEVICES=1 $PY "$SRC/eval_frozen_student.py" --headless \
    --task Isaac-Forge-PegInsert-TBCamera-v0 --checkpoint "$OUT/best.pt" \
    --norm_stats "$OUT/norm_stats.npz" \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag p0_tb_peg_n${nm} >> "$lg" 2>&1
  log "tb eval n=${nm} rc=$?"
done
log "P0_EXTRAP_DONE (tb retried on gpu1)"
