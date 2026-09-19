#!/usr/bin/env bash
# Make the BC-only vs +DAgger contrast a proper 3-vs-3 on the FULL noise axis.
#
# Tonight's seed-0 pair showed DAgger is worth ~nothing inside the training range
# (+0.8 @5mm) but +4.7 / +8.9 points at 7.5 / 10mm. That reframes the claim from
# "DAgger adds success" to "DAgger buys out-of-distribution robustness" -- a
# claim worth more than one seed, since it now goes in the paper.
#
# Two phases:
#   A  gate2_seed1 / gate2_seed2 (BC+DAgger) at 7.5 / 10mm -- checkpoints already
#      exist, only seed0 had breaking-point cells, so this runs immediately.
#   B  tb_bconly_seed1 / seed2 at 7.5 / 10mm -- waits for those trainings and
#      their own 0-5mm evals to finish first.
#
# Frozen protocol throughout (n=256, protocol_seed 42, dyn_rand on). One cell at
# a time on a dedicated GPU: an Isaac eval sharing a GPU with another heavy CUDA
# process dies with create_articulation_view on None (hit exactly that tonight).
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
MASTER=$LOGD/dagger_3seed_master.log
GPU=${1:-1}
GYM=Isaac-Forge-PegInsert-TBCamera-v0
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_mem(){ while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
  log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 240; done; }

run_cell(){ # <ckpt_dir> <tag_prefix> <noise>
  local out=$1 tag=$2 nm=$3
  local lg=$LOGD/${tag}_n${nm}.log
  grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP $tag n=$nm"; return; }
  [ -f "$out/best.pt" ] || { log "SKIP $tag: no best.pt"; return; }
  wait_mem
  log "$tag n=${nm}mm START"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_frozen_student.py" --headless \
    --task "$GYM" --checkpoint "$out/best.pt" --norm_stats "$out/norm_stats.npz" \
    --episodes 256 --num_envs 32 --protocol_seed 42 \
    --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "${tag}_n${nm}" >> "$lg" 2>&1
  log "$tag n=${nm} rc=$?"
}

# ---- phase A: BC+DAgger seeds 1,2 in the extrapolation region ---------------
for s in 1 2; do
  for nm in 7.5 10; do
    run_cell ~/forge_ts/student_ckpts/peg/gate2_seed${s} "dag3_gate2_s${s}" "$nm"
  done
done
log "PHASE_A_DONE (BC+DAgger seeds 1,2 extrapolation)"

# ---- phase B: BC-only seeds 1,2, after their own chains finish --------------
log "waiting for the BC-only seed chains"
while true; do
  a=$(grep -c "TB_BCONLY_SEED_DONE" $LOGD/tb_bconly_s1_master.log 2>/dev/null)
  b=$(grep -c "TB_BCONLY_SEED_DONE" $LOGD/tb_bconly_s2_master.log 2>/dev/null)
  [ "${a:-0}" -ge 1 ] && [ "${b:-0}" -ge 1 ] && break
  log "waiting: s1_done=${a:-0} s2_done=${b:-0}"
  sleep 600
done

for s in 1 2; do
  for nm in 7.5 10; do
    run_cell ~/forge_ts/student_ckpts/peg/tb_bconly_seed${s} "dag3_bconly_s${s}" "$nm"
  done
done
log "DAGGER_3SEED_DONE"
