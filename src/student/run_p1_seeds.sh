#!/usr/bin/env bash
# P1 architecture ablation: add seeds 1 and 2 for one architecture.
#   usage: run_p1_seeds.sh <concat|act> <gpu>
#
# Why: the "three fusion architectures are statistically indistinguishable"
# claim -- which is load-bearing for the contribution framing ("performance comes
# from the modality combination, not the fusion mechanism") -- currently rests on
# ONE seed per architecture. Tonight's BC-only run was a reminder of how badly a
# single seed can mislead when variance is real. The reference FMT arm already
# has 3 seeds (gate2_seed0/1/2, sd = 0.9 at 5mm), so two more seeds per
# competitor makes it a proper 3-vs-3-vs-3.
#
# Identical recipe to the seed-0 P1 runs: full BC+DAgger dataset, same trainer
# (train_student_arch.py), same early stopping, same eval protocol.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/student
LOGD=~/forge_ts/logs
DATA=/media/data/forge_ts_data
ARCH=${1:?arch: concat|act}
GPU=${2:?gpu}
GYM=Isaac-Forge-PegInsert-TBCamera-v0
MASTER=$LOGD/p1_seeds_${ARCH}_master.log
log(){ echo "=== [$(date -Is)] $*" >> "$MASTER"; }

wait_mem(){ while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 20 ]; do
  log "wait: available=$(free -g | awk '/^Mem:/{print $7}')G"; sleep 240; done; }

for seed in 1 2; do
  OUT=~/forge_ts/student_ckpts/peg/p1_${ARCH}_s${seed}
  if [ -f "$OUT/best.pt" ]; then
    log "SKIP train seed=$seed (best.pt exists)"
  else
    wait_mem
    mkdir -p "$OUT"
    log "train arch=$ARCH seed=$seed gpu=$GPU"
    $PY -u "$SRC/run_seed_earlystop.py" \
      --data_root "$DATA" --task peg --out_dir "$OUT" --gpu "$GPU" \
      --patience 6 --max_epochs 30 --seed "$seed" --num_workers 4 \
      --trainer "$SRC/train_student_arch.py" --extra_args "--arch $ARCH" \
      > "$LOGD/p1_${ARCH}_s${seed}_train.log" 2>&1
    log "train arch=$ARCH seed=$seed rc=$? best=$([ -f $OUT/best.pt ] && echo yes || echo NO)"
  fi
  [ -f "$OUT/best.pt" ] || { log "no best.pt for seed=$seed, skipping its eval"; continue; }

  for nm in 0 1 2.5 5; do
    lg=$LOGD/p1_${ARCH}_s${seed}_eval_n${nm}.log
    grep -q EVAL_SUMMARY "$lg" 2>/dev/null && { log "SKIP eval seed=$seed n=$nm"; continue; }
    wait_mem
    log "eval arch=$ARCH seed=$seed n=${nm}mm"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$SRC/eval_arch_student.py" --headless \
      --task "$GYM" --checkpoint "$OUT/best.pt" --norm_stats "$OUT/norm_stats.npz" \
      --episodes 256 --num_envs 32 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on \
      --tag p1_${ARCH}_s${seed}_n${nm} >> "$lg" 2>&1
    log "eval seed=$seed n=${nm} rc=$?"
  done
done
log "P1_SEEDS_DONE arch=$ARCH"
