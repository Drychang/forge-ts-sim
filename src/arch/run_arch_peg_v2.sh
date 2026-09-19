#!/usr/bin/env bash
# Fairness retest: retrain ARCH peg WITH EE velocity in obs (ARCH_EE_VEL=1,
# 33-dim) once GPU2 frees (after nut training). Pose-only peg gave 0% SR with
# a timid non-contacting policy -- likely my over-narrow reading of the paper's
# "end-effector (EE)" obs, since ARCH commands velocities.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/arch
LOGD=~/forge_ts/logs
MASTER=$LOGD/arch_v2_master.log
cd "$SRC"
echo "=== [$(date -Is)] waiting for GPU2 free (nut training to finish) ===" >> "$MASTER"
while pgrep -f "arch_nut_s0" > /dev/null 2>&1; do sleep 180; done
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)
  [ "$free" -ge 12000 ] && break
  echo "=== [$(date -Is)] waiting GPU2 memory (free=${free}MiB) ===" >> "$MASTER"; sleep 180
done
echo "=== [$(date -Is)] ARCH peg v2 (EE_VEL) train START on GPU2 ===" >> "$MASTER"
CUDA_VISIBLE_DEVICES=2 ARCH_EE_VEL=1 $PY train_arch.py --task Isaac-Forge-PegInsert-ARCH-v0 --headless \
  --num_envs 128 --seed 0 --max_iterations 200 \
  agent.params.config.full_experiment_name=arch_peg_v2_s0 >> "$LOGD/arch_train_peg_v2.log" 2>&1
echo "=== [$(date -Is)] ARCH peg v2 train rc=$? ===" >> "$MASTER"
ck=$(ls "$SRC/logs/rl_games/Forge/arch_peg_v2_s0/nn"/last_Forge_ep_200*.pth 2>/dev/null | head -1)
if [ -n "$ck" ]; then
  for nm in 0 5; do
    echo "=== [$(date -Is)] peg v2 eval noise=$nm START ===" >> "$MASTER"
    CUDA_VISIBLE_DEVICES=2 ARCH_EE_VEL=1 $PY eval_arch.py --headless --task Isaac-Forge-PegInsert-ARCH-v0 \
      --checkpoint "$ck" --num_envs 32 --episodes 256 --protocol_seed 42 \
      --fixed_pos_noise_mm "$nm" --dyn_rand on --tag "arch_pegv2_n${nm}" \
      >> "$LOGD/arch_eval_pegv2_n${nm}.log" 2>&1
    echo "=== [$(date -Is)] peg v2 eval noise=$nm rc=$? ===" >> "$MASTER"
  done
fi
echo "=== [$(date -Is)] ARCH_PEG_V2_DONE ===" >> "$MASTER"
