#!/usr/bin/env bash
# Design-validation ablation for C3-TC: run zero + adapter injection on peg @5mm
# on GPU1 (clean, no orphan) after the GPU1 teacher chain (peg->gear) finishes.
# true=0.9688 already obtained (valid). Expected: zero << true (noise slot is
# load-bearing for T-C, unlike T-B where zero was still 0.984), adapter between.
export OMNI_KIT_ACCEPT_EULA=Y
PY=~/miniconda3/envs/isaaclab/bin/python
SRC=~/forge_ts/src/corma
LOGD=~/forge_ts/logs
MASTER=$LOGD/tc_teachers_master.log
CK=~/forge_ts/src/corma/logs/rl_games/Forge/tc_peg_s0/nn/last_Forge_ep_200_rew_347.71625.pth
cd "$SRC"
echo "=== [$(date -Is)] ablation waiting for TC_GPU1_CHAIN_DONE (GPU1 free) ===" >> "$LOGD/tc_peg_ablation.log"
while ! grep -q "TC_GPU1_CHAIN_DONE" "$MASTER" 2>/dev/null; do sleep 120; done
sleep 30  # let GPU1 memory settle
echo "=== [$(date -Is)] GPU1 free; running zero+adapter on GPU1 ===" >> "$LOGD/tc_peg_ablation.log"
for mode in zero adapter; do
  env CUDA_VISIBLE_DEVICES=1 $PY eval_corma_c3_tc.py --headless \
    --task Isaac-Forge-PegInsert-TC-v0 --tb_checkpoint "$CK" \
    --adapter ~/forge_ts/adapter_ckpts/peg/adapter_best.pt \
    --adapter_norm ~/forge_ts/adapter_ckpts/peg/adapter_norm.npz \
    --num_envs 32 --episodes 64 --protocol_seed 42 --fixed_pos_noise_mm 5 --dyn_rand on \
    --inject_mode $mode --tag tc_peg_5mm_${mode}_gpu1 \
    >> "$LOGD/tc_eval_peg_${mode}.log" 2>&1
  echo "=== [$(date -Is)] ablation $mode rc=$?" >> "$LOGD/tc_peg_ablation.log"
done
echo "=== [$(date -Is)] TC_PEG_ABLATION_GPU1_DONE" >> "$LOGD/tc_peg_ablation.log"
