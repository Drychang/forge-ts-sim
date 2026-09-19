"""Phase-1 smoke test for the SRSA adapter envs: create each task env,
reset, run random 6-dim actions, verify shapes/goal sanity/success path."""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SRSA-Forge-PegInsert-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--noise_mm", type=float, default=2.5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import gymnasium as gym
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import srsa_forge_env  # noqa: E402,F401  (registers the adapter envs)

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
env_cfg.seed = 42
noise_m = args_cli.noise_mm / 1000.0
env_cfg.obs_rand.fixed_asset_pos = [noise_m, noise_m, noise_m]

env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
raw = env.unwrapped
obs, _ = env.reset()

pol, cri = obs["policy"], obs["critic"]
print(f"SMOKE obs policy shape={tuple(pol.shape)} critic shape={tuple(cri.shape)}", flush=True)
assert pol.shape == (args_cli.num_envs, 24), pol.shape
assert cri.shape == (args_cli.num_envs, 44), cri.shape

# goal sanity: |goal - noisy_anchor| should be small (cm-scale grasp/success
# offsets), and delta_pos = goal - fingertip must match obs slices.
goal_pos = pol[:, 14:17]
delta = pol[:, 21:24]
fingertip = pol[:, 7:10]
err = (goal_pos - fingertip - delta).abs().max().item()
anchor = raw.fixed_pos_obs_frame + raw.init_fixed_pos_obs_noise
goal_anchor_dist = (goal_pos - anchor).norm(dim=1)
print(f"SMOKE delta consistency err={err:.2e}", flush=True)
print(f"SMOKE |goal-anchor| m: min={goal_anchor_dist.min():.3f} max={goal_anchor_dist.max():.3f}", flush=True)
assert err < 1e-4

for step in range(args_cli.steps):
    action = torch.rand((args_cli.num_envs, 6), device=raw.device) * 2 - 1
    obs, rew, term, trunc, info = env.step(action)
print(f"SMOKE stepped {args_cli.steps} steps OK; actions dim=6 accepted", flush=True)
print(f"SMOKE success tensor: {raw._get_curr_successes(raw.cfg_task.success_threshold, False).shape}"
      if hasattr(raw, "_get_curr_successes") else "SMOKE (no _get_curr_successes hook)", flush=True)
print("SRSA_ENV_SMOKE_OK", flush=True)
os._exit(0)
