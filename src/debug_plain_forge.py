"""Minimal isolation test: plain ForgeEnv (no camera subclass) at a small num_envs."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("Isaac-Forge-PegInsert-Direct-v0", num_envs=args_cli.num_envs)
if args_cli.device is not None:
    env_cfg.sim.device = args_cli.device
print(f"[DEBUG] constructing plain ForgeEnv num_envs={args_cli.num_envs}", flush=True)
env = gym.make("Isaac-Forge-PegInsert-Direct-v0", cfg=env_cfg, render_mode=None)
print("[DEBUG] construction OK, calling reset()", flush=True)
obs, _ = env.reset()
print("[DEBUG] RESET_OK", flush=True)
for i in range(5):
    import torch
    act = torch.rand((env.unwrapped.num_envs, env.unwrapped.action_space.shape[-1]), device=env.unwrapped.device) * 2 - 1
    obs, rew, term, trunc, info = env.step(act)
print("[DEBUG] STEP_OK", flush=True)
env.close()
simulation_app.close()
