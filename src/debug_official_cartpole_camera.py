"""Experiment 7: does the OFFICIAL, zero-custom-code Isaac-Cartpole-RGB-Camera-Direct-v0
task -- a maintained DirectRLEnv+TiledCamera example going through the exact same
reset_idx() -> scene.reset(env_ids) -> sensor.reset(env_ids) codepath that crashes
for every Forge/Factory+camera variant tested so far -- also crash on this machine?

If YES: the regression is at the InteractiveScene/DirectRLEnv level in general,
unrelated to Factory/Forge/our code entirely -- a core IsaacLab-on-this-host bug.
If NO: something remains specific to the Factory/Forge task family itself (its
own _reset_idx chain, EventManager terms, multi-articulation scene, etc.) beyond
plain construction order (already ruled out in experiment 6).

usage: debug_official_cartpole_camera.py --num_envs 4
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401 -- registers Isaac-Cartpole-RGB-Camera-Direct-v0
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg("Isaac-Cartpole-RGB-Camera-Direct-v0", device=args_cli.device or "cuda:0",
                         num_envs=args_cli.num_envs)
print(f"[DEBUG] constructing official Isaac-Cartpole-RGB-Camera-Direct-v0 num_envs={args_cli.num_envs}", flush=True)
env = gym.make("Isaac-Cartpole-RGB-Camera-Direct-v0", cfg=env_cfg)
print("[DEBUG] construction OK, calling reset()", flush=True)
obs, info = env.reset()
print("[DEBUG] RESET_OK", flush=True)
act = torch.rand((env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device) * 2 - 1
env.step(act)
print("[DEBUG] STEP_OK", flush=True)
env.close()
simulation_app.close()
