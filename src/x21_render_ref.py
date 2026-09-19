#!/usr/bin/env python3
"""x21 -- render the sim reference frames AT THE REAL CAMERA POSES.

The Franka side has only `gear_ref_wrist.png`, taken at the SIM NOMINAL wrist
pose. jitcam_real_s2 was trained with the wrist at the MEASURED real pose, so
that reference is not a valid baseline for it -- they currently have nothing to
compare their live frames against.

This renders exactly what the policy ate during training: both cameras at the
measured real extrinsics, HFOV 55.7 deg, 256x256, straight out of
`raw_env.get_camera_images()` (the same call the collector uses).

Saves several episode-start frames plus a few steps in, so they can compare
composition, not just one lucky frame.

Usage:
    OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=2 python x21_render_ref.py \
        --headless --enable_cameras
"""
import argparse
import math
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Forge-GearMesh-TBCamera-v0")
parser.add_argument("--checkpoint", default=os.path.expanduser(
    "~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_gear_s0/nn/"
    "last_Forge_ep_200_rew_746.5314.pth"))
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--out", default=os.path.expanduser("~/forge_ts/logs/x21_ref"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from PIL import Image
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

sys.path.insert(0, os.path.expanduser("~/forge_ts/src"))
import tb_camera_env  # noqa: F401  registers the task


def save(img_t, path):
    a = img_t.detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[-1] == 4:
        a = a[..., :3]
    Image.fromarray(a).save(path)


def main():
    os.makedirs(args_cli.out, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_env = env.unwrapped
    wrapped = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate)

    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = raw_env.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()

    obs = wrapped.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print("[x21] cameras are at whatever TB_CAM_* env vars were exported; "
          "check the CAM ABS lines above.", flush=True)

    want = [0, 3, 6, 9, 12, 18, 24, 30]
    with torch.inference_mode():
        for step in range(args_cli.steps + 1):
            if step in want:
                imgs = raw_env.get_camera_images()
                relz = (raw_env.fingertip_midpoint_pos
                        - raw_env.fixed_pos_obs_frame)[:, 2] * 1000.0
                for e in range(min(3, raw_env.num_envs)):
                    save(imgs["tp_rgb"][e],
                         os.path.join(args_cli.out, f"sim_e{e}_s{step:04d}_tp.png"))
                    save(imgs["wrist_rgb"][e],
                         os.path.join(args_cli.out, f"sim_e{e}_s{step:04d}_wrist.png"))
                print(f"[x21] step {step:3d}  rel_z(mm) env0..2 = "
                      f"{[round(float(v),1) for v in relz[:3]]}", flush=True)
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, _, _ = wrapped.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

    print("[x21] saved to", args_cli.out, flush=True)
    for f in sorted(os.listdir(args_cli.out)):
        print("   ", f)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
