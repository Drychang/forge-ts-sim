#!/usr/bin/env python3
"""x20 -- measure the ACHIEVED fingertip roll/pitch tilt in sim rollouts.

WHY: the Franka side needs a tool-verticality gate threshold derived from data,
not guessed. They cannot read it off the training data because
`forge_env.py:77` does `noisy_fingertip_quat[:, [0, 3]] = 0.0` -- the ENV itself
zeroes quat w and z before the observation is formed, so every stored
`student_obs` reports a perfectly vertical tool regardless of the true pose.

The true pose lives in `raw_env.fingertip_midpoint_quat`, which never reaches
the policy. This logs it during teacher rollouts (the very distribution the
student was trained to imitate) and reports the tilt-from-vertical.

Tilt definition: angle between the tool's own +z axis and world -z (gripper
pointing straight down = 0 deg). Reported as a single COMBINED tilt, which is
what the Franka side proposed replacing the separate roll/pitch gates with.

Also segments by insertion depth so the gate can be tightened only where the
geometry actually binds (gear clearance gives a 1.452 deg limit at full
engagement, but the approach phase is unconstrained).

Usage:
    CUDA_VISIBLE_DEVICES=1 python x20_tilt_measure.py --headless --episodes 128
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.argv_backup = list(sys.argv)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Forge-GearMesh-TBCamera-v0")
parser.add_argument("--checkpoint", default=os.path.expanduser(
    "~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_gear_s0/nn/"
    "last_Forge_ep_200_rew_746.5314.pth"))
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default=os.path.expanduser("~/forge_ts/logs/x20_tilt.npz"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

sys.path.insert(0, os.path.expanduser("~/forge_ts/src"))
import tb_camera_env  # noqa: F401  registers the task


def quat_tilt_deg(q):
    """Angle between the tool's +z axis and world -z, in degrees. q is (w,x,y,z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # third column of R(q) = tool z axis expressed in world
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    # angle to world -z  => cos = -zz
    c = torch.clamp(-zz, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(c)), zx, zy


def main():
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

    vecenv.register("IsaacRlgWrapper",
                    lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
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

    max_ep_len = raw_env.max_episode_length
    n_envs = raw_env.num_envs
    need_eps = args_cli.episodes

    tilts, relz, phases = [], [], []
    done_eps = 0
    step_idx = 0
    t0 = time.time()

    print(f"[x20] task={args_cli.task} envs={n_envs} max_ep_len={max_ep_len} "
          f"target_eps={need_eps}", flush=True)
    print("[x20] NOTE fingertip_midpoint_quat is the TRUE pose; "
          "obs quat has w,z zeroed by forge_env.py:77", flush=True)

    while done_eps < need_eps:
        with torch.inference_mode():
            q = raw_env.fingertip_midpoint_quat.detach()
            tilt, _, _ = quat_tilt_deg(q)
            rel = (raw_env.fingertip_midpoint_pos.detach()
                   - raw_env.fixed_pos_obs_frame.detach())[:, 2]

            tilts.append(tilt.cpu().numpy().copy())
            relz.append(rel.cpu().numpy().copy())
            phases.append(np.full(n_envs, step_idx / max_ep_len, np.float32))

            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=False)
            obs, _, dones, _ = wrapped.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            step_idx += 1
            if step_idx >= max_ep_len:
                done_eps += n_envs
                step_idx = 0
                if agent.is_rnn:
                    agent.init_rnn()
                print(f"[x20] {done_eps}/{need_eps} episodes  "
                      f"{time.time()-t0:.0f}s", flush=True)

    T = np.concatenate(tilts)
    Z = np.concatenate(relz) * 1000.0
    P = np.concatenate(phases)
    np.savez_compressed(args_cli.out, tilt_deg=T, rel_z_mm=Z, phase=P)

    print()
    print("=" * 70)
    print("ACHIEVED FINGERTIP TILT FROM VERTICAL  (deg)  n=%d" % len(T))
    print("=" * 70)
    print("  mean %.4f   median %.4f   std %.4f" % (T.mean(), np.median(T), T.std()))
    for p in (50, 90, 95, 99, 99.9, 100):
        print("    p%-5s = %.4f" % (p, np.percentile(T, p)))
    print()
    print("by insertion depth (fingertip z relative to gear-post frame):")
    print("  %-16s %8s %8s %8s %8s %9s" % ("rel-z (mm)", "mean", "p95", "p99", "max", "n"))
    for lo, hi in [(-100, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 50), (50, 999)]:
        m = (Z >= lo) & (Z < hi)
        if m.sum() > 50:
            print("  %-16s %8.4f %8.4f %8.4f %8.4f %9d"
                  % ("%d..%d" % (lo, hi), T[m].mean(), np.percentile(T[m], 95),
                     np.percentile(T[m], 99), T[m].max(), m.sum()))
    print()
    print("by episode phase:")
    print("  %-16s %8s %8s %8s %9s" % ("phase", "mean", "p95", "max", "n"))
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        m = (P >= lo) & (P < hi)
        if m.sum() > 50:
            print("  %-16s %8.4f %8.4f %8.4f %9d"
                  % ("%.0f-%.0f%%" % (lo * 100, hi * 100), T[m].mean(),
                     np.percentile(T[m], 95), T[m].max(), m.sum()))
    print()
    print("saved -> %s" % args_cli.out)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
