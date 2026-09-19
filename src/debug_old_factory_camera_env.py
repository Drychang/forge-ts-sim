"""Sanity check: run the OLD, previously-proven factory_camera_env.py
VERBATIM (as a module, unmodified) to see if it still works today. If this
crashes too, the "Factory+camera is proven fine" premise itself has
regressed on this machine/IsaacLab version since it was last verified --
this is NOT a Forge-specific or new-code bug at all.

usage: debug_old_factory_camera_env.py --task_variant peg --num_envs 32
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task_variant", type=str, default="peg", choices=["peg", "gear", "nut"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/force_vla_research/src/envs"))
from factory_camera_env import (  # noqa: E402
    FactoryCameraEnv,
    FactoryCameraTaskPegInsertCfg,
    FactoryCameraTaskGearMeshCfg,
    FactoryCameraTaskNutThreadCfg,
)

CFGS = {"peg": FactoryCameraTaskPegInsertCfg, "gear": FactoryCameraTaskGearMeshCfg, "nut": FactoryCameraTaskNutThreadCfg}

cfg = CFGS[args_cli.task_variant]()
cfg.scene.num_envs = args_cli.num_envs
if args_cli.device is not None:
    cfg.sim.device = args_cli.device

print(f"[DEBUG] constructing OLD FactoryCameraEnv (verbatim, unmodified) task={args_cli.task_variant} "
      f"num_envs={args_cli.num_envs}", flush=True)
env = FactoryCameraEnv(cfg, render_mode=None)
print("[DEBUG] construction OK, calling reset()", flush=True)
env.reset()
print("[DEBUG] RESET_OK", flush=True)
act = torch.rand((env.num_envs, env.cfg.action_space), device=env.device) * 2 - 1
env.step(act)
print("[DEBUG] STEP_OK", flush=True)
env.close()
simulation_app.close()
