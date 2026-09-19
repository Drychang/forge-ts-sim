"""Control group for experiment 8: run the SAME find_matching_prims("/World/envs/env_.*")
query against the official, working Isaac-Cartpole-RGB-Camera-Direct-v0 env, to
confirm whether the USD stage genuinely shows num_envs separate env root prims
there (theory: yes) vs. only 1 in the Forge/Factory case (experiment 8 finding).

usage: debug_cartpole_prim_count.py --num_envs 4
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
import isaaclab_tasks  # noqa: E402, F401
from isaaclab.sim.utils.queries import find_matching_prims  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg("Isaac-Cartpole-RGB-Camera-Direct-v0", device=args_cli.device or "cuda:0",
                         num_envs=args_cli.num_envs)
print(f"[DEBUG] constructing official cartpole camera env num_envs={args_cli.num_envs}", flush=True)
env = gym.make("Isaac-Cartpole-RGB-Camera-Direct-v0", cfg=env_cfg)
print("[DEBUG] construction OK (NOT calling reset())", flush=True)

env_prims = find_matching_prims("/World/envs/env_.*")

print(f"[COUNT] requested num_envs        = {args_cli.num_envs}", flush=True)
print(f"[COUNT] env root prims (env_.*)   = {len(env_prims)}  paths={[p.GetPath().pathString for p in env_prims]}", flush=True)

# also just dump every direct child under /World/envs to see actual naming/structure
import omni.usd  # noqa: E402
stage = omni.usd.get_context().get_stage()
envs_prim = stage.GetPrimAtPath("/World/envs")
children = [c.GetPath().pathString for c in envs_prim.GetChildren()] if envs_prim else []
print(f"[COUNT] /World/envs direct children = {len(children)}  paths={children}", flush=True)

env.close()
simulation_app.close()
