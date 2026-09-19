"""Experiment 8: does the camera prim actually get spawned/cloned into all
num_envs environments, or only into env_0? This directly queries the USD
stage's prim count for the camera path regex vs. the robot path regex
(ground truth -- robot Articulation is known-good since headless Forge/Factory
training without cameras has run fine for hundreds of runs), WITHOUT calling
env.reset() -- so it never touches the crashing codepath and stays CUDA-clean.

If camera count != robot count (both should equal num_envs), the sensor's
_num_envs = len(find_matching_prims(env_prim_path_expr)) is undercounting
relative to what DirectRLEnv thinks num_envs is -- exactly the mismatch that
would make sensor_base.py:180's self._timestamp_last_update[env_ids] = 0.0
genuinely index out of bounds (not an async-reported red herring).

usage: debug_camera_prim_count.py --num_envs 4
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--fabric_clone", action="store_true", help="use copy_from_source=False instead of True")
parser.add_argument("--skip_table", action="store_true", help="skip the regex-path Table spawn (isolate its effect)")
parser.add_argument("--no_clone_in_fabric", action="store_true",
                     help="override cfg.scene.clone_in_fabric to False (Factory/Forge default True; cartpole camera cfg omits it -> False)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402
from isaaclab.sim.utils.queries import find_matching_prims  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg  # noqa: E402

CAM_WIDTH = 256
CAM_HEIGHT = 256


def _tp_cfg():
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/TPCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)),
        width=CAM_WIDTH, height=CAM_HEIGHT,
    )


@configclass
class BisectCfg(ForgeTaskPegInsertCfg):
    pass


class BisectEnv(ForgeEnv):
    def _setup_scene(self):
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        from isaaclab.assets import Articulation
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        if not args_cli.skip_table:
            cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
            cfg.func("/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711))

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=not args_cli.fabric_clone)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._tp_camera = TiledCamera(_tp_cfg())
        self.scene.sensors["tp_camera"] = self._tp_camera


env_cfg = BisectCfg()
env_cfg.scene.num_envs = args_cli.num_envs
if args_cli.device is not None:
    env_cfg.sim.device = args_cli.device
print(f"[DEBUG] cfg.scene.clone_in_fabric (before override) = {env_cfg.scene.clone_in_fabric}", flush=True)
if args_cli.no_clone_in_fabric:
    env_cfg.scene.clone_in_fabric = False

print(f"[DEBUG] constructing BisectEnv num_envs={args_cli.num_envs} clone_in_fabric={env_cfg.scene.clone_in_fabric}", flush=True)
env = BisectEnv(env_cfg, render_mode=None)
print("[DEBUG] construction OK (NOT calling reset() -- pure USD stage inspection)", flush=True)

env_prims = find_matching_prims("/World/envs/env_.*")
robot_prims = find_matching_prims("/World/envs/env_.*/Robot")
table_prims = find_matching_prims("/World/envs/env_.*/Table")
cam_prims = find_matching_prims("/World/envs/env_.*/TPCamera")

print(f"[COUNT] requested num_envs        = {args_cli.num_envs}", flush=True)
print(f"[COUNT] env root prims (env_.*)   = {len(env_prims)}  paths={[p.GetPath().pathString for p in env_prims]}", flush=True)
print(f"[COUNT] Robot prims                = {len(robot_prims)}  paths={[p.GetPath().pathString for p in robot_prims]}", flush=True)
print(f"[COUNT] Table prims                = {len(table_prims)}  paths={[p.GetPath().pathString for p in table_prims]}", flush=True)
print(f"[COUNT] TPCamera prims              = {len(cam_prims)}  paths={[p.GetPath().pathString for p in cam_prims]}", flush=True)

if len(cam_prims) != args_cli.num_envs:
    print(f"[VERDICT] MISMATCH: camera prim count ({len(cam_prims)}) != num_envs ({args_cli.num_envs}) "
          f"-- this alone explains the index-out-of-bounds assert.", flush=True)
else:
    print("[VERDICT] camera prim count matches num_envs -- mismatch theory REFUTED, cause is elsewhere.", flush=True)

print("[DEBUG] now attempting env.reset() + env.step() to confirm end-to-end...", flush=True)
import torch  # noqa: E402
env.reset()
print("[DEBUG] RESET_OK", flush=True)
act = torch.rand((env.num_envs, env.cfg.action_space), device=env.device) * 2 - 1
env.step(act)
print("[DEBUG] STEP_OK", flush=True)
img = env._tp_camera.data.output["rgb"]
print(f"[DEBUG] camera data read OK, shape={tuple(img.shape)} dtype={img.dtype}", flush=True)

env.close()
simulation_app.close()
