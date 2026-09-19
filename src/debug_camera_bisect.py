"""Bisection test: ForgeCameraEnv with a selectable subset of cameras.
usage: debug_camera_bisect.py --cameras none|tp|wrist|both --num_envs 32
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--cameras", type=str, default="both", choices=["none", "tp", "wrist", "both"])
parser.add_argument("--no_events", action="store_true", help="blank out cfg.events (isolate FORGE startup event terms)")
parser.add_argument(
    "--drop_event", type=str, default=None,
    help="name of a single EventCfg attribute to set to None (surgical isolation), "
    "e.g. object_scale_mass | held_physics_material | fixed_physics_material | robot_physics_material | dead_zone_thresholds",
)
parser.add_argument("--reset_twice", action="store_true", help="call env.reset() twice in a row (does 2nd reset also crash if 1st succeeds?)")
parser.add_argument("--fabric_clone", action="store_true", help="use copy_from_source=False (default fabric clone) instead of the USD clone")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg

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


def _wrist_cfg():
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)),
        width=CAM_WIDTH, height=CAM_HEIGHT,
    )


@configclass
class BisectCfg(ForgeTaskPegInsertCfg):
    pass


class BisectEnv(ForgeEnv):
    def __init__(self, cfg, cameras, render_mode=None, **kwargs):
        self._which_cameras = cameras
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        from isaaclab.assets import Articulation
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
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

        if self._which_cameras in ("tp", "both"):
            self._tp_camera = TiledCamera(_tp_cfg())
            self.scene.sensors["tp_camera"] = self._tp_camera
        if self._which_cameras in ("wrist", "both"):
            self._wrist_camera = TiledCamera(_wrist_cfg())
            self.scene.sensors["wrist_camera"] = self._wrist_camera


from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = BisectCfg()
env_cfg.scene.num_envs = args_cli.num_envs
if args_cli.device is not None:
    env_cfg.sim.device = args_cli.device
if args_cli.no_events:
    env_cfg.events = None
if args_cli.drop_event:
    for name in args_cli.drop_event.split(","):
        setattr(env_cfg.events, name.strip(), None)

print(f"[DEBUG] constructing BisectEnv cameras={args_cli.cameras} num_envs={args_cli.num_envs} "
      f"no_events={args_cli.no_events} drop_event={args_cli.drop_event} fabric_clone={args_cli.fabric_clone}", flush=True)
env = BisectEnv(env_cfg, cameras=args_cli.cameras, render_mode=None)
print("[DEBUG] construction OK, calling reset()", flush=True)
env.reset()
print("[DEBUG] RESET_OK", flush=True)
if args_cli.reset_twice:
    env.reset()
    print("[DEBUG] RESET2_OK", flush=True)
act = torch.rand((env.num_envs, env.cfg.action_space), device=env.device) * 2 - 1
env.step(act)
print("[DEBUG] STEP_OK", flush=True)
env.close()
simulation_app.close()
