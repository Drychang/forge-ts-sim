"""Experiment 5 (Phase 2 spec 2.2): does the crash follow TiledCamera specifically,
or is it generic to any camera sensor + Factory-family env on this machine?

Same BisectEnv scaffolding as debug_camera_bisect.py, but using the non-tiled
isaaclab.sensors.Camera / CameraCfg (one render product per env) instead of
TiledCamera / TiledCameraCfg. If this does NOT crash, TiledCamera's specific
codepath is the culprit and Camera is a usable (if slower) fallback for the
whole project. If it crashes identically, the regression is even more general
than "TiledCamera is broken" -- points at the renderer/replicator stack itself.

usage: debug_camera_bisect_standard.py --cameras none|tp|wrist|both --num_envs 32
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--cameras", type=str, default="both", choices=["none", "tp", "wrist", "both"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg  # noqa: E402

CAM_WIDTH = 256
CAM_HEIGHT = 256


def _tp_cfg():
    return CameraCfg(
        prim_path="/World/envs/env_.*/TPCamera",
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)),
        width=CAM_WIDTH, height=CAM_HEIGHT,
    )


def _wrist_cfg():
    return CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        offset=CameraCfg.OffsetCfg(pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"),
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

        self.scene.clone_environments(copy_from_source=True)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self._which_cameras in ("tp", "both"):
            self._tp_camera = Camera(_tp_cfg())
            self.scene.sensors["tp_camera"] = self._tp_camera
        if self._which_cameras in ("wrist", "both"):
            self._wrist_camera = Camera(_wrist_cfg())
            self.scene.sensors["wrist_camera"] = self._wrist_camera


env_cfg = BisectCfg()
env_cfg.scene.num_envs = args_cli.num_envs
if args_cli.device is not None:
    env_cfg.sim.device = args_cli.device

print(f"[DEBUG] constructing BisectEnv (standard Camera, not Tiled) cameras={args_cli.cameras} "
      f"num_envs={args_cli.num_envs}", flush=True)
env = BisectEnv(env_cfg, cameras=args_cli.cameras, render_mode=None)
print("[DEBUG] construction OK, calling reset()", flush=True)
env.reset()
print("[DEBUG] RESET_OK", flush=True)
act = torch.rand((env.num_envs, env.cfg.action_space), device=env.device) * 2 - 1
env.step(act)
print("[DEBUG] STEP_OK", flush=True)
if args_cli.cameras != "none":
    cam = env._tp_camera if args_cli.cameras in ("tp", "both") else env._wrist_camera
    img = cam.data.output["rgb"]
    print(f"[DEBUG] camera data read OK, shape={tuple(img.shape)} dtype={img.dtype}", flush=True)
env.close()
simulation_app.close()
