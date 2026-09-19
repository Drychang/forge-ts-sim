# Camera-augmented FORGE environments (Phase 0 prototype; Phase 2 data
# collection consumes get_camera_images() + get_wrench_history()).
#
# Ports the proven FactoryCameraEnv pattern (old project, same machine and
# IsaacLab commit 4927517) onto ForgeEnv:
#   - third-person + wrist TiledCameras, 256x256 rgb
#   - USD-based env cloning (fabric clone silently fails with cameras)
#   - capture_state()/reset_to_state() with the SAME state-dict layout as the
#     old reset-pool tooling, adapted to FORGE's extra per-episode randomization
# and adds a NEW physics-rate (per-substep) raw wrench history for FMT's
# frequency binning.
#
# Cameras require AppLauncher(enable_cameras=True). No gym registration:
# collectors/benchmarks import ForgeCameraEnv + a task cfg directly and call
# ForgeCameraEnv(cfg, render_mode=None), same as the old project.

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.forge import forge_utils
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import (
    ForgeEnvCfg,
    ForgeTaskPegInsertCfg,
    ForgeTaskGearMeshCfg,
    ForgeTaskNutThreadCfg,
)

CAM_WIDTH = 256
CAM_HEIGHT = 256


def _make_tp_camera_cfg():
    # "table_cam" pose from IsaacLab's Franka visuomotor stack env — proven
    # values, do not hand-tune quaternions. ForgeEnv does NOT override
    # FactoryEnv._setup_scene (same table pose, same /World/envs/env_.*/Robot),
    # so both camera poses carry over from the old FactoryCameraEnv unchanged.
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/TPCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
    )


def _make_wrist_camera_cfg():
    # "wrist_cam" pose from the same visuomotor stack env.
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
    )


@configclass
class ForgeCameraEnvCfg(ForgeEnvCfg):
    tp_camera: TiledCameraCfg = _make_tp_camera_cfg()
    wrist_camera: TiledCameraCfg = _make_wrist_camera_cfg()


@configclass
class ForgeCameraTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    tp_camera: TiledCameraCfg = _make_tp_camera_cfg()
    wrist_camera: TiledCameraCfg = _make_wrist_camera_cfg()


@configclass
class ForgeCameraTaskGearMeshCfg(ForgeTaskGearMeshCfg):
    tp_camera: TiledCameraCfg = _make_tp_camera_cfg()
    wrist_camera: TiledCameraCfg = _make_wrist_camera_cfg()


@configclass
class ForgeCameraTaskNutThreadCfg(ForgeTaskNutThreadCfg):
    tp_camera: TiledCameraCfg = _make_tp_camera_cfg()
    wrist_camera: TiledCameraCfg = _make_wrist_camera_cfg()


class ForgeCameraEnv(ForgeEnv):
    """ForgeEnv + third-person and wrist TiledCameras + per-substep wrench history.

    NOTE: _setup_scene is a full copy of FactoryEnv._setup_scene (which ForgeEnv
    inherits unmodified) with ONE change — clone_environments(copy_from_source=True).
    Factory's default fabric-based clone (copy_from_source=False) silently fails when
    cameras are enabled with num_envs > 1 ("Failed to clone in Fabric"), corrupting
    the PhysX GPU pipeline (device-side asserts on first kernel). USD-based cloning
    is slower at startup but correct. (old project decisions.md D14, proven on this
    exact machine/commit)
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        # DirectRLEnv renders whenever _sim_step_counter % cfg.sim.render_interval == 0
        # (direct_rl_env.py). FORGE cfgs never set this, so it defaults to 1 while
        # decimation=8 -> 8 RTX renders per policy step instead of 1 (8x too many).
        # Must be set before super().__init__ (which constructs the SimulationContext).
        cfg.sim.render_interval = cfg.decimation
        super().__init__(cfg, render_mode, **kwargs)
        # Ring buffer of RAW (unsmoothed) wrench samples, one per physics substep
        # of the current policy step. (num_envs, decimation=8, 6), world frame.
        self._wrench_history = torch.zeros((self.num_envs, self.cfg.decimation, 6), device=self.device)
        self._wrench_write_idx = 0

    def _setup_scene(self):
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        from isaaclab.assets import Articulation
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        # THE one-line change vs FactoryEnv: USD clone instead of fabric clone.
        self.scene.clone_environments(copy_from_source=True)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._tp_camera = TiledCamera(self.cfg.tp_camera)
        self._wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.scene.sensors["tp_camera"] = self._tp_camera
        self.scene.sensors["wrist_camera"] = self._wrist_camera

    # ------------------------------------------------------------------ cameras
    def get_camera_images(self):
        """Return {"tp_rgb", "wrist_rgb"} as uint8 numpy arrays of shape (N, H, W, 3)."""
        tp = self._tp_camera.data.output["rgb"]
        wr = self._wrist_camera.data.output["rgb"]

        def to_uint8(x):
            if x.dtype != torch.uint8:
                x = x * 255.0 if x.max() <= 1.5 else x
                x = x.clamp(0, 255).to(torch.uint8)
            return x[..., :3].cpu().numpy()

        return {"tp_rgb": to_uint8(tp), "wrist_rgb": to_uint8(wr)}

    # ------------------------------------------------- physics-rate wrench history
    def _pre_physics_step(self, action):
        super()._pre_physics_step(action)
        self._wrench_write_idx = 0

    def _apply_action(self):
        super()._apply_action()
        # DirectRLEnv.step() calls _apply_action() exactly cfg.decimation times per
        # policy step, each time immediately BEFORE sim.step() (verified in
        # direct_rl_env.py at target commit 4927517). Sample k therefore holds the
        # wrench after substep k-1 of this policy step; sample 0 holds the wrench
        # after the LAST substep of the previous policy step. All samples remain
        # uniformly spaced at physics_dt, which is what FMT's frequency binning
        # consumes — the one-substep lag is irrelevant to the spectrum.
        if self._wrench_write_idx < self.cfg.decimation:
            wrench = self._robot.root_physx_view.get_link_incoming_joint_force()[:, self.force_sensor_body_idx]
            self._wrench_history[:, self._wrench_write_idx] = wrench
            self._wrench_write_idx += 1

    def get_wrench_history(self):
        """RAW (unsmoothed) force-sensor wrench for the last policy step.

        Returns (num_envs, decimation=8, 6) tensor, world frame (the same
        convention as ForgeEnv.force_sensor_world; no EMA, no frame change,
        no observation noise). FMT's frequency binning consumes this directly.
        Zeros before the first policy step / right after reset_to_state().
        """
        return self._wrench_history.clone()

    # ---------------------------------------------------------------- diverse reset
    # Snapshot/restore the full resettable sim state, so data-collection episodes
    # can start from intermediate states sampled along successful teacher
    # trajectories. Layout is IDENTICAL to the old FactoryCameraEnv pool format
    # (same keys/shapes) so existing reset-pool tooling keeps working.
    # Velocities are zeroed on restore (start from rest — no momentum mismatch).
    def capture_state(self):
        return {
            "joint_pos": self._robot.data.joint_pos.detach().clone().cpu(),            # (N, 9)
            "fixed_pos": self._fixed_asset.data.root_pos_w.detach().clone().cpu(),     # (N, 3) world
            "fixed_quat": self._fixed_asset.data.root_quat_w.detach().clone().cpu(),   # (N, 4)
            "held_pos": self._held_asset.data.root_pos_w.detach().clone().cpu(),       # (N, 3) world
            "held_quat": self._held_asset.data.root_quat_w.detach().clone().cpu(),     # (N, 4)
            "ctrl_target_joint_pos": self.ctrl_target_joint_pos.detach().clone().cpu(),
            "fixed_pos_obs_frame": self.fixed_pos_obs_frame.detach().clone().cpu(),
            "init_fixed_pos_obs_noise": self.init_fixed_pos_obs_noise.detach().clone().cpu(),
        }

    def reset_to_state(self, state, env_ids=None):
        """Restore a capture_state() snapshot.

        FORGE's per-episode randomization (EMA factor, prop gains, pos/rot
        thresholds, contact-penalty thresholds, dead zones, quat flips, smoothed
        force reset) is NOT part of the snapshot — it is RE-DRAWN here with the
        same helpers ForgeEnv._reset_idx uses, so every restore sees a fresh
        dynamics draw (matches FORGE's training-time domain randomization; the
        pool stays purely kinematic and stays compatible with old tooling).

        Episode bookkeeping (episode_length_buf, reset_buf, success stats) is
        not touched; call env.reset() once before using restores, same as the
        old collector did. Like ForgeEnv._reset_idx, the randomization re-draw
        acts on ALL envs — intended usage is full-batch restores (factory
        assumes all envs reset together).
        """
        dev = self.device
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=dev)
        n = len(env_ids)
        zero_vel = torch.zeros((n, 6), device=dev)
        jp = state["joint_pos"].to(dev)
        self._robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
        fixed_pose = torch.cat([state["fixed_pos"].to(dev), state["fixed_quat"].to(dev)], dim=-1)
        held_pose = torch.cat([state["held_pos"].to(dev), state["held_quat"].to(dev)], dim=-1)
        self._fixed_asset.write_root_pose_to_sim(fixed_pose, env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self._held_asset.write_root_pose_to_sim(held_pose, env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        if self.cfg_task.name == "gear_mesh":
            # Gear pose is not independent state: FactoryEnv always re-poses both
            # gears onto the fixed asset's root pose on reset (factory_env.py
            # randomize_initial_state, small/large_gear_state[:,0:7]=fixed_state[:,0:7]).
            # reset_to_state bypasses that codepath entirely, so without this the
            # gears would stay at the PREVIOUS episode's (randomized) fixed-asset
            # pose -> visually/physically inconsistent scene after a pool restore.
            self._small_gear_asset.write_root_pose_to_sim(fixed_pose, env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
            self._large_gear_asset.write_root_pose_to_sim(fixed_pose, env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
        self.ctrl_target_joint_pos[:] = state["ctrl_target_joint_pos"].to(dev)
        self.fixed_pos_obs_frame[:] = state["fixed_pos_obs_frame"].to(dev)
        self.init_fixed_pos_obs_noise[:] = state["init_fixed_pos_obs_noise"].to(dev)
        for _ in range(2):  # extra settle so the restored grasp/contact stabilises
            self.step_sim_no_action()

        # Fresh per-episode dynamics draw + action bootstrap consistent with the
        # restored pose (mirrors ForgeEnv._reset_idx after super()._reset_idx).
        self._redraw_episode_randomization()
        self._reinit_actions_from_pose()
        self._wrench_history.zero_()
        self._wrench_write_idx = 0

        # step_sim_no_action uses render=False -> TiledCameras hold a stale frame.
        # Render once so the FIRST observation after reset reflects the restored
        # state (otherwise the policy's first action is computed on the pre-reset
        # image and can knock the carefully-centred peg off the hole).
        try:
            self.sim.render()
        except Exception:  # noqa: BLE001
            pass
        self._compute_intermediate_values(dt=self.physics_dt)

    def _redraw_episode_randomization(self):
        """Re-draw everything ForgeEnv._reset_idx randomizes per episode (verbatim logic).

        KNOWN GAP: does not re-run the EventManager 'reset'-mode terms (currently
        just held-asset mass, +-0.005 kg), since those normally fire inside
        DirectRLEnv._reset_idx, which reset_to_state bypasses. Held-asset mass
        stays at whatever the last natural reset drew across pool restores.
        """
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        prop_gains = forge_utils.get_random_prop_gains(
            self.default_gains.clone(), self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.default_pos_threshold.clone(), self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.default_rot_threshold.clone(), self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reinit_actions_from_pose(self):
        """Bootstrap actions from the current (restored) pose for correct EMA.

        Mirrors ForgeEnv._reset_idx: FORGE actions are targets relative to the
        fixed asset, so a zero action would command a jump to the bolt top.
        Requires _compute_intermediate_values to have run (settle steps do that).
        """
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.prev_actions)

        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )
        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )
        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0  # success prediction: "not succeeded"
