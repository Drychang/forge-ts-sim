"""P2-2 collection/eval environment: ForgeTBEnv (T-B privileged teacher -- the
ONLY thing that drives actions, completely unchanged) + dual TiledCamera
(tp+wrist, 256x256) + a per-substep raw wrench buffer + the official T-A
24-dim noisy obs exposed as `self.student_obs` every step.

Cameras/wrench/student_obs are strictly additive instrumentation for the
student's future training data -- none of it feeds back into the teacher's
decisions. T-B's own 64-dim privileged obs (unchanged from tb_env.ForgeTBEnv)
remains the only input to the policy that generates env.step()'s action.

Camera-crash fix this env depends on (see reference_mercury_sim_gotchas memory
/ plan doc §Gate 0 2026-07-05 addendum): Factory/Forge's shared
InteractiveSceneCfg sets clone_in_fabric=True. Under that mode, cloned
env_1..N-1 exist only in the Fabric acceleration layer, invisible to
Usd.Stage traversal -- so Camera/TiledCamera's find_matching_prims-based
_num_envs count collapses to 1 regardless of the real num_envs, and
sensor_base.py's self._timestamp_last_update[env_ids] = 0.0 genuinely indexes
out of bounds on reset. Articulations are unaffected (they use the PhysX
tensor view API, not USD stage traversal), which is why headless Factory/Forge
training has always worked fine without cameras. Fix: override
clone_in_fabric=False on the *_TaskCfg subclasses below (mirrors what the
official cartpole_camera_env.py already does -- its InteractiveSceneCfg simply
never sets the True default its no-camera sibling cartpole_env.py uses).

Wrench sampling: ForgeEnv._apply_action recomputes _compute_intermediate_values
(and therefore self.force_sensor_world, the RAW/unsmoothed world-frame 6D
force+torque reading) on every physics substep, not just once per policy
step -- see forge_env.py L148-151 (`last_update_timestamp` staleness check).
_apply_action is called exactly cfg.decimation times per env.step() call
(direct_rl_env.py's physics-stepping loop), so appending there and flattening
in _get_observations (called exactly once, after the loop, at both step() and
reset()) gives exactly `decimation` raw physics-rate samples per policy step,
oldest-first -- matching dataset.py's `wrench_raw: (E, L, decimation, 6)`
layout and eval_frozen_student.py's `get_wrench_history() -> (N, 8, 6)`
contract exactly.

Interface consumed by collect_camera_rollouts.py (P2-2) and
student/eval_frozen_student.py (P2-1):
  env.student_obs               (num_envs, 24) float32, official T-A noisy layout
  env.get_camera_images()       {"tp_rgb": (N,H,W,3) uint8, "wrist_rgb": (N,H,W,3) uint8}
  env.get_wrench_history()      (N, decimation, 6) float32, raw/unsmoothed, oldest-first,
                                 THIS policy step's substeps only (not a rolling window --
                                 callers that need a longer window roll it themselves, same
                                 as dataset.py's rolling-window reconstruction from shards)

gym ids: Isaac-Forge-{PegInsert,GearMesh,NutThread}-TBCamera-v0
"""
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.forge import agents as forge_agents

from tb_env import (
    ForgeTBEnv,
    ForgeTBTaskGearMeshCfg,
    ForgeTBTaskNutThreadCfg,
    ForgeTBTaskPegInsertCfg,
)

CAM_WIDTH = 256
CAM_HEIGHT = 256

# ---- optional camera-extrinsics jitter (sim2real sensitivity analysis) ----
# Default OFF (all env vars unset/0) -> byte-identical camera cfgs to the
# validated collection/eval setup. When enabled, applies ONE fixed perturbation
# per process (a mounting/calibration error is constant within a run, not
# per-episode):
#   TB_CAM_JITTER_POS_MM  translation magnitude (mm), random unit direction
#   TB_CAM_JITTER_ROT_DEG rotation magnitude (deg), random axis, composed in
#                          the camera's LOCAL frame (right-multiply) -- that is
#                          what a physical mount error does
#   TB_CAM_JITTER_SEED    seed; each camera derives its own stream from
#                          (seed, camera_name) so tp/wrist get independent
#                          but reproducible perturbations


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _maybe_jitter_offset(pos, rot_wxyz, cam_name):
    import math
    import os as _os

    import numpy as _np

    # Absolute pose override, applied BEFORE jitter so the two compose: this sets where the
    # camera nominally is, jitter then perturbs around it. Used to place the simulator's
    # camera at a pose measured on the real robot.
    _abs = _os.environ.get("TB_CAM_%s_POS" % cam_name.upper(), "").strip()
    _absr = _os.environ.get("TB_CAM_%s_ROT" % cam_name.upper(), "").strip()
    if _abs:
        pos = tuple(float(v) for v in _abs.split(","))
        print("[tb_camera_env] CAM ABS %s: pos -> %s" % (cam_name, pos), flush=True)
    if _absr:
        rot_wxyz = tuple(float(v) for v in _absr.split(","))
        print("[tb_camera_env] CAM ABS %s: rot -> %s" % (cam_name, rot_wxyz), flush=True)

    pos_mm = float(_os.environ.get("TB_CAM_JITTER_POS_MM", "0") or 0)
    rot_deg = float(_os.environ.get("TB_CAM_JITTER_ROT_DEG", "0") or 0)
    if pos_mm == 0.0 and rot_deg == 0.0:
        return pos, rot_wxyz
    # Which cameras to perturb. Unset (the default) means all of them, so results
    # recorded before this option existed stay comparable. Set it to "wrist" to isolate
    # the mount that is actually wrong on the real rig, or "tp" for the other one.
    _cams = _os.environ.get("TB_CAM_JITTER_CAMS", "").strip()
    if _cams:
        _want = [c.strip() for c in _cams.split(",") if c.strip()]
        if cam_name not in _want:
            print(f"[tb_camera_env] CAM JITTER {cam_name}: SKIPPED "
                  f"(TB_CAM_JITTER_CAMS={_cams})", flush=True)
            return pos, rot_wxyz

    seed = int(_os.environ.get("TB_CAM_JITTER_SEED", "0") or 0)
    rng = _np.random.default_rng([seed, sum(ord(c) for c in cam_name)])

    d = rng.normal(size=3)
    d /= max(float(_np.linalg.norm(d)), 1e-9)
    new_pos = tuple(float(p) + float(di) * pos_mm / 1000.0 for p, di in zip(pos, d))

    axis = rng.normal(size=3)
    axis /= max(float(_np.linalg.norm(axis)), 1e-9)
    half = math.radians(rot_deg) / 2.0
    q_delta = (math.cos(half), *(float(a) * math.sin(half) for a in axis))
    new_rot = _quat_mul_wxyz(rot_wxyz, q_delta)
    # renormalize against accumulated float error
    n = math.sqrt(sum(c * c for c in new_rot))
    new_rot = tuple(c / n for c in new_rot)

    print(
        f"[tb_camera_env] CAM JITTER {cam_name}: pos_mm={pos_mm:g} rot_deg={rot_deg:g} seed={seed} "
        f"pos {tuple(round(p, 5) for p in pos)} -> {tuple(round(p, 5) for p in new_pos)} "
        f"rot {tuple(round(c, 5) for c in rot_wxyz)} -> {tuple(round(c, 5) for c in new_rot)}",
        flush=True,
    )
    return new_pos, new_rot

# Official T-A policy-slot obs order (forge_env_cfg.py ForgeEnvCfg.obs_order,
# hardcoded here -- NOT read from self.cfg.obs_order, which ForgeTBTaskXCfg
# overrides to the 64-dim privileged TB_ORDER for the teacher's own obs).
TA_OBS_ORDER = [
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "ft_force",
    "force_threshold",
]



def _cam_aperture():
    import math, os as _os
    hfov = _os.environ.get("TB_CAM_HFOV_DEG", "").strip()
    if not hfov:
        return 20.955
    hfov = float(hfov)
    ap = 2.0 * 24.0 * math.tan(math.radians(hfov / 2.0))
    print(f"[tb_camera_env] HFOV override: {hfov:.1f} deg -> horizontal_aperture={ap:.4f}", flush=True)
    return ap


def _tp_cfg():
    pos, rot = _maybe_jitter_offset((1.0, 0.0, 0.4), (0.35355, -0.61237, -0.61237, 0.35355), "tp")
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/TPCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=_cam_aperture(), clipping_range=(0.1, 2)),
        width=CAM_WIDTH, height=CAM_HEIGHT,
    )


def _wrist_cfg():
    pos, rot = _maybe_jitter_offset((0.13, 0.0, -0.15), (-0.70614, 0.03701, 0.03701, -0.70614), "wrist")
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=_cam_aperture(), clipping_range=(0.1, 2)),
        width=CAM_WIDTH, height=CAM_HEIGHT,
    )


class ForgeTBCameraEnv(ForgeTBEnv):
    """ForgeTBEnv (T-B teacher, unchanged action path) + dual camera + raw
    per-substep wrench + T-A's official 24-dim noisy obs as self.student_obs."""

    def _setup_scene(self):
        super()._setup_scene()  # full official FactoryEnv scene incl. clone_environments()
        self._tp_camera = TiledCamera(_tp_cfg())
        self.scene.sensors["tp_camera"] = self._tp_camera
        self._wrist_camera = TiledCamera(_wrist_cfg())
        self.scene.sensors["wrist_camera"] = self._wrist_camera

        self._wrench_step_samples = []
        self._wrench_last_step = torch.zeros((self.num_envs, self.cfg.decimation, 6), device=self.device)
        self.student_obs = torch.zeros((self.num_envs, 24), device=self.device)

    def _apply_action(self):
        super()._apply_action()
        # self.force_sensor_world was just refreshed above (see module docstring) --
        # raw, unsmoothed, world-frame 6D wrench at physics rate. Read-only capture.
        self._wrench_step_samples.append(self.force_sensor_world.clone())

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Factory/Forge is lockstep (_get_dones is timeout-only and identical for all
        # envs, verified in eval_frozen.py) -- env_ids is always "all" here, so a full
        # clear is correct, not an approximation. Drops any not-yet-flushed samples from
        # the episode that just ended, so the following _get_observations() (called right
        # after reset, see module docstring) sees a clean zeroed history for the new episode.
        self._wrench_step_samples = []
        self._wrench_last_step.zero_()

    def _get_observations(self):
        # Official T-A 24-dim noisy obs. NOTE: we cannot just call
        # ForgeEnv._get_observations(self) unbound -- it internally reads
        # self.cfg.obs_order/state_order, which ForgeTBTaskXCfg overrides to
        # TB_ORDER (64-dim, privileged-only keys like joint_pos/fixed_pos_obs_noise
        # that don't exist in the policy-slot obs_dict) -- that's a data-level
        # override, not just a method override, so the unbound call KeyErrors.
        # Reimplement the exact same obs_dict construction from forge_env.py
        # instead, collapsed with the hardcoded official T-A order.
        obs_dict, _ = self._get_factory_obs_state_dict()
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0
        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
            }
        )
        self.student_obs = factory_utils.collapse_obs_dict(obs_dict, TA_OBS_ORDER + ["prev_actions"]).clone()

        if len(self._wrench_step_samples) > 0:
            self._wrench_last_step = torch.stack(self._wrench_step_samples, dim=1)
            self._wrench_step_samples = []

        # T-B's 64-dim privileged obs -- the ONLY input that drives the teacher's action.
        return super()._get_observations()

    def get_camera_images(self):
        return {
            "tp_rgb": self._tp_camera.data.output["rgb"].clone(),
            "wrist_rgb": self._wrist_camera.data.output["rgb"].clone(),
        }

    def get_wrench_history(self):
        """Raw (unsmoothed), world-frame wrench for the `decimation` substeps
        that just elapsed this policy step, oldest-first. Shape (num_envs, decimation, 6)."""
        return self._wrench_last_step.clone()


@configclass
class ForgeTBCameraTaskPegInsertCfg(ForgeTBTaskPegInsertCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)


@configclass
class ForgeTBCameraTaskGearMeshCfg(ForgeTBTaskGearMeshCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)


@configclass
class ForgeTBCameraTaskNutThreadCfg(ForgeTBTaskNutThreadCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)


gym.register(
    id="Isaac-Forge-PegInsert-TBCamera-v0",
    entry_point=f"{__name__}:ForgeTBCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForgeTBCameraTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{forge_agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-GearMesh-TBCamera-v0",
    entry_point=f"{__name__}:ForgeTBCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForgeTBCameraTaskGearMeshCfg",
        "rl_games_cfg_entry_point": f"{forge_agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-NutThread-TBCamera-v0",
    entry_point=f"{__name__}:ForgeTBCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForgeTBCameraTaskNutThreadCfg",
        "rl_games_cfg_entry_point": f"{forge_agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)

print(
    "[tb_camera_env] registered Isaac-Forge-{PegInsert,GearMesh,NutThread}-TBCamera-v0",
    flush=True,
)
