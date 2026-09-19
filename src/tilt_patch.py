"""tilt_patch.py -- zero-shot FIXTURE TILT for Isaac Lab Factory/Forge (2026-09-19).

Import AFTER Kit is up (after AppLauncher) and BEFORE gym.make. It monkey-patches
FactoryEnv.randomize_initial_state at class level, so every variant that calls
super() (ForgeEnv, ForgeTBEnv, ForgeTBCameraEnv, ...) inherits the tilt.

Why a patch and not a subclass: the fixture pose is written in the middle of the
official 110-line randomize_initial_state (step 1.d) and, for gear_mesh, the two
flanking gears are written later from the same *local* fixed_state. Wrapping the
asset write calls lets us tilt all of them consistently without copying the
function.

Environment variables
  TILT_DEG   float, default 0   tilt magnitude in degrees (fixture z-axis vs world z)
  TILT_MODE  fixed|uniform      fixed = every episode exactly TILT_DEG (default);
                                uniform = per-episode U[0, TILT_DEG]
  TILT_AXIS  random|x|y         axis of the tilt in the world xy-plane (default random azimuth)
  TILT_ABOUT root|tip           pivot. root = rotate about the asset root pose (hole bottom /
                                gear base / bolt head); tip = rotate about the observation
                                frame (top of fixture) so the tip position is unchanged
                                (default root)

Notes
  * The tilt quaternion is sampled for EVERY episode even when TILT_DEG=0, so the
    frozen-protocol RNG stream is identical across tilt levels (paired comparison).
    It is NOT identical to the stock env stream (one extra torch.rand per reset).
  * With TILT_DEG=0 the written poses are bit-identical to stock (identity quat).
  * Success / reward / fixed_pos_obs_frame all use the simulated fixed_quat
    (factory_env.py _compute_intermediate_values, randomize_initial_state step 1.e,
    factory_utils.get_target_held_base_pose), so they follow the tilted fixture.
    The success test itself stays world-xy / world-z (factory_env._get_curr_successes);
    for tilt <= 10 deg the cosine error on a 25 mm insertion is <= 0.4 mm.
  * Fixtures are fixed-root articulations (they float at z in [0, 0.1] m in the stock
    env), so a tilted pose is held without falling.
"""
import math
import os

import torch

import isaacsim.core.utils.torch as torch_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

TILT_DEG = float(os.environ.get("TILT_DEG", "0") or 0.0)
TILT_MODE = os.environ.get("TILT_MODE", "fixed")
TILT_AXIS = os.environ.get("TILT_AXIS", "random")
TILT_ABOUT = os.environ.get("TILT_ABOUT", "root")
assert TILT_MODE in ("fixed", "uniform"), TILT_MODE
assert TILT_AXIS in ("random", "x", "y"), TILT_AXIS
assert TILT_ABOUT in ("root", "tip"), TILT_ABOUT

_orig_randomize = FactoryEnv.randomize_initial_state


def _sample_tilt(n, device):
    """Return (quat wxyz [n,4], angle_rad [n]) for a world-frame tilt about an xy-plane axis."""
    if TILT_AXIS == "random":
        az = torch.rand((n,), device=device) * 2.0 * math.pi
    elif TILT_AXIS == "x":
        az = torch.zeros((n,), device=device)
    else:
        az = torch.full((n,), math.pi / 2.0, device=device)
    axis = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], dim=1)
    if TILT_MODE == "uniform":
        ang = torch.rand((n,), device=device) * math.radians(TILT_DEG)
    else:
        ang = torch.full((n,), math.radians(TILT_DEG), device=device)
    return torch_utils.quat_from_angle_axis(ang, axis), ang


def _tilt_pose(root_pose, tilt_q, tip_local):
    """Apply the world-frame tilt to a [n,7] (pos, quat wxyz) root pose."""
    out = root_pose.clone()
    q = out[:, 3:7]
    out[:, 3:7] = torch_utils.quat_mul(tilt_q, q)
    if TILT_ABOUT == "tip" and tip_local is not None:
        # keep the fixture tip where it was: p_root' = p_tip - R' t_tip
        tip_before = out[:, 0:3] + torch_utils.quat_apply(q, tip_local)
        out[:, 0:3] = tip_before - torch_utils.quat_apply(out[:, 3:7], tip_local)
    return out


def _tip_local(env, n):
    tip = torch.zeros((n, 3), device=env.device)
    tip[:, 2] += env.cfg_task.fixed_asset_cfg.height + env.cfg_task.fixed_asset_cfg.base_height
    if env.cfg_task.name == "gear_mesh":
        tip[:, 0] = env.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]
    return tip


def _patched_randomize_initial_state(self, env_ids):
    n = len(env_ids)
    tilt_q, tilt_ang = _sample_tilt(n, self.device)
    if not hasattr(self, "tilt_applied_rad"):
        self.tilt_applied_rad = torch.zeros((self.num_envs,), device=self.device)
    self.tilt_applied_rad[env_ids] = tilt_ang
    tip_local = _tip_local(self, n)

    assets = [self._fixed_asset]
    if self.cfg_task.name == "gear_mesh" and getattr(self.cfg_task, "add_flanking_gears", False):
        assets += [self._small_gear_asset, self._large_gear_asset]

    originals = []
    for asset in assets:
        orig = asset.write_root_pose_to_sim

        def make_wrapper(_orig):
            def wrapped(root_pose, env_ids=None, *a, **kw):
                # Only the randomize-step writes have exactly len(env_ids) rows; be defensive anyway.
                if root_pose.shape[0] == tilt_q.shape[0]:
                    root_pose = _tilt_pose(root_pose, tilt_q, tip_local)
                return _orig(root_pose, env_ids=env_ids, *a, **kw)
            return wrapped

        asset.write_root_pose_to_sim = make_wrapper(orig)
        originals.append((asset, orig))
    try:
        return _orig_randomize(self, env_ids)
    finally:
        for asset, orig in originals:
            asset.write_root_pose_to_sim = orig


FactoryEnv.randomize_initial_state = _patched_randomize_initial_state
print(
    f"[tilt_patch] FactoryEnv.randomize_initial_state patched: TILT_DEG={TILT_DEG} MODE={TILT_MODE} "
    f"AXIS={TILT_AXIS} ABOUT={TILT_ABOUT}",
    flush=True,
)


def measured_tilt_deg(env):
    """Actual angle between the simulated fixture z-axis and world z, per env (degrees)."""
    q = env.fixed_quat
    z = torch.zeros((q.shape[0], 3), device=q.device)
    z[:, 2] = 1.0
    zf = torch_utils.quat_apply(q, z)
    return torch.rad2deg(torch.acos(zf[:, 2].clamp(-1.0, 1.0)))
