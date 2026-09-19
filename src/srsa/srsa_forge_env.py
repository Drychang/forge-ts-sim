"""AutoMate-semantics adapter envs over the FORGE tasks, for the SRSA baseline.

Purpose (FORGE_TS_SRSA_PORT_PLAN_2026-07-19.md, Option C): run SRSA's
skill-library policies + SIL fine-tuning against our FORGE 3 tasks in SRSA's
NATIVE observation/action semantics, with the noise axis injected exactly the
way the frozen protocol injects it for T-A.

Faithfulness sources (read line-by-line 2026-07-20 -- no guessed semantics):
  automate/assembly_env.py      goal chain (296-306), obs dicts (351-385),
                                EMA smoothing (390-398), action pipeline (436-486)
  automate/assembly_env_cfg.py  obs_order/state_order, CtrlCfg constants
  factory/factory_env.py        noise chain: NOISY estimate = fixed_pos_obs_frame
                                + init_fixed_pos_obs_noise (lines 162/271);
                                obs_frame itself = true fixed-asset tip (655-669)
  factory/factory_utils.py      get_held_base_pose / get_target_held_base_pose

Deliberate deltas vs stock AutoMate (documented for the paper):
  1. fingertip_goal is anchored at the NOISY fixed estimate
     (fixed_pos_obs_frame + init_fixed_pos_obs_noise) -- the noise axis.
     Stock AutoMate computes the goal from the true pose and only pollutes
     its action clip frame; anchoring BOTH (goal + clip frame) at the same
     noisy estimate reproduces FORGE's action-frame trap semantics 1:1.
  2. AutoMate reads fingertip-goal offsets from per-assembly JSON. We instead
     MEASURE the equivalent chain from the env's own post-reset geometry:
     goal = target_held_base_pose(noisy anchor) ∘ (held_base -> fingertip
     grasp transform captured right after reset). Same meaning: fingertip
     pose when the held asset reaches its success pose under current grasp.
  3. Controller = AutoMate constants (EMA 0.2, thresholds 0.1m/0.01rad,
     upright lock, gains [100,100,100,30,30,30], rot_deriv_scale 10),
     re-applied after every reset (forge's reset randomizes gains -- we
     deliberately bypass that: SRSA's recipe has a fixed controller).
     Physics-level randomization (mass/friction events) is left untouched.
  4. Success / termination / episode length stay FORGE-native -- the SR
     numbers must be comparable with T-A/T-B/student rows.
  5. Rewards for fine-tuning are handled by the training-side wrapper
     (sparse-from-success first, FORGE shaped as fallback), not here.

Registered ids (state-only, no cameras):
  Isaac-SRSA-Forge-{PegInsert,GearMesh,NutThread}-v0
"""

import gymnasium as gym
import torch

from isaaclab.utils import configclass

import isaacsim.core.utils.torch as torch_utils  # noqa: E402  (same import factory/automate use)

from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env_cfg import (  # noqa: E402
    ForgeTaskGearMeshCfg,
    ForgeTaskNutThreadCfg,
    ForgeTaskPegInsertCfg,
)

import os as _os

# SRSA fine-tuning uses sparse rewards (AssemblySparseEnv line 89-91:
# rew = ep_succeeded.float()). Same env-var-gated pattern as TB_RANDOMIZE_NOISE;
# default 0 = FORGE shaped reward untouched (screening/eval unaffected).
SRSA_SPARSE_REWARD = _os.environ.get("SRSA_SPARSE_REWARD", "0") == "1"

# ---- AutoMate CtrlCfg constants (assembly_env_cfg.py 47-62, verbatim) ----
# DIAGNOSIS knob (2026-07-25): SRSA_FORGE_CTRL=forge switches the controller
# from AutoMate's soft/coarse constants to FORGE-native ones, to separate
# "adapter-env obs semantics" from "AutoMate controller too soft/coarse for
# 8mm tight-fit insertion" as the cause of the low SRSA scores.
_CTRL = _os.environ.get("SRSA_FORGE_CTRL", "automate")

AUTOMATE_EMA_FACTOR = 0.2
AUTOMATE_POS_THRESHOLD = (0.1, 0.1, 0.1)
AUTOMATE_ROT_THRESHOLD = (0.01, 0.01, 0.01)
AUTOMATE_POS_ACTION_BOUNDS = (0.1, 0.1)  # (lo, hi) magnitudes used in torch.clip
AUTOMATE_TASK_PROP_GAINS = (100.0, 100.0, 100.0, 30.0, 30.0, 30.0)
AUTOMATE_ROT_DERIV_SCALE = 10.0

if _CTRL == "forge":  # FORGE-native controller (forge_env_cfg / factory_env_cfg)
    AUTOMATE_TASK_PROP_GAINS = (565.0, 565.0, 565.0, 28.0, 28.0, 28.0)
    AUTOMATE_POS_THRESHOLD = (0.02, 0.02, 0.02)
    AUTOMATE_ROT_THRESHOLD = (0.097, 0.097, 0.097)
    AUTOMATE_POS_ACTION_BOUNDS = (0.05, 0.05)
    AUTOMATE_EMA_FACTOR = 0.0625  # forge ema_factor_range [0.025,0.1] midpoint
    print("[srsa_forge_env] DIAG: using FORGE-NATIVE controller constants", flush=True)

OBS_ORDER = ("joint_pos", "fingertip_pos", "fingertip_quat",
             "fingertip_goal_pos", "fingertip_goal_quat", "delta_pos")          # 7+3+4+3+4+3 = 24
STATE_ORDER = ("joint_pos", "joint_vel", "fingertip_pos", "fingertip_quat",
               "ee_linvel", "ee_angvel", "fingertip_goal_pos",
               "fingertip_goal_quat", "held_pos", "held_quat", "delta_pos")     # = 44


class SRSAForgeEnv(ForgeEnv):
    """FORGE task env exposed through AutoMate's obs/action interface."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Let forge size all its INTERNAL buffers exactly as stock (7-dim
        # actions, its own obs dims). We only re-shape the EXTERNAL spaces
        # afterwards -- zero risk of breaking forge internals.
        super().__init__(cfg, render_mode, **kwargs)

        self._grasp_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._grasp_quat[:, 0] = 1.0
        self._grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._automate_actions = torch.zeros((self.num_envs, 6), device=self.device)
        # NOTE: controller pinning happens in _reset_idx -- factory only
        # creates task_prop_gains during its first reset, not in __init__.

        # ---- external gym spaces: AutoMate interface (obs 24 / state 44 / act 6) ----
        inf = float("inf")
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(low=-inf, high=inf, shape=(24,)),
                "critic": gym.spaces.Box(low=-inf, high=inf, shape=(44,)),
            }
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        # rl_games' IsaacLab wrapper reads these cfg ints in some paths:
        self.cfg.observation_space = 24
        self.cfg.state_space = 44
        self.cfg.action_space = 6

    # ------------------------------------------------------------------ #
    def _apply_automate_controller(self):
        """AutoMate uses a FIXED controller; forge's reset re-randomizes /
        re-defaults gains and thresholds for ALL envs -- so re-pin the FULL
        tensors after every reset (idempotent constants, no slicing issues).
        Also safe on the very first reset, where factory CREATES
        task_prop_gains for the first time."""
        n = self.num_envs
        self.task_prop_gains = torch.tensor(AUTOMATE_TASK_PROP_GAINS, device=self.device).repeat(n, 1)
        self.pos_threshold = torch.tensor(AUTOMATE_POS_THRESHOLD, device=self.device).repeat(n, 1)
        self.rot_threshold = torch.tensor(AUTOMATE_ROT_THRESHOLD, device=self.device).repeat(n, 1)
        # AutoMate assembly_env._set_gains (lines 487-491), verbatim: critical
        # damping with rot deriv scale. factory has no such method (it uses
        # factory_utils.get_deriv_gains without the rot scale).
        self.task_deriv_gains = 2 * torch.sqrt(self.task_prop_gains)
        self.task_deriv_gains[:, 3:6] /= AUTOMATE_ROT_DERIV_SCALE

    def _capture_grasp_offset(self, env_ids):
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg,
            self.num_envs, self.device,
        )
        inv_quat, inv_pos = torch_utils.tf_inverse(held_base_quat[env_ids], held_base_pos[env_ids])
        g_quat, g_pos = torch_utils.tf_combine(
            inv_quat, inv_pos,
            self.fingertip_midpoint_quat[env_ids], self.fingertip_midpoint_pos[env_ids],
        )
        self._grasp_quat[env_ids] = g_quat
        self._grasp_pos[env_ids] = g_pos

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_automate_actions"):
            return  # construction-time reset before our buffers exist
        self._compute_intermediate_values(dt=self.physics_dt)
        self._capture_grasp_offset(env_ids)
        self._automate_actions[env_ids] = 0.0
        self._apply_automate_controller()

    # ------------------------------------------------------------------ #
    def _noisy_fixed_anchor(self):
        return self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

    def _compute_gripper_goal(self):
        tgt_pos, tgt_quat = factory_utils.get_target_held_base_pose(
            self._noisy_fixed_anchor(), self.fixed_quat,
            self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
        )
        goal_quat, goal_pos = torch_utils.tf_combine(tgt_quat, tgt_pos, self._grasp_quat, self._grasp_pos)
        return goal_quat, goal_pos

    def _get_observations(self):
        goal_quat, goal_pos = self._compute_gripper_goal()
        delta_pos = goal_pos - self.fingertip_midpoint_pos
        obs = torch.cat(
            [
                self.joint_pos[:, 0:7],
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                goal_pos,
                goal_quat,
                delta_pos,
            ],
            dim=-1,
        )
        state = torch.cat(
            [
                self.joint_pos[:, 0:7],
                self.joint_vel[:, 0:7],
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                self.fingertip_midpoint_linvel,
                self.fingertip_midpoint_angvel,
                goal_pos,
                goal_quat,
                self.held_pos,
                self.held_quat,
                delta_pos,
            ],
            dim=-1,
        )
        return {"policy": obs, "critic": state}

    # ------------------------------------------------------------------ #
    def _get_rewards(self):
        # Run forge's full reward path regardless -- it updates ep_succeeded,
        # success bookkeeping and logging. Sparse mode replaces only the
        # returned buffer, mirroring SRSA's AssemblySparseEnv semantics.
        rew = super()._get_rewards()
        if SRSA_SPARSE_REWARD:
            return self.ep_succeeded.float()
        return rew

    def _pre_physics_step(self, action):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)
        self._automate_actions = (
            AUTOMATE_EMA_FACTOR * action.clone().to(self.device)
            + (1 - AUTOMATE_EMA_FACTOR) * self._automate_actions
        )
        # forge internals (prev_actions in its own bookkeeping, reward terms)
        # expect a 7-dim self.actions; dim 7 (success-pred) is inert -> pad 0.
        self.actions = torch.cat(
            [self._automate_actions, torch.zeros((self.num_envs, 1), device=self.device)], dim=-1
        )

    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        pos_actions = self._automate_actions[:, 0:3] * torch.tensor(AUTOMATE_POS_THRESHOLD, device=self.device)
        rot_actions = self._automate_actions[:, 3:6] * torch.tensor(AUTOMATE_ROT_THRESHOLD, device=self.device)

        ctrl_target_pos = self.fingertip_midpoint_pos + pos_actions
        # forge's _get_rewards reads self.delta_pos / self.delta_yaw (action
        # penalty terms) -- normally set by forge's _apply_action, which we
        # replace. Provide the same semantics: pre-clip displacement command.
        self.delta_pos = pos_actions
        anchor = self._noisy_fixed_anchor()
        delta_pos = ctrl_target_pos - anchor
        pos_error_clipped = torch.clip(delta_pos, -AUTOMATE_POS_ACTION_BOUNDS[0], AUTOMATE_POS_ACTION_BOUNDS[1])
        self.ctrl_target_fingertip_midpoint_pos = anchor + pos_error_clipped

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / (angle.unsqueeze(-1) + 1e-9)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        target_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(target_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # AutoMate upright lock: roll=pi
        target_euler_xyz[:, 1] = 0.0      # pitch=0, yaw free
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        # delta_yaw for forge's action penalty (same wrap forge/automate use).
        import numpy as _np
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        curr_yaw = torch.where(curr_yaw > _np.deg2rad(235), curr_yaw - 2 * _np.pi, curr_yaw)
        tgt_yaw = target_euler_xyz[:, 2]
        tgt_yaw = torch.where(tgt_yaw > _np.deg2rad(235), tgt_yaw - 2 * _np.pi, tgt_yaw)
        self.delta_yaw = tgt_yaw - curr_yaw

        # factory's generate_ctrl_signals takes explicit targets (AutoMate's
        # variant reads members instead -- same math underneath).
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )


@configclass
class SRSAForgeTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    pass


@configclass
class SRSAForgeTaskGearMeshCfg(ForgeTaskGearMeshCfg):
    pass


@configclass
class SRSAForgeTaskNutThreadCfg(ForgeTaskNutThreadCfg):
    pass


for _gym_id, _cfg_name in [
    ("Isaac-SRSA-Forge-PegInsert-v0", "SRSAForgeTaskPegInsertCfg"),
    ("Isaac-SRSA-Forge-GearMesh-v0", "SRSAForgeTaskGearMeshCfg"),
    ("Isaac-SRSA-Forge-NutThread-v0", "SRSAForgeTaskNutThreadCfg"),
]:
    gym.register(
        id=_gym_id,
        entry_point=f"{__name__}:SRSAForgeEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
            "rl_games_cfg_entry_point": "SRSA.tasks.direct.srsa.agents:rl_games_ppo_sil_cfg.yaml",
        },
    )

print("[srsa_forge_env] registered Isaac-SRSA-Forge-{PegInsert,GearMesh,NutThread}-v0", flush=True)
