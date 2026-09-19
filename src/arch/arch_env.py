"""ARCH-INSERT primitive baseline (CoRL 2025) on the FORGE benchmark.

ARCH (arXiv 2409.16451, CoRL'25) is a hierarchical assembly method: model-based
primitives (GRASP/PLACE/MOVE via lazyPRM) + ONE RL-learned primitive (INSERT,
PPO in sim) + a DiT high-level policy over primitives. Only the INSERT
primitive is comparable to our single-part precision-insertion setting, so we
reimplement ITS RECIPE (the paper releases no per-task insertion numbers and
its full stack needs CPPF++ point-cloud pose estimation + IsaacLab 4.0 +
multi-part assets, none of which transfer).

ARCH INSERT recipe, per paper (only these three things are disclosed):
  obs    = end-effector pose + force-torque (FT) + EE pose RELATIVE TO GOAL
  action = end-effector VELOCITY commands
  algo   = PPO trained in simulation

Faithful mapping onto FORGE (everything not disclosed stays FORGE-native so
the SR is comparable with T-A / T-B / student / SRSA / C3):
  obs (27) = fingertip_pos(3) + fingertip_quat(4)          <- EE pose
           + ft_force_6d(6)                                 <- FULL 6D FT (ARCH
             uses force-torque; FORGE's own T-A obs only takes the 3D force,
             so this is genuinely ARCH's richer FT input)
           + fingertip_pos_rel_fixed(3) + rel_quat(4)       <- EE pose rel goal
           + prev_actions(7)
    The goal is the NOISY target estimate (fixed_pos_obs_frame + noise), which
    is exactly ARCH's own condition ("insertion goal poses are subject to
    inaccuracies from pose estimation errors") and our noise axis.
  action (6) = EE velocity (lin 3 + ang 3), integrated over one policy step
    (dt_policy = decimation/120 = 1/15 s) into a pose delta, then handed to
    FORGE's native task-space impedance controller. Velocity is scaled by
    ARCH_VEL_SCALE so the per-step displacement bound matches FORGE's own
    pos_action_bounds -- i.e. the same reachable action set as every other
    baseline, only re-parameterised as velocity (ARCH's parameterisation).
  reward/success/episode/dyn_rand = FORGE native, untouched.

Register: Isaac-Forge-{PegInsert,GearMesh,NutThread}-ARCH-v0
Train:    train_arch.py (rl_games PPO, same 200-epoch budget as T-A)
"""
import os as _os

import gymnasium as gym
import torch

from isaaclab.utils import configclass

import isaacsim.core.utils.torch as torch_utils  # noqa: E402

from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG  # noqa: E402
from isaaclab_tasks.direct.forge import agents as forge_agents  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv  # noqa: E402
from isaaclab_tasks.direct.forge.forge_env_cfg import (  # noqa: E402
    ForgeTaskGearMeshCfg,
    ForgeTaskNutThreadCfg,
    ForgeTaskPegInsertCfg,
)

# Velocity action scale: policy outputs v in [-1,1]; displacement over one
# policy step = v * ARCH_VEL_SCALE * dt_policy. With dt_policy = 1/15 s and
# scale 0.75 m/s the per-step bound is 0.05 m == FORGE's pos_action_bounds,
# so the reachable set matches all other baselines exactly.
# Both scales are chosen so that |v|=|w|=1 produces EXACTLY the same command
# forge receives from T-A's |action|=1 -- identical reachable set, only
# re-parameterised as velocity (ARCH's parameterisation):
#   position: v*0.75*(1/15) = 0.05 m  == pos_action_bounds -> same as T-A
#   rotation: w*15.0*(1/15) = 1.0 rad == rot_action_bounds -> same as T-A
# NOTE (2026-07-27): ARCH_ANGVEL_SCALE was originally 3.0, giving only 0.2 rad
# at |w|=1 -- a 5x rotation handicap vs T-A. peg/gear barely rotate so they
# were unaffected, but NutThread (whose success criterion REQUIRES a yaw
# rotation check) was crippled: plenty of contact (5.4N, 69% >5N) yet 0%
# success. Corrected to 15.0 for a fair, consistent recipe across all tasks.
ARCH_VEL_SCALE = float(_os.environ.get("ARCH_VEL_SCALE", "0.75"))      # m/s at |v|=1
ARCH_ANGVEL_SCALE = float(_os.environ.get("ARCH_ANGVEL_SCALE", "15.0"))  # rad/s at |w|=1

# New obs keys this env contributes (registered so obs_order can size them).
OBS_DIM_CFG["ft_force_6d"] = 6
OBS_DIM_CFG["fingertip_quat_rel_fixed"] = 4
STATE_DIM_CFG["ft_force_6d"] = 6
STATE_DIM_CFG["fingertip_quat_rel_fixed"] = 4

# ARCH INSERT obs: EE pose + FT(6D) + EE pose relative to goal. prev_actions
# is appended by the base class (+7) -> 20 + 7 = 27 dims.
# ARCH_EE_VEL=1 (default) reads the paper's "end-effector (EE)" as EE *state*
# = pose + velocity. Since ARCH commands VELOCITIES, velocity feedback is the
# standard/consistent reading, and without it the policy is blind to its own
# speed -- which empirically destroyed the tightest task (peg trained to a
# timid non-contacting policy: 1.23N mean force, 0% SR, reward plateaued at
# 79 vs gear's 622). ARCH_EE_VEL=0 reproduces the pose-only interpretation.
# DEFAULT 0: the peg/gear/nut checkpoints trained 2026-07-26/27 use the
# pose-only (27-dim) obs, so the default MUST stay pose-only or their evals
# break on a dim mismatch. The velocity-inclusive variant (33-dim) is opted
# into explicitly via ARCH_EE_VEL=1 for the v2 fairness retest.
ARCH_EE_VEL = _os.environ.get("ARCH_EE_VEL", "0") != "0"

ARCH_OBS_ORDER = [
    "fingertip_pos", "fingertip_quat",              # EE pose (3+4)
]
if ARCH_EE_VEL:
    ARCH_OBS_ORDER += ["ee_linvel", "ee_angvel"]    # EE velocity (3+3)
ARCH_OBS_ORDER += [
    "ft_force_6d",                                   # force-torque (6)
    "fingertip_pos_rel_fixed", "fingertip_quat_rel_fixed",  # rel to goal (3+4)
]


class ForgeARCHEnv(ForgeEnv):
    """FORGE task with ARCH-INSERT's obs/action parameterisation."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._dt_policy = self.cfg.decimation * self.cfg.sim.dt  # 8/120 = 1/15 s
        self._arch_actions = torch.zeros((self.num_envs, 6), device=self.device)
        # External action space is 6-dim velocity (FORGE internals stay 7-dim).
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.cfg.action_space = 6

    # ---- obs: ARCH's EE pose + 6D FT + relative-to-(noisy)goal pose ----
    def _get_observations(self):
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        # relative orientation EE->goal (goal orientation = fixed asset frame)
        rel_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.fixed_quat)
        )

        arch_extra = {
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "ft_force_6d": self.force_sensor_smooth,                      # full 6D FT
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat_rel_fixed": rel_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": self.noisy_force,
            "prev_actions": prev_actions,
        }
        obs_dict.update(arch_extra)
        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3],
                "ft_force_6d": self.force_sensor_smooth,
                "fingertip_quat_rel_fixed": rel_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    # ---- action: velocity commands integrated into a pose delta ----
    def _pre_physics_step(self, action):
        self._arch_actions = action.clone().to(self.device)
        # Integrate velocity over one policy step -> displacement/rotation delta,
        # expressed in FORGE's normalised action units so the native
        # _apply_action pipeline (EMA, clipping, impedance) is reused verbatim.
        lin = self._arch_actions[:, 0:3] * ARCH_VEL_SCALE * self._dt_policy   # metres
        ang = self._arch_actions[:, 3:6] * ARCH_ANGVEL_SCALE * self._dt_policy  # radians
        pos_norm = lin / torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        rot_norm = ang / torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device)
        seven = torch.cat(
            [pos_norm, rot_norm, torch.zeros((self.num_envs, 1), device=self.device)], dim=-1
        ).clamp(-1.0, 1.0)
        super()._pre_physics_step(seven)


@configclass
class ForgeARCHTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    obs_order: list = list(ARCH_OBS_ORDER)


@configclass
class ForgeARCHTaskGearMeshCfg(ForgeTaskGearMeshCfg):
    obs_order: list = list(ARCH_OBS_ORDER)


@configclass
class ForgeARCHTaskNutThreadCfg(ForgeTaskNutThreadCfg):
    obs_order: list = list(ARCH_OBS_ORDER)


for _gid, _cfg, _yaml in [
    ("Isaac-Forge-PegInsert-ARCH-v0", "ForgeARCHTaskPegInsertCfg", "rl_games_ppo_cfg.yaml"),
    ("Isaac-Forge-GearMesh-ARCH-v0", "ForgeARCHTaskGearMeshCfg", "rl_games_ppo_cfg.yaml"),
    ("Isaac-Forge-NutThread-ARCH-v0", "ForgeARCHTaskNutThreadCfg", "rl_games_ppo_cfg_nut_thread.yaml"),
]:
    gym.register(
        id=_gid,
        entry_point=f"{__name__}:ForgeARCHEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:{_cfg}",
            "rl_games_cfg_entry_point": f"{forge_agents.__name__}:{_yaml}",
        },
    )

print(f"[arch_env] ARCH_EE_VEL={ARCH_EE_VEL} obs_dims={sum({'fingertip_pos':3,'fingertip_quat':4,'ee_linvel':3,'ee_angvel':3,'ft_force_6d':6,'fingertip_pos_rel_fixed':3,'fingertip_quat_rel_fixed':4}[k] for k in ARCH_OBS_ORDER)+7}", flush=True)
print("[arch_env] registered Isaac-Forge-{PegInsert,GearMesh,NutThread}-ARCH-v0 "
      "(ARCH-INSERT primitive: EE pose + 6D FT + rel-goal obs, velocity actions)", flush=True)
