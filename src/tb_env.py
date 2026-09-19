"""T-B privileged teacher environments for the FORGE task family.

The actor observes the FULL privileged state (identical to what the official
FORGE critic sees: true fingertip/held/fixed pose, joint_pos, true velocity,
task_prop_gains, ema_factor, noise-free 3D force, thresholds) PLUS the
per-episode fixed-asset position-observation offset itself
(init_fixed_pos_obs_noise, 3 dims). Teacher never deploys, so this is legal.

WHY THE OFFSET IS NEEDED (not just clean state): FORGE anchors its pose-
target command to a NOISY action frame --
    forge_env._apply_action: fixed_pos_action_frame = fixed_pos_obs_frame + init_fixed_pos_obs_noise
-- so eval-time noise shifts where every policy's commanded pose actually
lands, regardless of how clean that policy's own observations are. Without
seeing the offset, a "clean state" actor still cannot precisely correct for
it. With the offset in its observation, T-B can learn an exact compensating
mapping and become a genuinely noise-invariant oracle upper bound.

TRAINING vs EVAL noise-distribution switch (env var, not a cfg field, so
neither train.py nor eval_frozen.py needs to be touched):
  TB_RANDOMIZE_NOISE=1 (default) -> per-episode std ~ U[0, tb_noise_std_max]
      (5mm default), offset ~ N(0, std). Training must see this wide range
      or it will be badly out-of-distribution at the high end of the frozen
      eval noise sweep (official default is a FIXED 1mm std).
  TB_RANDOMIZE_NOISE=0 -> this override is a no-op; the base FactoryEnv
      sampling (driven by cfg.obs_rand.fixed_asset_pos, exactly like T-A)
      takes over, so eval_frozen.py's --fixed_pos_noise_mm sweep controls
      the noise level unmodified. MUST be 0 for every eval run.

Full design rationale: Downloads/TB_IMPLEMENTATION_SPEC_2026-07-04.md
"""

import os

import gymnasium as gym
import torch

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG
from isaaclab_tasks.direct.forge import agents as forge_agents
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import (
    ForgeTaskGearMeshCfg,
    ForgeTaskNutThreadCfg,
    ForgeTaskPegInsertCfg,
)

TB_RANDOMIZE_NOISE = os.environ.get("TB_RANDOMIZE_NOISE", "1") != "0"

# Register the new privileged-only key, then backfill OBS_DIM_CFG with every
# STATE_DIM_CFG key so obs_order (== state_order for T-B) can look up any of
# them. Same pattern forge_env_cfg.py itself uses to extend these globals.
STATE_DIM_CFG["fixed_pos_obs_noise"] = 3
OBS_DIM_CFG.update(STATE_DIM_CFG)

# prev_actions (7 dims) is appended separately by _get_observations, exactly
# like the official obs_order/state_order lists -- do not list it here.
TB_ORDER = [
    "fingertip_pos", "fingertip_quat", "ee_linvel", "ee_angvel",
    "joint_pos", "held_pos", "held_pos_rel_fixed", "held_quat",
    "fixed_pos", "fixed_quat", "task_prop_gains", "ema_factor",
    "ft_force", "pos_threshold", "rot_threshold", "force_threshold",
    "fixed_pos_obs_noise",
]  # sums to 57 + action_space(7) = 64 dims (verified against OBS_DIM_CFG/STATE_DIM_CFG)


class ForgeTBEnv(ForgeEnv):
    """ForgeEnv variant whose actor is fully privileged (== critic)."""

    def randomize_initial_state(self, env_ids):
        # super() call chain: FactoryEnv.randomize_initial_state sets
        # self.init_fixed_pos_obs_noise[:] from cfg.obs_rand.fixed_asset_pos
        # (fixed 1mm std by default). We then override it with a wider,
        # per-episode-resampled distribution for training only.
        super().randomize_initial_state(env_ids)
        if TB_RANDOMIZE_NOISE:
            n = len(env_ids)
            std = torch.rand((n, 1), device=self.device) * self.cfg.tb_noise_std_max
            self.init_fixed_pos_obs_noise[env_ids] = torch.randn((n, 3), device=self.device) * std

    def _get_observations(self):
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        # Mirror ForgeEnv._get_observations' state_dict additions (CLEAN
        # values only -- deliberately never touch the noisy_* variants
        # ForgeEnv adds to obs_dict for the T-A actor).
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0
        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3],
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
                "fixed_pos_obs_noise": self.init_fixed_pos_obs_noise,
            }
        )

        # Actor == critic: fully privileged, symmetric observation.
        obs_dict = state_dict

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}


@configclass
class ForgeTBTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    obs_order: list = list(TB_ORDER)
    state_order: list = list(TB_ORDER)
    tb_noise_std_max: float = 0.005  # meters (5mm), matches the frozen eval sweep's high end


@configclass
class ForgeTBTaskGearMeshCfg(ForgeTaskGearMeshCfg):
    obs_order: list = list(TB_ORDER)
    state_order: list = list(TB_ORDER)
    tb_noise_std_max: float = 0.005


@configclass
class ForgeTBTaskNutThreadCfg(ForgeTaskNutThreadCfg):
    obs_order: list = list(TB_ORDER)
    state_order: list = list(TB_ORDER)
    tb_noise_std_max: float = 0.005


gym.register(
    id="Isaac-Forge-PegInsert-TB-v0",
    entry_point=f"{__name__}:ForgeTBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForgeTBTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{forge_agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-GearMesh-TB-v0",
    entry_point=f"{__name__}:ForgeTBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForgeTBTaskGearMeshCfg",
        "rl_games_cfg_entry_point": f"{forge_agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-NutThread-TB-v0",
    entry_point=f"{__name__}:ForgeTBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ForgeTBTaskNutThreadCfg",
        "rl_games_cfg_entry_point": f"{forge_agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)

print(
    f"[tb_env] registered Isaac-Forge-{{PegInsert,GearMesh,NutThread}}-TB-v0 "
    f"(TB_RANDOMIZE_NOISE={TB_RANDOMIZE_NOISE})",
    flush=True,
)
