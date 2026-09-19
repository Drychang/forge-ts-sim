"""T-C teacher env for CoRMA/C3: MINIMAL-privileged teacher.

Unlike T-B (fully privileged 64-dim, actor==critic, sees clean fixed_pos so the
noise slot is redundant -- proven useless for the adapter substitution), T-C's
ACTOR sees ONLY the deployable T-A 24-dim noisy obs + the 3-dim noise vector
(27-dim). It does NOT see clean fixed_pos anywhere in the actor obs, so the
noise vector is the ONLY channel to recover the true target frame -> the noise
is genuinely load-bearing, so replacing it at deploy with the adapter's
prediction actually tests online context inference (the CoRMA premise).

Actor obs (27) = T-A obs_order (17) + fixed_pos_obs_noise (3) + prev_actions (7).
Critic state = FORGE default privileged state_order (asymmetric AC, standard;
critic seeing clean state only speeds training, does not affect the actor's
dependence on the noise).

Register: Isaac-Forge-{PegInsert,GearMesh,NutThread}-TC-v0
Train: train_tc.py (== train_tb.py, imports tc_env). TC_RANDOMIZE_NOISE=1
(default) resamples wide per-episode noise for training, like T-B.
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

TC_RANDOMIZE_NOISE = os.environ.get("TC_RANDOMIZE_NOISE", "1") != "0"

# Register the noise key so obs_order can look it up (same pattern as tb_env).
OBS_DIM_CFG["fixed_pos_obs_noise"] = 3
STATE_DIM_CFG["fixed_pos_obs_noise"] = 3

# T-A actor obs_order + the noise vector. prev_actions (7) appended separately
# by _get_observations. Sums to 17 + 3 = 20, +7 = 27-dim actor obs.
TC_OBS_ORDER = [
    "fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel",
    "ft_force", "force_threshold", "fixed_pos_obs_noise",
]


class ForgeTCEnv(ForgeEnv):
    """ForgeEnv variant whose ACTOR obs = T-A noisy obs + noise (minimal privilege)."""

    def randomize_initial_state(self, env_ids):
        super().randomize_initial_state(env_ids)
        if TC_RANDOMIZE_NOISE:
            n = len(env_ids)
            std = torch.rand((n, 1), device=self.device) * self.cfg.tc_noise_std_max
            self.init_fixed_pos_obs_noise[env_ids] = torch.randn((n, 3), device=self.device) * std

    def _get_observations(self):
        # Replicate ForgeEnv._get_observations, but ADD fixed_pos_obs_noise to
        # the ACTOR obs_dict. Actor stays on NOISY variants (never clean
        # fixed_pos); critic keeps the default privileged state_dict.
        obs_dict, state_dict = self._get_factory_obs_state_dict()

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
                "fixed_pos_obs_noise": self.init_fixed_pos_obs_noise,  # the ONLY privileged channel
                "prev_actions": prev_actions,
            }
        )
        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3],
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}


@configclass
class ForgeTCTaskPegInsertCfg(ForgeTaskPegInsertCfg):
    obs_order: list = list(TC_OBS_ORDER)  # 27-dim actor; state_order stays default privileged
    tc_noise_std_max: float = 0.005  # 5mm, matches frozen-eval high end


@configclass
class ForgeTCTaskGearMeshCfg(ForgeTaskGearMeshCfg):
    obs_order: list = list(TC_OBS_ORDER)
    tc_noise_std_max: float = 0.005


@configclass
class ForgeTCTaskNutThreadCfg(ForgeTaskNutThreadCfg):
    obs_order: list = list(TC_OBS_ORDER)
    tc_noise_std_max: float = 0.005


for _gid, _cfg, _yaml in [
    ("Isaac-Forge-PegInsert-TC-v0", "ForgeTCTaskPegInsertCfg", "rl_games_ppo_cfg.yaml"),
    ("Isaac-Forge-GearMesh-TC-v0", "ForgeTCTaskGearMeshCfg", "rl_games_ppo_cfg.yaml"),
    ("Isaac-Forge-NutThread-TC-v0", "ForgeTCTaskNutThreadCfg", "rl_games_ppo_cfg_nut_thread.yaml"),
]:
    gym.register(
        id=_gid,
        entry_point=f"{__name__}:ForgeTCEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:{_cfg}",
            "rl_games_cfg_entry_point": f"{forge_agents.__name__}:{_yaml}",
        },
    )

print(f"[tc_env] registered Isaac-Forge-{{PegInsert,GearMesh,NutThread}}-TC-v0 "
      f"(27-dim actor, TC_RANDOMIZE_NOISE={TC_RANDOMIZE_NOISE})", flush=True)
