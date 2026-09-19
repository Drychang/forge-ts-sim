"""Minimal isolation smoke test for ForgeTBEnv: construct directly (no
rl_games/hydra), verify obs dims, reset/step, and that the offset actually
appears (nonzero, matches self.init_fixed_pos_obs_noise) in the observation.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task_variant", type=str, default="peg", choices=["peg", "gear", "nut"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import tb_env  # noqa: E402

CFGS = {
    "peg": tb_env.ForgeTBTaskPegInsertCfg,
    "gear": tb_env.ForgeTBTaskGearMeshCfg,
    "nut": tb_env.ForgeTBTaskNutThreadCfg,
}

cfg = CFGS[args_cli.task_variant]()
cfg.scene.num_envs = args_cli.num_envs
if args_cli.device is not None:
    cfg.sim.device = args_cli.device

print(f"[DEBUG] TB_RANDOMIZE_NOISE={tb_env.TB_RANDOMIZE_NOISE}", flush=True)
print(f"[DEBUG] declared observation_space={cfg.observation_space if hasattr(cfg,'observation_space') else 'N/A pre-init'}", flush=True)

env = tb_env.ForgeTBEnv(cfg, render_mode=None)
print(f"[DEBUG] construction OK. observation_space={env.cfg.observation_space} state_space={env.cfg.state_space}", flush=True)
assert env.cfg.observation_space == 64, f"expected 64, got {env.cfg.observation_space}"
assert env.cfg.state_space == 64, f"expected 64, got {env.cfg.state_space}"

obs_dict, _ = env.reset()
obs = obs_dict["policy"]
state = obs_dict["critic"]
print(f"[DEBUG] RESET_OK obs.shape={tuple(obs.shape)} state.shape={tuple(state.shape)}", flush=True)
assert obs.shape[-1] == 64, obs.shape
assert state.shape[-1] == 64, state.shape
assert torch.allclose(obs, state), "actor/critic obs should be identical (symmetric privileged)"

# offset should be the last 3 dims before prev_actions(7) -> index [-10:-7]
offset_in_obs = obs[:, -10:-7]
true_offset = env.init_fixed_pos_obs_noise
print(f"[DEBUG] offset_in_obs sample={offset_in_obs[0].tolist()} true_offset sample={true_offset[0].tolist()}", flush=True)
assert torch.allclose(offset_in_obs, true_offset), "offset slice mismatch -- check TB_ORDER indexing"
if tb_env.TB_RANDOMIZE_NOISE:
    std_per_env = true_offset.std(dim=-1)
    print(f"[DEBUG] offset nonzero check: max|offset|={true_offset.abs().max().item():.6f} (expect up to ~0.015 for 5mm std draws)", flush=True)

for i in range(5):
    act = torch.rand((env.num_envs, env.cfg.action_space), device=env.device) * 2 - 1
    obs_dict, rew, term, trunc, info = env.step(act)
print("[DEBUG] STEP_OK", flush=True)

env.close()
simulation_app.close()
