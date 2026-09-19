"""CoRMA/C3 Phase 1: collect adapter training data by rolling out T-B.

For each step of a T-B rollout, log the 24-dim DEPLOYABLE T-A-style obs
(the only signals a non-privileged adapter may see); per episode, log the
true init_fixed_pos_obs_noise (3-vec) as the regression LABEL. State-only
(no cameras) so this is fast.

The adapter (Phase 2) learns: sequence of 24-dim deployable obs -> 3-dim
noise. At deploy (Phase 3) T-B consumes the adapter's predicted noise in
its obs[54:57] slot instead of the true value.

24-dim T-A obs layout (forge_env_cfg obs_order + prev_actions), rebuilt from
env internals so no camera env is needed:
  [0:3]  fingertip_pos_rel_fixed = fingertip - (fixed_pos_obs_frame + noise)
  [3:7]  fingertip_quat
  [7:10] ee_linvel_fd
  [10:13] ee_angvel_fd
  [13:16] ft_force (force_sensor_smooth[:,0:3])
  [16]   force_threshold (contact_penalty_thresholds)
  [17:24] prev_actions (self.actions; [3:5] zeroed to match T-B convention)

Launch (from repo root, via eval_frozen_tb-style Kit bring-up):
  see run_collect_adapter.sh
"""
import argparse
import os

os.environ.setdefault("TB_RANDOMIZE_NOISE", "0")  # we drive noise via cfg, not the wide resample

import isaaclab.app  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, help="Isaac-Forge-*-TB-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--fixed_pos_noise_mm", type=float, default=5.0,
                    help="std of per-episode noise; 5mm gives broad 3-vec coverage incl small values")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True, help="output .npz path")
isaaclab.app.AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# defer tb_env import until Kit is up (train_tb.py convention)
_orig_init = isaaclab.app.AppLauncher.__init__
def _patched(self, *a, **k):
    _orig_init(self, *a, **k)
    import tb_env  # noqa: F401
isaaclab.app.AppLauncher.__init__ = _patched

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/ has tb_env.py
app_launcher = isaaclab.app.AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def build_deployable_obs24(e):
    """Rebuild the 24-dim T-A deployable obs from T-B env internals."""
    noisy_fixed = e.fixed_pos_obs_frame + e.init_fixed_pos_obs_noise
    prev = e.actions.clone()
    prev[:, 3:5] = 0.0
    return torch.cat([
        e.fingertip_midpoint_pos - noisy_fixed,      # 3
        e.fingertip_midpoint_quat,                    # 4
        e.ee_linvel_fd,                               # 3
        e.ee_angvel_fd,                               # 3
        e.force_sensor_smooth[:, 0:3],                # 3
        e.contact_penalty_thresholds[:, None],        # 1
        prev,                                         # 7
    ], dim=-1)


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    noise_m = args_cli.fixed_pos_noise_mm / 1000.0
    env_cfg.obs_rand.fixed_asset_pos = [noise_m, noise_m, noise_m]

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    agent_cfg["params"]["config"]["num_actors"] = raw.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(os.path.abspath(os.path.expanduser(args_cli.checkpoint)))
    agent.reset()

    n_envs = raw.num_envs
    sim_device = raw.device

    # Success must be read BEFORE DirectRLEnv._reset_idx re-randomizes (same
    # pitfall documented in eval_frozen.py). Capture inside a _get_rewards hook.
    cap = {}
    _orig_rew = raw._get_rewards
    def _hooked_rew():
        rew = _orig_rew()
        if bool(raw.reset_buf.any()):
            cap["success"] = raw._get_curr_successes(
                success_threshold=raw.cfg_task.success_threshold,
                check_rot=(raw.cfg_task.name == "nut_thread"),
            ).clone()
        return rew
    raw._get_rewards = _hooked_rew

    configure_seed(args_cli.seed)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    rounds = math.ceil(args_cli.episodes / n_envs)
    ep_obs_seqs, ep_noise, ep_success = [], [], []

    for r in range(rounds):
        step_buf = []                                   # list of (n_envs,24)
        noise_this_ep = raw.init_fixed_pos_obs_noise.clone()  # constant per episode
        round_done = False
        while not round_done:
            with torch.inference_mode():
                step_buf.append(build_deployable_obs24(raw).clone())
                a = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
                obs, _, dones, _ = env.step(a)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if bool(dones.any()):
                    round_done = True
                    succ = cap["success"]  # captured pre-reset by the hook
                    if agent.is_rnn and agent.states is not None:
                        for st in agent.states:
                            st[:, dones, :] = 0.0
        seq = torch.stack(step_buf, dim=1)              # (n_envs, T, 24)
        for i in range(n_envs):
            ep_obs_seqs.append(seq[i].cpu().numpy().astype(np.float32))
            ep_noise.append(noise_this_ep[i].cpu().numpy().astype(np.float32))
            ep_success.append(bool(succ[i]))
        print(f"COLLECT_ROUND {r+1}/{rounds} T={seq.shape[1]} eps={len(ep_obs_seqs)} "
              f"succ_rate={np.mean(ep_success):.3f}", flush=True)

    out = os.path.abspath(os.path.expanduser(args_cli.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # episodes share T within a round but T differs across tasks; store as object array
    np.savez_compressed(
        out,
        obs_seqs=np.array(ep_obs_seqs, dtype=object),
        noise=np.stack(ep_noise),
        success=np.array(ep_success),
        noise_mm=args_cli.fixed_pos_noise_mm,
        task=raw.cfg_task.name,
    )
    print(f"COLLECT_DONE saved {len(ep_obs_seqs)} eps -> {out} "
          f"(succ_rate={np.mean(ep_success):.3f}, obs24 sanity dim={ep_obs_seqs[0].shape})", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
