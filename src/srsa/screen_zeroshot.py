# SRSA ZERO-SHOT SCREENING: copy of eval_frozen.py (frozen original untouched)
# with three changes: (a) registers the SRSA adapter envs; (b) agent cfg is
# loaded from SRSA's rl_games_ppo_sil_cfg.yaml with algo swapped to the
# standard a2c_continuous player (SIL is training-only; the model class
# continuous_a2c_logstd is identical, so their .pth restores 1:1);
# (c) --agent_yaml CLI arg. Protocol mechanics are byte-identical.
# Frozen evaluation protocol for FORGE tasks (paper-wide, do not modify once results are logged).
#
# Runs a trained rl_games checkpoint on an Isaac-Forge-*-Direct-v0 task under a
# reproducible protocol and writes per-episode JSONL + a summary JSON + one line
# appended to <out_dir>/eval_summary.log.

_DESC_ = """Frozen evaluation protocol for FORGE tasks.

Usage (from the IsaacLab repo root):
  ./isaaclab.sh -p ~/forge_ts/src/eval_frozen.py --headless \\
      --task Isaac-Forge-PegInsert-Direct-v0 \\
      --checkpoint ~/forge_ts/checkpoints/peg.pth \\
      --episodes 256 --num_envs 128 --protocol_seed 42 \\
      --fixed_pos_noise_mm 1.0 --dyn_rand on --tag teacher_s42

WHAT "FROZEN" MEANS (verified against IsaacLab @ 4927517):
  Same protocol_seed + same num_envs + same episode schedule (episodes -> rounds)
  => identical initial-state sequence across every evaluated policy. Mechanism:
  1. env_cfg.seed = protocol_seed. DirectRLEnv.__init__ (direct_rl_env.py L98-100)
     calls self.seed(cfg.seed) -> isaaclab.utils.seed.configure_seed WHEN cfg.seed
     is not None (silently skipped + a warning logged otherwise) -- harmless here
     since protocol_seed is always a real int. configure_seed seeds random / numpy /
     torch CPU / torch CUDA (manual_seed + manual_seed_all) global generators.
  2. Verified in factory_env.py randomize_initial_state and forge_env.py _reset_idx:
     ALL initial-state randomization uses global-generator torch.rand/randn (no
     per-env generator, no env.seed path). So the reset draws are a function of
     the global RNG stream position only.
  3. We re-seed via configure_seed(protocol_seed) AFTER the rl_games player is
     created/restored and IMMEDIATELY BEFORE the first env.reset(). Network
     construction consumes RNG (weight init before checkpoint load), so without
     this the stream position at the first reset would depend on the policy
     architecture. With it, reset #0 draws from a fixed stream position.
  4. Per-step RNG draw counts are policy-independent (FORGE obs-noise draws in
     _compute_intermediate_values are fixed-shape every physics step; the
     deterministic player does not sample actions), so reset draws for round k
     happen at the same stream position for every policy.
  Caveat (honest): GPU PhysX and cuDNN are not bitwise deterministic run-to-run.
  The protocol freezes the COMMANDED randomization sequence (fixed-asset pose,
  hand pose, in-gripper pose, obs-noise offsets, dynamics parameters); residual
  physics jitter during the reset settling remains. Using one protocol_seed for
  all policies makes comparisons paired at the distribution level.

SUCCESS METRIC:
  Per-env success at the FINAL episode step, computed by the env's own
  _get_curr_successes (same definition as the env-logged extras["successes"]).
  DirectRLEnv.step calls _get_dones -> _get_rewards -> _reset_idx -> _get_obs,
  i.e. by the time env.step returns at a timeout step the env has already been
  re-randomized. Therefore we capture successes inside a wrapper around
  _get_rewards (runs after reset_buf is set, before _reset_idx). We also record
  ever_success from env.ep_succeeded (success at any step during the episode).
  Termination: FactoryEnv._get_dones is timeout-only and identical for all envs,
  so all envs in a round finish simultaneously (verified factory_env.py).

WRENCH METRIC:
  ||F|| from env.force_sensor_smooth[:, 0:3] (forge_env.py, EMA-smoothed force
  sensor in the noisy fixed-asset frame), accumulated per policy step inside the
  same _get_rewards wrapper (pre-reset, so the final step is included).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=_DESC_, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="e.g. Isaac-Forge-PegInsert-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to rl_games .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=256, help="Evaluated count is rounded up to a multiple of num_envs.")
parser.add_argument("--protocol_seed", type=int, default=42)
parser.add_argument(
    "--fixed_pos_noise_mm",
    type=float,
    default=1.0,
    help="Std (mm, per axis) of the per-episode fixed-asset position observation noise. 1.0 = FORGE default; 0 disables.",
)
parser.add_argument("--dyn_rand", type=str, choices=["on", "off"], default="on")
parser.add_argument("--out_dir", type=str, default="~/forge_ts/eval")
parser.add_argument("--tag", type=str, default=None, help="Free label for the summary line; defaults to checkpoint stem.")
parser.add_argument("--agent_yaml", type=str,
                    default="/extra_home3/user/SRSA/source/SRSA/SRSA/tasks/direct/srsa/agents/rl_games_ppo_sil_cfg.yaml")
# append AppLauncher cli args (--headless, --device, ...)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srsa_forge_env  # noqa: E402,F401  (registers Isaac-SRSA-Forge-*-v0)
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

FORCE_LIMIT_N = 5.0


def wilson95(successes, n):
    """95% Wilson score interval, no scipy."""
    if n == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def apply_frozen_dynamics(env_cfg):
    """Freeze every dynamics-randomization knob at its nominal value (--dyn_rand off).

    Knob inventory from forge_env.py _reset_idx + forge_env_cfg.py EventCfg +
    factory_env.py randomize_initial_state:
      [frozen here]
      1. ctrl.task_prop_gains_noise_level  -> 0 (prop gains = default_task_prop_gains)
      2. ctrl.pos_threshold_noise_level    -> 0
      3. ctrl.rot_threshold_noise_level    -> 0
      4. ctrl.ema_factor_range             -> collapsed to range mean
      5. ctrl.default_dead_zone            -> 0 (dead zone draw is rand*default_dead_zone)
      6. events.dead_zone_thresholds       -> None (interval EventTerm re-randomizing dead
         zones every 2 s; EventManager._prepare_terms skips None term cfgs)
      7. task.contact_penalty_threshold_range -> collapsed to mean (feeds the
         force_threshold obs and reward only; frozen for determinism)
      8. events.object_scale_mass mass_distribution_params -> (0, 0) (held-asset mass)
      9. events.fixed_physics_material static_friction_range (0.25, 1.25) -> (0.75, 0.75)
         (startup-mode; held/robot material events are already constant)
      [kept in ALL conditions -- task distribution / observation model, not dyn rand]
      - fixed-asset init pose noise, hand init pose noise, held-asset-in-gripper noise
      - obs noises: fingertip pos/rot, ft_force (FORGE defaults)
      - init_fixed_pos_obs_noise (controlled by --fixed_pos_noise_mm)
      - flip_quats quaternion-sign augmentation (obs-space double-cover augmentation)
    """
    env_cfg.ctrl.task_prop_gains_noise_level = [0.0] * 6
    env_cfg.ctrl.pos_threshold_noise_level = [0.0] * 3
    env_cfg.ctrl.rot_threshold_noise_level = [0.0] * 3
    ema_lo, ema_hi = env_cfg.ctrl.ema_factor_range
    ema_nominal = 0.5 * (ema_lo + ema_hi)
    env_cfg.ctrl.ema_factor_range = [ema_nominal, ema_nominal]
    env_cfg.ctrl.default_dead_zone = [0.0] * 6
    env_cfg.events.dead_zone_thresholds = None
    c_lo, c_hi = env_cfg.task.contact_penalty_threshold_range
    c_nominal = 0.5 * (c_lo + c_hi)
    env_cfg.task.contact_penalty_threshold_range = [c_nominal, c_nominal]
    env_cfg.events.object_scale_mass.params["mass_distribution_params"] = (0.0, 0.0)
    env_cfg.events.fixed_physics_material.params["static_friction_range"] = (0.75, 0.75)


def main():
    ckpt_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    out_dir = os.path.abspath(os.path.expanduser(args_cli.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = args_cli.tag if args_cli.tag is not None else os.path.splitext(os.path.basename(ckpt_path))[0]

    # ---- env cfg (all protocol modifications happen on the cfg BEFORE env creation) ----
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # seed like train.py: env_cfg.seed set before env creation; DirectRLEnv.__init__
    # then seeds torch/np/random via configure_seed (see module docstring).
    env_cfg.seed = args_cli.protocol_seed

    # noise axis: per-episode fixed-asset position OBS noise (factory_env.py
    # randomize_initial_state step 1.e draws init_fixed_pos_obs_noise as
    # randn @ diag(cfg.obs_rand.fixed_asset_pos), held constant for the episode).
    noise_m = args_cli.fixed_pos_noise_mm / 1000.0
    env_cfg.obs_rand.fixed_asset_pos = [noise_m, noise_m, noise_m]

    dyn_rand_on = args_cli.dyn_rand == "on"
    if not dyn_rand_on:
        apply_frozen_dynamics(env_cfg)

    # ---- agent cfg + env wiring: mirrors target-version play.py ----
    import yaml as _yaml
    with open(os.path.expanduser(args_cli.agent_yaml)) as f:
        agent_cfg = _yaml.safe_load(f)
    # SIL is a TRAINING algo; the standard continuous player restores the same
    # model. central_value nets are train-only -- drop so the player skips them.
    agent_cfg["params"]["algo"]["name"] = "a2c_continuous"
    agent_cfg["params"]["config"].pop("central_value_config", None)
    # rl_games Runner.load seeds torch from params.seed (time-based if unset) -- pin it.
    agent_cfg["params"]["seed"] = args_cli.protocol_seed
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = ckpt_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(ckpt_path)
    agent.reset()

    raw_env = env.unwrapped  # ForgeEnv
    num_envs = raw_env.num_envs
    sim_device = raw_env.device

    # ---- per-step capture hook ----
    # DirectRLEnv.step order: _get_dones -> _get_rewards -> _reset_idx -> _get_observations.
    # After env.step() returns at a timeout step the sim state is already re-randomized,
    # so final-step successes and forces MUST be read inside the rewards call (reset_buf
    # is set, reset has not happened yet). The hook is capture-only: no RNG, no behavior change.
    cap = {}
    orig_get_rewards = raw_env._get_rewards

    def hooked_get_rewards():
        rew = orig_get_rewards()
        f_norm = torch.linalg.vector_norm(raw_env.force_sensor_smooth[:, 0:3], dim=1)
        cap["force_sum"] += f_norm
        torch.maximum(cap["force_max"], f_norm, out=cap["force_max"])
        cap["force_over"] += (f_norm > FORCE_LIMIT_N).float()
        cap["steps"] += 1
        if bool(raw_env.reset_buf.any()):
            check_rot = raw_env.cfg_task.name == "nut_thread"
            cap["final_success"] = raw_env._get_curr_successes(
                success_threshold=raw_env.cfg_task.success_threshold, check_rot=check_rot
            ).clone()
            # ep_succeeded was just updated for the final step by _log_factory_metrics
            # inside orig_get_rewards; cleared only at the next _pre_physics_step.
            cap["ever_success"] = raw_env.ep_succeeded.clone().bool()
        return rew

    raw_env._get_rewards = hooked_get_rewards

    # ---- FREEZE POINT: pin the RNG stream right before the first reset ----
    configure_seed(args_cli.protocol_seed)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    # rl_games player: enable batched-obs mode, then init RNN states (2-layer LSTM 1024).
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    rounds = math.ceil(args_cli.episodes / num_envs)
    records = []
    t_start = time.time()
    steps_per_episode = None

    for round_idx in range(rounds):
        # fresh accumulators OUTSIDE inference mode so in-place updates in the hook are legal
        cap["force_sum"] = torch.zeros(num_envs, device=sim_device)
        cap["force_max"] = torch.zeros(num_envs, device=sim_device)
        cap["force_over"] = torch.zeros(num_envs, device=sim_device)
        cap["steps"] = 0
        cap["final_success"] = None
        cap["ever_success"] = None

        round_done = False
        while not round_done:
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=True)
                obs, _, dones, _ = env.step(actions)
                if bool(dones.any()):
                    if not bool(dones.all()):
                        raise RuntimeError(
                            "Protocol violated: envs terminated out of sync (FactoryEnv is timeout-only)."
                        )
                    round_done = True
                    # reset LSTM states at the episode boundary (dones is a bool mask)
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0

        if cap["final_success"] is None:
            raise RuntimeError("Success capture hook never fired at episode end.")
        n_steps = cap["steps"]
        steps_per_episode = n_steps
        final_s = cap["final_success"].cpu()
        ever_s = cap["ever_success"].cpu()
        mean_f = (cap["force_sum"] / n_steps).cpu()
        max_f = cap["force_max"].cpu()
        frac_over = (cap["force_over"] / n_steps).cpu()

        for env_idx in range(num_envs):
            records.append({
                "episode_idx": round_idx * num_envs + env_idx,
                "round": round_idx,
                "env_idx": env_idx,
                "success": bool(final_s[env_idx]),
                "ever_success": bool(ever_s[env_idx]),
                "mean_force_n": round(float(mean_f[env_idx]), 4),
                "max_force_n": round(float(max_f[env_idx]), 4),
                "frac_over_5n": round(float(frac_over[env_idx]), 4),
            })
        sr_round = sum(r["success"] for r in records[-num_envs:]) / num_envs
        print(
            f"EVAL_ROUND {round_idx + 1}/{rounds} steps={n_steps} sr_round={sr_round:.4f} "
            f"elapsed={time.time() - t_start:.0f}s",
            flush=True,
        )

    # ---- aggregate + write outputs ----
    n_eval = len(records)
    n_success = sum(r["success"] for r in records)
    sr = n_success / n_eval
    lo, hi = wilson95(n_success, n_eval)
    sr_ever = sum(r["ever_success"] for r in records) / n_eval

    timestamp = time.time()
    summary = {
        "task": args_cli.task,
        "checkpoint": ckpt_path,
        "tag": tag,
        "protocol_seed": args_cli.protocol_seed,
        "episodes_requested": args_cli.episodes,
        "episodes": n_eval,
        "num_envs": num_envs,
        "rounds": rounds,
        "steps_per_episode": steps_per_episode,
        "sr": sr,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "sr_ever": sr_ever,
        "mean_force_n_avg": sum(r["mean_force_n"] for r in records) / n_eval,
        "max_force_n_avg": sum(r["max_force_n"] for r in records) / n_eval,
        "max_force_n_max": max(r["max_force_n"] for r in records),
        "frac_over_5n_avg": sum(r["frac_over_5n"] for r in records) / n_eval,
        "noise_mm": args_cli.fixed_pos_noise_mm,
        "dyn_rand": args_cli.dyn_rand,
        "timestamp": timestamp,
    }

    task_short = args_cli.task.replace("Isaac-Forge-", "").replace("-Direct-v0", "")
    base = f"{task_short}_{tag}_noise{args_cli.fixed_pos_noise_mm:g}mm_dr{args_cli.dyn_rand}_seed{args_cli.protocol_seed}_{int(timestamp)}"
    jsonl_path = os.path.join(out_dir, base + ".jsonl")
    summary_path = os.path.join(out_dir, base + ".json")
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log_line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}] "
        f"task={task_short} tag={tag} seed={args_cli.protocol_seed} noise_mm={args_cli.fixed_pos_noise_mm:g} "
        f"dyn_rand={args_cli.dyn_rand} n={n_eval} sr={sr:.4f} ci95=[{lo:.4f},{hi:.4f}] "
        f"F_mean={summary['mean_force_n_avg']:.2f}N F_max_avg={summary['max_force_n_avg']:.2f}N "
        f"frac>5N={summary['frac_over_5n_avg']:.3f} ckpt={ckpt_path}"
    )
    with open(os.path.join(out_dir, "eval_summary.log"), "a") as f:
        f.write(log_line + "\n")

    print("EVAL_SUMMARY " + json.dumps(summary), flush=True)
    print(log_line, flush=True)
    print(f"EVAL_DONE jsonl={jsonl_path} summary={summary_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
