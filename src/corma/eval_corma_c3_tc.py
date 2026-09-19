"""CoRMA/C3 Phase 3 (T-C teacher variant): frozen-protocol eval of the RMA-style
adaptation baseline. Uses the 27-dim MINIMAL-privileged T-C teacher whose noise
slot [17:20] is load-bearing (no clean fixed_pos bypass), so the adapter
injection genuinely drives behavior -- unlike the T-B variant.

Runs the T-B teacher but replaces its privileged noise slot (obs[54:57]) with
the adapter online prediction from deployable-obs history -- i.e. T-B no
longer sees the true noise, only what a deployable adapter infers. This is the
"in the spirit of CoRMA" baseline: online context inference feeding a
privileged-trained policy, NO vision.

Frozen protocol identical to eval_frozen.py (seed 42, 256 eps, Wilson CI,
_get_rewards success hook, noise/dyn_rand axes). The ONLY change vs T-B eval
is the obs[54:57] overwrite with the adapter prediction.
"""
import argparse
import math
import os

os.environ["TB_RANDOMIZE_NOISE"] = "0"
os.environ["TC_RANDOMIZE_NOISE"] = "0"  # CRITICAL: tc_env reads THIS (not TB_); without it tc_env wide-resamples noise every episode and ignores --fixed_pos_noise_mm

import isaaclab.app  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, help="Isaac-Forge-*-TC-v0")
parser.add_argument("--tb_checkpoint", required=True)
parser.add_argument("--adapter", required=True, help="adapter_best.pt")
parser.add_argument("--adapter_norm", required=True, help="adapter_norm.npz")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--protocol_seed", type=int, default=42)
parser.add_argument("--fixed_pos_noise_mm", type=float, default=1.0)
parser.add_argument("--dyn_rand", choices=["on", "off"], default="on")
parser.add_argument("--out_dir", default=os.path.expanduser("~/forge_ts/eval"))
parser.add_argument("--tag", default=None)
parser.add_argument("--inject_mode", choices=["adapter", "zero", "true"], default="adapter",
                    help="adapter=predicted noise (the baseline); zero/true=validation ablations")
isaaclab.app.AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_here = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, ".."))  # src/ has tb_env
_orig = isaaclab.app.AppLauncher.__init__


def _patched(self, *a, **k):
    _orig(self, *a, **k)
    import tc_env  # noqa: F401


isaaclab.app.AppLauncher.__init__ = _patched

app_launcher = isaaclab.app.AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from adapter_model import CausalNoiseAdapter  # noqa: E402

FORCE_LIMIT_N = 5.0
NOISE_SLICE = slice(17, 20)  # fixed_pos_obs_noise in T-C 27-dim actor obs (17 T-A dims precede it: 3+4+3+3+3+1)


def wilson95(k, n):
    if n == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def apply_frozen_dynamics(env_cfg):
    env_cfg.ctrl.task_prop_gains_noise_level = [0.0] * 6
    env_cfg.ctrl.pos_threshold_noise_level = [0.0] * 3
    env_cfg.ctrl.rot_threshold_noise_level = [0.0] * 3
    lo, hi = env_cfg.ctrl.ema_factor_range
    env_cfg.ctrl.ema_factor_range = [0.5 * (lo + hi)] * 2
    env_cfg.ctrl.default_dead_zone = [0.0] * 6
    env_cfg.events.dead_zone_thresholds = None
    clo, chi = env_cfg.task.contact_penalty_threshold_range
    env_cfg.task.contact_penalty_threshold_range = [0.5 * (clo + chi)] * 2
    env_cfg.events.object_scale_mass.params["mass_distribution_params"] = (0.0, 0.0)
    env_cfg.events.fixed_physics_material.params["static_friction_range"] = (0.75, 0.75)


def build_obs24(e):
    noisy_fixed = e.fixed_pos_obs_frame + e.init_fixed_pos_obs_noise
    prev = e.actions.clone()
    prev[:, 3:5] = 0.0
    return torch.cat([e.fingertip_midpoint_pos - noisy_fixed, e.fingertip_midpoint_quat,
                      e.ee_linvel_fd, e.ee_angvel_fd, e.force_sensor_smooth[:, 0:3],
                      e.contact_penalty_thresholds[:, None], prev], dim=-1)


def main():
    out_dir = os.path.expanduser(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = args_cli.tag or "corma_c3_tc"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    stats = np.load(os.path.expanduser(args_cli.adapter_norm))
    H = int(stats["H"])
    scale = float(stats["target_scale"])
    om = torch.tensor(stats["obs_mean"], device=dev, dtype=torch.float32)
    os_ = torch.tensor(stats["obs_std"], device=dev, dtype=torch.float32)
    ack = torch.load(os.path.expanduser(args_cli.adapter), map_location=dev)
    adapter = CausalNoiseAdapter(H=H, target_scale=scale).to(dev)
    adapter.load_state_dict(ack["model"])
    adapter.eval()

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.protocol_seed
    nm = args_cli.fixed_pos_noise_mm / 1000.0
    env_cfg.obs_rand.fixed_asset_pos = [nm, nm, nm]
    if args_cli.dyn_rand == "off":
        apply_frozen_dynamics(env_cfg)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.protocol_seed
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_act)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    ckpt = os.path.abspath(os.path.expanduser(args_cli.tb_checkpoint))
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = ckpt
    agent_cfg["params"]["config"]["num_actors"] = raw.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(ckpt)
    agent.reset()

    n_envs = raw.num_envs
    sim_dev = raw.device
    cap = {}
    orig_rew = raw._get_rewards

    def hooked():
        rew = orig_rew()
        f = torch.linalg.vector_norm(raw.force_sensor_smooth[:, 0:3], dim=1)
        cap["fs"] += f
        torch.maximum(cap["fm"], f, out=cap["fm"])
        cap["fo"] += (f > FORCE_LIMIT_N).float()
        cap["steps"] += 1
        if bool(raw.reset_buf.any()):
            cap["succ"] = raw._get_curr_successes(
                success_threshold=raw.cfg_task.success_threshold,
                check_rot=(raw.cfg_task.name == "nut_thread")).clone()
        return rew

    raw._get_rewards = hooked

    configure_seed(args_cli.protocol_seed)
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    hist = torch.zeros(n_envs, H, 24, device=dev)
    hlen = torch.zeros(n_envs, dtype=torch.long, device=dev)
    rounds = math.ceil(args_cli.episodes / n_envs)
    records = []
    t0 = time.time()
    spe = None

    for r in range(rounds):
        cap.update(fs=torch.zeros(n_envs, device=sim_dev), fm=torch.zeros(n_envs, device=sim_dev),
                   fo=torch.zeros(n_envs, device=sim_dev), steps=0, succ=None, zmag=0.0, tmag=0.0, zn=0)
        done = False
        while not done:
            with torch.inference_mode():
                o24 = build_obs24(raw).to(dev)
                hist[:, :-1] = hist[:, 1:].clone()
                hist[:, -1] = (o24 - om) / os_
                hlen = torch.clamp(hlen + 1, max=H)
                mask = torch.arange(H, device=dev)[None, :] < (H - hlen)[:, None]
                z_mm = adapter(hist, mask)
                z_m = (z_mm / scale).to(obs.dtype)
                if args_cli.inject_mode == "zero":
                    z_m = torch.zeros_like(z_m)
                elif args_cli.inject_mode == "true":
                    z_m = raw.init_fixed_pos_obs_noise.to(obs.device, obs.dtype)
                obs = obs.clone()
                obs[:, NOISE_SLICE] = z_m
                cap["zmag"] += z_m.abs().mean().item() * 1000.0  # injected |noise| mm
                cap["tmag"] += raw.init_fixed_pos_obs_noise.abs().mean().item() * 1000.0
                cap["zn"] += 1
                a = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
                obs, _, dones, _ = env.step(a)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if bool(dones.any()):
                    done = True
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0
                    hist.zero_()
                    hlen.zero_()
        spe = cap["steps"]
        succ = cap["succ"].cpu()
        mf = (cap["fs"] / spe).cpu()
        mx = cap["fm"].cpu()
        fo = (cap["fo"] / spe).cpu()
        for i in range(n_envs):
            records.append(dict(success=bool(succ[i]), mean_force_n=float(mf[i]),
                                max_force_n=float(mx[i]), frac_over_5n=float(fo[i])))
        sr_round = float(np.mean([x["success"] for x in records[-n_envs:]]))
        print(f"EVAL_ROUND {r+1}/{rounds} steps={spe} sr={sr_round:.4f} "
              f"inj_mag_mm={cap['zmag']/max(cap['zn'],1):.3f} true_mag_mm={cap['tmag']/max(cap['zn'],1):.3f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    n = len(records)
    k = sum(x["success"] for x in records)
    sr = k / n
    lo, hi = wilson95(k, n)
    summary = dict(corma_c3=True, task=args_cli.task, tb_checkpoint=ckpt,
                   adapter=os.path.abspath(os.path.expanduser(args_cli.adapter)),
                   tag=tag, protocol_seed=args_cli.protocol_seed, episodes=n, num_envs=n_envs,
                   steps_per_episode=spe, sr=sr, wilson95_lo=lo, wilson95_hi=hi,
                   mean_force_n_avg=float(np.mean([x["mean_force_n"] for x in records])),
                   max_force_n_avg=float(np.mean([x["max_force_n"] for x in records])),
                   frac_over_5n_avg=float(np.mean([x["frac_over_5n"] for x in records])),
                   noise_mm=args_cli.fixed_pos_noise_mm, dyn_rand=args_cli.dyn_rand, timestamp=time.time())
    print("EVAL_SUMMARY " + json.dumps(summary), flush=True)
    with open(os.path.join(out_dir, f"corma_c3_{tag}.json"), "w") as f:
        json.dump(summary, f)
    print(f"EVAL_DONE sr={sr:.4f} ci=[{lo:.3f},{hi:.3f}] noise={args_cli.fixed_pos_noise_mm}mm", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
