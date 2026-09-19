# G1b: color-shift robustness eval. Copy of eval_frozen_student.py (DO NOT
# edit that file -- it's the frozen protocol) with ONE addition: a fixed-
# per-run photometric perturbation applied to tp_rgb/wrist_rgb before they
# reach the policy, simulating "deployed against a different fixed color/
# lighting setup" (as opposed to per-step flicker). Purpose: quantify whether
# --img_aug v1 (dataset.py _augment_rgb) training actually buys color
# robustness, by comparing an un-augmented checkpoint (e.g. gate2_seed0)
# against an augmented one (aug_v1) under the SAME injected color shift.
#
# The systematic transform parameters (per-channel gamma, contrast,
# brightness, saturation, channel permutation, polarity) are drawn ONCE per
# run per camera from --color_seed and held fixed for every step/episode;
# only the small per-step sensor-noise term is redrawn every call. This
# mirrors dataset.py's _augment_rgb ranges exactly -- keep the two in sync.

_DESC_ = """Color-shift robustness eval for the FMT student.

Usage (from the IsaacLab repo root):
  ./isaaclab.sh -p ~/forge_ts/src/student/eval_colorshift_student.py --headless \\
      --task Isaac-Forge-PegInsert-TBCamera-v0 \\
      --checkpoint ~/forge_ts/student_ckpts/peg/gate2_seed0/best.pt \\
      --norm_stats ~/forge_ts/student_ckpts/peg/gate2_seed0/norm_stats.npz \\
      --episodes 128 --num_envs 32 --protocol_seed 42 \\
      --fixed_pos_noise_mm 2.5 --dyn_rand on --color_seed 1 --tag colorshift_gate2seed0_cs1
"""

import argparse
import os

os.environ["TB_RANDOMIZE_NOISE"] = "0"  # MUST be set before tb_env/tb_camera_env import

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=_DESC_, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="e.g. Isaac-Forge-PegInsert-TBCamera-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a train_student_bc.py checkpoint (.pt).")
parser.add_argument("--norm_stats", type=str, required=True, help="Path to norm_stats.npz saved alongside the checkpoint.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--protocol_seed", type=int, default=42)
parser.add_argument("--fixed_pos_noise_mm", type=float, default=1.0)
parser.add_argument("--dyn_rand", type=str, choices=["on", "off"], default="on")
parser.add_argument("--color_seed", type=int, required=True, help="fixes the per-run color-shift params")
parser.add_argument("--out_dir", type=str, default="~/forge_ts/eval")
parser.add_argument("--tag", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows -- Kit is up, safe to import isaaclab_tasks/tb_camera_env now."""

import json
import math
import os as _os
import sys
import time

import cv2
import numpy as np
import torch

from isaaclab.utils.seed import configure_seed

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import tb_camera_env  # noqa: E402  (side effect: gym.register the TB-camera envs; safe now, Kit is up)
from student_fmt import StudentFMTConfig, StudentFMTPolicy  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

FORCE_LIMIT_N = 5.0
WRENCH_HORIZON = 32


def wilson95(successes, n):
    """95% Wilson score interval, no scipy. Copied verbatim from eval_frozen.py."""
    if n == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def apply_frozen_dynamics(env_cfg):
    """Identical knob inventory to eval_frozen.py's apply_frozen_dynamics -- keep in sync."""
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


def sample_colorshift_params(rng):
    """Draw ONE fixed systematic transform, same ranges as dataset.py's
    _augment_rgb (keep in sync). Returns a dict consumed by apply_colorshift."""
    return {
        "gamma": rng.uniform(0.5, 2.2, size=3),
        "contrast": rng.uniform(0.6, 1.4),
        "brightness": rng.uniform(0.5, 1.5),
        "sat": 0.0 if rng.random() < 0.15 else rng.uniform(0.0, 1.3),
        "permute": rng.permutation(3) if rng.random() < 0.3 else np.arange(3),
        "polarity_flip": rng.random() < 0.05,
    }


def apply_colorshift(img_uint8, params, noise_rng):
    """img_uint8: (N,H,W,3) uint8, torch.Tensor (GPU or CPU) or numpy array --
    tb_camera_env.get_camera_images() returns GPU torch tensors. Applies the
    FIXED systematic params (same for every call this run) + fresh per-call
    sensor noise; returns a numpy uint8 array (caller re-wraps with
    torch.as_tensor, same as eval_frozen_student.py's build_inputs)."""
    if torch.is_tensor(img_uint8):
        img_uint8 = img_uint8.detach().cpu().numpy()
    x01 = img_uint8.astype(np.float32) / 255.0
    for c in range(3):
        x01[..., c] = x01[..., c] ** params["gamma"][c]
    x = x01 * 255.0
    mean = x.mean(axis=(1, 2), keepdims=True)
    x = (x - mean) * params["contrast"] + mean
    x = x * params["brightness"]
    lum = (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    x = lum + (x - lum) * params["sat"]
    x = x[..., params["permute"]]
    if params["polarity_flip"]:
        x = 255.0 - x
    x = x + noise_rng.normal(0.0, 5.0, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def main():
    ckpt_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    out_dir = os.path.abspath(os.path.expanduser(args_cli.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = args_cli.tag if args_cli.tag is not None else os.path.splitext(os.path.basename(ckpt_path))[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- fixed per-run color-shift params, one set per camera ----
    cs_rng = np.random.default_rng(args_cli.color_seed)
    tp_params = sample_colorshift_params(cs_rng)
    wrist_params = sample_colorshift_params(cs_rng)
    noise_rng = np.random.default_rng(args_cli.color_seed + 100000)
    print(f"[eval_colorshift] tp_params={tp_params} wrist_params={wrist_params}", flush=True)

    # ---- load student model + normalization stats ----
    stats = dict(np.load(os.path.expanduser(args_cli.norm_stats)))
    state_mean = torch.as_tensor(stats["state_mean"], device=device, dtype=torch.float32)
    state_std = torch.as_tensor(stats["state_std"], device=device, dtype=torch.float32)
    wrench_mean = torch.as_tensor(stats["wrench_mean"], device=device, dtype=torch.float32)
    wrench_std = torch.as_tensor(stats["wrench_std"], device=device, dtype=torch.float32)

    ckpt = torch.load(ckpt_path, map_location=device)
    policy = StudentFMTPolicy(StudentFMTConfig(**ckpt["config"])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    # ---- env cfg (all protocol modifications happen on the cfg BEFORE env creation) ----
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.protocol_seed

    noise_m = args_cli.fixed_pos_noise_mm / 1000.0
    env_cfg.obs_rand.fixed_asset_pos = [noise_m, noise_m, noise_m]

    dyn_rand_on = args_cli.dyn_rand == "on"
    if not dyn_rand_on:
        apply_frozen_dynamics(env_cfg)

    import gymnasium as gym

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_env = env.unwrapped  # ForgeTBCameraEnv
    num_envs = raw_env.num_envs
    sim_device = raw_env.device

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
            cap["ever_success"] = raw_env.ep_succeeded.clone().bool()
        return rew

    raw_env._get_rewards = hooked_get_rewards

    configure_seed(args_cli.protocol_seed)

    obs, _ = env.reset()
    wrench_buf = torch.zeros((num_envs, WRENCH_HORIZON, 6), device=sim_device)

    def build_inputs():
        state = raw_env.student_obs.to(torch.float32)
        images = raw_env.get_camera_images()
        tp_shifted = apply_colorshift(images["tp_rgb"], tp_params, noise_rng)
        wrist_shifted = apply_colorshift(images["wrist_rgb"], wrist_params, noise_rng)
        tp_rgb = torch.as_tensor(tp_shifted, device=sim_device)
        wrist_rgb = torch.as_tensor(wrist_shifted, device=sim_device)
        state_n = (state - state_mean) / state_std
        wrench_n = (wrench_buf - wrench_mean) / wrench_std
        return state_n, wrench_n, tp_rgb, wrist_rgb

    def update_wrench_buf():
        new = raw_env.get_wrench_history()
        wrench_buf[:] = torch.cat([wrench_buf[:, 8:], new], dim=1)

    rounds = math.ceil(args_cli.episodes / num_envs)
    records = []
    t_start = time.time()
    steps_per_episode = None

    for round_idx in range(rounds):
        cap["force_sum"] = torch.zeros(num_envs, device=sim_device)
        cap["force_max"] = torch.zeros(num_envs, device=sim_device)
        cap["force_over"] = torch.zeros(num_envs, device=sim_device)
        cap["steps"] = 0
        cap["final_success"] = None
        cap["ever_success"] = None
        wrench_buf.zero_()

        round_done = False
        while not round_done:
            with torch.inference_mode():
                update_wrench_buf()
                state_n, wrench_n, tp_rgb, wrist_rgb = build_inputs()
                actions = policy(state_n, wrench_n, tp_rgb, wrist_rgb)
                obs, _, terminated, truncated, _ = env.step(actions)
                dones = terminated | truncated
                if bool(dones.any()):
                    if not bool(dones.all()):
                        raise RuntimeError(
                            "Protocol violated: envs terminated out of sync (FactoryEnv is timeout-only)."
                        )
                    round_done = True

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
        "color_seed": args_cli.color_seed,
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

    task_short = args_cli.task.replace("Isaac-Forge-", "").replace("-TBCamera-v0", "")
    base = f"{task_short}_{tag}_noise{args_cli.fixed_pos_noise_mm:g}mm_dr{args_cli.dyn_rand}_cs{args_cli.color_seed}_seed{args_cli.protocol_seed}_{int(timestamp)}"
    jsonl_path = os.path.join(out_dir, base + ".jsonl")
    summary_path = os.path.join(out_dir, base + ".json")
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log_line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}] "
        f"task={task_short} tag={tag} seed={args_cli.protocol_seed} color_seed={args_cli.color_seed} "
        f"noise_mm={args_cli.fixed_pos_noise_mm:g} dyn_rand={args_cli.dyn_rand} n={n_eval} sr={sr:.4f} "
        f"ci95=[{lo:.4f},{hi:.4f}] ckpt={ckpt_path}"
    )
    with open(os.path.join(out_dir, "eval_summary.log"), "a") as f:
        f.write(log_line + "\n")

    print("EVAL_SUMMARY " + json.dumps(summary), flush=True)
    print(log_line, flush=True)
    print(f"EVAL_DONE jsonl={jsonl_path} summary={summary_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    import os as _os2
    _os2._exit(0)  # skip Kit teardown hang, see record_labeled_clips.py precedent
