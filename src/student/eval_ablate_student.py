# T3: modality-ablation eval. Copy of eval_frozen_student.py (DO NOT edit that
# file -- frozen protocol) with ONE addition: --ablate zeroes the same modality
# that was zeroed during that checkpoint's training (train_student_bc.py
# --ablate). Zeroing semantics MUST match training: wrench is zeroed AFTER
# normalization (zero-then-normalize would inject a -mean/std constant bias,
# not "no information"); vision is zeroed as uint8 black images. A checkpoint
# trained with --ablate X must be evaluated with --ablate X -- evaluating an
# ablated checkpoint with live modalities (or vice versa) breaks the input
# contract and produces meaningless numbers.

_DESC_ = """Modality-ablation eval for the FMT student (T3).

Usage (from the IsaacLab repo root):
  ./isaaclab.sh -p ~/forge_ts/src/student/eval_ablate_student.py --headless \\
      --task Isaac-Forge-PegInsert-TBCamera-v0 \\
      --checkpoint ~/forge_ts/student_ckpts/peg/ablate_vision/best.pt \\
      --norm_stats ~/forge_ts/student_ckpts/peg/ablate_vision/norm_stats.npz \\
      --episodes 256 --num_envs 32 --protocol_seed 42 \\
      --fixed_pos_noise_mm 2.5 --dyn_rand on --ablate vision --tag ablate_vision
"""

import argparse
import os

os.environ["TB_RANDOMIZE_NOISE"] = "0"  # MUST be set before tb_env/tb_camera_env import

from isaaclab.app import AppLauncher

# Modality ablation sets. Expressed as sets so the legacy flag names ("both" meaning
# vision+force off, i.e. state-only) keep their exact historical meaning while the new
# state-off cells slot in beside them. Anything trained under a name MUST be evaluated
# under the same name -- eval_ablate_student.py imports this same table.
ABLATE_SETS = {
    "none":         frozenset(),
    "vision":       frozenset({"vision"}),                  # state + force
    "force":        frozenset({"force"}),                   # state + vision
    "both":         frozenset({"vision", "force"}),         # state only   (legacy name)
    "state":        frozenset({"state"}),                   # vision + force
    "state_force":  frozenset({"state", "force"}),          # vision only
    "state_vision": frozenset({"state", "vision"}),         # force only
}
ABLATE_CHOICES = list(ABLATE_SETS)


parser = argparse.ArgumentParser(description=_DESC_, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="e.g. Isaac-Forge-PegInsert-TBCamera-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a train_student_bc.py checkpoint (.pt).")
parser.add_argument("--norm_stats", type=str, required=True, help="Path to norm_stats.npz saved alongside the checkpoint.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--protocol_seed", type=int, default=42)
parser.add_argument("--fixed_pos_noise_mm", type=float, default=1.0)
parser.add_argument("--blank_cam", type=str, choices=["none", "wrist", "tp"], default="none",
                    help="zero ONE camera at eval time, on a model trained with both. "
                         "--ablate vision zeroes both and cannot express a single bad mount. "
                         "Applied after the --ablate zeroing, so 'vision' still wins.")
parser.add_argument("--dyn_rand", type=str, choices=["on", "off"], default="on")
parser.add_argument("--ablate", type=str, choices=ABLATE_CHOICES, required=True,
                    help="must match the flag the checkpoint was TRAINED with")
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


def main():
    ckpt_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    out_dir = os.path.abspath(os.path.expanduser(args_cli.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = args_cli.tag if args_cli.tag is not None else os.path.splitext(os.path.basename(ckpt_path))[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = dict(np.load(os.path.expanduser(args_cli.norm_stats)))
    state_mean = torch.as_tensor(stats["state_mean"], device=device, dtype=torch.float32)
    state_std = torch.as_tensor(stats["state_std"], device=device, dtype=torch.float32)
    wrench_mean = torch.as_tensor(stats["wrench_mean"], device=device, dtype=torch.float32)
    wrench_std = torch.as_tensor(stats["wrench_std"], device=device, dtype=torch.float32)

    ckpt = torch.load(ckpt_path, map_location=device)
    policy = StudentFMTPolicy(StudentFMTConfig(**ckpt["config"])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

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
        tp_rgb = torch.as_tensor(images["tp_rgb"], device=sim_device)
        wrist_rgb = torch.as_tensor(images["wrist_rgb"], device=sim_device)
        state_n = (state - state_mean) / state_std
        wrench_n = (wrench_buf - wrench_mean) / wrench_std
        # ablation zeroing -- same semantics as train_student_bc.py --ablate:
        # wrench AFTER normalization, images as black uint8
        _ab = ABLATE_SETS[args_cli.ablate]
        if "state" in _ab:
            state_n = torch.zeros_like(state_n)
        if "force" in _ab:
            wrench_n = torch.zeros_like(wrench_n)
        if "vision" in _ab:
            tp_rgb = torch.zeros_like(tp_rgb)
            wrist_rgb = torch.zeros_like(wrist_rgb)
        # One-camera blanking, after the ablation so --ablate vision still dominates.
        if args_cli.blank_cam == "wrist":
            wrist_rgb = torch.zeros_like(wrist_rgb)
        elif args_cli.blank_cam == "tp":
            tp_rgb = torch.zeros_like(tp_rgb)
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
        "ablate": args_cli.ablate,
        "blank_cam": args_cli.blank_cam,
        # Record the camera perturbation this run was produced under. Without it a
        # jittered result is indistinguishable from a clean one once it is in a table.
        "cam_jitter_pos_mm": float(os.environ.get("TB_CAM_JITTER_POS_MM", "0") or 0),
        "cam_jitter_rot_deg": float(os.environ.get("TB_CAM_JITTER_ROT_DEG", "0") or 0),
        "cam_jitter_seed": int(os.environ.get("TB_CAM_JITTER_SEED", "0") or 0),
        "cam_jitter_cams": os.environ.get("TB_CAM_JITTER_CAMS", "") or "all",
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
    base = (f"{task_short}_{tag}_noise{args_cli.fixed_pos_noise_mm:g}mm_dr{args_cli.dyn_rand}_"
            f"ab{args_cli.ablate}_seed{args_cli.protocol_seed}_{int(timestamp)}")
    jsonl_path = os.path.join(out_dir, base + ".jsonl")
    summary_path = os.path.join(out_dir, base + ".json")
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log_line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}] "
        f"task={task_short} tag={tag} seed={args_cli.protocol_seed} ablate={args_cli.ablate} "
        f"noise_mm={args_cli.fixed_pos_noise_mm:g} dyn_rand={args_cli.dyn_rand} "
        f"blank_cam={args_cli.blank_cam} n={n_eval} sr={sr:.4f} "
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
    os._exit(0)  # skip Kit teardown hang, see record_labeled_clips.py precedent
