"""Record labeled success/failure demo clips for a T-A (rl_games) or student
(StudentFMTPolicy) checkpoint under a given noise/dyn_rand condition.

Adapted from eval_frozen.py / eval_frozen_student.py: same env-cfg protocol
setup (seed, noise, dyn_rand knobs) and the same reward-hook success-capture
trick (DirectRLEnv.step order is _get_dones -> _get_rewards -> _reset_idx, so
final-step success/state must be read inside a wrapped _get_rewards, before
the post-timeout reset overwrites it). num_envs=1 so the rendered viewport
shows exactly one clear episode. Runs sequential episodes, renders every
step via env.render() (rgb_array), overlays a label, and keeps only the
first --n_success successes and --n_fail failures as separate mp4 files;
everything else is discarded (not written to disk).

usage (T-A):
  ./isaaclab.sh -p record_labeled_clips.py --headless \
    --task Isaac-Forge-PegInsert-Direct-v0 --model_type ta \
    --checkpoint .../ta_n5_peg_s0/nn/last_..pth \
    --fixed_pos_noise_mm 5 --dyn_rand on \
    --label_model "Noise-augT-A" --label_task PegInsert \
    --out_dir ~/forge_ts/videos

usage (student):
  ./isaaclab.sh -p record_labeled_clips.py --headless \
    --task Isaac-Forge-PegInsert-TBCamera-v0 --model_type student \
    --checkpoint ~/forge_ts/student_ckpts/peg/gate2_seed0/best.pt \
    --norm_stats ~/forge_ts/student_ckpts/peg/gate2_seed0/norm_stats.npz \
    --fixed_pos_noise_mm 5 --dyn_rand on \
    --label_model Student --label_task PegInsert \
    --out_dir ~/forge_ts/videos
"""
import argparse
import math
import os

os.environ.setdefault("TB_RANDOMIZE_NOISE", "0")  # harmless for T-A, required before tb_camera_env import

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--model_type", choices=["ta", "student"], required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--norm_stats", default=None, help="student only")
parser.add_argument("--fixed_pos_noise_mm", type=float, required=True)
parser.add_argument("--dyn_rand", choices=["on", "off"], default="on")
parser.add_argument("--protocol_seed", type=int, default=42)
parser.add_argument("--n_success", type=int, default=3)
parser.add_argument("--n_fail", type=int, default=3)
parser.add_argument("--max_episodes", type=int, default=60)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--label_model", required=True)
parser.add_argument("--label_task", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows -- Kit is up."""

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

FORCE_LIMIT_N = 5.0
VIEWER_EYE = (1.0, 0.0, 0.4)
VIEWER_LOOKAT = (0.4, 0.0, 0.05)


def apply_frozen_dynamics(env_cfg):
    """Verbatim from eval_frozen.py -- keep in sync if that changes."""
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


def overlay_label(frame_rgb, header, outcome, step_idx, force_n):
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.putText(frame, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"step={step_idx} |F|={force_n:.1f}N", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    color = (0, 200, 0) if outcome == "SUCCESS" else (0, 0, 220)
    (tw, th), _ = cv2.getTextSize(outcome, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.rectangle(frame, (w - tw - 24, h - th - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, outcome, (w - tw - 12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
    return frame


def write_clip(path, frames, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def build_env_cfg():
    num_envs = 1
    env_cfg = parse_env_cfg(args_cli.task, num_envs=num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.protocol_seed
    noise_m = args_cli.fixed_pos_noise_mm / 1000.0
    env_cfg.obs_rand.fixed_asset_pos = [noise_m, noise_m, noise_m]
    if args_cli.dyn_rand == "off":
        apply_frozen_dynamics(env_cfg)
    env_cfg.viewer.eye = VIEWER_EYE
    env_cfg.viewer.lookat = VIEWER_LOOKAT
    env_cfg.viewer.resolution = (1280, 720)
    return env_cfg


def run_ta():
    import gymnasium as gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = build_env_cfg()
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.protocol_seed
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    wrapped = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped})

    ckpt_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = ckpt_path
    agent_cfg["params"]["config"]["num_actors"] = 1

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(ckpt_path)
    agent.reset()

    raw_env = env.unwrapped
    sim_device = raw_env.device

    cap = {}
    orig_get_rewards = raw_env._get_rewards

    def hooked_get_rewards():
        rew = orig_get_rewards()
        f_norm = torch.linalg.vector_norm(raw_env.force_sensor_smooth[:, 0:3], dim=1)
        cap["force_now"] = float(f_norm[0])
        if bool(raw_env.reset_buf.any()):
            check_rot = raw_env.cfg_task.name == "nut_thread"
            cap["final_success"] = bool(raw_env._get_curr_successes(
                success_threshold=raw_env.cfg_task.success_threshold, check_rot=check_rot
            )[0])
        return rew

    raw_env._get_rewards = hooked_get_rewards

    configure_seed(args_cli.protocol_seed)
    obs = wrapped.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    def step_fn():
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs_holder[0])
            actions = agent.get_action(obs_t, is_deterministic=True)
            new_obs, _, dones, _ = wrapped.step(actions)
            if bool(dones.any()) and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            obs_holder[0] = new_obs
            return bool(dones.any())

    obs_holder = [obs]
    return raw_env, cap, step_fn


def run_student():
    import gymnasium as gym
    import numpy as np
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "student"))
    import tb_camera_env  # noqa: F401  side-effect gym.register
    from student_fmt import StudentFMTConfig, StudentFMTPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = dict(np.load(os.path.expanduser(args_cli.norm_stats)))
    state_mean = torch.as_tensor(stats["state_mean"], device=device, dtype=torch.float32)
    state_std = torch.as_tensor(stats["state_std"], device=device, dtype=torch.float32)
    wrench_mean = torch.as_tensor(stats["wrench_mean"], device=device, dtype=torch.float32)
    wrench_std = torch.as_tensor(stats["wrench_std"], device=device, dtype=torch.float32)

    ckpt = torch.load(os.path.abspath(os.path.expanduser(args_cli.checkpoint)), map_location=device)
    policy = StudentFMTPolicy(StudentFMTConfig(**ckpt["config"])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    env_cfg = build_env_cfg()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = env.unwrapped
    sim_device = raw_env.device
    WRENCH_HORIZON = 32

    cap = {}
    orig_get_rewards = raw_env._get_rewards

    def hooked_get_rewards():
        rew = orig_get_rewards()
        f_norm = torch.linalg.vector_norm(raw_env.force_sensor_smooth[:, 0:3], dim=1)
        cap["force_now"] = float(f_norm[0])
        if bool(raw_env.reset_buf.any()):
            check_rot = raw_env.cfg_task.name == "nut_thread"
            cap["final_success"] = bool(raw_env._get_curr_successes(
                success_threshold=raw_env.cfg_task.success_threshold, check_rot=check_rot
            )[0])
        return rew

    raw_env._get_rewards = hooked_get_rewards

    configure_seed(args_cli.protocol_seed)
    env.reset()
    wrench_buf = torch.zeros((1, WRENCH_HORIZON, 6), device=sim_device)

    def step_fn():
        with torch.inference_mode():
            new = raw_env.get_wrench_history()
            wrench_buf[:] = torch.cat([wrench_buf[:, 8:], new], dim=1)
            state = raw_env.student_obs.to(torch.float32)
            images = raw_env.get_camera_images()
            tp_rgb = torch.as_tensor(images["tp_rgb"], device=sim_device)
            wrist_rgb = torch.as_tensor(images["wrist_rgb"], device=sim_device)
            state_n = (state - state_mean) / state_std
            wrench_n = (wrench_buf - wrench_mean) / wrench_std
            actions = policy(state_n, wrench_n, tp_rgb, wrist_rgb)
            _, _, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated
            return bool(dones.any())

    def reset_wrench_at_boundary():
        wrench_buf.zero_()

    return raw_env, cap, step_fn, reset_wrench_at_boundary


def main():
    out_dir = os.path.abspath(os.path.expanduser(args_cli.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    header = f"{args_cli.label_model} | {args_cli.label_task} | noise={args_cli.fixed_pos_noise_mm:g}mm dr={args_cli.dyn_rand}"
    fps = 15.0

    reset_wrench = None
    if args_cli.model_type == "ta":
        raw_env, cap, step_fn = run_ta()
    else:
        raw_env, cap, step_fn, reset_wrench = run_student()

    n_ok, n_bad = 0, 0
    ep_idx = 0
    while (n_ok < args_cli.n_success or n_bad < args_cli.n_fail) and ep_idx < args_cli.max_episodes:
        cap["final_success"] = None
        cap["force_now"] = 0.0
        if reset_wrench is not None:
            reset_wrench()
        frames = []
        step_idx = 0
        done = False
        while not done:
            done = step_fn()
            raw_frame = raw_env.render()
            if raw_frame is not None:
                frames.append((step_idx, raw_frame, cap.get("force_now", 0.0)))
            step_idx += 1
        success = bool(cap["final_success"])
        outcome = "SUCCESS" if success else "FAILURE"
        keep = (success and n_ok < args_cli.n_success) or (not success and n_bad < args_cli.n_fail)
        print(f"CLIP_EP ep={ep_idx} steps={len(frames)} success={success} keep={keep}", flush=True)
        if keep:
            labeled = [overlay_label(f, header, outcome, s, force) for (s, f, force) in frames]
            idx = n_ok if success else n_bad
            fname = (f"{args_cli.label_model}_{args_cli.label_task}_noise{args_cli.fixed_pos_noise_mm:g}mm_"
                      f"dr{args_cli.dyn_rand}_{outcome.lower()}_{idx}.mp4").replace(" ", "")
            write_clip(os.path.join(out_dir, fname), labeled, fps)
            if success:
                n_ok += 1
            else:
                n_bad += 1
        ep_idx += 1

    print(f"RECORD_DONE label_model={args_cli.label_model} label_task={args_cli.label_task} "
          f"noise={args_cli.fixed_pos_noise_mm:g} dr={args_cli.dyn_rand} "
          f"n_success={n_ok} n_fail={n_bad} episodes_used={ep_idx}", flush=True)


if __name__ == "__main__":
    main()
    # HARD EXIT instead of simulation_app.close(): Kit teardown reliably hangs
    # for hours in this headless+render config (observed twice). Every clip is
    # already flushed to disk by write_clip()'s VideoWriter.release(), so
    # skipping teardown loses nothing and lets the driver loop proceed.
    os._exit(0)
