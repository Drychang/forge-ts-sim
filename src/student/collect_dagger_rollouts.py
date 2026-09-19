"""P2-4 DAgger data collector.

Unlike collect_camera_rollouts.py (P2-2, teacher drives, only success kept),
this script:
  1. Lets the STUDENT (not T-B) drive env.step() -- so the visited state
     distribution matches what the deployed student will actually see,
     including states that lead to failure. That distribution mismatch is
     exactly what DAgger exists to fix.
  2. Records, every step, the T-B 64-dim privileged obs that the env already
     computes internally (tb_camera_env.py's ForgeTBCameraEnv always returns
     it as obs["policy"] from step()/reset(), regardless of who is driving
     actions -- this is instrumentation, not a behavior change).
  3. At episode end, RELABELS the entire episode offline: replays that
     episode's T-B obs sequence, in order, through the T-B LSTM policy
     (hidden state reset once at t=0, never mid-episode -- rl_games'
     PpoPlayerContinuous.get_action carries self.states across calls
     automatically, so a plain sequential loop is correct here) and takes
     the deterministic mean action as the new label.
  4. Keeps EVERY episode, success or not (DAgger's value is precisely in
     the states the student visits but the teacher never had to correct
     before -- filtering to successes-only would throw away the most
     informative failure states).
  5. Writes shards in the IDENTICAL format to collect_camera_rollouts.py,
     so dataset.py / train_student_bc.py can train on original-BC and
     DAgger shards side by side with zero code changes.

Usage (from the IsaacLab repo root):
  ./isaaclab.sh -p ~/forge_ts/src/student/collect_dagger_rollouts.py --headless \\
      --task Isaac-Forge-PegInsert-TBCamera-v0 \\
      --student_checkpoint ~/forge_ts/student_ckpts/peg/best.pt \\
      --student_norm_stats ~/forge_ts/student_ckpts/peg/norm_stats.npz \\
      --teacher_checkpoint ~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_peg_s2/nn/last_Forge_ep_200_rew_380.91483.pth \\
      --num_envs 32 --target_episodes 1000 \\
      --out_dir /media/data/forge_ts_data/PegInsert
"""
import argparse
import os
import time

os.environ.setdefault("TB_RANDOMIZE_NOISE", "1")  # match original BC collection's noise coverage

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="e.g. Isaac-Forge-PegInsert-TBCamera-v0")
parser.add_argument("--student_checkpoint", type=str, required=True)
parser.add_argument("--student_norm_stats", type=str, required=True)
parser.add_argument("--teacher_checkpoint", type=str, required=True, help="T-B rl_games checkpoint (.pth) used to relabel.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--target_episodes", type=int, default=1000)
parser.add_argument("--shard_size", type=int, default=100)
parser.add_argument("--jpeg_quality", type=int, default=90)
parser.add_argument("--shard_start_idx", type=int, default=0, help="first shard index to write (avoid clobbering existing BC/DAgger shards).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows -- Kit is up, safe to import isaaclab_tasks/tb_camera_env now."""

import cv2  # noqa: E402
import math  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tb_camera_env  # noqa: E402, F401 -- side effect: gym.register
from student_fmt import StudentFMTConfig, StudentFMTPolicy  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def encode_batch_jpeg(img_batch_rgb_u8, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    out = []
    for i in range(img_batch_rgb_u8.shape[0]):
        bgr = cv2.cvtColor(img_batch_rgb_u8[i], cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, encode_param)
        assert ok, "JPEG encode failed"
        out.append(buf.tobytes())
    return out


def flush_shard(shard_data, out_dir, shard_idx):
    path = os.path.join(out_dir, f"shard_{shard_idx:04d}.npz")
    np.savez(
        path,
        student_obs=np.stack(shard_data["student_obs"]),
        wrench_raw=np.stack(shard_data["wrench_raw"]),
        tp_jpeg=np.stack(shard_data["tp_jpeg"]),
        wrist_jpeg=np.stack(shard_data["wrist_jpeg"]),
        action=np.stack(shard_data["action"]),
        ep_len=np.array(shard_data["ep_len"], dtype=np.int32),
        noise_std_mm=np.array(shard_data["noise_std_mm"], dtype=np.float32),
    )
    print(f"[DAGGER] wrote {path} ({len(shard_data['ep_len'])} episodes)", flush=True)


def relabel_episode_with_teacher(teacher_agent, tb_obs_seq):
    """tb_obs_seq: (L, num_envs, 64) tensor. Returns (num_envs, L, 7) relabeled
    deterministic actions. Hidden state is reset once at t=0 and carried
    across all L steps by rl_games' get_action internally (self.states) --
    replaying in order is what makes this a faithful "what would the teacher
    have done, having watched this whole episode unfold" relabel."""
    L = tb_obs_seq.shape[0]
    with torch.inference_mode():
        teacher_agent.init_rnn()
        actions = []
        for t in range(L):
            obs_t = teacher_agent.obs_to_torch(tb_obs_seq[t])
            act_t = teacher_agent.get_action(obs_t, is_deterministic=True)
            actions.append(act_t.detach().to("cpu", torch.float32))
    return torch.stack(actions, dim=1)  # (num_envs, L, action_dim)


def main():
    student_ckpt_path = os.path.abspath(os.path.expanduser(args_cli.student_checkpoint))
    teacher_ckpt_path = os.path.abspath(os.path.expanduser(args_cli.teacher_checkpoint))
    out_dir = os.path.abspath(os.path.expanduser(args_cli.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- student policy ----
    stats = dict(np.load(os.path.expanduser(args_cli.student_norm_stats)))
    state_mean = torch.as_tensor(stats["state_mean"], device=device, dtype=torch.float32)
    state_std = torch.as_tensor(stats["state_std"], device=device, dtype=torch.float32)
    wrench_mean = torch.as_tensor(stats["wrench_mean"], device=device, dtype=torch.float32)
    wrench_std = torch.as_tensor(stats["wrench_std"], device=device, dtype=torch.float32)

    ckpt = torch.load(student_ckpt_path, map_location=device)
    student = StudentFMTPolicy(StudentFMTConfig(**ckpt["config"])).to(device)
    student.load_state_dict(ckpt["model"])
    student.eval()

    # ---- env (student drives it) ----
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_env = env.unwrapped  # ForgeTBCameraEnv
    num_envs = raw_env.num_envs
    decimation = raw_env.cfg.decimation
    max_ep_len = raw_env.max_episode_length

    # ---- teacher (T-B), used ONLY for offline relabeling, never drives env ----
    # BasePlayer.__init__ calls self.create_env() (needs an 'rlgpu' vecenv
    # registered) UNLESS config['env_info'] is already provided -- since this
    # player never steps a real env (get_action/init_rnn/obs_to_torch only),
    # supplying env_info directly skips that whole registration dance.
    import gym as legacy_gym  # noqa: E402 -- rl_games' own dependency, not gymnasium

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": legacy_gym.spaces.Box(-float("inf"), float("inf"), (64,), dtype=np.float32),
        "action_space": legacy_gym.spaces.Box(-1.0, 1.0, (raw_env.cfg.action_space,), dtype=np.float32),
        "agents": 1,
        "value_size": 1,
    }
    runner = Runner()
    runner.load(agent_cfg)
    teacher_agent = runner.create_player()
    teacher_agent.restore(teacher_ckpt_path)
    teacher_agent.get_batch_size(torch.zeros(num_envs, 64, device=device), 1)
    teacher_agent.reset()

    shard_data = {k: [] for k in
                  ["student_obs", "wrench_raw", "tp_jpeg", "wrist_jpeg", "action", "ep_len", "noise_std_mm"]}
    shard_idx = args_cli.shard_start_idx
    n_episodes_total = 0
    n_success_total = 0
    t_start = time.time()

    print(f"[DAGGER] task={args_cli.task} num_envs={num_envs} max_ep_len={max_ep_len} "
          f"target_episodes={args_cli.target_episodes} out_dir={out_dir}", flush=True)

    # success capture hook (informational only -- DAgger keeps every episode
    # regardless, but the running SR is a useful health signal to log)
    cap = {"final_success": None}
    orig_get_rewards = raw_env._get_rewards
    check_rot = raw_env.cfg_task.name == "nut_thread"

    def hooked_get_rewards():
        rew = orig_get_rewards()
        if bool(raw_env.reset_buf.any()):
            cap["final_success"] = raw_env._get_curr_successes(
                success_threshold=raw_env.cfg_task.success_threshold, check_rot=check_rot
            ).clone()
        return rew

    raw_env._get_rewards = hooked_get_rewards

    while n_episodes_total < args_cli.target_episodes:
        obs, _ = env.reset()  # obs = {"policy": (N,64) T-B obs, "critic": (N,64)}

        buf_student_obs = torch.zeros((num_envs, max_ep_len, 24))
        buf_wrench = torch.zeros((num_envs, max_ep_len, decimation, 6))
        buf_tb_obs = torch.zeros((max_ep_len, num_envs, 64))  # time-major, for sequential relabel
        buf_tp_jpeg = [[None] * max_ep_len for _ in range(num_envs)]
        buf_wrist_jpeg = [[None] * max_ep_len for _ in range(num_envs)]
        wrench_roll = torch.zeros((num_envs, 32, 6), device=device)

        for t in range(max_ep_len):
            with torch.inference_mode():
                cur_student_obs = raw_env.student_obs.detach()
                new_wrench = raw_env.get_wrench_history().detach().to(torch.float32)  # (N, decimation, 6)
                wrench_roll = torch.cat([wrench_roll[:, decimation:], new_wrench], dim=1)
                images = raw_env.get_camera_images()
                tp_rgb = images["tp_rgb"]
                wrist_rgb = images["wrist_rgb"]

                state_n = (cur_student_obs.to(torch.float32) - state_mean) / state_std
                wrench_n = (wrench_roll - wrench_mean) / wrench_std
                action = student(state_n, wrench_n, tp_rgb, wrist_rgb)

                buf_student_obs[:, t] = cur_student_obs.cpu().to(torch.float32)
                buf_wrench[:, t] = new_wrench.cpu()
                buf_tb_obs[t] = obs["policy"].detach().cpu().to(torch.float32)
                tp_np = tp_rgb.cpu().numpy()
                wrist_np = wrist_rgb.cpu().numpy()

            tp_jpegs = encode_batch_jpeg(tp_np, args_cli.jpeg_quality)
            wrist_jpegs = encode_batch_jpeg(wrist_np, args_cli.jpeg_quality)
            for e in range(num_envs):
                buf_tp_jpeg[e][t] = tp_jpegs[e]
                buf_wrist_jpeg[e][t] = wrist_jpegs[e]

            obs, _, terminated, truncated, _ = env.step(action)
            dones = terminated | truncated
            if bool(dones.any()) and not bool(dones.all()):
                raise RuntimeError("Protocol violated: envs terminated out of sync (Factory/Forge is timeout-only).")

        if cap["final_success"] is None:
            raise RuntimeError("Success capture hook never fired at episode end.")
        final_success = cap["final_success"].cpu()
        cap["final_success"] = None
        noise_mm = torch.norm(raw_env.init_fixed_pos_obs_noise.detach().cpu(), dim=-1) * 1000.0

        relabeled_actions = relabel_episode_with_teacher(teacher_agent, buf_tb_obs.to(device))

        n_episodes_total += num_envs
        n_success_total += int(final_success.sum().item())

        for e in range(num_envs):
            shard_data["student_obs"].append(buf_student_obs[e].numpy().copy())
            shard_data["wrench_raw"].append(buf_wrench[e].numpy().copy())
            shard_data["tp_jpeg"].append(np.array(buf_tp_jpeg[e], dtype=object))
            shard_data["wrist_jpeg"].append(np.array(buf_wrist_jpeg[e], dtype=object))
            shard_data["action"].append(relabeled_actions[e].numpy().copy())
            shard_data["ep_len"].append(max_ep_len)
            shard_data["noise_std_mm"].append(float(noise_mm[e]))

        if len(shard_data["ep_len"]) >= args_cli.shard_size:
            flush_shard(shard_data, out_dir, shard_idx)
            shard_idx += 1
            shard_data = {k: [] for k in shard_data}

        elapsed = time.time() - t_start
        sr_running = n_success_total / max(1, n_episodes_total)
        print(f"[DAGGER] +{num_envs} episodes (student-driven, all kept) -> "
              f"total {n_episodes_total}/{args_cli.target_episodes} "
              f"(running_student_sr={sr_running:.3f}, elapsed={elapsed:.0f}s)", flush=True)

    if len(shard_data["ep_len"]) > 0:
        flush_shard(shard_data, out_dir, shard_idx)

    print(f"[DAGGER] DONE: {n_episodes_total} episodes collected+relabeled "
          f"(student's own SR while collecting={n_success_total/max(1,n_episodes_total):.3f}) "
          f"in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
