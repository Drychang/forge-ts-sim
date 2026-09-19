"""P2-2 data collector: roll out a T-B teacher checkpoint on ForgeTBCameraEnv,
keep only successful episodes, shard them to disk in the format dataset.py
(P2-1) expects.

Design notes (all deliberate, not oversights):
- Actions are SAMPLED from the teacher's action distribution (is_deterministic=
  False), not the deterministic mean -- gives free dither/state coverage
  instead of a single narrow trajectory per reset draw (plan spec §4.2).
- Factory/Forge is a FIXED-HORIZON, LOCKSTEP task family: _get_dones is
  timeout-only and identical for every env (verified in eval_frozen.py's
  protocol docstring and enforced here via the same "dones must be all-or-
  nothing" assertion), so every episode has exactly raw_env.max_episode_length
  steps and every env starts/ends its episode at the same global step. That's
  why a single scalar step_idx (not a per-env counter) is correct below.
- Success = FINAL-step success (same primary metric as eval_frozen.py's `sr`,
  not the more lenient `sr_ever`) -- keeps the collected data's success
  definition identical to the number Gate 2 will be judged against.
- noise_std_mm actually stores the L2 norm (mm) of the per-episode
  init_fixed_pos_obs_noise offset that was actually drawn this episode -- not
  the generating std parameter itself (tb_env.py's ForgeTBEnv.randomize_
  initial_state never exposes that scalar). Equivalent information for any
  downstream analysis of "how much noise was this episode collected under".

Usage (from the IsaacLab repo root):
  ./isaaclab.sh -p ~/forge_ts/src/collect_camera_rollouts.py --headless \\
      --task Isaac-Forge-PegInsert-TBCamera-v0 \\
      --checkpoint ~/force_vla_research/IsaacLab/logs/rl_games/Forge/tb_peg_s2/nn/last_Forge_ep_200_rew_380.91483.pth \\
      --num_envs 32 --target_success 2000 \\
      --out_dir /media/data/forge_ts_data/PegInsert

Best-SR-at-5mm T-B seed per task (queried from ~/forge_ts/eval/*tb_noisespec*, 2026-07-05):
  PegInsert : tb_peg_s2  (sr@5mm/dr_on = 1.000)  logs/rl_games/Forge/tb_peg_s2/nn/last_Forge_ep_200_rew_380.91483.pth
  GearMesh  : tb_gear_s1 (sr@5mm/dr_on = 0.992)  logs/rl_games/Forge/tb_gear_s1/nn/last_Forge_ep_200_rew_760.77905.pth
  NutThread : tb_nut_s1  (sr@5mm/dr_on = 1.000)  logs/rl_games/Forge/tb_nut_s1/nn/last_Forge_ep_200_rew_1138.2274.pth
"""
import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="e.g. Isaac-Forge-PegInsert-TBCamera-v0")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="T-A (24-dim, non-privileged) rl_games checkpoint (.pth) to roll out.")
parser.add_argument("--agent_task", type=str, default=None,
                    help="Task whose rl_games cfg built the checkpoint. Default: the -Direct- twin of "
                         "--task, i.e. T-A's own network/normalisation config.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--target_success", type=int, default=2000)
parser.add_argument("--shard_size", type=int, default=100)
parser.add_argument("--jpeg_quality", type=int, default=90)
parser.add_argument("--seed", type=int, default=0, help="collection RNG seed (not a frozen protocol seed).")
parser.add_argument("--out_dir", type=str, required=True, help="e.g. /media/data/forge_ts_data/PegInsert")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows -- Kit is up, safe to import isaaclab_tasks/tb_camera_env now."""

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import tb_camera_env  # noqa: E402, F401 -- side effect: gym.register the TBCamera envs
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import math  # noqa: E402

TA_OBS_DIM = 24
# --- P0-1 control: distil from the NON-privileged teacher -------------------
# Identical to collect_camera_rollouts.py in every respect (env, noise
# distribution, success-only filtering, sharding, JPEG path) EXCEPT which policy
# generates the actions. The camera env's policy slot normally carries T-B's
# 64-dim privileged obs; the same env already computes T-A's official 24-dim
# noisy obs every step as additive instrumentation (raw_env.student_obs -- also
# exactly what lands in the dataset). We declare observation_space=24 and swap
# the policy slot for student_obs inside _get_observations, the single funnel
# both reset() and step() go through, so the rl_games wrapper's clipping /
# concat / asymmetric-state handling stays byte-identical to the T-B path.
# Declared space == actual tensor, so nothing downstream is fooled.


def encode_batch_jpeg(img_batch_rgb_u8, quality):
    """img_batch: (N,H,W,3) uint8 numpy, RGB order (as returned by TiledCamera).
    Returns a python list of N raw JPEG byte strings (BGR-encoded, matching
    dataset.py's cv2.imdecode + COLOR_BGR2RGB decode convention)."""
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
    print(f"[COLLECT] wrote {path} ({len(shard_data['ep_len'])} episodes)", flush=True)


def main():
    ckpt_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    out_dir = os.path.abspath(os.path.expanduser(args_cli.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # obs_order (NOT observation_space) is the real lever: the env recomputes
    # observation_space from it, so this is what makes the policy slot 24-dim.
    env_cfg.obs_order = list(tb_camera_env.TA_OBS_ORDER)

    agent_task = args_cli.agent_task or args_cli.task.replace("-TBCamera-", "-Direct-")
    if agent_task == args_cli.task:
        raise SystemExit(f"cannot derive the -Direct- twin of {args_cli.task}; pass --agent_task")
    print(f"[TA] agent cfg from {agent_task} (obs_dim={TA_OBS_DIM})", flush=True)
    agent_cfg = load_cfg_from_registry(agent_task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["seed"] = args_cli.seed
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_env = env.unwrapped  # ForgeTBCameraEnv -- keep a handle before rl_games wraps it

    _orig_get_obs = raw_env._get_observations

    _checked = []

    def _ta_get_observations():
        obs_dict = _orig_get_obs()
        p = obs_dict["policy"]
        if p.shape[-1] != TA_OBS_DIM:
            raise RuntimeError(f"policy slot is {tuple(p.shape)}, expected (*, {TA_OBS_DIM}); "
                               "the obs_order override did not take effect")
        if not _checked:
            # Both tensors are T-A's official layout at the same physics state;
            # they differ only by FORGE's independent per-step obs-noise draws
            # (tb_camera_env computes the T-A obs twice -- see module note).
            d = (p - raw_env.student_obs).abs().amax(dim=0)
            print("[TA] policy slot == student_obs (the dataset tensor). Discarded "
                  f"second draw differed by max-abs per dim: {[round(float(v), 5) for v in d]}",
                  flush=True)
            _checked.append(True)
        # act on EXACTLY what the dataset records
        obs_dict["policy"] = raw_env.student_obs
        return obs_dict

    raw_env._get_observations = _ta_get_observations
    wrapped_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped_env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = ckpt_path
    agent_cfg["params"]["config"]["num_actors"] = raw_env.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(ckpt_path)
    agent.reset()

    num_envs = raw_env.num_envs
    sim_device = raw_env.device
    action_dim = raw_env.cfg.action_space
    decimation = raw_env.cfg.decimation
    max_ep_len = raw_env.max_episode_length
    check_rot = raw_env.cfg_task.name == "nut_thread"

    # ---- success capture hook (same mechanism as eval_frozen.py) ----
    cap = {"final_success": None}
    orig_get_rewards = raw_env._get_rewards

    def hooked_get_rewards():
        rew = orig_get_rewards()
        if bool(raw_env.reset_buf.any()):
            cap["final_success"] = raw_env._get_curr_successes(
                success_threshold=raw_env.cfg_task.success_threshold, check_rot=check_rot
            ).clone()
        return rew

    raw_env._get_rewards = hooked_get_rewards

    obs = wrapped_env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # per-episode scratch buffers -- fixed-horizon+lockstep (see module docstring)
    # means one scalar step_idx and one flat max_ep_len axis suffice.
    buf_student_obs = torch.zeros((num_envs, max_ep_len, 24))
    buf_wrench = torch.zeros((num_envs, max_ep_len, decimation, 6))
    buf_action = torch.zeros((num_envs, max_ep_len, action_dim))
    buf_tp_jpeg = [[None] * max_ep_len for _ in range(num_envs)]
    buf_wrist_jpeg = [[None] * max_ep_len for _ in range(num_envs)]
    step_idx = 0

    shard_data = {k: [] for k in
                  ["student_obs", "wrench_raw", "tp_jpeg", "wrist_jpeg", "action", "ep_len", "noise_std_mm"]}
    shard_idx = 0
    n_success_total = 0
    n_episodes_total = 0
    t_start = time.time()

    print(f"[COLLECT] task={args_cli.task} num_envs={num_envs} max_ep_len={max_ep_len} "
          f"target_success={args_cli.target_success} out_dir={out_dir}", flush=True)

    while n_success_total < args_cli.target_success:
        # Entire body must stay inside inference_mode: agent.states (LSTM hidden
        # state) is created under inference_mode by agent.get_action/init_rnn, so
        # any later in-place update to it (the episode-boundary reset below) MUST
        # also run inside inference_mode, or torch raises "Inplace update to
        # inference tensor outside InferenceMode is not allowed." Matches
        # eval_frozen.py's identical single-with-block structure.
        with torch.inference_mode():
            cur_student_obs = raw_env.student_obs.detach().to("cpu", torch.float32)
            cur_wrench = raw_env.get_wrench_history().detach().to("cpu", torch.float32)
            images = raw_env.get_camera_images()
            tp_np = images["tp_rgb"].detach().cpu().numpy()
            wrist_np = images["wrist_rgb"].detach().cpu().numpy()

            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=False)  # SAMPLE, not mu

            if step_idx == 0:
                # Snapshot the offset THIS episode is being run under. It must be
                # read here, not at episode end: step() resets terminated envs
                # internally and re-draws init_fixed_pos_obs_noise, so a read
                # after the final step returns the NEXT episode's value.
                ep_noise_mm = (torch.norm(raw_env.init_fixed_pos_obs_noise.detach(), dim=-1)
                               * 1000.0).cpu()

            buf_student_obs[:, step_idx] = cur_student_obs
            buf_wrench[:, step_idx] = cur_wrench
            buf_action[:, step_idx] = actions.detach().to("cpu", torch.float32)
            tp_jpegs = encode_batch_jpeg(tp_np, args_cli.jpeg_quality)
            wrist_jpegs = encode_batch_jpeg(wrist_np, args_cli.jpeg_quality)
            for e in range(num_envs):
                buf_tp_jpeg[e][step_idx] = tp_jpegs[e]
                buf_wrist_jpeg[e][step_idx] = wrist_jpegs[e]
            step_idx += 1

            obs, _, dones, _ = wrapped_env.step(actions)

            if bool(dones.any()):
                if not bool(dones.all()):
                    raise RuntimeError(
                        "Protocol violated: envs terminated out of sync (Factory/Forge is timeout-only)."
                    )
                if cap["final_success"] is None:
                    raise RuntimeError("Success capture hook never fired at episode end.")

                final_success = cap["final_success"].cpu()
                noise_mm = ep_noise_mm  # captured at this episode's first step, see above
                n_episodes_total += num_envs
                n_new_success = int(final_success.sum().item())

                for e in range(num_envs):
                    if bool(final_success[e]):
                        shard_data["student_obs"].append(buf_student_obs[e].numpy().copy())
                        shard_data["wrench_raw"].append(buf_wrench[e].numpy().copy())
                        shard_data["tp_jpeg"].append(np.array(buf_tp_jpeg[e], dtype=object))
                        shard_data["wrist_jpeg"].append(np.array(buf_wrist_jpeg[e], dtype=object))
                        shard_data["action"].append(buf_action[e].numpy().copy())
                        shard_data["ep_len"].append(step_idx)
                        shard_data["noise_std_mm"].append(float(noise_mm[e]))
                        n_success_total += 1

                if len(shard_data["ep_len"]) >= args_cli.shard_size:
                    flush_shard(shard_data, out_dir, shard_idx)
                    shard_idx += 1
                    shard_data = {k: [] for k in shard_data}

                cap["final_success"] = None
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                step_idx = 0

                elapsed = time.time() - t_start
                sr_running = n_success_total / max(1, n_episodes_total)
                print(f"[COLLECT] +{n_new_success}/{num_envs} success this round -> "
                      f"total {n_success_total}/{args_cli.target_success} "
                      f"(running_sr={sr_running:.3f}, elapsed={elapsed:.0f}s)", flush=True)

    if len(shard_data["ep_len"]) > 0:
        flush_shard(shard_data, out_dir, shard_idx)

    print(f"[COLLECT] DONE: {n_success_total} successful episodes from {n_episodes_total} rollouts "
          f"(sr={n_success_total/max(1,n_episodes_total):.3f}) in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
