"""Smoke test for tb_camera_env.py: gym.make the real registered id, run several
policy steps with random actions, verify student_obs/camera/wrench interfaces
all have correct shapes and actually vary (not stuck at zero), and verify the
wrench buffer zeroes out across a reset.

usage: smoke_tb_camera_env.py --task Isaac-Forge-PegInsert-TBCamera-v0 --num_envs 4
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-TBCamera-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import tb_camera_env  # noqa: E402, F401 -- side effect: gym.register
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
print(f"[SMOKE] cfg.scene.clone_in_fabric = {env_cfg.scene.clone_in_fabric}", flush=True)
assert env_cfg.scene.clone_in_fabric is False, "fix not applied!"

env = gym.make(args_cli.task, cfg=env_cfg)
raw_env = env.unwrapped
print(f"[SMOKE] constructed {args_cli.task} num_envs={raw_env.num_envs}", flush=True)

obs, info = env.reset()
print("[SMOKE] RESET_OK", flush=True)

# right after reset: wrench history must be all-zero (module docstring contract)
wrench0 = raw_env.get_wrench_history()
assert wrench0.shape == (raw_env.num_envs, raw_env.cfg.decimation, 6), f"bad wrench shape {wrench0.shape}"
assert torch.all(wrench0 == 0.0), "wrench history should be zero right after reset"
print(f"[SMOKE] wrench_history shape={tuple(wrench0.shape)} all-zero-after-reset OK", flush=True)

student_obs0 = raw_env.student_obs
assert student_obs0.shape == (raw_env.num_envs, 24), f"bad student_obs shape {student_obs0.shape}"
print(f"[SMOKE] student_obs shape={tuple(student_obs0.shape)}", flush=True)

images0 = raw_env.get_camera_images()
assert images0["tp_rgb"].shape == (raw_env.num_envs, 256, 256, 3), images0["tp_rgb"].shape
assert images0["wrist_rgb"].shape == (raw_env.num_envs, 256, 256, 3), images0["wrist_rgb"].shape
print(f"[SMOKE] camera images tp={tuple(images0['tp_rgb'].shape)} wrist={tuple(images0['wrist_rgb'].shape)} "
      f"dtype={images0['tp_rgb'].dtype}", flush=True)

# step a few times, confirm wrench actually varies (i.e. we're reading real physics,
# not a frozen buffer) and student_obs changes too
nonzero_wrench_seen = False
prev_student_obs = student_obs0.clone()
student_obs_changed = False
for i in range(8):
    act = torch.rand((raw_env.num_envs, raw_env.cfg.action_space), device=raw_env.device) * 2 - 1
    obs, rew, terminated, truncated, extras = env.step(act)
    w = raw_env.get_wrench_history()
    if torch.any(w != 0.0):
        nonzero_wrench_seen = True
    if not torch.allclose(raw_env.student_obs, prev_student_obs):
        student_obs_changed = True
    prev_student_obs = raw_env.student_obs.clone()
    print(f"[SMOKE] step {i}: wrench_absmax={w.abs().max().item():.4f} "
          f"student_obs_l2={raw_env.student_obs.norm().item():.4f}", flush=True)

assert nonzero_wrench_seen, "wrench history stayed all-zero across 8 steps -- not reading real physics"
assert student_obs_changed, "student_obs never changed across 8 steps"
print("[SMOKE] STEP_OK, wrench varies, student_obs varies", flush=True)

# reset again mid-stream, confirm wrench snaps back to zero
obs, info = env.reset()
wrench_after_reset2 = raw_env.get_wrench_history()
assert torch.all(wrench_after_reset2 == 0.0), "wrench history should be zero right after a mid-stream reset too"
print("[SMOKE] RESET2_OK, wrench correctly zeroed again", flush=True)

print("[SMOKE] ALL_CHECKS_PASSED", flush=True)
env.close()
simulation_app.close()
