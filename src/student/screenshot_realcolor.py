"""Capture what the student's tp/wrist cameras see, with original sim colors
(--realcolor off) or the real-print recolored assets (--realcolor on), so
the author can eyeball whether the recolor matches the physical parts BEFORE we
spend GPU-hours on the real-color rehearsal eval.

Zero-action episode: the robot just holds its randomized start pose above the
fixed asset while frames are saved -- no policy involved.

  CUDA_VISIBLE_DEVICES=2 ~/miniconda3/envs/isaaclab/bin/python \
      ~/forge_ts/src/student/screenshot_realcolor.py --headless \
      --task Isaac-Forge-PegInsert-TBCamera-v0 --realcolor on \
      --num_envs 2 --steps 12 --out_dir ~/forge_ts/screenshots_realcolor
"""
import argparse
import os

os.environ["TB_RANDOMIZE_NOISE"] = "0"  # MUST be set before tb_env/tb_camera_env import

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--realcolor", type=str, choices=["on", "off"], required=True)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--save_every", type=int, default=3)
parser.add_argument("--protocol_seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="~/forge_ts/screenshots_realcolor")
parser.add_argument("--table_color", type=str, default=None,
                    help='e.g. "0.92,0.90,0.86" -- flat-recolor the table after env creation')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tb_camera_env  # noqa: E402,F401  (side effect: registers the TBCamera envs)

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

SHORT = {"PegInsert": "peg", "GearMesh": "gear", "NutThread": "nut"}


def main():
    short = next(v for k, v in SHORT.items() if k in args_cli.task)
    out_dir = os.path.expanduser(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.protocol_seed
    if args_cli.realcolor == "on":
        from realcolor_override import apply_realcolor
        apply_realcolor(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if args_cli.table_color is not None:
        from realcolor_override import apply_table_color
        apply_table_color(tuple(float(x) for x in args_cli.table_color.split(",")))
    env.reset()
    mode = "real" if args_cli.realcolor == "on" else "sim"
    if args_cli.table_color is not None:
        mode += "_lt"
    act_dim = env.action_space.shape[-1]
    action = torch.zeros(args_cli.num_envs, act_dim, device=env.unwrapped.device)

    n_saved = 0
    for step in range(args_cli.steps):
        env.step(action)
        if step % args_cli.save_every != 0:
            continue
        imgs = env.unwrapped.get_camera_images()
        for cam in ("tp_rgb", "wrist_rgb"):
            arr = imgs[cam][..., :3].cpu().numpy().astype(np.uint8)
            for i in range(arr.shape[0]):
                fp = os.path.join(out_dir, f"{short}_{mode}_env{i}_step{step:02d}_{cam[:-4]}.png")
                cv2.imwrite(fp, cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
                n_saved += 1
    print(f"SCREENSHOT_DONE task={short} mode={mode} n_saved={n_saved} out={out_dir}", flush=True)
    os._exit(0)  # skip Kit teardown (known permanent hang in headless+render)


if __name__ == "__main__":
    main()
