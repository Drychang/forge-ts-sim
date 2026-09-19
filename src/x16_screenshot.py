#!/usr/bin/env python3
"""Capture screenshots from gear env with real vs sim camera poses.
Run BEFORE the full x16 training to visually verify camera setup.
Uses 4 envs (low VRAM) and saves PNG files for comparison.

Usage:
    CUDA_VISIBLE_DEVICES=0 python x16_screenshot.py --headless

Outputs to ~/forge_ts/logs/x16_screenshots/
"""
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.expanduser("~/forge_ts/logs/x16_screenshots")
    os.makedirs(out_dir, exist_ok=True)

    # We'll run two passes: real poses and sim nominal poses
    configs = [
        {
            "name": "real",
            "env": {
                "TB_CAM_WRIST_POS": "0.036666,-0.030747,-0.046328",
                "TB_CAM_WRIST_ROT": "0.702792,0.013969,0.004503,0.711244",
                "TB_CAM_TP_POS": "1.023540,-0.050991,0.378229",
                "TB_CAM_TP_ROT": "0.436309,-0.597628,-0.550977,0.385874",
                "TB_CAM_HFOV_DEG": "55.7",
            },
        },
        {
            "name": "sim_nominal",
            "env": {},  # no overrides = sim nominal
        },
    ]

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Capturing: {cfg['name']}")
        print(f"{'='*60}")

        # Set env vars
        for k, v in cfg["env"].items():
            os.environ[k] = v

        # Clear env vars not in this config
        for k in ["TB_CAM_WRIST_POS", "TB_CAM_WRIST_ROT", "TB_CAM_TP_POS",
                   "TB_CAM_TP_ROT", "TB_CAM_HFOV_DEG"]:
            if k not in cfg["env"]:
                os.environ.pop(k, None)

        # Import here so env vars are read fresh
        # Isaac Sim can only be initialized once per process, so we do this
        # differently: we'll use a subprocess for the second config
        if cfg["name"] == "sim_nominal":
            # For the second pass, we need a fresh process
            import subprocess
            env2 = os.environ.copy()
            for k in ["TB_CAM_WRIST_POS", "TB_CAM_WRIST_ROT", "TB_CAM_TP_POS",
                       "TB_CAM_TP_ROT", "TB_CAM_HFOV_DEG"]:
                env2.pop(k, None)
            env2["_X16_SCREENSHOT_PASS"] = "sim_nominal"
            env2["_X16_SCREENSHOT_OUTDIR"] = out_dir
            cmd = [sys.executable, __file__]
            if args.headless:
                cmd.append("--headless")
            subprocess.run(cmd, env=env2)
            continue

        _capture_pass(cfg["name"], out_dir, args.headless)
        break  # only first pass in this process

    print(f"\nScreenshots saved to {out_dir}/")
    print("Files:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png"):
            print(f"  {f}")


def _capture_pass(name, out_dir, headless):
    import numpy as np

    # Check if this is a subprocess call
    if "_X16_SCREENSHOT_PASS" in os.environ:
        name = os.environ["_X16_SCREENSHOT_PASS"]
        out_dir = os.environ["_X16_SCREENSHOT_OUTDIR"]

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "Y"

    from omni.isaac.lab.app import AppLauncher
    launcher = AppLauncher(headless=headless)
    simulation_app = launcher.app

    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.sensors import TiledCamera

    # Import env registration
    sys.path.insert(0, os.path.expanduser("~/forge_ts/src"))
    import tb_camera_env  # registers the task

    from omni.isaac.lab.envs import ManagerBasedRLEnv
    import gymnasium as gym

    task = "Isaac-Forge-GearMesh-TBCamera-v0"
    env = gym.make(task, num_envs=4, headless=headless)

    # Reset and step a few times to get stable images
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample() * 0  # zero action
        obs, _, _, _, _ = env.step(action)

    # Extract camera images
    tp_cam = env.unwrapped.scene.sensors["tp_camera"]
    wrist_cam = env.unwrapped.scene.sensors["wrist_camera"]

    tp_rgb = tp_cam.data.output["rgb"][0].cpu().numpy()  # (H, W, 4) RGBA
    wrist_rgb = wrist_cam.data.output["rgb"][0].cpu().numpy()

    # Save as PNG (drop alpha)
    from PIL import Image
    Image.fromarray(tp_rgb[:, :, :3]).save(os.path.join(out_dir, f"{name}_tp.png"))
    Image.fromarray(wrist_rgb[:, :, :3]).save(os.path.join(out_dir, f"{name}_wrist.png"))
    print(f"Saved {name}_tp.png and {name}_wrist.png")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    if "_X16_SCREENSHOT_PASS" in os.environ:
        _capture_pass(None, None, "--headless" in sys.argv)
    else:
        main()
