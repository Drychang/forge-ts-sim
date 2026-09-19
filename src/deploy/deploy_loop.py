"""15Hz student-policy deployment loop, driven through a RobotIO backend.

Backend-agnostic: works identically against SimRobotIO (loopback verification)
or a future FrankaRobotIO (franka_io_stub.py) -- this file never imports
isaaclab or libfranka directly.

Loopback verification (G4 acceptance criterion, run from ~/forge_ts/src/student
so student_fmt.py is importable, with SimRobotIO on sys.path):
  ./isaaclab.sh -p ~/forge_ts/src/deploy/deploy_loop.py \
    --backend sim --task Isaac-Forge-PegInsert-TBCamera-v0 \
    --checkpoint ~/forge_ts/student_ckpts/peg/gate2_seed0/best.pt \
    --norm_stats ~/forge_ts/student_ckpts/peg/gate2_seed0/norm_stats.npz \
    --episodes 128 --noise_mm 2.5 --dyn_rand on --protocol_seed 42
Compare the printed SR against eval_frozen_student.py under the identical
condition -- must match within ~2 points (both are stochastic in PhysX/cuDNN,
see eval_frozen.py's frozen-protocol docstring for why exact bit-match isn't
expected).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student"))

POLICY_HZ = 15.0
POLICY_DT = 1.0 / POLICY_HZ


def build_policy(checkpoint, norm_stats, device):
    from student_fmt import StudentFMTConfig, StudentFMTPolicy

    stats = dict(np.load(os.path.expanduser(norm_stats)))
    ckpt = torch.load(os.path.expanduser(checkpoint), map_location=device)
    policy = StudentFMTPolicy(StudentFMTConfig(**ckpt["config"])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    norm = {
        "state_mean": torch.as_tensor(stats["state_mean"], device=device, dtype=torch.float32),
        "state_std": torch.as_tensor(stats["state_std"], device=device, dtype=torch.float32),
        "wrench_mean": torch.as_tensor(stats["wrench_mean"], device=device, dtype=torch.float32),
        "wrench_std": torch.as_tensor(stats["wrench_std"], device=device, dtype=torch.float32),
    }
    return policy, norm


def run_episode(io, policy, norm, device, real_time=False, max_steps=1000):
    io.reset()
    steps = 0
    while not io.is_done() and steps < max_steps:
        t0 = time.time()
        # torch.as_tensor: accepts numpy (franka backend) AND torch tensors on
        # any device (sim backend) -- the RobotIO contract allows both.
        state = torch.as_tensor(io.get_obs24(), dtype=torch.float32).to(device)
        wrench = torch.as_tensor(io.get_wrench_window(), dtype=torch.float32).to(device)
        images = io.get_images()
        tp_rgb = torch.as_tensor(images["tp"], device=device).unsqueeze(0)
        wrist_rgb = torch.as_tensor(images["wrist"], device=device).unsqueeze(0)

        state_n = (state.unsqueeze(0) - norm["state_mean"]) / norm["state_std"]
        wrench_n = (wrench.unsqueeze(0) - norm["wrench_mean"]) / norm["wrench_std"]

        with torch.inference_mode():
            action = policy(state_n, wrench_n, tp_rgb, wrist_rgb)[0]

        io.send_action(action)
        steps += 1

        if real_time:
            dt = time.time() - t0
            if dt < POLICY_DT:
                time.sleep(POLICY_DT - dt)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "franka"], required=True)
    ap.add_argument("--task", required=True, help="sim: gym id, e.g. Isaac-Forge-PegInsert-TBCamera-v0")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--norm_stats", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--noise_mm", type=float, default=1.0, help="sim backend only")
    ap.add_argument("--dyn_rand", choices=["on", "off"], default="on", help="sim backend only")
    ap.add_argument("--protocol_seed", type=int, default=42, help="sim backend only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--real_time", action="store_true", help="pace to 15Hz wall-clock (franka backend needs this)")
    args, unknown = ap.parse_known_args()

    if args.backend == "sim":
        # AppLauncher must exist before importing isaaclab -- this script is
        # meant to run via isaaclab.sh -p, which handles that; if invoked
        # directly, replicate the minimal launch here.
        from isaaclab.app import AppLauncher

        app_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(app_parser)
        app_args = app_parser.parse_args(unknown + ["--headless"])
        app_args.enable_cameras = True  # TBCamera env needs this or TiledCamera raises at init
        app_launcher = AppLauncher(app_args)
        simulation_app = app_launcher.app  # noqa: F841

        from sim_io import SimRobotIO
        io = SimRobotIO(args.task, noise_mm=args.noise_mm, dyn_rand=args.dyn_rand,
                        protocol_seed=args.protocol_seed, device=args.device)
    else:
        from franka_io_stub import FrankaRobotIO
        io = FrankaRobotIO()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, norm = build_policy(args.checkpoint, args.norm_stats, device)

    successes = []
    t_start = time.time()
    for ep in range(args.episodes):
        n_steps = run_episode(io, policy, norm, device, real_time=args.real_time)
        s = io.last_success() if hasattr(io, "last_success") else None
        if s is not None:
            successes.append(s)
        print(f"DEPLOY_EP ep={ep+1}/{args.episodes} steps={n_steps} success={s}", flush=True)

    if successes:
        sr = sum(successes) / len(successes)
        print(f"DEPLOY_LOOP_SUMMARY n={len(successes)} sr={sr:.4f} "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)
    io.close()
    if args.backend == "sim":
        os._exit(0)  # skip Kit teardown (known to hang, see record_labeled_clips.py)


if __name__ == "__main__":
    main()
