"""Record episodes (policy-driven or teleop) into training-ready shards.

This is THE tool for both real-robot phases:
  Phase 3 zero-shot protocol : --mode policy  (+ --inject_offset_mm, --interactive)
  Phase 4 teleop demos       : --mode teleop  (actions arrive via UDP JSON)

Every episode is recorded in the EXACT shard schema ForgeRolloutDataset
expects (see shard_writer.py), so fine-tuning needs zero pipeline changes.
A sidecar trials.jsonl logs one line per episode (offset vector, steps,
verdict, kept-or-not) -- that file IS the zero-shot results table.

Works with the sim backend too (end-to-end pipeline validation without a
robot):
  ./isaaclab.sh -p record_real_episodes.py --backend sim \
      --task Isaac-Forge-PegInsert-TBCamera-v0 --mode policy \
      --checkpoint .../aug_v1/best.pt --norm_stats .../aug_v1/norm_stats.npz \
      --episodes 2 --out_dir /tmp/rec_smoke --shard_size 2 --keep all

Real robot (after FrankaRobotIO is implemented):
  python record_real_episodes.py --backend franka --task peg --mode policy \
      --checkpoint checkpoints/peg/best.pt --norm_stats checkpoints/peg/norm_stats.npz \
      --episodes 20 --max_steps 149 --inject_offset_mm 2.5 --interactive \
      --out_dir ~/real_data/PegInsert --shard_size 20 --keep all --real_time

Teleop UDP protocol (Phase 4): send one JSON datagram per policy tick to
127.0.0.1:<--teleop_port>, payload {"action": [a0..a6]} with each a in
[-1,1] (a[0:3]=delta-pos / pos_action_bounds=0.05m; a[3:6]=delta-rot;
a[6]=ignored). If no datagram arrives within one tick the previous action
is held. Send {"action": ..., "done": true} to end the episode.
"""
import argparse
import json
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student"))

POLICY_HZ = 15.0
POLICY_DT = 1.0 / POLICY_HZ
WRENCH_HORIZON = 32
SUBSTEPS = 8


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "franka"], required=True)
    ap.add_argument("--task", required=True,
                    help="sim: gym id (Isaac-Forge-*-TBCamera-v0); franka: peg|gear|nut")
    ap.add_argument("--mode", choices=["policy", "teleop"], required=True)
    ap.add_argument("--checkpoint", default=None, help="required for --mode policy")
    ap.add_argument("--norm_stats", default=None, help="required for --mode policy")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=149,
                    help="sim episode lengths: peg 149 / gear 299 / nut 449")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=20)
    ap.add_argument("--shard_start_idx", type=int, default=0)
    ap.add_argument("--keep", choices=["all", "success"], default="all",
                    help="teleop demos usually 'success'; zero-shot logging 'all'")
    ap.add_argument("--interactive", action="store_true",
                    help="ask operator for success verdict after every episode (real robot)")
    ap.add_argument("--inject_offset_mm", type=float, default=0.0,
                    help="magnitude of a random-direction offset injected into the estimated "
                         "target frame each episode (franka backend only)")
    ap.add_argument("--offset_seed", type=int, default=0)
    ap.add_argument("--teleop_port", type=int, default=5555)
    ap.add_argument("--real_time", action="store_true", help="pace to 15Hz (mandatory on franka)")
    ap.add_argument("--noise_mm", type=float, default=1.0, help="sim backend env noise")
    ap.add_argument("--dyn_rand", choices=["on", "off"], default="on", help="sim backend")
    ap.add_argument("--protocol_seed", type=int, default=42, help="sim backend")
    ap.add_argument("--device", default=None, help="sim backend device override (e.g. cuda:0)")
    return ap.parse_known_args()


class TeleopUDP:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.last = [0.0] * 7
        self.done = False
        print(f"[teleop] listening on udp://127.0.0.1:{port} "
              f'(send {{"action":[7 floats in -1..1]}}, optional "done":true)', flush=True)

    def get(self):
        while True:  # drain to newest
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                msg = json.loads(data.decode())
                if "action" in msg and len(msg["action"]) == 7:
                    self.last = [max(-1.0, min(1.0, float(v))) for v in msg["action"]]
                if msg.get("done"):
                    self.done = True
            except (ValueError, KeyError):
                print("[teleop] bad datagram ignored", flush=True)
        return list(self.last)

    def reset(self):
        self.last = [0.0] * 7
        self.done = False


def main():
    args, unknown = parse_args()
    import numpy as np

    # ---- backend ----
    if args.backend == "sim":
        from isaaclab.app import AppLauncher  # must exist before isaaclab imports

        app_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(app_parser)
        app_args = app_parser.parse_args(unknown + ["--headless"])
        app_args.enable_cameras = True
        app_launcher = AppLauncher(app_args)
        _sim_app = app_launcher.app  # noqa: F841
        import torch
        from sim_io import SimRobotIO
        io = SimRobotIO(args.task, noise_mm=args.noise_mm, dyn_rand=args.dyn_rand,
                        protocol_seed=args.protocol_seed, device=args.device)
    else:
        import torch
        from franka_io_stub import FrankaRobotIO
        io = FrankaRobotIO(task=args.task)

    from shard_writer import ShardWriter, encode_jpeg_rgb

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- action source ----
    policy = None
    norm = None
    teleop = None
    if args.mode == "policy":
        assert args.checkpoint and args.norm_stats, "--mode policy needs --checkpoint/--norm_stats"
        from deploy_loop import build_policy
        policy, norm = build_policy(args.checkpoint, args.norm_stats, device)
    else:
        teleop = TeleopUDP(args.teleop_port)

    writer = ShardWriter(args.out_dir, shard_size=args.shard_size, start_idx=args.shard_start_idx)
    trials_path = os.path.join(os.path.abspath(os.path.expanduser(args.out_dir)), "trials.jsonl")
    trials = open(trials_path, "a", buffering=1)
    rng = np.random.default_rng(args.offset_seed)

    kept = 0
    for ep in range(args.episodes):
        # per-episode offset injection (real noise-axis experiments)
        offset_mm_vec = [0.0, 0.0, 0.0]
        if args.inject_offset_mm > 0:
            d = rng.normal(size=3)
            d /= max(float(np.linalg.norm(d)), 1e-9)
            offset_m = d * args.inject_offset_mm / 1000.0
            applied = io.set_frame_offset(offset_m.tolist())
            if not applied:
                # Fail LOUD: silently proceeding would log inject_offset_mm=X in
                # trials.jsonl while the actually-applied offset is zero --
                # corrupting the zero-shot results table.
                raise RuntimeError(
                    "--inject_offset_mm requested but this backend's set_frame_offset "
                    "returned False (unsupported). Aborting to protect protocol "
                    "integrity. (Sim noise is driven by --noise_mm instead.)")
            offset_mm_vec = (offset_m * 1000.0).tolist()

        io.reset()
        if teleop:
            teleop.reset()
            print(f"[record] episode {ep+1}/{args.episodes}: teleop is LIVE, drive now", flush=True)

        obs_list, wrench_list, tp_list, wrist_list, act_list = [], [], [], [], []
        wrench_win = torch.zeros(1, WRENCH_HORIZON, 6, device=device)
        steps = 0
        while not io.is_done() and steps < args.max_steps and not (teleop and teleop.done):
            t0 = time.time()
            # torch.as_tensor is the ONE conversion that accepts numpy arrays,
            # CPU tensors, and CUDA tensors uniformly -- np.asarray() would
            # crash on CUDA tensors from the sim backend.
            step8 = io.get_wrench_step()          # (8,6) raw, this step
            step8_t = torch.as_tensor(step8, dtype=torch.float32).to(device).reshape(1, SUBSTEPS, 6)
            wrench_win = torch.cat([wrench_win[:, SUBSTEPS:], step8_t], dim=1)
            obs = io.get_obs24()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device).reshape(1, -1)
            images = io.get_images()
            tp = np.asarray(images["tp"] if not torch.is_tensor(images["tp"]) else images["tp"].cpu().numpy())
            wrist = np.asarray(images["wrist"] if not torch.is_tensor(images["wrist"]) else images["wrist"].cpu().numpy())

            if args.mode == "policy":
                tp_t = torch.as_tensor(tp, device=device).unsqueeze(0)
                wrist_t = torch.as_tensor(wrist, device=device).unsqueeze(0)
                state_n = (obs_t - norm["state_mean"]) / norm["state_std"]
                wrench_n = (wrench_win - norm["wrench_mean"]) / norm["wrench_std"]
                with torch.inference_mode():
                    action = policy(state_n, wrench_n, tp_t, wrist_t)[0]
                action_np = action.detach().cpu().numpy().astype(np.float32)
            else:
                action_np = np.asarray(teleop.get(), dtype=np.float32)
                action = torch.as_tensor(action_np, device=device)

            io.send_action(action)

            obs_list.append(obs_t[0].detach().cpu().numpy().astype(np.float32))
            wrench_list.append(step8_t[0].detach().cpu().numpy().astype(np.float32))
            tp_list.append(encode_jpeg_rgb(tp.astype(np.uint8)))
            wrist_list.append(encode_jpeg_rgb(wrist.astype(np.uint8)))
            act_list.append(action_np)
            steps += 1

            if args.real_time:
                dt = time.time() - t0
                if dt < POLICY_DT:
                    time.sleep(POLICY_DT - dt)

        # ---- verdict ----
        verdict = None
        if args.backend == "sim" and hasattr(io, "last_success"):
            s = io.last_success()
            verdict = "success" if s else ("fail" if s is not None else "unknown")
        if args.interactive:
            ans = ""
            while ans not in ("y", "n", "d"):
                ans = input(f"[record] episode {ep+1}: success? [y=success / n=fail / d=discard] ").strip().lower()
            verdict = {"y": "success", "n": "fail", "d": "discard"}[ans]
        if verdict is None:
            verdict = "unknown"

        keep_this = (verdict != "discard") and (args.keep == "all" or verdict == "success")
        if keep_this and steps > 0:
            writer.add_episode(obs_list, wrench_list, tp_list, wrist_list, act_list,
                               noise_std_mm=args.inject_offset_mm)
            kept += 1
        trials.write(json.dumps({
            "episode": ep, "steps": steps, "verdict": verdict, "kept": keep_this,
            "offset_mm_vec": offset_mm_vec, "inject_offset_mm": args.inject_offset_mm,
            "mode": args.mode, "backend": args.backend, "ts": time.time(),
        }) + "\n")
        print(f"RECORD_EP ep={ep+1}/{args.episodes} steps={steps} verdict={verdict} kept={keep_this}", flush=True)

    writer.close()
    trials.close()
    io.close()
    print(f"RECORD_DONE episodes={args.episodes} kept={kept} out_dir={args.out_dir} trials={trials_path}", flush=True)
    if args.backend == "sim":
        os._exit(0)  # skip Kit teardown hang


if __name__ == "__main__":
    main()
