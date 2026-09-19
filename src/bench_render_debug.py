_DESC_ = """Benchmark ForgeCameraEnv render throughput (Phase 0).

For each env count: build ForgeCameraEnv, warm up, then time policy steps with
random actions while fetching both camera images every step — that IS the
Phase 2 data-collection workload. Prints a markdown table at the end.

Safest usage is ONE env count per invocation (kit teardown between in-process
env builds can be flaky) — see run_bench.sh. Multiple counts in one process are
attempted sequentially with env.close() in between and OOM guarded.

  cd <IsaacLab repo root> && ./isaaclab.sh -p ~/forge_ts/src/bench_render.py \\
      --headless --task_variant peg --env_counts 32 --steps 200
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=_DESC_)
parser.add_argument("--task_variant", type=str, default="peg", choices=["peg", "gear", "nut"])
parser.add_argument(
    "--env_counts", type=str, default="32,48,64",
    help="comma-separated env counts; prefer one per invocation (run_bench.sh)",
)
parser.add_argument(
    "--steps", type=int, default=120,
    help="timed policy steps per env count (default 120: warmup(20)+120=140 stays "
    "below peg's 150-step episode timeout so no task embeds a reset+IK-settle "
    "outlier in the timed window; gear/nut timeouts are 300/450 so unaffected)",
)
parser.add_argument("--warmup", type=int, default=20, help="untimed warm-up policy steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # TiledCamera needs RTX sensors even when headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Below requires the running app."""
import gc
import os
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_camera_env_debug_norender import (  # noqa: E402
    ForgeCameraEnv,
    ForgeCameraTaskPegInsertCfg,
    ForgeCameraTaskGearMeshCfg,
    ForgeCameraTaskNutThreadCfg,
)

CFGS = {
    "peg": ForgeCameraTaskPegInsertCfg,
    "gear": ForgeCameraTaskGearMeshCfg,
    "nut": ForgeCameraTaskNutThreadCfg,
}


def gpu_mem_gib():
    """(used, total) GiB for the whole device — includes kit allocations that
    torch.cuda.max_memory_allocated cannot see."""
    free_b, total_b = torch.cuda.mem_get_info()
    return (total_b - free_b) / 2**30, total_b / 2**30


def bench_one(num_envs):
    cfg = CFGS[args_cli.task_variant]()
    cfg.scene.num_envs = num_envs
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    env = ForgeCameraEnv(cfg, render_mode=None)
    try:
        env.reset()
        with torch.inference_mode():
            for _ in range(args_cli.warmup):
                act = torch.rand((env.num_envs, 7), device=env.device) * 2.0 - 1.0
                env.step(act)
                env.get_camera_images()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(args_cli.steps):
                act = torch.rand((env.num_envs, 7), device=env.device) * 2.0 - 1.0
                env.step(act)
                env.get_camera_images()
                env.get_wrench_history()  # also part of the Phase 2 workload (~free)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
        steps_s = args_cli.steps / elapsed
        used_gib, total_gib = gpu_mem_gib()
        return {
            "num_envs": num_envs,
            "steps_s": steps_s,
            "env_steps_s": steps_s * num_envs,
            "frames_s": steps_s * num_envs * 2,  # 2 cameras per env
            "torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_used_gib": used_gib,
            "gpu_total_gib": total_gib,
        }
    finally:
        env.close()


def main():
    counts = [int(x) for x in args_cli.env_counts.split(",") if x.strip()]
    if len(counts) > 1:
        print(
            "[WARN] multiple env counts in one process rely on kit teardown between "
            "builds; if this wedges, use run_bench.sh (one count per invocation).",
            flush=True,
        )

    results = []
    for n in counts:
        print(f"[BENCH] task={args_cli.task_variant} num_envs={n} steps={args_cli.steps}", flush=True)
        try:
            res = bench_one(n)
            results.append(res)
            print(
                f"[BENCH] num_envs={n} policy_steps/s={res['steps_s']:.2f} "
                f"env_steps/s={res['env_steps_s']:.1f} frames/s={res['frames_s']:.1f} "
                f"torch_peak={res['torch_peak_gib']:.2f}GiB "
                f"gpu_used={res['gpu_used_gib']:.2f}/{res['gpu_total_gib']:.1f}GiB",
                flush=True,
            )
        except Exception as e:  # OOM / kit failure: record and try the next count
            traceback.print_exc()
            results.append({"num_envs": n, "error": f"{type(e).__name__}: {e}"})
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n## ForgeCameraEnv render throughput "
        f"(task={args_cli.task_variant}, steps={args_cli.steps}, 2x 256x256 rgb TiledCameras)"
    )
    print("| num_envs | policy steps/s | env-steps/s | frames/s | torch peak alloc (GiB) | GPU used/total (GiB) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['num_envs']} | FAILED: {r['error']} | - | - | - | - |")
        else:
            print(
                f"| {r['num_envs']} | {r['steps_s']:.2f} | {r['env_steps_s']:.1f} | {r['frames_s']:.1f} "
                f"| {r['torch_peak_gib']:.2f} | {r['gpu_used_gib']:.2f}/{r['gpu_total_gib']:.1f} |"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
