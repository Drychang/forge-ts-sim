"""Inference-latency measurement for the student policy.

Closes the "is the 15Hz real-time claim actually true?" gap. Measures the
deployed forward path only -- exactly what runs per control tick on the robot:
    uint8 dual-camera frames + 32x6 wrench window + 24-dim state -> 7-dim action
Normalisation happens inside the module, so no host-side preprocessing is hidden
from the timing. JPEG decode is NOT included (on the real robot the frames
arrive as raw tensors from the camera driver, not as JPEGs -- JPEG only exists in
the offline dataset).

Reports p50/p95/p99 over N timed iterations after warmup, with CUDA synchronised
per iteration so the numbers are wall-clock per-tick latency, not throughput.
Also times the three P1 architectures so the equivalence claim gains a compute
axis: if concat/act are statistically indistinguishable in SR, cost matters.

Usage: bench_latency.py [--iters 300] [--archs fmt,concat,act] [--device cuda]
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/forge_ts/src/student"))
from student_archs import build_policy          # noqa: E402
from student_fmt import StudentFMTConfig        # noqa: E402

CKPTS = {
    "fmt": "~/forge_ts/student_ckpts/peg/gate2_seed0/best.pt",
    "concat": "~/forge_ts/student_ckpts/peg/p1_concat/best.pt",
    "act": "~/forge_ts/student_ckpts/peg/p1_act/best.pt",
}


def bench(arch, iters, device, warmup=30):
    cfg = StudentFMTConfig(pretrained_backbone=False)  # weights come from the ckpt
    model = build_policy(arch, cfg).to(device).eval()

    path = os.path.expanduser(CKPTS[arch])
    loaded = "random-init"
    if os.path.isfile(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = f"{os.path.basename(os.path.dirname(path))} (missing={len(missing)}, unexpected={len(unexpected)})"

    n_params = sum(p.numel() for p in model.parameters())
    state = torch.randn(1, cfg.state_dim, device=device)
    wrench = torch.randn(1, cfg.wrench_horizon, cfg.wrench_dim, device=device)
    tp = torch.randint(0, 255, (1, 256, 256, 3), dtype=torch.uint8, device=device)
    wr = torch.randint(0, 255, (1, 256, 256, 3), dtype=torch.uint8, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(state, wrench, tp, wr)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(state, wrench, tp, wr)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)

    ts.sort()
    return {
        "arch": arch, "ckpt": loaded, "params_M": n_params / 1e6,
        "p50": ts[len(ts) // 2], "p95": ts[int(0.95 * len(ts))], "p99": ts[int(0.99 * len(ts))],
        "mean": statistics.fmean(ts), "min": ts[0], "max": ts[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--archs", default="fmt,concat,act")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    if dev.startswith("cuda"):
        print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    print(f"batch=1, dual 256x256 uint8 RGB + 32x6 wrench + 24-dim state, "
          f"{args.iters} timed iters, CUDA-synced per iter\n", flush=True)

    rows = [bench(a, args.iters, dev) for a in args.archs.split(",")]
    hdr = f"{'arch':8s} {'params':>8s} {'p50 ms':>8s} {'p95 ms':>8s} {'p99 ms':>8s} {'Hz@p95':>8s}  checkpoint"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arch']:8s} {r['params_M']:7.2f}M {r['p50']:8.2f} {r['p95']:8.2f} {r['p99']:8.2f} "
              f"{1000.0 / r['p95']:8.1f}  {r['ckpt']}")
    print("\nFORGE control rate is 15 Hz -> the per-tick budget is 66.7 ms.")
    for r in rows:
        head = 66.7 / r["p95"]
        print(f"  {r['arch']:8s} p95 {r['p95']:6.2f} ms = {100.0 * r['p95'] / 66.7:5.1f}% of budget "
              f"({head:.1f}x headroom)")
    print("\nLATENCY_JSON " + str({r["arch"]: round(r["p95"], 3) for r in rows}))


if __name__ == "__main__":
    main()
