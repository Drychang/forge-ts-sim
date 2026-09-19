#!/usr/bin/env python3
"""x27 -- does the +8.4 mm y frame compensation actually work?

x26 found the TP camera adds a constant y bias of -8.40 mm (stable across all
three heights) plus 6.4 mm of pose-dependent scatter, and I recommended the
robot side shift the taught fixture frame by +8.4 mm in y to cancel it.

That recommendation was never tested. It should be, before anyone moves a
fixture on the strength of it -- and it can be, offline, on the same 75 real
images.

The mechanism, spelled out, because the bookkeeping is easy to get backwards:

    obs[0:3] = fingertip - taught_frame        so shifting the frame by +d
                                               shifts obs[0:3] by -d
    belief   = action[0:3] * 50 mm             expressed relative to the frame
    where the robot ends up = taught_frame + belief

  So with the shift:  end = (true + d) + belief(obs - d)
      and the error is:  err(d) = d + belief(obs - d) - belief_target

If the policy simply ignored the shift, err would grow by exactly d and the
compensation would be self-defeating. If it tracks the state, belief moves by
roughly -d and the error shrinks. Which one happens is an empirical question
about this network, so ask it.

Sweeps d over a range so the optimum is visible rather than assumed.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/forge_ts/src"))
from student.student_fmt import StudentFMTPolicy, StudentFMTConfig

WRENCH_MEAN = np.array([2.9480, 2.5299, 1.7696, -0.25120, 0.30680, 0.0004])
WRENCH_STD = np.array([2.4241, 2.3212, 2.4910, 0.2497, 0.2682, 0.2179])


def build_state(rel_mm, quat, rng):
    s = np.zeros(24, np.float32)
    s[0:3] = np.asarray(rel_mm) / 1000.0
    s[3:7] = np.asarray(quat, dtype=np.float64)
    s[3] = 0.0; s[6] = 0.0
    s[7:10] = rng.normal(0.0, 0.030, 3)
    s[10] = 0.0; s[11] = 0.0
    s[12] = rng.normal(0.0, 0.1207)
    s[13:16] = [rng.normal(2.941, 2.544), rng.normal(2.527, 2.445),
                rng.normal(1.773, 2.455)]
    s[16] = 7.47862
    s[17:24] = 0.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=os.path.expanduser("~/ig"))
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/best.pt"))
    ap.add_argument("--norm", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/norm_stats.npz"))
    ap.add_argument("--draws", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    st = dict(np.load(a.norm))
    s_mean = torch.as_tensor(st["state_mean"], dtype=torch.float32)
    s_std = torch.as_tensor(st["state_std"], dtype=torch.float32)
    w_mean = torch.as_tensor(st["wrench_mean"], dtype=torch.float32)
    w_std = torch.as_tensor(st["wrench_std"], dtype=torch.float32)

    ck = torch.load(a.ckpt, map_location="cpu")
    pol = StudentFMTPolicy(StudentFMTConfig(**ck["config"]))
    pol.load_state_dict(ck["model"]); pol.eval()

    grid = json.load(open(os.path.join(a.grid, "imgrid.json")))
    samples = grid["samples"]

    # a modest pose subset keeps this to a few minutes; spread over the grid
    subset = samples[::3]
    print(f"[x27] {len(subset)} of {len(samples)} poses, draws={a.draws}", flush=True)

    imgs = []
    for sm in subset:
        dx, dy, z = sm["cmd_dx_dy_z_mm"]
        k = "x%+04d_y%+04d_z%+04d" % (dx, dy, z)
        tp = os.path.join(a.grid, k + "_tp_policy256.png")
        wr = os.path.join(a.grid, k + "_wrist_policy256.png")
        if not (os.path.exists(tp) and os.path.exists(wr)):
            imgs.append(None); continue
        imgs.append((np.asarray(Image.open(tp).convert("RGB"), np.uint8),
                     np.asarray(Image.open(wr).convert("RGB"), np.uint8)))

    print()
    print("shift d applied to the taught frame in +y; end-position error is")
    print("  err = d + belief_y(obs shifted by -d)      (target belief_y = 0)")
    print()
    print("   d (mm) | mean end-err y | |err| mean | |err| p90 | belief_y mean")
    print("  " + "-" * 68)

    best = None
    for d in (0.0, 2.0, 4.0, 6.0, 8.4, 11.0, 14.0):
        errs, bys = [], []
        rng = np.random.default_rng(23)
        for sm, im in zip(subset, imgs):
            if im is None:
                continue
            rel = np.array(sm["rel_to_tip_mm"], float)
            rel_shift = rel.copy()
            rel_shift[1] -= d                     # frame moved +d  ->  obs moves -d
            acc = []
            for _ in range(a.draws):
                s = build_state(rel_shift, sm["quat_wxyz"], rng)
                sn = ((torch.from_numpy(s) - s_mean) / s_std).unsqueeze(0)
                wraw = rng.normal(WRENCH_MEAN, WRENCH_STD, (32, 6)).astype(np.float32)
                wn = ((torch.from_numpy(wraw) - w_mean) / w_std).unsqueeze(0)
                with torch.inference_mode():
                    act = pol(sn, wn, torch.from_numpy(im[0]).unsqueeze(0),
                              torch.from_numpy(im[1]).unsqueeze(0))
                acc.append(float(act[0, 1]) * 50.0)
            by = float(np.mean(acc))
            bys.append(by)
            errs.append(d + by)
        e = np.array(errs)
        row = (d, e.mean(), np.abs(e).mean(), np.percentile(np.abs(e), 90),
               float(np.mean(bys)))
        print("  %7.1f | %14.2f | %10.2f | %9.2f | %13.2f" % row, flush=True)
        if best is None or np.abs(e).mean() < best[1]:
            best = (d, np.abs(e).mean())

    print()
    print("  best d = %.1f mm  (|err| mean %.2f mm)" % best)
    print()
    print("  If |err| is flat or rising with d, the policy is NOT tracking the")
    print("  shift and the compensation cannot work -- withdraw the advice.")
    print("  If it dips near 8.4, the recommendation holds.")


if __name__ == "__main__":
    main()
