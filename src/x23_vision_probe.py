#!/usr/bin/env python3
"""x23 -- clean vision-only probe.

x22 showed the policy's target belief goes 27-31 mm laterally with the real
images, but that test is contaminated: prev_actions was fed from the recorded
(already-drifted) trajectory, so the policy is partly just continuing its own
earlier mistake.

This removes the contamination. The state is PINNED to a canonical, perfectly
on-axis pose -- fingertip 30 mm directly above the gear post, tool vertical,
zero velocity, zero prev_actions, nominal force threshold -- and the ONLY thing
that varies is the image pair.

If the policy still commands a large lateral target under a state that says
"you are already centred", then vision alone is mislocalising the post, and no
amount of obs-assembly fixing on the robot side will help.

Reference: in 404 sim episodes the final lateral belief was 3.32 mm mean,
7.82 mm p95, 11.64 mm max.
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


def canonical_state(rel_mm, rng):
    """On-axis approach pose. Degenerate dims 3,6,10,11,20,21 held at zero."""
    s = np.zeros(24, np.float32)
    s[0:3] = np.array(rel_mm) / 1000.0
    # tool vertical: quat (w,x,y,z) with w=z=0 -> (0, 1, 0, 0) is roll=180, pitch=0
    s[3] = 0.0; s[4] = 1.0; s[5] = 0.0; s[6] = 0.0
    s[7:10] = rng.normal(0.0, 0.030, 3)      # ee_linvel: all three live
    s[10] = 0.0; s[11] = 0.0                 # ee_angvel x,y degenerate
    s[12] = rng.normal(0.0, 0.1207)
    s[13:16] = [0.0, 0.0, 0.5]               # near free space, tiny contact
    s[16] = 7.479                            # nominal force threshold
    s[17:24] = 0.0                           # NO prior action -- the key change
    return s


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.expanduser("~/forge_run02/run02"))
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/best.pt"))
    ap.add_argument("--norm", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/norm_stats.npz"))
    ap.add_argument("--draws", type=int, default=16)
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

    frames = sorted(int(f[1:5]) for f in os.listdir(os.path.join(a.run, "frames"))
                    if f.endswith("_tp.png"))
    blank = np.zeros((256, 256, 3), np.uint8)
    rng = np.random.default_rng(7)

    print("STATE IS PINNED ON-AXIS: rel = [0, 0, +30] mm, prev_actions = 0")
    print("so any lateral command below is produced by the IMAGES alone.")
    print()
    print("step |  both real cams   |  wrist only       |  tp only          |  no vision")
    print("-" * 80)

    agg = {k: [] for k in ("real", "wristonly", "tponly", "none")}
    for step in frames:
        tp = load(os.path.join(a.run, "frames", f"s{step:04d}_tp.png"))
        wr = load(os.path.join(a.run, "frames", f"s{step:04d}_wrist.png"))
        res = {}
        for key, (i1, i2) in (("real", (tp, wr)), ("wristonly", (blank, wr)),
                              ("tponly", (tp, blank)), ("none", (blank, blank))):
            outs = []
            for _ in range(a.draws):
                s = canonical_state([0, 0, 30], rng)
                sn = ((torch.from_numpy(s) - s_mean) / s_std).unsqueeze(0)
                wraw = rng.normal(WRENCH_MEAN, WRENCH_STD, (32, 6)).astype(np.float32)
                wn = ((torch.from_numpy(wraw) - w_mean) / w_std).unsqueeze(0)
                with torch.inference_mode():
                    act = pol(sn, wn,
                              torch.from_numpy(i1).unsqueeze(0),
                              torch.from_numpy(i2).unsqueeze(0))
                outs.append(act[0, :3].numpy() * 50.0)
            m = np.mean(outs, 0)
            res[key] = m
            agg[key].append(m)
        f = lambda m: "[%+6.1f %+6.1f]" % (m[0], m[1])
        print("%4d | %s | %s | %s | %s"
              % (step, f(res["real"]), f(res["wristonly"]), f(res["tponly"]), f(res["none"])),
              flush=True)

    print()
    print("=" * 80)
    print("lateral |xy| of the target belief, with the state pinned on-axis:")
    for k, label in (("real", "both real cameras"), ("wristonly", "wrist only"),
                     ("tponly", "TP only"), ("none", "no vision (state only)")):
        arr = np.array(agg[k])
        lat = np.hypot(arr[:, 0], arr[:, 1])
        print("  %-24s mean %5.1f   median %5.1f   max %5.1f mm"
              % (label, lat.mean(), np.median(lat), lat.max()))
    print()
    print("  sim reference (404 episodes): final lateral 3.32 mean / 7.82 p95 / 11.64 max")


if __name__ == "__main__":
    main()
