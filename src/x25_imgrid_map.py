#!/usr/bin/env python3
"""x25 -- map the policy's target-belief error across the 75-pose imgrid.

The robot side cannot change the TP scene, so the question shifts from "does
fixing the scene help" to "is the TP error correctable". Those need different
answers:

  GLOBAL CONSTANT OFFSET  -> shift the taught fixture frame by -offset and the
                             policy lands on the shaft. Deployable today.
  VARIES WITH ARM POSE    -> not a frame error, it is a localisation failure.
                             No constant compensation exists.

x23 could not tell these apart: it pinned the state to a single canonical pose,
so every frame got the same state and only the image varied. The imgrid fixes
that -- it carries the TRUE `rel_to_tip_mm` for all 75 poses, so the state can
be the real one and the belief error becomes a function of arm position.

Belief is read straight off the action: action[0:3] * 50 mm is where the policy
thinks the target sits relative to the fixture frame.

Unrecorded obs dims (force vector, wrench window, prev_actions) are drawn from
the sim training distribution over several draws; the per-pose spread is
reported so it is visible whether the conclusion survives that uncertainty.
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
SIM_FTHR = 7.47862


def build_state(rel_mm, quat_wxyz, rng):
    """24-dim obs per the sim contract. Degenerate dims 3,6,10,11,20,21 = 0."""
    s = np.zeros(24, np.float32)
    s[0:3] = np.asarray(rel_mm) / 1000.0
    q = np.asarray(quat_wxyz, dtype=np.float64)
    s[3:7] = q
    s[3] = 0.0                                   # qw  degenerate
    s[6] = 0.0                                   # qz  degenerate
    s[7:10] = rng.normal(0.0, 0.030, 3)          # ee_linvel: all three live
    s[10] = 0.0; s[11] = 0.0                     # ee_angvel x,y degenerate
    s[12] = rng.normal(0.0, 0.1207)
    f = np.array([rng.normal(2.941, 2.544), rng.normal(2.527, 2.445),
                  rng.normal(1.773, 2.455)])
    s[13:16] = f
    s[16] = SIM_FTHR
    s[17:24] = 0.0                               # no prior action
    return s


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=os.path.expanduser("~/ig"))
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/best.pt"))
    ap.add_argument("--norm", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/norm_stats.npz"))
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/forge_ts/logs/x25_map.json"))
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
    tip = np.array(grid["fixture_tip_m"])
    print(f"[x25] {len(samples)} poses, fixture_tip_m={tip}, yaw={grid['yaw_deg']:.2f} deg",
          flush=True)

    blank = np.zeros((256, 256, 3), np.uint8)
    rng = np.random.default_rng(11)
    out = []

    for i, sm in enumerate(samples):
        dx, dy, z = sm["cmd_dx_dy_z_mm"]
        key = "x%+04d_y%+04d_z%+04d" % (dx, dy, z)
        ptp = os.path.join(a.grid, key + "_tp_policy256.png")
        pwr = os.path.join(a.grid, key + "_wrist_policy256.png")
        if not (os.path.exists(ptp) and os.path.exists(pwr)):
            print(f"[x25] MISSING {key}", flush=True); continue
        tp, wr = load(ptp), load(pwr)
        rel = sm["rel_to_tip_mm"]

        res = {}
        for cond, (i1, i2) in (("both", (tp, wr)), ("wrist_only", (blank, wr)),
                               ("tp_only", (tp, blank)), ("none", (blank, blank))):
            acc = []
            for _ in range(a.draws):
                s = build_state(rel, sm["quat_wxyz"], rng)
                sn = ((torch.from_numpy(s) - s_mean) / s_std).unsqueeze(0)
                wraw = rng.normal(WRENCH_MEAN, WRENCH_STD, (32, 6)).astype(np.float32)
                wn = ((torch.from_numpy(wraw) - w_mean) / w_std).unsqueeze(0)
                with torch.inference_mode():
                    act = pol(sn, wn, torch.from_numpy(i1).unsqueeze(0),
                              torch.from_numpy(i2).unsqueeze(0))
                acc.append(act[0, :3].numpy() * 50.0)
            res[cond] = {"mean": np.mean(acc, 0).tolist(),
                         "std": np.std(acc, 0).tolist()}

        out.append({"cmd": [dx, dy, z], "rel_to_tip_mm": rel,
                    "tilt_deg": sm["tilt_deg"], "belief": res})
        b = res["both"]["mean"]
        print("[x25] %2d/%d %s  rel=[%+6.1f %+6.1f %+6.1f]  belief=[%+6.1f %+6.1f %+6.1f]"
              % (i + 1, len(samples), key, rel[0], rel[1], rel[2], b[0], b[1], b[2]),
              flush=True)

    json.dump({"fixture_tip_m": tip.tolist(), "poses": out}, open(a.out, "w"), indent=1)
    print(f"\n[x25] saved -> {a.out}", flush=True)

    # ---- is the lateral error a constant, or does it move with the arm? ----
    print()
    print("=" * 76)
    for cond in ("both", "tp_only", "wrist_only", "none"):
        B = np.array([p["belief"][cond]["mean"] for p in out])
        lat = np.hypot(B[:, 0], B[:, 1])
        print("%-11s  belief xy: mean=[%+6.2f %+6.2f]  sd=[%5.2f %5.2f]  "
              "|xy| mean %5.2f max %5.2f mm"
              % (cond, B[:, 0].mean(), B[:, 1].mean(), B[:, 0].std(), B[:, 1].std(),
                 lat.mean(), lat.max()))
    print()
    print("A constant frame error shows up as a large mean with a SMALL sd.")
    print("A localisation failure shows up as a large sd -- no constant can fix it.")
    print()
    B = np.array([p["belief"]["both"]["mean"] for p in out])
    C = np.array([p["cmd"] for p in out], float)
    for j, axis in enumerate("xy"):
        for k, cname in enumerate(("dx", "dy", "z")):
            r = np.corrcoef(C[:, k], B[:, j])[0, 1]
            if abs(r) > 0.35:
                print("  belief_%s correlates with %s : r=%+.2f" % (axis, cname, r))


if __name__ == "__main__":
    main()
