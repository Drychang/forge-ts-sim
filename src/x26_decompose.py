#!/usr/bin/env python3
"""x26 -- decompose the imgrid belief map.

x25's raw summary is misleading on its own: with the state varying across the
grid, the belief moves for two quite different reasons, and they need separating
before anything can be concluded.

  STATE branch   obs[0:3] changes with arm pose, so the action changes too.
                 That is the policy working as designed, not an error.
  VISION branch  the extra displacement the images add on top.

So the quantity that matters is the DIFFERENCE per pose:

    vision_effect(pose) = belief(with images) - belief(no images)

and then, for the robot side's actual question:

    constant part  = mean over poses  -> correctable by shifting the taught frame
    varying part   = sd over poses    -> a localisation failure, no constant fixes it

The within-pose draw spread (from sampling the unrecorded obs dims) is reported
alongside, so it is visible how much of the "varying part" is just my own
sampling uncertainty rather than real pose dependence.
"""
import json
import os

import numpy as np

P = os.path.expanduser("~/forge_ts/logs/x25_map.json")
d = json.load(open(P))
poses = d["poses"]

cmd = np.array([p["cmd"] for p in poses], float)          # dx, dy, z (commanded)
rel = np.array([p["rel_to_tip_mm"] for p in poses], float)
tilt = np.array([p["tilt_deg"] for p in poses], float)


def arr(cond, field="mean"):
    return np.array([p["belief"][cond][field] for p in poses], float)


B = {c: arr(c) for c in ("both", "tp_only", "wrist_only", "none")}
S = {c: arr(c, "std") for c in ("both", "tp_only", "wrist_only", "none")}

print("=" * 78)
print("x26  belief decomposition over the 75-pose imgrid   (jitcam_real_s2)")
print("=" * 78)
print(f"poses={len(poses)}  fixture_tip_m={d['fixture_tip_m']}")
print(f"tilt over the grid: {tilt.min():.2f} - {tilt.max():.2f} deg")
print()

print("--- 1. within-pose draw spread (MY sampling uncertainty, not the robot) ---")
for c in ("both", "tp_only", "wrist_only", "none"):
    print("  %-11s per-pose sd of draws: x %.2f  y %.2f  z %.2f mm"
          % (c, S[c][:, 0].mean(), S[c][:, 1].mean(), S[c][:, 2].mean()))
print()

print("--- 2. does the STATE branch alone already explain the movement? ---")
for c in ("none", "both"):
    rx = np.corrcoef(cmd[:, 0], B[c][:, 0])[0, 1]
    ry = np.corrcoef(cmd[:, 1], B[c][:, 1])[0, 1]
    print("  %-11s  r(dx, belief_x)=%+.2f   r(dy, belief_y)=%+.2f" % (c, rx, ry))
print("  -> if 'none' shows the same correlation, that movement is the state")
print("     branch responding to obs[0:3], i.e. the policy working normally.")
print()

print("--- 3. VISION EFFECT = belief(images) - belief(no images), per pose ---")
print("    %-14s %-26s %-26s" % ("", "constant part (mean)", "varying part (sd)"))
for c in ("both", "tp_only", "wrist_only"):
    V = B[c] - B["none"]
    lat = np.hypot(V[:, 0], V[:, 1])
    print("    %-14s x %+6.2f  y %+6.2f mm      x %6.2f  y %6.2f mm   |xy| mean %5.2f max %5.2f"
          % (c, V[:, 0].mean(), V[:, 1].mean(), V[:, 0].std(), V[:, 1].std(),
             lat.mean(), lat.max()))
print()

V = B["tp_only"] - B["none"]
cx, cy = V[:, 0].mean(), V[:, 1].mean()
sx, sy = V[:, 0].std(), V[:, 1].std()
print("--- 4. VERDICT for the TP camera ---")
print("    constant offset  : [%+.2f, %+.2f] mm  (|%.2f| mm)" % (cx, cy, np.hypot(cx, cy)))
print("    pose-dependent sd: [ %.2f,  %.2f] mm" % (sx, sy))
ratio = np.hypot(sx, sy) / max(np.hypot(cx, cy), 1e-6)
print("    varying / constant ratio = %.2f" % ratio)
if ratio > 1.5:
    print("    -> DOMINATED BY THE VARYING PART. A constant frame shift cannot fix this.")
elif ratio < 0.6:
    print("    -> DOMINATED BY THE CONSTANT PART. A frame shift of -[%.1f, %.1f] mm would help."
          % (cx, cy))
else:
    print("    -> MIXED. A frame shift removes some of it; a residual remains.")
print()

print("--- 5. does the vision effect track arm position? ---")
for j, ax in enumerate("xy"):
    for k, nm in enumerate(("dx", "dy", "z")):
        r = np.corrcoef(cmd[:, k], V[:, j])[0, 1]
        if abs(r) > 0.3:
            print("    tp vision effect_%s  vs %s : r=%+.2f" % (ax, nm, r))
r = np.corrcoef(tilt, np.hypot(V[:, 0], V[:, 1]))[0, 1]
print("    tp vision effect |xy| vs measured tilt : r=%+.2f" % r)
print()

print("--- 6. per-height slices of the TP vision effect ---")
for z in sorted(set(cmd[:, 2])):
    m = cmd[:, 2] == z
    lat = np.hypot(V[m, 0], V[m, 1])
    print("    z=%+5.0f mm (n=%2d): constant [%+6.2f %+6.2f]  |xy| mean %5.2f  max %5.2f"
          % (z, m.sum(), V[m, 0].mean(), V[m, 1].mean(), lat.mean(), lat.max()))
