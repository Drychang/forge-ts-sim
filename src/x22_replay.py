#!/usr/bin/env python3
"""x22 -- replay the real run's observations through jitcam_real_s2 and find out
whether the 31 mm lateral error comes from OBS ASSEMBLY or from the IMAGES.

Runs on CPU: GPU2 fell off the bus on Mercury and wedged the driver for new
CUDA contexts, but 26.5M params over ~21 frames is fine on CPU.

WHY THIS IS DECISIVE
--------------------
`action[0:3] * 50 mm` IS the policy's belief about where the target sits
relative to the fixture frame (forge_action: target_pre = action_frame +
clip(a[0:3],-1,1)*0.05). The Franka side recorded exactly that quantity as
`target_rel_fixture_mm`, so the policy's belief is directly observable.

So: assemble the 24-dim obs INDEPENDENTLY from the sim contract, feed it with
their recorded images, and compare the resulting belief against theirs.

  belief(mine + their images) ~= belief(theirs)
      -> their assembly is equivalent to mine; the images are what mislead it
  belief(mine + their images) != belief(theirs)
      -> their assembly differs from the contract; the diff localises the dim

Then ablate vision (blank both cameras) to see how much of the belief is
vision-driven at all.

The run only records the 8-dim robot state, not the assembled 24-dim obs, and
not the force vector or wrench history. Those unknown dims are SAMPLED from the
sim training distribution over many draws, and the spread is reported -- if the
conclusion is stable across draws it does not depend on the guess.
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

# --- exact per-dim sim statistics, measured from the forreal training shards ---
# degenerate dims (identically zero in training): 3 qw, 6 qz, 10 avx, 11 avy, 20 pa3, 21 pa4
SIM = {
    "lv":   (0.0,      0.03000),   # dims 7,8,9  -- ALL THREE live (not just z)
    "avz":  (0.00002,  0.12073),   # dim 12      -- dims 10,11 are exactly 0
    "fx":   (2.94138,  2.54388),
    "fy":   (2.52723,  2.44497),
    "fz":   (1.77260,  2.45545),
    "fthr": (7.47862,  1.41591),
    "pa0":  (0.20437,  0.14084),
    "pa1":  (-0.11128, 0.11614),
    "pa2":  (-0.27320, 0.16540),
    "pa5":  (0.21846,  0.11358),
    "pa6":  (0.70208,  0.57516),
}
WRENCH_MEAN = np.array([2.9480, 2.5299, 1.7696, -0.25120, 0.30680, 0.0004])
WRENCH_STD = np.array([2.4241, 2.3212, 2.4910, 0.2497, 0.2682, 0.2179])


def rpy_to_quat_wxyz(r, p, y):
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def build_state(tick, fixture, rng, prev_actions=None):
    """Assemble the 24-dim obs from the sim contract. Unknown dims are sampled."""
    s = np.zeros(24, np.float32)
    tcp = np.array(tick["state24"][0:3])
    s[0:3] = tcp - fixture                                   # fingertip_pos_rel_fixed

    r, p, y = tick["state24"][3:6]
    q = rpy_to_quat_wxyz(r, p, y)
    s[3:7] = q
    s[3] = 0.0                                               # qw  -- degenerate
    s[6] = 0.0                                               # qz  -- degenerate

    s[7:10] = rng.normal(SIM["lv"][0], SIM["lv"][1], 3)      # ee_linvel, all 3 live
    s[10] = 0.0                                              # avx -- degenerate
    s[11] = 0.0                                              # avy -- degenerate
    s[12] = rng.normal(*SIM["avz"])

    # force vector: only |F| was recorded, so draw a direction from the sim
    # distribution and rescale it to the measured magnitude
    f = np.array([rng.normal(*SIM["fx"]), rng.normal(*SIM["fy"]), rng.normal(*SIM["fz"])])
    n = np.linalg.norm(f)
    if n > 1e-6:
        f = f / n * float(tick["force_n"])
    s[13:16] = f
    s[16] = rng.normal(*SIM["fthr"])

    if prev_actions is None:
        s[17] = rng.normal(*SIM["pa0"]); s[18] = rng.normal(*SIM["pa1"])
        s[19] = rng.normal(*SIM["pa2"]); s[22] = rng.normal(*SIM["pa5"])
        s[23] = rng.normal(*SIM["pa6"])
    else:
        s[17:24] = prev_actions
    s[20] = 0.0                                              # pa3 -- degenerate
    s[21] = 0.0                                              # pa4 -- degenerate
    return s


def load_img(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    assert a.shape == (256, 256, 3), a.shape
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.expanduser("~/forge_run02/run02"))
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/best.pt"))
    ap.add_argument("--norm", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/norm_stats.npz"))
    ap.add_argument("--draws", type=int, default=32)
    a = ap.parse_args()

    dev = torch.device("cpu")
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))

    run = json.load(open(os.path.join(a.run, "run.json")))
    fixture = np.array(run["fixture_m"])
    ticks = {t["step"]: t for t in run["ticks"]}

    st = dict(np.load(a.norm))
    s_mean = torch.as_tensor(st["state_mean"], dtype=torch.float32)
    s_std = torch.as_tensor(st["state_std"], dtype=torch.float32)
    w_mean = torch.as_tensor(st["wrench_mean"], dtype=torch.float32)
    w_std = torch.as_tensor(st["wrench_std"], dtype=torch.float32)

    ck = torch.load(a.ckpt, map_location="cpu")
    pol = StudentFMTPolicy(StudentFMTConfig(**ck["config"]))
    pol.load_state_dict(ck["model"])
    pol.eval().to(dev)
    print(f"[x22] loaded {a.ckpt}  (epoch {ck.get('epoch')}, val {ck.get('best_val'):.5f})",
          flush=True)

    frames = sorted(int(f[1:5]) for f in os.listdir(os.path.join(a.run, "frames"))
                    if f.endswith("_tp.png"))
    print(f"[x22] {len(frames)} recorded frames: {frames}", flush=True)

    rng = np.random.default_rng(0)
    blank = np.zeros((256, 256, 3), np.uint8)

    # prev_actions[0:3] is recoverable: action[0:3] == target_rel_fixture_mm / 50.
    # The first version sampled it randomly, which injected a large spurious
    # spread into x; reconstruct it from the preceding tick instead.
    all_steps = sorted(ticks)
    prev_a = {}
    for idx, stp in enumerate(all_steps):
        if idx == 0:
            prev_a[stp] = None
            continue
        pv = np.array(ticks[all_steps[idx - 1]]["target_rel_fixture_mm"]) / 50.0
        pa = np.zeros(7, np.float32)
        pa[0:3] = np.clip(pv, -1, 1)
        pa[5] = SIM["pa5"][0]
        pa[6] = SIM["pa6"][0]
        prev_a[stp] = pa

    print()
    hdr = ("step | recorded target_rel (mm)        | REPLAY their imgs (mm)          "
           "| VISION BLANKED (mm)")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for step in frames:
        t = ticks[step]
        tp = load_img(os.path.join(a.run, "frames", f"s{step:04d}_tp.png"))
        wr = load_img(os.path.join(a.run, "frames", f"s{step:04d}_wrist.png"))

        outs = {"real": [], "blank": [], "no_tp": [], "no_wrist": []}
        for d in range(a.draws):
            s = build_state(t, fixture, rng, prev_actions=prev_a.get(step))
            sn = ((torch.from_numpy(s) - s_mean) / s_std).unsqueeze(0)
            wraw = rng.normal(WRENCH_MEAN, WRENCH_STD, size=(32, 6)).astype(np.float32)
            wn = ((torch.from_numpy(wraw) - w_mean) / w_std).unsqueeze(0)
            for key, (i1, i2) in (("real", (tp, wr)), ("blank", (blank, blank)),
                                  ("no_tp", (blank, wr)), ("no_wrist", (tp, blank))):
                with torch.inference_mode():
                    act = pol(sn,
                              wn,
                              torch.from_numpy(i1).unsqueeze(0),
                              torch.from_numpy(i2).unsqueeze(0))
                outs[key].append(act[0, :3].numpy() * 1000.0 * 0.05)

        rec = np.array(t["target_rel_fixture_mm"])
        mr = np.mean(outs["real"], 0); sr = np.std(outs["real"], 0)
        mb = np.mean(outs["blank"], 0)
        mnt = np.mean(outs["no_tp"], 0)
        mnw = np.mean(outs["no_wrist"], 0)
        rows.append((step, rec, mr, sr, mb, mnt, mnw))
        print("%4d | [%+7.1f %+7.1f %+7.1f]         | [%+7.1f %+7.1f %+7.1f] +-%.1f  "
              "| [%+7.1f %+7.1f %+7.1f]"
              % (step, rec[0], rec[1], rec[2], mr[0], mr[1], mr[2],
                 float(sr.mean()), mb[0], mb[1], mb[2]), flush=True)

    print()
    print("=" * 78)
    rec_all = np.array([r[1] for r in rows])
    rep_all = np.array([r[2] for r in rows])
    bl_all = np.array([r[4] for r in rows])
    nt_all = np.array([r[5] for r in rows])
    nw_all = np.array([r[6] for r in rows])
    d = np.linalg.norm(rep_all - rec_all, axis=1)
    print("REPLAY vs RECORDED  |diff| per frame: mean %.2f  max %.2f mm" % (d.mean(), d.max()))
    print("  -> small = my independent assembly reproduces theirs (assembly OK)")
    print("  -> large = their assembly differs from the sim contract")
    print()
    print("lateral (xy) magnitude of the policy's target belief:")
    for nm, arr in (("recorded (theirs)", rec_all),
                    ("replay  both cams", rep_all),
                    ("TP blanked   (wrist only)", nt_all),
                    ("WRIST blanked (tp only)", nw_all),
                    ("both blanked (state only)", bl_all)):
        lat = np.hypot(arr[:, 0], arr[:, 1])
        print("  %-22s first %5.1f   last %5.1f   max %5.1f mm"
              % (nm, lat[0], lat[-1], lat.max()))
    print()
    print("sim reference: final lateral 3.32 +- , p95 7.82, max 11.64 mm (404 episodes)")


if __name__ == "__main__":
    main()
