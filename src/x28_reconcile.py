#!/usr/bin/env python3
"""x28 -- reconcile the d=8.4 (offline probe) vs d=14 (real robot) conflict.

Three things x27 did that this fixes:

  1. x27 ran against the STALE fixture value (imgrid.json fixture_tip_m =
     [590.86, -18.01, 120.06]).  The robot side's d-scan ran against
     G_dscan = [597.55/597.83, -18.78/-19.1, 119.8/120.1].  Delta is
     [+6.97, -1.09, +0.04] mm.  Run BOTH origins.
  2. x27 swept d only to 14, so it could not see the far side of the optimum.
     The robot's own scan goes to 22 and falls off there.  Sweep to 30.
  3. x27 kept ONLY the y channel of the action and threw away x and z.
     The real success gate is radial bore offset <= 3.0 mm AND z <= -4.0 mm.
     Keep all three, and score with the real gate as well as with |err_y|.

Per-pose values are saved so standard errors and PAIRED comparisons between
d values are possible -- x27 reported bare means with no uncertainty at all.
"""
import json, os, sys, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/forge_ts/src"))
from student.student_fmt import StudentFMTPolicy, StudentFMTConfig

WRENCH_MEAN = np.array([2.9480, 2.5299, 1.7696, -0.25120, 0.30680, 0.0004])
WRENCH_STD  = np.array([2.4241, 2.3212, 2.4910, 0.2497, 0.2682, 0.2179])
SIM_FTHR = 7.47862

# imgrid.json was built against this frame (== the robot side's G_stale)
T_SCAN = np.array([590.86, -18.01, 120.06])
# the frame the robot side's d-scan actually ran against (G_dscan)
T_DSCAN = np.array([597.83, -19.10, 120.10])


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
    s[16] = SIM_FTHR
    s[17:24] = 0.0
    return s


def tail(pol, tp_tok, wrist_tok, state, wrench):
    """Everything after the two ResNet trunks (which depend only on the image)."""
    v = torch.cat([tp_tok, wrist_tok], dim=1) + pol.modal_embed_vision
    w = pol.wrench_tokenizer(wrench)
    w = w + pol.temporal_embed_wrench[:, : w.shape[1]] + pol.modal_embed_wrench
    v, w = pol.cross_attn(v, w)
    st = pol.state_encoder(state).unsqueeze(1)
    fused = torch.cat([pol.vision_pool(v), pol.wrench_pool(w), st], dim=1)
    return pol.action_head(fused.flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=os.path.expanduser("~/ig"))
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/best.pt"))
    ap.add_argument("--norm", default=os.path.expanduser(
        "~/forge_ts/student_ckpts/gear/jitcam_real_s2/norm_stats.npz"))
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", default=os.path.expanduser("~/forge_ts/logs/x28_reconcile.json"))
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    st = dict(np.load(a.norm))
    s_mean = torch.as_tensor(st["state_mean"], dtype=torch.float32)
    s_std  = torch.as_tensor(st["state_std"],  dtype=torch.float32)
    w_mean = torch.as_tensor(st["wrench_mean"], dtype=torch.float32)
    w_std  = torch.as_tensor(st["wrench_std"],  dtype=torch.float32)

    ck = torch.load(a.ckpt, map_location="cpu")
    pol = StudentFMTPolicy(StudentFMTConfig(**ck["config"]))
    pol.load_state_dict(ck["model"]); pol.eval()

    grid = json.load(open(os.path.join(a.grid, "imgrid.json")))
    samples = grid["samples"]
    tip = np.array(grid["fixture_tip_m"]) * 1000.0
    print(f"[x28] imgrid fixture_tip = {tip} mm   (T_SCAN assumed {T_SCAN})", flush=True)
    assert np.allclose(tip, T_SCAN, atol=1e-3), "imgrid fixture differs from assumption!"

    # ---- cache the ResNet trunks: they depend on the image alone ----
    print("[x28] caching vision tokens for 75 poses ...", flush=True)
    toks, keep = [], []
    with torch.inference_mode():
        for sm in samples:
            dx, dy, z = sm["cmd_dx_dy_z_mm"]
            k = "x%+04d_y%+04d_z%+04d" % (dx, dy, z)
            ptp = os.path.join(a.grid, k + "_tp_policy256.png")
            pwr = os.path.join(a.grid, k + "_wrist_policy256.png")
            if not (os.path.exists(ptp) and os.path.exists(pwr)):
                print("  MISSING", k, flush=True); continue
            tp = torch.from_numpy(np.asarray(Image.open(ptp).convert("RGB"), np.uint8)).unsqueeze(0)
            wr = torch.from_numpy(np.asarray(Image.open(pwr).convert("RGB"), np.uint8)).unsqueeze(0)
            toks.append((pol.tp_backbone(tp), pol.wrist_backbone(wr)))
            keep.append(sm)
    print(f"[x28] {len(keep)} poses cached", flush=True)

    D = [0, 2, 4, 6, 8, 8.4, 10, 12, 14, 16, 18, 20, 22, 26, 30]
    ORIGINS = {"scan_stale": np.zeros(3), "dscan_corrected": T_DSCAN - T_SCAN}

    results = {}
    with torch.inference_mode():
        for oname, delta in ORIGINS.items():
            print(f"\n[x28] origin = {oname}   delta = {delta} mm", flush=True)
            per_d = {}
            for d in D:
                rows = []
                rng = np.random.default_rng(23)
                for sm, (tpt, wrt) in zip(keep, toks):
                    rel = np.array(sm["rel_to_tip_mm"], float) - delta   # tcp - T_true
                    rel[1] -= d                                          # frame moved +d in y
                    acc = []
                    for _ in range(a.draws):
                        s = build_state(rel, sm["quat_wxyz"], rng)
                        sn = ((torch.from_numpy(s) - s_mean) / s_std).unsqueeze(0)
                        wraw = rng.normal(WRENCH_MEAN, WRENCH_STD, (32, 6)).astype(np.float32)
                        wn = ((torch.from_numpy(wraw) - w_mean) / w_std).unsqueeze(0)
                        acc.append((tail(pol, tpt, wrt, sn, wn)[0, :3].numpy() * 50.0))
                    b = np.mean(acc, 0)          # belief x,y,z in mm rel. to the shifted frame
                    rows.append({"cmd": sm["cmd_dx_dy_z_mm"],
                                 "belief": b.tolist(),
                                 "err_x": float(b[0]),
                                 "err_y": float(d + b[1]),
                                 "bel_z": float(b[2])})
                per_d[str(d)] = rows
                ex = np.array([r["err_x"] for r in rows])
                ey = np.array([r["err_y"] for r in rows])
                bz = np.array([r["bel_z"] for r in rows])
                rad = np.hypot(ex, ey)
                n = len(rows)
                print("  d=%5.1f | ey %+6.2f | |ey| %5.2f+-%4.2f | rad %5.2f+-%4.2f | "
                      "rate<=3mm %5.1f%% | ex %+6.2f | bel_z %+6.2f"
                      % (d, ey.mean(), np.abs(ey).mean(), np.abs(ey).std(ddof=1)/np.sqrt(n),
                         rad.mean(), rad.std(ddof=1)/np.sqrt(n),
                         100.0*(rad <= 3.0).mean(), ex.mean(), bz.mean()), flush=True)
            results[oname] = per_d

    json.dump({"T_SCAN": T_SCAN.tolist(), "T_DSCAN": T_DSCAN.tolist(),
               "draws": a.draws, "D": D, "results": results}, open(a.out, "w"))
    print(f"\n[x28] saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
