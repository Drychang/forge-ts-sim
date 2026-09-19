"""Compare the noise distribution of the T-A vs T-B demonstration datasets.

This is the mechanism figure for P0-1. Both collections drew per-episode noise
from the same law (std ~ U[0,5mm], noise ~ N(0,std)) and kept only SUCCESSES.
The privileged teacher succeeds at ~98% everywhere, so its dataset inherits the
sampling law almost unchanged. The non-privileged teacher only succeeds ~32% at
5mm, so success-filtering silently truncates its dataset's high-noise tail --
the student then never sees the compensation behaviour it is supposed to learn.

Usage: noise_hist.py [--edges 0,1,2,2.5,3,4,5,99]
"""
import argparse
import glob
import os

import numpy as np

ROOTS = [
    ("T-B (privileged)", "/media/data/forge_ts_data/PegInsert", 16),
    ("T-A (non-privileged)", "/media/data/forge_ts_data_ta/PegInsert", None),
]


def load_noise(root, max_shards):
    paths = sorted(glob.glob(os.path.join(root, "shard_*.npz")))
    if max_shards is not None:
        paths = paths[:max_shards]  # BC stage only, to match the T-A arm
    vals = []
    for p in paths:
        with np.load(p) as z:
            vals.append(np.asarray(z["noise_std_mm"], dtype=np.float64))
    if not vals:
        return None, 0
    return np.concatenate(vals), len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="0,1,2,3,4,5,99")
    args = ap.parse_args()
    edges = [float(x) for x in args.edges.split(",")]

    rows = {}
    for name, root, ms in ROOTS:
        v, n = load_noise(root, ms)
        if v is None:
            print(f"{name}: no shards under {root} yet")
            continue
        rows[name] = v
        print(f"{name}: {len(v)} episodes from {n} shards  "
              f"mean={v.mean():.2f}mm  median={np.median(v):.2f}mm  p90={np.percentile(v, 90):.2f}mm  max={v.max():.2f}mm")

    if len(rows) < 2:
        print("\n(need both datasets for the comparison)")
        return

    print("\n%-14s %s" % ("bin (mm)", "  ".join("%-22s" % k for k in rows)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells = []
        for k, v in rows.items():
            m = (v >= lo) & (v < hi)
            cells.append("%-22s" % f"{m.sum():5d}  ({100.0 * m.mean():5.1f}%)")
        print("%-14s %s" % (f"[{lo:g}, {hi:g})", "  ".join(cells)))

    # the headline number: how much of the high-noise regime got truncated
    a, b = list(rows.values())
    for thr in (2.5, 4.0):
        fa = 100.0 * (a >= thr).mean()
        fb = 100.0 * (b >= thr).mean()
        print(f"\nshare of demos with noise >= {thr:g}mm:  "
              f"{list(rows)[0]} {fa:.1f}%   {list(rows)[1]} {fb:.1f}%   "
              f"(ratio {fb / fa if fa else float('nan'):.2f}x)")


if __name__ == "__main__":
    main()
