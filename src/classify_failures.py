"""Failure-case classification from the per-episode eval records. Analysis only:
no new rollouts, it re-reads the jsonl the frozen protocol already wrote.

Closes the "what do the remaining failures actually look like?" gap, and turns
the breaking-point curve into a mechanism story: WHICH failure mode grows as the
observation noise pushes past the training range.

Each episode record carries: success (at the final step), ever_success (at any
step), mean_force_n, max_force_n, frac_over_5n. That supports three disjoint
failure classes:

  SLIPPED   ever_success and not success
            -> the parts WERE mated at some point and then came apart. A control
               /stiffness problem, not a perception one.
  NO_INSERT not ever_success, but contact forces in the normal range
            -> found the part and pressed on it, never got in. Alignment.
  NO_TOUCH  not ever_success, and mean force below the 5th percentile of the
            SUCCESSFUL episodes in the same cell
            -> never developed normal contact at all: searched the wrong place.
               This is the mode that should dominate once the frame offset
               exceeds what vision can still resolve.

Usage: classify_failures.py [--arm student|ta] [--tasks peg,gear,nut]
"""
import argparse
import collections
import glob
import json
import os
import re

EVALD = os.path.expanduser("~/forge_ts/eval")
GYM = {"peg": "PegInsert", "gear": "GearMesh", "nut": "NutThread"}


def find_jsonl(task, arm, noise):
    """bp chains tag student cells bp_<task>_n<noise> and T-A cells bp_ta_<task>_n<noise>."""
    tag = f"bp_{task}_n{noise}" if arm == "student" else f"bp_ta_{task}_n{noise}"
    pat = os.path.join(EVALD, f"{GYM[task]}_{tag}_noise{noise}mm_dron_seed42_*.jsonl")
    hits = sorted(glob.glob(pat))
    if not hits and arm == "student":
        # the 0-5mm cells came from the main gate2 chain, tagged differently
        pat = os.path.join(EVALD, f"{GYM[task]}_gate2_seed0_noise{noise}mm_dron_seed42_*.jsonl")
        hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def classify(path):
    eps = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = [e for e in eps if e["success"]]
    bad = [e for e in eps if not e["success"]]
    if not eps:
        return None
    # data-driven contact floor: below the 5th percentile of successful episodes'
    # mean force, the episode never developed normal contact
    if ok:
        f = sorted(e["mean_force_n"] for e in ok)
        floor = f[max(0, int(0.05 * len(f)) - 1)]
    else:
        floor = 1.0
    c = collections.Counter()
    members = collections.defaultdict(list)
    for e in bad:
        if e.get("ever_success"):
            k = "SLIPPED"
        elif e["mean_force_n"] >= floor:
            k = "NO_INSERT"
        else:
            k = "NO_TOUCH"
        c[k] += 1
        members[k].append(e)

    def med(cls, field):
        v = sorted(e.get(field, 0.0) for e in members[cls])
        return v[len(v) // 2] if v else float("nan")

    return {"n": len(eps), "sr": len(ok) / len(eps), "fail": len(bad), "floor_n": floor,
            **{k: c[k] for k in ("SLIPPED", "NO_INSERT", "NO_TOUCH")},
            # force evidence per class, so the labels are backed rather than asserted
            "ok_over5": med("SLIPPED", "frac_over_5n") if False else (
                sorted(e["frac_over_5n"] for e in ok)[len(ok) // 2] if ok else float("nan")),
            "ni_over5": med("NO_INSERT", "frac_over_5n"),
            "nt_over5": med("NO_TOUCH", "frac_over_5n"),
            "ni_force": med("NO_INSERT", "mean_force_n"),
            "nt_force": med("NO_TOUCH", "mean_force_n")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="student", choices=["student", "ta"])
    ap.add_argument("--tasks", default="peg,gear,nut")
    ap.add_argument("--levels", default="5,7.5,10,15,20")
    args = ap.parse_args()

    hdr = (f"{'task':6s} {'noise':>6s} {'n':>4s} {'SR%':>6s} {'fails':>6s} | {'slipped':>8s} "
           f"{'no_insert':>10s} {'no_touch':>9s} | median frac_over_5N (ok/no_insert/no_touch)")
    print(hdr)
    print("-" * len(hdr))
    for t in args.tasks.split(","):
        for nm in args.levels.split(","):
            p = find_jsonl(t, args.arm, nm)
            if p is None:
                print(f"{t:6s} {nm:>6s}   -- (no per-episode file yet)")
                continue
            r = classify(p)
            if r is None:
                continue
            f = r["fail"] or 1
            print(f"{t:6s} {nm:>6s} {r['n']:4d} {100 * r['sr']:6.1f} {r['fail']:6d} | "
                  f"{r['SLIPPED']:3d} ({100 * r['SLIPPED'] / f:4.0f}%) "
                  f"{r['NO_INSERT']:4d} ({100 * r['NO_INSERT'] / f:4.0f}%) "
                  f"{r['NO_TOUCH']:3d} ({100 * r['NO_TOUCH'] / f:4.0f}%) | "
                  f"{r['ok_over5']:.2f} / {r['ni_over5']:.2f} / {r['nt_over5']:.2f}")
        print()


if __name__ == "__main__":
    main()
