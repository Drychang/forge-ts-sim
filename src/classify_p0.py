"""Failure-mode classification for the P0-1 matched pair (analysis only).

Same three disjoint classes as classify_failures.py:
  SLIPPED    mated at some point, came apart      -> control/stiffness
  NO_INSERT  contacted normally, never got in     -> fine alignment
  NO_TOUCH   never developed sustained contact    -> searched the wrong place
             (mean force below the 5th percentile of the SAME cell's successes)

The question this answers: WHEN the non-privileged-teacher student fails, does it
fail the way its teacher fails? T-A cannot recover the true hole position from a
displaced frame, so its failures should be dominated by never finding the target.
If the ta arm inherits that signature while the tb arm does not, the mechanism
behind the 65-point gap is pinned down, not just asserted.
"""
import collections
import glob
import json
import os

EVALD = os.path.expanduser("~/forge_ts/eval")


def find(arm, noise):
    hits = sorted(glob.glob(os.path.join(
        EVALD, f"PegInsert_p0_{arm}_peg_n{noise}_noise{noise}mm_dron_seed42_*.jsonl")))
    return hits[-1] if hits else None


def classify(path):
    eps = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = [e for e in eps if e["success"]]
    bad = [e for e in eps if not e["success"]]
    if ok:
        f = sorted(e["mean_force_n"] for e in ok)
        floor = f[max(0, int(0.05 * len(f)) - 1)]
    else:
        floor = 1.0
    c, members = collections.Counter(), collections.defaultdict(list)
    for e in bad:
        k = ("SLIPPED" if e.get("ever_success")
             else "NO_INSERT" if e["mean_force_n"] >= floor else "NO_TOUCH")
        c[k] += 1
        members[k].append(e)

    def med(cls, field):
        v = sorted(e.get(field, 0.0) for e in members[cls])
        return v[len(v) // 2] if v else float("nan")

    return {"n": len(eps), "sr": len(ok) / len(eps) if eps else 0.0, "fail": len(bad),
            "floor": floor, "c": c,
            "ok5": (sorted(e["frac_over_5n"] for e in ok)[len(ok) // 2] if ok else float("nan")),
            "ni5": med("NO_INSERT", "frac_over_5n"), "nt5": med("NO_TOUCH", "frac_over_5n")}


hdr = (f"{'arm':4s} {'noise':>6s} {'n':>4s} {'SR%':>6s} {'fails':>6s} | "
       f"{'slipped':>9s} {'no_insert':>11s} {'no_touch':>10s} | med frac>5N ok/ni/nt")
print(hdr)
print("-" * len(hdr))
for arm in ("ta", "tb"):
    for nm in ("0", "1", "2.5", "5", "7.5", "10"):
        p = find(arm, nm)
        if p is None:
            print(f"{arm:4s} {nm:>6s}   -- (no per-episode file)")
            continue
        r = classify(p)
        f = r["fail"] or 1
        print(f"{arm:4s} {nm:>6s} {r['n']:4d} {100 * r['sr']:6.1f} {r['fail']:6d} | "
              f"{r['c']['SLIPPED']:4d} ({100 * r['c']['SLIPPED'] / f:3.0f}%) "
              f"{r['c']['NO_INSERT']:5d} ({100 * r['c']['NO_INSERT'] / f:3.0f}%) "
              f"{r['c']['NO_TOUCH']:4d} ({100 * r['c']['NO_TOUCH'] / f:3.0f}%) | "
              f"{r['ok5']:.2f} / {r['ni5']:.2f} / {r['nt5']:.2f}")
    print()
