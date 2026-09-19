"""Aggregate the breaking-point matrix: student vs T-A vs T-B oracle over the
full noise axis, including the beyond-training extrapolation levels.

Rather than guessing filenames, this indexes every log by the CHECKPOINT it
evaluated, which is what actually identifies the arm:
    student_ckpts/<task>/gate2_seed0  -> student (vision+force, distilled)
    Forge/repro_<task>_s0             -> T-A (official FORGE state-only)
    Forge/tb_<task>_s0               -> T-B (privileged oracle)
Seed 0 only, dyn_rand=on only, so the main-sweep cells (0/1/2.5/5mm) are on the
same footing as the breaking-point cells (7.5/10/15/20mm) -- note these are
single-seed numbers and will differ slightly from the 3-seed means in the report.

Usage: aggregate_bp.py [--md]
"""
import argparse
import glob
import os
import re

LOGD = os.path.expanduser("~/forge_ts/logs")
TASKS = ["peg", "gear", "nut"]
LEVELS = ["0", "1", "2.5", "5", "7.5", "10", "15", "20"]
ARMS = [("student", "Student (ours)"), ("ta", "T-A (official)"), ("tb", "T-B (oracle)")]

LINE = re.compile(
    r"noise_mm=(?P<noise>[0-9.]+)\s+dyn_rand=(?P<dr>on|off)\s+n=(?P<n>\d+)\s+sr=(?P<sr>[0-9.]+)"
    r".*?ckpt=(?P<ckpt>\S+)"
)


def classify(ckpt):
    for t in TASKS:
        if f"student_ckpts/{t}/gate2_seed0" in ckpt:
            return "student", t
        if f"Forge/repro_{t}_s0" in ckpt:
            return "ta", t
        if f"Forge/tb_{t}_s0" in ckpt:
            return "tb", t
    return None, None


# Only the CLEAN frozen-protocol runs. Without this allowlist the index is
# poisoned by the perturbation studies, which also evaluate gate2_seed0 at
# dyn_rand=on / 2.5mm noise (colour shift, camera jitter, latency, realcolour,
# modality ablation) -- their whole point is that they are NOT the clean setting.
ALLOWED_PREFIXES = (
    "eval_gate2_",       # student, main sweep (one chain log per task, 8 cells)
    "bp_student_",       # student, breaking point
    "noisespec_s0_",     # T-A, main sweep
    "bp_ta_",            # T-A, breaking point
    "tb_noisespec_s0_",  # T-B, main sweep
    "bp_tb_",            # T-B, breaking point
)


def build_index():
    idx = {}
    for path in sorted(glob.glob(os.path.join(LOGD, "*.log"))):
        name = os.path.basename(path)
        if not name.startswith(ALLOWED_PREFIXES):
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = LINE.search(line)
                if not m or m.group("dr") != "on":
                    continue
                arm, task = classify(m.group("ckpt"))
                if arm is None:
                    continue
                noise = "%g" % float(m.group("noise"))
                key = (arm, task, noise)
                if key in idx and idx[key][0] != float(m.group("sr")):
                    print(f"WARNING duplicate {key}: {idx[key][2]}={idx[key][0]} vs {name}={m.group('sr')}")
                idx[key] = (float(m.group("sr")), int(m.group("n")), name)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    idx = build_index()

    if args.md:
        print("| Task | Arm | " + " | ".join(f"{l}mm" for l in LEVELS) + " |")
        print("|---|---|" + "---|" * len(LEVELS))
    for t in TASKS:
        for key, label in ARMS:
            vals = []
            for l in LEVELS:
                hit = idx.get((key, t, l))
                vals.append("--" if hit is None else f"{100 * hit[0]:.1f}")
            if args.md:
                print(f"| {t} | {label} | " + " | ".join(vals) + " |")
            else:
                print(f"{t:5s} {label:16s} " + "  ".join(f"{v:>6s}" for v in vals))
        if not args.md:
            print()

    missing = [(t, k, l) for t in TASKS for k, _ in ARMS for l in LEVELS if (k, t, l) not in idx]
    print(f"\nmissing cells: {len(missing)}" + ("" if missing else "  (matrix complete)"))
    for t, k, l in missing:
        print(f"  {t} {k} {l}mm")


if __name__ == "__main__":
    main()
