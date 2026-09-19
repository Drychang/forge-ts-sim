"""Breaking-point figure: student vs the official state-only T-A, across the full
noise axis including the beyond-training extrapolation region.

Data source: the frozen protocol logs (protocol_seed 42, dyn_rand on, n=256,
seed 0 throughout so the 0-5mm cells sit on the same footing as the 7.5-20mm
breaking-point cells). Indexed by the CHECKPOINT each log evaluated, with an
allowlist of clean-protocol log prefixes, so the perturbation studies (colour
shift, camera jitter, latency) cannot leak in.

The T-B oracle is drawn as a thin dashed reference, not as a competitor: it reads
the true noise vector, so it is the information-theoretic ceiling. Without it a
drop in the student's curve is ambiguous between "distillation failed to
extrapolate" and "the task itself became unsolvable".
"""
import glob
import os
import re

import numpy as np

LOGD = os.path.expanduser("~/forge_ts/logs")
OUT = os.path.expanduser("~/forge_ts/figs")
os.makedirs(OUT, exist_ok=True)

TASKS = [("peg", "PegInsert"), ("gear", "GearMesh"), ("nut", "NutThread")]
LEVELS = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
TRAIN_MAX = 5.0

ARMS = [
    ("student", "Student (ours): vision + force", "#1f77b4", 2.6, "-", 6),
    ("ta", "T-A (official FORGE): state only", "#2ca02c", 2.0, "-", 5),
    ("tb", "T-B oracle (reads the true noise) — ceiling", "0.45", 1.4, "--", 3),
]

ALLOWED = ("eval_gate2_", "bp_student_", "noisespec_s0_", "bp_ta_",
           "tb_noisespec_s0_", "bp_tb_")
LINE = re.compile(r"noise_mm=(?P<n>[0-9.]+)\s+dyn_rand=on\s+n=(?P<N>\d+)\s+sr=(?P<sr>[0-9.]+)"
                  r".*?ckpt=(?P<ckpt>\S+)")


def classify(ckpt):
    for t, _ in TASKS:
        if f"student_ckpts/{t}/gate2_seed0" in ckpt:
            return "student", t
        if f"Forge/repro_{t}_s0" in ckpt:
            return "ta", t
        if f"Forge/tb_{t}_s0" in ckpt:
            return "tb", t
    return None, None


def build():
    idx = {}
    for p in sorted(glob.glob(os.path.join(LOGD, "*.log"))):
        if not os.path.basename(p).startswith(ALLOWED):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            m = LINE.search(line)
            if not m:
                continue
            arm, task = classify(m.group("ckpt"))
            if arm is None:
                continue
            N = int(m.group("N"))
            k = int(round(float(m.group("sr")) * N))
            idx[(arm, task, float(m.group("n")))] = (k, N)
    return idx


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = build()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True)

    for ax, (t, T) in zip(axes, TASKS):
        # shade the region the teacher never trained in
        ax.axvspan(TRAIN_MAX, LEVELS[-1] + 0.6, color="#f2c9c9", alpha=0.32, zorder=0, lw=0)
        ax.axvline(TRAIN_MAX, color="#b03a3a", lw=1.2, ls=":", zorder=1)

        for arm, label, colour, lw, ls, zo in ARMS:
            xs, ys, lo, hi = [], [], [], []
            for nm in LEVELS:
                hit = idx.get((arm, t, nm))
                if hit is None:
                    continue
                k, n = hit
                a, b = wilson(k, n)
                xs.append(nm); ys.append(100 * k / n)
                lo.append(100 * (k / n - a)); hi.append(100 * (b - k / n))
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=5.5, capsize=3,
                        lw=lw, ls=ls, color=colour, label=label, zorder=zo,
                        markerfacecolor="white" if arm == "tb" else colour)

        # annotate the student's own degradation
        st = [(nm, idx[("student", t, nm)]) for nm in LEVELS if ("student", t, nm) in idx]
        if st:
            n5 = dict(st)[5.0]; n20 = dict(st)[20.0]
            s5, s20 = 100 * n5[0] / n5[1], 100 * n20[0] / n20[1]
            ax.annotate(f"{s5:.1f}%", xy=(5.0, s5), xytext=(1.0, 68),
                        fontsize=9.5, fontweight="bold", color="#1f77b4",
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="#1f77b4"))
            ax.annotate(f"{s20:.1f}%", xy=(20.0, s20), xytext=(15.2, 34),
                        fontsize=9.5, fontweight="bold", color="#1f77b4",
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="#1f77b4"))
            ta20 = idx[("ta", t, 20.0)]
            ax.annotate(f"{100*ta20[0]/ta20[1]:.1f}%", xy=(20.0, 100 * ta20[0] / ta20[1]),
                        xytext=(13.2, 18), fontsize=9.5, fontweight="bold", color="#2ca02c",
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="#2ca02c"))

        ax.set_title({"peg": "PegInsert", "gear": "GearMesh", "nut": "NutThread"}[t], fontsize=12.5)
        ax.set_xlabel("Fixed-asset pose observation noise, std (mm)", fontsize=11)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([f"{v:g}" for v in LEVELS], fontsize=9.5)
        ax.set_xlim(-0.8, LEVELS[-1] + 0.6)
        ax.set_ylim(-3, 105)
        ax.grid(alpha=0.3, ls=":", zorder=0)

    axes[0].set_ylabel("Success Rate (%)", fontsize=12)
    for _ax in axes:
        _ax.text((TRAIN_MAX + LEVELS[-1]) / 2, 2.0,
                 "extrapolation (beyond the teacher's 5 mm training range)",
                 fontsize=8.3, color="#8a2b2b", ha="center", va="bottom", alpha=0.9)
    axes[0].legend(fontsize=9, loc="lower left", framealpha=0.94)

    fig.suptitle("Breaking-Point Analysis — where does the method actually fail?",
                 fontsize=14.5, y=0.985)
    fig.text(0.5, 0.005,
             "Frozen protocol: n=256 per cell, protocol_seed 42, dyn_rand on, seed 0 throughout. "
             "Error bars: Wilson 95% CI. Shaded region is beyond the teacher's 5 mm training range, "
             "so every point there is extrapolation.",
             ha="center", fontsize=8.6, color="0.32")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    p = os.path.join(OUT, "fig3_breaking_point.png")
    fig.savefig(p, dpi=200)
    print("WROTE", p)

    # ---- retention table, printed for the caption ----
    print("\nstudent / oracle retention (%):")
    print(f"{'task':6s} " + "".join(f"{v:>7g}" for v in LEVELS))
    for t, _ in TASKS:
        row = []
        for nm in LEVELS:
            s, o = idx.get(("student", t, nm)), idx.get(("tb", t, nm))
            row.append(f"{100*(s[0]/s[1])/(o[0]/o[1]):>7.0f}" if s and o else "      -")
        print(f"{t:6s} " + "".join(row))
    print("\nstudent - T-A gap (points):")
    for t, _ in TASKS:
        row = []
        for nm in LEVELS:
            s, a = idx.get(("student", t, nm)), idx.get(("ta", t, nm))
            row.append(f"{100*(s[0]/s[1] - a[0]/a[1]):>7.1f}" if s and a else "      -")
        print(f"{t:6s} " + "".join(row))


if __name__ == "__main__":
    main()
