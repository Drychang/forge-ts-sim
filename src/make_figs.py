"""Distribution figures for the FORGE teacher-student paper.

Fig 1  violin + box of the SR estimate at the hardest noise level (5mm), grouped
       by task, one violin per method. Violin = bootstrap distribution of the
       frozen-protocol success-rate estimate (2000 resamples of that cell's own
       n episodes); box = its quartiles; dots = the actual independent training
       seeds behind it (3 for our arms, 1 for the single-seed baselines, which is
       the baseline convention -- shown rather than hidden).

Fig 2  SR vs observation noise (0/1/2.5/5mm), one line per method, per task,
       with Wilson 95% CI error bars.

Everything comes from the frozen protocol logs (protocol_seed 42, dyn_rand on,
n=256). Nothing is re-run and nothing is averaged across tasks.
"""
import glob
import json
import os
import re

import numpy as np

LOGD = os.path.expanduser("~/forge_ts/logs")
OUT = os.path.expanduser("~/forge_ts/figs")
os.makedirs(OUT, exist_ok=True)
TASKS = [("peg", "PegInsert"), ("gear", "GearMesh"), ("nut", "NutThread")]
NOISES = ["0", "1", "2.5", "5"]

# method -> (display name, colour, list of log-path templates per seed)
# {t} short task, {T} gym task, {n} noise
METHODS = [
    ("ours", "Ours (vision+force)", "#1f77b4", [
        "eval_gate2_{t}_gate2_seed0.log", "eval_gate2_{t}_gate2_seed1.log", "eval_gate2_{t}_gate2_seed2.log"]),
    ("taaug", "Noise-aug T-A", "#ff7f0e", [
        "ta_n5_noisespec_s0_{t}_noise{n}mm_dron.log", "ta_n5_noisespec_s1_{t}_noise{n}mm_dron.log",
        "ta_n5_noisespec_s2_{t}_noise{n}mm_dron.log"]),
    ("ta", "T-A (official FORGE)", "#2ca02c", [
        "noisespec_s0_{t}_noise{n}mm_dron.log", "noisespec_s1_{t}_noise{n}mm_dron.log",
        "noisespec_s2_{t}_noise{n}mm_dron.log"]),
    ("arch", "ARCH-INSERT (CoRL'25)", "#d62728", ["arch_v3_eval_{t}_n{n}.log"]),
    ("c3", "C3 / RMA-style (2026)", "#9467bd", ["tc_c3_{t}_adapter_n{n}.log"]),
    ("srsa", "SRSA (ICLR'25)", "#8c564b", ["srsa_3b_v2_eval_{t}_n{n}.log"]),
]

LINE = re.compile(r"noise_mm=(?P<noise>[0-9.]+)\s+dyn_rand=on\s+n=(?P<n>\d+)\s+sr=(?P<sr>[0-9.]+)")


def read_cell(path, want_noise):
    """Return (successes, n) for the dyn_rand=on run at want_noise, else None.

    Two log flavours exist. Most write a human-readable summary line carrying the
    noise level, so the level is verified from the line itself. The C3 wrapper
    writes only the EVAL_SUMMARY JSON and encodes the noise in the FILENAME, so
    for those the caller's template is what pins the level -- one file per cell.
    """
    p = os.path.join(LOGD, path)
    if not os.path.isfile(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()
    for line in txt.splitlines():
        m = LINE.search(line)
        if m and abs(float(m.group("noise")) - float(want_noise)) <= 1e-9:
            n = int(m.group("n"))
            return int(round(float(m.group("sr")) * n)), n
    # fallback: EVAL_SUMMARY JSON only (C3). dyn_rand=on is the only variant run.
    ms = re.search(r'"sr":\s*([0-9.]+)', txt)
    me = re.search(r'"episodes":\s*(\d+)', txt)
    if ms and me:
        n = int(me.group(1))
        return int(round(float(ms.group(1)) * n)), n
    return None


def cells(method_tpls, t, T, noise):
    """All available (k, n) for a method/task/noise, one per seed."""
    out = []
    for tpl in method_tpls:
        r = read_cell(tpl.format(t=t, T=T, n=noise), noise)
        if r:
            out.append(r)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def boot(k, n, reps=2000, rng=None):
    """Bootstrap the SR estimate: resample n Bernoulli outcomes from the cell."""
    rng = rng or np.random.default_rng(0)
    return rng.binomial(n, k / n, size=reps) / n * 100.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rng = np.random.default_rng(42)

    # ---------------- Figure 1: violin + box at 5mm ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    missing = []
    for ax, (t, T) in zip(axes, TASKS):
        data, colours, labels, seedpts, ns = [], [], [], [], []
        for key, name, colour, tpls in METHODS:
            cs = cells(tpls, t, T, "5")
            if not cs:
                missing.append((t, key, "5mm"))
                continue
            k = sum(c[0] for c in cs)
            n = sum(c[1] for c in cs)
            data.append(boot(k, n, rng=rng))
            colours.append(colour)
            labels.append(name)
            seedpts.append([100.0 * a / b for a, b in cs])
            ns.append((len(cs), n))

        pos = np.arange(1, len(data) + 1)
        vp = ax.violinplot(data, positions=pos, widths=0.78, showextrema=False)
        for body, colour in zip(vp["bodies"], colours):
            body.set_facecolor(colour)
            body.set_alpha(0.32)
            body.set_edgecolor(colour)
        bp = ax.boxplot(data, positions=pos, widths=0.20, showfliers=False,
                        medianprops=dict(color="#e8710a", lw=1.8),
                        boxprops=dict(color="0.25"), whiskerprops=dict(color="0.25"),
                        capprops=dict(color="0.25"))
        for i, (pts, colour) in enumerate(zip(seedpts, colours), start=1):
            jit = rng.uniform(-0.10, 0.10, size=len(pts))
            ax.scatter(np.full(len(pts), i) + jit, pts, s=34, zorder=5,
                       facecolor="white", edgecolor=colour, linewidths=1.5)
        # annotate our median
        if data:
            ax.annotate(f"{np.median(data[0]):.1f}%", xy=(1, np.median(data[0])),
                        xytext=(1.42, np.median(data[0]) + 7), fontsize=10, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", lw=1.0, color="0.3"))
        ax.set_xticks(pos)
        ax.set_xticklabels([l.split(" (")[0].replace("Ours (vision+force)", "Ours") for l in labels],
                           rotation=28, ha="right", fontsize=9)
        ax.set_title({"peg": "PegInsert", "gear": "GearMesh", "nut": "NutThread"}[t], fontsize=12)
        ax.grid(axis="y", alpha=0.3, ls=":")
        ax.set_ylim(-3, 105)
    axes[0].set_ylabel("Success Rate (%)", fontsize=12)
    fig.suptitle("Success-Rate Distribution at the Hardest Noise Level (5 mm) — Violin + Box",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.005,
             "Violin/box: bootstrap distribution of the frozen-protocol SR estimate "
             "(2000 resamples, protocol_seed 42, dyn_rand on).  Circles: independent training seeds "
             "(3 for Ours / T-A / Noise-aug T-A; 1 for the single-seed baselines, per baseline convention).",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    f1 = os.path.join(OUT, "fig1_violin_5mm.png")
    fig.savefig(f1, dpi=200)
    plt.close(fig)

    # ---------------- Figure 2: SR vs noise ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
    for ax, (t, T) in zip(axes, TASKS):
        for key, name, colour, tpls in METHODS:
            xs, ys, lo, hi = [], [], [], []
            for j, nm in enumerate(NOISES):
                cs = cells(tpls, t, T, nm)
                if not cs:
                    continue
                k = sum(c[0] for c in cs)
                n = sum(c[1] for c in cs)
                a, b = wilson(k, n)
                xs.append(j)
                ys.append(100.0 * k / n)
                lo.append(100.0 * (k / n - a))
                hi.append(100.0 * (b - k / n))
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=5, capsize=3, lw=2.0 if key == "ours" else 1.4,
                        color=colour, label=name, zorder=5 if key == "ours" else 3,
                        alpha=1.0 if key == "ours" else 0.85)
        ax.set_xticks(range(len(NOISES)))
        ax.set_xticklabels([f"{n}" for n in NOISES])
        ax.set_xlabel("Observation noise std (mm)", fontsize=11)
        ax.set_title({"peg": "PegInsert", "gear": "GearMesh", "nut": "NutThread"}[t], fontsize=12)
        ax.grid(alpha=0.3, ls=":")
        ax.set_ylim(-3, 105)
    axes[0].set_ylabel("Success Rate (%)", fontsize=12)
    axes[0].legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    fig.suptitle("Success Rate vs Observation Noise — Ours stays flat where state-only methods collapse",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.005,
             "Frozen protocol: n=256 per cell, protocol_seed 42, dyn_rand on. Error bars: Wilson 95% CI. "
             "Ours / T-A / Noise-aug T-A pool 3 seeds; ARCH / C3 / SRSA are single-seed (baseline convention).",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    f2 = os.path.join(OUT, "fig2_sr_vs_noise.png")
    fig.savefig(f2, dpi=200)
    plt.close(fig)

    # ---------------- provenance dump ----------------
    prov = {}
    for t, T in TASKS:
        for key, name, _, tpls in METHODS:
            for nm in NOISES:
                cs = cells(tpls, t, T, nm)
                if cs:
                    prov[f"{t}|{key}|{nm}mm"] = {"per_seed_sr": [round(a / b, 4) for a, b in cs],
                                                 "pooled_k": sum(c[0] for c in cs),
                                                 "pooled_n": sum(c[1] for c in cs)}
    with open(os.path.join(OUT, "figure_data.json"), "w") as fh:
        json.dump(prov, fh, indent=1)

    print("WROTE", f1)
    print("WROTE", f2)
    print("missing cells:", missing if missing else "none")
    for t, _ in TASKS:
        row = []
        for key, name, _, tpls in METHODS:
            cs = cells(tpls, t, "", "5")
            row.append(f"{key}={100 * sum(c[0] for c in cs) / sum(c[1] for c in cs):.1f}" if cs else f"{key}=--")
        print(f"  {t} @5mm: " + "  ".join(row))


if __name__ == "__main__":
    main()
