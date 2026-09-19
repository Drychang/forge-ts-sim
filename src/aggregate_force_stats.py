"""T1: aggregate contact-force statistics across T-A / noise-aug T-A / T-B / student.

Reads the frozen-protocol summary JSONs already on disk (no new eval runs).
For each (model, task, noise_mm, dyn_rand) picks the LATEST file per seed if
duplicates exist (reruns), then reports 3-seed mean for:
  mean_force_n_avg, max_force_n_avg, max_force_n_max, frac_over_5n_avg
"""
import glob
import json
import os
import statistics

EVAL_DIR = os.path.expanduser("~/forge_ts/eval")
TASKS = ["PegInsert", "GearMesh", "NutThread"]
NOISE = [0, 1, 2.5, 5]
DR = ["on", "off"]

MODELS = {
    "T-A (1mm)": {"pattern": "{task}_noisespec_s{seed}_noise{noise}mm_dr{dr}_seed42_*.json", "exclude": "ta_n5"},
    "Noise-aug T-A (5mm)": {"pattern": "{task}_ta_n5_noisespec_s{seed}_noise{noise}mm_dr{dr}_seed42_*.json", "exclude": None},
    "T-B (oracle)": {"pattern": "{task}-TB-v0_tb_noisespec_s{seed}_noise{noise}mm_dr{dr}_seed42_*.json", "exclude": None},
    "Student": {"pattern": "{task}_gate2_seed{seed}_noise{noise}mm_dr{dr}_seed42_*.json", "exclude": None},
}

FIELDS = ["mean_force_n_avg", "max_force_n_avg", "max_force_n_max", "frac_over_5n_avg"]


def load_latest(pattern):
    matches = sorted(glob.glob(os.path.join(EVAL_DIR, pattern)))
    if not matches:
        return None
    # filename ends in _<timestamp>.json -- pick the largest timestamp
    matches.sort(key=lambda p: int(os.path.splitext(p)[0].rsplit("_", 1)[-1]))
    with open(matches[-1]) as f:
        return json.load(f)


rows = {}  # (model, task, noise, dr) -> list of dicts (per seed)
for model, spec in MODELS.items():
    for task in TASKS:
        for noise in NOISE:
            for dr in DR:
                vals = []
                for seed in [0, 1, 2]:
                    fname_glob = spec["pattern"].format(task=task, seed=seed, noise=noise, dr=dr)
                    if spec["exclude"]:
                        # emulate "exclude ta_n5" by checking candidate paths don't contain it
                        cands = sorted(glob.glob(os.path.join(EVAL_DIR, fname_glob)))
                        cands = [c for c in cands if spec["exclude"] not in os.path.basename(c)]
                        if not cands:
                            continue
                        cands.sort(key=lambda p: int(os.path.splitext(p)[0].rsplit("_", 1)[-1]))
                        with open(cands[-1]) as f:
                            d = json.load(f)
                    else:
                        d = load_latest(fname_glob)
                    if d is not None:
                        vals.append(d)
                if vals:
                    rows[(model, task, noise, dr)] = vals

print(f"{'model':22} {'task':10} {'noise':6} {'dr':4} {'n':3} "
      f"{'meanF':>8} {'maxF_avg':>9} {'maxF_max':>9} {'frac>5N':>8}")
for model in MODELS:
    for task in TASKS:
        for noise in NOISE:
            for dr in DR:
                key = (model, task, noise, dr)
                vals = rows.get(key)
                if not vals:
                    continue
                n = len(vals)
                agg = {f: statistics.mean(v[f] for v in vals) for f in FIELDS}
                print(f"{model:22} {task:10} {noise:<6} {dr:4} {n:3} "
                      f"{agg['mean_force_n_avg']:8.2f} {agg['max_force_n_avg']:9.2f} "
                      f"{agg['max_force_n_max']:9.2f} {agg['frac_over_5n_avg']:8.3f}")

# Write markdown summary
out_path = os.path.join(EVAL_DIR, "force_stats_summary.md")
with open(out_path, "w") as f:
    f.write("# Contact-force statistics summary (T1)\n\n")
    f.write("mean_force_n_avg = avg over episode of ||F|| (N); max_force_n_avg = avg per-episode peak "
            "||F||; max_force_n_max = worst single peak across all episodes; frac_over_5n_avg = fraction "
            "of policy steps with ||F||>5N (avg over episodes).\n\n")
    for task in TASKS:
        f.write(f"## {task}\n\n")
        f.write("| Model | noise(mm) | dr | n_seeds | mean F (N) | max F avg (N) | max F worst (N) | frac>5N |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in MODELS:
            for noise in NOISE:
                for dr in DR:
                    key = (model, task, noise, dr)
                    vals = rows.get(key)
                    if not vals:
                        continue
                    n = len(vals)
                    agg = {fl: statistics.mean(v[fl] for v in vals) for fl in FIELDS}
                    f.write(f"| {model} | {noise} | {dr} | {n} | {agg['mean_force_n_avg']:.2f} | "
                            f"{agg['max_force_n_avg']:.2f} | {agg['max_force_n_max']:.2f} | "
                            f"{agg['frac_over_5n_avg']:.3f} |\n")
        f.write("\n")

print(f"\nWROTE {out_path}")
