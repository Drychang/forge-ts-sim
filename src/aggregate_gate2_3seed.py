import glob
import json
import os
import statistics

LOG_DIR = os.path.expanduser("~/forge_ts/logs")
TASKS = ["peg", "gear", "nut"]

rows = {}  # (task, noise_mm, dyn_rand) -> list of sr

for task in TASKS:
    for seed in [0, 1, 2]:
        path = os.path.join(LOG_DIR, f"eval_gate2_{task}_gate2_seed{seed}.log")
        with open(path) as f:
            for line in f:
                if "EVAL_SUMMARY " in line:
                    d = json.loads(line.split("EVAL_SUMMARY ", 1)[1])
                    key = (task, d["noise_mm"], d["dyn_rand"])
                    rows.setdefault(key, []).append(d["sr"] * 100.0)

print(f"{'task':6} {'noise':6} {'dr':4} {'seeds':6} {'mean':8} {'std':7} {'min':7} {'max':7}")
for task in TASKS:
    for noise in [0, 1, 2.5, 5]:
        for dr in ["on", "off"]:
            key = (task, noise, dr)
            vals = rows.get(key, [])
            if len(vals) != 3:
                print(f"{task:6} {noise:<6} {dr:4} n={len(vals)} <<< MISSING, expected 3")
                continue
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals)
            print(f"{task:6} {noise:<6} {dr:4} {len(vals):<6} {mean:7.2f}% {std:6.2f} {min(vals):6.2f}% {max(vals):6.2f}%")

# dump raw per-seed for spot check
print()
print("=== raw per-seed sr (%) ===")
for task in TASKS:
    for noise in [0, 1, 2.5, 5]:
        for dr in ["on", "off"]:
            key = (task, noise, dr)
            vals = rows.get(key, [])
            print(f"{task} noise={noise} dr={dr}: {[round(v,2) for v in vals]}")
