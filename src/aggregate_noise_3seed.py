import json
import glob
import re
import statistics

records = {}  # (task, seed, noise, dr) -> latest record (by timestamp)

for f in glob.glob("/home/user/forge_ts/eval/*noisespec_s*.json"):
    d = json.load(open(f))
    m = re.search(r"repro_[a-z]+_s(\d)", d["checkpoint"])
    if not m:
        continue
    seed = int(m.group(1))
    task = d["task"].replace("Isaac-Forge-", "").replace("-Direct-v0", "")
    key = (task, seed, d["noise_mm"], d["dyn_rand"])
    if key not in records or d["timestamp"] > records[key]["timestamp"]:
        records[key] = d

# sanity: expect exactly 3 tasks x 3 seeds x 4 noise x 2 dr = 72 unique keys
print(f"# unique (task,seed,noise,dr) combos: {len(records)} (expect 72)")

groups = {}  # (task, noise, dr) -> list of sr across seeds
for (task, seed, noise, dr), d in records.items():
    groups.setdefault((task, noise, dr), []).append((seed, d["sr"] * 100, d["mean_force_n_avg"]))

print()
header = "{:12s} {:>6s} {:>4s} {:>8s} {:>8s}  seeds".format("task", "noise", "dr", "mean_sr%", "std_sr%")
print(header)
for task in ["PegInsert", "GearMesh", "NutThread"]:
    for noise in [0.0, 1.0, 2.5, 5.0]:
        for dr in ["on", "off"]:
            key = (task, noise, dr)
            if key not in groups:
                print(f"{task:12s} {noise:>5.1f}mm {dr:>4s}  MISSING")
                continue
            vals = sorted(groups[key])
            srs = [v[1] for v in vals]
            mean_sr = statistics.mean(srs)
            std_sr = statistics.stdev(srs) if len(srs) > 1 else 0.0
            seed_str = ", ".join(f"s{v[0]}={v[1]:.1f}%" for v in vals)
            print(f"{task:12s} {noise:>5.1f}mm {dr:>4s} {mean_sr:>7.2f}% {std_sr:>7.2f}%  [{seed_str}]")
