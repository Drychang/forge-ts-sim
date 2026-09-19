import json
import glob

rows = []
for f in glob.glob("/home/user/forge_ts/eval/*noisespec_s0*.json"):
    d = json.load(open(f))
    rows.append(d)
rows.sort(key=lambda d: (d["task"], float(d["noise_mm"]), d["dyn_rand"]))

header = "{:22s} {:>6s} {:>4s} {:>7s} {:>6s} {:>6s} {:>7s}".format(
    "task", "noise", "dr", "sr%", "ci_lo", "ci_hi", "F_mean"
)
print(header)
for d in rows:
    task_short = d["task"].replace("Isaac-Forge-", "").replace("-Direct-v0", "")
    line = "{:22s} {:>5.1f}mm {:>4s} {:>6.2f}% {:>6.1f} {:>6.1f} {:>6.2f}N".format(
        task_short,
        d["noise_mm"],
        d["dyn_rand"],
        d["sr"] * 100,
        d["wilson95_lo"] * 100,
        d["wilson95_hi"] * 100,
        d["mean_force_n_avg"],
    )
    print(line)
