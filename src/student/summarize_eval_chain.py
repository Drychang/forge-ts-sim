import json
import sys

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        if "EVAL_SUMMARY " in line:
            d = json.loads(line.split("EVAL_SUMMARY ", 1)[1])
            rows.append(d)

rows.sort(key=lambda r: (r["noise_mm"], r["dyn_rand"]))
print("noise_mm  dyn_rand      sr    ci_lo    ci_hi")
for r in rows:
    print("{:>8}  {:>8}  {:6.2f}%  {:6.2f}%  {:6.2f}%".format(
        r["noise_mm"], r["dyn_rand"], r["sr"] * 100, r["wilson95_lo"] * 100, r["wilson95_hi"] * 100
    ))
print(f"\ntotal conditions: {len(rows)}")
