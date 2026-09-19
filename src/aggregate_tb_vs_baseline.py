import json
import glob
import re
import statistics

def load(glob_pattern, seed_pattern):
    records = {}
    for f in glob.glob(glob_pattern):
        d = json.load(open(f))
        m = re.search(seed_pattern, d["checkpoint"])
        if not m:
            continue
        seed = int(m.group(1))
        task = d["task"].replace("Isaac-Forge-", "").replace("-Direct-v0", "").replace("-TB-v0", "")
        key = (task, seed, d["noise_mm"], d["dyn_rand"])
        if key not in records or d["timestamp"] > records[key]["timestamp"]:
            records[key] = d
    return records

# T-A checkpoints live under repro_{task}_s{seed}/, T-B under tb_{task}_s{seed}/ --
# filter on the checkpoint path (not just the tag) so nothing leaks across.
ta_records = {k: v for k, v in load("/home/user/forge_ts/eval/*.json", r"repro_[a-z]+_s(\d)").items()
              if "/tb_" not in v["checkpoint"]}
tb_records = {k: v for k, v in load("/home/user/forge_ts/eval/*.json", r"tb_[a-z]+_s(\d)").items()
              if "/tb_" in v["checkpoint"]}

print(f"# T-A unique combos: {len(ta_records)} (expect 72)")
print(f"# T-B unique combos: {len(tb_records)} (expect 72)")
print()

def group_by_task_noise_dr(records):
    groups = {}
    for (task, seed, noise, dr), d in records.items():
        groups.setdefault((task, noise, dr), []).append(d["sr"] * 100)
    return groups

ta_groups = group_by_task_noise_dr(ta_records)
tb_groups = group_by_task_noise_dr(tb_records)

header = "{:12s} {:>6s} {:>4s} {:>10s} {:>10s} {:>9s}".format("task", "noise", "dr", "T-A(base)", "T-B", "delta")
print(header)
for task in ["PegInsert", "GearMesh", "NutThread"]:
    for noise in [0.0, 1.0, 2.5, 5.0]:
        for dr in ["on", "off"]:
            key = (task, noise, dr)
            ta = ta_groups.get(key, [])
            tb = tb_groups.get(key, [])
            ta_m = statistics.mean(ta) if ta else float("nan")
            ta_s = statistics.stdev(ta) if len(ta) > 1 else 0.0
            tb_m = statistics.mean(tb) if tb else float("nan")
            tb_s = statistics.stdev(tb) if len(tb) > 1 else 0.0
            delta = tb_m - ta_m if ta and tb else float("nan")
            print(f"{task:12s} {noise:>5.1f}mm {dr:>4s} {ta_m:>6.2f}±{ta_s:<4.1f} {tb_m:>6.2f}±{tb_s:<4.1f} {delta:>+8.2f}")
