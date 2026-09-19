"""BC-only vs BC+DAgger, 3 seeds, across the full noise axis including the
extrapolation region. Reads the frozen-protocol logs directly.

Seed 0 of each arm comes from its original run (p0_tb_* / eval_gate2_* and the
breaking-point cells); seeds 1-2 come from tonight's follow-ups.
"""
import os
import re
import statistics as st

L = os.path.expanduser("~/forge_ts/logs")


def sr(path, want_noise=None):
    p = os.path.join(L, path)
    if not os.path.isfile(p):
        return None
    for line in open(p, errors="replace"):
        if "dyn_rand=on" not in line:
            continue
        m = re.search(r"noise_mm=([0-9.]+).*?sr=([0-9.]+)", line)
        if not m:
            continue
        if want_noise is not None and abs(float(m.group(1)) - float(want_noise)) > 1e-9:
            continue
        return float(m.group(2))
    return None


IN_RANGE = ["0", "1", "2.5", "5"]
EXTRAP = ["7.5", "10"]


def bconly(nm):
    if nm in IN_RANGE:
        return [sr(f"p0_tb_eval_peg_n{nm}.log", nm),
                sr(f"tb_bconly_s1_eval_peg_n{nm}.log", nm),
                sr(f"tb_bconly_s2_eval_peg_n{nm}.log", nm)]
    return [sr(f"p0_tb_eval_peg_n{nm}.log", nm),
            sr(f"dag3_bconly_s1_n{nm}.log", nm),
            sr(f"dag3_bconly_s2_n{nm}.log", nm)]


def dagger(nm):
    if nm in IN_RANGE:
        return [sr(f"eval_gate2_peg_gate2_seed{s}.log", nm) for s in (0, 1, 2)]
    return [sr(f"bp_student_peg_n{nm}.log", nm),
            sr(f"dag3_gate2_s1_n{nm}.log", nm),
            sr(f"dag3_gate2_s2_n{nm}.log", nm)]


print("| noise | BC-only (mean ± sd) | BC+DAgger (mean ± sd) | DAgger 貢獻 |")
print("|---|---|---|---|")
for nm in IN_RANGE + EXTRAP:
    b, d = bconly(nm), dagger(nm)
    if any(v is None for v in b + d):
        print(f"| {nm}mm | pending {b} | pending {d} | -- |")
        continue
    bm, bs = 100 * st.fmean(b), 100 * st.stdev(b)
    dm, ds = 100 * st.fmean(d), 100 * st.stdev(d)
    tag = "" if nm in IN_RANGE else "(外推)"
    print(f"| {nm}mm {tag} | {bm:.1f} ± {bs:.1f} | {dm:.1f} ± {ds:.1f} | **+{dm - bm:.1f}** |")

print()
for nm in IN_RANGE + EXTRAP:
    b, d = bconly(nm), dagger(nm)
    if any(v is None for v in b + d):
        continue
    print(f"{nm}mm  BC-only {[round(100 * v, 1) for v in b]}   BC+DAgger {[round(100 * v, 1) for v in d]}")
