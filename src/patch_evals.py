"""Generate *_tilt.py copies of the two frozen-protocol evaluators with tilt_patch wired in.

Usage: python patch_evals.py <src_root>   (src_root = dir containing eval_frozen.py, tilt_patch.py, student/)
Originals are never modified. Generated files are overwritten.
"""
import os
import re
import sys

root = os.path.abspath(sys.argv[1])
student_dir = os.path.join(root, "student")


def must_replace(text, old, new, label, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"[patch_evals] {label}: expected {count} occurrence(s) of anchor, found {n}: {old[:60]!r}")
    return text.replace(old, new)


SUMMARY_FIELDS = (
    '        "tilt_deg": tilt_patch.TILT_DEG,\n'
    '        "tilt_mode": tilt_patch.TILT_MODE,\n'
    '        "tilt_axis": tilt_patch.TILT_AXIS,\n'
    '        "tilt_about": tilt_patch.TILT_ABOUT,\n'
    '        "tilt_measured_deg_mean": float(tilt_patch.measured_tilt_deg(raw_env).mean().item()),\n'
    '        "tilt_measured_deg_max": float(tilt_patch.measured_tilt_deg(raw_env).max().item()),\n'
)

# ---------------------------------------------------------------- eval_frozen.py (T-A / rl_games)
src = os.path.join(root, "eval_frozen.py")
dst = os.path.join(root, "eval_frozen_tilt.py")
t = open(src, encoding="utf-8").read()
t = must_replace(
    t,
    "from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg\n",
    "from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg\n"
    "\n"
    "import sys as _sys\n"
    "_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
    "import tilt_patch  # noqa: E402  (class-level patch of FactoryEnv.randomize_initial_state; TILT_DEG env var)\n",
    "eval_frozen import",
)
t = must_replace(
    t,
    "    obs = env.reset()\n",
    "    obs = env.reset()\n"
    "    _mt = tilt_patch.measured_tilt_deg(raw_env)\n"
    "    print(f\"[tilt_patch] measured fixture tilt after first reset: mean={_mt.mean().item():.3f} \"\n"
    "          f\"min={_mt.min().item():.3f} max={_mt.max().item():.3f} deg (requested {tilt_patch.TILT_DEG})\", flush=True)\n",
    "eval_frozen reset print",
)
t = must_replace(
    t,
    '        "noise_mm": args_cli.fixed_pos_noise_mm,\n',
    '        "noise_mm": args_cli.fixed_pos_noise_mm,\n' + SUMMARY_FIELDS,
    "eval_frozen summary",
)
t = t.replace("_DESC_ = ", "_DESC_ = 'TILT VARIANT (tilt_patch, env TILT_DEG). ' + ", 1) if "_DESC_ = " in t else t
open(dst, "w", encoding="utf-8").write(t)
print("[patch_evals] wrote", dst)

# ---------------------------------------------------------------- student/eval_ablate_student.py
src = os.path.join(student_dir, "eval_ablate_student.py")
dst = os.path.join(student_dir, "eval_ablate_student_tilt.py")
t = open(src, encoding="utf-8").read()
t = must_replace(
    t,
    "import tb_camera_env  # noqa: E402",
    "import tb_camera_env  # noqa: E402\n"
    "import tilt_patch  # noqa: E402  (parent dir is on sys.path; class-level patch, TILT_DEG env var)",
    "student import",
)
# find the first env.reset() call inside main and print the measured tilt right after it
m = re.search(r"\n(\s*)([\w, _]+?)\s*=\s*env\.reset\(\)\n", t)
if not m:
    raise SystemExit("[patch_evals] student: could not find env.reset() assignment")
indent = m.group(1)
insert = (
    f"{indent}_mt = tilt_patch.measured_tilt_deg(env.unwrapped)\n"
    f"{indent}print(f\"[tilt_patch] measured fixture tilt after first reset: mean={{_mt.mean().item():.3f}} \"\n"
    f"{indent}      f\"min={{_mt.min().item():.3f}} max={{_mt.max().item():.3f}} deg (requested {{tilt_patch.TILT_DEG}})\", flush=True)\n"
)
t = t[: m.end()] + insert + t[m.end():]
anchor = '        "cam_jitter_pos_mm":'
if t.count(anchor) != 1:
    raise SystemExit(f"[patch_evals] student summary anchor count = {t.count(anchor)}")
fields = SUMMARY_FIELDS.replace("raw_env", "env.unwrapped")
t = t.replace(anchor, fields + anchor, 1)
open(dst, "w", encoding="utf-8").write(t)
print("[patch_evals] wrote", dst)
