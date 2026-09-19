"""Frozen-protocol eval wrapper for the ARCH-INSERT baseline: registers the
Isaac-Forge-*-ARCH-v0 ids then hands off to eval_frozen.py UNMODIFIED (runpy),
exactly like eval_frozen_tb.py does for T-B. Protocol (seed 42, 256 eps,
Wilson CI, success hook, noise/dyn_rand axes) is therefore byte-identical to
every other baseline in this project.

  python eval_arch.py --task Isaac-Forge-PegInsert-ARCH-v0 \
      --checkpoint <arch ckpt> --fixed_pos_noise_mm 5 --dyn_rand on \
      --episodes 256 --num_envs 32 --tag arch_peg_n5 --headless
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # arch_env
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # eval_frozen.py

import isaaclab.app  # noqa: E402

_orig_init = isaaclab.app.AppLauncher.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    import arch_env  # noqa: F401  (registers ARCH envs once Kit is up)


isaaclab.app.AppLauncher.__init__ = _patched_init

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_frozen.py"),
    run_name="__main__",
)
