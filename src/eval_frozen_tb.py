"""Eval wrapper for T-B: forces TB_RANDOMIZE_NOISE off (so the wide training-
time noise resampling in tb_env.ForgeTBEnv is a no-op and the frozen
protocol's own --fixed_pos_noise_mm / --dyn_rand sweep in eval_frozen.py
drives the noise level unmodified, exactly like T-A), registers the
Isaac-Forge-*-TB-v0 gym ids, then hands off to eval_frozen.py unmodified.

Usage: identical to eval_frozen.py, just point --task at a *-TB-v0 id.
  ./isaaclab.sh -p eval_frozen_tb.py --task Isaac-Forge-PegInsert-TB-v0 \
      --checkpoint <tb ckpt> --fixed_pos_noise_mm 5 --dyn_rand on --tag tb_noisespec_s0 --headless

IMPORTANT: see train_tb.py's docstring -- `import tb_env` must not happen
until Kit has started (AppLauncher constructed), so it's deferred via an
AppLauncher.__init__ monkeypatch rather than a top-level import.
"""
import os
import runpy
import sys

os.environ["TB_RANDOMIZE_NOISE"] = "0"  # MUST be set before tb_env is ever imported

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.app  # noqa: E402  (safe pre-Kit import; see docstring)

_orig_init = isaaclab.app.AppLauncher.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    import tb_env  # noqa: F401  (side effect: gym.register the TB envs; safe now, Kit is up)


isaaclab.app.AppLauncher.__init__ = _patched_init

EVAL_FROZEN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_frozen.py")
runpy.run_path(EVAL_FROZEN_PY, run_name="__main__")
