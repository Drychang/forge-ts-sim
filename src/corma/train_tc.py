"""Train wrapper for T-C (27-dim minimal-privileged teacher): registers
Isaac-Forge-*-TC-v0, then hands off to IsaacLab's rl_games train.py (runpy),
identical pattern to train_tb.py.

  ./isaaclab.sh -p train_tc.py --task Isaac-Forge-PegInsert-TC-v0 --headless \
      --num_envs 128 --seed 0 agent.params.config.full_experiment_name=tc_peg_s0

TC_RANDOMIZE_NOISE defaults to 1 (wide per-episode noise for training).
tc_env import is deferred to after Kit start via the AppLauncher monkeypatch
(same rationale as train_tb.py: isaaclab_tasks transitively needs pxr).
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.app  # noqa: E402

_orig_init = isaaclab.app.AppLauncher.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    import tc_env  # noqa: F401  (registers the TC envs; safe now, Kit is up)


isaaclab.app.AppLauncher.__init__ = _patched_init

ISAACLAB_TRAIN_PY = os.path.expanduser(
    "~/force_vla_research/IsaacLab/scripts/reinforcement_learning/rl_games/train.py"
)
runpy.run_path(ISAACLAB_TRAIN_PY, run_name="__main__")
