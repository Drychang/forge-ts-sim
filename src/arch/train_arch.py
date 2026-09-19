"""Train wrapper for the ARCH-INSERT primitive baseline: registers
Isaac-Forge-*-ARCH-v0 then hands off to IsaacLab's rl_games train.py (runpy),
same pattern as train_tb.py / train_tc.py.

  python train_arch.py --task Isaac-Forge-PegInsert-ARCH-v0 --headless \
      --num_envs 128 --seed 0 --max_iterations 200 \
      agent.params.config.full_experiment_name=arch_peg_s0
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.app  # noqa: E402

_orig_init = isaaclab.app.AppLauncher.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    import arch_env  # noqa: F401  (registers ARCH envs after Kit is up)


isaaclab.app.AppLauncher.__init__ = _patched_init

runpy.run_path(
    os.path.expanduser("~/force_vla_research/IsaacLab/scripts/reinforcement_learning/rl_games/train.py"),
    run_name="__main__",
)
