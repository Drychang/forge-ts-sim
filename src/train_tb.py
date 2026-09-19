"""Train wrapper for T-B: registers Isaac-Forge-*-TB-v0 gym ids, then hands
off to IsaacLab's own rl_games train.py unmodified (runpy so its
@hydra_task_config decorator resolves against our already-registered ids).

Usage: identical to the official train.py, just point --task at a *-TB-v0 id.
  ./isaaclab.sh -p train_tb.py --task Isaac-Forge-PegInsert-TB-v0 --headless \
      --num_envs 128 --seed 0 agent.params.config.full_experiment_name=tb_peg_s0

TB_RANDOMIZE_NOISE defaults to "1" (wide per-episode noise resampling) which
is exactly what training needs -- no env var setting required for training.

IMPORTANT: `import tb_env` must NOT happen until AFTER Isaac Sim/Kit has
started (AppLauncher constructed) -- tb_env pulls in isaaclab_tasks, which
transitively needs `pxr` (USD bindings) that only become importable once
Kit's own extensions are loaded. `isaaclab.app` itself is safe to import
early (its whole job is to bootstrap Kit before anything else needs it).
We monkeypatch AppLauncher.__init__ so the import fires the instant
train.py's own AppLauncher(args_cli) call (inside the runpy'd code below)
finishes starting Kit -- well before train.py's @hydra_task_config
decorator (which needs our gym ids already registered) is evaluated.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.app  # noqa: E402  (safe pre-Kit import; see docstring)

_orig_init = isaaclab.app.AppLauncher.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    import tb_env  # noqa: F401  (side effect: gym.register the TB envs; safe now, Kit is up)


isaaclab.app.AppLauncher.__init__ = _patched_init

ISAACLAB_TRAIN_PY = os.path.expanduser(
    "~/force_vla_research/IsaacLab/scripts/reinforcement_learning/rl_games/train.py"
)
runpy.run_path(ISAACLAB_TRAIN_PY, run_name="__main__")
