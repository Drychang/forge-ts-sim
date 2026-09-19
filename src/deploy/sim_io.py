"""SimRobotIO: RobotIO backed by the Isaac-Forge-*-TBCamera-v0 env.

Used for the G4 loopback verification: run deploy_loop.py through THIS
backend and confirm SR matches eval_frozen_student.py within noise. If it
does, the deployment loop/adapter code has zero gap vs the validated eval
harness, and any future real-robot discrepancy is isolable to hardware/
perception, not the deployment software stack.

num_envs is fixed at 1 -- this backend drives one episode at a time, same as
a real robot. Noise/dyn_rand knobs are still configurable (same env_cfg
overrides as eval_frozen_student.py) so loopback can be tested across the
same condition grid.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # tb_camera_env.py lives in src/
os.environ.setdefault("TB_RANDOMIZE_NOISE", "0")

from robot_io import RobotIO

WRENCH_HORIZON = 32
SUBSTEPS_PER_STEP = 8


def apply_frozen_dynamics(env_cfg):
    """Verbatim from eval_frozen_student.py -- keep in sync if that changes."""
    env_cfg.ctrl.task_prop_gains_noise_level = [0.0] * 6
    env_cfg.ctrl.pos_threshold_noise_level = [0.0] * 3
    env_cfg.ctrl.rot_threshold_noise_level = [0.0] * 3
    ema_lo, ema_hi = env_cfg.ctrl.ema_factor_range
    ema_nominal = 0.5 * (ema_lo + ema_hi)
    env_cfg.ctrl.ema_factor_range = [ema_nominal, ema_nominal]
    env_cfg.ctrl.default_dead_zone = [0.0] * 6
    env_cfg.events.dead_zone_thresholds = None
    c_lo, c_hi = env_cfg.task.contact_penalty_threshold_range
    c_nominal = 0.5 * (c_lo + c_hi)
    env_cfg.task.contact_penalty_threshold_range = [c_nominal, c_nominal]
    env_cfg.events.object_scale_mass.params["mass_distribution_params"] = (0.0, 0.0)
    env_cfg.events.fixed_physics_material.params["static_friction_range"] = (0.75, 0.75)


class SimRobotIO(RobotIO):
    def __init__(self, task_gym_id, noise_mm=1.0, dyn_rand="on", protocol_seed=42, device=None):
        import gymnasium as gym

        import tb_camera_env  # noqa: F401  side-effect: gym.register
        from isaaclab.utils.seed import configure_seed
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(task_gym_id, num_envs=1)
        if device is not None:
            env_cfg.sim.device = device
        env_cfg.seed = protocol_seed
        noise_m = noise_mm / 1000.0
        env_cfg.obs_rand.fixed_asset_pos = [noise_m, noise_m, noise_m]
        if dyn_rand == "off":
            apply_frozen_dynamics(env_cfg)

        self.env = gym.make(task_gym_id, cfg=env_cfg, render_mode=None)
        self.raw_env = self.env.unwrapped
        self.device = self.raw_env.device
        self._configure_seed = configure_seed
        self._protocol_seed = protocol_seed
        self._wrench_buf = torch.zeros((1, WRENCH_HORIZON, 6), device=self.device)

        # success-capture hook, identical mechanism to eval_frozen_student.py
        self._cap = {}
        orig_get_rewards = self.raw_env._get_rewards

        def hooked_get_rewards():
            rew = orig_get_rewards()
            if bool(self.raw_env.reset_buf.any()):
                check_rot = self.raw_env.cfg_task.name == "nut_thread"
                self._cap["final_success"] = bool(self.raw_env._get_curr_successes(
                    success_threshold=self.raw_env.cfg_task.success_threshold, check_rot=check_rot
                )[0])
            return rew

        self.raw_env._get_rewards = hooked_get_rewards
        self._done = False
        self._seeded_once = False

    def reset(self):
        if not self._seeded_once:
            # freeze point, once: matches eval_frozen_student.py's protocol
            self._configure_seed(self._protocol_seed)
            self._seeded_once = True
        self.env.reset()
        self._wrench_buf.zero_()
        self._cap["final_success"] = None
        self._done = False

    def get_obs24(self):
        return self.raw_env.student_obs.to(torch.float32)[0]

    def get_wrench_window(self):
        new = self.raw_env.get_wrench_history()  # (1, 8, 6) this-step substeps
        self._wrench_buf[:] = torch.cat([self._wrench_buf[:, SUBSTEPS_PER_STEP:], new], dim=1)
        return self._wrench_buf[0]

    def get_wrench_step(self):
        """(8,6) newest physics-rate samples this policy step. Idempotent per
        step (get_wrench_history returns a snapshot set once per env step),
        so calling both this and get_wrench_window in the same step is safe."""
        return self.raw_env.get_wrench_history()[0]

    def get_images(self):
        images = self.raw_env.get_camera_images()
        return {"tp": images["tp_rgb"][0], "wrist": images["wrist_rgb"][0]}

    def send_action(self, action7):
        a = action7.reshape(1, -1).to(self.device)
        _, _, terminated, truncated, _ = self.env.step(a)
        self._done = bool((terminated | truncated).any())

    def is_done(self):
        return self._done

    def last_success(self):
        """Sim-only convenience (real robot has no ground-truth success flag)."""
        return self._cap.get("final_success")

    def close(self):
        self.env.close()
