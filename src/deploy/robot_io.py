"""Abstract deployment I/O contract for the FORGE-TS student policy.

Any concrete backend (sim, real Franka) implements this interface. deploy_loop.py
is written entirely against RobotIO -- it never imports isaaclab or libfranka
directly, so `SimRobotIO` (loopback verification, see sim_io.py) and a future
`FrankaRobotIO` (franka_io_stub.py) are drop-in swaps.

Units/frames (must match what train_student_bc.py's norm_stats.npz saw):
  - get_obs24(): 24-dim official noisy-state layout, SAME layout/order as
    tb_camera_env.TA_OBS_ORDER (fingertip_pos_rel_fixed[3], fingertip_quat[4],
    ee_linvel[3], ee_angvel[3], ft_force[3], force_threshold[1], prev_actions[7]
    -- see tb_camera_env.py for the exact concat order and field widths).
  - get_wrench_window(): (32, 6) float32, raw/unsmoothed 6D wrench
    [Fx,Fy,Fz,Tx,Ty,Tz] in the SAME frame eval_frozen_student.py used
    (world-frame force_sensor_world, oldest-first, 32 = 4 policy steps x
    8 physics substeps at 120Hz -- a real F/T sensor sampled at ~120Hz and
    windowed the same way satisfies this without resampling).
  - get_images(): {"tp": (256,256,3) uint8 RGB, "wrist": (256,256,3) uint8 RGB}.
  - send_action(a7): 7-dim, only a7[0:6] have any physical effect (3 position +
    3 rotation delta, see forge_env.py _apply_action). a7[6] is a training-time
    auxiliary "policy_success_pred" signal used ONLY in the sim reward function
    (forge_env._get_rewards line ~102) -- it has NO control effect and does
    NOT need to be sent to a real robot controller. Kept in the signature only
    so the same tensor shape flows through unmodified from the network output.
"""
import abc


class RobotIO(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        """Move to the episode start pose. Returns nothing; call the getters after."""

    @abc.abstractmethod
    def get_obs24(self):
        """-> (24,) float32 (numpy or torch, caller casts)."""

    @abc.abstractmethod
    def get_wrench_window(self):
        """-> (32, 6) float32, oldest-first."""

    @abc.abstractmethod
    def get_images(self):
        """-> {"tp": (256,256,3) uint8, "wrist": (256,256,3) uint8}."""

    @abc.abstractmethod
    def send_action(self, action7):
        """action7: (7,) float32 in [-1,1] (already the raw network output,
        pos/rot components still need the backend's own pos_action_bounds /
        rot_action_bounds scaling -- see forge_env._apply_action step (0))."""

    @abc.abstractmethod
    def is_done(self):
        """-> bool. Sim: episode timeout. Real: wall-clock step budget or
        operator-triggered stop (e-stop / success button)."""

    def get_wrench_step(self):
        """-> (8, 6) float32: THIS policy step's 8 newest physics-rate (120Hz)
        wrench samples, oldest-first. Required by record_real_episodes.py
        (the shard schema stores per-step 8x6 blocks, from which the training
        pipeline reconstructs 32-sample windows). Inference-only backends may
        skip this; the recorder needs it."""
        raise NotImplementedError(f"{type(self).__name__} does not implement get_wrench_step()")

    def set_frame_offset(self, offset_xyz_m):
        """Inject a KNOWN offset (meters, base frame) into the estimated
        target frame -- used to replicate the sim noise axis on the real
        robot (zero-shot offset-sweep protocol). Returns True if applied.
        Default: unsupported (sim's noise is driven by env cfg instead)."""
        return False

    def close(self):
        """Optional cleanup hook; default no-op."""
        return None
