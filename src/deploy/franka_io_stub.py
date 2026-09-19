"""FrankaRobotIO skeleton -- fill in once the lab Franka is available.

This is INTENTIONALLY incomplete (TODOs mark every hardware-dependent piece).
Its purpose now is to lock the 24-dim obs layout, wrench convention, and
action semantics against forge_env.py source truth, so implementation later
is "fill in the TODOs" rather than "figure out the contract from scratch".

Prereqs to install/verify when hardware is available (not done yet):
  - libfranka + franka_ros2 (or raw libfranka C++ bound via pybind), matched
    to the arm's firmware version.
  - pyrealsense2, 2x RealSense (tp + wrist mount).
  - Read FORGE's real paper appendix / IsaacLab Factory README for whether
    the ORIGINAL FORGE authors used O_F_ext_hat_K or an external F/T sensor --
    matching their choice removes a variable when comparing to published
    real-robot numbers.

24-dim obs layout (MUST match tb_camera_env.TA_OBS_ORDER exactly, source:
tb_camera_env.py + forge_env.py _get_observations):
  [0:3]   fingertip_pos_rel_fixed = fingertip_pos_world - estimated_target_pos_world
          "estimated_target_pos_world" = wherever your perception/calibration
          pipeline PLACES the fixed-asset frame (hole/gear-base/bolt tip).
          Whatever error is baked into that estimate IS the real-world analog
          of --fixed_pos_noise_mm -- no noise needs to be added artificially.
  [3:7]   fingertip_quat (wxyz, Isaac convention -- ROS is xyzw, MUST convert)
  [7:10]  ee_linvel   (finite-difference of fingertip_pos, or from O_dP_EE_c)
  [10:13] ee_angvel   (finite-difference of fingertip_quat, or O_dP_EE_c ang part)
  [13:16] ft_force    (3D force in the SAME frame convention forge_env.py
                       uses -- see force_sensor_smooth; likely the fixed-asset/
                       noisy frame, verify by reading forge_env._get_observations
                       before wiring this up)
  [16]    force_threshold = a scalar YOU choose per episode (sim randomizes
                       this per cfg_task.contact_penalty_threshold_range; for
                       deployment just pick the sim's nominal/mean value
                       unless you have a specific reason to vary it)
  [17:24] prev_actions (7,) -- the action YOU sent last step; zero at episode start

Wrench window (32,6): raw, unsmoothed, oldest-first, 32 samples covering the
last 4 policy steps x 8 physics substeps (sim: 120Hz). A real F/T source
sampled at ~120Hz and ring-buffered the same way satisfies this without
resampling; if your sensor runs at a different rate, resample to 120Hz
equivalent BEFORE windowing, don't just take the last 32 raw samples at a
different native rate (that silently changes the time window meaning: 32
sim samples = 267ms elapsed, per the original student-training pipeline).

Camera images: 256x256 RGB uint8, matching tp/wrist mounting implied by
tb_camera_env.py's TiledCameraCfg offsets (tp: pos=(1.0,0,0.4) relative to
robot base, wrist: pos=(0.13,0,-0.15) relative to panda_hand -- use these as
the STARTING point for real camera placement/FOV matching, see execution
queue G0-era discussion: FOV crop to ~47 deg to match sim's PinholeCameraCfg).

Action (7,): only [0:6] have any physical effect (see robot_io.py docstring
and forge_env._apply_action -- action[6] is a sim-training-only auxiliary
signal with zero control effect, do not try to "port" it).
  [0:3] position delta in the SAME estimated-target frame as the obs above,
        scaled by pos_action_bounds=[0.05,0.05,0.05] m (i.e. network output
        in [-1,1] * 0.05m), then further clipped to pos_threshold of current
        EE pose per-step (see forge_env._apply_action step 2.a) -- replicate
        both the scale AND the per-step clip, not just the scale.
  [3:6] rotation delta, rot_action_bounds=[1,1,1] (see _apply_action for the
        yaw joint-limit remapping -- this logic is Franka-joint-limit-specific
        and should carry over unchanged since it's the same robot).
"""
from robot_io import RobotIO


class FrankaRobotIO(RobotIO):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Hardware not available yet -- see module docstring for the exact "
            "contract to implement. Do not guess at frame conventions; verify "
            "each one against forge_env.py before wiring up the real robot."
        )

    def reset(self):
        raise NotImplementedError  # TODO: move to a fixed/randomized start pose via move_to_joint_position or a Cartesian motion; open gripper if applicable, grasp held part

    def get_obs24(self):
        raise NotImplementedError  # TODO: assemble per the layout above from robot_state + your perception pipeline's target-frame estimate

    def get_wrench_window(self):
        raise NotImplementedError  # TODO: ring buffer over O_F_ext_hat_K (or external F/T), resampled to 120Hz-equivalent windowing

    def get_images(self):
        raise NotImplementedError  # TODO: pyrealsense2 frame grab + resize/crop to 256x256

    def send_action(self, action7):
        raise NotImplementedError  # TODO: task-space impedance/admittance command using action7[0:6] only, per the scaling/clipping rules above

    def is_done(self):
        raise NotImplementedError  # TODO: wall-clock step budget matching the sim episode_length_s for this task, OR an operator e-stop/success signal

    def get_wrench_step(self):
        raise NotImplementedError  # TODO: return (8,6) float32 -- the 8 NEWEST samples of the 120Hz-resampled F/T ring buffer covering THIS 15Hz policy tick, oldest-first. Required by record_real_episodes.py (shard schema). NOT the last 8 raw 1kHz samples -- resample to 120Hz-equivalent first.

    def set_frame_offset(self, offset_xyz_m):
        raise NotImplementedError  # TODO: add offset (meters, robot base frame) to the estimated target pose used for obs[0:3] and the action frame; return True. Required by --inject_offset_mm (Phase 3 zero-shot offset sweep) -- record_real_episodes ABORTS if this returns False while an offset was requested.
