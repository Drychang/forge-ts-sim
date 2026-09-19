"""ShardWriter: writes episodes into the EXACT npz shard schema that
student/dataset.py's ForgeRolloutDataset expects (same schema the sim
collector collect_camera_rollouts.py produced), so real-robot data can be
fed to train_student_bc.py with ZERO changes:

  student_obs   : (E, L, 24) float32   -- RAW (un-normalized) official obs layout
  wrench_raw    : (E, L, 8, 6) float32 -- per-policy-step 8 newest physics-rate samples
  tp_jpeg       : (E, L) object array of JPEG bytes (padding cells hold b"")
  wrist_jpeg    : (E, L) object array of JPEG bytes
  action        : (E, L, 7) float32    -- executed action in [-1,1]
  ep_len        : (E,) int32           -- valid steps per episode (padded rows beyond are junk)
  noise_std_mm  : (E,) float32         -- injected offset magnitude in mm (0 if none)

Padding: episodes shorter than the longest in the shard are zero-padded;
ForgeRolloutDataset only ever indexes t < ep_len so padding is never read.
"""
import os

import cv2
import numpy as np

JPEG_QUALITY = 90


def encode_jpeg_rgb(img_rgb_uint8):
    """RGB uint8 (H,W,3) -> JPEG bytes. Mirror of dataset._decode_jpeg, which
    does imdecode -> cvtColor(BGR2RGB); so we must cvtColor(RGB2BGR) -> imencode."""
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


class ShardWriter:
    def __init__(self, out_dir, shard_size=100, start_idx=0):
        self.out_dir = os.path.abspath(os.path.expanduser(out_dir))
        os.makedirs(self.out_dir, exist_ok=True)
        self.shard_size = shard_size
        self.next_idx = start_idx
        self._eps = []

    def add_episode(self, student_obs, wrench_raw, tp_jpeg, wrist_jpeg, action, noise_std_mm=0.0):
        """student_obs (L,24) f32; wrench_raw (L,8,6) f32; tp_jpeg/wrist_jpeg:
        list of L JPEG-bytes; action (L,7) f32; noise_std_mm: float."""
        L = len(student_obs)
        assert len(wrench_raw) == L and len(tp_jpeg) == L and len(wrist_jpeg) == L and len(action) == L, \
            f"length mismatch: obs={len(student_obs)} wrench={len(wrench_raw)} tp={len(tp_jpeg)} " \
            f"wrist={len(wrist_jpeg)} act={len(action)}"
        self._eps.append({
            "student_obs": np.asarray(student_obs, dtype=np.float32),
            "wrench_raw": np.asarray(wrench_raw, dtype=np.float32),
            "tp_jpeg": list(tp_jpeg),
            "wrist_jpeg": list(wrist_jpeg),
            "action": np.asarray(action, dtype=np.float32),
            "noise_std_mm": float(noise_std_mm),
        })
        if len(self._eps) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._eps:
            return None
        E = len(self._eps)
        Lmax = max(ep["student_obs"].shape[0] for ep in self._eps)

        student_obs = np.zeros((E, Lmax, 24), dtype=np.float32)
        wrench_raw = np.zeros((E, Lmax, 8, 6), dtype=np.float32)
        action = np.zeros((E, Lmax, 7), dtype=np.float32)
        ep_len = np.zeros((E,), dtype=np.int32)
        noise_std_mm = np.zeros((E,), dtype=np.float32)
        tp_jpeg = np.empty((E, Lmax), dtype=object)
        wrist_jpeg = np.empty((E, Lmax), dtype=object)
        tp_jpeg[:] = b""
        wrist_jpeg[:] = b""

        for e, ep in enumerate(self._eps):
            L = ep["student_obs"].shape[0]
            student_obs[e, :L] = ep["student_obs"]
            wrench_raw[e, :L] = ep["wrench_raw"]
            action[e, :L] = ep["action"]
            ep_len[e] = L
            noise_std_mm[e] = ep["noise_std_mm"]
            for t in range(L):
                tp_jpeg[e, t] = ep["tp_jpeg"][t]
                wrist_jpeg[e, t] = ep["wrist_jpeg"][t]

        path = os.path.join(self.out_dir, f"shard_{self.next_idx:04d}.npz")
        np.savez(path, student_obs=student_obs, wrench_raw=wrench_raw,
                 tp_jpeg=tp_jpeg, wrist_jpeg=wrist_jpeg, action=action,
                 ep_len=ep_len, noise_std_mm=noise_std_mm)
        print(f"[ShardWriter] wrote {path} ({E} episodes, Lmax={Lmax})", flush=True)
        self.next_idx += 1
        self._eps = []
        return path

    def close(self):
        return self.flush()
