"""Dataset for BC training on collected teacher rollouts.

Expected on-disk layout (written by collect_camera_rollouts.py, P2-2):
  <data_root>/<task>/shard_XXXX.npz, each containing arrays for up to 100
  SUCCESSFUL episodes:
    student_obs   : (E, L, 24) float32   -- noisy state, official T-A obs layout
    wrench_raw    : (E, L, 8, 6) float32 -- per-substep raw wrench, physics-rate
    tp_jpeg       : object array of E*L JPEG-encoded bytes (variable length)
    wrist_jpeg    : object array of E*L JPEG-encoded bytes
    action        : (E, L, 7) float32    -- teacher action actually executed
    ep_len        : (E,) int32           -- valid length per episode (<=L, padded with 0)
    noise_std_mm  : (E,) float32         -- per-episode fixed_pos_obs_noise std used at collection

Each __getitem__ returns ONE (state, wrench_hist, tp_rgb, wrist_rgb, action)
transition, with wrench_hist built as a rolling window of the last
`wrench_horizon` physics-rate samples (32 = 4 policy steps x 8 substeps),
zero-padded at episode start -- matching how the same window is built
online during rollout/eval (see eval_frozen_student.py).
"""

import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

WRENCH_HORIZON = 32  # 4 policy steps x 8 substeps/step


class ForgeRolloutDataset(Dataset):
    def __init__(self, data_root, task, split="train", val_fraction=0.1, seed=0, shard_cache=None,
                 augment=False):
        """shard_cache: pass an already-loaded {shard_idx: {...}} dict (e.g. from
        a sibling train/val split's .shard_cache) to avoid a second full
        np.load of every shard -- train_ds and val_ds otherwise each hold an
        independent full copy of the entire dataset in RAM simultaneously.

        augment: photometric-only image augmentation (sim->real color/lighting
        robustness). Train split only by convention -- keep val clean so
        val_loss stays comparable across runs. Geometry is never touched:
        the spatial layout must stay consistent with state/action labels.
        """
        self.task = task
        self.augment = augment
        shard_paths = sorted(glob.glob(os.path.join(data_root, task, "shard_*.npz")))
        if not shard_paths:
            raise FileNotFoundError(f"no shards found under {os.path.join(data_root, task)}")

        # Split by EPISODE (not by transition) so val episodes are fully held out.
        episodes = []  # list of dicts, one per episode, holding references into loaded shards
        if shard_cache is not None:
            self._shard_cache = shard_cache
            for shard_idx in range(len(shard_paths)):
                n_ep = self._shard_cache[shard_idx]["ep_len"].shape[0]
                for e in range(n_ep):
                    episodes.append((shard_idx, e))
        else:
            self._shard_cache = {}
            for shard_idx, path in enumerate(shard_paths):
                d = np.load(path, allow_pickle=True)
                n_ep = d["ep_len"].shape[0]
                for e in range(n_ep):
                    episodes.append((shard_idx, e))
                self._shard_cache[shard_idx] = {
                    "path": path,
                    "student_obs": d["student_obs"],
                    "wrench_raw": d["wrench_raw"],
                    "tp_jpeg": d["tp_jpeg"],
                    "wrist_jpeg": d["wrist_jpeg"],
                    "action": d["action"],
                    "ep_len": d["ep_len"],
                }

        rng = np.random.RandomState(seed)
        order = rng.permutation(len(episodes))
        n_val = max(1, int(len(episodes) * val_fraction))
        val_idx = set(order[:n_val].tolist())
        keep = [episodes[i] for i in range(len(episodes)) if (i in val_idx) == (split == "val")]
        self.episodes = keep

        # Precompute a flat transition index: (episode_pos, t) for every valid step.
        self._index = []
        for ep_pos, (shard_idx, e) in enumerate(self.episodes):
            ep_len = int(self._shard_cache[shard_idx]["ep_len"][e])
            for t in range(ep_len):
                self._index.append((ep_pos, t))

    def __len__(self):
        return len(self._index)

    def _decode_jpeg(self, jpeg_bytes):
        arr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

    def _augment_rgb(self, img):
        """Photometric augmentation ("v1"): brightness/contrast/saturation/
        channel-permutation/polarity/noise/blur. Ranges deliberately wide
        enough to cover the real black-parts-on-white-bases palette from a
        sim palette of gold/blue/gray -- the goal is color-INVARIANCE, not
        matching any specific real color.

        Randomness comes from torch (DataLoader seeds torch per worker);
        seeding numpy's global RNG here would silently duplicate streams
        across forked workers.
        """
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(seed)

        # per-channel random tone curve (gamma) FIRST, on normalized [0,1].
        # This is the step that actually decorrelates relative brightness
        # between differently-colored objects -- e.g. the sim peg is yellow
        # (low B channel) while the sim gripper/table is light gray (B
        # comparable to R/G); a strong per-channel gamma pushes the
        # low-B object toward black while a high-B object stays light. The
        # affine brightness/contrast/saturation steps below are GLOBAL and
        # preserve relative ordering -- they alone can never turn a bright
        # object dark while a differently-colored neighbor stays light,
        # which is exactly the black-part-on-white-base relationship in the
        # printed real parts.
        x01 = np.clip(img.astype(np.float32), 0.0, 255.0) / 255.0
        for c in range(3):
            gamma = rng.uniform(0.5, 2.2)
            x01[..., c] = x01[..., c] ** gamma
        x = x01 * 255.0

        # contrast about the per-image mean, then overall brightness scale
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = (x - mean) * rng.uniform(0.6, 1.4) + mean
        x = x * rng.uniform(0.5, 1.5)
        # saturation toward/away from luminance; occasionally full grayscale
        lum = (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        sat = 0.0 if rng.random() < 0.15 else rng.uniform(0.0, 1.3)
        x = lum + (x - lum) * sat
        # coarse hue coverage: random channel permutation
        if rng.random() < 0.3:
            x = x[..., rng.permutation(3)]
        # rare polarity flip: covers contrast-inverted palettes
        if rng.random() < 0.05:
            x = 255.0 - x
        # sensor noise + occasional defocus/motion blur proxy
        x = x + rng.normal(0.0, rng.uniform(0.0, 10.0), size=x.shape).astype(np.float32)
        if rng.random() < 0.3:
            k = int(rng.choice([3, 5]))
            x = cv2.GaussianBlur(x, (k, k), 0)
        return np.clip(x, 0.0, 255.0).astype(np.uint8)

    def __getitem__(self, idx):
        ep_pos, t = self._index[idx]
        shard_idx, e = self.episodes[ep_pos]
        shard = self._shard_cache[shard_idx]

        state = shard["student_obs"][e, t].astype(np.float32)  # (24,)
        action = shard["action"][e, t].astype(np.float32)  # (7,)

        # Rolling wrench window: last WRENCH_HORIZON physics-rate samples
        # ending at policy step t (inclusive), zero-padded at episode start.
        # wrench_raw is (L, 8, 6) per episode -> flatten policy-step axis with
        # substep axis to get a single physics-rate timeline of length L*8.
        wrench_flat = shard["wrench_raw"][e, : t + 1].reshape(-1, 6)  # ((t+1)*8, 6)
        if wrench_flat.shape[0] >= WRENCH_HORIZON:
            wrench_hist = wrench_flat[-WRENCH_HORIZON:]
        else:
            pad = np.zeros((WRENCH_HORIZON - wrench_flat.shape[0], 6), dtype=np.float32)
            wrench_hist = np.concatenate([pad, wrench_flat], axis=0)
        wrench_hist = wrench_hist.astype(np.float32)

        tp_rgb = self._decode_jpeg(shard["tp_jpeg"][e, t])
        wrist_rgb = self._decode_jpeg(shard["wrist_jpeg"][e, t])
        if self.augment:
            # independent draws per camera: two real cameras won't share
            # exposure/white-balance errors
            tp_rgb = self._augment_rgb(tp_rgb)
            wrist_rgb = self._augment_rgb(wrist_rgb)

        return {
            "state": torch.from_numpy(state),
            "wrench": torch.from_numpy(wrench_hist),
            "tp_rgb": torch.from_numpy(tp_rgb),
            "wrist_rgb": torch.from_numpy(wrist_rgb),
            "action": torch.from_numpy(action),
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: dataset.py <data_root> <task> [aug]")
        sys.exit(1)
    use_aug = len(sys.argv) > 3 and sys.argv[3] == "aug"
    ds = ForgeRolloutDataset(sys.argv[1], sys.argv[2], split="train", augment=use_aug)
    print(f"SMOKE: {len(ds)} transitions across {len(ds.episodes)} episodes (augment={use_aug})")
    item = ds[0]
    for k, v in item.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    if use_aug:
        # same index twice -> different pixels (aug active), identical labels
        a, b = ds[0], ds[0]
        diff = (a["tp_rgb"].float() - b["tp_rgb"].float()).abs().mean().item()
        same_action = bool(torch.equal(a["action"], b["action"]))
        print(f"AUG_CHECK: mean_abs_pixel_diff_between_two_draws={diff:.2f} (expect >1), "
              f"action_identical={same_action} (expect True)")
