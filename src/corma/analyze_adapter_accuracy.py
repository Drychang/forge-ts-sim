"""Offline analysis: trained noise-adapter prediction accuracy, broken down by
noise level and by in-episode contact history length. No Kit/sim needed.

Informs the 27-dim-teacher decision: if the adapter infers 5mm noise to within
~1mm even at episode end (max contact context), an eventual RMA baseline fed
this estimate would be a STRONG competitor; if it degrades at high noise, weak.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter_model import CausalNoiseAdapter


def predict_at(model, seq_np, om, os_, H, dev, end_t):
    """Predict noise from the window ending at step end_t (1-indexed)."""
    lo = max(0, end_t - H)
    w = seq_np[lo:end_t]
    w = (w - om) / os_
    pad = H - w.shape[0]
    mask = np.zeros(H, dtype=bool)
    if pad > 0:
        w = np.concatenate([np.zeros((pad, 24), np.float32), w], 0)
        mask[:pad] = True
    with torch.inference_mode():
        wt = torch.from_numpy(w.astype(np.float32))[None].to(dev)
        mt = torch.from_numpy(mask)[None].to(dev)
        return model(wt, mt)[0].cpu().numpy()  # (3,) mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/forge_ts/adapter_data"))
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/forge_ts/adapter_ckpts"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'task':6} {'noise':>6} {'RMSE_end_mm':>12} {'RMSE_mid_mm':>12} {'RMSE_early_mm':>13} {'n_eps':>6}")
    print("-" * 62)
    for task in ["peg", "gear", "nut"]:
        norm = np.load(os.path.join(args.ckpt_dir, task, "adapter_norm.npz"))
        H = int(norm["H"]); scale = float(norm["target_scale"])
        om, os_ = norm["obs_mean"], norm["obs_std"]
        ck = torch.load(os.path.join(args.ckpt_dir, task, "adapter_best.pt"), map_location=dev)
        model = CausalNoiseAdapter(H=H, target_scale=scale).to(dev)
        model.load_state_dict(ck["model"]); model.eval()

        for nm in ["0", "1", "2.5", "5"]:
            files = glob.glob(os.path.join(args.data_dir, f"{task}_n{nm}.npz"))
            if not files:
                continue
            d = np.load(files[0], allow_pickle=True)
            seqs, noise = d["obs_seqs"], d["noise"]
            errs_end, errs_mid, errs_early = [], [], []
            for i in range(len(seqs)):
                s = seqs[i].astype(np.float32); T = s.shape[0]
                true_mm = noise[i] * scale
                pe = predict_at(model, s, om, os_, H, dev, T)          # end (max context)
                pm = predict_at(model, s, om, os_, H, dev, max(1, T // 2))
                pear = predict_at(model, s, om, os_, H, dev, min(T, 5))
                errs_end.append(np.linalg.norm(pe - true_mm))
                errs_mid.append(np.linalg.norm(pm - true_mm))
                errs_early.append(np.linalg.norm(pear - true_mm))
            print(f"{task:6} {nm:>6} {np.sqrt(np.mean(np.square(errs_end))):>12.3f} "
                  f"{np.sqrt(np.mean(np.square(errs_mid))):>12.3f} "
                  f"{np.sqrt(np.mean(np.square(errs_early))):>13.3f} {len(seqs):>6}")
    print("\nRMSE = 3D noise-vector prediction error (mm). end=window at episode end "
          "(max contact history), mid=halfway, early=first 5 steps.")
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
