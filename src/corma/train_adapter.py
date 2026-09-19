"""CoRMA/C3 Phase 2: train the noise-regression adapter on collected T-B data.

Loads {task}_n*.npz (obs_seqs object array of (T,24), noise (N,3) in meters),
builds (H,24)-window -> noise(mm) regression pairs, trains CausalNoiseAdapter.
Saves adapter.pt + norm_stats.npz. State-only, fast (no cameras, no Isaac).
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter_model import CausalNoiseAdapter


def load_task(data_dir, task):
    obs_eps, noise_eps = [], []
    files = sorted(glob.glob(os.path.join(data_dir, f"{task}_n*.npz")))
    if not files:
        raise SystemExit(f"no data for task {task} in {data_dir}")
    for f in files:
        d = np.load(f, allow_pickle=True)
        for i in range(len(d["obs_seqs"])):
            obs_eps.append(d["obs_seqs"][i].astype(np.float32))   # (T,24)
            noise_eps.append(d["noise"][i].astype(np.float32))    # (3,) meters
    return obs_eps, noise_eps, files


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, obs_eps, noise_eps, H, obs_mean, obs_std, target_scale, samples_per_ep=8):
        self.obs, self.noise, self.H = obs_eps, noise_eps, H
        self.m, self.s, self.scale = obs_mean, obs_std, target_scale
        self.spe = samples_per_ep
        self.index = [(e, k) for e in range(len(obs_eps)) for k in range(samples_per_ep)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        e, _ = self.index[idx]
        seq = self.obs[e]                                          # (T,24)
        T = seq.shape[0]
        t = np.random.randint(1, T + 1)                           # window ends at t
        lo = max(0, t - self.H)
        w = seq[lo:t]                                             # (<=H,24)
        w = (w - self.m) / self.s
        pad = self.H - w.shape[0]
        mask = np.zeros(self.H, dtype=bool)
        if pad > 0:
            w = np.concatenate([np.zeros((pad, 24), np.float32), w], 0)
            mask[:pad] = True                                     # left-pad masked
        label = self.noise[e] * self.scale                       # mm
        return (torch.from_numpy(w.astype(np.float32)),
                torch.from_numpy(mask),
                torch.from_numpy(label.astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/forge_ts/adapter_data"))
    ap.add_argument("--task", required=True, choices=["peg", "gear", "nut"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)

    obs_eps, noise_eps, files = load_task(args.data_dir, args.task)
    n = len(obs_eps)
    print(f"[adapter:{args.task}] {n} episodes from {len(files)} files", flush=True)

    # obs norm stats over all steps
    allobs = np.concatenate([o for o in obs_eps], 0)
    m = allobs.mean(0); s = allobs.std(0) + 1e-6
    scale = 1000.0

    idx = np.random.permutation(n)
    nval = max(1, int(n * args.val_frac))
    val_i, tr_i = set(idx[:nval].tolist()), idx[nval:].tolist()
    tr = WindowDataset([obs_eps[i] for i in tr_i], [noise_eps[i] for i in tr_i], args.H, m, s, scale)
    va = WindowDataset([obs_eps[i] for i in val_i], [noise_eps[i] for i in val_i], args.H, m, s, scale, samples_per_ep=4)
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = CausalNoiseAdapter(H=args.H, target_scale=scale).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val = float("inf")

    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for w, mask, y in tl:
            w, mask, y = w.to(dev), mask.to(dev), y.to(dev)
            pred = model(w, mask)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        # val: report RMSE in mm
        model.eval(); vt = 0.0; vn = 0
        with torch.no_grad():
            for w, mask, y in vl:
                w, mask, y = w.to(dev), mask.to(dev), y.to(dev)
                vt += ((model(w, mask) - y) ** 2).mean().item(); vn += 1
        vrmse = (vt / max(vn, 1)) ** 0.5
        print(f"epoch {ep+1}/{args.epochs} train_mse={tot/max(nb,1):.4f} val_rmse_mm={vrmse:.4f}", flush=True)
        if vrmse < best_val:
            best_val = vrmse
            torch.save({"model": model.state_dict(),
                        "config": {"H": args.H, "target_scale": scale}},
                       os.path.join(os.path.expanduser(args.out_dir), "adapter_best.pt"))
    np.savez(os.path.join(os.path.expanduser(args.out_dir), "adapter_norm.npz"),
             obs_mean=m, obs_std=s, target_scale=scale, H=args.H)
    print(f"ADAPTER_TRAIN_DONE task={args.task} best_val_rmse_mm={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
