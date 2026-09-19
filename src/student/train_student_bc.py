"""Behavior-cloning trainer for StudentFMTPolicy.

usage:
  python train_student_bc.py --data_root /media/data/forge_ts_data \
      --task peg --out_dir ~/forge_ts/student_ckpts/peg --epochs 80 --seed 0

State/wrench are z-scored using statistics computed once from the training
split (stored alongside checkpoints as norm_stats.npz, and consumed again by
eval_frozen_student.py so train/eval normalization always matches). Images
are NOT normalized here -- StudentFMTPolicy.CameraBackbone does ImageNet
normalization internally on raw uint8 input.
"""
import argparse
import os
import time

import numpy as np

# Modality ablation sets. Expressed as sets so the legacy flag names ("both" meaning
# vision+force off, i.e. state-only) keep their exact historical meaning while the new
# state-off cells slot in beside them. Anything trained under a name MUST be evaluated
# under the same name -- eval_ablate_student.py imports this same table.
ABLATE_SETS = {
    "none":         frozenset(),
    "vision":       frozenset({"vision"}),                  # state + force
    "force":        frozenset({"force"}),                   # state + vision
    "both":         frozenset({"vision", "force"}),         # state only   (legacy name)
    "state":        frozenset({"state"}),                   # vision + force
    "state_force":  frozenset({"state", "force"}),          # vision only
    "state_vision": frozenset({"state", "vision"}),         # force only
}
ABLATE_CHOICES = list(ABLATE_SETS)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ForgeRolloutDataset
from student_fmt import StudentFMTConfig, StudentFMTPolicy


def compute_norm_stats(dataset, n_samples=2000, seed=0):
    """Estimate per-channel mean/std for state and wrench from a subsample."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    states, wrenches = [], []
    for i in idx:
        item = dataset[int(i)]
        states.append(item["state"].numpy())
        wrenches.append(item["wrench"].numpy())
    states = np.stack(states)  # (N, 24)
    wrenches = np.concatenate(wrenches, axis=0)  # (N*32, 6)
    return {
        "state_mean": states.mean(0), "state_std": states.std(0) + 1e-6,
        "wrench_mean": wrenches.mean(0), "wrench_std": wrenches.std(0) + 1e-6,
    }


class Normalizer:
    def __init__(self, stats, device):
        self.state_mean = torch.as_tensor(stats["state_mean"], device=device, dtype=torch.float32)
        self.state_std = torch.as_tensor(stats["state_std"], device=device, dtype=torch.float32)
        self.wrench_mean = torch.as_tensor(stats["wrench_mean"], device=device, dtype=torch.float32)
        self.wrench_std = torch.as_tensor(stats["wrench_std"], device=device, dtype=torch.float32)

    def state(self, x):
        return (x - self.state_mean) / self.state_std

    def wrench(self, x):
        return (x - self.wrench_mean) / self.wrench_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["peg", "gear", "nut"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--img_aug", type=str, choices=["none", "v1"], default="none",
                   help="photometric image augmentation (train split only); 'v1' for sim2real color robustness")
    p.add_argument("--ablate", type=str, choices=ABLATE_CHOICES, default="none",
                   help="modality ablation (T3): zero out the named modalities in BOTH train and val. "
                        "Zeroing happens AFTER normalization -- zero-then-normalize would inject a "
                        "-mean/std constant bias instead of no-information. "
                        "'vision'=black images, 'force'=zero wrench, 'both'=state-only (legacy name), "
                        "'state'=vision+force, 'state_force'=vision only, 'state_vision'=force only. "
                        "Eval must use eval_ablate_student.py with the SAME flag.")
    p.add_argument("--dropout_modalities", type=str, default="state,vision,force",
                   help="which modalities --modality_dropout may drop, comma separated. "
                        "The default (all three) made peg MORE vision-dependent, because "
                        "vision is the only channel that can carry the task alone and "
                        "dropping the others just pushes the optimiser onto it. Set to "
                        "'vision' to add only the 'operate without cameras' signal.")
    p.add_argument("--modality_dropout", type=float, default=0.0,
                   help="probability that a training sample has ONE modality zeroed "
                        "(chosen uniformly from state/vision/force). At most one is ever "
                        "dropped, so two always survive. Applied AFTER normalization, on the "
                        "TRAIN split only -- val stays clean so val_loss stays comparable to "
                        "runs without it. 0.0 (default) reproduces the existing recipe "
                        "exactly. Eval needs no matching flag: this shapes the weights, it is "
                        "not part of the observation contract.")
    # ---- real-robot fine-tune options ----
    p.add_argument("--init_from", type=str, default=None,
                   help="warm-start MODEL WEIGHTS ONLY from this checkpoint (fresh optimizer, "
                        "epoch 0). Use for fine-tuning; --resume (weights+optimizer+epoch) is "
                        "for continuing an interrupted run of the SAME experiment.")
    p.add_argument("--freeze_backbone", type=str, choices=["none", "early"], default="none",
                   help="'early' freezes both camera backbones' stem+layer1-3 (BN kept in eval "
                        "mode) and trains only layer4+proj+fusion+head -- the fine-tune recipe "
                        "for small real datasets.")
    p.add_argument("--real_data_dir", type=str, default=None,
                   help="directory of REAL-robot shards (from record_real_episodes.py). When "
                        "given: train = sim-train + real-train x oversample, val = REAL val "
                        "split (early stopping then optimizes real-world fit).")
    p.add_argument("--real_oversample", type=int, default=4,
                   help="how many times the real training episodes are repeated per epoch "
                        "relative to their natural size (real data is tiny vs sim)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_gym_name = {"peg": "PegInsert", "gear": "GearMesh", "nut": "NutThread"}[args.task]
    train_ds = ForgeRolloutDataset(args.data_root, task_gym_name, split="train", seed=0,
                                    augment=(args.img_aug == "v1"))
    # Reuse train_ds's already-loaded shard cache instead of a second full
    # np.load of every shard -- otherwise train_ds+val_ds hold two
    # independent full copies of the dataset in RAM at once (this is what
    # was OOM-killing NutThread even fully serialized with num_workers=1-2).
    # val stays UN-augmented so val_loss is comparable across runs.
    val_ds = ForgeRolloutDataset(args.data_root, task_gym_name, split="val", seed=0,
                                  shard_cache=train_ds._shard_cache)

    if args.real_data_dir:
        # Real shards live directly in real_data_dir (no per-task subfolder):
        # pass "." as task so the glob hits real_data_dir/shard_*.npz.
        real_train = ForgeRolloutDataset(args.real_data_dir, ".", split="train", seed=0,
                                          augment=(args.img_aug == "v1"))
        real_val = ForgeRolloutDataset(args.real_data_dir, ".", split="val", seed=0,
                                        shard_cache=real_train._shard_cache)
        if len(real_train) == 0 or len(real_val) == 0:
            raise SystemExit(
                f"[train_student_bc] ABORT: real data split degenerate "
                f"(train={len(real_train)}, val={len(real_val)} transitions). The val split "
                f"takes at least 1 episode, so you need >=2 real episodes (>=10 recommended -- "
                f"with fewer, a single noisy episode decides best.pt). Record more demos.")
        if len(real_train.episodes) < 9:
            print(f"[train_student_bc] WARNING: only {len(real_train.episodes)} real train episodes "
                  f"-- early stopping will be noisy; >=10 total recommended", flush=True)
        from torch.utils.data import ConcatDataset
        train_ds_final = ConcatDataset([train_ds] + [real_train] * max(1, args.real_oversample))
        val_ds_final = real_val  # early-stop on REAL fit -- that's the fine-tune objective
        print(f"[train_student_bc] FINETUNE MIX: sim_train={len(train_ds)} + "
              f"real_train={len(real_train)}x{args.real_oversample}, val=REAL({len(real_val)})", flush=True)
    else:
        train_ds_final = train_ds
        val_ds_final = val_ds

    print(f"[train_student_bc] train={len(train_ds_final)} transitions, val={len(val_ds_final)} transitions "
          f"img_aug={args.img_aug} ablate={args.ablate} freeze={args.freeze_backbone} "
          f"modality_dropout={args.modality_dropout} "
          f"dropout_modalities={args.dropout_modalities}", flush=True)

    stats_path = os.path.join(args.out_dir, "norm_stats.npz")
    if os.path.exists(stats_path):
        stats = dict(np.load(stats_path))
    elif args.init_from:
        # Fine-tune MUST use the stats the warm-started weights were trained
        # with (recomputing could silently shift the z-scoring). Inherit the
        # norm_stats.npz sitting next to the init_from checkpoint.
        src_stats = os.path.join(os.path.dirname(os.path.expanduser(args.init_from)), "norm_stats.npz")
        if not os.path.isfile(src_stats):
            raise SystemExit(f"[train_student_bc] ABORT: --init_from given but no norm_stats.npz found "
                             f"next to it ({src_stats}) and none in out_dir. Copy the pretrain run's "
                             f"norm_stats.npz into out_dir first.")
        stats = dict(np.load(src_stats))
        np.savez(stats_path, **stats)
        print(f"[train_student_bc] inherited norm_stats from {src_stats}", flush=True)
    else:
        print("[train_student_bc] computing normalization stats...", flush=True)
        stats = compute_norm_stats(train_ds)
        np.savez(stats_path, **stats)
    norm = Normalizer(stats, device)

    train_loader = DataLoader(train_ds_final, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds_final, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    policy = StudentFMTPolicy(StudentFMTConfig()).to(device)

    if args.init_from:
        ckpt = torch.load(os.path.expanduser(args.init_from), map_location=device)
        policy.load_state_dict(ckpt["model"])
        print(f"[train_student_bc] warm-started weights from {args.init_from} (fresh optimizer)", flush=True)

    frozen_modules = []
    if args.freeze_backbone == "early":
        for bb in (policy.tp_backbone, policy.wrist_backbone):
            for m in (bb.stem, bb.layer1, bb.layer2, bb.layer3):
                for prm in m.parameters():
                    prm.requires_grad_(False)
                frozen_modules.append(m)
        n_train = sum(p_.numel() for p_ in policy.parameters() if p_.requires_grad)
        n_total = sum(p_.numel() for p_ in policy.parameters())
        print(f"[train_student_bc] freeze_backbone=early: trainable {n_train:,}/{n_total:,} params", flush=True)

    def apply_freeze_eval_mode():
        # frozen BN layers must stay in eval mode or their running stats
        # drift toward the fine-tune distribution despite frozen weights
        for m in frozen_modules:
            m.eval()

    opt = torch.optim.AdamW((p_ for p_ in policy.parameters() if p_.requires_grad),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    loss_fn = nn.HuberLoss(delta=1.0)

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        resume_path = os.path.expanduser(args.resume)
        if not os.path.isfile(resume_path):
            raise SystemExit(f"[train_student_bc] ABORT: --resume file not found: {resume_path} "
                             f"(silently starting fresh would overwrite this run's checkpoints)")
        if args.init_from:
            print("[train_student_bc] WARNING: both --resume and --init_from given; "
                  "--resume wins (init_from weights discarded)", flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        if ckpt.get("freeze_backbone", "none") != args.freeze_backbone:
            print(f"[train_student_bc] WARNING: checkpoint freeze_backbone="
                  f"{ckpt.get('freeze_backbone')} != current {args.freeze_backbone}; optimizer "
                  f"load will likely fail -- rerun with the matching flag", flush=True)
        policy.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        best_val = ckpt.get("best_val", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        print(f"[train_student_bc] resumed from {resume_path} at epoch {start_epoch} "
              f"(best_val so far {best_val:.5f})", flush=True)

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        policy.train()
        apply_freeze_eval_mode()
        t0 = time.time()
        running_loss = 0.0
        for batch in train_loader:
            state = norm.state(batch["state"].to(device, non_blocking=True))
            wrench = norm.wrench(batch["wrench"].to(device, non_blocking=True))
            tp_rgb = batch["tp_rgb"].to(device, non_blocking=True)
            wrist_rgb = batch["wrist_rgb"].to(device, non_blocking=True)
            target_action = batch["action"].to(device, non_blocking=True)
            _ab = ABLATE_SETS[args.ablate]
            if "force" in _ab:
                wrench = torch.zeros_like(wrench)
            if "vision" in _ab:
                tp_rgb = torch.zeros_like(tp_rgb)
                wrist_rgb = torch.zeros_like(wrist_rgb)
            if "state" in _ab:
                state = torch.zeros_like(state)

            if args.modality_dropout > 0.0:
                # At most one modality per sample, so two always remain to fall back on.
                # Only the modalities named in --dropout_modalities are eligible; with a
                # single name the choice is degenerate and that one is dropped every time
                # the sample is hit.
                b = state.shape[0]
                hit = torch.rand(b, device=state.device) < args.modality_dropout
                _mods = [m.strip() for m in args.dropout_modalities.split(",") if m.strip()]
                which = torch.randint(0, len(_mods), (b,), device=state.device)
                _ix = {m: i for i, m in enumerate(_mods)}

                def _m(flag, ref):
                    return flag.view(-1, *([1] * (ref.dim() - 1)))

                def _sel(name):
                    return hit & (which == _ix[name]) if name in _ix else torch.zeros_like(hit)

                d_state = _sel("state")
                d_vis = _sel("vision")
                d_force = _sel("force")
                state = torch.where(_m(d_state, state), torch.zeros_like(state), state)
                wrench = torch.where(_m(d_force, wrench), torch.zeros_like(wrench), wrench)
                tp_rgb = torch.where(_m(d_vis, tp_rgb), torch.zeros_like(tp_rgb), tp_rgb)
                wrist_rgb = torch.where(_m(d_vis, wrist_rgb), torch.zeros_like(wrist_rgb), wrist_rgb)

            pred_action = policy(state, wrench, tp_rgb, wrist_rgb)
            loss = loss_fn(pred_action, target_action)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            running_loss += loss.item()
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1

        train_loss = running_loss / len(train_loader)

        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                state = norm.state(batch["state"].to(device))
                wrench = norm.wrench(batch["wrench"].to(device))
                tp_rgb = batch["tp_rgb"].to(device)
                wrist_rgb = batch["wrist_rgb"].to(device)
                target_action = batch["action"].to(device)
                _ab = ABLATE_SETS[args.ablate]
                if "force" in _ab:
                    wrench = torch.zeros_like(wrench)
                if "vision" in _ab:
                    tp_rgb = torch.zeros_like(tp_rgb)
                    wrist_rgb = torch.zeros_like(wrist_rgb)
                if "state" in _ab:
                    state = torch.zeros_like(state)
                pred_action = policy(state, wrench, tp_rgb, wrist_rgb)
                val_loss += loss_fn(pred_action, target_action).item()
        val_loss /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
              f"lr={sched.get_last_lr()[0]:.2e} time={dt:.1f}s", flush=True)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        ckpt = {"model": policy.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                "config": StudentFMTConfig().__dict__,
                "sched": sched.state_dict(), "best_val": best_val,
                "freeze_backbone": args.freeze_backbone}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

    writer.close()
    print("TRAIN_BC_DONE", flush=True)


if __name__ == "__main__":
    main()
