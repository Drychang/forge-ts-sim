"""External early-stopping wrapper around train_student_bc.py.

train_student_bc.py itself has no early stopping -- it always runs the full
--epochs range and just tracks best.pt by lowest val_loss seen so far. Since
best.pt is always safe to use at any point, we can watch its stdout for
"epoch N/M ... val_loss=X" lines and kill the child once val_loss hasn't
improved for `--patience` epochs, instead of burning hours past the trough.

usage:
  python run_seed_earlystop.py --task peg --seed 0 \
      --data_root /media/data/forge_ts_data \
      --out_dir ~/forge_ts/student_ckpts/peg/gate2_seed0 --gpu 1 \
      --patience 6 --max_epochs 30
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["peg", "gear", "nut"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--img_aug", type=str, default="none")
    ap.add_argument("--ablate", type=str, default="none")
    # P1 architecture ablation (backward compatible: defaults reproduce the
    # original behaviour exactly -- train_student_bc.py with no extra args).
    ap.add_argument("--trainer", type=str, default=None,
                    help="path to an alternative trainer script (default: train_student_bc.py)")
    ap.add_argument("--extra_args", type=str, default="",
                    help="extra CLI args appended verbatim to the trainer command")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    trainer_path = args.trainer or os.path.join(script_dir, "train_student_bc.py")
    cmd = [
        sys.executable, "-u", trainer_path,
        "--data_root", args.data_root, "--task", args.task,
        "--out_dir", out_dir, "--epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size), "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--img_aug", args.img_aug,
        "--ablate", args.ablate,
    ]
    if args.extra_args.strip():
        cmd += args.extra_args.split()

    log_path = os.path.join(out_dir, "train.log")
    logf = open(log_path, "a", buffering=1)
    logf.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} cmd={' '.join(cmd)} ===\n")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=script_dir, env=env,
        preexec_fn=os.setsid,
    )

    best_val = float("inf")
    best_epoch = -1
    pat = re.compile(r"epoch (\d+)/\d+ train_loss=([\d.eE+-]+) val_loss=([\d.eE+-]+)")

    def stop_child(reason):
        logf.write(f"[run_seed_earlystop] {reason}\n")
        print(f"[run_seed_earlystop] {reason}", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(10):
            if proc.poll() is not None:
                return
            time.sleep(1)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        for line in proc.stdout:
            logf.write(line)
            print(line, end="", flush=True)
            m = pat.search(line)
            if m:
                cur_epoch = int(m.group(1))
                vloss = float(m.group(3))
                if vloss < best_val:
                    best_val = vloss
                    best_epoch = cur_epoch
                if cur_epoch - best_epoch >= args.patience:
                    stop_child(
                        f"EARLY_STOP epoch={cur_epoch} best_epoch={best_epoch} "
                        f"best_val={best_val:.5f} patience={args.patience}"
                    )
                    logf.write("RUN_SEED_EARLYSTOP_DONE reason=early_stop\n")
                    print("RUN_SEED_EARLYSTOP_DONE reason=early_stop", flush=True)
                    return
            if "TRAIN_BC_DONE" in line:
                logf.write("RUN_SEED_EARLYSTOP_DONE reason=natural_finish\n")
                print("RUN_SEED_EARLYSTOP_DONE reason=natural_finish", flush=True)
                return
    except KeyboardInterrupt:
        stop_child("interrupted")
        raise

    rc = proc.wait()
    logf.write(f"RUN_SEED_EARLYSTOP_DONE reason=process_ended rc={rc}\n")
    print(f"RUN_SEED_EARLYSTOP_DONE reason=process_ended rc={rc}", flush=True)


if __name__ == "__main__":
    main()
