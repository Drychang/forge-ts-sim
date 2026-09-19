# Seeing Through the Displaced Frame — simulation code

Training and evaluation code for the paper *Seeing Through the Displaced Frame: Privileged Noise
Distillation for Vision-Force Precision Assembly* (under review, ICRA 2027).

A privileged teacher observes the fixture-pose offset in simulation, or clean demonstrations are
relabelled into the displaced frame in closed form. Either way the compensation ends up in the
training data, and the deployed student sees only noisy state, a raw wrench window and two RGB
cameras.

> **Under double-anonymous review.** Absolute paths in the run scripts have had the account name
> replaced with `user`, so they are examples rather than working paths. Set them for your machine.

## Layout

```
src/
    student/            the deployed policy: architecture, BC and DAgger training, evaluation
        student_fmt.py          force-modulated transformer, the deployed student
        student_archs.py        the architecture variants used in the ablation
        train_student_bc.py     behaviour cloning
        train_student_arch.py   the same, for the architecture sweep
        collect_dagger_rollouts.py
        dataset.py
        eval_*.py               noise sweep, ablations, tilt, colour shift, latency
    deploy/             the sim-side half of the hardware bridge
        deploy_loop.py, robot_io.py, sim_io.py, shard_writer.py
        record_real_episodes.py
    arch/               architecture experiments
    corma/              CoRMA comparison
    srsa/               SRSA comparison
    collect_ta_rollouts.py      teacher rollout collection (state baseline)
    collect_camera_rollouts.py  teacher rollout collection (privileged, with cameras)
    aggregate_*.py              the tables in the paper
    classify_failures.py        failure-mode breakdown
    run_*.sh, chain_*.sh        the experiment chains as they were run
```

## Requirements

The environments are the three unmodified FORGE tasks. You need:

- Isaac Lab with the Factory/Forge task suite, and a GPU that can run it
- `rl_games` for the teacher policies
- PyTorch, NumPy, and the usual scientific stack

The Isaac Lab install is not vendored here. Point the `FORGE` / `--data_root` style variables at
the top of the run scripts at your own checkout and data directory.

Two environment gotchas that cost time to find, in case they save you the same:

- Forge/Factory with cameras needs `clone_in_fabric=False`, otherwise it crashes on reset.
- Rollout results are not reproducible run to run, so small-N success counts are not a result.

## Reproducing the main table

Roughly, in order:

1. Train or obtain the teachers. `run_p0_ta_teacher.sh` for the state baseline (T-A), the
   privileged oracle (T-B) alongside it.
2. Collect demonstrations. `collect_ta_rollouts.py` for the state route,
   `collect_camera_rollouts.py` for the privileged route with cameras.
3. Train the student. `student/train_student_bc.py`, then one DAgger round with
   `student/collect_dagger_rollouts.py` and a second BC pass.
4. Evaluate across the noise spectrum and aggregate with `aggregate_noise_3seed.py`.

`run_ta_n5_pipeline.sh` and the `chain_*.sh` scripts are the actual chains used, and are the most
honest description of what was run.

## Not included

Checkpoints, collected demonstrations, evaluation dumps and rendered videos. Those run to tens of
gigabytes and live outside the repository.

## Citation

Anonymous while under review. The entry will be updated on acceptance.
