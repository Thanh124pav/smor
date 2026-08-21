"""Budget x mixture x seed sweep -> per-seed results table (spec §36, milestone 2).

For each (budget B, mixture p, seed): draw a UNIQUE-trajectory subset from the PH/MG pools
(``sample_dataset``), train the FIXED BC learner on it, and record the clean-target validation
loss (+ optional robosuite closed-loop success). Writes one row per run to results/scaling_runs.csv.

    python -m experiments.scaling.run_grid --config configs/scaling/two_source_mvp.yaml
    python -m experiments.scaling.run_grid --budgets 20 40 80 --mixtures 0 0.5 1 --seeds 0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.scaling.generate_pools import SourcePools, build_pools
from smor.data.trajectory_dataset import TrajectoryDataset
from smor.learners.bc import BCLearner
from smor.reweighting.grouping import make_groups
from smor.scaling.records import ScalingObservation
from smor.scaling.results_store import append_observation
from smor.scaling.sampler import sample_dataset
from smor.utils.seeding import resolve_device, seed_everything


def _norm(trajs, mu, sd):
    return [((o - mu) / sd, a) for o, a in trajs]


def _build_train(pools: SourcePools, sampled) -> TrajectoryDataset:
    mu, sd = pools.obs_mu, pools.obs_sd
    ph_ids = sampled.source_ids["ph"]; mg_ids = sampled.source_ids["mg"]
    trajs = ([pools.ph_pool[i] for i in ph_ids] + [pools.mg_pool[i] for i in mg_ids])
    trajs = _norm(trajs, mu, sd)
    # single group (uniform BC, no reweighting); source identity lives in the results row
    return TrajectoryDataset.from_trajectories(trajs, [0] * len(trajs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/scaling/two_source_mvp.yaml")
    p.add_argument("--task", default="lift")
    p.add_argument("--pools-cache", default="data/scaling/pools_lift.pt")
    p.add_argument("--budgets", type=int, nargs="+", default=None)
    p.add_argument("--mixtures", type=float, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--n-val-ph", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/scaling/scaling_runs.csv")
    p.add_argument("--rollout", action="store_true", help="also eval robosuite success (slow)")
    p.add_argument("--eval-episodes", type=int, default=25)
    args = p.parse_args()

    from smor.scaling.config import ScalingConfig
    cfg = ScalingConfig.from_yaml(args.config) if Path(args.config).exists() else ScalingConfig()
    budgets = args.budgets or cfg.budgets_all
    mixtures = args.mixtures if args.mixtures is not None else cfg.mixtures
    seeds = args.seeds or cfg.seeds
    epochs = args.epochs or cfg.epochs
    device = str(resolve_device(args.device))

    # pools (cache to disk so repeated sweeps skip HDF5 reads)
    cache = Path(args.pools_cache)
    if cache.exists():
        pools = SourcePools.load(cache)
    else:
        pools = build_pools(task=args.task, n_val_ph=args.n_val_ph, seed=0)
        pools.save(cache)
    print(f"pools: PH={len(pools.ph_pool)} MG={len(pools.mg_pool)} val={len(pools.val_trajs)} "
          f"obs={pools.obs_dim} act={pools.act_dim} device={device}")

    # fixed clean-target validation set (normalized once)
    val_ds = TrajectoryDataset.from_trajectories(
        _norm(pools.val_trajs, pools.obs_mu, pools.obs_sd), [0] * len(pools.val_trajs))

    env = None
    if args.rollout:
        from smor.envs.robosuite_env import RobosuiteVecEnv
        env = RobosuiteVecEnv(task=args.task, dtype="ph", horizon=300, device=device)

    out = Path(args.out)
    if out.exists():
        out.unlink()  # fresh sweep
    n_runs = len(budgets) * len(mixtures) * len(seeds)
    done = 0
    t0 = time.time()
    for B in budgets:
        for pmix in mixtures:
            for seed in seeds:
                seed_everything(seed)
                sampled = sample_dataset({"ph": len(pools.ph_pool), "mg": len(pools.mg_pool)},
                                         budget=B, mixture=[pmix, 1 - pmix], seed=seed)
                train_ds = _build_train(pools, sampled)
                ga = make_groups(np.zeros(train_ds.num_trajectories, dtype=np.int64),
                                 group_size=train_ds.num_trajectories, whole_fidelity=True)
                lr = BCLearner(train_ds, ga, hidden=tuple(args.hidden), lr=1e-3,
                               batch_size=args.batch_size, device=device, val_data=val_ds,
                               env=env, seed=seed, data_device="cpu")
                steps = epochs * max(1, math.ceil(train_ds.num_transitions / args.batch_size))
                ts = time.time()
                for _ in range(steps):
                    lr.train_step({0: 1.0}, lr.sample_batches([0]))
                metrics = lr.evaluate(n_episodes=args.eval_episodes if env is not None else 0)
                obs = ScalingObservation(
                    budget=int(B), mixture=sampled.mixture, source_counts=sampled.source_counts,
                    seed=int(seed), val_loss=float(metrics["val_loss"]),
                    success_rate=float(metrics.get("success_rate", float("nan"))),
                    task=args.task, learner="bc", train_steps=int(steps),
                    wall_time=float(time.time() - ts))
                append_observation(out, obs)
                done += 1
                print(f"[{done}/{n_runs}] B={B} p={pmix:.1f} seed={seed} "
                      f"N=({sampled.source_counts[0]},{sampled.source_counts[1]}) "
                      f"val_loss={obs.val_loss:.4f}"
                      + (f" succ={obs.success_rate:.2f}" if env is not None else "")
                      + f" ({obs.wall_time:.0f}s)")
    if env is not None:
        env.close()
    print(f"\nsaved {out}  ({done} runs, {time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
