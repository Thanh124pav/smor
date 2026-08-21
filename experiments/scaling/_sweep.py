"""Source-agnostic budget x mixture x seed sweep (shared by the RoboMimic and point-mass runners).

Given generic source pools (``{name: list[(obs, act)]}``), a fixed validation set, and an optional
closed-loop eval env, run the sweep and append one :class:`ScalingObservation` per run to a CSV.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from smor.data.trajectory_dataset import TrajectoryDataset
from smor.learners.bc import BCLearner
from smor.reweighting.grouping import make_groups
from smor.scaling.records import ScalingObservation
from smor.scaling.results_store import append_observation
from smor.scaling.sampler import sample_dataset
from smor.utils.seeding import seed_everything

Traj = Tuple[np.ndarray, np.ndarray]


def run_sweep(
    source_pools: Dict[str, List[Traj]],
    val_ds: TrajectoryDataset,
    budgets: Sequence[int],
    mixtures: Sequence[float],
    seeds: Sequence[int],
    out: str | Path,
    task: str = "",
    env=None,
    epochs: int = 40,
    batch_size: int = 256,
    hidden: Sequence[int] = (256, 256),
    lr: float = 1e-3,
    device: str = "cpu",
    eval_episodes: int = 25,
    fresh: bool = True,
) -> Path:
    """Run the full sweep; source 0 is the first key in ``source_pools`` (mixture = p of source 0)."""
    names = list(source_pools.keys())
    if len(names) != 2:
        raise ValueError("this MVP sweep expects exactly 2 sources.")
    out = Path(out)
    if fresh and out.exists():
        out.unlink()
    n_runs = len(budgets) * len(mixtures) * len(seeds)
    done, t0 = 0, time.time()
    for B in budgets:
        for pmix in mixtures:
            for seed in seeds:
                seed_everything(seed)
                sampled = sample_dataset({n: len(source_pools[n]) for n in names},
                                         budget=B, mixture=[pmix, 1 - pmix], seed=seed)
                trajs = ([source_pools[names[0]][i] for i in sampled.source_ids[names[0]]]
                         + [source_pools[names[1]][i] for i in sampled.source_ids[names[1]]])
                train_ds = TrajectoryDataset.from_trajectories(trajs, [0] * len(trajs))
                ga = make_groups(np.zeros(train_ds.num_trajectories, dtype=np.int64),
                                 group_size=train_ds.num_trajectories, whole_fidelity=True)
                learner = BCLearner(train_ds, ga, hidden=tuple(hidden), lr=lr,
                                    batch_size=batch_size, device=device, val_data=val_ds,
                                    env=env, seed=seed, data_device="cpu")
                steps = epochs * max(1, math.ceil(train_ds.num_transitions / batch_size))
                ts = time.time()
                for _ in range(steps):
                    learner.train_step({0: 1.0}, learner.sample_batches([0]))
                metrics = learner.evaluate(n_episodes=eval_episodes if env is not None else 0)
                obs = ScalingObservation(
                    budget=int(B), mixture=sampled.mixture, source_counts=sampled.source_counts,
                    seed=int(seed), val_loss=float(metrics["val_loss"]),
                    success_rate=float(metrics.get("success_rate", float("nan"))),
                    task=task, learner="bc", train_steps=int(steps),
                    wall_time=float(time.time() - ts))
                append_observation(out, obs)
                done += 1
                print(f"[{done}/{n_runs}] B={B} p={pmix:.2f} seed={seed} "
                      f"N=({sampled.source_counts[0]},{sampled.source_counts[1]}) "
                      f"val={obs.val_loss:.4f}"
                      + (f" succ={obs.success_rate:.2f}" if env is not None else "")
                      + f" ({obs.wall_time:.0f}s)")
    print(f"\nsaved {out}  ({done} runs, {time.time()-t0:.0f}s)")
    return out
