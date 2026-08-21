"""Point-mass 2-source scaling sweep — a controllable setting with a SCALE-DEPENDENT optimum.

RoboMimic PH-vs-MG turned out to be a dominance setting (MG never helps). Here we construct two
genuinely complementary sources with a **bias-variance trade-off** whose optimal mixture SHIFTS
with budget (the §45 "strongest result"). All trajectories are re-simulated in the point-mass env
(the executed, possibly corrupted, action drives the dynamics — §5 compliant):

  * source A "unbiased_jittery" — unbiased expert but HIGH action noise (low bias, high variance);
    averages to the correct policy only with enough data.
  * source B "biased_precise"   — low-noise but systematically undershooting (gain<1): low variance,
    but a non-zero asymptotic error floor.

At small budgets variance dominates -> prefer the precise source B; at large budgets the noise
averages out -> the unbiased source A wins. So p*_B shifts from B-heavy to A-heavy as B grows.

    python -m experiments.scaling.run_grid_pointmass --out results/scaling/pm_runs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.scaling._sweep import run_sweep
from smor.data.trajectory_dataset import TrajectoryDataset
from smor.envs.demos import collect_demonstrations
from smor.envs.point_mass import PointMassConfig, PointMassEnv
from smor.utils.seeding import resolve_device


def _to_trajs(obs, act):
    return [(obs[i].numpy(), act[i].numpy()) for i in range(obs.shape[0])]


def _gen(n, noise, gain, rot, horizon, seed, device):
    env = PointMassEnv(PointMassConfig(horizon=horizon), device=device, seed=seed)
    o, a = collect_demonstrations(env, n, noise=noise, gain=gain, rotation_deg=rot, seed=seed)
    return _to_trajs(o, a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1600])
    p.add_argument("--mixtures", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--pool-size", type=int, default=1700, help="trajectories per source pool")
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--a-noise", type=float, default=1.2, help="source A action noise (variance)")
    p.add_argument("--a-gain", type=float, default=1.0)
    p.add_argument("--a-rot", type=float, default=0.0)
    p.add_argument("--b-noise", type=float, default=0.02)
    p.add_argument("--b-gain", type=float, default=1.0)
    p.add_argument("--b-rot", type=float, default=30.0, help="source B directional bias (deg)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--no-rollout", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/scaling/pm_runs.csv")
    args = p.parse_args()

    device = str(resolve_device(args.device))
    ps = max(args.pool_size, max(args.budgets))
    pool_a = _gen(ps, args.a_noise, args.a_gain, args.a_rot, args.horizon, seed=1, device="cpu")
    pool_b = _gen(ps, args.b_noise, args.b_gain, args.b_rot, args.horizon, seed=2, device="cpu")
    val_o, val_a = collect_demonstrations(
        PointMassEnv(PointMassConfig(horizon=args.horizon), device="cpu", seed=999),
        args.n_val, noise=0.0, gain=1.0, seed=999)  # clean full-region deployment target
    val_ds = TrajectoryDataset.from_trajectories(_to_trajs(val_o, val_a), [0] * args.n_val)
    print(f"pools: A(unbiased_jittery,noise={args.a_noise})={len(pool_a)}  "
          f"B(biased_precise,rot={args.b_rot})={len(pool_b)}  val={args.n_val}  device={device}")

    env = None
    if not args.no_rollout:
        env = PointMassEnv(PointMassConfig(horizon=args.horizon), device=device, seed=7)

    run_sweep({"A_unbiased_jittery": pool_a, "B_biased_precise": pool_b}, val_ds,
              budgets=args.budgets, mixtures=args.mixtures, seeds=args.seeds,
              out=args.out, task="pointmass", env=env, epochs=args.epochs,
              batch_size=args.batch_size, hidden=args.hidden, device=device,
              eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
