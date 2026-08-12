"""Generate + save expert/noisy point-mass demonstrations (PLAN.md §10)."""

from __future__ import annotations

import argparse
from pathlib import Path

from smor.envs.demos import make_two_fidelity_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Collect two-fidelity point-mass demonstrations.")
    p.add_argument("--out", type=str, default="data/pointmass")
    p.add_argument("--n-expert", type=int, default=40)
    p.add_argument("--n-noisy", type=int, default=40)
    p.add_argument("--noise", type=float, default=0.6)
    p.add_argument("--random-prob", type=float, default=0.3)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    train = make_two_fidelity_dataset(
        n_expert=args.n_expert, n_noisy=args.n_noisy, noise=args.noise,
        random_prob=args.random_prob, horizon=args.horizon, seed=args.seed,
    )
    val = make_two_fidelity_dataset(
        n_expert=args.n_val, n_noisy=0, noise=0.0, random_prob=0.0,
        horizon=args.horizon, seed=args.seed + 4242,
    )
    train.save(out / "train.pt")
    val.save(out / "val.pt")
    print(f"saved train ({train.num_trajectories} traj) + val ({val.num_trajectories} traj) to {out}/")
    print(f"  fidelity counts (train): expert={int((train.fidelity==0).sum())} "
          f"noisy={int((train.fidelity==1).sum())}")


if __name__ == "__main__":
    main()
