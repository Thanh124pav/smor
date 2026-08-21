"""Build the two RoboMimic source pools for the Module-2 scaling sweep (spec §5, milestone 1).

Sources (Phase 1 decision): PH (proficient-human) vs MG (machine-generated) on lift. Both are real
simulator-collected trajectories (satisfies §5 — no offline action relabeling). A fixed slice of PH
is reserved as the clean deployment/validation target used by every run; the rest of PH and all of
MG form the acquisition pools that the sampler draws unique trajectories from.

    python -m experiments.scaling.generate_pools --task lift --cache data/scaling/pools_lift.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from smor.data.robomimic.loader import DEFAULT_OBS_KEYS, _read_trajectories
from smor.data.robomimic.registry import ensure_dataset

Traj = Tuple[np.ndarray, np.ndarray]


@dataclass
class SourcePools:
    ph_pool: List[Traj]        # PH training pool (real trajectories)
    mg_pool: List[Traj]        # MG training pool
    val_trajs: List[Traj]      # reserved clean PH held-out (deployment/val target)
    obs_mu: np.ndarray         # global obs mean (for consistent normalization)
    obs_sd: np.ndarray         # global obs std
    obs_dim: int
    act_dim: int
    task: str

    def save(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SourcePools":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def build_pools(task: str = "lift", obs_keys=DEFAULT_OBS_KEYS, n_val_ph: int = 20,
                seed: int = 0, root: str | Path | None = None) -> SourcePools:
    ph_path = ensure_dataset(task, "ph", root=root)
    mg_path = ensure_dataset(task, "mg", root=root)
    ph = _read_trajectories(ph_path, obs_keys, tier=None)
    mg = _read_trajectories(mg_path, obs_keys, tier=None)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ph))
    val_idx, pool_idx = order[:n_val_ph], order[n_val_ph:]
    val_trajs = [ph[i] for i in val_idx]
    ph_pool = [ph[i] for i in pool_idx]

    # global normalization stats over ALL pool transitions (fixed across every run)
    all_obs = np.concatenate([o for o, _ in (ph_pool + mg)], axis=0)
    mu = all_obs.mean(0); sd = all_obs.std(0) + 1e-6
    return SourcePools(ph_pool=ph_pool, mg_pool=mg, val_trajs=val_trajs,
                       obs_mu=mu.astype(np.float32), obs_sd=sd.astype(np.float32),
                       obs_dim=ph[0][0].shape[-1], act_dim=ph[0][1].shape[-1], task=task)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="lift")
    p.add_argument("--n-val-ph", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=None)
    p.add_argument("--cache", default="data/scaling/pools_lift.pt")
    args = p.parse_args()
    pools = build_pools(task=args.task, n_val_ph=args.n_val_ph, seed=args.seed, root=args.root)
    path = pools.save(args.cache)
    print(f"PH pool={len(pools.ph_pool)}  MG pool={len(pools.mg_pool)}  val={len(pools.val_trajs)}  "
          f"obs={pools.obs_dim} act={pools.act_dim} -> {path}")


if __name__ == "__main__":
    main()
