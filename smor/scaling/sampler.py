"""Budget x mixture -> unique-trajectory subset sampler (spec §3, §8).

Acquisition scaling counts **unique trajectories**: given a total budget ``B`` and a mixture
``p`` over ``K`` source pools, allocate ``N_i = floor(B p_i / c_i)`` (equal cost ``c_i=1`` in the
MVP) with ``sum_i N_i = B``, then sample ``N_i`` trajectories *without replacement* from pool ``i``.
Training may reuse trajectories across epochs, but ``N_i`` is always the unique-data count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np


@dataclass
class SampledDataset:
    source_ids: Dict[str, np.ndarray]   # selected trajectory ids per source (no replacement)
    source_counts: np.ndarray           # (K,) N_i, sum = budget
    mixture: np.ndarray                 # (K,) proportions actually used
    budget: int
    seed: int
    source_order: Sequence[str]

    @property
    def total(self) -> int:
        return int(self.source_counts.sum())


def allocate_counts(budget: int, mixture: Sequence[float],
                    costs: Sequence[float] | None = None) -> np.ndarray:
    """Integer per-source counts summing exactly to ``budget`` (largest-remainder rounding).

    With costs, allocates ``N_i = floor(B p_i / c_i)``; the residual (from flooring) is handed to
    the sources with the largest fractional parts so ``sum N_i`` is conserved.
    """
    p = np.asarray(mixture, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("mixture must be a non-empty 1-D vector.")
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError(f"mixture must sum to 1, got {p.sum()}")
    if (p < -1e-9).any():
        raise ValueError("mixture entries must be non-negative.")
    if costs is None:
        # equal cost: conserve sum_i N_i = budget via largest-remainder rounding
        raw = budget * p
        base = np.floor(raw).astype(np.int64)
        remainder = int(round(budget - base.sum()))
        frac = raw - base
        order = np.argsort(-frac)
        for k in range(max(0, remainder)):
            base[order[k % len(order)]] += 1
        return base
    # unequal cost: N_i = floor(B p_i / c_i); only sum_i c_i N_i <= budget is guaranteed (§3)
    c = np.asarray(costs, dtype=np.float64)
    return np.floor(budget * p / c).astype(np.int64)


def sample_dataset(
    source_pools: Mapping[str, Sequence],
    budget: int,
    mixture: Sequence[float],
    seed: int = 0,
    costs: Sequence[float] | None = None,
) -> SampledDataset:
    """Sample a unique-trajectory subset for ``(budget, mixture, seed)``.

    ``source_pools`` maps ``source_id -> pool`` where ``pool`` is a sequence of trajectory ids (or
    an int pool size, interpreted as ``range(size)``). Returns a :class:`SampledDataset`.
    """
    order = list(source_pools.keys())
    counts = allocate_counts(budget, mixture, costs)
    if counts.shape[0] != len(order):
        raise ValueError("mixture length must match number of source pools.")
    rng = np.random.default_rng(seed)
    ids: Dict[str, np.ndarray] = {}
    for i, sid in enumerate(order):
        pool = source_pools[sid]
        pool_ids = np.arange(int(pool)) if np.isscalar(pool) else np.asarray(list(pool))
        n = int(counts[i])
        if n > pool_ids.shape[0]:
            raise ValueError(
                f"source '{sid}' needs {n} unique trajectories but pool has {pool_ids.shape[0]}. "
                f"Enlarge the pool or lower the budget."
            )
        ids[sid] = rng.choice(pool_ids, size=n, replace=False) if n > 0 else pool_ids[:0]
    return SampledDataset(source_ids=ids, source_counts=counts,
                          mixture=np.asarray(mixture, dtype=np.float64), budget=int(budget),
                          seed=int(seed), source_order=order)
