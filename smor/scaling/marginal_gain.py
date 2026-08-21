"""Per-source marginal acquisition gain curves (spec §29).

``G_i(N) = -dL/dN_i`` = expected loss reduction from one additional unique sample of source ``i``;
with cost, ``G_i^cost = G_i / c_i``. Provided as sweepable curves for the marginal-utility plots
(milestone 8).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def marginal_gain_curve(law, source: int, n_grid: Sequence[int],
                        other_counts: Sequence[int], cost: float = 1.0) -> Dict:
    """Sweep ``N_source`` over ``n_grid`` (other sources fixed) -> marginal gain per (unit) sample."""
    K = law.K
    other = np.asarray(other_counts, dtype=np.float64)
    if other.shape[0] != K:
        raise ValueError("other_counts must have length K.")
    gains, gains_cost = [], []
    for n in n_grid:
        counts = other.copy(); counts[source] = float(n)
        g = float(np.atleast_1d(law.marginal_gain(counts))[source])
        gains.append(g); gains_cost.append(g / cost)
    return {"source": int(source), "N": np.asarray(n_grid, dtype=np.float64),
            "gain": np.asarray(gains), "gain_per_cost": np.asarray(gains_cost)}
