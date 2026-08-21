"""Predicted optimal mixture / acquisition allocation (spec §25, §30).

* ``predict_optimal_mixture`` — from a fitted (B, p) law, ``p*_B = argmin_p L(B, p)`` (§25).
* ``kkt_allocation`` — from a fitted additive source-count law and a budget, the cost-constrained
  optimal counts via the KKT condition ``a_i alpha_i N_i^{-(alpha_i+1)} / c_i = lambda`` (§30).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.optimize import minimize


def predict_optimal_mixture(law, budget: float) -> np.ndarray:
    """``p*_B`` from a fitted (B, p) budget-mixture law (§25)."""
    return law.optimal_mixture(budget)


def kkt_allocation(law, budget: float, costs: Sequence[float] | None = None,
                   n_restarts: int = 8, seed: int = 0) -> Dict:
    """Optimal integer-relaxed source counts minimizing a fitted additive law s.t. budget (§30).

    Minimizes ``L(N)`` over ``N_i >= 0`` with ``sum_i c_i N_i = B`` (project onto the budget
    simplex via a softmax parameterization; multi-start to avoid poor local optima).
    """
    K = law.K
    c = np.ones(K) if costs is None else np.asarray(costs, dtype=np.float64)
    B = float(budget)
    rng = np.random.default_rng(seed)

    def counts_from_logits(z):
        w = np.exp(z - z.max()); w = w / w.sum()
        return (B * w) / c  # sum_i c_i N_i = B

    def obj(z):
        return float(law.predict(counts_from_logits(z)))

    best = None
    for _ in range(n_restarts):
        z0 = rng.normal(size=K)
        res = minimize(obj, z0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    counts = counts_from_logits(best.x)
    mixture = c * counts / B
    return {"budget": B, "counts": counts, "mixture": mixture,
            "predicted_loss": float(best.fun),
            "marginal_gain_per_cost": law.marginal_gain(counts) / c}
