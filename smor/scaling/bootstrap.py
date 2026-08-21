"""Bootstrap confidence intervals for scaling estimates (spec §27).

Resamples seed-level observations, refits the law, and records the quantities of interest
(scaling exponent, asymptotic loss, optimal mixture, target-budget regret) so each comes with a
95% CI rather than a single point estimate.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd


def _resample_seed_level(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample WITH replacement over seeds within each (budget, mixture) cell."""
    pcol = "p_source_0" if "p_source_0" in df.columns else "mixture"
    parts = []
    for _, cell in df.groupby(["budget", pcol]):
        idx = rng.integers(0, len(cell), size=len(cell))
        parts.append(cell.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def bootstrap_law(
    df: pd.DataFrame,
    make_law: Callable[[], object],
    budgets: List[int],
    n_resamples: int = 500,
    seed: int = 0,
) -> Dict:
    """Bootstrap a (B, p) law's optimal mixture (+ params) over ``n_resamples`` (spec §27)."""
    rng = np.random.default_rng(seed)
    p_stars = {int(b): [] for b in budgets}
    params = []
    for _ in range(n_resamples):
        boot = _resample_seed_level(df, rng)
        try:
            law = make_law().fit(boot)
        except Exception:
            continue
        params.append(np.asarray(law.params, dtype=np.float64))
        for b in budgets:
            p_stars[int(b)].append(float(law.optimal_mixture(b)[0]))

    def ci(vals):
        v = np.asarray(vals, dtype=np.float64)
        return {"mean": float(v.mean()), "lo95": float(np.percentile(v, 2.5)),
                "hi95": float(np.percentile(v, 97.5)), "std": float(v.std())}

    out = {"n_effective": len(params),
           "p_star": {int(b): ci(p_stars[int(b)]) for b in budgets if p_stars[int(b)]}}
    if params:
        P = np.stack(params)
        out["params"] = {i: ci(P[:, i]) for i in range(P.shape[1])}
    return out
