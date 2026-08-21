"""Bootstrap CI for the optimal mixture (spec §27, §42)."""

import numpy as np
import pandas as pd

from smor.scaling.bootstrap import bootstrap_law
from smor.scaling.laws import PowerScalingLaw


def _noisy_grid(fn, budgets, ps, seeds, noise, rng):
    rows = []
    for B in budgets:
        for p in ps:
            for s in seeds:
                rows.append({"budget": B, "p_source_0": p, "seed": s,
                             "val_loss": float(fn(B, p) + noise * rng.standard_normal())})
    return pd.DataFrame(rows)


def test_bootstrap_ci_contains_truth():
    rng = np.random.default_rng(0)
    # optimum at p=0.3, independent of B
    df = _noisy_grid(lambda B, p: (p - 0.3) ** 2 + B ** -0.5,
                     [50, 100, 200, 400], np.linspace(0, 1, 11), seeds=(0, 1, 2),
                     noise=0.002, rng=rng)
    out = bootstrap_law(df, PowerScalingLaw, budgets=[400], n_resamples=80, seed=1)
    ci = out["p_star"][400]
    assert ci["lo95"] <= 0.3 <= ci["hi95"]
    assert out["n_effective"] > 40
