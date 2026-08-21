"""Known-optimum recovery + oracle/regret (spec §42, §24-26)."""

import numpy as np
import pandas as pd

from smor.scaling.laws import PowerScalingLaw
from smor.scaling.oracle import oracle_mixture, target_budget_regret


def _grid_df(fn, budgets, ps, seeds=(0,)):
    rows = []
    for B in budgets:
        for p in ps:
            for s in seeds:
                rows.append({"budget": B, "p_source_0": p, "seed": s,
                             "val_loss": float(fn(B, p))})
    return pd.DataFrame(rows)


def test_known_optimum():
    # L(B,p) = (p-0.3)^2 + B^-0.5  -> p* = 0.3 for all B
    df = _grid_df(lambda B, p: (p - 0.3) ** 2 + B ** -0.5,
                  budgets=[50, 100, 200, 400, 800], ps=np.linspace(0, 1, 11))
    law = PowerScalingLaw().fit(df)
    for B in [100, 800, 4000]:
        p_star = law.optimal_mixture(B)[0]
        assert abs(p_star - 0.3) < 0.05, (B, p_star)


def test_oracle_and_regret():
    df = _grid_df(lambda B, p: (p - 0.3) ** 2 + B ** -0.5,
                  budgets=[100, 800], ps=np.round(np.linspace(0, 1, 11), 2))
    orc = oracle_mixture(df, budget=800, metric="val_loss")
    assert abs(orc["p_star"] - 0.3) < 1e-6
    # predicting the true optimum gives ~zero regret; a bad mixture gives positive regret
    r_good = target_budget_regret(df, 800, predicted_p=0.3)
    r_bad = target_budget_regret(df, 800, predicted_p=1.0)
    assert r_good["regret"] <= 1e-9
    assert r_bad["regret"] > r_good["regret"]
