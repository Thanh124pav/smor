"""Synthetic scaling-law recovery (spec §42)."""

import numpy as np
import pandas as pd

from smor.scaling.laws import PowerScalingLaw


def _grid_df(fn, budgets, ps):
    rows = []
    for B in budgets:
        for p in ps:
            rows.append({"budget": B, "p_source_0": p, "val_loss": float(fn(B, p))})
    return pd.DataFrame(rows)


def test_power_law_recovers_alpha():
    # y = 1 + 3 B^-0.4 (p-independent)
    df = _grid_df(lambda B, p: 1.0 + 3.0 * B ** -0.4,
                  budgets=[50, 100, 200, 400, 800, 1600], ps=[0.0, 0.5, 1.0])
    law = PowerScalingLaw().fit(df)
    params = dict(zip(law.param_names, law.params))
    assert abs(params["alpha"] - 0.4) < 0.05
    assert law.fit_result.rmse < 1e-3
    assert abs(law.predict(3200, 0.5) - (1.0 + 3.0 * 3200 ** -0.4)) < 1e-2


def test_power_law_captures_mixture_dependence():
    # y = (1 + 0.5 p) + 3 B^-0.5 : asymptote grows with p
    df = _grid_df(lambda B, p: (1.0 + 0.5 * p) + 3.0 * B ** -0.5,
                  budgets=[50, 100, 200, 400, 800], ps=[0.0, 0.25, 0.5, 0.75, 1.0])
    law = PowerScalingLaw().fit(df)
    # lower mixture (p=0) should be predicted better (lower loss) than p=1 at large budget
    assert law.predict(4000, 0.0) < law.predict(4000, 1.0)
