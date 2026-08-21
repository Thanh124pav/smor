"""Full-pipeline integration on a synthetic source-count law (spec §43).

Must pass before running the expensive real-robot sweep: generate observations from a known
additive law, fit, extrapolate, predict optimal allocation, estimate marginal gain, bootstrap.
"""

import numpy as np
import pandas as pd

from smor.scaling.laws import AdditiveSourceScalingLaw
from smor.scaling.marginal_gain import marginal_gain_curve
from smor.scaling.optimize_mixture import kkt_allocation


def _true(N1, N2):
    return 1.0 + 2.0 * (N1 + 1) ** -0.5 + 3.0 * (N2 + 1) ** -0.3


def _make_counts_df(grid):
    rows = []
    for n1 in grid:
        for n2 in grid:
            rows.append({"n_source_0": n1, "n_source_1": n2, "val_loss": _true(n1, n2)})
    return pd.DataFrame(rows)


def test_additive_law_recovers_params():
    df = _make_counts_df([1, 5, 10, 25, 50, 100, 200, 400])
    law = AdditiveSourceScalingLaw(n_sources=2).fit(df)
    P = dict(zip(law.param_names, law.params))
    assert abs(P["Linf"] - 1.0) < 0.1
    assert abs(P["a0"] - 2.0) < 0.3 and abs(P["a1"] - 3.0) < 0.4
    assert abs(P["alpha0"] - 0.5) < 0.06 and abs(P["alpha1"] - 0.3) < 0.06


def test_additive_law_extrapolates():
    fit_df = _make_counts_df([1, 5, 10, 25, 50, 100])
    law = AdditiveSourceScalingLaw(n_sources=2).fit(fit_df)
    for n1, n2 in [(800, 800), (1600, 50), (50, 1600)]:
        pred = float(law.predict([n1, n2]))
        assert abs(pred - _true(n1, n2)) < 0.05


def test_marginal_gain_positive_and_decreasing():
    df = _make_counts_df([1, 5, 10, 25, 50, 100, 200])
    law = AdditiveSourceScalingLaw(n_sources=2).fit(df)
    curve = marginal_gain_curve(law, source=0, n_grid=[1, 10, 100, 1000], other_counts=[50, 50])
    g = curve["gain"]
    assert (g > 0).all()                 # more data always helps
    assert np.all(np.diff(g) < 0)        # diminishing returns


def test_kkt_allocation_runs_and_conserves_budget():
    df = _make_counts_df([1, 5, 10, 25, 50, 100, 200])
    law = AdditiveSourceScalingLaw(n_sources=2).fit(df)
    alloc = kkt_allocation(law, budget=300, costs=[1.0, 1.0], seed=0)
    assert abs(alloc["counts"].sum() - 300) < 1.0
    # source 1 has the flatter exponent (0.3) -> gets meaningful share (interior allocation)
    assert alloc["counts"].min() > 1.0
