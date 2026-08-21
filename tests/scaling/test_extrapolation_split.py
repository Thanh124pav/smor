"""Held-out scale extrapolation + no-leakage (spec §22, §42)."""

import numpy as np
import pandas as pd

from smor.scaling.laws import PowerScalingLaw, ShiftedPowerScalingLaw
from smor.scaling.model_selection import evaluate_model, select_model
from smor.scaling.records import ScalingDataset, ScalingObservation


def _obs_grid(fn, budgets, ps, seeds=(0, 1, 2)):
    ds = ScalingDataset()
    for B in budgets:
        for p in ps:
            for s in seeds:
                ds.add(ScalingObservation(budget=B, mixture=[p, 1 - p],
                                          source_counts=[int(round(B * p)), B - int(round(B * p))],
                                          seed=s, val_loss=float(fn(B, p))))
    return ds


def test_filter_budgets_no_leakage():
    ds = _obs_grid(lambda B, p: 1 + 3 * B ** -0.5, [50, 100, 200, 400, 800, 1600], [0.0, 0.5, 1.0])
    fit = ds.filter_budgets([50, 100, 200, 400]).to_dataframe()
    held = ds.filter_budgets([800, 1600]).to_dataframe()
    assert set(fit["budget"].unique()) == {50, 100, 200, 400}
    assert set(held["budget"].unique()) == {800, 1600}
    assert not set(fit["budget"]).intersection(held["budget"])  # no leakage


def test_extrapolation_accuracy():
    fn = lambda B, p: (1.0 + 0.3 * p) + 3.0 * B ** -0.5
    ds = _obs_grid(fn, [50, 100, 200, 400, 800, 1600], [0.0, 0.5, 1.0])
    fit_df = ds.filter_budgets([50, 100, 200, 400]).to_dataframe()
    held_df = ds.filter_budgets([800, 1600]).to_dataframe()
    reports = []
    for make in (PowerScalingLaw, ShiftedPowerScalingLaw):
        law = make().fit(fit_df)
        reports.append(evaluate_model(law, fit_df, held_df))
    best = select_model(reports, criterion="heldout_rmse")
    # clean power-law data -> held-out extrapolation should be accurate
    assert best["heldout"]["rmse"] < 0.05
