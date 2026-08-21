"""Model selection + held-out extrapolation scoring (spec §21, §22).

A law is judged NOT by train R^2 alone but by AIC/BIC and — most importantly — its held-out scale
extrapolation error: fit on small budgets, predict large held-out budgets (§22).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _pcol(df: pd.DataFrame) -> str:
    return "p_source_0" if "p_source_0" in df.columns else "mixture"


def extrapolation_error(law, df_heldout: pd.DataFrame, target_col: str = "val_loss") -> Dict:
    """MAE / RMSE / mean relative error of ``law`` on held-out rows (spec §22)."""
    pcol = _pcol(df_heldout)
    y = df_heldout[target_col].to_numpy(dtype=np.float64)
    pred = np.array([law.predict(int(b), float(p))
                     for b, p in zip(df_heldout["budget"], df_heldout[pcol])], dtype=np.float64)
    err = pred - y
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "rel_error": float(np.mean(np.abs(err) / (np.abs(y) + 1e-9))),
            "n": int(y.shape[0])}


def evaluate_model(law, train_df: pd.DataFrame, heldout_df: pd.DataFrame | None = None,
                   target_col: str = "val_loss") -> Dict:
    """Full report: train RMSE/AIC/BIC (+ held-out extrapolation if provided)."""
    fr = law.fit_result
    report = {"model": law.name, "train_rmse": float(fr.rmse),
              "aic": float(fr.aic()), "bic": float(fr.bic()),
              "n_params": int(fr.n_params), "status": fr.status,
              "params": {nm: float(v) for nm, v in zip(law.param_names, law.params)}}
    if heldout_df is not None and len(heldout_df):
        report["heldout"] = extrapolation_error(law, heldout_df, target_col)
    return report


def select_model(reports: List[Dict], criterion: str = "heldout_rmse") -> Dict:
    """Pick the best report. ``criterion`` in {heldout_rmse, heldout_mae, aic, bic, train_rmse}."""
    def key(r):
        if criterion.startswith("heldout"):
            metric = criterion.split("_", 1)[1]
            return r.get("heldout", {}).get(metric, np.inf)
        return r.get(criterion, np.inf)
    return min(reports, key=key)
