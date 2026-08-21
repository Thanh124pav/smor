"""Grid oracle optimal mixture (spec §24).

At a (held-out) budget, evaluate the seed-averaged loss/success over the observed mixture grid and
take the argmin/argmax. The oracle is only approximate (finite grid), and is the reference against
which a fitted law's predicted ``p*`` is scored via target-budget regret (§26).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _pcol(df: pd.DataFrame) -> str:
    return "p_source_0" if "p_source_0" in df.columns else "mixture"


def oracle_mixture(df: pd.DataFrame, budget: int, metric: str = "val_loss",
                   maximize: bool | None = None) -> Dict:
    """Return the grid-optimal mixture at ``budget`` (min loss / max success), seed-averaged."""
    if maximize is None:
        maximize = metric != "val_loss"
    pcol = _pcol(df)
    sub = df[df["budget"] == budget]
    if sub.empty:
        raise ValueError(f"no rows at budget={budget}.")
    agg = sub.groupby(pcol)[metric].mean()
    p_star = float(agg.idxmax() if maximize else agg.idxmin())
    val = float(agg.max() if maximize else agg.min())
    return {"budget": int(budget), "p_star": p_star,
            "mixture": np.array([p_star, 1.0 - p_star]), metric: val,
            "grid": {float(k): float(v) for k, v in agg.items()}}


def target_budget_regret(df: pd.DataFrame, budget: int, predicted_p: float,
                         metric: str = "val_loss") -> Dict:
    """Regret of a predicted mixture vs the grid oracle at ``budget`` (spec §26).

    Uses the seed-averaged metric at the grid point nearest ``predicted_p`` (the sweep is on a
    grid, so the predicted continuous ``p`` is snapped to the closest evaluated mixture).
    """
    maximize = metric != "val_loss"
    pcol = _pcol(df)
    sub = df[df["budget"] == budget]
    agg = sub.groupby(pcol)[metric].mean()
    grid_p = np.array(list(agg.index), dtype=np.float64)
    nearest = float(grid_p[np.argmin(np.abs(grid_p - predicted_p))])
    val_pred = float(agg.loc[nearest])
    val_oracle = float(agg.max() if maximize else agg.min())
    regret = (val_oracle - val_pred) if maximize else (val_pred - val_oracle)
    return {"budget": int(budget), "predicted_p": float(predicted_p),
            "nearest_grid_p": nearest, "value_predicted": val_pred,
            "value_oracle": val_oracle, "regret": float(regret)}
