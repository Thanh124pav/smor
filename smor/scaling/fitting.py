"""Robust nonlinear least-squares fitting for scaling laws (spec §20).

Wraps :func:`scipy.optimize.least_squares` with parameter bounds, multiple initializations, and a
fit-status/diagnostics record, so a single bad local optimum does not decide the law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass
class FitResult:
    params: np.ndarray
    rmse: float
    cost: float
    n_params: int
    n_points: int
    status: str
    param_names: Sequence[str] = field(default_factory=list)

    def aic(self) -> float:
        n, k = self.n_points, self.n_params
        rss = max(self.cost * 2.0, 1e-300)
        return n * np.log(rss / n) + 2 * k

    def bic(self) -> float:
        n, k = self.n_points, self.n_params
        rss = max(self.cost * 2.0, 1e-300)
        return n * np.log(rss / n) + k * np.log(n)

    def as_dict(self) -> dict:
        return {"params": {nm: float(v) for nm, v in zip(self.param_names, self.params)}
                if self.param_names else self.params.tolist(),
                "rmse": float(self.rmse), "aic": float(self.aic()), "bic": float(self.bic()),
                "status": self.status}


def robust_fit(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    p0_list: Sequence[Sequence[float]],
    bounds: Tuple[Sequence[float], Sequence[float]],
    param_names: Sequence[str] | None = None,
    max_nfev: int = 10000,
) -> FitResult:
    """Fit ``model_fn(params, X) ~= y`` from several starts; keep the lowest-cost solution.

    ``model_fn`` maps ``(params, X) -> predictions`` of shape ``(len(y),)``.
    """
    y = np.asarray(y, dtype=np.float64)
    best = None
    for p0 in p0_list:
        try:
            res = least_squares(
                lambda p: model_fn(np.asarray(p), X) - y,
                x0=np.asarray(p0, dtype=np.float64), bounds=bounds, max_nfev=max_nfev,
            )
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("all fit initializations failed.")
    resid = model_fn(best.x, X) - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return FitResult(
        params=np.asarray(best.x), rmse=rmse, cost=float(best.cost),
        n_params=len(best.x), n_points=int(y.shape[0]),
        status="ok" if best.success else "maxfev",
        param_names=list(param_names) if param_names else [],
    )
