"""Scaling-law base classes (spec §35).

``ScalingLaw`` is the abstract interface (fit / predict / optimal_mixture). ``BudgetMixtureLaw`` is
a concrete base for 2-source ``(B, p)`` laws where the mixture is the single proportion
``p = p_source_0`` and the second source is ``1 - p``. Subclasses only implement the model form
``_model(params, B, p)`` plus initial guesses and bounds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from smor.scaling.fitting import FitResult, robust_fit


class ScalingLaw(ABC):
    name: str = "scaling_law"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "ScalingLaw": ...

    @abstractmethod
    def predict(self, budget, mixture): ...

    def optimal_mixture(self, budget: float) -> np.ndarray:
        raise NotImplementedError


class BudgetMixtureLaw(ScalingLaw):
    """2-source ``(B, p)`` law; ``p`` is the proportion of source 0, source 1 gets ``1 - p``."""

    param_names: List[str] = []
    target_col: str = "val_loss"

    def __init__(self):
        self.fit_result: FitResult | None = None
        self.params: np.ndarray | None = None

    # ---- subclass hooks -------------------------------------------------
    @abstractmethod
    def _model(self, params: np.ndarray, B: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Predicted loss at budgets ``B`` and source-0 proportions ``p``."""

    @abstractmethod
    def _p0_and_bounds(self, B: np.ndarray, p: np.ndarray, y: np.ndarray
                       ) -> Tuple[List[Sequence[float]], Tuple[Sequence[float], Sequence[float]]]:
        ...

    # ---- fit / predict --------------------------------------------------
    def _extract(self, df: pd.DataFrame):
        B = df["budget"].to_numpy(dtype=np.float64)
        pcol = "p_source_0" if "p_source_0" in df.columns else "mixture"
        p = df[pcol].to_numpy(dtype=np.float64)
        y = df[self.target_col].to_numpy(dtype=np.float64)
        return B, p, y

    def fit(self, df: pd.DataFrame) -> "BudgetMixtureLaw":
        B, p, y = self._extract(df)
        p0_list, bounds = self._p0_and_bounds(B, p, y)
        X = np.column_stack([B, p])
        self.fit_result = robust_fit(
            lambda params, XX: self._model(params, XX[:, 0], XX[:, 1]),
            X, y, p0_list, bounds, param_names=self.param_names)
        self.params = self.fit_result.params
        return self

    def predict(self, budget, mixture) -> np.ndarray | float:
        if self.params is None:
            raise RuntimeError("call fit() first.")
        B = np.atleast_1d(np.asarray(budget, dtype=np.float64))
        p = self._mixture_to_p(mixture, B.shape[0])
        out = self._model(self.params, B, p)
        return float(out[0]) if out.shape[0] == 1 else out

    @staticmethod
    def _mixture_to_p(mixture, n: int) -> np.ndarray:
        m = np.asarray(mixture, dtype=np.float64)
        if m.ndim == 0:
            p = np.full(n, float(m))
        elif m.ndim == 1:
            p = np.full(n, float(m[0]))   # source-0 proportion
        else:
            p = m[:, 0].astype(np.float64)
        return p

    def optimal_mixture(self, budget: float) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("call fit() first.")
        B = float(budget)
        res = minimize_scalar(
            lambda p: float(self._model(self.params, np.array([B]), np.array([p]))[0]),
            bounds=(0.0, 1.0), method="bounded")
        p = float(np.clip(res.x, 0.0, 1.0))
        return np.array([p, 1.0 - p])
