"""Additive per-source-count scaling law (spec §18, §29, §43).

``L(N_1..N_K) = Linf + sum_i a_i (N_i + N0_i)^{-alpha_i}`` — acquisition acts directly on source
counts. Exposes the per-source marginal gain ``G_i = -dL/dN_i = a_i alpha_i (N_i+N0_i)^{-(alpha_i+1)}``
(spec §29), the bridge to acquisition optimization / KKT allocation (§30).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from smor.scaling.fitting import FitResult, robust_fit
from smor.scaling.laws.base import ScalingLaw


class AdditiveSourceScalingLaw(ScalingLaw):
    name = "additive_source"

    def __init__(self, n_sources: int = 2):
        self.K = int(n_sources)
        self.fit_result: FitResult | None = None
        self.params: np.ndarray | None = None

    @property
    def param_names(self):
        return (["Linf"] + [f"a{i}" for i in range(self.K)]
                + [f"alpha{i}" for i in range(self.K)] + [f"N0_{i}" for i in range(self.K)])

    def _unpack(self, params):
        K = self.K
        Linf = params[0]
        a = params[1:1 + K]
        alpha = params[1 + K:1 + 2 * K]
        N0 = params[1 + 2 * K:1 + 3 * K]
        return Linf, a, alpha, N0

    def _model(self, params, counts):
        Linf, a, alpha, N0 = self._unpack(params)
        counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))  # (M, K)
        terms = a[None, :] * np.power(counts + N0[None, :], -alpha[None, :])
        return Linf + terms.sum(axis=1)

    # ---- fit ------------------------------------------------------------
    def fit(self, counts, y=None) -> "AdditiveSourceScalingLaw":
        """Fit to ``(counts (M,K), y (M,))``; also accepts a dataframe with ``n_source_*`` cols."""
        if isinstance(counts, pd.DataFrame):
            df = counts
            n_cols = sorted([c for c in df.columns if c.startswith("n_source_")])
            self.K = len(n_cols)
            counts = df[n_cols].to_numpy(dtype=np.float64)
            y = df["val_loss"].to_numpy(dtype=np.float64)
        counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        K = self.K
        y0 = float(np.min(y)); yr = float(np.ptp(y)) + 1e-3
        p0_list = [
            [y0] + [yr] * K + [0.5] * K + [1.0] * K,
            [y0] + [yr * 2] * K + [0.3] * K + [1.0] * K,
            [y0 * 0.5] + [yr] * K + [0.8] * K + [10.0] * K,
        ]
        lo = [-1e4] + [0.0] * K + [1e-3] * K + [1e-6] * K
        hi = [1e4] + [1e7] * K + [3.0] * K + [1e4] * K
        self.fit_result = robust_fit(lambda p, X: self._model(p, X), counts, y,
                                     p0_list, (lo, hi), param_names=self.param_names)
        self.params = self.fit_result.params
        return self

    # ---- predict / gain -------------------------------------------------
    def predict(self, counts) -> np.ndarray | float:
        if self.params is None:
            raise RuntimeError("call fit() first.")
        out = self._model(self.params, counts)
        return float(out[0]) if out.shape[0] == 1 else out

    def marginal_gain(self, counts) -> np.ndarray:
        """``G_i = -dL/dN_i = a_i alpha_i (N_i + N0_i)^{-(alpha_i+1)}`` (spec §29)."""
        if self.params is None:
            raise RuntimeError("call fit() first.")
        _, a, alpha, N0 = self._unpack(self.params)
        counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
        g = a[None, :] * alpha[None, :] * np.power(counts + N0[None, :], -(alpha[None, :] + 1.0))
        return g[0] if g.shape[0] == 1 else g
