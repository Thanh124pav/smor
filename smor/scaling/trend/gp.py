"""Gaussian-process scaling surface over ``(log B, p)`` (spec §15).

A GP gives a smooth mean ``mu_L(B, p)`` and uncertainty ``sigma_L(B, p)`` — useful both for
discovery (is the surface smooth?) and, later, uncertainty-aware acquisition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pcol(df: pd.DataFrame) -> str:
    return "p_source_0" if "p_source_0" in df.columns else "mixture"


class GPScalingModel:
    def __init__(self, target_col: str = "val_loss", alpha: float = 1e-4):
        self.target_col = target_col
        self.alpha = alpha
        self.gp = None
        self._mu = 0.0
        self._sd = 1.0

    def fit(self, df: pd.DataFrame) -> "GPScalingModel":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        pcol = _pcol(df)
        X = np.column_stack([np.log(df["budget"].to_numpy(float)), df[pcol].to_numpy(float)])
        y = df[self.target_col].to_numpy(float)
        self._mu, self._sd = float(y.mean()), float(y.std() + 1e-8)
        kernel = ConstantKernel(1.0) * RBF([1.0, 0.3]) + WhiteKernel(1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha,
                                           normalize_y=False, n_restarts_optimizer=3)
        self.gp.fit(X, (y - self._mu) / self._sd)
        return self

    def predict(self, budget, mixture, return_std: bool = True):
        B = np.atleast_1d(np.asarray(budget, float))
        p = np.atleast_1d(np.asarray(mixture, float))
        if p.ndim > 1:
            p = p[:, 0]
        p = np.broadcast_to(p, B.shape)
        X = np.column_stack([np.log(B), p])
        mean, std = self.gp.predict(X, return_std=True)
        mean = mean * self._sd + self._mu
        std = std * self._sd
        if return_std:
            return (float(mean[0]), float(std[0])) if B.shape[0] == 1 else (mean, std)
        return float(mean[0]) if B.shape[0] == 1 else mean
