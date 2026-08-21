"""Additive spline GAM over ``(log B, p)`` (spec §14), implemented with sklearn (no pygam dep).

Model: ``L = f1(log B) + f2(p) + f12(log B, p) + eps``. ``f1, f2`` are B-spline bases; the
interaction ``f12`` is the tensor product of the two bases. ``interaction_strength()`` reports how
much of the fitted signal lives in the interaction block — the empirical test of whether the
optimal mixture shifts with scale (§14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pcol(df: pd.DataFrame) -> str:
    return "p_source_0" if "p_source_0" in df.columns else "mixture"


class GAMScalingModel:
    def __init__(self, target_col: str = "val_loss", n_splines: int = 5, alpha: float = 1e-2):
        self.target_col = target_col
        self.n_splines = n_splines
        self.alpha = alpha

    def _features(self, logB, p, fit: bool):
        from sklearn.preprocessing import SplineTransformer
        if fit:
            self._sb = SplineTransformer(n_knots=self.n_splines, degree=3, include_bias=False)
            self._sp = SplineTransformer(n_knots=self.n_splines, degree=3, include_bias=False)
            Fb = self._sb.fit_transform(logB.reshape(-1, 1))
            Fp = self._sp.fit_transform(p.reshape(-1, 1))
        else:
            Fb = self._sb.transform(logB.reshape(-1, 1))
            Fp = self._sp.transform(p.reshape(-1, 1))
        # interaction = tensor product of the two bases
        Finter = (Fb[:, :, None] * Fp[:, None, :]).reshape(Fb.shape[0], -1)
        self._nb, self._np, self._ni = Fb.shape[1], Fp.shape[1], Finter.shape[1]
        return np.concatenate([Fb, Fp, Finter], axis=1)

    def fit(self, df: pd.DataFrame) -> "GAMScalingModel":
        from sklearn.linear_model import Ridge
        pcol = _pcol(df)
        logB = np.log(df["budget"].to_numpy(float))
        p = df[pcol].to_numpy(float)
        y = df[self.target_col].to_numpy(float)
        self._ymean = float(y.mean())
        F = self._features(logB, p, fit=True)
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=True).fit(F, y - self._ymean)
        return self

    def predict(self, budget, mixture):
        B = np.atleast_1d(np.asarray(budget, float))
        p = np.atleast_1d(np.asarray(mixture, float))
        if p.ndim > 1:
            p = p[:, 0]
        p = np.broadcast_to(p, B.shape)
        F = self._features(np.log(B), p, fit=False)
        out = self.ridge.predict(F) + self._ymean
        return float(out[0]) if B.shape[0] == 1 else out

    def interaction_strength(self) -> float:
        """Fraction of coefficient L2 energy in the interaction block (0 = separable in B,p)."""
        c = self.ridge.coef_
        inter = c[self._nb + self._np:]
        total = np.linalg.norm(c) + 1e-12
        return float(np.linalg.norm(inter) / total)
