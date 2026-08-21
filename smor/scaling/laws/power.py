"""Power-law and log-trend (B, p) scaling laws (spec §16 Model 1 & 4, §17)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from smor.scaling.laws.base import BudgetMixtureLaw


def _quad(coef3: np.ndarray, p: np.ndarray) -> np.ndarray:
    return coef3[0] + coef3[1] * p + coef3[2] * p ** 2


class PowerScalingLaw(BudgetMixtureLaw):
    r"""``L(B,p) = Linf(p) + A(p) B^{-alpha}`` with ``Linf, A`` quadratic in ``p`` (§16, §17)."""

    name = "power"
    param_names = ["Linf0", "Linf1", "Linf2", "A0", "A1", "A2", "alpha"]

    def _model(self, params, B, p):
        Linf = _quad(params[0:3], p)
        A = _quad(params[3:6], p)
        alpha = params[6]
        return Linf + A * np.power(B, -alpha)

    def _p0_and_bounds(self, B, p, y):
        y0 = float(np.min(y))
        yr = float(np.ptp(y)) + 1e-3
        p0_list = [
            [y0, 0, 0, yr, 0, 0, 0.5],
            [y0, 0, 0, yr, 0, 0, 0.3],
            [y0 * 0.5, 0, 0, yr * 2, 0, 0, 0.8],
        ]
        lo = [-1e3, -1e3, -1e3, 0.0, -1e3, -1e3, 1e-3]
        hi = [1e3, 1e3, 1e3, 1e6, 1e3, 1e3, 3.0]
        return p0_list, (lo, hi)


class LogScalingLaw(BudgetMixtureLaw):
    r"""Log-trend baseline ``L(B,p) = a(p) - b(p) log B`` (§16 Model 4)."""

    name = "log"
    param_names = ["a0", "a1", "a2", "b0", "b1", "b2"]

    def _model(self, params, B, p):
        a = _quad(params[0:3], p)
        b = _quad(params[3:6], p)
        return a - b * np.log(B)

    def _p0_and_bounds(self, B, p, y):
        p0_list = [[float(np.max(y)), 0, 0, 0.1, 0, 0],
                   [float(np.mean(y)), 0, 0, 0.01, 0, 0]]
        lo = [-1e3] * 6
        hi = [1e3] * 6
        return p0_list, (lo, hi)
