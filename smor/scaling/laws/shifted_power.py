"""Shifted power-law (B, p) scaling law (spec §16 Model 2)."""

from __future__ import annotations

import numpy as np

from smor.scaling.laws.base import BudgetMixtureLaw
from smor.scaling.laws.power import _quad


class ShiftedPowerScalingLaw(BudgetMixtureLaw):
    r"""``L(B,p) = Linf(p) + A(p) (B + B0)^{-alpha}`` (§16 Model 2).

    The offset ``B0`` captures the finite-data regime where a plain power law bends; often the
    best small-budget fit and the most reliable extrapolator to held-out large budgets.
    """

    name = "shifted_power"
    param_names = ["Linf0", "Linf1", "Linf2", "A0", "A1", "A2", "alpha", "B0"]

    def _model(self, params, B, p):
        Linf = _quad(params[0:3], p)
        A = _quad(params[3:6], p)
        alpha, B0 = params[6], params[7]
        return Linf + A * np.power(B + B0, -alpha)

    def _p0_and_bounds(self, B, p, y):
        y0 = float(np.min(y))
        yr = float(np.ptp(y)) + 1e-3
        p0_list = [
            [y0, 0, 0, yr, 0, 0, 0.5, 10.0],
            [y0, 0, 0, yr * 3, 0, 0, 0.4, 50.0],
            [y0 * 0.5, 0, 0, yr, 0, 0, 0.8, 1.0],
        ]
        lo = [-1e3, -1e3, -1e3, 0.0, -1e3, -1e3, 1e-3, 0.0]
        hi = [1e3, 1e3, 1e3, 1e7, 1e3, 1e3, 3.0, 1e4]
        return p0_list, (lo, hi)
