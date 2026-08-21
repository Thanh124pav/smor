"""Exponential-saturation (B, p) scaling law (spec §16 Model 3)."""

from __future__ import annotations

import numpy as np

from smor.scaling.laws.base import BudgetMixtureLaw
from smor.scaling.laws.power import _quad


class ExponentialScalingLaw(BudgetMixtureLaw):
    r"""``L(B,p) = Linf(p) + A(p) e^{-k B}`` (§16 Model 3).

    Saturating form: loss approaches ``Linf(p)`` exponentially. Distinguishable from a power law
    mainly by its held-out extrapolation, which is why several candidates are always compared.
    """

    name = "exponential"
    param_names = ["Linf0", "Linf1", "Linf2", "A0", "A1", "A2", "k"]

    def _model(self, params, B, p):
        Linf = _quad(params[0:3], p)
        A = _quad(params[3:6], p)
        k = params[6]
        return Linf + A * np.exp(-k * B)

    def _p0_and_bounds(self, B, p, y):
        y0 = float(np.min(y))
        yr = float(np.ptp(y)) + 1e-3
        bscale = float(np.median(B)) + 1.0
        p0_list = [
            [y0, 0, 0, yr, 0, 0, 1.0 / bscale],
            [y0, 0, 0, yr * 2, 0, 0, 3.0 / bscale],
        ]
        lo = [-1e3, -1e3, -1e3, 0.0, -1e3, -1e3, 1e-6]
        hi = [1e3, 1e3, 1e3, 1e6, 1e3, 1e3, 10.0]
        return p0_list, (lo, hi)
