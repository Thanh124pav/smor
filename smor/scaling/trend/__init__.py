"""Flexible trend discovery (spec §13-15): fit a smoother BEFORE assuming a parametric law.

* :class:`GPScalingModel` — Gaussian process over ``(log B, p)`` with uncertainty (sklearn).
* :class:`GAMScalingModel` — additive spline model ``f1(log B) + f2(p) + f12(log B, p)`` whose
  interaction term measures scale x mixture interaction (the key question of §14).
"""

from smor.scaling.trend.gam import GAMScalingModel
from smor.scaling.trend.gp import GPScalingModel

__all__ = ["GAMScalingModel", "GPScalingModel"]
