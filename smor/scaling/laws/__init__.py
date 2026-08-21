"""Parametric scaling laws (spec §16-19).

Two families:
  * ``(B, p)`` budget-mixture laws (2-source): PowerScalingLaw, ShiftedPowerScalingLaw,
    ExponentialScalingLaw, LogScalingLaw — with ``predict(budget, mixture)`` and
    ``optimal_mixture(budget)``.
  * source-count laws: AdditiveSourceScalingLaw ``L(N_1..N_K)`` — with ``predict(counts)``.
"""

from smor.scaling.laws.additive_source import AdditiveSourceScalingLaw
from smor.scaling.laws.base import BudgetMixtureLaw, ScalingLaw
from smor.scaling.laws.exponential import ExponentialScalingLaw
from smor.scaling.laws.power import LogScalingLaw, PowerScalingLaw
from smor.scaling.laws.shifted_power import ShiftedPowerScalingLaw

__all__ = [
    "ScalingLaw", "BudgetMixtureLaw",
    "PowerScalingLaw", "ShiftedPowerScalingLaw", "ExponentialScalingLaw", "LogScalingLaw",
    "AdditiveSourceScalingLaw",
]
