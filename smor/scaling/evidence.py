"""Module 2 output object (spec §48).

``ScalingEvidence`` bundles everything Module 2 produces: the fitted surface/law, extrapolation
metrics, the predicted optimal mixture per budget, bootstrap intervals, marginal-gain curves, and
oracle regret — the analogue of Module 1's ``ReweightingEvidence`` and the input to the future
acquisition-inference operator (§49).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ScalingEvidence:
    fitted_law: Any = None
    law_name: str = ""
    law_parameters: Dict[str, float] = field(default_factory=dict)
    extrapolation_metrics: Dict[str, Any] = field(default_factory=dict)
    optimal_mixture_by_budget: Dict[int, Any] = field(default_factory=dict)
    bootstrap_intervals: Dict[str, Any] = field(default_factory=dict)
    marginal_gain_curves: Dict[str, Any] = field(default_factory=dict)
    oracle_regret: Dict[int, float] = field(default_factory=dict)
    trend_interaction_strength: float = float("nan")

    def summary(self) -> Dict[str, Any]:
        return {
            "law": self.law_name, "params": self.law_parameters,
            "extrapolation": self.extrapolation_metrics,
            "optimal_mixture_by_budget": {int(k): (v.tolist() if hasattr(v, "tolist") else v)
                                          for k, v in self.optimal_mixture_by_budget.items()},
            "oracle_regret": self.oracle_regret,
            "trend_interaction_strength": self.trend_interaction_strength,
        }
