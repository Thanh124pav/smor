"""SMOR Module 2 — Scaling Laws for Data Ratio.

Studies the mapping ``(B, p) -> J`` (acquisition budget x source mixture -> downstream
performance / loss) and predicts the budget-dependent optimal mixture ``p*_B``. Core principle
(see ``SMOR_MODULE2_SCALING_LAWS_IMPLEMENTATION.md``):

    Discover first, parameterize second, extrapolate third.

Layout:
    records / results_store  — ScalingObservation, ScalingDataset, CSV persistence
    sampler                  — budget x mixture -> unique-trajectory subset (no replacement)
    laws/                    — parametric scaling laws (power / shifted / exponential / additive)
    trend/                   — flexible smoothers (GP, spline-GAM) for discovery
    fitting / model_selection — robust nonlinear fits + AIC/BIC/held-out extrapolation
    oracle / optimize_mixture — grid oracle p* and predicted p* from a fitted law
    bootstrap / marginal_gain — CIs and per-source marginal utility G_i = -dL/dN_i
"""

from smor.scaling.records import ScalingDataset, ScalingObservation
from smor.scaling.sampler import SampledDataset, sample_dataset

__all__ = ["ScalingObservation", "ScalingDataset", "SampledDataset", "sample_dataset"]
