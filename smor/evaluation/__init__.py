"""Evaluation: real env rollouts, correlation metrics, long-horizon utility."""

from smor.evaluation.rollout import rollout_policy
from smor.evaluation.metrics import (
    sign_accuracy,
    spearman,
    pearson,
    cosine_similarity,
)
from smor.evaluation.long_horizon_utility import (
    estimate_group_hypergradients,
    realized_long_horizon_deltas,
)

__all__ = [
    "rollout_policy",
    "sign_accuracy",
    "spearman",
    "pearson",
    "cosine_similarity",
    "estimate_group_hypergradients",
    "realized_long_horizon_deltas",
]
