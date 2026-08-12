"""Long-horizon utility-prediction experiment (PLAN.md §8).

The central validation of the K>1 motivation: does the estimated hypergradient ``h_j^{(K)}``
predict the *realized* effect of upweighting group ``j`` and training for a much longer
horizon? We measure both and compare (sign accuracy, Spearman, Pearson, cosine).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from smor.reweighting.grouping import GroupAssignment
from smor.reweighting.hypergradient import group_hypergradient
from smor.reweighting.outer_objective import OuterObjective, ValidationLoss


def _uniform_weights(M: int) -> Dict[int, float]:
    return {j: 1.0 / M for j in range(M)}


def _perturb(weights: Dict[int, float], j: int, epsilon: float) -> Dict[int, float]:
    w = dict(weights)
    w[j] = w[j] + epsilon
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


@torch.no_grad()
def _outer_value(learner, outer_objective: OuterObjective) -> float:
    return float(outer_objective.loss(learner).detach())


def estimate_group_hypergradients(
    learner,
    group_assignment: GroupAssignment,
    K: int,
    neumann_lr: float,
    damping: float,
    outer_objective: Optional[OuterObjective] = None,
    n_batches: int = 8,
    weights: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Return the expected ``h_j^{(K)}`` per group, averaged over ``n_batches`` resamples."""
    outer_objective = outer_objective or ValidationLoss()
    M = group_assignment.num_groups
    gids = list(range(M))
    weights = weights or _uniform_weights(M)
    acc = np.zeros(M)
    for _ in range(n_batches):
        batches = learner.sample_batches(gids)
        group_losses = learner.per_group_losses(batches)
        outer_loss = outer_objective.loss(learner)
        inner_loss = None
        if K > 1:
            for gid in gids:
                term = weights[gid] * group_losses[gid]
                inner_loss = term if inner_loss is None else inner_loss + term
        hg = group_hypergradient(
            group_losses, outer_loss, learner.parameters_for_reweighting(),
            K=K, neumann_lr=neumann_lr, damping=damping, inner_loss=inner_loss,
        )
        acc += np.array([hg[g] for g in gids])
    return acc / n_batches


def realized_long_horizon_deltas(
    learner,
    group_assignment: GroupAssignment,
    epsilon: float,
    oracle_steps: int,
    outer_objective: Optional[OuterObjective] = None,
    groups: Optional[List[int]] = None,
) -> np.ndarray:
    """Realized change in outer loss from upweighting each group and training ``oracle_steps``.

    Uses a fixed base checkpoint: every condition restarts from the same policy+optimizer
    state so the measured delta isolates the effect of the beta perturbation.
    Returns ``Delta_j^long`` (negative == upweighting group j *helped*).
    """
    outer_objective = outer_objective or ValidationLoss()
    M = group_assignment.num_groups
    gids = list(range(M))
    groups = groups if groups is not None else gids
    base_state = copy.deepcopy(learner.state_dict())
    base_weights = _uniform_weights(M)

    def _run(weights: Dict[int, float]) -> float:
        learner.load_state_dict(copy.deepcopy(base_state))
        for _ in range(oracle_steps):
            learner.train_step(weights, learner.sample_batches(gids))
        return _outer_value(learner, outer_objective)

    L0 = _run(base_weights)
    deltas = np.zeros(M)
    for j in groups:
        Lj = _run(_perturb(base_weights, j, epsilon))
        deltas[j] = Lj - L0

    learner.load_state_dict(base_state)  # leave learner as we found it
    return deltas
