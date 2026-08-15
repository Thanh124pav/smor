"""Outer objective interface (PLAN.md §7).

For the first solver-comparison experiment the outer objective is a held-out high-quality
validation loss, so K=1 vs K>1 is isolated to the hypergradient solver. The interface is
modular so ranking / closed-loop-return objectives can be added later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class OuterObjective(ABC):
    """Computes a differentiable outer loss ``L_out(theta)`` from a learner."""

    @abstractmethod
    def loss(self, learner) -> torch.Tensor:
        """Return a differentiable scalar outer loss."""


class ValidationLoss(OuterObjective):
    """Held-out validation (expert) loss — the default open-loop outer objective for BC."""

    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size

    def loss(self, learner) -> torch.Tensor:
        val = learner.validation_loss(batch_size=self.batch_size)
        if val is None:
            raise ValueError("learner has no validation data for ValidationLoss.")
        return val


class ClosedLoopReturn(OuterObjective):
    """Differentiable closed-loop return surrogate (PLAN.md §7).

    Backpropagates through the (differentiable) env dynamics of a policy rollout, so the outer
    signal reflects *task* performance (compounding action errors) rather than open-loop action
    matching. Requires a learner exposing ``closed_loop_loss(n_episodes, horizon)``.
    """

    def __init__(self, n_episodes: int = 128, horizon: int | None = None):
        self.n_episodes = n_episodes
        self.horizon = horizon

    def loss(self, learner) -> torch.Tensor:
        fn = getattr(learner, "closed_loop_loss", None)
        if fn is None:
            raise ValueError("learner does not support ClosedLoopReturn (no closed_loop_loss).")
        return fn(self.n_episodes, self.horizon)
