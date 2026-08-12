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
    """Held-out validation (expert) loss — the default outer objective for BC."""

    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size

    def loss(self, learner) -> torch.Tensor:
        val = learner.validation_loss(batch_size=self.batch_size)
        if val is None:
            raise ValueError("learner has no validation data for ValidationLoss.")
        return val
