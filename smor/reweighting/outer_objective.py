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


class CAILRankingLoss(OuterObjective):
    """Common-backbone CAIL-style confidence objective (PLAN.md §3.2, §7).

    A faithful adaptation of CAIL's key mechanism — learning demonstration *confidence* from a
    quality ranking — to the shared BC backbone (no adversarial AIRL/RL loop; the original
    AIRL-CAIL is Stage A, see smor/baselines/cail/). Given per-group quality labels, the outer
    loss is a pairwise margin-ranking penalty over the policy's per-group BC losses: a
    higher-quality group should be fit at least ``margin`` better than a lower-quality one,

        L_out = mean over (i>j) of  softplus(margin + loss_i - loss_j).

    Upweighting a high-quality group lowers its loss and this penalty, so the one-step
    (K=1) confidence/hypergradient update concentrates beta on higher-quality demonstrations —
    exactly CAIL's confidence behavior.
    """

    def __init__(self, group_quality: dict, margin: float = 0.02):
        self.group_quality = {int(k): float(v) for k, v in group_quality.items()}
        self.margin = float(margin)

    def loss(self, learner) -> torch.Tensor:
        import torch.nn.functional as F

        gids = list(self.group_quality)
        batches = learner.sample_batches(gids)
        losses = learner.per_group_losses(batches)
        terms = []
        for i in gids:
            for j in gids:
                if self.group_quality[i] > self.group_quality[j]:
                    terms.append(F.softplus(self.margin + losses[i] - losses[j]))
        if not terms:
            total = None
            for v in losses.values():
                total = v if total is None else total + v
            return total / max(1, len(losses))
        return torch.stack(terms).mean()


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
