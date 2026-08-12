"""WeightedLearner contract (PLAN.md §11).

The reweighting module depends ONLY on this interface, never on AIRL/BC internals.
A ``batch_by_group`` is a mapping ``{group_id: batch}`` where ``batch`` is whatever the
concrete learner consumes (for BC: a ``(states, actions)`` tuple of tensors).
Group weights are passed as ``{group_id: weight}`` mappings, decoupling the learner from
the beta parameterization and group ordering used by :class:`OnlineReweighter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping

import torch


Batch = Any
BatchByGroup = Mapping[int, Batch]
GroupWeights = Mapping[int, float]


class WeightedLearner(ABC):
    """Abstract learner driven by group-weighted data.

    Contract summary:
        per_group_losses      -> differentiable L_j(theta) for each active group
        weighted_inner_loss   -> sum_j w_j L_j(theta), differentiable in theta
        parameters_for_reweighting -> the theta tensors used for grads/HVPs
        train_step            -> a real SGD update of theta under given weights
        evaluate              -> task metrics (e.g. episodic return, success, val loss)
    """

    device: torch.device

    # ---- differentiable losses -----------------------------------------
    @abstractmethod
    def per_group_losses(self, batch_by_group: BatchByGroup) -> Dict[int, torch.Tensor]:
        """Return ``{group_id: scalar loss tensor}``, differentiable w.r.t. theta."""

    def weighted_inner_loss(
        self, weights: GroupWeights, batch_by_group: BatchByGroup
    ) -> torch.Tensor:
        """L_in = sum_j w_j L_j(theta). Default: reuse :meth:`per_group_losses`."""
        losses = self.per_group_losses(batch_by_group)
        total: torch.Tensor | None = None
        for gid, loss in losses.items():
            w = weights[gid]
            term = w * loss if not isinstance(w, torch.Tensor) else w.to(loss.device) * loss
            total = term if total is None else total + term
        if total is None:
            raise ValueError("weighted_inner_loss received no groups.")
        return total

    # ---- parameters -----------------------------------------------------
    @abstractmethod
    def parameters_for_reweighting(self) -> Iterable[torch.nn.Parameter]:
        """Return the theta parameters used for hypergradient / HVP computation."""

    # ---- real optimization ---------------------------------------------
    @abstractmethod
    def train_step(
        self, weights: GroupWeights, batch_by_group: BatchByGroup
    ) -> Dict[str, float]:
        """Perform one real optimizer step of theta under ``weights``; return metrics."""

    # ---- evaluation & checkpointing ------------------------------------
    @abstractmethod
    def evaluate(self, **kwargs: Any) -> Dict[str, float]:
        """Return task metrics (episodic return, success rate, validation loss, ...)."""

    @abstractmethod
    def sample_batches(self, group_ids: Iterable[int]) -> Dict[int, Batch]:
        """Sample a fresh training batch for each requested group id."""

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
