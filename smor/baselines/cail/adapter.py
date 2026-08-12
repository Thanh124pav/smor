"""CAIL / AIRL learner adapter — STUB (PLAN.md §3.1, §11 adapter #1).

The original CAIL baseline (Zhang et al., NeurIPS 2021) is AIRL-based with confidence
reweighting. Reproducing it requires the upstream Stanford-ILIAD environment (MuJoCo +
mujoco_py + legacy gym/PyTorch) and is intentionally **not** run inside Module 1 (see
``README.md`` in this directory). This stub fixes the intended interface so the adapter can be
filled in during the dedicated reproduction pass, wrapping AIRL internals behind the same
:class:`WeightedLearner` contract the reweighting core already drives for BC.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch

from smor.learners.base import WeightedLearner

_NOT_IMPLEMENTED = (
    "CAILAIRLAdapter is a stub. The original-CAIL reproduction (Stage A) is a separate pass "
    "requiring the upstream MuJoCo/mujoco_py environment; see smor/baselines/cail/README.md."
)


class CAILAIRLAdapter(WeightedLearner):
    """Planned adapter around AIRL + CAIL confidence reweighting. Not yet implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def per_group_losses(self, batch_by_group) -> Dict[int, torch.Tensor]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def parameters_for_reweighting(self) -> Iterable[torch.nn.Parameter]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def train_step(self, weights, batch_by_group) -> Dict[str, float]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def evaluate(self, **kwargs) -> Dict[str, float]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def sample_batches(self, group_ids) -> Dict[int, object]:
        raise NotImplementedError(_NOT_IMPLEMENTED)
