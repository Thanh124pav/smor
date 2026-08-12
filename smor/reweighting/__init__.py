"""Online reweighting core (PLAN.md §5, §6, §1.2, Stages B-E).

Lazy attribute access (PEP 562) keeps the flat public API while allowing individual
submodules to be imported without pulling in the whole chain.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "OnlineReweighterConfig": "smor.reweighting.config",
    "GroupAssignment": "smor.reweighting.grouping",
    "make_groups": "smor.reweighting.grouping",
    "BetaDistribution": "smor.reweighting.beta",
    "ReweightScheduler": "smor.reweighting.scheduler",
    "hvp": "smor.reweighting.hvp",
    "apply_pk": "smor.reweighting.neumann",
    "group_hypergradient": "smor.reweighting.hypergradient",
    "one_step_hypergradient": "smor.reweighting.hypergradient",
    "OnlineReweighter": "smor.reweighting.online_reweighter",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
