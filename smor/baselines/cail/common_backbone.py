"""Common-backbone CAIL-style confidence baseline (PLAN.md §3.2).

This is the apples-to-apples CAIL baseline used in the solver comparison: the SAME BC backbone,
grouping, optimizer and update schedule as SMOR, differing only in (a) the hypergradient depth
(K=1, the one-step / CAIL-style confidence update) and (b) optionally CAIL's confidence-ranking
outer objective (:class:`smor.reweighting.outer_objective.CAILRankingLoss`).

It is NOT the original AIRL-CAIL (Zhang et al., NeurIPS 2021), which adds an adversarial reward
and an RL policy loop — that reproduction is Stage A (see this directory's README.md). Reporting:
always label this "CAIL-style (common backbone)", never plain "CAIL".
"""

from __future__ import annotations

from dataclasses import replace

from smor.reweighting.config import OnlineReweighterConfig
from smor.reweighting.grouping import GroupAssignment


def cail_style_config(cfg: OnlineReweighterConfig) -> OnlineReweighterConfig:
    """Return a copy of ``cfg`` forced to the CAIL-style one-step solver (K=1)."""
    return replace(cfg, K=1)


def group_quality_from_sources(ga: GroupAssignment, sources: list[dict]) -> dict:
    """Per-group quality label for CAIL's ranking loss: higher = cleaner source.

    Quality is derived from each source's nominal corruption (lower noise + gain nearer 1.0 =
    higher quality). ``ga.group_fidelity[g]`` indexes into ``sources``.
    """
    quality = {}
    for g in range(ga.num_groups):
        s = sources[int(ga.group_fidelity[g])]
        noise = float(s.get("noise", 0.0))
        gain_err = abs(float(s.get("gain", 1.0)) - 1.0)
        blunder = float(s.get("random_prob", 0.0))
        quality[g] = -(noise + gain_err + blunder)
    return quality
