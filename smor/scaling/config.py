"""MVP scaling-experiment config (spec §41)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class SourceSpec:
    id: str
    path: str = ""
    cost: float = 1.0


@dataclass
class ScalingConfig:
    name: str = "two_source_scaling_mvp"
    sources: List[SourceSpec] = field(default_factory=list)
    budgets_all: List[int] = field(default_factory=lambda: [50, 100, 200, 400, 800, 1600])
    fit_budgets: List[int] = field(default_factory=lambda: [50, 100, 200, 400])
    heldout_budgets: List[int] = field(default_factory=lambda: [800, 1600])
    mixtures: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    learner: str = "bc"
    epochs: int = 50
    batch_size: int = 256
    num_rollouts: int = 100
    metric: str = "success_rate"
    scaling_models: List[str] = field(default_factory=lambda: ["power", "shifted_power", "exponential"])
    bootstrap_resamples: int = 500

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScalingConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        exp = raw.get("experiment", {})
        budgets = raw.get("budgets", {})
        mix = raw.get("mixtures", {})
        learner = raw.get("learner", {})
        ev = raw.get("evaluation", {})
        boot = raw.get("bootstrap", {})
        sources = [SourceSpec(id=s["id"], path=s.get("path", ""), cost=float(s.get("cost", 1.0)))
                   for s in raw.get("sources", [])]
        mix_list = mix.get("source_A") or mix.get(next(iter(mix), ""), []) if mix else []
        return cls(
            name=exp.get("name", "two_source_scaling_mvp"), sources=sources,
            budgets_all=budgets.get("all", cls().budgets_all),
            fit_budgets=budgets.get("fit", cls().fit_budgets),
            heldout_budgets=budgets.get("heldout", cls().heldout_budgets),
            mixtures=mix_list or cls().mixtures, seeds=raw.get("seeds", cls().seeds),
            learner=learner.get("name", "bc"), epochs=int(learner.get("epochs", 50)),
            batch_size=int(learner.get("batch_size", 256)),
            num_rollouts=int(ev.get("num_rollouts", 100)),
            metric=ev.get("metric", "success_rate"),
            scaling_models=raw.get("scaling_models", cls().scaling_models),
            bootstrap_resamples=int(boot.get("num_resamples", 500)),
        )
