"""Core data structures for scaling observations (spec §34, §11).

A :class:`ScalingObservation` is one training run at a given ``(budget, mixture, seed)`` with its
measured ``val_loss`` and ``success_rate``. A :class:`ScalingDataset` is a collection of them with
a tidy-dataframe view (one row per seed — never pre-aggregated, per §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class ScalingObservation:
    budget: int
    mixture: np.ndarray          # (K,) acquisition proportions, sum=1
    source_counts: np.ndarray    # (K,) unique-trajectory counts N_i, sum=budget
    seed: int
    val_loss: float
    success_rate: float = float("nan")
    task: str = ""
    learner: str = "bc"
    train_steps: int = 0
    wall_time: float = 0.0

    def __post_init__(self):
        self.mixture = np.asarray(self.mixture, dtype=np.float64)
        self.source_counts = np.asarray(self.source_counts, dtype=np.int64)

    @property
    def num_sources(self) -> int:
        return int(self.mixture.shape[0])

    def to_row(self) -> dict:
        row = {
            "task": self.task, "learner": self.learner, "budget": int(self.budget),
            "seed": int(self.seed), "val_loss": float(self.val_loss),
            "success_rate": float(self.success_rate), "train_steps": int(self.train_steps),
            "wall_time": float(self.wall_time),
        }
        for i in range(self.num_sources):
            row[f"p_source_{i}"] = float(self.mixture[i])
            row[f"n_source_{i}"] = int(self.source_counts[i])
        return row


class ScalingDataset:
    """A collection of :class:`ScalingObservation` with a tidy-dataframe view."""

    def __init__(self, observations: Sequence[ScalingObservation] | None = None):
        self.observations: List[ScalingObservation] = list(observations or [])

    def add(self, obs: ScalingObservation) -> None:
        self.observations.append(obs)

    def __len__(self) -> int:
        return len(self.observations)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([o.to_row() for o in self.observations])

    def filter_budgets(self, budgets: Sequence[int]) -> "ScalingDataset":
        keep = set(int(b) for b in budgets)
        return ScalingDataset([o for o in self.observations if int(o.budget) in keep])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ScalingDataset":
        p_cols = sorted([c for c in df.columns if c.startswith("p_source_")])
        n_cols = sorted([c for c in df.columns if c.startswith("n_source_")])
        obs = []
        for _, r in df.iterrows():
            obs.append(ScalingObservation(
                budget=int(r["budget"]),
                mixture=np.array([r[c] for c in p_cols], dtype=np.float64),
                source_counts=np.array([r[c] for c in n_cols], dtype=np.int64),
                seed=int(r["seed"]), val_loss=float(r["val_loss"]),
                success_rate=float(r.get("success_rate", np.nan)),
                task=str(r.get("task", "")), learner=str(r.get("learner", "bc")),
                train_steps=int(r.get("train_steps", 0)), wall_time=float(r.get("wall_time", 0.0)),
            ))
        return cls(obs)
