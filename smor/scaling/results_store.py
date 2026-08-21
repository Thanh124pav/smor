"""Per-seed results table persistence (spec §11).

CSV with one row per (task, learner, budget, mixture, seed) — never pre-aggregated, so downstream
statistics (bootstrap, mixed-effects) keep the seed-level variation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smor.scaling.records import ScalingDataset, ScalingObservation


def append_observation(path: str | Path, obs: ScalingObservation) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([obs.to_row()])
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


def save_results(path: str | Path, ds: ScalingDataset) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_dataframe().to_csv(path, index=False)
    return path


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
