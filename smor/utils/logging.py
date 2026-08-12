"""Structured JSONL logging + run metadata (PLAN.md §13)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


@dataclass
class RunMetadata:
    """Run-level metadata logged once at the start of every run (PLAN.md §13)."""

    seed: int
    config: dict
    env: str
    n: int
    K: int
    damping: float
    neumann_lr: float
    beta_lr: float
    reweight_interval: int
    n_demos: int = 0
    fidelity_counts: dict = field(default_factory=dict)
    device: str = "cpu"
    git_commit: Optional[str] = field(default_factory=_git_commit)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])

    def to_dict(self) -> dict:
        return asdict(self)


class JsonlLogger:
    """Append-only JSONL logger. One record per line; flushed each write."""

    def __init__(self, path: str | Path, echo: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        self.echo = echo

    def log(self, record_type: str, **fields: Any) -> None:
        record = {"type": record_type, **_jsonify(fields)}
        line = json.dumps(record)
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.echo:
            print(line)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion of tensors/arrays/paths to JSON-serializable values."""
    # Local imports to keep logging importable without heavy deps at module load.
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    # torch tensors: import lazily.
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return obj
