"""Reproducible seeding and device resolution."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and torch RNGs. Returns the seed for convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Best-effort determinism; may reduce throughput.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device string. ``None``/``"auto"`` -> cuda if available else cpu."""
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
