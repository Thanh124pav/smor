"""Utilities: seeding, device selection, JSONL logging, safety checks."""

from smor.utils.seeding import seed_everything, resolve_device
from smor.utils.logging import JsonlLogger, RunMetadata
from smor.utils.checks import (
    SafetyError,
    check_finite,
    check_simplex,
    check_positive_int,
)

__all__ = [
    "seed_everything",
    "resolve_device",
    "JsonlLogger",
    "RunMetadata",
    "SafetyError",
    "check_finite",
    "check_simplex",
    "check_positive_int",
]
