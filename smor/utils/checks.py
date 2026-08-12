"""Safety / stability checks (PLAN.md §14). Fail loudly, never silently."""

from __future__ import annotations

from typing import Iterable

import torch


class SafetyError(RuntimeError):
    """Raised when a numerical / structural invariant is violated."""


def check_finite(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Raise if ``x`` contains NaN or Inf."""
    if not torch.isfinite(x).all():
        n_nan = int(torch.isnan(x).sum())
        n_inf = int(torch.isinf(x).sum())
        raise SafetyError(
            f"{name} contains non-finite values (nan={n_nan}, inf={n_inf})."
        )
    return x


def check_simplex(beta: torch.Tensor, name: str = "beta", atol: float = 1e-4) -> torch.Tensor:
    """Raise if ``beta`` is not a valid probability simplex vector."""
    check_finite(beta, name)
    if (beta < -atol).any():
        raise SafetyError(f"{name} has negative entries (min={float(beta.min()):.3e}).")
    total = float(beta.sum())
    if abs(total - 1.0) > atol:
        raise SafetyError(f"{name} does not sum to 1 (sum={total:.6f}, atol={atol}).")
    return beta


def check_positive_int(value: int, name: str) -> int:
    """Raise if ``value`` is not an integer >= 1 (used for n and K)."""
    if not isinstance(value, (int,)) or isinstance(value, bool):
        raise SafetyError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 1:
        raise SafetyError(f"{name} must be >= 1, got {value}.")
    return value


def check_no_explosion(values: Iterable[float], name: str, limit: float = 1e8) -> None:
    """Raise if any magnitude exceeds ``limit`` (detects diverging Neumann iterates)."""
    for i, v in enumerate(values):
        if not (abs(v) < limit):
            raise SafetyError(
                f"{name} iterate {i} exploded (|value|={abs(v):.3e} >= {limit:.1e})."
            )
