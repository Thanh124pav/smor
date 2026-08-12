"""Reweight schedule (PLAN.md §6).

Train theta for ``reweight_interval`` (R) policy steps under the current beta, then estimate
the outer signal and update beta, repeated ``n_beta_updates`` times, after an optional warmup
of policy-only steps.
"""

from __future__ import annotations

from typing import Iterator, Tuple


class ReweightScheduler:
    """Yields ``("train", i)`` and ``("reweight", u)`` events for the outer loop."""

    def __init__(self, reweight_interval: int, n_beta_updates: int, warmup_steps: int = 0):
        if reweight_interval < 1:
            raise ValueError("reweight_interval must be >= 1")
        if n_beta_updates < 0:
            raise ValueError("n_beta_updates must be >= 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.reweight_interval = int(reweight_interval)
        self.n_beta_updates = int(n_beta_updates)
        self.warmup_steps = int(warmup_steps)

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        step = 0
        for _ in range(self.warmup_steps):
            yield ("train", step)
            step += 1
        for u in range(self.n_beta_updates):
            for _ in range(self.reweight_interval):
                yield ("train", step)
                step += 1
            yield ("reweight", u)

    @property
    def total_policy_steps(self) -> int:
        return self.warmup_steps + self.n_beta_updates * self.reweight_interval

    @property
    def total_beta_updates(self) -> int:
        return self.n_beta_updates
