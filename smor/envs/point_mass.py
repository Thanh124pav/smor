"""A small 2D point-mass goal-reaching environment (batched, GPU-friendly).

Dynamics are pure tensor ops so ``B`` episodes run in parallel on the target device:

    obs      = concat(pos, goal)            in R^4
    action   = velocity command in [-1, 1]^2
    pos_{t+1}= clip(pos_t + max_step * action, -world, world)
    reward   = -||pos - goal||              (dense)
    success  = ||pos - goal|| < goal_radius

This is deliberately tiny so a full reweighting run finishes quickly on CPU or GPU. It also
provides two demonstration fidelities (expert vs noisy) matching CAIL's varying-optimality
setting via :func:`expert_action` plus configurable action noise (see ``envs/demos.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PointMassConfig:
    horizon: int = 40
    max_step: float = 0.1
    world: float = 1.0
    goal_radius: float = 0.15
    min_start_goal_dist: float = 0.5


class PointMassEnv:
    obs_dim: int = 4
    act_dim: int = 2

    def __init__(self, config: PointMassConfig | None = None, device="cpu", seed: int = 0):
        self.cfg = config or PointMassConfig()
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._pos: torch.Tensor | None = None
        self._goal: torch.Tensor | None = None
        self._t = 0

    # ---- core API -------------------------------------------------------
    def reset(self, batch_size: int) -> torch.Tensor:
        """Sample ``batch_size`` start/goal pairs; return observation ``(B, 4)``."""
        w = self.cfg.world
        # sample on CPU generator for reproducibility, then move to device
        pos = (torch.rand(batch_size, 2, generator=self.generator) * 2 - 1) * w
        goal = (torch.rand(batch_size, 2, generator=self.generator) * 2 - 1) * w
        # enforce a minimum start-goal distance so the task is non-trivial
        for _ in range(10):
            close = (pos - goal).norm(dim=-1) < self.cfg.min_start_goal_dist
            if not close.any():
                break
            new_goal = (torch.rand(batch_size, 2, generator=self.generator) * 2 - 1) * w
            goal = torch.where(close.unsqueeze(-1), new_goal, goal)
        self._pos = pos.to(self.device)
        self._goal = goal.to(self.device)
        self._t = 0
        return self._obs()

    def step(self, action: torch.Tensor):
        """Advance all episodes one step. Returns ``(obs, reward, done, info)``."""
        assert self._pos is not None, "call reset() first"
        action = torch.clamp(action.to(self.device), -1.0, 1.0)
        self._pos = torch.clamp(
            self._pos + self.cfg.max_step * action, -self.cfg.world, self.cfg.world
        )
        self._t += 1
        dist = (self._pos - self._goal).norm(dim=-1)
        reward = -dist
        done = self._t >= self.cfg.horizon
        info = {
            "distance": dist,
            "success": (dist < self.cfg.goal_radius),
        }
        done_mask = torch.full((self._pos.shape[0],), bool(done), device=self.device)
        return self._obs(), reward, done_mask, info

    def _obs(self) -> torch.Tensor:
        return torch.cat([self._pos, self._goal], dim=-1)

    @property
    def horizon(self) -> int:
        return self.cfg.horizon


def expert_action(obs: torch.Tensor, max_step: float = 0.1) -> torch.Tensor:
    """Near-optimal action: move straight toward the goal at full feasible speed.

    ``obs = [pos_x, pos_y, goal_x, goal_y]``. Returns velocity commands in ``[-1, 1]^2``.
    """
    pos = obs[..., :2]
    goal = obs[..., 2:]
    to_goal = goal - pos
    # full speed until within one step of the goal, then proportional slow-down
    action = torch.clamp(to_goal / max_step, -1.0, 1.0)
    return action
