"""Behavior-cloning learner adapter (PLAN.md §11, learner #2).

A tiny MLP policy trained by group-weighted behavior cloning. Implements the
:class:`WeightedLearner` contract so the reweighting core can drive it unchanged. All tensors
live on the configured device (GPU-friendly).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from smor.envs.demos import DemoDataset
from smor.learners.base import WeightedLearner
from smor.reweighting.grouping import GroupAssignment
from smor.utils.seeding import resolve_device


class MLPPolicy(nn.Module):
    """obs -> action in [-1, 1]^act_dim (tanh output)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        dims = [obs_dim, *hidden]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers += [nn.Linear(dims[-1], act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


Batch = Tuple[torch.Tensor, torch.Tensor]


class BCLearner(WeightedLearner):
    def __init__(
        self,
        dataset: DemoDataset,
        group_assignment: GroupAssignment,
        hidden: Tuple[int, ...] = (64, 64),
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str | torch.device = "auto",
        val_data: Optional[DemoDataset] = None,
        env=None,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = resolve_device(device)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.env = env
        self._rng = torch.Generator(device="cpu").manual_seed(seed)

        obs_flat, act_flat, traj_id = dataset.flatten()
        self.obs = obs_flat.to(self.device, dtype)
        self.act = act_flat.to(self.device, dtype)
        self._traj_id = traj_id
        self.group_assignment = group_assignment

        # Precompute transition-index pool per group (immutable for the run).
        self._group_pool: Dict[int, torch.Tensor] = {}
        member_set = {j: set(m.tolist()) for j, m in enumerate(group_assignment.members)}
        traj_np = traj_id.numpy()
        for j in range(group_assignment.num_groups):
            mask = np.isin(traj_np, list(member_set[j]))
            idx = torch.from_numpy(np.where(mask)[0])
            if idx.numel() == 0:
                raise ValueError(f"group {j} has no transitions.")
            self._group_pool[j] = idx

        # Held-out validation transitions for the outer objective.
        if val_data is not None:
            vo, va, _ = val_data.flatten()
            self._val_obs = vo.to(self.device, dtype)
            self._val_act = va.to(self.device, dtype)
        else:
            self._val_obs = self._val_act = None

        self.policy = MLPPolicy(dataset.obs_dim, dataset.act_dim, hidden).to(self.device, dtype)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # fixed start/goal batch for the differentiable closed-loop outer objective
        self._cl_pos: Optional[torch.Tensor] = None
        self._cl_goal: Optional[torch.Tensor] = None

    # ---- WeightedLearner contract --------------------------------------
    def sample_batches(self, group_ids: Iterable[int]) -> Dict[int, Batch]:
        out: Dict[int, Batch] = {}
        for gid in group_ids:
            pool = self._group_pool[int(gid)]
            sel = torch.randint(0, pool.numel(), (self.batch_size,), generator=self._rng)
            idx = pool[sel].to(self.device)
            out[int(gid)] = (self.obs[idx], self.act[idx])
        return out

    def per_group_losses(self, batch_by_group) -> Dict[int, torch.Tensor]:
        losses = {}
        for gid, (obs, act) in batch_by_group.items():
            pred = self.policy(obs)
            losses[gid] = ((pred - act) ** 2).mean()
        return losses

    def parameters_for_reweighting(self) -> Iterable[nn.Parameter]:
        return list(self.policy.parameters())

    def train_step(self, weights, batch_by_group) -> Dict[str, float]:
        loss = self.weighted_inner_loss(weights, batch_by_group)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"inner_loss": float(loss.detach())}

    def evaluate(self, n_episodes: int = 64) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.env is not None:
            from smor.evaluation.rollout import rollout_policy

            self.policy.eval()
            with torch.no_grad():
                metrics.update(rollout_policy(
                    self.env,
                    lambda o: self.policy(o.to(self.device, self.dtype)),
                    n_episodes=n_episodes,
                ))
            self.policy.train()
        vl = self.validation_loss()
        if vl is not None:
            metrics["val_loss"] = float(vl.detach())
        return metrics

    # ---- outer-objective helpers ---------------------------------------
    def validation_loss(self, batch_size: Optional[int] = None) -> Optional[torch.Tensor]:
        """Differentiable MSE on held-out expert transitions (outer signal)."""
        if self._val_obs is None:
            return None
        if batch_size is None or batch_size >= self._val_obs.shape[0]:
            obs, act = self._val_obs, self._val_act
        else:
            sel = torch.randint(0, self._val_obs.shape[0], (batch_size,), generator=self._rng)
            sel = sel.to(self.device)
            obs, act = self._val_obs[sel], self._val_act[sel]
        pred = self.policy(obs)
        return ((pred - act) ** 2).mean()

    def closed_loop_loss(self, n_episodes: int = 128, horizon: Optional[int] = None) -> torch.Tensor:
        """Differentiable closed-loop return surrogate (PLAN.md §7).

        Rolls the current policy through the point-mass dynamics (which are pure torch, hence
        differentiable) from a FIXED set of start/goal pairs and returns the mean
        distance-to-goal over the horizon. Minimizing it maximizes episodic return, so it is a
        closed-loop outer objective — unlike open-loop validation MSE it accounts for how action
        errors compound over a rollout (e.g. penalizes systematic over/undershoot).
        """
        if self.env is None:
            raise ValueError("closed_loop_loss requires an env.")
        cfg = self.env.cfg
        H = int(horizon or self.env.horizon)
        if self._cl_pos is None or self._cl_pos.shape[0] != n_episodes:
            g = torch.Generator().manual_seed(12345)
            pos = (torch.rand(n_episodes, 2, generator=g) * 2 - 1) * cfg.world
            goal = (torch.rand(n_episodes, 2, generator=g) * 2 - 1) * cfg.world
            for _ in range(10):
                close = (pos - goal).norm(dim=-1) < cfg.min_start_goal_dist
                if not close.any():
                    break
                ng = (torch.rand(n_episodes, 2, generator=g) * 2 - 1) * cfg.world
                goal = torch.where(close.unsqueeze(-1), ng, goal)
            self._cl_pos = pos.to(self.device, self.dtype)
            self._cl_goal = goal.to(self.device, self.dtype)
        pos, goal = self._cl_pos, self._cl_goal
        total = pos.new_zeros(())
        for _ in range(H):
            obs = torch.cat([pos, goal], dim=-1)
            action = self.policy(obs)
            pos = torch.clamp(pos + cfg.max_step * action, -cfg.world, cfg.world)
            total = total + (pos - goal).norm(dim=-1).mean()
        return total / H

    # ---- checkpointing --------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def clone_params(self) -> dict:
        """Deep copy of policy weights only (for long-horizon perturbation experiments)."""
        return {k: v.detach().clone() for k, v in self.policy.state_dict().items()}

    def restore_params(self, snapshot: dict) -> None:
        self.policy.load_state_dict(snapshot)
