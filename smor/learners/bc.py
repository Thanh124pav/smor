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
        data_device: str | torch.device | None = None,
    ):
        self.device = resolve_device(device)
        # Where the (potentially large) demonstration tensors live. Keep on CPU for big
        # datasets to avoid GPU OOM; batches are moved to self.device on sampling.
        self.data_device = self.device if data_device is None else torch.device(data_device)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.env = env
        self._rng = torch.Generator(device="cpu").manual_seed(seed)

        obs_flat, act_flat, traj_id = dataset.flatten()
        self.obs = obs_flat.to(self.data_device, dtype)
        self.act = act_flat.to(self.data_device, dtype)
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
            idx = pool[sel].to(self.data_device)
            out[int(gid)] = (self.obs[idx].to(self.device), self.act[idx].to(self.device))
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
        if self.env is not None and n_episodes > 0:
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

    # ---- closed-loop (policy-gradient) outer objectives (PLAN.md §7) ----
    #
    # A NON-differentiable env (robosuite/MuJoCo/Meta-World) cannot be back-propagated through, so
    # the gradient of the task return w.r.t. theta is estimated with a policy-gradient (score-
    # function) surrogate L_out = -E_t[ logpi_theta(a_t|s_t) * A_t ], grad_theta L_out = -grad E[R].
    # Three advantage estimators are provided (REINFORCE / GRPO / PPO); all share one rollout.
    # Because the outer signal is TASK RETURN (not action-matching to a privileged clean source),
    # this avoids the trivial-corner trap of ValidationLoss. Pair with cfg.normalize_group_grads to
    # stop the noisy g_out from being hijacked by whichever group has the largest gradient.

    @torch.no_grad()
    def _rollout_collect(self, n_episodes: int, H: int, std: float):
        """Roll the current policy with Gaussian exploration; return (obs, act, rew, active)."""
        obs_log, act_log, rew_log, active_log = [], [], [], []
        self.policy.eval()
        obs = self.env.reset(n_episodes)
        done = torch.zeros(n_episodes, dtype=torch.bool, device=obs.device)
        for _ in range(H):
            mean = self.policy(obs.to(self.device, self.dtype))
            a = torch.clamp(mean + std * torch.randn_like(mean), -1.0, 1.0)
            obs_log.append(obs.to(self.device, self.dtype))
            act_log.append(a.detach())
            active_log.append((~done).to(self.device))
            obs, reward, done_step, _ = self.env.step(a)
            rew_log.append(reward.to(self.device))
            done = done | done_step.to(done.device).bool()
            if bool(done.all()):
                break
        self.policy.train()
        return (torch.stack(obs_log), torch.stack(act_log),
                torch.stack(rew_log), torch.stack(active_log).float())

    def _pg_surrogate(self, obs_all, act_all, adv_t, active, std) -> torch.Tensor:
        """-E[ logpi(a|s) * A ] over active steps (differentiable in theta)."""
        T, B = act_all.shape[0], act_all.shape[1]
        mean_flat = self.policy(obs_all.reshape(T * B, -1))           # WITH grad
        a_flat = act_all.reshape(T * B, -1)
        logp = -0.5 * (((a_flat - mean_flat) / std) ** 2).sum(-1)     # (T*B,) up to const
        w = (active * adv_t).reshape(-1)
        return -(logp * w).sum() / active.sum().clamp_min(1.0)

    def rollout_pg_surrogate(self, n_episodes: int = 16, horizon: Optional[int] = None,
                             explore_std: float = 0.1) -> torch.Tensor:
        """REINFORCE: episodic return with a MEAN baseline (highest variance)."""
        if self.env is None:
            raise ValueError("rollout_pg_surrogate requires an attached env.")
        H, std = int(horizon or self.env.horizon), float(explore_std)
        obs_all, act_all, rew_all, active = self._rollout_collect(n_episodes, H, std)
        returns = (rew_all * active).sum(0)
        adv = returns - returns.mean()
        return self._pg_surrogate(obs_all, act_all, adv.unsqueeze(0).expand_as(active), active, std)

    def rollout_grpo_surrogate(self, n_episodes: int = 32, horizon: Optional[int] = None,
                               explore_std: float = 0.1) -> torch.Tensor:
        """GRPO: GROUP-RELATIVE advantage ``(R_i - mean)/std`` over the rollout group (no critic)."""
        if self.env is None:
            raise ValueError("rollout_grpo_surrogate requires an attached env.")
        H, std = int(horizon or self.env.horizon), float(explore_std)
        obs_all, act_all, rew_all, active = self._rollout_collect(n_episodes, H, std)
        returns = (rew_all * active).sum(0)
        adv = (returns - returns.mean()) / (returns.std() + 1e-6)     # group-normalized
        return self._pg_surrogate(obs_all, act_all, adv.unsqueeze(0).expand_as(active), active, std)

    def rollout_ppo_surrogate(self, n_episodes: int = 32, horizon: Optional[int] = None,
                              explore_std: float = 0.1, gamma: float = 0.99) -> torch.Tensor:
        """PPO-style: per-step REWARD-TO-GO advantage with a per-timestep baseline, whitened.

        Uses temporal credit assignment (discounted return-to-go G_t minus a batch time-baseline
        b_t = mean_i G_t^i) rather than the episode-level return of REINFORCE/GRPO — lower-variance,
        the same advantage PPO/GAE build on (a learned critic + clipped ratio can be added later).
        """
        if self.env is None:
            raise ValueError("rollout_ppo_surrogate requires an attached env.")
        H, std = int(horizon or self.env.horizon), float(explore_std)
        obs_all, act_all, rew_all, active = self._rollout_collect(n_episodes, H, std)
        T, B = rew_all.shape
        # discounted reward-to-go per step
        g = torch.zeros(B, device=rew_all.device)
        rtg = torch.zeros_like(rew_all)
        for t in range(T - 1, -1, -1):
            g = rew_all[t] * active[t] + gamma * g
            rtg[t] = g
        baseline = (rtg * active).sum(1, keepdim=True) / active.sum(1, keepdim=True).clamp_min(1)
        adv = rtg - baseline                                          # (T, B)
        adv = adv / (adv[active > 0].std() + 1e-6)
        return self._pg_surrogate(obs_all, act_all, adv, active, std)

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
