"""Demonstration collection with two fidelity levels (PLAN.md §10).

Fidelity 0 = expert (clean near-optimal actions); fidelity 1 = noisy (Gaussian action noise
plus occasional uniform-random actions). Reweighting should learn to downweight the noisy
group. Data is stored as CPU float32 tensors and moved to the device by the learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from smor.envs.point_mass import PointMassEnv, PointMassConfig, expert_action


@dataclass
class DemoDataset:
    """Trajectory-structured demonstrations with fidelity labels."""

    obs: torch.Tensor        # (N, T, obs_dim) float32 (cpu)
    act: torch.Tensor        # (N, T, act_dim) float32 (cpu)
    fidelity: torch.Tensor   # (N,) int64 (cpu)

    def __post_init__(self):
        assert self.obs.shape[0] == self.act.shape[0] == self.fidelity.shape[0]

    @property
    def num_trajectories(self) -> int:
        return int(self.obs.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.obs.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.act.shape[-1])

    def flatten(self):
        """Return ``(obs_flat, act_flat, traj_id)`` over all transitions."""
        N, T = self.obs.shape[0], self.obs.shape[1]
        obs_flat = self.obs.reshape(N * T, -1)
        act_flat = self.act.reshape(N * T, -1)
        traj_id = torch.arange(N).repeat_interleave(T)
        return obs_flat, act_flat, traj_id

    def fidelity_labels(self) -> np.ndarray:
        return self.fidelity.numpy()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"obs": self.obs, "act": self.act, "fidelity": self.fidelity}, path
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DemoDataset":
        d = torch.load(path, map_location="cpu")
        return cls(obs=d["obs"], act=d["act"], fidelity=d["fidelity"])


@torch.no_grad()
def collect_demonstrations(
    env: PointMassEnv,
    n_traj: int,
    noise: float = 0.0,
    random_prob: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out an (optionally corrupted) expert; return ``(obs, act)`` of shape (n_traj, T, ·).

    ``noise``: std of Gaussian action noise. ``random_prob``: probability a step is replaced
    by a uniform random action in ``[-1, 1]^2`` (models a suboptimal demonstrator).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    env.generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    T = env.horizon
    obs = env.reset(n_traj)  # (n_traj, obs_dim) on env.device
    obs_buf = torch.empty(n_traj, T, env.obs_dim)
    act_buf = torch.empty(n_traj, T, env.act_dim)
    for t in range(T):
        base = expert_action(obs, max_step=env.cfg.max_step)
        a = base
        if noise > 0:
            a = a + noise * torch.randn(a.shape, generator=gen).to(a.device)
        if random_prob > 0:
            rand_a = (torch.rand(a.shape, generator=gen) * 2 - 1).to(a.device)
            mask = (torch.rand(n_traj, 1, generator=gen).to(a.device) < random_prob)
            a = torch.where(mask, rand_a, a)
        a = torch.clamp(a, -1.0, 1.0)
        obs_buf[:, t] = obs.cpu()
        act_buf[:, t] = a.cpu()
        obs, _, _, _ = env.step(a)
    return obs_buf, act_buf


def make_two_fidelity_dataset(
    n_expert: int = 40,
    n_noisy: int = 40,
    noise: float = 0.6,
    random_prob: float = 0.3,
    horizon: int = 40,
    seed: int = 0,
    device: str = "cpu",
) -> DemoDataset:
    """Build an expert (fidelity 0) + noisy (fidelity 1) demonstration dataset."""
    cfg = PointMassConfig(horizon=horizon)
    env = PointMassEnv(cfg, device=device, seed=seed)
    exp_obs, exp_act = collect_demonstrations(env, n_expert, noise=0.0, random_prob=0.0, seed=seed)
    noisy_obs, noisy_act = collect_demonstrations(
        env, n_noisy, noise=noise, random_prob=random_prob, seed=seed + 100
    )
    obs = torch.cat([exp_obs, noisy_obs], dim=0)
    act = torch.cat([exp_act, noisy_act], dim=0)
    fidelity = torch.cat([
        torch.zeros(n_expert, dtype=torch.int64),
        torch.ones(n_noisy, dtype=torch.int64),
    ])
    return DemoDataset(obs=obs, act=act, fidelity=fidelity)
