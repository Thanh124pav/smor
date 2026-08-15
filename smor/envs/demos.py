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
def _rotate(a: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate 2D action vectors (..., 2) by ``angle_rad`` (systematic directional error)."""
    if angle_rad == 0.0:
        return a
    import math
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    ax, ay = a[..., 0], a[..., 1]
    return torch.stack([c * ax - s * ay, s * ax + c * ay], dim=-1)


def collect_demonstrations(
    env: PointMassEnv,
    n_traj: int,
    noise: float = 0.0,
    random_prob: float = 0.0,
    rotation_deg: float = 0.0,
    gain: float = 1.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out an (optionally corrupted) expert; return ``(obs, act)`` of shape (n_traj, T, ·).

    Corruption models a demonstration *source*/device:
      ``rotation_deg``: systematic directional miscalibration (rotate every action) — the key
        knob for the "different error structure" fidelity model (e.g. SpaceMouse vs teleop);
      ``gain``: systematic magnitude error (overshoot/undershoot);
      ``noise``: std of zero-mean Gaussian action noise (device jitter);
      ``random_prob``: probability a step is replaced by a uniform random action (blunders).
    The executed (corrupted) action is both recorded as the BC target and applied to the env, so
    the induced state distribution reflects the source's behavior.
    """
    import math
    gen = torch.Generator(device="cpu").manual_seed(seed)
    env.generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    rot = math.radians(rotation_deg)
    T = env.horizon
    obs = env.reset(n_traj)  # (n_traj, obs_dim) on env.device
    obs_buf = torch.empty(n_traj, T, env.obs_dim)
    act_buf = torch.empty(n_traj, T, env.act_dim)
    for t in range(T):
        base = expert_action(obs, max_step=env.cfg.max_step)
        a = gain * _rotate(base, rot)
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


# --- multi-source ("different error structure") demonstrations ---------------
#
# Models genuinely different demonstration *devices* (e.g. SpaceMouse vs human teleop): each
# covers the whole task but has its own systematic error (a directional rotation bias + gain +
# jitter). No source is globally best, so the optimal reweighting is a NON-trivial interior
# mixture whose weighted biases cancel — training on a single source (or naive uniform) is
# suboptimal. Fidelity label = source index.

# Two default device profiles with OPPOSITE rotation bias of DIFFERENT magnitude, so the
# bias-cancelling mixture is interior AND asymmetric (beats uniform and either source alone).
# Different *error structures*, both full-coverage, neither globally best:
#   spacemouse -> small precise deflections that UNDERSHOOT (gain<1) + slight +rotation, low jitter
#   teleop     -> jerky inputs that OVERSHOOT (gain>1) + slight -rotation, more jitter + blunders
# The magnitude errors are opposite, so only a weighted mixture restores unit gain (best task
# performance). A single source under/overshoots and naive uniform doesn't balance the different
# magnitudes -> the optimal beta is an interior, asymmetric mixture.
DEFAULT_SOURCES = [
    {"name": "spacemouse",  "rotation_deg": +8.0,  "gain": 0.80, "noise": 0.05, "random_prob": 0.00},
    {"name": "teleop",      "rotation_deg": -12.0, "gain": 1.25, "noise": 0.10, "random_prob": 0.05},
    {"name": "kinesthetic", "rotation_deg": +22.0, "gain": 1.00, "noise": 0.14, "random_prob": 0.02},
]


def make_multisource_dataset(
    sources: list[dict] | None = None,
    n_per_source: int = 40,
    horizon: int = 40,
    seed: int = 0,
    device: str = "cpu",
) -> tuple["DemoDataset", list[str]]:
    """Build a dataset from several demonstration *sources* with different error structures.

    Each entry in ``sources`` is a dict with keys ``name`` and any of
    ``rotation_deg``/``gain``/``noise``/``random_prob``/``n_traj``. Returns
    ``(dataset, source_names)`` where ``dataset.fidelity`` holds the source index.
    """
    sources = sources if sources is not None else DEFAULT_SOURCES
    cfg = PointMassConfig(horizon=horizon)
    env = PointMassEnv(cfg, device=device, seed=seed)
    obs_list, act_list, fid_list, names = [], [], [], []
    for i, src in enumerate(sources):
        n = int(src.get("n_traj", n_per_source))
        o, a = collect_demonstrations(
            env, n,
            noise=float(src.get("noise", 0.0)),
            random_prob=float(src.get("random_prob", 0.0)),
            rotation_deg=float(src.get("rotation_deg", 0.0)),
            gain=float(src.get("gain", 1.0)),
            seed=seed + 100 * (i + 1),
        )
        obs_list.append(o); act_list.append(a)
        fid_list.append(torch.full((n,), i, dtype=torch.int64))
        names.append(str(src.get("name", f"source{i}")))
    ds = DemoDataset(obs=torch.cat(obs_list), act=torch.cat(act_list),
                     fidelity=torch.cat(fid_list))
    return ds, names


def make_clean_target_dataset(n: int = 20, horizon: int = 40, seed: int = 0,
                              device: str = "cpu") -> "DemoDataset":
    """Clean expert demonstrations = the deployment target for the outer objective (no device
    error). Used as the held-out validation set when training on imperfect sources."""
    cfg = PointMassConfig(horizon=horizon)
    env = PointMassEnv(cfg, device=device, seed=seed)
    o, a = collect_demonstrations(env, n, noise=0.0, random_prob=0.0, seed=seed)
    return DemoDataset(obs=o, act=a, fidelity=torch.zeros(n, dtype=torch.int64))
