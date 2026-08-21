"""Variable-length ("ragged") trajectory dataset (PLAN.md §10/§11 data contract).

``DemoDataset`` (``smor.envs.demos``) stores demonstrations as a dense ``(N, T, dim)`` tensor and
therefore assumes a *fixed* horizon — fine for the synthetic point-mass / Meta-World rollouts,
but real datasets (RoboMimic, RoboTurk, Open-X, ...) have trajectories of different lengths.

:class:`TrajectoryDataset` stores demonstrations as a single flat transition table plus an
explicit per-transition ``traj_id`` and a per-trajectory ``fidelity`` label. It exposes exactly
the attributes/methods the reweighting stack reads off a dataset —
``flatten() -> (obs, act, traj_id)``, ``num_trajectories``, ``obs_dim``, ``act_dim``,
``fidelity_labels()`` — so :class:`smor.learners.bc.BCLearner`,
:func:`smor.reweighting.grouping.make_groups`, and the outer objectives all work unchanged (they
duck-type on this interface, never on ``DemoDataset`` internals).

Tensors are kept on CPU float32; the learner moves batches to its device on sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch


@dataclass
class TrajectoryDataset:
    """Flat, variable-length trajectory demonstrations with per-trajectory fidelity labels.

    Attributes:
        obs:      (M, obs_dim) float32 — all transitions concatenated over trajectories.
        act:      (M, act_dim) float32 — aligned with ``obs``.
        traj_id:  (M,) int64 — dense trajectory index in ``[0, N)`` for each transition; must be
                  contiguous & sorted (all transitions of trajectory ``i`` form one block).
        fidelity: (N,) int64 — fidelity / source label per trajectory (the reweighting group key).
    """

    obs: torch.Tensor
    act: torch.Tensor
    traj_id: torch.Tensor
    fidelity: torch.Tensor

    def __post_init__(self):
        self.obs = self.obs.to(torch.float32).cpu()
        self.act = self.act.to(torch.float32).cpu()
        self.traj_id = self.traj_id.to(torch.int64).cpu()
        self.fidelity = self.fidelity.to(torch.int64).cpu()
        if not (self.obs.shape[0] == self.act.shape[0] == self.traj_id.shape[0]):
            raise ValueError("obs/act/traj_id must share the transition count M.")
        n_from_ids = int(self.traj_id.max().item()) + 1 if self.traj_id.numel() else 0
        if n_from_ids != self.fidelity.shape[0]:
            raise ValueError(
                f"traj_id references {n_from_ids} trajectories but fidelity has "
                f"{self.fidelity.shape[0]} entries."
            )

    # ---- DemoDataset-compatible interface ------------------------------
    def flatten(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(obs_flat, act_flat, traj_id)`` over all transitions (already flat)."""
        return self.obs, self.act, self.traj_id

    def fidelity_labels(self) -> np.ndarray:
        return self.fidelity.numpy()

    @property
    def num_trajectories(self) -> int:
        return int(self.fidelity.shape[0])

    @property
    def num_transitions(self) -> int:
        return int(self.obs.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.act.shape[-1])

    @property
    def horizon(self) -> int:
        """Mean trajectory length (informational only; horizons are not uniform)."""
        n = self.num_trajectories
        return int(round(self.num_transitions / n)) if n else 0

    def traj_lengths(self) -> np.ndarray:
        return np.bincount(self.traj_id.numpy(), minlength=self.num_trajectories)

    # ---- construction --------------------------------------------------
    @classmethod
    def from_trajectories(
        cls,
        trajectories: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        fidelity: Sequence[int],
    ) -> "TrajectoryDataset":
        """Build from a list of ``(obs (T,·), act (T,·))`` pairs + one fidelity label each."""
        if len(trajectories) != len(fidelity):
            raise ValueError("need exactly one fidelity label per trajectory.")
        if not trajectories:
            raise ValueError("no trajectories provided.")
        obs_parts, act_parts, id_parts = [], [], []
        for i, (o, a) in enumerate(trajectories):
            o = torch.as_tensor(o, dtype=torch.float32)
            a = torch.as_tensor(a, dtype=torch.float32)
            if o.shape[0] != a.shape[0]:
                raise ValueError(f"trajectory {i}: obs/act length mismatch.")
            obs_parts.append(o)
            act_parts.append(a)
            id_parts.append(torch.full((o.shape[0],), i, dtype=torch.int64))
        return cls(
            obs=torch.cat(obs_parts, dim=0),
            act=torch.cat(act_parts, dim=0),
            traj_id=torch.cat(id_parts, dim=0),
            fidelity=torch.as_tensor(list(fidelity), dtype=torch.int64),
        )

    def subset(self, traj_indices: Sequence[int]) -> "TrajectoryDataset":
        """Return a new dataset with only the given trajectories (re-indexed densely)."""
        keep = {int(i) for i in traj_indices}
        remap = {old: new for new, old in enumerate(sorted(keep))}
        tid = self.traj_id.numpy()
        mask = np.isin(tid, list(keep))
        new_ids = np.array([remap[int(t)] for t in tid[mask]], dtype=np.int64)
        new_fid = np.array([int(self.fidelity[old]) for old in sorted(keep)], dtype=np.int64)
        return TrajectoryDataset(
            obs=self.obs[mask],
            act=self.act[mask],
            traj_id=torch.from_numpy(new_ids),
            fidelity=torch.from_numpy(new_fid),
        )

    # ---- persistence ---------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"obs": self.obs, "act": self.act, "traj_id": self.traj_id, "fidelity": self.fidelity},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryDataset":
        d = torch.load(path, map_location="cpu")
        return cls(obs=d["obs"], act=d["act"], traj_id=d["traj_id"], fidelity=d["fidelity"])
