"""Loader for the official CAIL (Zhang et al., NeurIPS 2021) demonstration buffers.

CAIL — *Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality*
(https://arxiv.org/abs/2110.14754, https://github.com/Stanford-ILIAD/Confidence-Aware-Imitation-Learning)
— ships mixed-optimality demonstration buffers used to benchmark learning-from-imperfect-demos.
Each buffer is a flat transition table (``.pth`` dict: ``state``/``action``/``reward``/``done``/
``next_state``) built by rolling out **5 policy checkpoints of decreasing quality** (e.g. Ant-v2:
200 trajectories, 40 per policy, mean returns 4787 / 3740 / 2947 / 2115 / 789).

This loader splits the buffer back into trajectories (on ``done``), computes each trajectory's
return, and returns a :class:`~smor.data.trajectory_dataset.TrajectoryDataset` — so SMOR's
learner-agnostic online reweighting can reweight the CAIL demonstrations directly, at any
granularity (use ``n=1`` for per-demonstration confidence, the CAIL setting).

Download the official buffers with::

    cd data/cail && gdown 1oohGvjlEqhwZwof5vHnwr_mx2D0AjlzU -O buffers.tar.gz && tar xzf buffers.tar.gz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from smor.data.trajectory_dataset import TrajectoryDataset


@dataclass
class CAILData:
    """A CAIL buffer split into trajectories, with per-trajectory returns + source-policy ids."""

    dataset: TrajectoryDataset          # obs/act with fidelity = source-policy index (0=best)
    returns: np.ndarray                 # (N,) true episodic return per trajectory
    source: np.ndarray                  # (N,) source-policy block index (0=best .. 4=worst)
    env_id: str

    @property
    def num_trajectories(self) -> int:
        return int(self.returns.shape[0])


def _default_buffer(env_id: str, root: Path) -> Path:
    hits = sorted((root / env_id).glob("size*_reward_*.pth"))
    if not hits:
        raise FileNotFoundError(
            f"no CAIL buffer for {env_id} under {root/env_id}. Download with "
            f"`cd data/cail && gdown 1oohGvjlEqhwZwof5vHnwr_mx2D0AjlzU -O buffers.tar.gz "
            f"&& tar xzf buffers.tar.gz`."
        )
    return hits[0]


def load_cail_buffer(
    env_id: str = "Ant-v2",
    path: str | Path | None = None,
    root: str | Path = "data/cail",
    n_sources: int = 5,
) -> CAILData:
    """Load a CAIL ``.pth`` buffer into trajectory form with returns + source labels.

    The buffer is a flat transition table; trajectories are cut on the ``done`` flag. The buffer
    concatenates ``n_sources`` equal-size policy blocks in *decreasing* quality order, so the
    source-policy index (0 = best expert .. n_sources-1 = worst) is assigned by contiguous block.
    """
    path = Path(path) if path is not None else _default_buffer(env_id, Path(root))
    d = torch.load(path, map_location="cpu")
    state, action = d["state"].float(), d["action"].float()
    reward = d["reward"].float().squeeze(-1)
    done = d["done"].float().squeeze(-1)

    # cut into trajectories on done
    ends = (done > 0.5).nonzero(as_tuple=True)[0].tolist()
    if not ends or ends[-1] != len(done) - 1:
        ends.append(len(done) - 1)
    trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    returns: List[float] = []
    start = 0
    for e in ends:
        sl = slice(start, e + 1)
        trajs.append((state[sl].numpy(), action[sl].numpy()))
        returns.append(float(reward[sl].sum()))
        start = e + 1
    n = len(trajs)

    # source-policy index by contiguous equal-size blocks (decreasing quality)
    per = int(np.ceil(n / n_sources))
    source = np.minimum(np.arange(n) // per, n_sources - 1).astype(np.int64)

    dataset = TrajectoryDataset.from_trajectories(trajs, source.tolist())
    return CAILData(dataset=dataset, returns=np.asarray(returns, dtype=np.float64),
                    source=source, env_id=env_id)
