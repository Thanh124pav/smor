"""Fixed trajectory grouping with configurable granularity ``n`` (PLAN.md Stage C).

Group *trajectories*, never individual timesteps. Never create a group that crosses a
fidelity/domain boundary. Groups are seeded, reproducible, and immutable for a whole run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class GroupAssignment:
    """Immutable output of :func:`make_groups`.

    Attributes:
        group_id:       (N,) int array, group index per trajectory.
        group_fidelity: (M,) int array, fidelity label of each group.
        group_sizes:    (M,) int array, number of trajectories per group.
        members:        list of length M; members[j] = trajectory indices in group j.
        n:              requested group size (granularity).
    """

    group_id: np.ndarray
    group_fidelity: np.ndarray
    group_sizes: np.ndarray
    members: List[np.ndarray]
    n: int

    @property
    def num_groups(self) -> int:
        return int(self.group_fidelity.shape[0])

    @property
    def num_trajectories(self) -> int:
        return int(self.group_id.shape[0])

    def fidelity_to_groups(self) -> Dict[int, List[int]]:
        """Map each fidelity label -> list of group ids."""
        out: Dict[int, List[int]] = {}
        for j, fid in enumerate(self.group_fidelity.tolist()):
            out.setdefault(int(fid), []).append(j)
        return out


def make_groups(
    fidelity_labels: Sequence[int],
    group_size: int,
    seed: int = 0,
    whole_fidelity: bool = False,
) -> GroupAssignment:
    """Partition trajectories into fixed groups of ~``group_size`` within each fidelity.

    Args:
        fidelity_labels: length-N sequence, fidelity/domain label per trajectory.
        group_size:      ``n`` — target trajectories per group (>= 1). Ignored if
                         ``whole_fidelity`` is True.
        seed:            RNG seed for the deterministic within-fidelity shuffle.
        whole_fidelity:  if True, one group per fidelity level (n = |fidelity|).

    Rules enforced:
        * every trajectory appears in exactly one group;
        * no group crosses a fidelity boundary;
        * the final group in a fidelity may be smaller (uneven allowed);
        * a fidelity with zero trajectories produces no group.
    """
    labels = np.asarray(list(fidelity_labels))
    if labels.ndim != 1:
        raise ValueError("fidelity_labels must be 1-D.")
    N = labels.shape[0]
    if N == 0:
        raise ValueError("no trajectories to group.")
    if not whole_fidelity and group_size < 1:
        raise ValueError(f"group_size (n) must be >= 1, got {group_size}")

    rng = np.random.default_rng(seed)

    group_id = np.full(N, -1, dtype=np.int64)
    group_fidelity: List[int] = []
    members: List[np.ndarray] = []

    # Deterministic order over fidelity labels.
    unique_fids = sorted(int(f) for f in np.unique(labels))
    next_gid = 0
    for fid in unique_fids:
        idx = np.where(labels == fid)[0]
        if idx.size == 0:
            continue
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        if whole_fidelity:
            chunks = [shuffled]
        else:
            n_chunks = int(np.ceil(shuffled.size / group_size))
            chunks = np.array_split(shuffled, n_chunks)
        for chunk in chunks:
            if chunk.size == 0:
                continue
            group_id[chunk] = next_gid
            group_fidelity.append(fid)
            members.append(np.sort(chunk))
            next_gid += 1

    if (group_id < 0).any():
        missing = int((group_id < 0).sum())
        raise RuntimeError(f"{missing} trajectories were not assigned to any group.")

    group_sizes = np.array([m.size for m in members], dtype=np.int64)
    if (group_sizes == 0).any():
        raise RuntimeError("a group ended up with zero trajectories.")

    effective_n = int(group_sizes.max()) if whole_fidelity else int(group_size)
    return GroupAssignment(
        group_id=group_id,
        group_fidelity=np.asarray(group_fidelity, dtype=np.int64),
        group_sizes=group_sizes,
        members=members,
        n=effective_n,
    )
