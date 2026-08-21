"""Load RoboMimic HDF5 demonstrations into SMOR's trajectory contract.

Public entry point: :func:`load_robomimic_mix`, which turns a list of :class:`Component`
specifications (each = one task variant / quality tier) into a train
:class:`~smor.data.trajectory_dataset.TrajectoryDataset` (``fidelity`` = source index) plus a
held-out clean-target validation dataset, source names, and a per-source quality score for the
CAIL-style ranking baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from smor.data.robomimic.registry import ensure_dataset
from smor.data.trajectory_dataset import TrajectoryDataset

# Canonical robomimic low-dim observation set (object state + proprio). 19 dims for lift/can/
# square; the loader concatenates whichever of these keys the file actually contains, in order.
DEFAULT_OBS_KEYS: Tuple[str, ...] = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

# Higher = cleaner / more optimal source. Used for the static-quality and CAIL-ranking baselines.
TIER_QUALITY = {
    "ph": 1.0,
    "better": 0.9,
    "okay": 0.5,
    "worse": 0.1,
    "mg": 0.2,
    "mg_success": 0.4,
    "mg_fail": 0.0,
}


@dataclass
class Component:
    """One source in a RoboMimic mix = a (task, variant[, quality tier]) slice.

    Attributes:
        task:    robomimic task, e.g. ``"lift"``, ``"can"``, ``"square"``.
        dtype:   variant: ``"ph"`` | ``"mh"`` | ``"mg"``.
        tier:    quality filter within the variant. For ``mh``: ``better``/``okay``/``worse``
                 (or any HDF5 mask key). For ``mg``: ``mg_success``/``mg_fail`` to split by
                 rollout success. ``None`` = use every demo in the file.
        n:       optional cap on the number of *training* trajectories kept from this source.
        name:    display name (defaults to ``task-dtype[:tier]``).
        quality: quality score override (defaults to :data:`TIER_QUALITY`).
        target:  if True, this source is (part of) the clean deployment target — its held-out
                 split becomes the outer-objective validation set. If no component sets it, the
                 highest-quality component is used automatically.
    """

    task: str
    dtype: str
    tier: Optional[str] = None
    n: Optional[int] = None
    name: Optional[str] = None
    quality: Optional[float] = None
    target: bool = False

    def resolved_name(self) -> str:
        if self.name:
            return self.name
        return f"{self.task}-{self.dtype}" + (f":{self.tier}" if self.tier else "")

    def resolved_quality(self) -> float:
        if self.quality is not None:
            return float(self.quality)
        key = self.tier if self.tier in TIER_QUALITY else self.dtype
        return float(TIER_QUALITY.get(key, 0.5))


def _obs_vector(obs_group, obs_keys: Sequence[str]) -> np.ndarray:
    """Concatenate the requested low-dim obs keys along the feature axis -> (T, D)."""
    present = [k for k in obs_keys if k in obs_group]
    if not present:
        raise KeyError(
            f"none of obs_keys={list(obs_keys)} found; file has {list(obs_group.keys())}"
        )
    return np.concatenate([np.asarray(obs_group[k], dtype=np.float32) for k in present], axis=-1)


def _demo_success(demo) -> bool:
    """True if a (machine-generated) rollout reached the goal, from its sparse reward / dones."""
    if "rewards" in demo:
        r = np.asarray(demo["rewards"])
        if r.size and float(r.max()) > 0.0:
            return True
    if "dones" in demo:
        d = np.asarray(demo["dones"])
        if d.size and int(d[-1]) == 1:
            return True
    return False


def _select_demo_names(f, tier: Optional[str]) -> List[str]:
    """Resolve the list of demo names for a quality tier (HDF5 mask key or mg success split)."""
    data = f["data"]
    all_demos = list(data.keys())
    if tier is None:
        return all_demos
    if tier in ("mg_success", "mg_fail"):
        want = tier == "mg_success"
        return [d for d in all_demos if _demo_success(data[d]) == want]
    mask = f.get("mask")
    if mask is not None and tier in mask:
        return [n.decode() if isinstance(n, bytes) else str(n) for n in mask[tier][:]]
    raise KeyError(
        f"tier '{tier}' is not an available mask key. "
        f"Available: {sorted(mask.keys()) if mask is not None else '<no mask group>'}"
    )


def _read_trajectories(
    path: str | Path, obs_keys: Sequence[str], tier: Optional[str]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    import h5py

    out: List[Tuple[np.ndarray, np.ndarray]] = []
    with h5py.File(str(path), "r") as f:
        names = _select_demo_names(f, tier)
        data = f["data"]
        for name in names:
            demo = data[name]
            obs = _obs_vector(demo["obs"], obs_keys)
            act = np.asarray(demo["actions"], dtype=np.float32)
            n = min(obs.shape[0], act.shape[0])
            out.append((obs[:n], act[:n]))
    return out


def load_robomimic_mix(
    components: Sequence[Component],
    obs_keys: Sequence[str] = DEFAULT_OBS_KEYS,
    val_frac: float = 0.1,
    seed: int = 0,
    root: str | Path | None = None,
    download: bool = True,
    val_mode: str = "stratified",
    val_per_source: int | None = None,
) -> Tuple[TrajectoryDataset, TrajectoryDataset, List[str], List[float]]:
    """Assemble a multi-source RoboMimic mix.

    Each component's trajectories get ``fidelity = component_index``. How the held-out validation
    set (used by the outer objective ``ValidationLoss``) is drawn is controlled by ``val_mode``:

    * ``"stratified"`` (default) — hold out a slice from **every** source (``val_frac`` of each, or
      exactly ``val_per_source`` trajectories each), then pool them. The outer signal is then
      balanced across sources, so it does NOT privilege one "clean" source — which is what stops
      the confidence weights from collapsing onto whichever training source happens to match a
      single-source validation set. ``val`` keeps each held-out trajectory's source index as its
      fidelity label (so the val set itself is inspectable / can be balanced downstream).
    * ``"target"`` — legacy behaviour: hold out ``val_frac`` of the *target* component(s) only
      (the ``*``-marked, else highest-quality source). This makes the outer loss "match the clean
      target", which drives the weights to a near-corner solution — kept only for comparison.

    Returns ``(train, val, source_names, source_quality)``.
    """
    if not components:
        raise ValueError("need at least one component.")
    if val_mode not in ("stratified", "target"):
        raise ValueError(f"val_mode must be 'stratified' or 'target', got {val_mode!r}")
    rng = np.random.default_rng(seed)

    names = [c.resolved_name() for c in components]
    quality = [c.resolved_quality() for c in components]
    targets = [i for i, c in enumerate(components) if c.target]
    if not targets:  # auto-pick the highest-quality source as the clean target
        targets = [int(np.argmax(quality))]

    # read every component once, then decide the held-out split (needs global sizes to balance)
    comp_trajs: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    comp_order: List[np.ndarray] = []
    for i, comp in enumerate(components):
        path = (
            ensure_dataset(comp.task, comp.dtype, root=root)
            if download
            else Path(root or "data/robomimic") / comp.task / comp.dtype
        )
        trajs = _read_trajectories(path, obs_keys, comp.tier)
        if not trajs:
            raise ValueError(f"component {names[i]} yielded 0 trajectories.")
        comp_trajs.append(trajs)
        comp_order.append(rng.permutation(len(trajs)))

    # balanced stratified default: equal #val trajectories per source (so a large source like MG
    # cannot dominate the outer signal). Falls back to val_frac-of-the-smallest-source if unset.
    if val_mode == "stratified" and val_per_source is None:
        val_per_source = max(1, min(int(round(val_frac * len(t))) for t in comp_trajs))

    train_trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    train_fid: List[int] = []
    val_trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    val_fid: List[int] = []

    for i, comp in enumerate(components):
        trajs, order = comp_trajs[i], comp_order[i]
        # how many trajectories this source contributes to the held-out validation set
        if val_mode == "stratified":
            n_val = min(max(int(val_per_source), 1), len(trajs) - 1)
        else:  # "target": only the target source(s) contribute to val (legacy; collapses beta)
            n_val = (min(max(int(round(val_frac * len(trajs))), 1), len(trajs) - 1)
                     if i in targets else 0)
        val_idx, train_idx = order[:n_val], order[n_val:]
        if comp.n is not None:
            train_idx = train_idx[: comp.n]
        for j in train_idx:
            train_trajs.append(trajs[j])
            train_fid.append(i)
        for j in val_idx:
            val_trajs.append(trajs[j])
            val_fid.append(i)

    if not val_trajs:
        raise ValueError("no validation trajectories produced (check val_frac / val_mode).")

    train = TrajectoryDataset.from_trajectories(train_trajs, train_fid)
    # Val keeps the source index as fidelity so the held-out set is stratified/inspectable;
    # ValidationLoss pools all val transitions regardless of this label.
    val = TrajectoryDataset.from_trajectories(val_trajs, val_fid)
    return train, val, names, quality


def group_quality_from_labels(group_fidelity: Sequence[int], source_quality: Sequence[float]) -> dict:
    """Per-group quality dict (for :class:`CAILRankingLoss`) from per-source quality scores."""
    return {g: float(source_quality[int(fid)]) for g, fid in enumerate(group_fidelity)}
