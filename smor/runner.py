"""Convenience factory that wires the point-mass task, demos, grouping and BC learner.

Keeps experiment scripts thin and consistent. Everything is device-aware (GPU-friendly):
the env and learner share the resolved device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from smor.envs.demos import (
    DemoDataset, make_clean_target_dataset, make_multisource_dataset,
    make_two_fidelity_dataset,
)
from smor.envs.point_mass import PointMassConfig, PointMassEnv
from smor.learners.bc import BCLearner
from smor.reweighting.config import OnlineReweighterConfig
from smor.reweighting.grouping import GroupAssignment, make_groups
from smor.utils.seeding import resolve_device, seed_everything


@dataclass
class DataConfig:
    n_expert: int = 40
    n_noisy: int = 40
    noise: float = 0.6
    random_prob: float = 0.3
    horizon: int = 30
    n_val: int = 20
    hidden: Tuple[int, ...] = (64, 64)
    policy_lr: float = 1e-2


@dataclass
class PointMassRun:
    learner: BCLearner
    group_assignment: GroupAssignment
    env: PointMassEnv
    train: DemoDataset
    val: DemoDataset
    device: str
    source_names: list | None = None


def build_pointmass_run(
    cfg: OnlineReweighterConfig,
    data: DataConfig | None = None,
    whole_fidelity: bool = False,
    seed: int | None = None,
) -> PointMassRun:
    """Build a reproducible point-mass BC run consistent with ``cfg`` (n, batch_size, device)."""
    data = data or DataConfig()
    seed = cfg.seed if seed is None else seed
    seed_everything(seed)
    device = str(resolve_device(cfg.device))

    train = make_two_fidelity_dataset(
        n_expert=data.n_expert, n_noisy=data.n_noisy, noise=data.noise,
        random_prob=data.random_prob, horizon=data.horizon, seed=seed,
    )
    val = make_two_fidelity_dataset(
        n_expert=data.n_val, n_noisy=0, noise=0.0, random_prob=0.0,
        horizon=data.horizon, seed=seed + 4242,
    )
    env = PointMassEnv(PointMassConfig(horizon=data.horizon), device=device, seed=seed + 7)
    ga = make_groups(
        train.fidelity_labels(), group_size=cfg.n, seed=seed, whole_fidelity=whole_fidelity
    )
    learner = BCLearner(
        train, ga, hidden=tuple(data.hidden), lr=data.policy_lr,
        batch_size=cfg.batch_size, device=device, val_data=val, env=env, seed=seed,
    )
    return PointMassRun(learner=learner, group_assignment=ga, env=env,
                        train=train, val=val, device=device)


def build_multisource_run(
    cfg: OnlineReweighterConfig,
    sources: list[dict] | None = None,
    n_per_source: int = 40,
    horizon: int = 30,
    n_val: int = 40,
    hidden: Tuple[int, ...] = (64, 64),
    policy_lr: float = 1e-2,
    whole_fidelity: bool = True,
    seed: int | None = None,
) -> PointMassRun:
    """Build a run where training data comes from several *sources* with different error
    structures (e.g. SpaceMouse vs teleop). The validation set / outer objective is the CLEAN
    deployment target, so the reweighter must learn the bias-cancelling source mixture.
    Defaults to whole-fidelity grouping (one beta per source) — the clean interior-mixture test.
    """
    seed = cfg.seed if seed is None else seed
    seed_everything(seed)
    device = str(resolve_device(cfg.device))

    train, names = make_multisource_dataset(
        sources=sources, n_per_source=n_per_source, horizon=horizon, seed=seed, device=device)
    val = make_clean_target_dataset(n=n_val, horizon=horizon, seed=seed + 4242, device=device)
    env = PointMassEnv(PointMassConfig(horizon=horizon), device=device, seed=seed + 7)
    ga = make_groups(train.fidelity_labels(), group_size=cfg.n, seed=seed,
                     whole_fidelity=whole_fidelity)
    learner = BCLearner(
        train, ga, hidden=tuple(hidden), lr=policy_lr, batch_size=cfg.batch_size,
        device=device, val_data=val, env=env, seed=seed)
    return PointMassRun(learner=learner, group_assignment=ga, env=env,
                        train=train, val=val, device=device, source_names=names)
