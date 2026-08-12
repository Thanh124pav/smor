"""Shared helpers for experiment drivers: config IO and CLI overrides."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from smor.reweighting.config import OnlineReweighterConfig
from smor.runner import DataConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _filter(d: Dict[str, Any], cls) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def build_configs(
    raw: Dict[str, Any], overrides: Dict[str, Any] | None = None
) -> Tuple[OnlineReweighterConfig, DataConfig, Dict[str, Any]]:
    """Split a config dict into (reweighter cfg, data cfg, run params) and apply overrides."""
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    rw = {**_filter(raw.get("reweighter", {}), OnlineReweighterConfig)}
    data = {**_filter(raw.get("data", {}), DataConfig)}
    run = dict(raw.get("run", {}))

    # apply flat overrides to whichever section owns the key
    rw_names = {f.name for f in fields(OnlineReweighterConfig)}
    data_names = {f.name for f in fields(DataConfig)}
    for k, v in overrides.items():
        if k in rw_names:
            rw[k] = v
        elif k in data_names:
            data[k] = v
        else:
            run[k] = v

    cfg = OnlineReweighterConfig(**rw)
    dcfg = DataConfig(**data)
    return cfg, dcfg, run


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--n", type=int, default=None, help="reweighting granularity (trajectories/cell)")
    p.add_argument("--K", type=int, default=None, help="hypergradient / Neumann depth")
    p.add_argument("--steps", type=int, default=None, dest="n_beta_updates",
                   help="number of beta updates")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    p.add_argument("--outdir", type=str, default="runs", help="output directory")
    return p


def overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    keys = ["n", "K", "n_beta_updates", "seed", "device"]
    return {k: getattr(args, k, None) for k in keys}
