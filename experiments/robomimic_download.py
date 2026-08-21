"""Pre-download official RoboMimic low-dim datasets into the local cache.

    python -m experiments.robomimic_download --tasks lift can --types ph mh
    python -m experiments.robomimic_download --mix mh-tiers          # download what a mix needs

Files land under ``$SMOR_ROBOMIMIC_ROOT`` (default ``data/robomimic/{task}/{dtype}/``). The loader
also downloads on demand, so this is only for pre-fetching / offline prep.
"""

from __future__ import annotations

import argparse

from smor.data.robomimic import parse_mix
from smor.data.robomimic.registry import TASK_VARIANTS, ensure_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Download RoboMimic low-dim datasets.")
    p.add_argument("--tasks", nargs="+", default=["lift"], choices=sorted(TASK_VARIANTS))
    p.add_argument("--types", nargs="+", default=["ph", "mh"], choices=["ph", "mh", "mg"])
    p.add_argument("--mix", type=str, default=None,
                   help="instead of tasks/types, download exactly what this mix spec needs")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.mix:
        wanted = {(c.task, c.dtype) for c in parse_mix(args.mix)}
    else:
        wanted = {(t, d) for t in args.tasks for d in args.types if d in TASK_VARIANTS[t]}

    for task, dtype in sorted(wanted):
        path = ensure_dataset(task, dtype, root=args.root, force=args.force)
        print(f"  ready: {task}/{dtype} -> {path}")


if __name__ == "__main__":
    main()
