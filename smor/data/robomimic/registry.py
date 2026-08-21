"""RoboMimic dataset registry + downloader (no robomimic package required).

Datasets are the official low-dim HDF5 files hosted by the RoboMimic authors at
``downloads.cs.stanford.edu/downloads/rt_benchmark`` (the v1.4.1 release used by robomimic
>=0.3). We hard-code the URL scheme so downloading needs only ``requests`` — the heavy robomimic
/ robosuite / MuJoCo stack is only needed for the *optional* rollout evaluation, never for
loading + training.

URL scheme (verified against the host):
    {BASE}/{task}/{dtype}/{filename}
      ph, mh   -> low_dim_v141.hdf5
      mg       -> low_dim_sparse_v141.hdf5   (sparse-reward machine-generated variant)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

BASE_URL = "http://downloads.cs.stanford.edu/downloads/rt_benchmark"

# Which variants exist per task in the low-dim v1.4.1 release.
TASK_VARIANTS: Dict[str, Set[str]] = {
    "lift": {"ph", "mh", "mg"},
    "can": {"ph", "mh", "mg"},
    "square": {"ph", "mh"},
    "transport": {"ph", "mh"},
    "tool_hang": {"ph"},
}

# The quality tiers (HDF5 ``mask/`` filter keys) available inside a multi-human dataset.
MH_TIERS = ("better", "okay", "worse")


def dataset_filename(dtype: str) -> str:
    if dtype in ("ph", "mh"):
        return "low_dim_v141.hdf5"
    if dtype == "mg":
        # sparse-reward MG (matches robomimic's low-dim MG release); dense also exists.
        return "low_dim_sparse_v141.hdf5"
    raise ValueError(f"unknown robomimic dtype '{dtype}' (expected ph|mh|mg).")


def dataset_url(task: str, dtype: str) -> str:
    _validate(task, dtype)
    return f"{BASE_URL}/{task}/{dtype}/{dataset_filename(dtype)}"


def default_root() -> Path:
    """Where datasets are cached. Override with ``$SMOR_ROBOMIMIC_ROOT``."""
    env = os.environ.get("SMOR_ROBOMIMIC_ROOT")
    return Path(env) if env else Path("data/robomimic")


def local_path(task: str, dtype: str, root: str | Path | None = None) -> Path:
    root = Path(root) if root is not None else default_root()
    return root / task / dtype / dataset_filename(dtype)


def _validate(task: str, dtype: str) -> None:
    if task not in TASK_VARIANTS:
        raise ValueError(f"unknown robomimic task '{task}'. Known: {sorted(TASK_VARIANTS)}")
    if dtype not in TASK_VARIANTS[task]:
        raise ValueError(
            f"task '{task}' has no '{dtype}' variant. Available: {sorted(TASK_VARIANTS[task])}"
        )


def ensure_dataset(
    task: str,
    dtype: str,
    root: str | Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Return the local path to ``{task}/{dtype}``, downloading it (with resume) if missing."""
    _validate(task, dtype)
    dest = local_path(task, dtype, root)
    if dest.exists() and not force:
        return dest
    url = dataset_url(task, dtype)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download_with_resume(url, dest, quiet=quiet)
    return dest


def _download_with_resume(url: str, dest: Path, quiet: bool = False) -> None:
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    if not quiet:
        print(f"[robomimic] downloading {url}" + (f" (resume @ {existing}B)" if existing else ""))
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        done = existing
        with open(tmp, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if not quiet and total:
                    pct = 100.0 * done / total
                    print(f"\r[robomimic]   {done/1e6:8.1f} / {total/1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
    if not quiet:
        print()
    tmp.rename(dest)
