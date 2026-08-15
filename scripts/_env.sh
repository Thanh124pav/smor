#!/usr/bin/env bash
# Shared setup for SMOR experiment scripts: activate the conda env, cd to repo root,
# and pick a device. Source this from other scripts:  source "$(dirname "$0")/_env.sh"
set -euo pipefail

# --- conda env (Python 3.12, torch 2.11 + CUDA) ---
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-deeplearning}"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# --- repo root (parent of scripts/) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- device: DEVICE=cuda|cpu|auto (default auto -> cuda if available) ---
DEVICE="${DEVICE:-auto}"
OUTDIR="${OUTDIR:-results}"

echo "[env] conda=$CONDA_ENV  repo=$REPO_ROOT  device=$DEVICE  outdir=$OUTDIR"
python - <<'PY'
import torch
print(f"[env] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}"
      + (f"  gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY
