"""Correlation / agreement metrics for hypergradient-vs-realized-utility comparison (§8)."""

from __future__ import annotations

import numpy as np


def _v(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def sign_accuracy(pred, target) -> float:
    """Fraction of entries where sign(pred) == sign(target) (zeros treated as matching)."""
    p, t = _v(pred), _v(target)
    return float(np.mean(np.sign(p) == np.sign(t)))


def pearson(pred, target) -> float:
    p, t = _v(pred), _v(target)
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), dependency-free."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # resolve ties by averaging
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman(pred, target) -> float:
    p, t = _v(pred), _v(target)
    if len(p) < 2:
        return float("nan")
    return pearson(_rankdata(p), _rankdata(t))


def cosine_similarity(pred, target) -> float:
    p, t = _v(pred), _v(target)
    denom = (np.linalg.norm(p) * np.linalg.norm(t))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(p, t) / denom)
