"""Sampler tests (spec §42): budget conservation, mixture correctness, no replacement."""

import numpy as np

from smor.scaling.sampler import allocate_counts, sample_dataset


def test_budget_conservation():
    for B in [50, 100, 137, 400, 1600]:
        for p in [0.0, 0.2, 0.3333, 0.5, 0.81, 1.0]:
            counts = allocate_counts(B, [p, 1 - p])
            assert counts.sum() == B
            assert (counts >= 0).all()


def test_mixture_correctness():
    counts = allocate_counts(100, [0.3, 0.7])
    assert list(counts) == [30, 70]


def test_three_source_conservation():
    counts = allocate_counts(100, [0.2, 0.3, 0.5])
    assert counts.sum() == 100
    assert list(counts) == [20, 30, 50]


def test_no_replacement_and_pool_ids():
    pools = {"A": 500, "B": 500}
    s = sample_dataset(pools, budget=200, mixture=[0.4, 0.6], seed=0)
    assert s.source_counts.sum() == 200
    assert list(s.source_counts) == [80, 120]
    for sid, ids in s.source_ids.items():
        assert len(ids) == len(set(ids.tolist()))  # unique -> no replacement


def test_pool_too_small_raises():
    import pytest
    pools = {"A": 10, "B": 500}
    with pytest.raises(ValueError):
        sample_dataset(pools, budget=200, mixture=[0.9, 0.1], seed=0)  # needs 180 from pool of 10


def test_costs_allocation():
    # cost 2 for source A halves its count per unit budget
    counts = allocate_counts(100, [0.5, 0.5], costs=[2.0, 1.0])
    assert counts.sum() <= 100  # cost-weighted floor
    assert counts[0] * 2 + counts[1] * 1 <= 100 + 1
