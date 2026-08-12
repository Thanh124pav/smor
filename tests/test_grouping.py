"""Tests for fixed trajectory grouping (PLAN.md Stage C)."""

import numpy as np
import pytest

from smor.reweighting.grouping import make_groups


def _labels(n_expert=20, n_noisy=13):
    return np.array([0] * n_expert + [1] * n_noisy)


def test_every_trajectory_assigned_once():
    labels = _labels()
    ga = make_groups(labels, group_size=8, seed=0)
    # union of members equals 0..N-1, no repeats
    all_members = np.concatenate(ga.members)
    assert np.array_equal(np.sort(all_members), np.arange(labels.size))
    assert ga.group_id.min() >= 0
    assert ga.num_trajectories == labels.size


def test_no_group_crosses_fidelity():
    labels = _labels()
    ga = make_groups(labels, group_size=8, seed=1)
    for j, mem in enumerate(ga.members):
        fids = set(labels[mem].tolist())
        assert len(fids) == 1
        assert ga.group_fidelity[j] in fids


def test_group_sizes_respect_n_with_uneven_tail():
    labels = _labels(n_expert=20, n_noisy=13)
    ga = make_groups(labels, group_size=8, seed=2)
    # expert: 20 -> ceil(20/8)=3 chunks (sizes ~7,7,6); noisy: 13 -> 2 chunks (7,6)
    assert ga.num_groups == 3 + 2
    assert ga.group_sizes.max() <= 8
    assert ga.group_sizes.sum() == labels.size


def test_seeded_reproducible_and_immutable():
    labels = _labels()
    a = make_groups(labels, group_size=8, seed=7)
    b = make_groups(labels, group_size=8, seed=7)
    assert np.array_equal(a.group_id, b.group_id)
    assert np.array_equal(a.group_fidelity, b.group_fidelity)
    # different seed -> generally different assignment
    c = make_groups(labels, group_size=8, seed=8)
    assert not np.array_equal(a.group_id, c.group_id)


def test_n_equals_one_is_per_trajectory():
    labels = _labels(5, 5)
    ga = make_groups(labels, group_size=1, seed=0)
    assert ga.num_groups == labels.size
    assert np.all(ga.group_sizes == 1)


def test_whole_fidelity_one_group_per_level():
    labels = _labels(20, 13)
    ga = make_groups(labels, group_size=999, seed=0, whole_fidelity=True)
    assert ga.num_groups == 2
    assert set(ga.group_fidelity.tolist()) == {0, 1}
    assert sorted(ga.group_sizes.tolist()) == [13, 20]


def test_rejects_invalid_n():
    with pytest.raises(ValueError):
        make_groups(_labels(), group_size=0, seed=0)


def test_fidelity_to_groups_mapping():
    labels = _labels(20, 13)
    ga = make_groups(labels, group_size=8, seed=0)
    mapping = ga.fidelity_to_groups()
    assert set(mapping.keys()) == {0, 1}
    assert sum(len(v) for v in mapping.values()) == ga.num_groups
