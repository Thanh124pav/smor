"""Tests for the RoboMimic official-dataset loader + ragged TrajectoryDataset.

The TrajectoryDataset unit tests always run. The tests that touch real RoboMimic HDF5 files are
skipped unless the ``lift/mh`` low-dim dataset is already present locally (they never download).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from smor.data.trajectory_dataset import TrajectoryDataset
from smor.data.robomimic.registry import local_path
from smor.data.robomimic.spec import PRESETS, parse_mix


def _have(task="lift", dtype="mh") -> bool:
    try:
        import h5py  # noqa: F401
    except Exception:
        return False
    return local_path(task, dtype).exists()


# --- TrajectoryDataset (pure, no data) --------------------------------------

def _toy_trajs():
    # three variable-length trajectories, 2-dim obs, 1-dim act
    return [
        (torch.zeros(3, 2), torch.zeros(3, 1)),
        (torch.ones(5, 2), torch.ones(5, 1)),
        (torch.full((2, 2), 2.0), torch.full((2, 1), 2.0)),
    ]


def test_trajectory_dataset_shapes_and_flatten():
    ds = TrajectoryDataset.from_trajectories(_toy_trajs(), fidelity=[0, 1, 1])
    assert ds.num_trajectories == 3
    assert ds.num_transitions == 10
    assert ds.obs_dim == 2 and ds.act_dim == 1
    o, a, tid = ds.flatten()
    assert o.shape == (10, 2) and a.shape == (10, 1) and tid.shape == (10,)
    assert int(tid.min()) == 0 and int(tid.max()) == 2
    assert np.array_equal(ds.traj_lengths(), np.array([3, 5, 2]))
    assert np.array_equal(ds.fidelity_labels(), np.array([0, 1, 1]))


def test_trajectory_dataset_validates_fidelity_count():
    with pytest.raises(ValueError):
        TrajectoryDataset.from_trajectories(_toy_trajs(), fidelity=[0, 1])  # wrong length


def test_trajectory_dataset_subset_reindexes():
    ds = TrajectoryDataset.from_trajectories(_toy_trajs(), fidelity=[0, 1, 2])
    sub = ds.subset([0, 2])
    assert sub.num_trajectories == 2
    _, _, tid = sub.flatten()
    assert sorted(set(tid.tolist())) == [0, 1]
    assert np.array_equal(sub.fidelity_labels(), np.array([0, 2]))


def test_trajectory_dataset_roundtrip(tmp_path):
    ds = TrajectoryDataset.from_trajectories(_toy_trajs(), fidelity=[0, 1, 1])
    p = ds.save(tmp_path / "traj.pt")
    ds2 = TrajectoryDataset.load(p)
    assert torch.equal(ds.obs, ds2.obs) and torch.equal(ds.fidelity, ds2.fidelity)


# --- mix-spec parsing (pure) ------------------------------------------------

def test_parse_mix_preset():
    comps = parse_mix("mh-tiers")
    assert [c.tier for c in comps] == ["better", "okay", "worse"]
    assert comps[0].target is True
    assert comps[0].resolved_quality() > comps[-1].resolved_quality()


def test_parse_mix_dsl_and_target_star():
    comps = parse_mix("lift:ph*,lift:mh:worse,lift:mg:mg_fail:150")
    assert comps[0].dtype == "ph" and comps[0].target is True
    assert comps[1].tier == "worse"
    assert comps[2].tier == "mg_fail" and comps[2].n == 150


def test_parse_mix_task_override():
    comps = parse_mix("mh-tiers", task="can")
    assert all(c.task == "can" for c in comps)


def test_all_presets_parse():
    for name in PRESETS:
        assert len(parse_mix(name)) >= 1


# --- real RoboMimic data (skipped if not downloaded) ------------------------

@pytest.mark.skipif(not _have("lift", "mh"), reason="lift/mh RoboMimic dataset not downloaded")
def test_load_robomimic_mh_tiers():
    from smor.data.robomimic import load_robomimic_mix
    from smor.reweighting.grouping import make_groups

    train, val, names, quality = load_robomimic_mix(parse_mix("mh-tiers"), val_frac=0.1, seed=0)
    assert train.obs_dim == 19 and train.act_dim == 7
    assert len(names) == 3 and len(quality) == 3
    assert quality[0] > quality[1] > quality[2]
    # every fidelity tier is represented in the training set
    assert set(np.unique(train.fidelity_labels())) == {0, 1, 2}
    assert val.num_trajectories >= 1
    ga = make_groups(train.fidelity_labels(), group_size=8, seed=0, whole_fidelity=True)
    assert ga.num_groups == 3


@pytest.mark.skipif(not _have("lift", "ph"), reason="lift/ph RoboMimic dataset not downloaded")
def test_multisource_shared_states_shapes_and_calibration():
    from smor.data.robomimic.multisource import (
        DEFAULT_DEVICE_PROFILES, apply_device_calibration, make_robomimic_multisource,
    )

    n_dev = len(DEFAULT_DEVICE_PROFILES)
    train, val, names, quality = make_robomimic_multisource(
        task="lift", dtype="ph", val_frac=0.1, seed=0, shared_states=True)
    assert len(names) == n_dev and train.obs_dim == 19 and train.act_dim == 7
    # shared states => every device relabels the same base demos => equal group trajectory counts
    counts = np.bincount(train.fidelity_labels(), minlength=n_dev)
    assert set(counts.tolist()) == {int(counts[0])}  # all devices have the same #trajectories
    # calibration actually changes the actions (systematic bias), gripper channel untouched
    a = np.random.default_rng(0).uniform(-0.5, 0.5, size=(20, 7)).astype("float32")
    from smor.data.robomimic.multisource import DeviceProfile
    cal = apply_device_calibration(a, DeviceProfile("d", rot_deg=20, gain=1.3, noise=0.0), np.random.default_rng(0))
    assert not np.allclose(cal[:, 0:6], a[:, 0:6])
    assert np.allclose(cal[:, 6], np.clip(a[:, 6], -1, 1))  # gripper preserved


@pytest.mark.skipif(not (_have("lift", "ph") and _have("lift", "mg")),
                    reason="lift ph+mg RoboMimic datasets not downloaded")
def test_combined_good_plus_poison():
    from smor.data.robomimic.multisource import DEFAULT_DEVICE_PROFILES, make_robomimic_combined

    n_good = len(DEFAULT_DEVICE_PROFILES)
    train, val, names, quality = make_robomimic_combined(
        task="lift", base_dtype="ph", poison_n=50, val_frac=0.1, seed=0)
    assert len(names) == n_good + 1  # goods + one poison source
    assert names[-1] == "mg_fail"
    # poison is labelled far lower quality than any good source
    assert quality[-1] < min(quality[:n_good])
    counts = np.bincount(train.fidelity_labels())
    assert counts[-1] == 50  # poison capped
    assert train.obs_dim == 19 and train.act_dim == 7
