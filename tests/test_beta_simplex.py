"""Tests for the simplex beta parameterization (PLAN.md §5)."""

import numpy as np
import torch

from smor.reweighting.beta import BetaDistribution, project_to_simplex_with_floor
from smor.utils.checks import SafetyError


def test_init_is_uniform_simplex():
    b = BetaDistribution(num_groups=5)
    beta = b.as_numpy()
    assert np.allclose(beta, 1.0 / 5)
    assert abs(beta.sum() - 1.0) < 1e-9
    assert (beta >= 0).all()


def test_stays_on_simplex_after_updates_exp_grad():
    b = BetaDistribution(num_groups=6, update="exp_grad", lr=1.0, floor=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        h = rng.normal(size=6)
        info = b.step(h)
        beta = info["beta"]
        assert abs(beta.sum() - 1.0) < 1e-6
        assert (beta >= 1e-3 - 1e-9).all()
        assert np.isfinite(beta).all()


def test_stays_on_simplex_after_updates_logit():
    b = BetaDistribution(num_groups=4, update="logit", lr=0.5, temperature=1.0, floor=1e-4)
    rng = np.random.default_rng(1)
    for _ in range(50):
        info = b.step(rng.normal(size=4))
        assert abs(info["beta"].sum() - 1.0) < 1e-6
        assert (info["beta"] >= 0).all()


def test_floor_prevents_zero_mass():
    b = BetaDistribution(num_groups=3, update="exp_grad", lr=5.0, floor=0.05)
    # Hammer group 0 with large positive hypergrad -> mass flees, but floor holds.
    for _ in range(100):
        b.step(np.array([10.0, 0.0, 0.0]))
    beta = b.as_numpy()
    assert beta[0] >= 0.05 - 1e-9
    assert abs(beta.sum() - 1.0) < 1e-6


def test_helpful_group_gains_mass():
    # Negative hypergradient on group 1 => exp_grad increases its weight.
    b = BetaDistribution(num_groups=3, update="exp_grad", lr=0.5, floor=1e-4)
    start = b.as_numpy()[1]
    for _ in range(10):
        b.step(np.array([0.0, -1.0, 0.0]))
    assert b.as_numpy()[1] > start


def test_project_with_floor_properties():
    beta = torch.tensor([0.0, 0.01, 0.99], dtype=torch.float64)
    out = project_to_simplex_with_floor(beta, floor=0.05)
    assert abs(float(out.sum()) - 1.0) < 1e-9
    assert (out.numpy() >= 0.05 - 1e-9).all()


def test_entropy_reg_pulls_toward_uniform():
    b = BetaDistribution(num_groups=4, update="exp_grad", lr=0.5,
                         entropy_reg=1.0, floor=1e-4, init=np.array([0.7, 0.1, 0.1, 0.1]))
    ent0 = b.entropy()
    for _ in range(30):
        b.step(np.zeros(4))
    # entropy should rise toward log(4) under entropy regularization with zero task grad
    assert b.entropy() > ent0


def test_nan_hypergradient_fails_loudly():
    b = BetaDistribution(num_groups=3)
    try:
        b.step(np.array([np.nan, 0.0, 0.0]))
        assert False, "expected SafetyError on NaN hypergradient"
    except SafetyError:
        pass


def test_weight_dict_alignment():
    b = BetaDistribution(num_groups=3)
    wd = b.weight_dict([10, 20, 30])
    assert set(wd.keys()) == {10, 20, 30}
    assert abs(sum(wd.values()) - 1.0) < 1e-9
