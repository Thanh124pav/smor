"""Toy bilevel sanity checks on the BC learner (PLAN.md Stage D tests).

* upweighting a helpful (expert) group lowers the outer (validation) loss;
* the expert group receives a more-negative K=1 hypergradient than the noisy group;
* beta receives gradients and moves; nothing goes NaN.
"""

import numpy as np
import torch

from smor.envs.demos import make_two_fidelity_dataset
from smor.reweighting.grouping import make_groups
from smor.reweighting.beta import BetaDistribution
from smor.reweighting.hypergradient import group_hypergradient
from smor.learners.bc import BCLearner
from smor.utils.seeding import seed_everything


def _build(seed=0):
    seed_everything(seed)
    train = make_two_fidelity_dataset(n_expert=30, n_noisy=30, horizon=30, seed=seed)
    val = make_two_fidelity_dataset(n_expert=20, n_noisy=0, horizon=30, seed=seed + 7)
    ga = make_groups(train.fidelity_labels(), group_size=999, seed=seed, whole_fidelity=True)
    learner = BCLearner(train, ga, hidden=(32, 32), lr=1e-2, batch_size=128,
                        device="cpu", val_data=val, seed=seed)
    # group id -> fidelity (0 expert, 1 noisy)
    fid = {j: int(f) for j, f in enumerate(ga.group_fidelity)}
    return learner, ga, fid


def _train_fixed_beta(seed, weights, steps=250):
    learner, ga, fid = _build(seed)
    gids = list(range(ga.num_groups))
    for _ in range(steps):
        batches = learner.sample_batches(gids)
        learner.train_step(weights, batches)
    return float(learner.validation_loss().detach())


def test_upweighting_expert_lowers_outer_loss():
    # map fidelity->group id
    _, ga, fid = _build(0)
    expert_gid = [g for g, f in fid.items() if f == 0][0]
    noisy_gid = [g for g, f in fid.items() if f == 1][0]

    expert_weights = {expert_gid: 1.0, noisy_gid: 0.0}
    noisy_weights = {expert_gid: 0.0, noisy_gid: 1.0}

    val_expert = _train_fixed_beta(0, expert_weights)
    val_noisy = _train_fixed_beta(0, noisy_weights)
    assert np.isfinite(val_expert) and np.isfinite(val_noisy)
    assert val_expert < val_noisy  # upweighting the helpful group lowers outer loss


def test_expert_group_has_more_negative_hypergradient():
    learner, ga, fid = _build(1)
    gids = list(range(ga.num_groups))
    # do a little training so gradients are informative
    w = {g: 1.0 / ga.num_groups for g in gids}
    for _ in range(50):
        learner.train_step(w, learner.sample_batches(gids))

    # Single-batch hypergradients are noisy; compare the *expected* hypergradient by
    # averaging over several resampled batches (the mechanism is a distributional claim).
    acc = {g: [] for g in gids}
    for _ in range(25):
        batches = learner.sample_batches(gids)
        h = group_hypergradient(learner.per_group_losses(batches), learner.validation_loss(),
                                learner.parameters_for_reweighting(), K=1, neumann_lr=0.1)
        for g in gids:
            acc[g].append(h[g])
    mean_h = {g: float(np.mean(v)) for g, v in acc.items()}
    assert all(np.isfinite(v) for v in mean_h.values())
    expert_gid = [g for g, f in fid.items() if f == 0][0]
    noisy_gid = [g for g, f in fid.items() if f == 1][0]
    assert mean_h[expert_gid] < mean_h[noisy_gid]


def test_beta_receives_gradient_and_moves():
    learner, ga, fid = _build(2)
    gids = list(range(ga.num_groups))
    beta = BetaDistribution(ga.num_groups, update="exp_grad", lr=0.5, device="cpu")
    start = beta.as_numpy().copy()

    w = {g: 1.0 / ga.num_groups for g in gids}
    for _ in range(30):
        learner.train_step(w, learner.sample_batches(gids))

    batches = learner.sample_batches(gids)
    h = group_hypergradient(learner.per_group_losses(batches), learner.validation_loss(),
                            learner.parameters_for_reweighting(), K=1, neumann_lr=0.1)
    h_vec = [h[g] for g in gids]
    info = beta.step(h_vec)
    assert np.isfinite(info["beta"]).all()
    assert info["beta_l1_movement"] > 0
    assert not np.allclose(beta.as_numpy(), start)
