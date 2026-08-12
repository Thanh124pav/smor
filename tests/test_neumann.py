"""Truncated damped Neumann P_K tests (PLAN.md §1.2, Stage E, test_neumann)."""

import torch

from smor.reweighting.neumann import apply_pk
from smor.utils.checks import SafetyError


def _spd_quadratic(eigs):
    """Return (theta_param, loss_builder, H) for loss = 0.5 theta^T H theta."""
    d = len(eigs)
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    H = Q @ torch.diag(torch.tensor(eigs, dtype=torch.float64)) @ Q.t()
    H = 0.5 * (H + H.t())
    theta = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))

    def loss():
        return 0.5 * theta @ (H @ theta)

    return theta, loss, H


def _explicit_pk(H, K, eta, lam):
    d = H.shape[0]
    I = torch.eye(d, dtype=torch.float64)
    Ht = H + lam * I
    M = I - eta * Ht
    acc = torch.zeros_like(H)
    Mk = I.clone()
    for _ in range(K):
        acc = acc + Mk
        Mk = Mk @ M
    return eta * acc


def test_p1_is_eta_times_identity():
    theta, loss, H = _spd_quadratic([1.0, 2.0, 3.0])
    v = torch.randn(3, dtype=torch.float64)
    out = apply_pk(v, loss(), [theta], K=1, neumann_lr=0.1, damping=0.0)
    assert torch.allclose(out, 0.1 * v, atol=1e-12)


def test_apply_pk_matches_explicit_matrix():
    theta, loss, H = _spd_quadratic([0.5, 1.0, 2.0, 4.0])
    v = torch.randn(4, dtype=torch.float64)
    eta, lam = 0.15, 0.3
    for K in [1, 2, 4, 8]:
        got = apply_pk(v, loss(), [theta], K=K, neumann_lr=eta, damping=lam)
        expected = _explicit_pk(H, K, eta, lam) @ v
        assert torch.allclose(got, expected, atol=1e-9), (K, (got - expected).abs().max())


def test_pk_converges_to_damped_inverse():
    theta, loss, H = _spd_quadratic([0.5, 1.0, 2.0, 4.0])
    v = torch.randn(4, dtype=torch.float64)
    eta, lam = 0.2, 0.5
    Ht = H + lam * torch.eye(4, dtype=torch.float64)
    target = torch.linalg.solve(Ht, v)
    prev = None
    for K in [1, 2, 4, 8, 16, 32]:
        got = apply_pk(v, loss(), [theta], K=K, neumann_lr=eta, damping=lam)
        err = float((got - target).norm())
        if prev is not None:
            assert err <= prev + 1e-12
        prev = err
    assert prev < 1e-3  # converged to (H + lambda I)^{-1} v


def test_pk_converges_to_true_inverse_when_undamped():
    theta, loss, H = _spd_quadratic([1.0, 2.0, 4.0])
    v = torch.randn(3, dtype=torch.float64)
    eta = 0.2  # < 2 / max_eig
    target = torch.linalg.solve(H, v)
    errs = []
    for K in [1, 2, 4, 8, 32, 128]:
        got = apply_pk(v, loss(), [theta], K=K, neumann_lr=eta, damping=0.0)
        errs.append(float((got - target).norm()))
    for a, b in zip(errs[:-1], errs[1:]):
        assert b <= a + 1e-12
    assert errs[-1] < 1e-3


def test_damping_improves_stability_near_singularity():
    # Near-singular H: tiny eigenvalue makes H^{-1} ill-conditioned.
    theta, loss, H = _spd_quadratic([1e-3, 1.0, 3.0])
    v = torch.randn(3, dtype=torch.float64)
    eta = 0.3
    K = 40
    # undamped target is huge / slow to reach; damped target is bounded and reached fast.
    damped = apply_pk(v, loss(), [theta], K=K, neumann_lr=eta, damping=0.5)
    undamped = apply_pk(v, loss(), [theta], K=K, neumann_lr=eta, damping=0.0)
    damped_target = torch.linalg.solve(H + 0.5 * torch.eye(3, dtype=torch.float64), v)
    undamped_target = torch.linalg.solve(H, v)
    rel_damped = float((damped - damped_target).norm() / damped_target.norm())
    rel_undamped = float((undamped - undamped_target).norm() / undamped_target.norm())
    # damping yields far better convergence at the same (modest) K
    assert rel_damped < rel_undamped
    assert damped.norm() < undamped_target.norm()  # bounded estimate


def test_explosion_raises():
    theta, loss, H = _spd_quadratic([1.0, 2.0, 5.0])
    v = torch.randn(3, dtype=torch.float64)
    # eta too large -> |1 - eta*eig| > 1 for the largest eigenvalue -> diverges
    try:
        apply_pk(v, loss(), [theta], K=200, neumann_lr=1.0, damping=0.0)
        assert False, "expected SafetyError on diverging Neumann series"
    except SafetyError:
        pass


def test_group_hypergradient_k_gt_1_matches_explicit():
    from smor.reweighting.hypergradient import group_hypergradient, gather_flat_grad

    torch.manual_seed(5)
    d = 4
    theta = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))
    Hin = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    c_out = torch.randn(d, dtype=torch.float64)
    centers = [torch.randn(d, dtype=torch.float64) for _ in range(2)]
    eta, lam, K = 0.15, 0.2, 3

    inner = 0.5 * theta @ (Hin @ theta)            # Hessian == Hin
    group_losses = {j: 0.5 * ((theta - c) ** 2).sum() for j, c in enumerate(centers)}
    outer = 0.5 * ((theta - c_out) ** 2).sum()

    h = group_hypergradient(group_losses, outer, [theta], K=K,
                            neumann_lr=eta, damping=lam, inner_loss=inner)

    g_out = gather_flat_grad(0.5 * ((theta - c_out) ** 2).sum(), [theta], retain_graph=True)
    Pk = _explicit_pk(Hin, K, eta, lam)
    for j, c in enumerate(centers):
        g_j = gather_flat_grad(0.5 * ((theta - c) ** 2).sum(), [theta], retain_graph=True)
        ref = float(-torch.dot(g_out, Pk @ g_j))
        assert abs(h[j] - ref) < 1e-8


def test_rejects_k_less_than_one():
    theta, loss, H = _spd_quadratic([1.0, 2.0])
    v = torch.randn(2, dtype=torch.float64)
    try:
        apply_pk(v, loss(), [theta], K=0, neumann_lr=0.1, damping=0.0)
        assert False
    except ValueError:
        pass
