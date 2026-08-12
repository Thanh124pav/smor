"""HVP correctness on a tiny explicit Hessian (PLAN.md Stage E, test_hvp)."""

import torch

from smor.reweighting.hvp import hvp, flat_grad_with_graph


def test_hvp_matches_explicit_hessian_quadratic():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d, dtype=torch.float64)
    H = A @ A.t() + d * torch.eye(d, dtype=torch.float64)  # SPD, known Hessian
    b = torch.randn(d, dtype=torch.float64)

    theta = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))
    # loss = 0.5 theta^T H theta + b^T theta  =>  Hessian == H exactly
    loss = 0.5 * theta @ (H @ theta) + b @ theta

    for _ in range(5):
        v = torch.randn(d, dtype=torch.float64)
        got = hvp(loss, [theta], v)
        expected = H @ v
        assert torch.allclose(got, expected, atol=1e-9), (got - expected).abs().max()


def test_hvp_matches_autograd_hessian_nonlinear():
    torch.manual_seed(1)
    d = 4
    theta = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))
    W = torch.randn(d, d, dtype=torch.float64)

    def loss_fn(t):
        return (torch.tanh(W @ t) ** 2).sum()

    loss = loss_fn(theta)
    H_full = torch.autograd.functional.hessian(loss_fn, theta.detach())
    for _ in range(5):
        v = torch.randn(d, dtype=torch.float64)
        got = hvp(loss, [theta], v)
        assert torch.allclose(got, H_full @ v, atol=1e-8)


def test_hvp_reuses_precomputed_grad():
    torch.manual_seed(2)
    d = 5
    theta = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))
    H = torch.diag(torch.arange(1, d + 1, dtype=torch.float64))
    loss = 0.5 * theta @ (H @ theta)

    g = flat_grad_with_graph(loss, [theta])
    for _ in range(3):
        v = torch.randn(d, dtype=torch.float64)
        got = hvp(loss, [theta], v, grad=g)
        assert torch.allclose(got, H @ v, atol=1e-9)


def test_hvp_multi_param_shapes():
    torch.manual_seed(3)
    p1 = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    p2 = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float64))
    loss = (p1 ** 2).sum() + (p2 ** 3).sum()  # block-diagonal Hessian
    n = p1.numel() + p2.numel()
    v = torch.randn(n, dtype=torch.float64)
    got = hvp(loss, [p1, p2], v)
    # Hessian: diag(2,2,2) for p1 ; diag(6*p2_ij) for p2
    diag = torch.cat([2 * torch.ones(3, dtype=torch.float64),
                      (6 * p2.detach()).reshape(-1)])
    assert torch.allclose(got, diag * v, atol=1e-9)
