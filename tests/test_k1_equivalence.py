"""K=1 hypergradient equivalence (PLAN.md Stage D, test_k1_equivalence).

Verifies that the K=1 group hypergradient reduces to the gradient-alignment formula
``h_j = -eta_h * g_out^T g_j`` and matches the true one-step pseudo-update beta-gradient
in the small-step limit.
"""

import torch

from smor.reweighting.hypergradient import gather_flat_grad, group_hypergradient


def _toy(seed=0, dim=4, n_groups=3):
    torch.manual_seed(seed)
    theta = torch.nn.Parameter(torch.randn(dim, dtype=torch.float64))
    centers = [torch.randn(dim, dtype=torch.float64) for _ in range(n_groups)]
    c_out = torch.randn(dim, dtype=torch.float64)
    return theta, centers, c_out


def _group_losses(theta, centers):
    return {j: 0.5 * ((theta - c) ** 2).sum() for j, c in enumerate(centers)}


def _outer_loss(theta, c_out):
    return 0.5 * ((theta - c_out) ** 2).sum()


def test_k1_matches_gradient_alignment_formula():
    theta, centers, c_out = _toy()
    eta_h = 0.137
    gl = _group_losses(theta, centers)
    ol = _outer_loss(theta, c_out)

    h = group_hypergradient(gl, ol, [theta], K=1, neumann_lr=eta_h)

    # Manual reference: -eta_h * g_out . g_j
    g_out = gather_flat_grad(_outer_loss(theta, c_out), [theta], retain_graph=True)
    for j, c in enumerate(centers):
        g_j = gather_flat_grad(0.5 * ((theta - c) ** 2).sum(), [theta], retain_graph=True)
        ref = float(-eta_h * torch.dot(g_out, g_j))
        assert abs(h[j] - ref) < 1e-9


def test_k1_approaches_true_one_step_gradient_as_eta_shrinks():
    theta, centers, c_out = _toy(seed=1)

    def true_pseudo_beta_grad(eta):
        beta = torch.ones(len(centers), dtype=torch.float64, requires_grad=True)
        # inner gradient of L_in = sum_j beta_j L_j at theta (theta fixed leaf)
        gl = _group_losses(theta, centers)
        g_in = None
        for j, Lj in gl.items():
            gj = gather_flat_grad(Lj, [theta], create_graph=True, retain_graph=True)
            term = beta[j] * gj
            g_in = term if g_in is None else g_in + term
        theta_prime = theta.detach().flatten() - eta * g_in
        L = 0.5 * ((theta_prime - c_out) ** 2).sum()
        (grad_beta,) = torch.autograd.grad(L, beta)
        return grad_beta.detach()

    prev_err = None
    for eta in [0.1, 0.01, 0.001]:
        est = group_hypergradient(_group_losses(theta, centers),
                                  _outer_loss(theta, c_out), [theta],
                                  K=1, neumann_lr=eta)
        est_vec = torch.tensor([est[j] for j in range(len(centers))], dtype=torch.float64)
        true_vec = true_pseudo_beta_grad(eta)
        # both scale ~ eta; compare direction-normalized error
        err = float((est_vec - true_vec).norm() / (true_vec.norm() + 1e-12))
        if prev_err is not None:
            assert err < prev_err + 1e-9
        prev_err = err
    assert prev_err < 1e-2  # tight agreement at the smallest step


def test_helpful_group_has_negative_hypergradient():
    # Put a group's center exactly at the outer target: its gradient points to reducing L_out.
    theta = torch.nn.Parameter(torch.tensor([2.0, 0.0], dtype=torch.float64))
    c_out = torch.tensor([0.0, 0.0], dtype=torch.float64)
    centers = [c_out.clone(), torch.tensor([5.0, 5.0], dtype=torch.float64)]
    h = group_hypergradient(_group_losses(theta, centers), _outer_loss(theta, c_out),
                            [theta], K=1, neumann_lr=0.1)
    assert h[0] < 0  # aligned/helpful
    # group 1 pulls away from the target -> unhelpful
    assert h[1] > h[0]


def test_rejects_k_less_than_one():
    theta, centers, c_out = _toy()
    try:
        group_hypergradient(_group_losses(theta, centers), _outer_loss(theta, c_out),
                            [theta], K=0)
        assert False, "expected ValueError for K<1"
    except ValueError:
        pass
