"""Truncated, damped Neumann operator P_K (PLAN.md §1.2, Stage E).

With damped curvature ``H~ = H + lambda I`` and step ``eta_h``,

    P_K = eta_h * sum_{k=0}^{K-1} (I - eta_h * H~)^k

approximates ``H~^{-1}`` as K grows (for eta_h small enough that the series contracts).
Computed HVP-only via the recursion

    v_0 = g
    v_{k+1} = v_k - eta_h * (HVP(v_k) + lambda * v_k)
    P_K g   = eta_h * sum_{k=0}^{K-1} v_k

so no d x d matrix is ever formed. For K=1, P_1 g = eta_h * g.
"""

from __future__ import annotations

from typing import Iterable

import torch

from smor.reweighting.hvp import hvp, flat_grad_with_graph
from smor.utils.checks import SafetyError


def estimate_lambda_max(
    inner_loss: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
    damping: float = 0.0,
    n_iters: int = 10,
    seed: int = 0,
) -> float:
    """Estimate the largest eigenvalue of the damped Hessian ``H + damping*I`` via power
    iteration on HVPs (no matrix formed). Used to auto-scale the Neumann step so the series
    contracts: ``eta_h < 2 / lambda_max``.
    """
    params = list(params)
    n = sum(p.numel() for p in params)
    dev = params[0].device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n, generator=gen).to(dev, params[0].dtype)
    v = v / (v.norm() + 1e-12)
    grad = flat_grad_with_graph(inner_loss, params)  # reused across iterations (fixed theta)
    lam = 0.0
    for _ in range(max(1, n_iters)):
        Hv = hvp(inner_loss, params, v, grad=grad) + damping * v
        lam = float(torch.dot(v, Hv))          # Rayleigh quotient (v is unit-norm)
        nrm = float(Hv.norm())
        if nrm < 1e-12:
            break
        v = Hv / nrm
    return abs(lam)


def apply_pk(
    vector: torch.Tensor,
    inner_loss: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
    K: int,
    neumann_lr: float,
    damping: float,
    explode_limit: float = 1e8,
) -> torch.Tensor:
    """Apply the truncated damped Neumann operator to ``vector``.

    Args:
        vector:     flat seed vector ``g`` (length = concatenated param dim).
        inner_loss: differentiable ``L_in(theta)`` whose Hessian defines ``H``.
        params:     theta parameters.
        K:          truncation depth (>= 1).
        neumann_lr: ``eta_h``.
        damping:    ``lambda`` (>= 0).
        explode_limit: raise if an iterate norm exceeds this (diverging series).

    Returns ``P_K @ vector`` as a flat tensor.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    params = list(params)
    v = vector.reshape(-1).detach().clone()
    accum = v.clone()  # k = 0 term

    # Reuse a single graph-carrying first gradient for all HVPs at this theta.
    grad = flat_grad_with_graph(inner_loss, params) if K > 1 else None

    for _ in range(1, K):
        Hv = hvp(inner_loss, params, v, grad=grad)
        v = v - neumann_lr * (Hv + damping * v)
        vnorm = float(v.norm())
        if not torch.isfinite(v).all() or vnorm >= explode_limit:
            raise SafetyError(
                f"Neumann iterate diverged (norm={vnorm:.3e}); reduce neumann_lr or raise damping."
            )
        accum = accum + v

    return neumann_lr * accum
