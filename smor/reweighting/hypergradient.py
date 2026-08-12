"""Group hypergradients (PLAN.md Stage D / Stage E, §1.2).

For the bilevel objective with inner loss ``L_in = sum_j beta_j L_j(theta)`` and outer loss
``L_out(theta)``, the response of the outer loss to upweighting group ``j`` is estimated as

    h_j^{(K)} = - g_out^T P_K g_j

where ``g_out = grad_theta L_out``, ``g_j = grad_theta L_j`` and ``P_K`` is the truncated,
damped Neumann operator (see :mod:`smor.reweighting.neumann`). For ``K=1``, ``P_1 = eta_h * I``
so ``h_j^{(1)} = -eta_h * g_out^T g_j`` — the one-step / gradient-alignment / CAIL-style
common-backbone baseline.

A negative ``h_j`` means upweighting group ``j`` lowers the outer loss (helpful group).
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch

from smor.utils.checks import check_finite


def gather_flat_grad(
    scalar: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
    create_graph: bool = False,
    retain_graph: Optional[bool] = None,
) -> torch.Tensor:
    """Flattened gradient of ``scalar`` w.r.t. ``params`` (missing grads -> zeros)."""
    params = list(params)
    if retain_graph is None:
        retain_graph = create_graph
    grads = torch.autograd.grad(
        scalar, params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat = []
    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.reshape(-1))
    return torch.cat(flat)


def one_step_hypergradient(g_out: torch.Tensor, g_j: torch.Tensor, neumann_lr: float) -> float:
    """K=1 hypergradient: ``-eta_h * g_out^T g_j`` (P_1 = eta_h I)."""
    return float(-neumann_lr * torch.dot(g_out.detach().flatten(), g_j.detach().flatten()))


def group_hypergradient(
    group_losses: Mapping[int, torch.Tensor],
    outer_loss: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
    K: int = 1,
    neumann_lr: float = 0.1,
    damping: float = 1.0,
    inner_loss: Optional[torch.Tensor] = None,
    hvp_clip: float = 0.0,
    fallback_k1_on_invalid: bool = True,
    return_meta: bool = False,
) -> Dict[int, float] | Tuple[Dict[int, float], dict]:
    """Estimate ``h_j^{(K)}`` for every group.

    Args:
        group_losses: ``{group_id: differentiable scalar L_j(theta)}``.
        outer_loss:   differentiable scalar ``L_out(theta)``.
        params:       theta parameters (shared by all losses).
        K:            Neumann truncation depth (>= 1).
        neumann_lr:   ``eta_h``.
        damping:      ``lambda`` in ``(H + lambda I)``.
        inner_loss:   differentiable ``L_in(theta, beta)`` used for HVPs (required if K>1).
        hvp_clip:     if > 0, clip the norm of ``P_K g_j`` (stability).
        fallback_k1_on_invalid: on a non-finite/diverged curvature update, warn and fall back
                                to the K=1 estimate for that group (PLAN.md §14).
        return_meta:  if True, also return a dict with ``hvp_count``, ``n_fallbacks`` and
                      per-group ``pk_norm``.

    Returns ``{group_id: h_j}`` (python floats), or ``(hgrad, meta)`` if ``return_meta``.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    params = list(params)
    from smor.utils.checks import SafetyError  # local import to avoid cycles

    g_out = gather_flat_grad(outer_loss, params, create_graph=False, retain_graph=True).detach()
    check_finite(g_out, "g_out")

    out: Dict[int, float] = {}
    pk_norms: Dict[int, float] = {}
    n_fallbacks = 0
    hvp_count = 0
    for gid, L_j in group_losses.items():
        g_j = gather_flat_grad(L_j, params, create_graph=False, retain_graph=True).detach()
        check_finite(g_j, f"g_{gid}")
        if K == 1:
            v = neumann_lr * g_j
        else:
            if inner_loss is None:
                raise ValueError("inner_loss is required for K > 1.")
            from smor.reweighting.neumann import apply_pk  # lazy: avoids hard dep at K=1

            try:
                v = apply_pk(g_j, inner_loss, params, K, neumann_lr, damping)
                if not torch.isfinite(v).all():
                    raise SafetyError("non-finite P_K g")
                hvp_count += (K - 1)
            except SafetyError:
                if not fallback_k1_on_invalid:
                    raise
                warnings.warn(
                    f"group {gid}: curvature update invalid; falling back to K=1 (§14).",
                    RuntimeWarning, stacklevel=2,
                )
                v = neumann_lr * g_j
                n_fallbacks += 1
        if hvp_clip and hvp_clip > 0:
            norm = v.norm()
            if float(norm) > hvp_clip:
                v = v * (hvp_clip / (float(norm) + 1e-12))
        pk_norms[gid] = float(v.norm())
        out[gid] = float(-torch.dot(g_out, v))

    if return_meta:
        meta = {
            "hvp_count": hvp_count,
            "n_fallbacks": n_fallbacks,
            "pk_norm": pk_norms,
            "g_out_norm": float(g_out.norm()),
        }
        return out, meta
    return out
