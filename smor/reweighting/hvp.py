"""Hessian-vector products via autograd (PLAN.md Stage E).

Never materialize the d x d Hessian. HVP(v) = d/dtheta (grad_theta L . v), computed with a
double backward pass. Works with a flat vector matching the concatenated parameter dimension.
"""

from __future__ import annotations

from typing import Iterable, List

import torch


def _flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _params_numel(params: List[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def hvp(
    loss: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
    vector: torch.Tensor,
    grad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``H @ vector`` where ``H = d^2 loss / dtheta^2``.

    Args:
        loss:   scalar with a graph connected to ``params``.
        params: parameters theta.
        vector: flat tensor of length ``sum(p.numel())`` (the concatenated param dim).
        grad:   optionally a precomputed flat first gradient built with ``create_graph=True``
                (reused across multiple HVPs at the same theta to save a backward pass).

    Returns a flat tensor of the same length as ``vector``.
    """
    params = list(params)
    n = _params_numel(params)
    vector = vector.reshape(-1)
    if vector.numel() != n:
        raise ValueError(f"vector has {vector.numel()} elems, expected {n}.")

    if grad is None:
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        grad = _flatten(grads)
    elif not grad.requires_grad:
        raise ValueError("precomputed grad must require_grad (build with create_graph=True).")

    dot = torch.dot(grad, vector.to(grad.dtype))
    hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
    flat = []
    for g, p in zip(hv, params):
        flat.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
    return torch.cat(flat)


def flat_grad_with_graph(
    loss: torch.Tensor, params: Iterable[torch.nn.Parameter]
) -> torch.Tensor:
    """First gradient flattened, keeping the graph so it can seed repeated HVPs."""
    params = list(params)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    return _flatten(grads)
