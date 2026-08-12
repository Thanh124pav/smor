"""Simplex beta parameterization and updates (PLAN.md §5).

beta lives on the probability simplex (beta_j >= 0, sum_j beta_j = 1). Two update rules:

    exp_grad : exponentiated-gradient / mirror descent
               beta_j <- beta_j * exp(-eta * h_j), then renormalize
    logit    : store logits z, beta = softmax(z / temperature),
               z <- z - eta * h  (gradient step in logit space)

Both apply a floor epsilon (to prevent irreversible zero mass), optional entropy
regularization, temperature, and score clipping. Raw hypergradients and resulting beta
are both available for logging.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from smor.utils.checks import check_finite, check_simplex


def _to_tensor(x, device, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def project_to_simplex_with_floor(beta: torch.Tensor, floor: float) -> torch.Tensor:
    """Clamp to ``floor`` and renormalize so entries sum to 1 while staying >= floor.

    Assumes ``M * floor < 1``. Uses a simple clamp-and-renormalize of the *excess* mass
    above the floor, which keeps every entry >= floor and the total at 1.
    """
    M = beta.shape[-1]
    if floor <= 0.0:
        return beta / beta.sum(dim=-1, keepdim=True)
    if M * floor >= 1.0:
        raise ValueError(f"beta_floor={floor} too large for M={M} groups (M*floor>=1).")
    beta = torch.clamp(beta, min=0.0)
    excess = beta - floor
    excess = torch.clamp(excess, min=0.0)
    denom = excess.sum(dim=-1, keepdim=True)
    # If all mass at/below floor, fall back to uniform excess.
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    scale = (1.0 - M * floor)
    return floor + scale * excess / denom


class BetaDistribution:
    """Learnable simplex weights over ``M`` groups.

    Maintains ``beta`` (probabilities) and, for the ``logit`` rule, backing logits ``z``.
    """

    def __init__(
        self,
        num_groups: int,
        update: str = "exp_grad",
        lr: float = 0.5,
        temperature: float = 1.0,
        floor: float = 1e-4,
        entropy_reg: float = 0.0,
        score_clip: float = 0.0,
        standardize: bool = True,
        device: str | torch.device = "cpu",
        init: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float64,
    ):
        if num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if update not in {"exp_grad", "logit"}:
            raise ValueError(f"unknown update rule: {update}")
        self.M = int(num_groups)
        self.update = update
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.floor = float(floor)
        self.entropy_reg = float(entropy_reg)
        self.score_clip = float(score_clip)
        self.standardize = bool(standardize)
        self.device = torch.device(device)
        self.dtype = dtype

        if init is None:
            beta = torch.full((self.M,), 1.0 / self.M, device=self.device, dtype=dtype)
        else:
            beta = _to_tensor(init, self.device, dtype)
            beta = beta / beta.sum()
        self.beta = project_to_simplex_with_floor(beta, self.floor)
        # logits consistent with beta (softmax with temperature).
        self.z = torch.log(self.beta.clamp_min(1e-12)) * self.temperature
        check_simplex(self.beta, "beta_init")

    # ---- accessors ------------------------------------------------------
    def probs(self) -> torch.Tensor:
        return self.beta

    def as_numpy(self) -> np.ndarray:
        return self.beta.detach().cpu().numpy()

    def weight_dict(self, group_ids) -> dict:
        """Return ``{group_id: beta_j}`` as python floats for the given ordered ids."""
        vals = self.beta.detach().cpu().tolist()
        return {int(g): float(vals[i]) for i, g in enumerate(group_ids)}

    def entropy(self) -> float:
        b = self.beta.clamp_min(1e-12)
        return float(-(b * b.log()).sum())

    # ---- update ---------------------------------------------------------
    def _prepare_scores(self, hypergrad: torch.Tensor) -> torch.Tensor:
        h = _to_tensor(hypergrad, self.device, self.dtype)
        check_finite(h, "hypergradient")
        # entropy regularization pushes beta toward uniform: d/dbeta_j [-tau*H] = tau*(log beta_j + 1)
        if self.entropy_reg != 0.0:
            h = h + self.entropy_reg * (self.beta.clamp_min(1e-12).log() + 1.0)
        # center scores (only relative utility matters on the simplex).
        h = h - h.mean()
        # standardize to unit std so the beta step is invariant to the (K/n-dependent)
        # scale of the raw hypergradients; if there is no spread, produce no movement.
        if self.standardize:
            std = h.std()
            h = h / (std + 1e-8) if float(std) > 1e-12 else torch.zeros_like(h)
        if self.score_clip and self.score_clip > 0:
            h = torch.clamp(h, min=-self.score_clip, max=self.score_clip)
        return h

    def step(self, hypergrad) -> dict:
        """Apply one beta update given per-group hypergradients ``h_j``.

        Returns a dict of logging quantities (raw + normalized scores, movement, entropy).
        """
        raw = _to_tensor(hypergrad, self.device, self.dtype)
        scores = self._prepare_scores(raw)
        prev = self.beta.clone()

        if self.update == "exp_grad":
            new = self.beta * torch.exp(-self.lr * scores)
            new = new / new.sum()
        else:  # logit / softmax
            self.z = self.z - self.lr * scores
            new = torch.softmax(self.z / self.temperature, dim=-1)

        new = project_to_simplex_with_floor(new, self.floor)
        check_simplex(new, "beta")
        # keep logits consistent after projection (for logit rule continuity)
        self.z = torch.log(new.clamp_min(1e-12)) * self.temperature
        self.beta = new

        l1 = float((new - prev).abs().sum())
        l2 = float((new - prev).norm())
        return {
            "raw_hypergrad": raw.detach().cpu().numpy(),
            "score": scores.detach().cpu().numpy(),
            "beta": new.detach().cpu().numpy(),
            "entropy": self.entropy(),
            "beta_l1_movement": l1,
            "beta_l2_movement": l2,
        }
