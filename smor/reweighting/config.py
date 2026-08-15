"""Configuration for the online reweighter (single source of knobs)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class OnlineReweighterConfig:
    """All reweighting knobs. ``n`` and ``K`` are the two independent resolution axes."""

    # --- resolution axes ---
    n: int = 8                      # reweighting granularity: trajectories per beta cell
    K: int = 1                      # hypergradient / curvature depth (Neumann truncation)

    # --- schedule (§6) ---
    reweight_interval: int = 5      # policy steps per beta update (R)
    n_beta_updates: int = 40        # number of beta updates over the run
    warmup_steps: int = 0           # policy-only steps before the first beta update

    # --- beta parameterization (§5) ---
    beta_lr: float = 0.5            # exponentiated-gradient / logit step size
    beta_update: str = "exp_grad"   # {"exp_grad", "logit"}
    temperature: float = 1.0
    beta_floor: float = 1e-4        # minimum mass per group
    entropy_reg: float = 0.0        # optional entropy regularization on beta
    score_clip: float = 10.0        # clip |normalized hypergradient score|
    beta_standardize: bool = True   # standardize scores to unit std (scale-invariant step)

    # --- curvature operator (§1.2) ---
    neumann_lr: float = 0.1         # eta_h (used when neumann_auto is False)
    neumann_auto: bool = False      # auto-scale eta_h = neumann_safety / lambda_max(H+lambda I)
    neumann_safety: float = 1.0     # contraction margin (<2); eta_h*lambda_max = safety
    neumann_power_iters: int = 10   # power-iteration steps for lambda_max estimate
    damping: float = 1.0            # lambda in (H + lambda I)
    hvp_clip: float = 0.0           # 0 disables; else clip HVP vector norm

    # --- inner batch / eval ---
    batch_size: int = 64            # per-group batch size for losses
    seed: int = 0
    device: str = "auto"

    # --- safety ---
    fallback_k1_on_invalid: bool = True   # log + fall back to K=1 on invalid curvature

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "OnlineReweighterConfig":
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.reweight_interval < 1:
            raise ValueError(f"reweight_interval must be >= 1, got {self.reweight_interval}")
        if self.beta_update not in {"exp_grad", "logit"}:
            raise ValueError(f"unknown beta_update: {self.beta_update}")
        if not (0.0 <= self.beta_floor < 1.0):
            raise ValueError(f"beta_floor must be in [0,1), got {self.beta_floor}")
        return self
