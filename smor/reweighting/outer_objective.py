"""Outer objective interface (PLAN.md §7).

For the first solver-comparison experiment the outer objective is a held-out high-quality
validation loss, so K=1 vs K>1 is isolated to the hypergradient solver. The interface is
modular so ranking / closed-loop-return objectives can be added later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class OuterObjective(ABC):
    """Computes a differentiable outer loss ``L_out(theta)`` from a learner."""

    @abstractmethod
    def loss(self, learner) -> torch.Tensor:
        """Return a differentiable scalar outer loss."""


class ValidationLoss(OuterObjective):
    """Held-out validation (expert) loss — the default open-loop outer objective for BC."""

    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size

    def loss(self, learner) -> torch.Tensor:
        val = learner.validation_loss(batch_size=self.batch_size)
        if val is None:
            raise ValueError("learner has no validation data for ValidationLoss.")
        return val


class CAILRankingLoss(OuterObjective):
    """Common-backbone CAIL-style confidence objective (PLAN.md §3.2, §7).

    A faithful adaptation of CAIL's key mechanism — learning demonstration *confidence* from a
    quality ranking — to the shared BC backbone (no adversarial AIRL/RL loop; the original
    AIRL-CAIL is Stage A, see smor/baselines/cail/). Given per-group quality labels, the outer
    loss is a pairwise margin-ranking penalty over the policy's per-group BC losses: a
    higher-quality group should be fit at least ``margin`` better than a lower-quality one,

        L_out = mean over (i>j) of  softplus(margin + loss_i - loss_j).

    Upweighting a high-quality group lowers its loss and this penalty, so the one-step
    (K=1) confidence/hypergradient update concentrates beta on higher-quality demonstrations —
    exactly CAIL's confidence behavior.
    """

    def __init__(self, group_quality: dict, margin: float = 0.02):
        self.group_quality = {int(k): float(v) for k, v in group_quality.items()}
        self.margin = float(margin)

    def loss(self, learner) -> torch.Tensor:
        import torch.nn.functional as F

        gids = list(self.group_quality)
        batches = learner.sample_batches(gids)
        losses = learner.per_group_losses(batches)
        terms = []
        for i in gids:
            for j in gids:
                if self.group_quality[i] > self.group_quality[j]:
                    terms.append(F.softplus(self.margin + losses[i] - losses[j]))
        if not terms:
            total = None
            for v in losses.values():
                total = v if total is None else total + v
            return total / max(1, len(losses))
        return torch.stack(terms).mean()


class ClosedLoopReturn(OuterObjective):
    """Differentiable closed-loop return surrogate (PLAN.md §7).

    Backpropagates through the (differentiable) env dynamics of a policy rollout, so the outer
    signal reflects *task* performance (compounding action errors) rather than open-loop action
    matching. Requires a learner exposing ``closed_loop_loss(n_episodes, horizon)``.

    Only usable when the env dynamics are differentiable (the torch point-mass). For a real,
    non-differentiable simulator (robosuite/MuJoCo, Meta-World) use :class:`ClosedLoopRolloutReturn`.
    """

    def __init__(self, n_episodes: int = 128, horizon: int | None = None):
        self.n_episodes = n_episodes
        self.horizon = horizon

    def loss(self, learner) -> torch.Tensor:
        fn = getattr(learner, "closed_loop_loss", None)
        if fn is None:
            raise ValueError("learner does not support ClosedLoopReturn (no closed_loop_loss).")
        return fn(self.n_episodes, self.horizon)


class ClosedLoopRolloutReturn(OuterObjective):
    """Closed-loop **task return** outer objective for a NON-differentiable env (robosuite/MuJoCo).

    The outer signal is the real episodic return obtained by *interacting with the environment*:
    the policy is rolled out (with Gaussian exploration), and — because we cannot backprop through
    the simulator — the gradient of the return w.r.t. theta is estimated with the score-function /
    REINFORCE (policy-gradient) estimator. Concretely, the learner returns the differentiable
    surrogate

        L_out(theta) = - E_t[ log pi_theta(a_t | s_t) * A_t ],    A_t = R(tau) - baseline

    whose gradient equals -grad_theta E[return]. Minimizing it drives theta (and hence, through the
    hypergradient, the group weights beta) toward demonstrations that *actually improve task
    return* — a closed-loop signal, not open-loop action matching.

    Trade-offs vs :class:`ValidationLoss`: this rolls out the env at every beta update (expensive)
    and the REINFORCE gradient is higher-variance, so use enough episodes and expect noisier beta.
    Requires a learner exposing the rollout surrogate for the chosen ``variant`` and an attached env.

    Three policy-gradient advantage estimators are selectable via ``variant`` (all share one
    rollout; pair with ``config.normalize_group_grads=True`` for a stable hypergradient):

    * ``"reinforce"`` -- episodic return, mean baseline (highest variance).
    * ``"grpo"`` -- group-relative advantage ``(R_i - mean)/std`` over the rollout group, no critic.
    * ``"ppo"`` -- per-step reward-to-go advantage with a time-baseline (temporal credit assignment).
    """

    _METHOD = {"reinforce": "rollout_pg_surrogate", "grpo": "rollout_grpo_surrogate",
               "ppo": "rollout_ppo_surrogate"}

    def __init__(self, n_episodes: int = 32, horizon: int | None = None,
                 explore_std: float = 0.1, variant: str = "grpo"):
        if variant not in self._METHOD:
            raise ValueError(f"variant must be one of {list(self._METHOD)}, got {variant!r}")
        self.n_episodes = int(n_episodes)
        self.horizon = horizon
        self.explore_std = float(explore_std)
        self.variant = variant

    def loss(self, learner) -> torch.Tensor:
        fn = getattr(learner, self._METHOD[self.variant], None)
        if fn is None:
            raise ValueError(
                f"learner does not support ClosedLoopRolloutReturn variant '{self.variant}'.")
        return fn(n_episodes=self.n_episodes, horizon=self.horizon,
                  explore_std=self.explore_std)
