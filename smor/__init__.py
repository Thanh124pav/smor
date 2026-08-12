"""SMOR Module 1 — online data reweighting / utility estimation.

A learner-agnostic bilevel data-reweighting component:

    inner:  theta*(beta) = argmin_theta  sum_j beta_j L_{G_j}(theta)
    outer:  min_beta      L_out(theta*(beta))

Two independent resolution knobs (see PLAN.md):
    n : reweighting granularity  (demonstrations represented by one beta cell)
    K : hypergradient / curvature fidelity (truncated Neumann depth, HVP-only)
"""

from smor.evidence import ReweightingEvidence

__all__ = ["ReweightingEvidence"]
__version__ = "0.1.0"
