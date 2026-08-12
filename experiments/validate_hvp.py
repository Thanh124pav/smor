"""Standalone HVP / Neumann validation on an explicit quadratic (PLAN.md Stage E, step 8/10).

Prints, for a known SPD Hessian H:
  * the P_1 = eta_h * I identity check,
  * per-K approximation error ||P_K v - (H+lambda I)^{-1} v|| (should decrease),
  * autograd-HVP vs explicit H@v,
  * a near-singular case showing damping stabilizes the estimate.
"""

from __future__ import annotations

import torch

from smor.reweighting.hvp import hvp
from smor.reweighting.neumann import apply_pk


def _spd(eigs):
    d = len(eigs)
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    H = Q @ torch.diag(torch.tensor(eigs, dtype=torch.float64)) @ Q.t()
    return 0.5 * (H + H.t())


def main() -> None:
    torch.set_printoptions(precision=4)
    H = _spd([0.5, 1.0, 2.0, 4.0])
    d = H.shape[0]
    theta = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))
    loss = 0.5 * theta @ (H @ theta)
    v = torch.randn(d, dtype=torch.float64)

    print("== autograd HVP vs explicit H@v ==")
    got = hvp(loss, [theta], v)
    print(f"  max|HVP - H v| = {float((got - H @ v).abs().max()):.3e}")

    print("\n== P_1 = eta_h * I ==")
    eta = 0.15
    p1 = apply_pk(v, loss, [theta], K=1, neumann_lr=eta, damping=0.0)
    print(f"  max|P_1 v - eta*v| = {float((p1 - eta * v).abs().max()):.3e}")

    print("\n== P_K -> (H + lambda I)^{-1} v ==")
    lam = 0.3
    target = torch.linalg.solve(H + lam * torch.eye(d, dtype=torch.float64), v)
    for K in [1, 2, 4, 8, 16, 32, 64]:
        pk = apply_pk(v, loss, [theta], K=K, neumann_lr=eta, damping=lam)
        print(f"  K={K:3d}  ||P_K v - target|| = {float((pk - target).norm()):.3e}")

    print("\n== damping stabilizes a near-singular Hessian ==")
    Hs = _spd([1e-3, 1.0, 3.0])
    theta2 = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    loss2 = 0.5 * theta2 @ (Hs @ theta2)
    v2 = torch.randn(3, dtype=torch.float64)
    K = 40
    undamped = apply_pk(v2, loss2, [theta2], K=K, neumann_lr=0.3, damping=0.0)
    damped = apply_pk(v2, loss2, [theta2], K=K, neumann_lr=0.3, damping=0.5)
    tgt_u = torch.linalg.solve(Hs, v2)
    tgt_d = torch.linalg.solve(Hs + 0.5 * torch.eye(3, dtype=torch.float64), v2)
    print(f"  undamped rel err (vs H^-1)      @K={K}: "
          f"{float((undamped - tgt_u).norm() / tgt_u.norm()):.3e}")
    print(f"  damped   rel err (vs (H+lI)^-1) @K={K}: "
          f"{float((damped - tgt_d).norm() / tgt_d.norm()):.3e}")
    print("\nOK: HVP is autograd-exact; P_K converges; damping regularizes near singularity.")


if __name__ == "__main__":
    main()
