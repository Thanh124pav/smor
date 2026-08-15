# Findings — reweighting on HalfCheetah (varying-optimality)

Task: seals/HalfCheetah-v1. Three demo sources by corrupting a pretrained PPO expert:
expert (ret ≈1660), medium (≈758), noisy (≈−109). Correct behaviour = concentrate β on expert.
Metric = eval return; oracle "only-expert BC" ≈ 1424; uniform BC ≈ 895.

## 1. Reweighting works on a real task — at K=1
`compare_airl_smor` / `ksweep`:

| method / K | learned β (expert/med/noisy) | eval return |
|---|---|---|
| only-expert (oracle) | — | 1424 |
| uniform BC | — | 895 |
| **SMOR K=1** | **1.0 / 0 / 0** | **1447** |

SMOR at K=1 recovers the expert (β→1.0) and matches the oracle, clearly beating uniform.

## 2. K>1 with the default Neumann step is UNSTABLE on a 256×256 net
`ksweep --Ks 1 2 4 8 16` (neumann_lr=0.1, damping=1.0):

| K | β(expert/med/noisy) | return | expert-mass start→end |
|---|---|---|---|
| 1 | 1.0 / 0 / 0 | 1447 | 0.33 → 1.00 |
| 2 | 0.16 / 0.07 / 0.77 | 344 | 0.33 → 0.16 |
| 4 | 0.35 / 0.07 / 0.58 | 208 | 0.33 → 0.35 |
| 8 | 0.26 / 0.03 / 0.71 | 375 | 0.33 → 0.26 |
| 16 | 0.02 / 0.34 / 0.64 | 184 | 0.33 → 0.02 |

K>1 drives β to the **noisy** source and return collapses — β starts near-uniform and drifts the
wrong way.

## 3. Diagnosis + fix: scale the Neumann step to the Hessian
The truncated Neumann operator `P_K = η_h Σ (I − η_h(H+λI))^k` only contracts when
`η_h·λ_max(H+λI) < 2`. On the tiny point-mass net η_h=0.1 is fine; on the 256×256 policy the
Hessian spectrum is far larger, so the series diverges / points the wrong way and the estimated
hypergradient is garbage.

`ksweep --Ks 2 4 8 --neumann-lr 0.005 --damping 10`:

| K | β(expert/med/noisy) | return |
|---|---|---|
| 2 | 1.0 / 0 / 0 | 1320 |
| 4 | 1.0 / 0 / 0 | 1396 |
| 8 | 1.0 / 0 / 0 | 1416 |

With a contracting step, K>1 recovers correct behaviour (β→expert, return ≈ oracle).

## 4. Fix landed: auto-scaled Neumann step (`neumann_auto`)
`estimate_lambda_max` (power iteration over HVPs) estimates `lambda_max(H+lambda I)`; the
reweighter then sets `eta_h = neumann_safety / lambda_max` each beta update so the series always
contracts, with no per-model tuning. `ksweep --Ks 1 2 4 8 --neumann-auto`:

| K | β(expert/med/noisy) | return |
|---|---|---|
| 1 | 1.0 / 0 / 0 | 1447 |
| 2 | 1.0 / 0 / 0 | 1443 |
| 4 | 1.0 / 0 / 0 | 1488 |
| 8 | 1.0 / 0 / 0 | 1346 |

Every K now recovers β→expert and return ≈ oracle — the instability is gone without hand-tuning.
Enabled by default in `configs/curvature_reweight.yaml`.

## Takeaways
- The reweighting mechanism is **validated on a real MuJoCo task**, not just point-mass.
- **K=1 is sufficient here** (K>1 matches but does not beat it) — consistent with PLAN §16.6.
- The earlier real-task failures (Meta-World push, HalfCheetah K=2) were a **numerical
  instability of the curvature step**, not a conceptual flaw: the fix is to scale `neumann_lr`
  (or damping) to the model's curvature. A principled improvement is to auto-set
  `η_h ≈ c/λ_max` via a few power-iteration HVPs so K>1 is stable across model sizes.
- AIRL (imitation, 300k steps, CPU) is undertrained (return ≈ −32) and needs GPU + ≫1M steps for
  a fair number; infrastructure is in place (`baselines_airl/`).
