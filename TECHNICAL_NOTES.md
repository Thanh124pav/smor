# SMOR — Technical Notes

*Scaling-aware Mixture / Online Reweighting for imitation learning from heterogeneous
demonstrations.* These notes summarize the reformulated motivation, the experiments run so far, and
the analysis, for reporting. Full numbers and the codebase index are at the end; a companion
experiments dashboard (HTML) is published separately.

---

## 1. Motivation — Utilization and Acquisition, in harmony

Real robot-learning datasets are assembled from **many heterogeneous sources** (different teleop
devices, operators, machine-generated rollouts, tasks). Two questions arise, usually studied in
isolation:

- **Utilization** — *given a fixed collected dataset*, how should we **weight** each source so the
  learned policy performs best? (This is online data reweighting / confidence learning — CAIL and
  our SMOR reweighter live here.)
- **Acquisition** — *given a collection budget* `B`, what **mixture** `p` of sources should we
  **collect** to maximize downstream performance, and how does the optimal mixture change with
  scale? (This is a data-scaling-law question.)

**Reformulation (the thesis of this work): utilization and acquisition must be solved together.**
They are two views of the same per-source value signal:

- Utilization produces, as a by-product, a **confidence / weight** `β` and a **hypergradient**
  `h` per source — an estimate of *how much each source helps the true objective*.
- Acquisition needs exactly that signal, extrapolated over quantity: *collect more of the sources
  whose marginal value stays high*.

So a **pilot utilization pass** (cheap reweighting on a small mixed pilot set) should **drive
acquisition** (what to collect at the target budget), and the **acquisition scaling law** should in
turn tell utilization *how* to weight at that budget. The linchpin that makes this loop trustworthy
is the **outer-loss design**: utilization only produces a useful value signal if its outer
objective is aligned with the true deployment goal (task return), not a proxy (matching a
privileged "clean" source). This is where the novelty concentrates (§4).

```
  small mixed pilot ──utilization(reweight)──▶  β, h  (per-source value)
          ▲                                        │
          │                                        ▼
   final reweighting ◀──acquisition(scaling law)── p*(B_target)  ──collect──▶ target data
```

---

## 2. Setup

- **Learner (fixed across all baselines):** Behavior Cloning, MLP (256×256 / 512×512, tanh),
  Adam 1e-3. Isolates the *weighting* mechanism.
- **Weighting methods:** `uniform` · `only:<source>` · `static_quality` (best-labelled source) ·
  **CAIL** (one-step hypergradient, `K=1`, ranking outer loss) · **SMOR** (curvature-aware
  hypergradient `K>1`) · `AIRL` (adversarial baseline, HalfCheetah only).
- **Outer losses:** open-loop `ValidationLoss` / `CAILRankingLoss`; closed-loop
  `ClosedLoopReturn` (differentiable point-mass) and `ClosedLoopRolloutReturn` with
  **REINFORCE / GRPO / PPO** estimators (non-differentiable sims).
- **Metric (primary): closed-loop success / return** — *not* validation MSE (see §4, §5).
- **Data:** *official* — CAIL Ant-v2 buffer, Minari (D4RL-modern) mujoco, RoboMimic (robosuite);
  *self-simulated* — point-mass and robosuite device-calibration (each source = one systematic
  error, re-simulated).

---

## 3. Utilization results — two regimes of source heterogeneity

### 3.1 Regime A — sources differ by NOISE (complementary; no source dominates) → non-trivial β ✔

Each source is the same task through a differently **mis-calibrated device** (rotation / gain /
per-joint bias). No source is globally best, so the loss-minimizing weighting is a genuine
**interior mixture**.

| metric | Point-mass 3-dev | RoboMimic-lift 3-dev | RoboMimic-lift 3 good + 1 poison |
|---|---|---|---|
| SMOR learns β | **[.17, .51, .33] interior** | interior | **[.27, .32, .41, 0] interior** |
| SMOR vs uniform (**success**) | tie (success saturates =1.0) | **0.87 vs 0.20** ✔ | **1.00 vs 0.00** ✔ |
| CAIL (success) | ≈ SMOR | 0.40 | — |

**Finding.** Reweighting recovers a **non-trivial interior β**. On **closed-loop success** SMOR
clearly beats uniform and CAIL on robosuite (0.87 vs 0.20 vs 0.40; with a harmful "poison" source,
1.00 vs 0.00) — even though on validation-MSE it merely ties uniform. When the sources are *purely*
complementary (bias-cancelling) and the task is easy (point-mass), uniform is already a strong
baseline (averaging also cancels bias) and success saturates, so the win shows up only on harder,
compounding, closed-loop settings. *(RoboMimic device numbers currently 1 seed.)*

### 3.2 Regime B — sources differ by QUALITY (clean vs dirty) → trivial β, but enables filtering

When one source is strictly cleaner (proficient-human vs machine-generated, expert vs noisy), the
optimal weighting **collapses to a corner** (put all mass on the clean source):

| dataset (official) | uniform | SMOR | learned β |
|---|---|---|---|
| RoboMimic lift MH+MG (val↓) | 0.044 | **0.023** | corner {better .999} |
| RoboMimic lift PH+MH (val↓) | 0.039 | 0.030 | corner {ph .986} |

The optimum is *trivial* (a ranking, not a mixture) — **but that is itself useful**: the learned
confidence/weight is a **data-filtering signal**. It reliably separates clean from dirty sources
(on the CAIL Ant-v2 buffer, SMOR's per-demo weight ranks demonstrations by their true return with
Spearman **0.67**). So Regime B does not give an interesting *mixture*, but it gives a **curator**:
"keep these, drop those." This is the natural bridge to acquisition (§5): *don't buy more of the
dirty source.*

### 3.3 SMOR vs CAIL on standard varying-optimality benchmarks

Across CAIL Ant-v2 (official, n=1, env-free), and modern Ant-v1 / HalfCheetah (return, 3 seeds):
**SMOR ≈ CAIL** — curvature (`K>1`) matches but does not beat the one-step update (`K=1`) because
these benchmarks are well-conditioned; K=1 already concentrates on the good source. *The
contribution of SMOR is therefore not the curvature knob, but the outer-loss design below.*

---

## 4. Outer-loss design — where the novelty is (and CAIL's failure mode)

**CAIL's outer loss collapses.** CAIL's confidence objective (rank/match the labelled high-quality
demos, one-step) rewards *matching the "expert" source*. But the true goal is **task return**. When
BC covariate-shift makes expert-only policies **brittle**, "match expert" *diverges* from "maximize
return", and the reweighting confidently optimizes the wrong thing:

> **Minari HalfCheetah (official, return, 3 seeds):** uniform **2897**, CAIL-ranking **88**,
> SMOR-ranking **154**. The reweighters put β → [1,0,0] (all expert) and *lose badly to uniform.*

This is a **misalignment**, not a bug: an open-loop / ranking proxy is not the deployment objective.

**GRPO closed-loop return fixes it.** We replace the proxy with the *actual task return*, estimated
by a policy-gradient (GRPO: group-relative advantage, no critic) surrogate over real rollouts, plus
a cosine-normalized hypergradient (`g_j/‖g_j‖`) so a noisy return-gradient is not hijacked by
whichever source has the largest gradient:

| outer loss (Minari, return, 3 seeds) | HalfCheetah | Hopper | Walker2d |
|---|---|---|---|
| uniform | 1694 | 516 | 811 |
| CAIL (ranking) | 88 | 552 | 597 |
| **SMOR + GRPO closed-loop** | **2636** | 572 | **1264** |

On **HalfCheetah** the GRPO reweighter learns an **interior** mixture (`β≈[.64,.04,.32]`, one seed
reached 4353) and **beats uniform**; on **Walker2d** likewise (1264 vs 811). On robosuite
manipulation (RoboMimic **PH+MH+MG**, success): **SMOR+GRPO 0.80 vs uniform/CAIL 0.60**, learning
an interior mixture that *upweights `mg_success`* (machine rollouts that actually succeeded) —
which a quality-ranking would have discarded. → GRPO optimizes *what works*, not *what looks clean*.

**The risk of GRPO — all-failed trajectories.** The return signal is only informative when returns
**vary**. With a sparse reward and an under-trained policy (all rollouts fail, return ≈ constant),
the advantage is pure noise and the reweighter drifts (early runs collapsed β onto the *poison*
source, weight 0.48–0.74; the `g_j` normalization brought this down to 0.20, and dense reward /
enough warm-up removes it). So GRPO needs: (i) dense/shaped reward or a policy warm-started to
competence, (ii) enough rollout episodes to reduce variance. On "easy" envs where the proxy already
works (Hopper), GRPO is a wash. **This trade-off — power vs. the all-failed regime — is the central
design tension of the outer loss.**

---

## 5. Acquisition results — scaling laws, and the harmony with utilization

We fit `(B, p) → performance` and predict the budget-dependent optimal mixture `p*_B`, then
extrapolate to held-out budgets (pipeline: sweep → GAM/GP trend → parametric law → held-out
extrapolation → oracle + regret + bootstrap).

- **Regime A (complementary noise), point-mass bias-variance, 210 runs:** the optimal mixture
  **shifts with budget** — `p*` moves 0.0 (small budget → prefer the *precise* source) → 0.4–0.8
  (large budget → the *unbiased* source's diversity wins). The fitted law extrapolates to held-out
  budgets (RMSE 0.016), with target-budget regret ≤ 0.0035. **This is the interesting acquisition
  regime: what to buy depends on how much you buy.**
- **Regime B (clean/dirty), RoboMimic PH vs MG, 90 runs:** **dominance** — `p*=1.0` at every
  budget (always buy the clean source). Consistent with §3.2: the utilization signal ("MG doesn't
  help") *is* the acquisition recommendation ("don't buy MG").

**Harmony.** The two modules read the *same* per-source value signal from opposite ends:
utilization measures value at the current pilot scale; acquisition extrapolates it over quantity.
In Regime A the signal is a *mixture that moves with scale* (utilization β at budget `B` ≈ the
acquisition `p*_B`); in Regime B it is a *filter* (utilization drops the dirty source ⇒ acquisition
buys none of it). And both are only trustworthy when utilization's outer loss is aligned (§4) —
which is why the closed-loop/GRPO outer loss is the keystone connecting the two.

---

## 6. Summary of claims

1. **Two regimes.** Complementary-noise sources → **non-trivial interior β** (good utilization,
   scale-dependent acquisition). Clean/dirty sources → **trivial corner**, but a useful **data
   filter** and a dominance acquisition rule.
2. **Curvature is not the story.** SMOR (`K>1`) ≈ CAIL (`K=1`) on varying-optimality benchmarks.
3. **Outer loss is the novelty.** CAIL's ranking/open-loop loss **collapses** under covariate-shift
   (uniform ≫ reweight); a **closed-loop return (GRPO)** loss realigns it and **beats uniform**
   (HalfCheetah 2636 vs 1694; robosuite success 0.80 vs 0.60), at the cost of an **all-failed-
   trajectory failure mode** that needs dense reward / warm-up / enough episodes.
4. **Unified framing.** Utilization and acquisition are one value-estimation problem seen at two
   scales, coupled through the (aligned) outer loss.

---

## 7. Codebase index & experiments

**Utilization (Module 1)** — `smor/reweighting/` (bilevel reweighter, Neumann `K`, β on simplex),
`smor/learners/bc.py` (BC + REINFORCE/GRPO/PPO rollout surrogates), `smor/data/` (RoboMimic, CAIL,
ragged TrajectoryDataset), `smor/envs/robosuite_env.py` (headless rollout),
`smor/reweighting/outer_objective.py` (4 outer losses).
Drivers: `experiments/robomimic_reweight.py` (`--outer val|ranking|grpo`),
`experiments/robomimic_multisource.py` (`--poison`), `experiments/cail_compare.py`,
`baselines_airl/{cail_return,minari_reweight}.py`.

**Acquisition (Module 2)** — `smor/scaling/` (sampler, parametric laws, fitting, model selection,
oracle+regret, KKT allocation, bootstrap, marginal gain, GP/GAM trend, ScalingEvidence).
Drivers: `experiments/scaling/{generate_pools,run_grid,run_grid_pointmass,fit_laws,plot_scaling}.py`.

**Status.** ~66 unit tests pass (incl. the Module-2 synthetic-law integration test). Datasets:
CAIL Ant-v2, Minari (halfcheetah/hopper/walker2d), RoboMimic (lift/square). ManiSkill installed but
its env cannot run on this WSL box (Vulkan) → future work on a Vulkan-capable machine.

## 8. Limitations / next steps

- Several RoboMimic-device and GRPO numbers are **1 seed / high-variance** (GRPO is policy-gradient);
  multi-seed runs are in progress.
- The **all-failed-trajectory** failure mode of the closed-loop loss deserves a principled fix
  (value baseline / curriculum warm-up / PPO-clip); the estimator variants are implemented.
- **Pilot → acquisition bridge** (§1 loop): learn `F: {β, h}_pilot → p*(B_target)` — the
  scaling infrastructure and the reweighting evidence object are in place; the bridge is the next
  milestone.
