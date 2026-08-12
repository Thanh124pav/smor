# PLAN.md — SMOR Online Reweighting Module

## 0. Objective

Implement **Module 1 of SMOR: online data reweighting / utility estimation** as a reusable, learner-agnostic component.

The immediate research goal is **not** to solve data acquisition yet. The acquisition/scaling module will be implemented later and should consume the outputs of this module.

The online reweighting module should:

1. Reproduce a faithful **CAIL baseline**.
2. Implement a **CAIL-style one-step bilevel baseline** under a common backbone.
3. Implement the proposed **cluster-granular, curvature-aware bilevel reweighting**.
4. Expose two independent resolution knobs:
   - `n`: data/reweighting granularity — number of demonstrations represented by one beta cell.
   - `K`: hypergradient depth / curvature fidelity.
5. Log a full time series of data-utility evidence for future acquisition inference:
   - `beta_history`
   - `hypergradient_history`
   - cluster statistics
   - policy metrics
   - compute overhead

The module must be usable twice in the final SMOR pipeline:

```text
uniform pilot dataset
    -> OnlineReweighter
    -> pilot evidence
    -> [future acquisition/scaling module]
    -> target collection
    -> OnlineReweighter
    -> final policy
```

Do **not** implement the acquisition/scaling module in this task.

---

# 1. Research framing

We study a bilevel problem

\[
\theta^*(\beta)
=
\arg\min_\theta
L_{\mathrm{in}}(\theta,\beta),
\]

\[
\min_\beta
L_{\mathrm{out}}(\theta^*(\beta)).
\]

CAIL approximates the dependence of the outer objective on the data weights by a one-step pseudo-update.

Our proposed module adds two independent axes.

## 1.1 Data granularity: `n`

Let a demonstration dataset contain \(N\) trajectories. Partition it into \(M\) groups

\[
\mathcal G_1,\ldots,\mathcal G_M,
\qquad
|\mathcal G_j|\approx n,
\qquad
M\approx N/n.
\]

Use

\[
\beta_t=(\beta_{1,t},\ldots,\beta_{M,t})\in\Delta^M.
\]

Each beta cell represents one group of approximately `n` **demonstrations/trajectories**, not one state-action transition.

Interpretation:

```text
n = 1                  -> per-demonstration weighting
1 < n < domain size    -> cluster-level weighting
n = entire domain      -> domain/fidelity-level weighting
```

Call `n` **reweighting granularity** or **group resolution**, not "approximation error".

For the first MVP, cluster only **within the known fidelity/domain label**. Never mix expert and low-quality trajectories into the same group.

Initial grouping implementation:
- deterministic shuffled grouping within fidelity;
- fixed groups for an entire run;
- seeded and reproducible.

Later optional extension:
- embedding-based clustering;
- gradient-similarity clustering;
- state-coverage clustering.

Do not implement those extensions until the basic method works.

---

## 1.2 Hypergradient fidelity: `K`

Let

\[
H = \nabla_\theta^2 L_{\mathrm{in}}(\theta,\beta).
\]

Define the damped curvature operator

\[
\widetilde H = H+\lambda I.
\]

Use the truncated Neumann approximation

\[
P_K
=
\eta_h
\sum_{k=0}^{K-1}
(I-\eta_h\widetilde H)^k.
\]

For group \(j\),

\[
g_j = \nabla_\theta L_{\mathcal G_j},
\qquad
g_{\mathrm{out}} = \nabla_\theta L_{\mathrm{out}}.
\]

Estimate the hypergradient

\[
h_j^{(K)}
=
-
g_{\mathrm{out}}^\top
P_K g_j.
\]

Interpretation:

```text
K = 1
    P_1 = eta_h * I
    -> one-step / gradient-alignment / CAIL-like local approximation

K > 1
    -> progressively curvature-aware approximation

large K under suitable local assumptions
    -> approaches an implicit/influence-style inverse-Hessian effect
```

Do **not** claim that finite or large `K` is exact influence estimation in a non-convex neural network.

The code must never materialize the full Hessian.

Compute Hessian-vector products (HVPs) only.

Recursive computation:

\[
v_0 = g_j,
\]

\[
v_{k+1}
=
v_k-\eta_h(\mathrm{HVP}(v_k)+\lambda v_k),
\]

\[
P_K g_j
=
\eta_h\sum_{k=0}^{K-1}v_k.
\]

---

# 2. Important conceptual separation

`n` and `K` are independent.

```text
n = data-space resolution
K = optimization/curvature resolution
```

Required experiment matrix:

| granularity | K=1 | K=2 | K=4 |
|---|---:|---:|---:|
| n=1 | yes | yes | yes |
| n=8 | yes | yes | yes |
| n=32 | yes | yes | yes |
| whole fidelity | yes | yes | yes |

Do not assume the best `n` or `K` in advance.

---

# 3. Baselines

## 3.1 Primary baseline: original CAIL

Paper:
**Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality**
Songyuan Zhang, Zhangjie Cao, Dorsa Sadigh, Yanan Sui, NeurIPS 2021.

Official code:
https://github.com/Stanford-ILIAD/Confidence-Aware-Imitation-Learning

Requirements:
- first reproduce at least one reported CAIL setting before changing the algorithm;
- do not silently modify the original CAIL objective when reporting "CAIL";
- log the exact commit / dependency versions used;
- preserve a separate config named `cail_original`.

CAIL is an AIRL-based instantiation with confidence reweighting, a bilevel formulation, a pseudo-update, and a one-step approximation.

The original baseline and our common-backbone ablations must be reported separately.

---

## 3.2 Common-backbone CAIL-style baseline

We need an apples-to-apples solver comparison.

Implement:

```text
same dataset
same policy architecture
same inner loss
same outer loss
same optimizer
same update frequency
same grouping
```

and only change the hypergradient approximation.

`K=1` must be treated as the common-backbone CAIL-style one-step baseline.

This baseline is critical because comparing original AIRL-CAIL against a different learner would confound:
- policy architecture;
- inner objective;
- outer objective;
- reweighting solver.

---

## 3.3 Other cheap baselines

Implement from the beginning:

1. `uniform`
   \[
   \beta_j = 1/M
   \]

2. `static_quality`
   - fixed weights from known fidelity/quality labels;
   - diagnostic only, since it uses oracle metadata.

3. `final_beta_only`
   - use only the terminal learned weight vector when analysis needs a static summary.

4. `whole_fidelity_K1`
   - only one beta per fidelity/domain;
   - tests whether fine granularity is actually necessary.

Later, after the core method is stable:
- Re-Mix / Group-DRO-style domain reweighting;
- DoGE-style domain reweighting;
- CUPID-style influence baseline.

Do not block the MVP on these later baselines.

---

# 4. Experimental stages

## Stage A — Reproduce CAIL

Goal:
prove the environment, demonstrations, evaluation code, and baseline implementation are correct.

Tasks:
1. Set up the official CAIL environment.
2. Reproduce one small MuJoCo setting first.
3. Verify:
   - learning curve shape;
   - final return is in a plausible range;
   - confidence weights correlate with demonstration quality.
4. Save:
   - config;
   - seed;
   - logs;
   - checkpoint;
   - reproduction notes.

Do not begin the proposed method until a minimal CAIL reproduction passes.

---

## Stage B — Build a generic reweighting interface

Create an interface approximately like:

```python
result = reweighter.fit(
    learner=learner,
    train_dataset=train_dataset,
    outer_data=outer_data,
    evaluator=evaluator,
)

result.policy
result.beta_history
result.hypergradient_history
result.group_assignments
result.group_stats
result.train_metrics
result.eval_metrics
result.compute_metrics
```

The acquisition module must later be able to consume `result` without knowing the learner implementation.

---

## Stage C — Implement grouping / granularity `n`

Input:
- trajectories;
- fidelity/domain label per trajectory;
- `group_size=n`;
- RNG seed.

Output:
- `group_id` per trajectory;
- group-to-fidelity map;
- group sizes;
- immutable group assignments.

Rules:
- group trajectories, not individual time steps;
- never create a group crossing fidelity/domain boundaries;
- allow uneven final group size;
- validate every trajectory appears exactly once;
- groups remain fixed during a run.

---

## Stage D — Implement K=1 first

Before any HVP code, implement the one-step/common-backbone baseline.

For current beta:

\[
L_{\mathrm{in}}
=
\sum_j
\beta_j L_{\mathcal G_j}.
\]

Pseudo update:

\[
\theta'
=
\theta
-
\eta_\theta
\nabla_\theta L_{\mathrm{in}}(\theta,\beta).
\]

Outer gradient:

\[
\nabla_\beta
L_{\mathrm{out}}(\theta').
\]

Update beta and then perform the real policy update using the new beta.

Tests:
- beta stays on the simplex;
- no NaNs;
- increasing a helpful group's weight should lower the toy outer loss;
- gradients reach beta;
- pseudo parameters do not overwrite the actual policy before the real update.

---

## Stage E — Implement curvature-aware K>1

Implement:
- HVP utility;
- damped HVP;
- truncated Neumann operator;
- group hypergradient.

Never instantiate an \(d\times d\) Hessian.

Suggested API:

```python
def hvp(loss, params, vector):
    ...

def apply_pk(
    vector,
    inner_loss,
    params,
    K,
    neumann_lr,
    damping,
):
    ...

def group_hypergradient(
    group_grad,
    outer_grad,
    inner_loss,
    params,
    K,
    neumann_lr,
    damping,
):
    ...
```

Required numerical tests on a tiny quadratic problem:

1. Construct a small positive-definite quadratic where \(H^{-1}\) is explicitly computable.
2. Check:
   \[
   P_1v = \eta_h v.
   \]
3. Check approximation error
   \[
   \|P_Kv-H^{-1}v\|
   \]
   decreases for valid `K`/step-size settings.
4. Compare autograd HVP against explicit Hessian multiplication.
5. Verify damping improves stability near singularity.

Only after these tests pass, connect HVP code to the policy model.

---

# 5. Beta parameterization

Use a simplex distribution:

\[
\beta_j\ge0,
\qquad
\sum_j\beta_j=1.
\]

Preferred implementation:
store unconstrained logits `z` and define

```python
beta = softmax(z / temperature)
```

or use an exponentiated-gradient update.

For an outer-loss hypergradient \(h_j=dL_{\mathrm{out}}/d\beta_j\),

\[
\beta_j^{t+1}
\propto
\beta_j^t
\exp(-\eta_\beta h_j).
\]

Add:
- minimum floor / epsilon to prevent irreversible zero mass;
- optional entropy regularization;
- temperature config;
- gradient clipping / score normalization.

Log both:
- beta probabilities;
- raw hypergradient/utility scores.

---

# 6. Update schedule

Do not update beta at every policy SGD step by default.

Expose:

```yaml
reweight_interval: R
```

Procedure:

```text
train theta for R steps using current beta
-> estimate outer signal / hypergradient
-> update beta
-> continue training theta
```

Start with:
- small `R` for debugging;
- then test several values.

Reason:
- lower computational cost;
- reduce noisy rapid beta oscillation;
- allows the policy to respond to the current weighting before re-estimation.

Log:
- policy steps per beta update;
- HVP wall time;
- beta-update wall time;
- total overhead relative to uniform training.

---

# 7. Outer objective

For the **first solver-comparison experiment**, use the same outer objective as the CAIL/common-backbone setup so that K=1 vs K>1 is isolated.

Do not change both solver and outer objective in the same first experiment.

After the solver is validated, add a modular outer-objective interface:

```python
class OuterObjective:
    def loss(self, policy, batch_or_rollouts):
        ...
```

Candidate future objectives:
1. CAIL ranking loss.
2. Held-out expert / high-quality validation loss.
3. Closed-loop return or success surrogate.
4. CUPID-inspired policy-performance influence signal.

The first MVP does not need all four.

---

# 8. Critical validation experiment: does K>1 predict long-horizon utility better?

This experiment is more important than immediately running many robot benchmarks.

For each selected group \(G_j\):

1. Start from a fixed policy checkpoint.
2. Perturb/upweight the group:
   \[
   \beta_j \leftarrow \beta_j+\epsilon
   \]
   with renormalization.
3. Train for a much longer inner horizon \(H_{\mathrm{oracle}}\).
4. Measure the realized outer-performance change:
   \[
   \Delta_j^{\mathrm{long}}.
   \]
5. Compare this empirical long-horizon effect with:
   \[
   h_j^{(1)},h_j^{(2)},h_j^{(4)},...
   \]

Metrics:
- sign accuracy;
- Spearman rank correlation;
- Pearson correlation;
- cosine similarity with a finite-difference beta gradient where feasible.

Primary research question:

> Does increasing K improve prediction of the long-horizon effect of changing data weights?

This is the direct test of the motivation for moving beyond a one-step CAIL-style approximation.

---

# 9. Granularity experiment: role of n

Run:

```text
n ∈ {1, 8, 32, whole-fidelity}
```

Measure:
- final return / success;
- beta stability;
- hypergradient variance;
- long-horizon utility rank correlation;
- wall-clock time;
- peak VRAM;
- number of HVPs;
- number of beta dimensions.

Expected trade-off to test, not assume:

```text
small n
-> high resolution
-> more beta dimensions
-> noisier/more expensive estimates

large n
-> stable/cheap
-> possible aggregation bias
-> may hide useful heterogeneity within a fidelity
```

Do not hard-code a preferred n before this experiment.

---

# 10. Initial dataset / environment scope

Start cheap.

Phase 1:
- low-dimensional MuJoCo-style tasks compatible with CAIL reproduction;
- varying-optimality demonstrations.

Phase 2:
- one continuous-control robot manipulation benchmark;
- two fidelity levels first:
  - expert/high-quality;
  - low-quality/suboptimal.

Do not start with:
- VLA;
- image-heavy Diffusion Policy;
- real robot;
- many fidelity levels.

The purpose of Phase 1 is to validate the reweighting algorithm and hypergradient, not demonstrate final SMOR scale.

---

# 11. Learner abstraction

The reweighting module must not depend directly on AIRL internals.

Define a learner contract approximately as:

```python
class WeightedLearner:
    def per_group_losses(self, batch_by_group):
        """Return differentiable L_j(theta) for every active group."""
        ...

    def weighted_inner_loss(self, beta, batch_by_group):
        ...

    def parameters_for_reweighting(self):
        ...

    def train_step(self, beta, batch_by_group):
        ...

    def evaluate(self, ...):
        ...
```

Implement adapters in order:

1. CAIL/AIRL adapter for reproduction.
2. Simple BC adapter for debugging and cheap ablations.
3. Later: BC-RNN.
4. Later: IQL/offline RL.

Do not implement IQL in the first milestone.

---

# 12. Code organization

Recommended structure:

```text
smor/
├── baselines/
│   └── cail/
│       ├── README.md
│       ├── adapter.py
│       └── configs/
│
├── learners/
│   ├── base.py
│   ├── bc.py
│   └── cail_airl_adapter.py
│
├── reweighting/
│   ├── __init__.py
│   ├── grouping.py
│   ├── beta.py
│   ├── hvp.py
│   ├── neumann.py
│   ├── hypergradient.py
│   ├── outer_objective.py
│   ├── scheduler.py
│   └── online_reweighter.py
│
├── evaluation/
│   ├── metrics.py
│   ├── rollout.py
│   └── long_horizon_utility.py
│
├── experiments/
│   ├── reproduce_cail.py
│   ├── validate_hvp.py
│   ├── compare_K.py
│   ├── compare_n.py
│   └── run_reweighting_grid.py
│
├── configs/
│   ├── cail_reproduction.yaml
│   ├── one_step.yaml
│   └── curvature_reweight.yaml
│
└── tests/
    ├── test_grouping.py
    ├── test_beta_simplex.py
    ├── test_hvp.py
    ├── test_neumann.py
    ├── test_k1_equivalence.py
    └── test_toy_bilevel.py
```

Keep the future `acquisition/` module absent or stubbed only.

---

# 13. Logging schema

Every run must log:

## Run metadata
- git commit;
- config;
- seed;
- environment;
- dataset composition;
- total number of demonstrations;
- fidelity counts;
- n;
- K;
- damping;
- Neumann step size;
- beta learning rate;
- beta update interval.

## Per beta update
- beta vector;
- raw hypergradient per group;
- normalized score per group;
- group loss;
- outer loss;
- group fidelity;
- HVP count;
- HVP norm;
- beta entropy;
- beta L1/L2 movement from previous update.

## Evaluation
- episodic return;
- success rate where available;
- policy validation loss;
- wall-clock;
- GPU time;
- peak VRAM.

Store results in a machine-readable format (`jsonl`, `parquet`, or structured CSV) and checkpoints separately.

---

# 14. Safety/stability checks

Fail loudly if:
- beta contains NaN/Inf;
- beta sum differs materially from 1;
- HVP contains NaN/Inf;
- Neumann iterates explode;
- group assignment changes mid-run;
- a group has zero demonstrations;
- K < 1;
- n < 1.

Add configurable:
- damping;
- HVP norm clipping;
- beta entropy regularization;
- beta floor;
- score clipping;
- fallback to K=1 if a curvature update is numerically invalid, while logging the fallback.

Do not silently fall back without a warning.

---

# 15. Required ablations for the first research report

Minimum:

```text
Uniform
CAIL original
Common-backbone K=1
Proposed K=2
Proposed K=4
```

at:

```text
n = 1
n = moderate cluster
whole-fidelity
```

If compute is limited, prioritize:
1. one environment;
2. more method ablations;
3. then add environments.

The first goal is mechanism validation, not benchmark breadth.

---

# 16. Definition of success for Module 1

Module 1 is considered ready for integration with future acquisition/scaling work only if:

1. CAIL reproduction is plausible.
2. `K=1` passes the expected one-step equivalence test.
3. HVP / Neumann tests pass on a known quadratic.
4. The proposed K>1 method is numerically stable.
5. Beta trajectories are nontrivial and reproducible.
6. At least one moderate K improves long-horizon utility prediction over K=1, **or** experiments convincingly show it does not.
7. Granularity `n` produces a measurable resolution/compute trade-off.
8. The module exports a stable evidence object for downstream acquisition.

Suggested output schema:

```python
@dataclass
class ReweightingEvidence:
    final_policy_path: str
    beta_history: np.ndarray
    hypergradient_history: np.ndarray
    group_assignments: np.ndarray
    group_fidelity: np.ndarray
    group_stats: dict
    eval_history: dict
    compute_history: dict
    config: dict
```

---

# 17. What NOT to do yet

Do not:
- implement the Bayesian/scaling acquisition operator;
- map beta directly to p;
- claim beta equals acquisition ratio;
- add VLA;
- run Diffusion Policy;
- add real-robot experiments;
- implement every related-work baseline;
- optimize cluster semantics before basic n/K ablations work;
- claim K->infinity is exact influence in deep non-convex models.

---

# 18. Research questions Module 1 should answer

RQ-R1:
Does one-step bilevel reweighting provide a reliable estimate of the longer-horizon effect of changing demonstration weights?

RQ-R2:
Does curvature-aware K>1 hypergradient estimation improve that estimate enough to justify its extra compute?

RQ-R3:
What reweighting granularity n gives the best trade-off between fine-grained utility resolution, estimator variance, and compute?

RQ-R4:
Do cluster-level beta trajectories expose meaningful heterogeneity inside the same nominal fidelity level?

RQ-R5:
Are the learned beta trajectories sufficiently stable across seeds/checkpoints to be useful as evidence for a later acquisition/scaling module?

---

# 19. Relevant papers to read while implementing

## Core baseline
- Zhang et al., **Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality**, NeurIPS 2021.
  https://arxiv.org/abs/2110.14754
  https://github.com/Stanford-ILIAD/Confidence-Aware-Imitation-Learning

## Bilevel / data reweighting
- Ren et al., **Learning to Reweight Examples for Robust Deep Learning**, ICML 2018.
- Lorraine et al., **Optimizing Millions of Hyperparameters by Implicit Differentiation**, AISTATS 2020.
- DoGE, **Domain Reweighting with Generalization Estimation**, ICML 2024.

## Influence
- Koh & Liang, **Understanding Black-box Predictions via Influence Functions**, ICML 2017.
- CUPID, **Curating Data your Robot Loves with Influence Functions**, 2025.

## Post-CAIL imperfect-demonstration work
- Jung et al., **Sample-efficient Adversarial Imitation Learning**, JMLR 2024.
  This paper directly compares against CAIL in experiments with varying-optimality demonstrations.
- Cao et al., **Limited Preference Aided Imitation Learning from Imperfect Demonstrations (PAIL)**, ICML 2024.
  Uses preference-derived reward to reweight imperfect demonstrations and augment the dataset; important as a newer alternative to CAIL-style confidence learning.
- **PN-GAIL: Leveraging Non-optimal Information from ...**
  Directly compares against CAIL/T-REX/D-REX and uses the CAIL codebase as an implementation reference. Check final venue/version before treating it as an archival baseline.
- Fan et al., **Imitation Learning from Suboptimal Demonstrations via Meta-Learning An Action Ranker (ILMAR)**, 2024 preprint.
  Cites CAIL and uses CAIL's codebase for algorithmic dependencies, but its main reported baseline table does not include CAIL; treat as adjacent implementation/methodological evidence, not a direct CAIL baseline.

---

# 20. First implementation order for Claude

Execute strictly in this order:

1. Inspect official CAIL repository and document its environment/dependencies.
2. Reproduce one small CAIL experiment.
3. Create the generic `WeightedLearner` and `OnlineReweighter` interfaces.
4. Implement fixed trajectory grouping with configurable `n`.
5. Implement common-backbone K=1.
6. Add unit tests for K=1.
7. Implement HVP.
8. Validate HVP on a tiny explicit Hessian.
9. Implement damped truncated Neumann `P_K`.
10. Validate convergence on a quadratic toy problem.
11. Connect K>1 to group hypergradients.
12. Add logging and compute instrumentation.
13. Run the long-horizon utility-prediction test.
14. Run small n/K grid.
15. Produce a short `RESULTS_REWEIGHTING.md` summarizing:
    - correctness;
    - failures;
    - performance;
    - compute;
    - whether K>1 improves over K=1;
    - whether intermediate n is useful.

Do not move to acquisition/scaling until this report exists.
