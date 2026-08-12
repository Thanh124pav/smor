# SMOR — Module 1: Online Data Reweighting

A **learner-agnostic online data reweighting / utility-estimation** component. Given a set of
demonstrations of *varying optimality*, it learns per-group weights `beta` on the simplex by
solving a bilevel problem

```
inner:  theta*(beta) = argmin_theta  sum_j beta_j L_j(theta)      # weighted training
outer:  min_beta      L_out(theta*(beta))                          # held-out / validation loss
```

and exposes two **independent** resolution knobs:

- **`n`** — reweighting granularity: how many demonstrations (trajectories) one `beta` cell
  represents (`n=1` per-demo … whole-fidelity domain weighting).
- **`K`** — hypergradient / curvature fidelity: the truncated, damped **Neumann** depth used to
  approximate the inverse-Hessian effect via **Hessian-vector products only** (the Hessian is
  never materialized). `K=1` is the one-step / gradient-alignment / CAIL-style baseline; `K>1`
  is curvature-aware.

It logs a full time series of data-utility evidence (`beta_history`, `hypergradient_history`,
cluster stats, policy metrics, compute overhead) and returns a stable `ReweightingEvidence`
object for a **future** acquisition/scaling module (not implemented here — see
[PLAN.md](PLAN.md) §17).

This module inspired by the CAIL bilevel confidence-learning setup
([Stanford-ILIAD/Confidence-Aware-Imitation-Learning](https://github.com/Stanford-ILIAD/Confidence-Aware-Imitation-Learning),
Zhang et al., NeurIPS 2021).

## Layout

```
smor/
  reweighting/   grouping (n), simplex beta, hvp, neumann P_K, hypergradient, scheduler,
                 online_reweighter (the fit loop), outer_objective, config
  learners/      WeightedLearner contract + BCLearner (behavior cloning)
  envs/          small GPU-friendly point-mass goal env + two-fidelity demo generator
  evaluation/    real env rollouts, correlation metrics, long-horizon utility (§8)
  baselines/cail/  CAIL/AIRL adapter STUB + original config + reproduction notes (Stage A)
  evidence.py    ReweightingEvidence output object (§16)
  runner.py      point-mass run factory
experiments/     collect_demos, train_reweighting, validate_hvp, compare_K, compare_n,
                 run_reweighting_grid
configs/         one_step.yaml (K=1), curvature_reweight.yaml (K>1), cail_reproduction.yaml
tests/           grouping, beta simplex, hvp, neumann, K=1 equivalence, toy bilevel
```

## Setup

Uses the existing conda env (Python 3.12, torch 2.11 + CUDA, numpy, scipy, pytest):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate deeplearning
```

Everything is **device-aware** and **GPU-friendly**: a single `device` flag (`auto` by default
→ `cuda` if available, else `cpu`) threads through the env, learner, and HVP/Neumann. Models are
tiny MLPs and default configs are small, so a full run finishes in seconds/minutes on CPU.

## Quickstart

Run the tests (the numerical correctness core):

```bash
python -m pytest -q
```

Validate the HVP / Neumann machinery on an explicit quadratic:

```bash
python -m experiments.validate_hvp
```

Collect demonstrations, then **train + evaluate** a reweighted BC policy:

```bash
python -m experiments.collect_demos --out data/pointmass
python -m experiments.train_reweighting --config configs/curvature_reweight.yaml --n 8 --K 2
# add --device cpu to force CPU; artifacts land in runs/n8_K2_seed0/
```

`train_reweighting` prints a summary and writes `policy.pt`, `evidence.npz`, `log.jsonl`,
`summary.json`. It reports **episodic return + success rate** (real env rollouts) before/after,
plus how much `beta` mass moved onto the expert (high-fidelity) group.

## Experiment matrix (`n` × `K`)

```bash
python -m experiments.compare_K   --config configs/curvature_reweight.yaml --n 8   # RQ-R2 (§8)
python -m experiments.compare_n   --config configs/curvature_reweight.yaml         # RQ-R3 (§9)
python -m experiments.run_reweighting_grid --config configs/curvature_reweight.yaml
```

- `compare_K` — does `K>1` predict the *realized long-horizon* effect of upweighting a group
  better than `K=1`? Reports sign accuracy / Spearman / Pearson / cosine of `h^{(K)}` vs
  `Delta^long`.
- `compare_n` — the resolution/compute trade-off across `n ∈ {1, 8, 32, whole-fidelity}`.
- `run_reweighting_grid` — trains + evaluates every `(n, K)` cell and checks cross-seed
  reproducibility of the final `beta`.

Results and a written interpretation are in
[RESULTS_REWEIGHTING.md](RESULTS_REWEIGHTING.md).

## Using the module programmatically

```python
from smor.reweighting import OnlineReweighter, OnlineReweighterConfig, make_groups
from smor.runner import build_pointmass_run

cfg = OnlineReweighterConfig(n=8, K=2, n_beta_updates=40, device="auto")
setup = build_pointmass_run(cfg)                      # any WeightedLearner works here
evidence = OnlineReweighter(cfg).fit(setup.learner, setup.group_assignment)
evidence.save("runs/evidence.npz")                   # -> ReweightingEvidence for downstream use
```

To plug in a different learner (e.g. the future CAIL/AIRL adapter), implement the
`WeightedLearner` contract in `smor/learners/base.py`; the reweighter needs nothing else.

## Scope

Implements PLAN.md steps 3–14 (minus live MuJoCo). The acquisition/scaling module, the live
original-CAIL reproduction (Stage A — see `smor/baselines/cail/README.md`), and heavier
learners/benchmarks are **future work** (PLAN.md §17).
