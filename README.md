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
  data/          official-dataset loaders: ragged TrajectoryDataset + robomimic/ (PH/MH/MG)
  envs/          small GPU-friendly point-mass goal env + two-fidelity demo generator
                 + robosuite_env.py (optional real RoboMimic rollout eval)
  evaluation/    real env rollouts, correlation metrics, long-horizon utility (§8)
  baselines/cail/  CAIL/AIRL adapter STUB + original config + reproduction notes (Stage A)
  evidence.py    ReweightingEvidence output object (§16)
  runner.py      point-mass run factory
experiments/     collect_demos, train_reweighting, validate_hvp, compare_K, compare_n,
                 run_reweighting_grid, metaworld_reweight,
                 robomimic_reweight + robomimic_download (official RoboMimic data)
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

## Scripts

Convenience wrappers in `scripts/` (they activate the `deeplearning` conda env, `cd` to the
repo root, and pick a device). Control the device with `DEVICE=cuda|cpu|auto` (default `auto`).

```bash
bash scripts/smoke.sh              # fast end-to-end flow check (tests + a tiny run of every driver)
bash scripts/run_experiments.sh    # full suite -> results/ (reproduces RESULTS_REWEIGHTING.md)
DEVICE=cpu bash scripts/smoke.sh   # force CPU
```

The models are tiny MLPs: a full run peaks at ~**20 MB VRAM**, so a 4 GB GPU (e.g. GTX 1650) is
more than enough; CPU also works. The only slow cells are `n=1` (one β cell per trajectory →
many HVPs) — override granularities via `NS_COMPARE`/`NS_GRID` env vars if needed.

## Real datasets — RoboMimic (official, multi-fidelity)

Besides the synthetic point-mass / Meta-World sources, SMOR runs directly on the **official
[RoboMimic](https://robomimic.github.io/) datasets**, which ship demonstrations of *labelled,
varying human quality* — the real version of the multi-fidelity problem this module reweights:

- **`ph`** proficient-human (200 clean teleop demos), **`mh`** multi-human (300 demos from 6
  operators, split by skill into `better` / `okay` / `worse` tiers), **`mg`** machine-generated
  (SAC rollouts of mixed success). The **fidelity label is the real quality tier / variant.**

The loader reads the low-dim HDF5 files (needs only `h5py`; no MuJoCo for training), concatenates
the canonical low-dim obs keys, and returns a variable-length
[`TrajectoryDataset`](smor/data/trajectory_dataset.py) — the reweighting core and `BCLearner` run
unchanged. Datasets download on demand into `data/robomimic/` (or `$SMOR_ROBOMIMIC_ROOT`).

```bash
# optional pre-download (else it fetches on first use)
python -m experiments.robomimic_download --mix mh-tiers

# canonical multi-human quality-tier reweighting (held-out val-loss eval — default, no MuJoCo)
python -m experiments.robomimic_reweight --config configs/robomimic_reweight.yaml \
    --mix mh-tiers --K 4 --steps 200 --seeds 0 1 2

# arbitrary mixes via a DSL: proficient target (*) + weak human tier + machine failures
python -m experiments.robomimic_reweight --mix "lift:ph*,lift:mh:worse,lift:mg:mg_fail"

# real success-rate eval in the reconstructed robosuite env (needs: pip install robomimic robosuite)
python -m experiments.robomimic_reweight --mix mh-tiers --rollout
```

### Non-trivial reweighting: heterogeneous sources with different influence

The quality-tier mix above has a *trivial* optimum (put all weight on the cleanest tier). For a
**non-trivial** study, [`robomimic_multisource.py`](experiments/robomimic_multisource.py) treats
the same real task collected through several mis-calibrated teleop **devices** (each a different
anisotropic gain / rotation bias on the real actions — see
[`smor/data/robomimic/multisource.py`](smor/data/robomimic/multisource.py)). No single device
works (rollout success ~0.1–0.6), so you **must combine** them, and SMOR learns an **interior**
mixture:

```bash
# device-calibration sources; single sources fail, SMOR finds an interior blend (val-loss eval)
python -m experiments.robomimic_multisource --task lift --dtype ph --K 4 --steps 200 --seeds 0 1 2

# + real "poison" (MG failed rollouts): now uniform is suboptimal, so SMOR BEATS uniform AND every
#   single source while keeping an interior blend of the good sources and driving poison -> 0
python -m experiments.robomimic_multisource --poison --K 4 --steps 200 --seeds 0 1 2

# closed-loop success-rate eval in robosuite (headless, no rendering) instead of val-loss
python -m experiments.robomimic_multisource --poison --rollout
```

On real `lift/ph` + MG-fail poison the `--poison` run gives e.g. `beta ≈ [0.27, 0.32, 0.41, 0.0]`
(interior over the three good devices, poison rejected) at `val ≈ 0.028` vs `uniform ≈ 0.035` and
`best-single ≈ 0.037`. Without `--poison`, the interior optimum instead *ties* uniform — because
the complementary biases already cancel under naive averaging (a real, documented finding, not a
bug). The optional robosuite rollout needs `pip install "robosuite>=1.4,<1.5" mujoco` (state-based,
no OpenGL). Closed-loop **magnifies** the gap — on `lift/ph` + poison the systematic errors
compound over the episode:

| method | rollout success | val loss |
|---|---|---|
| uniform (keeps poison) | **0.00** | 0.031 |
| best single device | 0.80 | 0.032 |
| **SMOR interior mixture** | **1.00** | 0.025 |

(uniform's validation loss looks only slightly worse, but its rollout success collapses to 0 — the
poison actions compound; the interior mixture that rejects the poison lifts every time.)

Presets: `mh-tiers`, `ph-plus-mh`, `ph-plus-mg`, `full-spectrum` (see
[`smor/data/robomimic/spec.py`](smor/data/robomimic/spec.py)); DSL components are
`task:dtype[:tier][:n]`, `*` marks the clean deployment target. Tasks: `lift`, `can`, `square`,
`transport`, `tool_hang` (override a preset's task with `--task can`). The two eval options
(held-out val-loss vs robosuite rollout success rate) are both implemented; **val-loss is the
default** and needs no MuJoCo.

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
