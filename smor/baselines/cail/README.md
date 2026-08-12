# CAIL baseline (Stage A) — reproduction notes

**Status: not run inside Module 1.** This directory pins the *interface* and the *exact upstream
setting* for the original CAIL baseline. The live reproduction is a separate pass because it
needs the upstream MuJoCo environment, which is incompatible with this repo's lightweight,
GPU-friendly core.

## Paper & code
- Zhang, Cao, Sadigh, Sui. *Confidence-Aware Imitation Learning from Demonstrations with Varying
  Optimality.* NeurIPS 2021. https://arxiv.org/abs/2110.14754
- Upstream: https://github.com/Stanford-ILIAD/Confidence-Aware-Imitation-Learning

CAIL is an **AIRL-based** instantiation with confidence reweighting, a bilevel formulation, a
pseudo-update, and a one-step approximation. Our common-backbone `K=1` (see
`smor/reweighting/`) is the apples-to-apples solver analogue; the *original* AIRL-CAIL and our
common-backbone ablations must be reported **separately** (PLAN.md §3.1–§3.2).

## Upstream dependencies (legacy)
- Python 3.6+ (the upstream target)
- MuJoCo + `mujoco_py`
- PyTorch, OpenAI Gym, NumPy, SciPy, Pandas, Matplotlib, TensorBoard, tqdm

These do **not** coexist cleanly with this repo's `deeplearning` env (Python 3.12, torch 2.11).
Use a **separate** virtualenv/conda env for the reproduction.

## Reproduction steps (Stage A)
1. Create an isolated env; install MuJoCo + `mujoco_py` + the upstream `requirements.txt`.
2. Clone upstream at a pinned commit; record the commit hash in
   `configs/cail_original.yaml` (`upstream_commit`).
3. Train one small MuJoCo setting first, e.g.:
   ```bash
   python train_imitation.py --algo cail --env-id Ant-v2 \
       --buffer <demo_buffer> --label 0.05 --lr-conf 0.1 --pre-train 5000000
   ```
4. Verify: learning-curve shape, final return in a plausible range, and that confidence weights
   correlate with demonstration quality.
5. Save config, seed, logs, checkpoint, and reproduction notes (PLAN.md Stage A / §13).

## Adapter
`adapter.py` defines `CAILAIRLAdapter(WeightedLearner)` as a stub. When Stage A is set up, wrap
the upstream AIRL discriminator/policy behind the `WeightedLearner` contract so the reweighting
core can drive CAIL exactly as it drives the BC learner today — no changes to the reweighter.

Do not begin the proposed method's CAIL comparison until a minimal reproduction passes
(PLAN.md Stage A).
