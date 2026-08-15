# AIRL baseline on MuJoCo (isolated env)

Real AIRL (adversarial IRL) baseline via the `imitation` library, compared head-to-head with
SMOR-BC on **HalfCheetah varying-optimality** demonstrations — CAIL's native benchmark family.

## Why a separate environment
`imitation` pins `gymnasium~=0.29`, which is **incompatible with metaworld 3.1** (needs
gymnasium 1.x). Installing it into the main `deeplearning` env would downgrade gymnasium and
break the Meta-World setup. So AIRL lives in its own conda env; both AIRL and SMOR-BC run there
on the same MuJoCo demos for an apples-to-apples comparison.

## Setup (already done on this machine)
```bash
conda create -n smor-airl python=3.11 -y
PIP=~/miniconda3/envs/smor-airl/bin/pip
$PIP install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
$PIP install gymnasium==0.29.1 stable_baselines3==2.2.1 imitation==1.0.1 mujoco==3.1.6 imageio
$PIP install "numpy<2"        # torch 2.2 needs numpy<2
```
Verified: `AIRL_OK gym 0.29.1 sb3 2.2.1 torch 2.2.2+cpu`; pretrained expert returns ~1573 on
`seals/HalfCheetah-v1`.

## Pipeline (run with the smor-airl python)
```bash
PY=~/miniconda3/envs/smor-airl/bin/python
# 1. varying-optimality demos (expert / medium / noisy) from a corrupted pretrained expert
$PY -m baselines_airl.gen_demos --n-per-source 40 --out results/airl_mujoco
# 2. AIRL vs SMOR-BC (+ uniform / expert-only) on the same demos; metric = eval return
$PY -m baselines_airl.compare_airl_smor --K 2 --bc-steps 400 --airl-steps 300000
```

Demo quality gradient (seed 0): expert ≈ 1660, medium ≈ 758, noisy ≈ −109.

## Notes / limitations
- torch is **CPU** in this env, so AIRL (PPO + discriminator) training is slow; 300k steps is a
  first data point, not converged imitation (HalfCheetah AIRL typically needs ≫1M steps).
- This is the standard `imitation` AIRL (no confidence). CAIL's confidence learning on top of
  AIRL is the remaining piece; the common-backbone CAIL-style confidence (K=1 + ranking) already
  lives in `smor/baselines/cail/` for the solver comparison.
