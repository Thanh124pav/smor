"""SMOR vs CAIL — RETURN-based, on a MODERN MuJoCo env (seals/Ant-v1, seals/HalfCheetah-v1).

Complements the env-free comparison on the official CAIL Ant-v2 buffer (`experiments/cail_compare`):
here the metric is real episodic RETURN via rollout, on a current-gymnasium env whose demos we
generate ourselves (corrupted pretrained PPO expert -> expert/medium/noisy). Same backbone + same
CAIL ranking outer objective; only the hypergradient depth K differs (K=1 = CAIL, K>1 = SMOR).

Runs in the `smor-airl` conda env (gymnasium 0.29 + imitation).

    PY=~/miniconda3/envs/smor-airl/bin/python
    PYTHONPATH=. $PY -m baselines_airl.cail_return --env-id seals/Ant-v1 --seeds 0 1 2 --Ks 1 4
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

warnings.filterwarnings("ignore")

QUALITY = [{"name": "expert", "noise": 0.0}, {"name": "medium", "noise": 0.5},
           {"name": "noisy", "noise": 1.2}]


def gen_sources(env_id, n_per, seed):
    import seals  # noqa
    from imitation.policies.serialize import load_policy
    from imitation.util.util import make_vec_env
    from baselines_airl.gen_demos import corrupt_rollouts
    venv = make_vec_env(env_id, rng=np.random.default_rng(seed), n_envs=1)
    expert = load_policy("ppo-huggingface", venv=venv, env_name=env_id)
    obs_l, act_l, fid_l = [], [], []
    for i, s in enumerate(QUALITY):
        trajs = corrupt_rollouts(expert, venv, n_per, s["noise"], seed=seed + 100 * (i + 1))
        for t in trajs:
            obs_l.append(torch.tensor(t.obs[:-1], dtype=torch.float32))
            act_l.append(torch.tensor(t.acts, dtype=torch.float32))
            fid_l.append(i)
    T = min(o.shape[0] for o in obs_l)
    from smor.envs.demos import DemoDataset
    obs = torch.stack([o[:T] for o in obs_l]); act = torch.stack([a[:T] for a in act_l])
    fid = torch.tensor(fid_l, dtype=torch.int64)
    return DemoDataset(obs=obs, act=act, fidelity=fid)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="seals/Ant-v1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--Ks", type=int, nargs="+", default=[1, 4])
    p.add_argument("--n-per-source", type=int, default=30)
    p.add_argument("--bc-steps", type=int, default=1500)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--out", default="results/cail_return")
    args = p.parse_args()

    from smor.reweighting.grouping import make_groups
    from smor.reweighting.config import OnlineReweighterConfig
    from smor.reweighting.online_reweighter import OnlineReweighter
    from smor.reweighting.outer_objective import CAILRankingLoss
    from smor.learners.bc import BCLearner
    from baselines_airl.mujoco_env import MujocoVecEnv

    methods = ["uniform", "only:expert", "cail_K1"] + [f"smor_K{k}" for k in args.Ks if k > 1]
    agg = {m: [] for m in methods}
    beta_log = {m: [] for m in methods if m.startswith(("cail", "smor"))}

    for seed in args.seeds:
        ds = gen_sources(args.env_id, args.n_per_source, seed)
        env = MujocoVecEnv(args.env_id, horizon=ds.horizon, device="cpu", seed=seed + 7)
        ga = make_groups(ds.fidelity_labels(), group_size=999, seed=seed, whole_fidelity=True)
        gids = list(range(ga.num_groups)); fid = ga.group_fidelity
        quality = {g: float(2 - fid[g]) for g in gids}  # expert(2) > medium(1) > noisy(0)
        print(f"[seed {seed}] {args.env_id} traj={ds.num_trajectories} groups={ga.num_groups}", flush=True)

        def learner():
            return BCLearner(ds, ga, hidden=(256, 256), lr=1e-3, batch_size=256,
                             device="cpu", env=env, seed=seed, data_device="cpu")

        def train_fixed(w):
            lr = learner()
            for _ in range(args.bc_steps):
                lr.train_step(w, lr.sample_batches(gids))
            return lr.evaluate(n_episodes=args.eval_eps)["return_mean"]

        agg["uniform"].append(train_fixed({g: 1.0 / len(gids) for g in gids}))
        eg = [g for g in gids if fid[g] == 0][0]
        agg["only:expert"].append(train_fixed({g: (1.0 if g == eg else 0.0) for g in gids}))

        for K in [1] + [k for k in args.Ks if k > 1]:
            name = "cail_K1" if K == 1 else f"smor_K{K}"
            cfg = OnlineReweighterConfig(n=999, K=K, reweight_interval=5,
                                         n_beta_updates=args.bc_steps // 5, beta_lr=0.3,
                                         neumann_auto=True, batch_size=256, device="cpu", seed=seed)
            lr = learner()
            ev = OnlineReweighter(cfg).fit(lr, ga, outer_objective=CAILRankingLoss(quality),
                                           eval_every=10_000, eval_episodes=args.eval_eps,
                                           env_name=args.env_id)
            agg[name].append(float(ev.eval_history["return_mean"][-1]))
            beta_log[name].append([round(float(x), 3) for x in ev.final_beta])
            print(f"  {name}: return={agg[name][-1]:.0f} beta={beta_log[name][-1]}", flush=True)
        env.close()

    print(f"\n=== {args.env_id}: SMOR vs CAIL (return, seeds={args.seeds}) ===")
    rows = {}
    for m in methods:
        v = agg[m]; mu = mean(v); sd = pstdev(v) if len(v) > 1 else 0.0
        rows[m] = {"return_mean": mu, "return_std": sd}
        print(f"  {m:12s} return = {mu:8.0f} ± {sd:.0f}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"cail_return_{args.env_id.split('/')[-1]}.json").write_text(json.dumps(
        {"env": args.env_id, "seeds": args.seeds, "rows": rows, "beta": beta_log}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
