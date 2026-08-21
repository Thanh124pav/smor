"""SMOR vs CAIL on OFFICIAL Minari offline datasets (D4RL-modern), RETURN-based.

Minari (Farama) ships official offline datasets ``mujoco/<env>/{expert,medium,simple}-v0`` with
stored observations, on current gymnasium MuJoCo (no legacy mujoco_py). We treat the three quality
levels as the varying-optimality sources and compare uniform / only-expert / CAIL(K=1) / SMOR(K>1)
with the CAIL ranking outer objective; metric = rollout return in the dataset's own env.

Runs in the `smor-minari` conda env.

    PY=~/miniconda3/envs/smor-minari/bin/python
    PYTHONPATH=. $PY -m baselines_airl.minari_reweight --env halfcheetah --seeds 0 1 2 --Ks 1 4
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

QUALITIES = ["expert", "medium", "simple"]   # index 0 best .. 2 worst


def load_sources(env_name, n_per_source, seed):
    import minari
    from smor.envs.demos import DemoDataset
    rng = np.random.default_rng(seed)
    obs_l, act_l, fid_l = [], [], []
    env_id = None
    for i, q in enumerate(QUALITIES):
        d = minari.load_dataset(f"mujoco/{env_name}/{q}-v0", download=True)
        env_id = d.spec.env_spec.id
        eps = list(d.iterate_episodes())
        pick = rng.choice(len(eps), size=min(n_per_source, len(eps)), replace=False)
        for j in pick:
            ep = eps[int(j)]
            o = np.asarray(ep.observations, dtype=np.float32)[:-1]
            a = np.asarray(ep.actions, dtype=np.float32)
            n = min(o.shape[0], a.shape[0])
            obs_l.append(torch.tensor(o[:n])); act_l.append(torch.tensor(a[:n])); fid_l.append(i)
    T = min(o.shape[0] for o in obs_l)
    obs = torch.stack([o[:T] for o in obs_l])
    mu = obs.reshape(-1, obs.shape[-1]).mean(0); sd = obs.reshape(-1, obs.shape[-1]).std(0) + 1e-6
    obs = (obs - mu) / sd  # z-score (HalfCheetah/Ant obs have large-scale velocities)
    ds = DemoDataset(obs=obs, act=torch.stack([a[:T] for a in act_l]),
                     fidelity=torch.tensor(fid_l, dtype=torch.int64))
    return ds, env_id, mu.numpy(), sd.numpy()


class _NormVecEnv:
    """Wrap MujocoVecEnv to feed z-scored observations (matching the trained policy's inputs)."""
    def __init__(self, base, mu, sd):
        self.base, self.mu, self.sd = base, torch.tensor(mu), torch.tensor(sd)
        self.horizon, self.obs_dim, self.act_dim = base.horizon, base.obs_dim, base.act_dim

    def _n(self, o):
        return (o - self.mu.to(o.device)) / self.sd.to(o.device)

    def reset(self, b):
        return self._n(self.base.reset(b))

    def step(self, a):
        o, r, d, i = self.base.step(a)
        return self._n(o), r, d, i

    def close(self):
        self.base.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="halfcheetah", help="halfcheetah|hopper|walker2d|ant")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--Ks", type=int, nargs="+", default=[1, 4])
    p.add_argument("--n-per-source", type=int, default=40)
    p.add_argument("--bc-steps", type=int, default=2000)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--grpo", action="store_true",
                   help="add a SMOR arm using GRPO closed-loop RETURN outer loss (vs ranking)")
    p.add_argument("--grpo-episodes", type=int, default=12)
    p.add_argument("--out", default="results/minari")
    args = p.parse_args()

    from smor.reweighting.grouping import make_groups
    from smor.reweighting.config import OnlineReweighterConfig
    from smor.reweighting.online_reweighter import OnlineReweighter
    from smor.reweighting.outer_objective import CAILRankingLoss, ClosedLoopRolloutReturn
    from smor.learners.bc import BCLearner
    from baselines_airl.mujoco_env import MujocoVecEnv

    methods = ["uniform", "only:expert", "cail_K1"] + [f"smor_K{k}" for k in args.Ks if k > 1]
    if args.grpo:
        methods.append("smor_grpo_K4")
    agg = {m: [] for m in methods}
    beta_log = {m: [] for m in methods if m.startswith(("cail", "smor"))}
    env_id = None

    for seed in args.seeds:
        ds, env_id, mu, sd = load_sources(args.env, args.n_per_source, seed)
        env = _NormVecEnv(MujocoVecEnv(env_id, horizon=ds.horizon, device="cpu", seed=seed + 7), mu, sd)
        ga = make_groups(ds.fidelity_labels(), group_size=999, seed=seed, whole_fidelity=True)
        gids = list(range(ga.num_groups)); fid = ga.group_fidelity
        quality = {g: float(2 - fid[g]) for g in gids}  # expert(2)>medium(1)>simple(0)
        print(f"[seed {seed}] {args.env} ({env_id}) traj={ds.num_trajectories} T={ds.horizon}", flush=True)

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
                                           env_name=env_id)
            agg[name].append(float(ev.eval_history["return_mean"][-1]))
            beta_log[name].append([round(float(x), 3) for x in ev.final_beta])
            print(f"  {name}: return={agg[name][-1]:.0f} beta={beta_log[name][-1]}", flush=True)

        if args.grpo:
            # GRPO closed-loop RETURN outer loss (vs ranking): outer signal = real task return,
            # so it should NOT blindly favour expert when diverse data yields higher return.
            cfg = OnlineReweighterConfig(n=999, K=4, reweight_interval=50, n_beta_updates=40,
                                         warmup_steps=0, beta_lr=0.3, neumann_auto=True,
                                         batch_size=256, device="cpu", seed=seed,
                                         normalize_group_grads=True)
            lr = learner()
            ev = OnlineReweighter(cfg).fit(
                lr, ga, outer_objective=ClosedLoopRolloutReturn(
                    n_episodes=args.grpo_episodes, variant="grpo"),
                eval_every=10_000, eval_episodes=args.eval_eps, env_name=env_id)
            agg["smor_grpo_K4"].append(float(ev.eval_history["return_mean"][-1]))
            beta_log["smor_grpo_K4"].append([round(float(x), 3) for x in ev.final_beta])
            print(f"  smor_grpo_K4: return={agg['smor_grpo_K4'][-1]:.0f} "
                  f"beta={beta_log['smor_grpo_K4'][-1]}", flush=True)
        env.close()

    print(f"\n=== Minari {args.env} ({env_id}): SMOR vs CAIL (return, seeds={args.seeds}) ===")
    rows = {}
    for m in methods:
        v = agg[m]; mu = mean(v); sd = pstdev(v) if len(v) > 1 else 0.0
        rows[m] = {"return_mean": mu, "return_std": sd}
        print(f"  {m:12s} return = {mu:8.0f} ± {sd:.0f}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"minari_{args.env}.json").write_text(json.dumps(
        {"env": env_id, "dataset": f"mujoco/{args.env}", "seeds": args.seeds,
         "rows": rows, "beta": beta_log}, indent=2))
    print(f"saved {out}/minari_{args.env}.json")


if __name__ == "__main__":
    main()
