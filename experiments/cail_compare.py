"""SMOR vs CAIL online reweighting on the **official CAIL dataset** (n=1, per-demonstration).

This isolates the *reweighting update* itself. CAIL (Zhang et al., NeurIPS 2021) learns a per-
demonstration confidence via a bilevel objective solved with a **one-step** hypergradient
approximation; SMOR is the same bilevel with a **curvature-aware K>1** hypergradient. We run BOTH
on the same official CAIL buffer, with the SAME BC backbone, the SAME schedule, and the SAME
CAIL supervision (a small fraction of trajectories labelled with their return, used as a pairwise
ranking outer objective) — the ONLY difference is the hypergradient depth K. Grouping is n=1: one
confidence weight per demonstration (multi-fidelity is not used here).

Because the CAIL buffers were collected in legacy Ant-v2/Reacher-v2 (mujoco_py) and do not
transfer to the current gymnasium MuJoCo (a BC policy that perfectly fits the demos still falls
over in Ant-v4), evaluation is **env-free** and measures reweighting quality directly:
  * expert_mse  — action-MSE of the final reweighted policy on HELD-OUT expert (top-policy) demos
                  that are never used for training or labelling (lower = better imitation);
  * spearman    — rank correlation of the learned per-demo weight with the demo's TRUE return
                  (higher = the reweighting correctly identifies good demonstrations);
  * mass_top1/2 — fraction of weight the method puts on the best (top-2) source policies.

    python -m experiments.cail_compare --env-id Ant-v2 --label 0.05 --Ks 1 2 4 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from smor.data.cail import load_cail_buffer
from smor.data.trajectory_dataset import TrajectoryDataset
from smor.learners.bc import BCLearner
from smor.reweighting.config import OnlineReweighterConfig
from smor.reweighting.grouping import make_groups
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.reweighting.outer_objective import CAILRankingLoss
from smor.utils.seeding import resolve_device, seed_everything


def _normalize(ds: TrajectoryDataset, mu, sd) -> TrajectoryDataset:
    return TrajectoryDataset(obs=(ds.obs - mu) / sd, act=ds.act, traj_id=ds.traj_id,
                             fidelity=ds.fidelity)


def _spearman(a, b) -> float:
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    ar = (ar - ar.mean()); br = (br - br.mean())
    denom = np.sqrt((ar**2).sum() * (br**2).sum())
    return float((ar * br).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", type=str, default="Ant-v2")
    p.add_argument("--root", type=str, default="data/cail")
    p.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-per-source", type=int, default=None,
                   help="subsample this many demos per source policy (n=1 over all demos is slow); "
                        "None uses all")
    p.add_argument("--label", type=float, default=0.05, help="fraction of demos labelled by return")
    p.add_argument("--n-holdout-expert", type=int, default=10,
                   help="top-policy demos held out (never trained/labelled) for the eval metric")
    p.add_argument("--beta-updates", type=int, default=150)
    p.add_argument("--reweight-interval", type=int, default=5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--beta-lr", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="results/cail_compare")
    args = p.parse_args()

    methods = ["uniform"] + [(f"cail_K1" if k == 1 else f"smor_K{k}") for k in args.Ks]
    agg = {m: {"expert_mse": [], "spearman": [], "mass_top1": [], "mass_top2": []} for m in methods}

    cail = load_cail_buffer(env_id=args.env_id, root=args.root)
    ds, returns, source = cail.dataset, cail.returns, cail.source
    if args.n_per_source is not None:
        keep = []
        for s in np.unique(source):
            keep += np.where(source == s)[0][: args.n_per_source].tolist()
        keep = sorted(keep)
        ds = ds.subset(keep)
        returns, source = returns[keep], source[keep]
    N = ds.num_trajectories
    print(f"CAIL {args.env_id}: {N} demos, obs {ds.obs_dim}, act {ds.act_dim}, "
          f"returns [{returns.min():.0f},{returns.max():.0f}], sources {np.bincount(source).tolist()}")

    for seed in args.seeds:
        seed_everything(seed)
        device = str(resolve_device(args.device))
        rng = np.random.default_rng(seed)

        # hold out the best (top-return) expert demos as the eval target (never trained/labelled)
        expert_pool = np.where(source == 0)[0]
        holdout = expert_pool[np.argsort(returns[expert_pool])[::-1][: args.n_holdout_expert]]
        train_idx = np.array([i for i in range(N) if i not in set(holdout.tolist())])

        train_ds = ds.subset(train_idx.tolist())
        train_ret = returns[train_idx]
        train_src = source[train_idx]
        val_ds = ds.subset(holdout.tolist())

        # obs normalization from training demos (shared by all methods)
        mu = train_ds.obs.mean(0, keepdim=True); sd = train_ds.obs.std(0, keepdim=True) + 1e-6
        train_ds = _normalize(train_ds, mu, sd); val_ds = _normalize(val_ds, mu, sd)

        # n=1: one group (confidence) per demonstration
        ga = make_groups(np.zeros(train_ds.num_trajectories, dtype=np.int64),
                         group_size=1, seed=seed)
        gids = list(range(ga.num_groups))
        # group -> its trajectory's true return / source (n=1 => one member each)
        grp_ret = np.array([train_ret[ga.members[g][0]] for g in gids])
        grp_src = np.array([train_src[ga.members[g][0]] for g in gids])

        # CAIL supervision: label `--label` fraction of demos with their return (ranking)
        n_lab = max(2, int(round(args.label * len(gids))))
        labelled = rng.choice(gids, size=n_lab, replace=False)
        quality = {int(g): float(grp_ret[g]) for g in labelled}
        print(f"[seed {seed}] train demos={len(gids)} labelled={n_lab} holdout_expert={len(holdout)}")

        def new_learner():
            return BCLearner(train_ds, ga, hidden=tuple(args.hidden), lr=1e-3,
                             batch_size=args.batch_size, device=device, val_data=val_ds,
                             seed=seed, data_device=args.data_device)

        def metrics_from_beta(beta_vec, expert_mse):
            sp = _spearman(beta_vec, grp_ret)
            m1 = float(beta_vec[grp_src == 0].sum())
            m2 = float(beta_vec[np.isin(grp_src, [0, 1])].sum())
            return {"expert_mse": expert_mse, "spearman": sp, "mass_top1": m1, "mass_top2": m2}

        total_steps = args.warmup + args.beta_updates * args.reweight_interval

        # uniform baseline
        lr = new_learner()
        w = {g: 1.0 / len(gids) for g in gids}
        for _ in range(total_steps):
            lr.train_step(w, lr.sample_batches(gids))
        um = lr.evaluate(n_episodes=0)["val_loss"]
        beta_u = np.full(len(gids), 1.0 / len(gids))
        r = metrics_from_beta(beta_u, um)
        for k in agg["uniform"]:
            agg["uniform"][k].append(r[k])
        print(f"  uniform   expert_mse={um:.4f}")

        # CAIL (K=1) and SMOR (K>1): same ranking outer objective, only K differs
        for K in args.Ks:
            name = "cail_K1" if K == 1 else f"smor_K{K}"
            cfg = OnlineReweighterConfig(
                n=1, K=K, reweight_interval=args.reweight_interval,
                n_beta_updates=args.beta_updates, warmup_steps=args.warmup,
                beta_lr=args.beta_lr, batch_size=args.batch_size, neumann_auto=True,
                device=device, seed=seed)
            lr = new_learner()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev = OnlineReweighter(cfg).fit(lr, ga, outer_objective=CAILRankingLoss(quality),
                                               eval_every=10_000, eval_episodes=0,
                                               env_name=f"cail:{args.env_id}")
            em = ev.eval_history["val_loss"][-1]
            r = metrics_from_beta(np.asarray(ev.final_beta), em)
            for k in agg[name]:
                agg[name][k].append(r[k])
            print(f"  {name:8s} expert_mse={em:.4f} spearman={r['spearman']:.3f} "
                  f"mass_top1={r['mass_top1']:.3f} mass_top2={r['mass_top2']:.3f}")

    # summary
    print(f"\n=== CAIL {args.env_id}: SMOR vs CAIL online reweighting (n=1, label={args.label}, "
          f"seeds={args.seeds}) ===")
    print(f"{'method':>10} {'expert_mse':>18} {'spearman(beta,ret)':>20} {'mass_top1':>12} {'mass_top2':>12}")
    rows = {}
    for m in methods:
        def ms(key):
            v = agg[m][key]
            return mean(v), (pstdev(v) if len(v) > 1 else 0.0)
        em, ems = ms("expert_mse"); sp, sps = ms("spearman")
        m1, _ = ms("mass_top1"); m2, _ = ms("mass_top2")
        rows[m] = {"expert_mse": em, "expert_mse_std": ems, "spearman": sp,
                   "mass_top1": m1, "mass_top2": m2}
        print(f"{m:>10} {em:>10.4f}±{ems:<5.4f} {sp:>18.3f} {m1:>12.3f} {m2:>12.3f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"cail_compare_{args.env_id}.json").write_text(json.dumps(
        {"env_id": args.env_id, "label": args.label, "Ks": args.Ks, "seeds": args.seeds,
         "n_holdout_expert": args.n_holdout_expert, "rows": rows}, indent=2))
    print(f"\nsaved {outdir}/cail_compare_{args.env_id}.json")


if __name__ == "__main__":
    main()
