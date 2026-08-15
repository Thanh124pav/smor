"""SMOR vs CAIL(-style) vs baselines on Meta-World multi-fidelity demonstrations.

Real manipulation task; SUCCESS RATE is the primary (non-saturated) metric. Demonstrations come
from several corrupted-scripted "device" sources of different quality. Compares, on the clean
deployment target: only:<source>, uniform, static_quality, CAIL-style (K=1 + confidence ranking),
and SMOR (K>1). Demonstrations are generated ONCE per seed and cached, then reused across all
methods; large datasets are kept on CPU and batched to the GPU.

    python -m experiments.metaworld_reweight --task push-v3 --n-per-source 2000 \
        --outer ranking --seeds 0 1 2 --steps 200 --save-video
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from statistics import mean, pstdev

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.baselines.cail import cail_style_config, group_quality_from_sources
from smor.envs.metaworld_env import METAWORLD_SOURCES, MetaWorldVecEnv, make_metaworld_datasets
from smor.learners.bc import BCLearner
from smor.reweighting.grouping import make_groups
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.reweighting.outer_objective import CAILRankingLoss, ValidationLoss
from smor.utils.seeding import resolve_device, seed_everything


def _total_steps(cfg):
    return cfg.warmup_steps + cfg.n_beta_updates * cfg.reweight_interval


def _save_video(learner, task, horizon, path, seed):
    import numpy as np
    import torch
    from smor.evaluation.video import render_metaworld_video

    def policy_fn(o):
        t = torch.as_tensor(np.asarray(o), dtype=learner.dtype, device=learner.device).unsqueeze(0)
        with torch.no_grad():
            return learner.policy(t)[0].cpu().numpy()

    try:
        render_metaworld_video(task, policy_fn, path, n_episodes=3, horizon=horizon, seed=seed + 500)
    except Exception as e:
        print(f"[video] skipped ({type(e).__name__}: {e})")


def _source_mass(ga, beta_vec):
    fid = ga.group_fidelity
    return {int(f): float(sum(beta_vec[j] for j in range(len(beta_vec)) if fid[j] == f))
            for f in sorted(set(fid.tolist()))}


def main() -> None:
    p = common_parser("SMOR vs CAIL-style on Meta-World multi-fidelity demonstrations.")
    p.add_argument("--task", type=str, default="push-v3")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-per-source", type=int, default=500)
    p.add_argument("--n-sources", type=int, default=3)
    p.add_argument("--demo-horizon", type=int, default=150)
    p.add_argument("--eval-horizon", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=25)
    p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--outer", choices=["val", "ranking"], default="ranking",
                   help="SMOR outer objective (CAIL always uses ranking)")
    p.add_argument("--data-device", type=str, default="cpu",
                   help="where demo tensors live (cpu keeps GPU free for big data)")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    sources = METAWORLD_SOURCES[: args.n_sources]
    names = [s["name"] for s in sources]
    n_src = len(sources)
    best_label = int(min(range(n_src), key=lambda i: sources[i].get("noise", 1.0)
                         + abs(sources[i].get("gain", 1.0) - 1.0)))
    methods = ([f"only:{names[i]}" for i in range(n_src)]
               + ["uniform", "static_quality", "cail", "smor"])
    agg = {m: {"succ": [], "val": []} for m in methods}
    smor_mass = []

    for seed in args.seeds:
        cfg, _, _ = build_configs(raw, {**overrides_from_args(args), "seed": seed})
        device = str(resolve_device(cfg.device))
        seed_everything(seed)

        train, val, names = make_metaworld_datasets(
            task=args.task, n_per_source=args.n_per_source, n_sources=args.n_sources,
            demo_horizon=args.demo_horizon, n_val=args.n_val, seed=seed)
        env = MetaWorldVecEnv(task=args.task, horizon=args.eval_horizon, device=device, seed=seed + 7)
        ga = make_groups(train.fidelity_labels(), group_size=cfg.n, seed=seed, whole_fidelity=True)
        quality = group_quality_from_sources(ga, sources)
        print(f"[seed {seed}] task={args.task} traj={train.num_trajectories} "
              f"groups={ga.num_groups} device={device} data_device={args.data_device}")

        def new_learner():
            return BCLearner(train, ga, hidden=tuple(args.hidden), lr=1e-3,
                             batch_size=cfg.batch_size, device=device, val_data=val, env=env,
                             seed=seed, data_device=args.data_device)

        gids = list(range(ga.num_groups))
        fid = ga.group_fidelity

        def train_fixed(weights):
            lr = new_learner()
            for _ in range(_total_steps(cfg)):
                lr.train_step(weights, lr.sample_batches(gids))
            return lr.evaluate(n_episodes=args.eval_episodes)

        for i in range(n_src):
            w = {g: (1.0 if fid[g] == i else 0.0) for g in gids}
            tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
            m = train_fixed(w)
            agg[f"only:{names[i]}"]["succ"].append(float(m["success_rate"]))
            agg[f"only:{names[i]}"]["val"].append(float(m["val_loss"]))

        m = train_fixed({g: 1.0 / len(gids) for g in gids})
        agg["uniform"]["succ"].append(float(m["success_rate"])); agg["uniform"]["val"].append(float(m["val_loss"]))

        w = {g: (1.0 if fid[g] == best_label else 0.0) for g in gids}
        tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
        m = train_fixed(w)
        agg["static_quality"]["succ"].append(float(m["success_rate"])); agg["static_quality"]["val"].append(float(m["val_loss"]))

        # CAIL-style: common backbone, K=1 one-step + confidence ranking.
        lr_c = new_learner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev_c = OnlineReweighter(cail_style_config(cfg)).fit(
                lr_c, ga, outer_objective=CAILRankingLoss(quality),
                eval_every=10_000, eval_episodes=args.eval_episodes)
        agg["cail"]["succ"].append(float(ev_c.eval_history["success_rate"][-1]))
        agg["cail"]["val"].append(float(ev_c.eval_history["val_loss"][-1]))
        if args.save_video and seed == args.seeds[0]:
            _save_video(lr_c, args.task, args.eval_horizon, f"{args.outdir}/video_cail_{args.task}", seed)

        # SMOR: curvature-aware K>1 + chosen outer objective.
        outer = CAILRankingLoss(quality) if args.outer == "ranking" else ValidationLoss()
        lr_s = new_learner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = OnlineReweighter(cfg).fit(lr_s, ga, outer_objective=outer,
                                           eval_every=10_000, eval_episodes=args.eval_episodes)
        agg["smor"]["succ"].append(float(ev.eval_history["success_rate"][-1]))
        agg["smor"]["val"].append(float(ev.eval_history["val_loss"][-1]))
        smor_mass.append(_source_mass(ga, ev.final_beta))
        if args.save_video and seed == args.seeds[0]:
            _save_video(lr_s, args.task, args.eval_horizon, f"{args.outdir}/video_smor_{args.task}", seed)
        env.close()
        print(f"[seed {seed}] smor mix={ {names[k]: round(v,3) for k,v in smor_mass[-1].items()} }")

    print(f"\ntask={args.task}  sources={names}  naive_best={names[best_label]}  "
          f"seeds={args.seeds}  K={cfg.K}  outer={args.outer}  n_per_source={args.n_per_source}\n")
    print(f"{'method':>18} {'success':>16} {'val_loss':>14}")
    rows = {}
    for m in methods:
        sm, ss = mean(agg[m]["succ"]), (pstdev(agg[m]["succ"]) if len(agg[m]["succ"]) > 1 else 0.0)
        rows[m] = {"success_mean": sm, "success_std": ss, "val_mean": mean(agg[m]["val"])}
        print(f"{m:>18} {sm:>10.3f} ± {ss:<4.3f} {rows[m]['val_mean']:>14.4f}")
    mix = {names[k]: mean([mm[k] for mm in smor_mass]) for k in range(n_src)}
    print(f"\nSMOR learned source mixture: { {k: round(v,3) for k,v in mix.items()} }")
    best_static = max((m for m in methods if m not in ("smor", "cail")),
                      key=lambda m: rows[m]["success_mean"])
    verdict = (f"SMOR({rows['smor']['success_mean']:.3f}) vs CAIL({rows['cail']['success_mean']:.3f}) "
               f"vs best-static {best_static}({rows[best_static]['success_mean']:.3f})")
    print(verdict)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metaworld_reweight.json").write_text(json.dumps(
        {"task": args.task, "seeds": args.seeds, "sources": sources, "outer": args.outer,
         "n_per_source": args.n_per_source, "config": cfg.to_dict(), "rows": rows,
         "smor_mixture": mix, "verdict": verdict}, indent=2))
    print(f"\nsaved {outdir/'metaworld_reweight.json'}")


if __name__ == "__main__":
    main()
