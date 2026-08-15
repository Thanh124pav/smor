"""SMOR on Meta-World: reweighting multi-fidelity (corrupted-scripted) demonstrations.

Real manipulation task where SUCCESS RATE is not saturated. Compares, on the clean
deployment target: only:<source>, uniform, static_quality (naive trust-lowest-noise), and SMOR
(learned reweighting, ValidationLoss outer objective). Reports success rate (primary) + val loss
+ the learned source mixture, averaged over seeds.

    python -m experiments.metaworld_reweight --task reach-v3 --seeds 0 1 --steps 60
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from dataclasses import replace

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.baselines.cail import cail_style_config, group_quality_from_sources
from smor.envs.metaworld_env import METAWORLD_SOURCES
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.reweighting.outer_objective import CAILRankingLoss, ValidationLoss
from smor.runner import build_metaworld_run


def _total_steps(cfg):
    return cfg.warmup_steps + cfg.n_beta_updates * cfg.reweight_interval


def _train_fixed(run, weights, steps, eval_episodes):
    learner, ga = run.learner, run.group_assignment
    gids = list(range(ga.num_groups))
    for _ in range(steps):
        learner.train_step(weights, learner.sample_batches(gids))
    return learner.evaluate(n_episodes=eval_episodes)


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
    except Exception as e:  # video is best-effort, never fail the experiment
        print(f"[video] skipped ({type(e).__name__}: {e})")


def _source_mass(ga, beta_vec):
    fid = ga.group_fidelity
    return {int(f): float(sum(beta_vec[j] for j in range(len(beta_vec)) if fid[j] == f))
            for f in sorted(set(fid.tolist()))}


def main() -> None:
    p = common_parser("SMOR on Meta-World multi-fidelity demonstrations.")
    p.add_argument("--task", type=str, default="reach-v3")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-per-source", type=int, default=40)
    p.add_argument("--demo-horizon", type=int, default=150)
    p.add_argument("--eval-horizon", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=25)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--save-video", action="store_true", help="render SMOR + CAIL rollouts to mp4")
    args = p.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    names = [s["name"] for s in METAWORLD_SOURCES]
    n_src = len(METAWORLD_SOURCES)
    best_label = int(min(range(n_src), key=lambda i: METAWORLD_SOURCES[i].get("noise", 0.0)))
    methods = ([f"only:{names[i]}" for i in range(n_src)]
               + ["uniform", "static_quality", "cail", "smor"])
    agg = {m: {"succ": [], "val": []} for m in methods}
    smor_mass = []

    def _build(seed):
        return build_metaworld_run(
            cfg, task=args.task, n_per_source=args.n_per_source,
            demo_horizon=args.demo_horizon, eval_horizon=args.eval_horizon,
            n_val=args.n_val, whole_fidelity=True, seed=seed)

    for seed in args.seeds:
        cfg, _, _ = build_configs(raw, {**overrides_from_args(args), "seed": seed})

        for i in range(n_src):
            run = _build(seed); gids = list(range(run.group_assignment.num_groups))
            fid = run.group_assignment.group_fidelity
            w = {g: (1.0 if fid[g] == i else 0.0) for g in gids}
            tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
            m = _train_fixed(run, w, _total_steps(cfg), args.eval_episodes)
            agg[f"only:{names[i]}"]["succ"].append(float(m["success_rate"]))
            agg[f"only:{names[i]}"]["val"].append(float(m["val_loss"]))
            run.env.close()

        run = _build(seed); gids = list(range(run.group_assignment.num_groups)); M = len(gids)
        m = _train_fixed(run, {g: 1.0 / M for g in gids}, _total_steps(cfg), args.eval_episodes)
        agg["uniform"]["succ"].append(float(m["success_rate"])); agg["uniform"]["val"].append(float(m["val_loss"]))
        run.env.close()

        run = _build(seed); gids = list(range(run.group_assignment.num_groups))
        fid = run.group_assignment.group_fidelity
        w = {g: (1.0 if fid[g] == best_label else 0.0) for g in gids}
        tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
        m = _train_fixed(run, w, _total_steps(cfg), args.eval_episodes)
        agg["static_quality"]["succ"].append(float(m["success_rate"])); agg["static_quality"]["val"].append(float(m["val_loss"]))
        run.env.close()

        # CAIL-style baseline: common backbone, K=1 one-step + CAIL confidence-ranking loss.
        run = _build(seed)
        cail_cfg = cail_style_config(cfg)
        quality = group_quality_from_sources(run.group_assignment, METAWORLD_SOURCES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev_c = OnlineReweighter(cail_cfg).fit(
                run.learner, run.group_assignment,
                outer_objective=CAILRankingLoss(quality),
                eval_every=10_000, eval_episodes=args.eval_episodes)
        agg["cail"]["succ"].append(float(ev_c.eval_history["success_rate"][-1]))
        agg["cail"]["val"].append(float(ev_c.eval_history["val_loss"][-1]))
        if args.save_video and seed == args.seeds[0]:
            _save_video(run.learner, args.task, args.eval_horizon,
                        f"{args.outdir}/video_cail_{args.task}", seed)
        run.env.close()

        # SMOR: curvature-aware (K from config) + validation outer objective.
        run = _build(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = OnlineReweighter(cfg).fit(run.learner, run.group_assignment,
                                           outer_objective=ValidationLoss(),
                                           eval_every=10_000, eval_episodes=args.eval_episodes)
        agg["smor"]["succ"].append(float(ev.eval_history["success_rate"][-1]))
        agg["smor"]["val"].append(float(ev.eval_history["val_loss"][-1]))
        smor_mass.append(_source_mass(run.group_assignment, ev.final_beta))
        if args.save_video and seed == args.seeds[0]:
            _save_video(run.learner, args.task, args.eval_horizon,
                        f"{args.outdir}/video_smor_{args.task}", seed)
        run.env.close()
        print(f"[seed {seed}] smor mix="
              f"{ {names[k]: round(v,3) for k,v in smor_mass[-1].items()} }")

    print(f"\ntask={args.task}  sources={names}  naive_best={names[best_label]}  "
          f"seeds={args.seeds}  K={cfg.K}\n")
    print(f"{'method':>18} {'success':>16} {'val_loss':>14}")
    rows = {}
    for m in methods:
        sm, ss = mean(agg[m]["succ"]), (pstdev(agg[m]["succ"]) if len(agg[m]["succ"]) > 1 else 0.0)
        vv = mean(agg[m]["val"])
        rows[m] = {"success_mean": sm, "success_std": ss, "val_mean": vv}
        print(f"{m:>18} {sm:>10.3f} ± {ss:<4.3f} {vv:>14.4f}")
    mix = {names[k]: mean([mm[k] for mm in smor_mass]) for k in range(n_src)}
    print(f"\nSMOR learned source mixture: { {k: round(v,3) for k,v in mix.items()} }")
    best_static = max((m for m in methods if m != "smor"), key=lambda m: rows[m]["success_mean"])
    verdict = ("SMOR >= best static on success"
               if rows["smor"]["success_mean"] >= rows[best_static]["success_mean"] - 1e-9
               else f"SMOR below {best_static}")
    print(f"best static = {best_static} (succ {rows[best_static]['success_mean']:.3f}); "
          f"smor {rows['smor']['success_mean']:.3f} -> {verdict}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metaworld_reweight.json").write_text(json.dumps(
        {"task": args.task, "seeds": args.seeds, "sources": METAWORLD_SOURCES,
         "config": cfg.to_dict(), "rows": rows, "smor_mixture": mix, "verdict": verdict}, indent=2))
    print(f"\nsaved {outdir/'metaworld_reweight.json'}")


if __name__ == "__main__":
    main()
