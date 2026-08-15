"""Multi-source reweighting: SMOR vs single-source / uniform (different error structures).

Models two demonstration devices (e.g. SpaceMouse vs teleop) that BOTH cover the whole task
but have different *systematic* error (opposite rotation bias of different magnitude + jitter).
No source is globally best, so the bias-cancelling optimum is a non-trivial interior mixture.

Baselines trained for the same policy-step budget:
  * only:<src>      — all weight on one source
  * uniform         — equal weight on every source
  * static_quality  — naive "trust the lower-noise device" (all weight on it)
  * smor            — learned online reweighting

Success criterion (per the design): SMOR should beat every static baseline on the CLEAN
deployment target (validation loss / return), and its learned beta should be interior.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.envs.demos import DEFAULT_SOURCES
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.reweighting.outer_objective import ClosedLoopReturn, ValidationLoss
from smor.runner import build_multisource_run


def _total_steps(cfg):
    return cfg.warmup_steps + cfg.n_beta_updates * cfg.reweight_interval


def _train_fixed(run, weights, steps):
    learner, ga = run.learner, run.group_assignment
    gids = list(range(ga.num_groups))
    for _ in range(steps):
        learner.train_step(weights, learner.sample_batches(gids))
    return learner.evaluate(n_episodes=128)


def _source_mass(ga, beta_vec):
    fid = ga.group_fidelity
    return {int(f): float(sum(beta_vec[j] for j in range(len(beta_vec)) if fid[j] == f))
            for f in sorted(set(fid.tolist()))}


def main() -> None:
    parser = common_parser("Multi-source: SMOR vs single-source/uniform.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-per-source", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=22,
                        help="shorter horizon makes return more sensitive to systematic error")
    parser.add_argument("--whole-fidelity", action="store_true", default=True)
    parser.add_argument("--outer", choices=["val", "return"], default="return",
                        help="SMOR outer objective: open-loop val MSE or differentiable "
                             "closed-loop return surrogate")
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    names = [s["name"] for s in DEFAULT_SOURCES]
    n_src = len(DEFAULT_SOURCES)
    # naive "best fidelity" = lowest-noise source
    best_label = int(min(range(n_src), key=lambda i: DEFAULT_SOURCES[i].get("noise", 0.0)))

    methods = [f"only:{names[i]}" for i in range(n_src)] + ["uniform", "static_quality", "smor"]
    agg = {m: {"val": [], "ret": []} for m in methods}
    smor_mass = []

    for seed in args.seeds:
        cfg, _, _ = build_configs(raw, {**overrides_from_args(args), "seed": seed})

        for i in range(n_src):
            run = build_multisource_run(cfg, n_per_source=args.n_per_source, horizon=args.horizon,
                                        whole_fidelity=args.whole_fidelity, seed=seed)
            gids = list(range(run.group_assignment.num_groups))
            fid = run.group_assignment.group_fidelity
            w = {g: (1.0 if fid[g] == i else 0.0) for g in gids}
            tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
            m = _train_fixed(run, w, _total_steps(cfg))
            agg[f"only:{names[i]}"]["val"].append(float(m["val_loss"]))
            agg[f"only:{names[i]}"]["ret"].append(float(m["return_mean"]))

        run = build_multisource_run(cfg, n_per_source=args.n_per_source, horizon=args.horizon,
                                    whole_fidelity=args.whole_fidelity, seed=seed)
        gids = list(range(run.group_assignment.num_groups))
        M = len(gids)
        m = _train_fixed(run, {g: 1.0 / M for g in gids}, _total_steps(cfg))
        agg["uniform"]["val"].append(float(m["val_loss"])); agg["uniform"]["ret"].append(float(m["return_mean"]))

        run = build_multisource_run(cfg, n_per_source=args.n_per_source, horizon=args.horizon,
                                    whole_fidelity=args.whole_fidelity, seed=seed)
        gids = list(range(run.group_assignment.num_groups))
        fid = run.group_assignment.group_fidelity
        w = {g: (1.0 if fid[g] == best_label else 0.0) for g in gids}
        tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
        m = _train_fixed(run, w, _total_steps(cfg))
        agg["static_quality"]["val"].append(float(m["val_loss"])); agg["static_quality"]["ret"].append(float(m["return_mean"]))

        run = build_multisource_run(cfg, n_per_source=args.n_per_source, horizon=args.horizon,
                                    whole_fidelity=args.whole_fidelity, seed=seed)
        outer = ClosedLoopReturn() if args.outer == "return" else ValidationLoss()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = OnlineReweighter(cfg).fit(run.learner, run.group_assignment,
                                           outer_objective=outer,
                                           eval_every=10_000, eval_episodes=128)
        agg["smor"]["val"].append(float(ev.eval_history["val_loss"][-1]))
        agg["smor"]["ret"].append(float(ev.eval_history["return_mean"][-1]))
        smor_mass.append(_source_mass(run.group_assignment, ev.final_beta))
        print(f"[seed {seed}] done  smor beta(source)="
              f"{ {names[k]: round(v,3) for k,v in _source_mass(run.group_assignment, ev.final_beta).items()} }")

    print(f"\nsources: {names}  (naive best-fidelity = {names[best_label]})  seeds={args.seeds}  K={cfg.K}\n")
    print(f"{'method':>18} {'val_loss':>16} {'return':>16}")
    rows = {}
    for m in methods:
        vv, vs = mean(agg[m]["val"]), (pstdev(agg[m]["val"]) if len(agg[m]["val"]) > 1 else 0.0)
        rv, rs = mean(agg[m]["ret"]), (pstdev(agg[m]["ret"]) if len(agg[m]["ret"]) > 1 else 0.0)
        rows[m] = {"val_mean": vv, "val_std": vs, "ret_mean": rv, "ret_std": rs}
        print(f"{m:>18} {vv:>10.4f} ± {vs:<4.4f} {rv:>9.2f} ± {rs:<5.2f}")

    mean_mass = {names[k]: mean([mm[k] for mm in smor_mass]) for k in range(n_src)}
    print(f"\nSMOR learned source mixture (mean): "
          f"{ {k: round(v,3) for k,v in mean_mass.items()} }")

    best_static = min((m for m in methods if m != "smor"), key=lambda m: rows[m]["val_mean"])
    verdict = ("SMOR beats every static baseline"
               if rows["smor"]["val_mean"] < rows[best_static]["val_mean"]
               else f"SMOR does NOT beat {best_static}")
    print(f"best static baseline = {best_static} (val {rows[best_static]['val_mean']:.4f}); "
          f"smor val {rows['smor']['val_mean']:.4f}  ->  {verdict}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "compare_sources.json").write_text(json.dumps(
        {"seeds": args.seeds, "sources": DEFAULT_SOURCES, "config": cfg.to_dict(),
         "rows": rows, "smor_mixture": mean_mass, "verdict": verdict}, indent=2))
    print(f"\nsaved {outdir/'compare_sources.json'}")


if __name__ == "__main__":
    main()
