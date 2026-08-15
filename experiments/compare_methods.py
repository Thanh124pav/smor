"""Head-to-head: uniform vs SMOR (learned reweighting), on the light point-mass task.

Baselines (PLAN.md §3.3):
  * uniform         — fixed beta_j = 1/M (no reweighting)
  * static_quality  — oracle: all mass on the expert (high-fidelity) groups (diagnostic;
                      uses fidelity labels, so it is an upper-reference, not a fair method)
  * smor            — online curvature-aware reweighting (K from the config)

All methods share the same dataset, grouping, env and policy init per seed, and train for the
SAME number of policy-gradient steps, so the only difference is how the group weights are set.
Reports episodic return + success rate + validation loss, averaged over seeds.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.runner import build_pointmass_run


def _total_policy_steps(cfg) -> int:
    return cfg.warmup_steps + cfg.n_beta_updates * cfg.reweight_interval


def _train_fixed(setup, weights, steps):
    learner, ga = setup.learner, setup.group_assignment
    gids = list(range(ga.num_groups))
    for _ in range(steps):
        learner.train_step(weights, learner.sample_batches(gids))
    metrics = learner.evaluate(n_episodes=128)
    fid = ga.group_fidelity
    expert_mass = float(sum(weights[g] for g in gids if fid[g] == 0))
    return metrics, expert_mass


def run_uniform(cfg, dcfg, seed):
    setup = build_pointmass_run(cfg, dcfg, seed=seed)
    M = setup.group_assignment.num_groups
    w = {g: 1.0 / M for g in range(M)}
    return _train_fixed(setup, w, _total_policy_steps(cfg))


def run_static_quality(cfg, dcfg, seed):
    setup = build_pointmass_run(cfg, dcfg, seed=seed)
    ga = setup.group_assignment
    fid = ga.group_fidelity
    expert = [g for g in range(ga.num_groups) if fid[g] == 0]
    floor = cfg.beta_floor
    w = {g: (floor) for g in range(ga.num_groups)}
    share = (1.0 - floor * ga.num_groups) / max(1, len(expert))
    for g in expert:
        w[g] += share
    return _train_fixed(setup, w, _total_policy_steps(cfg))


def run_smor(cfg, dcfg, seed):
    setup = build_pointmass_run(cfg, dcfg, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = OnlineReweighter(cfg).fit(
            setup.learner, setup.group_assignment, eval_every=10_000, eval_episodes=128,
        )
    fid = setup.group_assignment.group_fidelity
    fb = ev.final_beta
    metrics = {
        "return_mean": ev.eval_history["return_mean"][-1],
        "success_rate": ev.eval_history["success_rate"][-1],
        "val_loss": ev.eval_history["val_loss"][-1],
    }
    return metrics, float(fb[fid == 0].sum())


METHODS = {"uniform": run_uniform, "static_quality": run_static_quality, "smor": run_smor}


def main() -> None:
    parser = common_parser("Compare uniform vs SMOR on the point-mass task.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["uniform", "static_quality", "smor"])
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}

    agg: dict[str, dict[str, list]] = {m: {"return": [], "success": [], "val": [], "emass": []}
                                       for m in args.methods}
    for seed in args.seeds:
        cfg, dcfg, _ = build_configs(raw, {**overrides_from_args(args), "seed": seed})
        for m in args.methods:
            metrics, emass = METHODS[m](cfg, dcfg, seed)
            agg[m]["return"].append(float(metrics["return_mean"]))
            agg[m]["success"].append(float(metrics["success_rate"]))
            agg[m]["val"].append(float(metrics.get("val_loss", float("nan"))))
            agg[m]["emass"].append(emass)
        print(f"[seed {seed}] done")

    def ms(xs):
        return (mean(xs), pstdev(xs) if len(xs) > 1 else 0.0)

    print(f"\nenv=point_mass  seeds={args.seeds}  K={cfg.K}  n={cfg.n}  "
          f"groups(M) fixed per seed\n")
    print(f"{'method':>15} {'return':>18} {'success':>14} {'val_loss':>14} {'expert_mass':>13}")
    rows = {}
    for m in args.methods:
        r_m, r_s = ms(agg[m]["return"]); s_m, s_s = ms(agg[m]["success"])
        v_m, _ = ms(agg[m]["val"]); e_m, _ = ms(agg[m]["emass"])
        rows[m] = {"return_mean": r_m, "return_std": r_s, "success_mean": s_m,
                   "val_loss_mean": v_m, "expert_mass_mean": e_m}
        print(f"{m:>15} {r_m:>10.2f} ± {r_s:<5.2f} {s_m:>10.2f} ± {s_s:<3.2f} "
              f"{v_m:>13.4f} {e_m:>13.3f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "compare_methods.json").write_text(json.dumps(
        {"seeds": args.seeds, "config": cfg.to_dict(), "rows": rows}, indent=2))
    print(f"\nsaved {outdir/'compare_methods.json'}")


if __name__ == "__main__":
    main()
