"""Fit scaling laws + extrapolate + predict optimal mixture (spec §37-38, milestones 4-7).

Loads results/scaling/scaling_runs.csv, discovers the trend (GAM interaction strength), fits the
candidate parametric laws on the FIT budgets, scores held-out scale extrapolation, selects the
best law, predicts p*_B, and compares to the grid oracle (target-budget regret) with bootstrap CIs.
Produces a ScalingEvidence summary (spec §48).

    python -m experiments.scaling.fit_laws --config configs/scaling/two_source_mvp.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smor.scaling.bootstrap import bootstrap_law
from smor.scaling.config import ScalingConfig
from smor.scaling.evidence import ScalingEvidence
from smor.scaling.laws import (ExponentialScalingLaw, LogScalingLaw, PowerScalingLaw,
                               ShiftedPowerScalingLaw)
from smor.scaling.model_selection import evaluate_model, select_model
from smor.scaling.oracle import oracle_mixture, target_budget_regret
from smor.scaling.results_store import load_results
from smor.scaling.trend.gam import GAMScalingModel

LAWS = {"power": PowerScalingLaw, "shifted_power": ShiftedPowerScalingLaw,
        "exponential": ExponentialScalingLaw, "log": LogScalingLaw}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/scaling/two_source_mvp.yaml")
    p.add_argument("--results", default="results/scaling/scaling_runs.csv")
    p.add_argument("--metric", default="val_loss")
    p.add_argument("--models", nargs="+", default=["power", "shifted_power", "exponential", "log"])
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--out", default="results/scaling/scaling_evidence.json")
    args = p.parse_args()

    cfg = ScalingConfig.from_yaml(args.config) if Path(args.config).exists() else ScalingConfig()
    df = load_results(args.results)
    for law in LAWS.values():
        law.target_col = args.metric  # class attr used by BudgetMixtureLaw._extract
    fit_df = df[df["budget"].isin(cfg.fit_budgets)].copy()
    held_df = df[df["budget"].isin(cfg.heldout_budgets)].copy()
    all_budgets = sorted(df["budget"].unique())
    print(f"loaded {len(df)} rows | fit budgets {cfg.fit_budgets} | held-out {cfg.heldout_budgets}")

    # --- Stage 2: flexible trend (scale x mixture interaction) ---
    gam = GAMScalingModel(target_col=args.metric).fit(fit_df)
    interaction = gam.interaction_strength()
    print(f"GAM scale×mixture interaction strength = {interaction:.3f} "
          f"({'shift likely' if interaction > 0.1 else 'near scale-invariant'})")

    # --- Stage 3: candidate laws + held-out extrapolation ---
    reports, fitted = [], {}
    for name in args.models:
        law = LAWS[name]()
        law.target_col = args.metric
        law.fit(fit_df)
        fitted[name] = law
        rep = evaluate_model(law, fit_df, held_df, target_col=args.metric)
        reports.append(rep)
        hd = rep.get("heldout", {})
        print(f"  {name:14s} train_rmse={rep['train_rmse']:.4f} aic={rep['aic']:.1f} "
              f"heldout_rmse={hd.get('rmse', float('nan')):.4f}")
    best = select_model(reports, criterion="heldout_rmse")
    best_name = best["model"]
    best_law = fitted[best_name]
    print(f"selected law: {best_name} (lowest held-out extrapolation RMSE)")

    # --- Stage 4: predicted optimal mixture vs oracle (regret) ---
    opt_by_budget, regret = {}, {}
    print(f"\n{'budget':>7} {'p*_hat':>8} {'p*_oracle':>10} {'regret':>10}")
    for B in all_budgets:
        p_hat = best_law.optimal_mixture(int(B))
        orc = oracle_mixture(df, int(B), metric=args.metric)
        reg = target_budget_regret(df, int(B), float(p_hat[0]), metric=args.metric)
        opt_by_budget[int(B)] = p_hat
        regret[int(B)] = reg["regret"]
        tag = "  (held-out)" if B in cfg.heldout_budgets else ""
        print(f"{int(B):>7} {p_hat[0]:>8.3f} {orc['p_star']:>10.3f} {reg['regret']:>10.4f}{tag}")

    # --- bootstrap CI on p* at held-out budgets ---
    boot = bootstrap_law(fit_df, LAWS[best_name], cfg.heldout_budgets,
                         n_resamples=args.bootstrap, seed=0)
    for B in cfg.heldout_budgets:
        ci = boot["p_star"].get(int(B))
        if ci:
            print(f"bootstrap p*_{B} = {ci['mean']:.3f} [{ci['lo95']:.3f}, {ci['hi95']:.3f}]")

    ev = ScalingEvidence(
        fitted_law=best_law, law_name=best_name,
        law_parameters={k: float(v) for k, v in zip(best_law.param_names, best_law.params)},
        extrapolation_metrics=best.get("heldout", {}),
        optimal_mixture_by_budget=opt_by_budget, bootstrap_intervals=boot,
        oracle_regret=regret, trend_interaction_strength=interaction)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev.summary(), indent=2, default=lambda o: o.tolist()))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
