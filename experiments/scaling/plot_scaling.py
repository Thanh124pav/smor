"""Raw scaling visualizations (spec §12, milestone 3): Plot A/B/C/D.

    python -m experiments.scaling.plot_scaling --results results/scaling/scaling_runs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/scaling/scaling_runs.csv")
    ap.add_argument("--outdir", default="results/scaling/plots")
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    piv = df.groupby(["budget", "p_source_0"])["val_loss"].mean().unstack()
    budgets = piv.index.to_numpy()
    ps = piv.columns.to_numpy()
    cmap = plt.get_cmap("viridis")

    # Plot A: B -> L for each mixture
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, p in enumerate(ps):
        ax.plot(budgets, piv[p].to_numpy(), "o-", color=cmap(j / max(1, len(ps) - 1)),
                label=f"p_PH={p:.1f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("budget B (unique trajectories)"); ax.set_ylabel("val loss")
    ax.set_title("Plot A: scaling by budget, per mixture"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "A_budget_scaling.png", dpi=130); plt.close(fig)

    # Plot B: p -> L for each budget
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, B in enumerate(budgets):
        ax.plot(ps, piv.loc[B].to_numpy(), "s-", color=cmap(i / max(1, len(budgets) - 1)),
                label=f"B={int(B)}")
    ax.set_yscale("log")
    ax.set_xlabel("mixture p_PH (source-0 share)"); ax.set_ylabel("val loss")
    ax.set_title("Plot B: mixture dependence, per budget"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "B_mixture_dependence.png", dpi=130); plt.close(fig)

    # Plot D: B -> grid-optimal p*
    fig, ax = plt.subplots(figsize=(6, 4))
    pstar = piv.idxmin(axis=1).to_numpy()
    ax.plot(budgets, pstar, "o-", color="crimson")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("budget B"); ax.set_ylabel("grid-optimal p*_PH")
    ax.set_title("Plot D: optimal mixture vs scale")
    fig.tight_layout(); fig.savefig(outdir / "D_optimal_mixture.png", dpi=130); plt.close(fig)

    saved = ["A_budget_scaling.png", "B_mixture_dependence.png", "D_optimal_mixture.png"]

    # Plot C: (B,p) -> success heatmap, if success measured
    if "success_rate" in df.columns and df["success_rate"].notna().any():
        sp = df.groupby(["budget", "p_source_0"])["success_rate"].mean().unstack()
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(sp.to_numpy(), aspect="auto", origin="lower", cmap="magma",
                       extent=[sp.columns.min(), sp.columns.max(), sp.index.min(), sp.index.max()])
        ax.set_xlabel("mixture p_PH"); ax.set_ylabel("budget B")
        ax.set_title("Plot C: closed-loop success (B, p)"); fig.colorbar(im, ax=ax, label="success")
        fig.tight_layout(); fig.savefig(outdir / "C_success_heatmap.png", dpi=130); plt.close(fig)
        saved.append("C_success_heatmap.png")

    print("saved:", ", ".join(str(outdir / s) for s in saved))


if __name__ == "__main__":
    main()
