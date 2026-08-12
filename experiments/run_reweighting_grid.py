"""n x K experiment matrix (PLAN.md §2). Train + evaluate each cell; collect a metrics table.

Also checks beta-trajectory reproducibility across two seeds (PLAN.md §16.5).

    python -m experiments.run_reweighting_grid --steps 40
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.runner import build_pointmass_run


def _cell(cfg, dcfg, whole_fidelity, eval_every, eval_episodes, outdir):
    setup = build_pointmass_run(cfg, dcfg, whole_fidelity=whole_fidelity)
    tag = ("wholefid" if whole_fidelity else f"n{cfg.n}") + f"_K{cfg.K}_seed{cfg.seed}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = OnlineReweighter(cfg).fit(
            setup.learner, setup.group_assignment,
            eval_every=eval_every, eval_episodes=eval_episodes,
            checkpoint_path=str(outdir / f"{tag}_policy.pt"),
        )
    ev.save(outdir / f"{tag}_evidence.npz")
    fid = setup.group_assignment.group_fidelity
    fb = ev.final_beta
    return {
        "granularity": "whole_fidelity" if whole_fidelity else str(cfg.n),
        "K": cfg.K, "seed": cfg.seed,
        "M": int(setup.group_assignment.num_groups),
        "return_final": float(ev.eval_history["return_mean"][-1]),
        "success_final": float(ev.eval_history["success_rate"][-1]),
        "expert_beta_mass": float(fb[fid == 0].sum()),
        "total_hvps": int(np.sum(ev.compute_history.get("hvp_count", [0]))),
    }, np.asarray(ev.beta_history)


def main() -> None:
    parser = common_parser("Run the n x K reweighting grid.")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--with-whole-fidelity", action="store_true", default=True)
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    eval_every = int((raw.get("run", {})).get("eval_every", 10))
    eval_episodes = int((raw.get("run", {})).get("eval_episodes", 128))
    outdir = Path(args.outdir) / "grid"; outdir.mkdir(parents=True, exist_ok=True)

    grid = [(str(n), n, False) for n in args.ns]
    if args.with_whole_fidelity:
        grid.append(("whole_fidelity", args.ns[-1], True))

    rows = []
    for _, n, whole in grid:
        for K in args.Ks:
            cfg, dcfg, _ = build_configs(raw, {**overrides_from_args(args), "n": n, "K": K})
            row, _ = _cell(cfg, dcfg, whole, eval_every, eval_episodes, outdir)
            rows.append(row)
            print(f"  n={row['granularity']:>14} K={K} M={row['M']:>3} "
                  f"ret={row['return_final']:.2f} succ={row['success_final']:.2f} "
                  f"expert_mass={row['expert_beta_mass']:.3f} hvps={row['total_hvps']}")

    # reproducibility across seeds (§16.5): same n,K, two seeds -> correlated beta trajectories
    cfg_a, dcfg, _ = build_configs(raw, {**overrides_from_args(args), "n": args.ns[1] if len(args.ns) > 1 else args.ns[0], "K": args.Ks[0], "seed": 0})
    cfg_b, _, _ = build_configs(raw, {**overrides_from_args(args), "n": cfg_a.n, "K": cfg_a.K, "seed": 1})
    _, beta_a = _cell(cfg_a, dcfg, False, eval_every, eval_episodes, outdir)
    _, beta_b = _cell(cfg_b, dcfg, False, eval_every, eval_episodes, outdir)
    # correlate final expert-mass trajectory shape via per-step correlation of sorted betas
    corr = float(np.corrcoef(beta_a[-1], beta_b[-1])[0, 1]) if beta_a.shape[1] > 1 else 1.0

    result = {"rows": rows, "final_beta_cross_seed_corr": corr}
    (outdir / "grid.json").write_text(json.dumps(result, indent=2))
    print(f"\nfinal-beta cross-seed correlation (n={cfg_a.n},K={cfg_a.K}): {corr:.3f}")
    print(f"saved {outdir/'grid.json'}")


if __name__ == "__main__":
    main()
