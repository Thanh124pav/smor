"""Pilot-dataset size sweep (the SMOR "uniform pilot -> OnlineReweighter" step).

Varies the size of a small pilot dataset (equal expert + noisy demonstrations) and asks:
does online reweighting still recover the signal from a *small* pilot? For each size we
compare uniform vs SMOR on the same data/seed, with a fixed held-out validation set so the
outer objective is comparable across sizes.

Reports, per pilot size: #trajectories, #groups M, uniform vs SMOR validation loss + return,
and the expert beta-mass SMOR learns.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from experiments.compare_methods import run_smor, run_uniform
from smor.runner import build_pointmass_run


def main() -> None:
    parser = common_parser("Sweep small pilot-dataset size: uniform vs SMOR.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                        help="per-fidelity demo count (total = 2x this: expert + noisy)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    rows = []
    print(f"{'total':>6} {'M':>4} | {'uni_val':>9} {'smor_val':>9} {'ratio':>7} | "
          f"{'uni_ret':>9} {'smor_ret':>9} | {'smor_emass':>10}")
    for size in args.sizes:
        u_val, s_val, u_ret, s_ret, s_em = [], [], [], [], []
        M = None
        for seed in args.seeds:
            ov = {**overrides_from_args(args), "seed": seed,
                  "n_expert": size, "n_noisy": size}
            cfg, dcfg, _ = build_configs(raw, ov)
            if M is None:
                M = build_pointmass_run(cfg, dcfg, seed=seed).group_assignment.num_groups
            um, _ = run_uniform(cfg, dcfg, seed)
            sm, sem = run_smor(cfg, dcfg, seed)
            u_val.append(float(um["val_loss"])); s_val.append(float(sm["val_loss"]))
            u_ret.append(float(um["return_mean"])); s_ret.append(float(sm["return_mean"]))
            s_em.append(sem)
        uv, sv = mean(u_val), mean(s_val)
        ratio = uv / sv if sv > 0 else float("inf")
        row = {"total_traj": 2 * size, "per_fidelity": size, "num_groups": M,
               "uniform_val": uv, "smor_val": sv, "val_ratio": ratio,
               "uniform_return": mean(u_ret), "smor_return": mean(s_ret),
               "uniform_return_std": pstdev(u_ret) if len(u_ret) > 1 else 0.0,
               "smor_return_std": pstdev(s_ret) if len(s_ret) > 1 else 0.0,
               "smor_expert_mass": mean(s_em)}
        rows.append(row)
        print(f"{2*size:>6} {M:>4} | {uv:>9.4f} {sv:>9.4f} {ratio:>6.1f}x | "
              f"{mean(u_ret):>9.2f} {mean(s_ret):>9.2f} | {mean(s_em):>10.3f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pilot_size_sweep.json").write_text(json.dumps(
        {"seeds": args.seeds, "config": cfg.to_dict(), "rows": rows}, indent=2))
    print(f"\nsaved {outdir/'pilot_size_sweep.json'}")


if __name__ == "__main__":
    main()
