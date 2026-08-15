"""Robustness sweep: how does SMOR's advantage over uniform grow with demo corruption?

For each noise level we corrupt the *noisy*-fidelity demonstrations more or less, then compare
uniform vs SMOR (learned reweighting) on the same data/seed. Reports the validation-loss and
return gap, averaged over seeds. Expectation: as the noisy demos get worse, uniform degrades
while SMOR downweights them and holds up.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from experiments.compare_methods import run_smor, run_uniform


def main() -> None:
    parser = common_parser("Sweep demo-noise level: uniform vs SMOR.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9],
                        help="noise std levels; random_prob is set to level/2")
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    results = []
    print(f"{'noise':>6} {'rand_p':>7} | "
          f"{'uniform_val':>12} {'smor_val':>10} {'val_gain':>9} | "
          f"{'uniform_ret':>12} {'smor_ret':>10} {'smor_emass':>11}")
    for lvl in args.levels:
        rand_p = round(lvl / 2, 3)
        u_val, s_val, u_ret, s_ret, s_em = [], [], [], [], []
        for seed in args.seeds:
            ov = {**overrides_from_args(args), "seed": seed, "noise": lvl, "random_prob": rand_p}
            cfg, dcfg, _ = build_configs(raw, ov)
            um, _ = run_uniform(cfg, dcfg, seed)
            sm, sem = run_smor(cfg, dcfg, seed)
            u_val.append(float(um["val_loss"])); s_val.append(float(sm["val_loss"]))
            u_ret.append(float(um["return_mean"])); s_ret.append(float(sm["return_mean"]))
            s_em.append(sem)
        uv, sv = mean(u_val), mean(s_val)
        gain = uv / sv if sv > 0 else float("inf")
        row = {"noise": lvl, "random_prob": rand_p,
               "uniform_val": uv, "smor_val": sv, "val_ratio": gain,
               "uniform_return": mean(u_ret), "smor_return": mean(s_ret),
               "smor_expert_mass": mean(s_em)}
        results.append(row)
        print(f"{lvl:>6.2f} {rand_p:>7.2f} | {uv:>12.4f} {sv:>10.4f} {gain:>8.1f}x | "
              f"{mean(u_ret):>12.2f} {mean(s_ret):>10.2f} {mean(s_em):>11.3f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "noise_sweep.json").write_text(json.dumps(
        {"seeds": args.seeds, "config": cfg.to_dict(), "rows": results}, indent=2))
    print(f"\nsaved {outdir/'noise_sweep.json'}")


if __name__ == "__main__":
    main()
