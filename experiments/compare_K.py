"""Does K>1 predict long-horizon data utility better than K=1? (PLAN.md §8, RQ-R2)

Procedure:
  1. Train a base policy under uniform beta to a fixed checkpoint.
  2. For each K, estimate the group hypergradients h_j^{(K)} at that checkpoint.
  3. Measure the *realized* long-horizon effect Delta_j^long of upweighting each group and
     training for a much longer oracle horizon.
  4. Report sign accuracy / Spearman / Pearson / cosine of h^{(K)} vs Delta^long.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.evaluation.long_horizon_utility import (
    estimate_group_hypergradients, realized_long_horizon_deltas,
)
from smor.evaluation.metrics import cosine_similarity, pearson, sign_accuracy, spearman
from smor.reweighting.outer_objective import ValidationLoss
from smor.runner import build_pointmass_run


def main() -> None:
    parser = common_parser("Compare K vs realized long-horizon utility.")
    parser.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--oracle-steps", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--hg-batches", type=int, default=16)
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    cfg, dcfg, _ = build_configs(raw, overrides_from_args(args))
    setup = build_pointmass_run(cfg, dcfg)
    learner, ga = setup.learner, setup.group_assignment
    gids = list(range(ga.num_groups))
    outer = ValidationLoss()

    # 1. base policy under uniform beta
    w = {g: 1.0 / ga.num_groups for g in gids}
    for _ in range(args.base_steps):
        learner.train_step(w, learner.sample_batches(gids))

    # 3. realized long-horizon deltas (computed once; the ground truth)
    print(f"measuring realized long-horizon deltas (oracle_steps={args.oracle_steps}, "
          f"eps={args.epsilon}, groups={ga.num_groups}) ...")
    deltas = realized_long_horizon_deltas(
        learner, ga, epsilon=args.epsilon, oracle_steps=args.oracle_steps, outer_objective=outer,
    )

    # 2 + 4. predictions per K and correlations vs deltas
    rows = []
    for K in args.Ks:
        h = estimate_group_hypergradients(
            learner, ga, K=K, neumann_lr=cfg.neumann_lr, damping=cfg.damping,
            outer_objective=outer, n_batches=args.hg_batches,
        )
        rows.append({
            "K": K,
            "sign_acc": sign_accuracy(h, deltas),
            "spearman": spearman(h, deltas),
            "pearson": pearson(h, deltas),
            "cosine": cosine_similarity(h, deltas),
        })

    print(f"\n{'K':>3} {'sign_acc':>9} {'spearman':>9} {'pearson':>9} {'cosine':>9}")
    for r in rows:
        print(f"{r['K']:>3} {r['sign_acc']:>9.3f} {r['spearman']:>9.3f} "
              f"{r['pearson']:>9.3f} {r['cosine']:>9.3f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": cfg.to_dict(),
        "base_steps": args.base_steps, "oracle_steps": args.oracle_steps,
        "epsilon": args.epsilon, "num_groups": ga.num_groups,
        "realized_deltas": deltas.tolist(),
        "group_fidelity": ga.group_fidelity.tolist(),
        "rows": rows,
    }
    (outdir / "compare_K.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved {outdir/'compare_K.json'}")


if __name__ == "__main__":
    main()
