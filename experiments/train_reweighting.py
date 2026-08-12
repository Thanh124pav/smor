"""Main train + evaluate entry point (PLAN.md Stage B / §16).

Trains a BC policy under online-learned beta on the point-mass task, evaluates via real env
rollouts, and saves a policy checkpoint, a ``ReweightingEvidence`` object and a JSONL log.

    python -m experiments.train_reweighting --config configs/curvature_reweight.yaml --n 8 --K 2
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from experiments._common import (
    build_configs, common_parser, load_yaml, overrides_from_args,
)
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.runner import build_pointmass_run
from smor.utils.logging import JsonlLogger


def main() -> None:
    parser = common_parser("Train + evaluate online reweighting on point-mass.")
    parser.add_argument("--whole-fidelity", action="store_true",
                        help="one beta per fidelity level (n = whole fidelity)")
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    cfg, dcfg, run = build_configs(raw, overrides_from_args(args))
    eval_every = int(run.get("eval_every", 10))
    eval_episodes = args.eval_episodes or int(run.get("eval_episodes", 128))

    outdir = Path(args.outdir) / f"n{cfg.n}_K{cfg.K}_seed{cfg.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    setup = build_pointmass_run(cfg, dcfg, whole_fidelity=args.whole_fidelity)
    print(f"device={setup.device}  groups={setup.group_assignment.num_groups}  "
          f"trajectories={setup.group_assignment.num_trajectories}  n={cfg.n} K={cfg.K}")

    logger = JsonlLogger(outdir / "log.jsonl")
    reweighter = OnlineReweighter(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("always", RuntimeWarning)
        evidence = reweighter.fit(
            setup.learner, setup.group_assignment,
            logger=logger, eval_every=eval_every, eval_episodes=eval_episodes,
            checkpoint_path=str(outdir / "policy.pt"),
        )
    logger.close()

    evidence.save(outdir / "evidence.npz")

    fid = setup.group_assignment.group_fidelity
    fb = evidence.final_beta
    summary = {
        "n": cfg.n, "K": cfg.K, "seed": cfg.seed, "device": setup.device,
        "num_groups": setup.group_assignment.num_groups,
        "expert_beta_mass": float(fb[fid == 0].sum()),
        "noisy_beta_mass": float(fb[fid == 1].sum()) if (fid == 1).any() else 0.0,
        "return_initial": float(evidence.eval_history["return_mean"][0]),
        "return_final": float(evidence.eval_history["return_mean"][-1]),
        "success_final": float(evidence.eval_history["success_rate"][-1]),
        "val_loss_final": float(evidence.eval_history["val_loss"][-1]),
        "total_hvps": int(np.sum(evidence.compute_history.get("hvp_count", [0]))),
        "total_hvp_time_s": float(np.sum(evidence.compute_history.get("hvp_time", [0.0]))),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nartifacts in {outdir}/ : policy.pt  evidence.npz  log.jsonl  summary.json")


if __name__ == "__main__":
    main()
