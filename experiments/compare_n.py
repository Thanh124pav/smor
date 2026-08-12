"""Granularity experiment: role of n (PLAN.md §9, RQ-R3).

Runs the reweighter at several granularities n in {1, 8, 32, whole-fidelity} and reports the
resolution / stability / compute trade-off: final return, beta stability, hypergradient
variance, wall-clock, #HVPs, and #beta dimensions.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.runner import build_pointmass_run


def _run_one(cfg, dcfg, whole_fidelity, eval_every, eval_episodes):
    setup = build_pointmass_run(cfg, dcfg, whole_fidelity=whole_fidelity)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = OnlineReweighter(cfg).fit(
            setup.learner, setup.group_assignment,
            eval_every=eval_every, eval_episodes=eval_episodes,
        )
    wall = time.perf_counter() - t0
    fid = setup.group_assignment.group_fidelity
    beta_hist = np.asarray(ev.beta_history)
    hg_hist = np.asarray(ev.hypergradient_history)
    return {
        "M": int(setup.group_assignment.num_groups),
        "return_final": float(ev.eval_history["return_mean"][-1]),
        "success_final": float(ev.eval_history["success_rate"][-1]),
        "expert_beta_mass": float(beta_hist[-1][fid == 0].sum()),
        "beta_step_l2_mean": float(np.linalg.norm(np.diff(beta_hist, axis=0), axis=1).mean()),
        "hypergrad_var": float(hg_hist.var()) if hg_hist.size else 0.0,
        "total_hvps": int(np.sum(ev.compute_history.get("hvp_count", [0]))),
        "wall_time_s": wall,
    }


def main() -> None:
    parser = common_parser("Compare reweighting granularity n.")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--include-whole-fidelity", action="store_true", default=True)
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    eval_every = int((raw.get("run", {})).get("eval_every", 10))
    eval_episodes = int((raw.get("run", {})).get("eval_episodes", 128))

    rows = []
    for n in args.ns:
        cfg, dcfg, _ = build_configs(raw, {**overrides_from_args(args), "n": n})
        r = _run_one(cfg, dcfg, False, eval_every, eval_episodes)
        r["granularity"] = str(n)
        rows.append(r)
    if args.include_whole_fidelity:
        cfg, dcfg, _ = build_configs(raw, overrides_from_args(args))
        r = _run_one(cfg, dcfg, True, eval_every, eval_episodes)
        r["granularity"] = "whole_fidelity"
        rows.append(r)

    hdr = ["granularity", "M", "return_final", "success_final", "expert_beta_mass",
           "beta_step_l2_mean", "hypergrad_var", "total_hvps", "wall_time_s"]
    print(" ".join(f"{h:>16}" for h in hdr))
    for r in rows:
        print(" ".join(f"{str(round(r[h],4)) if isinstance(r[h],float) else str(r[h]):>16}"
                       for h in hdr))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "compare_n.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {outdir/'compare_n.json'}")


if __name__ == "__main__":
    main()
