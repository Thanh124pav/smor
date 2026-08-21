"""SMOR on RoboMimic **device-calibration multi-source** demos — the non-trivial-interior study.

Unlike the quality-tier mix (``robomimic_reweight.py``, where the optimum is trivially "use the
cleanest source"), here every source is the SAME real RoboMimic task collected through a different
mis-calibrated teleop *device* (a distinct rotation bias + gain error + jitter). No single device
is correct and naive uniform does not cancel the (asymmetric) biases, so the loss-minimizing group
weighting is a genuine **interior mixture** — the reweighting question is non-trivial and rewards
curvature-aware K>1. See :mod:`smor.data.robomimic.multisource` for the mechanism + how the
default profiles were chosen (closed-form Gram analysis of the action residuals).

Compares, on the clean (un-calibrated) deployment target: only:<device>, uniform, static_quality
(least-biased single device), CAIL-style (K=1 + confidence ranking), and SMOR (K>1). Reports the
learned device mixture and how far it beats every corner + uniform.

    python -m experiments.robomimic_multisource --task lift --dtype ph --K 4 --steps 200 --seeds 0 1 2
    python -m experiments.robomimic_multisource --rollout   # + robosuite success rate
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from statistics import mean, pstdev

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.baselines.cail import cail_style_config
from smor.data.robomimic.loader import group_quality_from_labels
from smor.data.robomimic.multisource import (
    DEFAULT_DEVICE_PROFILES,
    DEFAULT_POISON,
    make_robomimic_combined,
    make_robomimic_multisource,
)
from smor.learners.bc import BCLearner
from smor.reweighting.grouping import make_groups
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.reweighting.outer_objective import CAILRankingLoss, ValidationLoss
from smor.utils.seeding import resolve_device, seed_everything


def _total_steps(cfg):
    return cfg.warmup_steps + cfg.n_beta_updates * cfg.reweight_interval


def _source_mass(group_fidelity, beta_vec):
    return {int(f): float(sum(beta_vec[j] for j in range(len(beta_vec)) if group_fidelity[j] == f))
            for f in sorted(set(int(x) for x in group_fidelity))}


def main() -> None:
    p = common_parser("SMOR on RoboMimic device-calibration multi-source (interior-optimum study).")
    p.add_argument("--task", type=str, default="lift")
    p.add_argument("--dtype", type=str, default="ph", choices=["ph", "mh", "mg"],
                   help="which real variant supplies the clean base demos")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-per-source", type=int, default=None, help="cap demos per device")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-horizon", type=int, default=400)
    p.add_argument("--eval-episodes", type=int, default=25)
    p.add_argument("--outer", choices=["val", "ranking"], default="val")
    p.add_argument("--poison", action="store_true",
                   help="add real bad demos (MG-fail) as poison sources: uniform keeps them and is "
                        "suboptimal, so SMOR beats uniform by driving poison->0 while keeping an "
                        "interior blend of the good calibrated sources")
    p.add_argument("--poison-n", type=int, default=180, help="demos per poison source")
    p.add_argument("--rollout", action="store_true",
                   help="also evaluate success rate in the robosuite env (needs robomimic+robosuite)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    args = p.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    profiles = DEFAULT_DEVICE_PROFILES
    n_src = len(profiles) + (len(DEFAULT_POISON) if args.poison else 0)
    names0 = [pf["name"] for pf in profiles] + (
        [p.get("name", p["dtype"]) for p in DEFAULT_POISON] if args.poison else [])
    methods = ([f"only:{n}" for n in names0] + ["uniform", "static_quality", "cail", "smor"])
    agg = {m: {"succ": [], "val": []} for m in methods}
    smor_mass = []
    quality = None

    for seed in args.seeds:
        cfg, _, _ = build_configs(raw, {**overrides_from_args(args), "seed": seed})
        device = str(resolve_device(cfg.device))
        seed_everything(seed)

        if args.poison:
            train, val, names, quality = make_robomimic_combined(
                task=args.task, base_dtype=args.dtype, profiles=profiles,
                n_per_source=args.n_per_source, poison_n=args.poison_n,
                val_frac=args.val_frac, seed=seed, root=args.data_root)
        else:
            train, val, names, quality = make_robomimic_multisource(
                task=args.task, dtype=args.dtype, profiles=profiles,
                n_per_source=args.n_per_source, val_frac=args.val_frac, seed=seed, root=args.data_root)
        best_label = int(max(range(n_src), key=lambda i: quality[i]))
        ga = make_groups(train.fidelity_labels(), group_size=cfg.n, seed=seed, whole_fidelity=True)
        gq = group_quality_from_labels(ga.group_fidelity, quality)

        env = None
        if args.rollout:
            from smor.envs.robosuite_env import RobosuiteVecEnv
            env = RobosuiteVecEnv(task=args.task, dtype=args.dtype, horizon=args.eval_horizon,
                                  device=device, seed=seed + 7, root=args.data_root)
        print(f"[seed {seed}] task={args.task} devices={names} traj={train.num_trajectories} "
              f"groups={ga.num_groups} device={device} rollout={bool(env)}")

        def new_learner():
            return BCLearner(train, ga, hidden=tuple(args.hidden), lr=1e-3,
                             batch_size=cfg.batch_size, device=device, val_data=val, env=env,
                             seed=seed, data_device=args.data_device)

        gids = list(range(ga.num_groups))
        fid = ga.group_fidelity

        def record(method, m):
            agg[method]["succ"].append(float(m.get("success_rate", float("nan"))))
            agg[method]["val"].append(float(m["val_loss"]))

        def train_fixed(weights):
            lr = new_learner()
            for _ in range(_total_steps(cfg)):
                lr.train_step(weights, lr.sample_batches(gids))
            return lr.evaluate(n_episodes=args.eval_episodes)

        for i in range(n_src):
            w = {g: (1.0 if fid[g] == i else 0.0) for g in gids}
            tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
            record(f"only:{names[i]}", train_fixed(w))

        record("uniform", train_fixed({g: 1.0 / len(gids) for g in gids}))

        w = {g: (1.0 if fid[g] == best_label else 0.0) for g in gids}
        tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
        record("static_quality", train_fixed(w))

        lr_c = new_learner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev_c = OnlineReweighter(cail_style_config(cfg)).fit(
                lr_c, ga, outer_objective=CAILRankingLoss(gq),
                eval_every=10_000, eval_episodes=args.eval_episodes, env_name=f"robomimic-ms:{args.task}")
        record("cail", {"success_rate": ev_c.eval_history.get("success_rate", [float("nan")])[-1],
                        "val_loss": ev_c.eval_history["val_loss"][-1]})

        outer = CAILRankingLoss(gq) if args.outer == "ranking" else ValidationLoss()
        lr_s = new_learner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = OnlineReweighter(cfg).fit(lr_s, ga, outer_objective=outer,
                                           eval_every=10_000, eval_episodes=args.eval_episodes,
                                           env_name=f"robomimic-ms:{args.task}")
        record("smor", {"success_rate": ev.eval_history.get("success_rate", [float("nan")])[-1],
                        "val_loss": ev.eval_history["val_loss"][-1]})
        smor_mass.append(_source_mass(ga.group_fidelity, ev.final_beta))
        if env is not None:
            env.close()
        print(f"[seed {seed}] smor mixture={ {names[k]: round(v,3) for k,v in smor_mass[-1].items()} }")

    print(f"\ntask={args.task} devices={names0} least_biased={names0[int(max(range(n_src), key=lambda i: quality[i]))]}  "
          f"seeds={args.seeds}  K={cfg.K}  outer={args.outer}\n")
    print(f"{'method':>22} {'success':>16} {'val_loss':>14}")
    rows = {}
    for m in methods:
        succ = [s for s in agg[m]["succ"] if not math.isnan(s)]
        sm = mean(succ) if succ else float("nan")
        ss = (pstdev(succ) if len(succ) > 1 else 0.0)
        rows[m] = {"success_mean": sm, "success_std": ss,
                   "val_mean": mean(agg[m]["val"]), "val_std": pstdev(agg[m]["val"]) if len(agg[m]["val"]) > 1 else 0.0}
        sstr = f"{sm:>10.3f} ± {ss:<4.3f}" if succ else f"{'n/a':>16}"
        print(f"{m:>22} {sstr} {rows[m]['val_mean']:>10.4f} ± {rows[m]['val_std']:.4f}")
    mix = {names0[k]: mean([mm.get(k, 0.0) for mm in smor_mass]) for k in range(n_src)}
    print(f"\nSMOR learned device mixture: { {k: round(v,3) for k,v in mix.items()} }")
    best_corner = min((rows[f'only:{n}']['val_mean'] for n in names0))
    print(f"best single device val={best_corner:.4f}  uniform val={rows['uniform']['val_mean']:.4f}  "
          f"SMOR val={rows['smor']['val_mean']:.4f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "robomimic_multisource.json").write_text(json.dumps(
        {"task": args.task, "dtype": args.dtype, "profiles": profiles, "seeds": args.seeds,
         "quality": quality, "outer": args.outer, "rollout": args.rollout,
         "config": cfg.to_dict(), "rows": rows, "smor_mixture": mix}, indent=2))
    print(f"\nsaved {outdir/'robomimic_multisource.json'}")


if __name__ == "__main__":
    main()
