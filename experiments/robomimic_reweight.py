"""SMOR vs CAIL-style vs baselines on **RoboMimic** multi-fidelity demonstrations.

Uses the official RoboMimic datasets (PH / MH / MG) as the data source. The fidelity groups are
*real* human-operator quality tiers (multi-human ``better``/``okay``/``worse``) and/or variant
quality (proficient-human vs machine-generated), not synthetic corruption. On the clean
deployment target it compares: only:<source>, uniform, static_quality (train on the single
best-labelled source), CAIL-style (K=1 + confidence ranking), and SMOR (curvature-aware K>1).

Evaluation:
  * default  -> held-out validation loss on the clean target (needs only h5py; no MuJoCo).
  * --rollout -> also roll the policy in the reconstructed robosuite env for a real success rate
                 (needs `pip install robomimic robosuite`).

    python -m experiments.robomimic_reweight --mix mh-tiers --K 4 --steps 200 --seeds 0 1 2
    python -m experiments.robomimic_reweight --mix "lift:ph*,lift:mh:worse,lift:mg:mg_fail" --rollout
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

from experiments._common import build_configs, common_parser, load_yaml, overrides_from_args
from smor.baselines.cail import cail_style_config
from smor.data.robomimic import load_robomimic_mix, parse_mix
from smor.data.robomimic.loader import group_quality_from_labels
from smor.learners.bc import BCLearner
from smor.reweighting.grouping import make_groups
from smor.reweighting.online_reweighter import OnlineReweighter
from smor.reweighting.outer_objective import CAILRankingLoss, ValidationLoss
from smor.utils.seeding import resolve_device, seed_everything


def _total_steps(cfg):
    return cfg.warmup_steps + cfg.n_beta_updates * cfg.reweight_interval


def _source_mass(group_fidelity, beta_vec):
    fid = group_fidelity
    return {int(f): float(sum(beta_vec[j] for j in range(len(beta_vec)) if fid[j] == f))
            for f in sorted(set(int(x) for x in fid))}


def _make_env(args, task, seed):
    if not (args.rollout or args.outer == "grpo"):
        return None
    from smor.envs.robosuite_env import RobosuiteVecEnv
    # Env metadata comes from the target component's variant; default to ph if present else mh.
    dtype = args.rollout_dtype
    return RobosuiteVecEnv(task=task, dtype=dtype, horizon=args.eval_horizon,
                           device=str(resolve_device(args.device or "auto")), seed=seed + 7,
                           root=args.data_root, reward_shaping=(args.outer == "grpo"))


def main() -> None:
    p = common_parser("SMOR vs CAIL-style on RoboMimic multi-fidelity demonstrations.")
    p.add_argument("--mix", type=str, default="mh-tiers",
                   help="preset name or DSL 'task:dtype[:tier][:n],...' (append * = clean target)")
    p.add_argument("--task", type=str, default=None,
                   help="override the task of every mix component (e.g. can, square)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--cap", type=int, default=None, help="max training trajectories per source")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-mode", choices=["stratified", "target"], default="stratified",
                   help="how the outer-loss validation set is drawn: 'stratified' holds out from "
                        "EVERY source (balanced outer signal, avoids beta collapse); 'target' holds "
                        "out only the clean target source (legacy; collapses beta to a near-corner)")
    p.add_argument("--val-per-source", type=int, default=None,
                   help="stratified: hold out exactly this many val trajectories per source "
                        "(equal/balanced stratification); default uses val_frac of each source")
    p.add_argument("--eval-horizon", type=int, default=400)
    p.add_argument("--eval-episodes", type=int, default=25)
    p.add_argument("--outer", choices=["val", "ranking", "grpo"], default="val",
                   help="SMOR outer objective: val (open-loop MSE) | ranking (CAIL) | grpo "
                        "(closed-loop RETURN via GRPO — needs robosuite env, reward_shaping)")
    p.add_argument("--grpo-episodes", type=int, default=12)
    p.add_argument("--rollout", action="store_true",
                   help="also evaluate success rate in the robosuite env (needs robomimic+robosuite)")
    p.add_argument("--rollout-dtype", type=str, default="ph",
                   help="which variant's env metadata to reconstruct for rollout eval")
    p.add_argument("--data-root", type=str, default=None, help="robomimic dataset cache dir")
    p.add_argument("--data-device", type=str, default="cpu",
                   help="where demo tensors live (cpu keeps GPU free for big data)")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    args = p.parse_args()

    raw = load_yaml(args.config) if args.config else {}
    components = parse_mix(args.mix, task=args.task)
    if args.cap is not None:
        for c in components:
            c.n = args.cap if c.n is None else min(c.n, args.cap)

    methods = None  # filled once we know source names
    agg: dict = {}
    smor_mass = []
    names_ref = None

    for seed in args.seeds:
        cfg, _, _ = build_configs(raw, {**overrides_from_args(args), "seed": seed})
        device = str(resolve_device(cfg.device))
        seed_everything(seed)

        train, val, names, quality = load_robomimic_mix(
            components, val_frac=args.val_frac, seed=seed, root=args.data_root,
            val_mode=args.val_mode, val_per_source=args.val_per_source)
        n_src = len(names)
        best_label = int(max(range(n_src), key=lambda i: quality[i]))
        if methods is None:
            names_ref = names
            methods = ([f"only:{names[i]}" for i in range(n_src)]
                       + ["uniform", "static_quality", "cail", "smor"])
            agg = {m: {"succ": [], "val": []} for m in methods}

        ga = make_groups(train.fidelity_labels(), group_size=cfg.n, seed=seed, whole_fidelity=True)
        gq = group_quality_from_labels(ga.group_fidelity, quality)
        env = _make_env(args, components[best_label].task, seed)
        print(f"[seed {seed}] mix={[ (c.task,c.dtype,c.tier) for c in components ]} "
              f"traj={train.num_trajectories} groups={ga.num_groups} device={device} "
              f"rollout={bool(env)}")

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

        # CAIL-style: shared backbone, K=1 one-step + confidence ranking.
        lr_c = new_learner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev_c = OnlineReweighter(cail_style_config(cfg)).fit(
                lr_c, ga, outer_objective=CAILRankingLoss(gq),
                eval_every=10_000, eval_episodes=args.eval_episodes, env_name=f"robomimic:{args.mix}")
        record("cail", {"success_rate": ev_c.eval_history.get("success_rate", [float("nan")])[-1],
                        "val_loss": ev_c.eval_history["val_loss"][-1]})

        # SMOR: curvature-aware K>1 + chosen outer objective.
        if args.outer == "grpo":
            from smor.reweighting.outer_objective import ClosedLoopRolloutReturn
            outer = ClosedLoopRolloutReturn(n_episodes=args.grpo_episodes, variant="grpo")
            cfg = replace(cfg, normalize_group_grads=True)
        else:
            outer = CAILRankingLoss(gq) if args.outer == "ranking" else ValidationLoss()
        lr_s = new_learner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = OnlineReweighter(cfg).fit(lr_s, ga, outer_objective=outer,
                                           eval_every=10_000, eval_episodes=args.eval_episodes,
                                           env_name=f"robomimic:{args.mix}")
        record("smor", {"success_rate": ev.eval_history.get("success_rate", [float("nan")])[-1],
                        "val_loss": ev.eval_history["val_loss"][-1]})
        smor_mass.append(_source_mass(ga.group_fidelity, ev.final_beta))
        if env is not None:
            env.close()
        print(f"[seed {seed}] smor mix={ {names[k]: round(v,3) for k,v in smor_mass[-1].items()} }")

    names = names_ref
    n_src = len(names)
    best_label = int(max(range(n_src), key=lambda i: quality[i]))
    print(f"\nmix={args.mix}  sources={names}  best_labelled={names[best_label]}  "
          f"seeds={args.seeds}  K={cfg.K}  outer={args.outer}\n")
    hdr = f"{'method':>26} {'success':>16} {'val_loss':>14}"
    print(hdr)
    rows = {}
    import math
    for m in methods:
        succ = [s for s in agg[m]["succ"] if not math.isnan(s)]
        sm = mean(succ) if succ else float("nan")
        ss = (pstdev(succ) if len(succ) > 1 else 0.0)
        rows[m] = {"success_mean": sm, "success_std": ss, "val_mean": mean(agg[m]["val"])}
        sstr = f"{sm:>10.3f} ± {ss:<4.3f}" if succ else f"{'n/a':>16}"
        print(f"{m:>26} {sstr} {rows[m]['val_mean']:>14.4f}")
    mix = {names[k]: mean([mm.get(k, 0.0) for mm in smor_mass]) for k in range(n_src)}
    print(f"\nSMOR learned source mixture: { {k: round(v,3) for k,v in mix.items()} }")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "robomimic_reweight.json").write_text(json.dumps(
        {"mix": args.mix, "seeds": args.seeds, "sources": names, "quality": quality,
         "outer": args.outer, "rollout": args.rollout, "config": cfg.to_dict(),
         "rows": rows, "smor_mixture": mix}, indent=2))
    print(f"\nsaved {outdir/'robomimic_reweight.json'}")


if __name__ == "__main__":
    main()
