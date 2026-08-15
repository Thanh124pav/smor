"""Different-style demonstrations (SpaceMouse vs teleop vs kinesthetic): does reweighting find a
NON-TRIVIAL mixture that beats every single source AND uniform?

Each source has a different systematic torque bias, so no single device is clean and the
bias-cancelling optimum is an interior, asymmetric beta. Metric = eval return; val target =
clean expert. Runs in the smor-airl env."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def main():
    import sys
    sys.path.insert(0, ".")
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default="results/airl_styles")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--bc-steps", type=int, default=400)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--out", type=str, default="results/airl_styles")
    args = p.parse_args()

    from baselines_airl.compare_airl_smor import load_demos, make_val, run_bc
    from baselines_airl.mujoco_env import MujocoVecEnv
    from smor.reweighting.grouping import make_groups
    from smor.reweighting.config import OnlineReweighterConfig

    train, names, env_id = load_demos(args.demos)
    val = make_val(env_id, n=10)
    print(f"demos {train.num_trajectories} traj | sources {names} | val=clean expert\n")

    methods = [f"only:{n}" for n in names] + ["uniform", "smor"]
    agg = {m: [] for m in methods}
    betas = []
    for seed in args.seeds:
        env = MujocoVecEnv(env_id, horizon=train.horizon, device="cpu", seed=7 + seed)
        ga = make_groups(train.fidelity_labels(), group_size=999, seed=seed, whole_fidelity=True)
        gids = list(range(ga.num_groups)); fid = ga.group_fidelity
        cfg = OnlineReweighterConfig(n=999, K=args.K, reweight_interval=5,
                                     n_beta_updates=args.bc_steps // 5, beta_lr=0.3,
                                     neumann_auto=True, neumann_safety=1.0, damping=1.0,
                                     batch_size=256, device="cpu", seed=seed)
        for i, nm in enumerate(names):
            w = {g: (1.0 if fid[g] == i else 0.0) for g in gids}
            tot = sum(w.values()); w = {g: v / tot for g, v in w.items()}
            r, _ = run_bc(train, val, env, ga, w, cfg, args.bc_steps, args.eval_eps)
            agg[f"only:{nm}"].append(r)
        r, _ = run_bc(train, val, env, ga, {g: 1.0 / len(gids) for g in gids}, cfg,
                      args.bc_steps, args.eval_eps)
        agg["uniform"].append(r)
        r, beta = run_bc(train, val, env, ga, None, cfg, args.bc_steps, args.eval_eps,
                         K=args.K, smor=True)
        agg["smor"].append(r)
        # source-level beta
        sm = {names[f]: float(sum(beta[j] for j in range(len(beta)) if fid[j] == f))
              for f in sorted(set(fid.tolist()))}
        betas.append(sm)
        env.close()
        print(f"[seed {seed}] smor beta={ {k: round(v,3) for k,v in sm.items()} }")

    from statistics import mean, pstdev
    print(f"\ntask={env_id}  K={args.K}  seeds={args.seeds}  (val target = clean expert)\n")
    print(f"{'method':>16} {'eval return':>18}")
    rows = {}
    for m in methods:
        mu = mean(agg[m]); sd = pstdev(agg[m]) if len(agg[m]) > 1 else 0.0
        rows[m] = {"mean": mu, "std": sd}
        print(f"{m:>16} {mu:>12.0f} ± {sd:<5.0f}")
    mix = {k: mean([b[k] for b in betas]) for k in names}
    print(f"\nSMOR mixture (mean): { {k: round(v,3) for k,v in mix.items()} }")
    best_single = max((f"only:{n}" for n in names), key=lambda m: rows[m]["mean"])
    print(f"best single = {best_single} ({rows[best_single]['mean']:.0f}); "
          f"uniform {rows['uniform']['mean']:.0f}; SMOR {rows['smor']['mean']:.0f}")
    verdict = ("SMOR beats best-single AND uniform"
               if rows["smor"]["mean"] > max(rows[best_single]["mean"], rows["uniform"]["mean"])
               else "SMOR does NOT beat both")
    print("VERDICT:", verdict)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "styles_compare.json").write_text(json.dumps(
        {"env": env_id, "sources": names, "K": args.K, "rows": rows,
         "smor_mixture": mix, "verdict": verdict}, indent=2))
    print(f"saved {args.out}/styles_compare.json")


if __name__ == "__main__":
    main()
