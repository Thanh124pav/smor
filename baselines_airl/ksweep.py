"""Research sweep: which K makes the reweighting hypergradient point the right way?

On HalfCheetah varying-optimality demos (source 0=expert best, 2=noisy worst), the correct
behaviour is to concentrate beta on the expert source. We sweep K (Neumann/curvature depth) and
record, per K: the final per-source beta, the eval return, and the beta DRIFT over updates
(expert-mass at start / mid / end) to see whether a one-step (K=1) gradient sends beta the wrong
way and whether larger K fixes it.
"""
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
    p.add_argument("--demos", type=str, default="results/airl_mujoco")
    p.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--bc-steps", type=int, default=400)
    p.add_argument("--reweight-interval", type=int, default=5)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--beta-lr", type=float, default=0.3)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--neumann-lr", type=float, default=0.1)
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--neumann-auto", action="store_true",
                   help="auto-scale eta_h = safety/lambda_max via power iteration")
    p.add_argument("--neumann-safety", type=float, default=1.0)
    p.add_argument("--out", type=str, default="results/airl_mujoco")
    args = p.parse_args()

    from baselines_airl.compare_airl_smor import load_demos, make_val
    from baselines_airl.mujoco_env import MujocoVecEnv
    from smor.reweighting.grouping import make_groups
    from smor.reweighting.config import OnlineReweighterConfig
    from smor.reweighting.online_reweighter import OnlineReweighter
    from smor.reweighting.outer_objective import ValidationLoss
    from smor.learners.bc import BCLearner

    train, names, env_id = load_demos(args.demos)
    val = make_val(env_id, n=10)
    ga = make_groups(train.fidelity_labels(), group_size=999, seed=0, whole_fidelity=True)
    fid = ga.group_fidelity
    exp_groups = [g for g in range(ga.num_groups) if fid[g] == 0]

    def source_mass(beta_vec):
        return {names[f]: float(sum(beta_vec[j] for j in range(len(beta_vec)) if fid[j] == f))
                for f in sorted(set(fid.tolist()))}

    rows = []
    print(f"demos {train.num_trajectories} traj | sources {names} | expert=source0\n")
    print(f"{'K':>3} {'beta(expert/med/noisy)':>28} {'return':>8} "
          f"{'exp-mass start/mid/end':>26}")
    for K in args.Ks:
        env = MujocoVecEnv(env_id, horizon=train.horizon, device="cpu", seed=7)
        learner = BCLearner(train, ga, hidden=(256, 256), lr=1e-3, batch_size=256,
                            device="cpu", val_data=val, env=env, seed=0, data_device="cpu")
        cfg = OnlineReweighterConfig(
            n=999, K=K, reweight_interval=args.reweight_interval,
            n_beta_updates=args.bc_steps // args.reweight_interval, beta_lr=args.beta_lr,
            neumann_lr=args.neumann_lr, damping=args.damping, batch_size=256, device="cpu",
            neumann_auto=args.neumann_auto, neumann_safety=args.neumann_safety,
            beta_standardize=not args.no_standardize)
        ev = OnlineReweighter(cfg).fit(learner, ga, outer_objective=ValidationLoss(),
                                       eval_every=10_000, eval_episodes=args.eval_eps)
        env.close()
        bh = np.asarray(ev.beta_history)  # (T+1, M)
        exp_idx = [j for j in range(ga.num_groups) if fid[j] == 0]
        exp_mass = bh[:, exp_idx].sum(axis=1)
        m0, mm, me = exp_mass[0], exp_mass[len(exp_mass) // 2], exp_mass[-1]
        ret = ev.eval_history["return_mean"][-1]
        sm = source_mass(ev.final_beta)
        rows.append({"K": K, "beta": sm, "return": ret,
                     "exp_mass_start": float(m0), "exp_mass_mid": float(mm), "exp_mass_end": float(me)})
        print(f"{K:>3} {str({k: round(v,2) for k,v in sm.items()}):>28} {ret:>8.0f} "
              f"{m0:>7.2f}/{mm:.2f}/{me:.2f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "ksweep.json").write_text(json.dumps(
        {"env": env_id, "sources": names, "beta_lr": args.beta_lr,
         "standardize": not args.no_standardize, "rows": rows}, indent=2))
    print(f"\nsaved {args.out}/ksweep.json")
    best = max(rows, key=lambda r: r["return"])
    print(f"best-return K={best['K']} (return {best['return']:.0f}, "
          f"expert-mass end {best['exp_mass_end']:.2f})")


if __name__ == "__main__":
    main()
