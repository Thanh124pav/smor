"""Compare AIRL vs SMOR-BC (and baselines) on HalfCheetah varying-optimality demos.

Runs in the smor-airl env. Same demonstrations feed every method; evaluation is episodic return
via real env rollouts.
    uniform-BC      : BC on all demos, equal weight
    only:expert     : BC on expert-fidelity demos only
    SMOR            : BC + online reweighting (K>1, validation outer objective)
    AIRL            : imitation AIRL (adversarial reward + PPO) on the mixed demos
"""
import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


def load_demos(path):
    from smor.envs.demos import DemoDataset
    with open(Path(path) / "trajs.pkl", "rb") as f:
        d = pickle.load(f)
    names = d["sources"]
    obs_l, act_l, fid_l = [], [], []
    for i, name in enumerate(names):
        for t in d["trajs"][name]:
            obs_l.append(torch.tensor(t.obs[:-1], dtype=torch.float32))
            act_l.append(torch.tensor(t.acts, dtype=torch.float32))
            fid_l.append(torch.full((len(t.acts),), i, dtype=torch.int64))
    # pad/stack as fixed-length trajectories (seals episodes are fixed length)
    T = obs_l[0].shape[0]
    obs = torch.stack([o for o in obs_l if o.shape[0] == T])
    act = torch.stack([a for a in act_l if a.shape[0] == T])
    fid = torch.tensor([int(f[0]) for f, o in zip(fid_l, obs_l) if o.shape[0] == T])
    return DemoDataset(obs=obs, act=act, fidelity=fid), names, d["env"]


def make_val(env_id, n=10, seed=999):
    """Clean held-out expert demos = validation target for SMOR's outer objective."""
    import seals  # noqa
    from imitation.util.util import make_vec_env
    from imitation.data.wrappers import RolloutInfoWrapper
    from imitation.policies.serialize import load_policy
    from imitation.data import rollout
    from smor.envs.demos import DemoDataset
    venv = make_vec_env(env_id, rng=np.random.default_rng(seed), n_envs=1,
                        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)])
    exp = load_policy("ppo-huggingface", venv=venv, env_name=env_id)
    r = rollout.rollout(exp, venv, rollout.make_sample_until(min_episodes=n),
                        rng=np.random.default_rng(seed))
    T = len(r[0].acts)
    obs = torch.stack([torch.tensor(t.obs[:-1], dtype=torch.float32) for t in r if len(t.acts) == T])
    act = torch.stack([torch.tensor(t.acts, dtype=torch.float32) for t in r if len(t.acts) == T])
    return DemoDataset(obs=obs, act=act, fidelity=torch.zeros(obs.shape[0], dtype=torch.int64))


def run_bc(train, val, env, ga, weights_or_smor, cfg, steps, eval_eps, K=1, smor=False):
    from smor.learners.bc import BCLearner
    from smor.reweighting.online_reweighter import OnlineReweighter
    from smor.reweighting.outer_objective import ValidationLoss
    learner = BCLearner(train, ga, hidden=(256, 256), lr=1e-3, batch_size=256,
                        device="cpu", val_data=val, env=env, seed=0, data_device="cpu")
    if smor:
        ev = OnlineReweighter(cfg).fit(learner, ga, outer_objective=ValidationLoss(),
                                       eval_every=10_000, eval_episodes=eval_eps)
        return ev.eval_history["return_mean"][-1], ev.final_beta
    gids = list(range(ga.num_groups))
    for _ in range(steps):
        learner.train_step(weights_or_smor, learner.sample_batches(gids))
    return learner.evaluate(n_episodes=eval_eps)["return_mean"], None


def run_airl(path, env_id, total_steps, eval_eps, seed=0):
    import seals  # noqa
    from imitation.algorithms.adversarial.airl import AIRL
    from imitation.rewards.reward_nets import BasicShapedRewardNet
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env
    from imitation.data.wrappers import RolloutInfoWrapper
    from imitation.data import rollout
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3.common.evaluation import evaluate_policy

    with open(Path(path) / "trajs.pkl", "rb") as f:
        d = pickle.load(f)
    demos = [t for name in d["sources"] for t in d["trajs"][name]]  # mixed varying-optimality

    venv = make_vec_env(env_id, rng=np.random.default_rng(seed), n_envs=8,
                        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)])
    learner = PPO(MlpPolicy, venv, n_steps=1024, batch_size=256, n_epochs=5,
                  ent_coef=0.0, learning_rate=3e-4, gamma=0.98, seed=seed, device="cpu")
    reward_net = BasicShapedRewardNet(venv.observation_space, venv.action_space,
                                      normalize_input_layer=RunningNorm)
    airl = AIRL(demonstrations=demos, demo_batch_size=1024, gen_replay_buffer_capacity=2048,
                n_disc_updates_per_round=4, venv=venv, gen_algo=learner, reward_net=reward_net)
    airl.train(total_steps)
    rew, _ = evaluate_policy(learner.policy, venv, n_eval_episodes=eval_eps)
    return float(rew)


def main():
    import sys
    sys.path.insert(0, ".")
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default="results/airl_mujoco")
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--bc-steps", type=int, default=400)
    p.add_argument("--airl-steps", type=int, default=300_000)
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--out", type=str, default="results/airl_mujoco")
    args = p.parse_args()

    from smor.reweighting.grouping import make_groups
    from smor.reweighting.config import OnlineReweighterConfig
    from baselines_airl.mujoco_env import MujocoVecEnv

    train, names, env_id = load_demos(args.demos)
    print(f"demos: {train.num_trajectories} traj, obs {train.obs_dim}, act {train.act_dim}, sources {names}")
    val = make_val(env_id, n=10)
    env = MujocoVecEnv(env_id, horizon=train.horizon, device="cpu", seed=7)
    ga = make_groups(train.fidelity_labels(), group_size=999, seed=0, whole_fidelity=True)
    cfg = OnlineReweighterConfig(n=999, K=args.K, reweight_interval=5, n_beta_updates=args.bc_steps // 5,
                                 beta_lr=0.3, neumann_lr=0.1, damping=1.0, batch_size=256, device="cpu")
    results = {}

    t = time.time()
    gids = list(range(ga.num_groups))
    results["uniform_bc"], _ = run_bc(train, val, env, ga, {g: 1.0 / len(gids) for g in gids},
                                      cfg, args.bc_steps, args.eval_eps)
    print(f"uniform-BC return {results['uniform_bc']:.0f}  ({time.time()-t:.0f}s)")

    fid = ga.group_fidelity
    exp_g = [g for g in gids if fid[g] == 0][0]
    w = {g: (1.0 if g == exp_g else 0.0) for g in gids}
    results["only_expert"], _ = run_bc(train, val, env, ga, w, cfg, args.bc_steps, args.eval_eps)
    print(f"only:expert-BC return {results['only_expert']:.0f}")

    results["smor_bc"], beta = run_bc(train, val, env, ga, None, cfg, args.bc_steps, args.eval_eps,
                                      K=args.K, smor=True)
    print(f"SMOR-BC return {results['smor_bc']:.0f}  beta={np.round(beta,3)}")

    t = time.time()
    results["airl"] = run_airl(args.demos, env_id, args.airl_steps, args.eval_eps)
    print(f"AIRL return {results['airl']:.0f}  ({time.time()-t:.0f}s, {args.airl_steps} steps)")

    print("\n=== HalfCheetah varying-optimality: eval return ===")
    for k, v in results.items():
        print(f"  {k:>16}: {v:.0f}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    import json
    (Path(args.out) / "compare_airl_smor.json").write_text(json.dumps(
        {"env": env_id, "sources": names, "K": args.K, "airl_steps": args.airl_steps,
         "smor_beta": [float(x) for x in (beta if beta is not None else [])],
         "results": results}, indent=2))
    print(f"saved {args.out}/compare_airl_smor.json")


if __name__ == "__main__":
    main()
