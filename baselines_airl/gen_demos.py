"""Generate varying-optimality demonstrations on seals/HalfCheetah-v1 (CAIL setting).

Runs in the isolated `smor-airl` conda env (gymnasium 0.29 + imitation). Three fidelity
sources are produced by corrupting a pretrained PPO expert's actions:
    expert  : clean expert
    medium  : expert + moderate Gaussian action noise
    noisy   : expert + heavy noise
Saves (a) imitation Trajectory lists per source for AIRL, and (b) a flat obs/act/fidelity npz
for the SMOR BC learner. Both baselines thus consume the SAME demonstrations.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import seals  # noqa: F401  registers seals/ envs
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

ENV = "seals/HalfCheetah-v1"
SOURCES = [
    {"name": "expert", "noise": 0.0},
    {"name": "medium", "noise": 0.5},
    {"name": "noisy",  "noise": 1.2},
]


def corrupt_rollouts(expert, venv, n_eps, noise, seed):
    """Collect n_eps trajectories where the executed action = expert + noise (recorded as target)."""
    rng = np.random.default_rng(seed)
    trajs = []
    from imitation.data.types import TrajectoryWithRew
    for ep in range(n_eps):
        obs = venv.envs[0].reset(seed=seed + ep)[0]
        O, A, R = [], [], []
        done = False
        while not done:
            a_exp, _ = expert.predict(obs, deterministic=True)
            a = np.clip(a_exp + noise * rng.standard_normal(a_exp.shape).astype(np.float32),
                        venv.action_space.low, venv.action_space.high)
            O.append(np.asarray(obs, dtype=np.float32)); A.append(np.asarray(a, dtype=np.float32))
            obs, r, term, trunc, info = venv.envs[0].step(a)
            R.append(float(r)); done = term or trunc
        O.append(np.asarray(obs, dtype=np.float32))
        trajs.append(TrajectoryWithRew(obs=np.array(O), acts=np.array(A),
                                       rews=np.array(R, dtype=np.float32), infos=None, terminal=True))
    return trajs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-source", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/airl_mujoco")
    args = p.parse_args()

    venv = make_vec_env(ENV, rng=np.random.default_rng(args.seed), n_envs=1,
                        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)])
    expert = load_policy("ppo-huggingface", venv=venv, env_name=ENV)

    all_trajs, flat_obs, flat_act, flat_fid, names = {}, [], [], [], []
    for i, s in enumerate(SOURCES):
        trajs = corrupt_rollouts(expert, venv, args.n_per_source, s["noise"], args.seed + 100 * (i + 1))
        rets = np.array([t.rews.sum() for t in trajs])
        print(f"source {s['name']:>7} (noise {s['noise']}): return {rets.mean():.0f} +- {rets.std():.0f}")
        all_trajs[s["name"]] = trajs
        names.append(s["name"])
        for t in trajs:
            flat_obs.append(t.obs[:-1]); flat_act.append(t.acts)
            flat_fid.append(np.full(len(t.acts), i, dtype=np.int64))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajs.pkl", "wb") as f:
        pickle.dump({"sources": names, "trajs": all_trajs, "env": ENV}, f)
    np.savez(out / "flat.npz", obs=np.concatenate(flat_obs), act=np.concatenate(flat_act),
             fidelity=np.concatenate(flat_fid), names=np.array(names))
    print(f"saved {out}/trajs.pkl + flat.npz  (obs_dim {flat_obs[0].shape[1]}, act_dim {flat_act[0].shape[1]})")


if __name__ == "__main__":
    main()
