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

# "quality" mode: a pure quality gradient (expert>medium>noisy) -> beta trivially -> expert.
QUALITY_SOURCES = [
    {"name": "expert", "noise": 0.0},
    {"name": "medium", "noise": 0.5},
    {"name": "noisy",  "noise": 1.2},
]

# "styles" mode: genuinely DIFFERENT devices (SpaceMouse vs teleop vs kinesthetic). Each has its
# own systematic miscalibration, NONE is clean, so the best policy needs a NON-TRIVIAL MIXTURE
# (an undershooting + an overshooting device blend toward unit gain; a per-limb-biased device is
# only useful in part). This is the setting where beta=1.0 is wrong and reweighting matters.
STYLE_SOURCES = [
    # Complementary ALTERNATING per-joint biases (moderate, not catastrophic on HalfCheetah) with
    # opposite patterns and DIFFERENT magnitude: every single device is moderately suboptimal, and
    # only an interior, asymmetric mixture cancels the bias back to clean-expert behaviour
    # (optimal beta_A/beta_B ~= 0.2/0.3), so uniform leaves residual bias and no single wins.
    {"name": "spacemouse",  "bias": [+0.3, -0.3, +0.3, -0.3, +0.3, -0.3], "noise": 0.10},
    {"name": "teleop",      "bias": [-0.2, +0.2, -0.2, +0.2, -0.2, +0.2], "noise": 0.10},
    {"name": "kinesthetic", "bias": [+0.25, +0.25, -0.25, -0.25, 0.0, 0.0], "noise": 0.10},
]


def corrupt_rollouts(expert, venv, n_eps, noise, seed, gain=1.0, bias=None):
    """Collect n_eps trajectories; executed action = gain*expert + bias + noise (recorded target)."""
    rng = np.random.default_rng(seed)
    trajs = []
    bias = np.asarray(bias, dtype=np.float32) if bias is not None else 0.0
    from imitation.data.types import TrajectoryWithRew
    for ep in range(n_eps):
        obs = venv.envs[0].reset(seed=seed + ep)[0]
        O, A, R = [], [], []
        done = False
        while not done:
            a_exp, _ = expert.predict(obs, deterministic=True)
            a = np.clip(gain * a_exp + bias
                        + noise * rng.standard_normal(a_exp.shape).astype(np.float32),
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
    p.add_argument("--mode", choices=["quality", "styles"], default="styles")
    p.add_argument("--out", type=str, default="results/airl_mujoco")
    args = p.parse_args()

    SOURCES = STYLE_SOURCES if args.mode == "styles" else QUALITY_SOURCES
    venv = make_vec_env(ENV, rng=np.random.default_rng(args.seed), n_envs=1,
                        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)])
    expert = load_policy("ppo-huggingface", venv=venv, env_name=ENV)

    all_trajs, flat_obs, flat_act, flat_fid, names = {}, [], [], [], []
    for i, s in enumerate(SOURCES):
        trajs = corrupt_rollouts(expert, venv, args.n_per_source, s.get("noise", 0.0),
                                 args.seed + 100 * (i + 1),
                                 gain=s.get("gain", 1.0), bias=s.get("bias"))
        rets = np.array([t.rews.sum() for t in trajs])
        print(f"source {s['name']:>12} (gain {s.get('gain',1.0)}, noise {s.get('noise',0.0)}): "
              f"return {rets.mean():.0f} +- {rets.std():.0f}")
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
