"""Meta-World integration (state-based) — a real manipulation benchmark for SMOR.

Provides:
  * MetaWorldVecEnv  — a batched wrapper exposing the same duck-typed interface as
    :class:`PointMassEnv` (``reset(batch_size)`` / ``step(action)`` / ``horizon`` /
    ``obs_dim`` / ``act_dim`` with ``info['success']``), so ``rollout_policy`` and
    ``BCLearner.evaluate`` work unchanged;
  * scripted-policy lookup + demonstration collection with per-source corruption, to build
    multi-fidelity demonstration datasets on a task where success rate is NOT saturated.

Note: MuJoCo dynamics are not differentiable here, so the closed-loop return objective is
unavailable; use the held-out (clean/proficient) validation loss as the outer objective.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def scripted_policy_for(task: str):
    """Return an instance of the Meta-World scripted policy for ``task`` (e.g. 'reach-v3')."""
    import metaworld.policies as P

    parts = task.split("-")
    version = parts[-1] if parts[-1].startswith("v") else ""
    words = parts[:-1] if version else parts
    camel = "".join(w.capitalize() for w in words)
    name = f"Sawyer{camel}{version.upper()}Policy"
    if not hasattr(P, name):
        raise ValueError(f"no scripted policy '{name}' for task '{task}'. "
                         f"Available: {[n for n in dir(P) if n.endswith('Policy')][:10]}...")
    return getattr(P, name)()


class MetaWorldVecEnv:
    """Batched Meta-World single-task env with a PointMassEnv-compatible interface."""

    def __init__(self, task: str = "reach-v3", horizon: int = 200,
                 device="cpu", seed: int = 0):
        import gymnasium as gym  # local import: heavy
        import metaworld  # noqa: F401  registers the Meta-World namespace
        self._gym = gym
        self.task = task
        self.cfg_horizon = int(horizon)
        self.device = torch.device(device)
        self._seed = seed
        self._envs: list = []
        self._make = lambda: gym.make("Meta-World/MT1", env_name=task)
        probe = self._make()
        obs, _ = probe.reset(seed=seed)
        self.obs_dim = int(np.asarray(obs).shape[0])
        self.act_dim = int(probe.action_space.shape[0])
        probe.close()
        self._done: Optional[np.ndarray] = None
        self._last_obs: Optional[np.ndarray] = None

    @property
    def horizon(self) -> int:
        return self.cfg_horizon

    def _ensure(self, batch_size: int):
        if len(self._envs) != batch_size:
            for e in self._envs:
                e.close()
            self._envs = [self._make() for _ in range(batch_size)]

    def reset(self, batch_size: int) -> torch.Tensor:
        self._ensure(batch_size)
        obs = []
        for i, e in enumerate(self._envs):
            o, _ = e.reset(seed=self._seed + 1000 + i)
            obs.append(np.asarray(o, dtype=np.float32))
        self._last_obs = np.stack(obs)
        self._done = np.zeros(batch_size, dtype=bool)
        self._success = np.zeros(batch_size, dtype=bool)
        return torch.from_numpy(self._last_obs).to(self.device)

    def step(self, action: torch.Tensor):
        a = action.detach().cpu().numpy().astype(np.float32)
        a = np.clip(a, -1.0, 1.0)
        obs, rew = [], []
        for i, e in enumerate(self._envs):
            if self._done[i]:
                obs.append(self._last_obs[i]); rew.append(0.0); continue
            o, r, term, trunc, info = e.step(a[i])
            self._success[i] = self._success[i] or bool(info.get("success", 0.0))
            self._done[i] = bool(term or trunc)
            obs.append(np.asarray(o, dtype=np.float32)); rew.append(float(r))
        self._last_obs = np.stack(obs)
        obs_t = torch.from_numpy(self._last_obs).to(self.device)
        rew_t = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done_t = torch.from_numpy(self._done.copy()).to(self.device)
        info = {"success": torch.from_numpy(self._success.copy()).to(self.device)}
        return obs_t, rew_t, done_t, info

    def close(self):
        for e in self._envs:
            e.close()
        self._envs = []


# --- multi-fidelity demonstrations from (corrupted) scripted policies ---------
#
# Each "source" = a scripted expert corrupted like a real teleop device: magnitude gain
# (over/undershoot), Gaussian jitter, and occasional blunders. Success rate on the task is
# sensitive to these, so — unlike the saturated point-mass — the differences are meaningful.
METAWORLD_SOURCES = [
    {"name": "spacemouse",  "gain": 0.85, "noise": 0.10, "random_prob": 0.00},
    {"name": "teleop",      "gain": 1.20, "noise": 0.20, "random_prob": 0.05},
    {"name": "kinesthetic", "gain": 1.00, "noise": 0.30, "random_prob": 0.02},
    {"name": "keyboard",    "gain": 0.70, "noise": 0.15, "random_prob": 0.03},
    {"name": "vr_control",  "gain": 1.35, "noise": 0.12, "random_prob": 0.08},
]


def collect_metaworld_demos(task, n_traj, gain=1.0, noise=0.0, random_prob=0.0,
                            demo_horizon=150, seed=0):
    """Roll a (corrupted) scripted policy for ``n_traj`` episodes; return (obs, act) tensors
    of shape (n_traj, demo_horizon, ·). The corrupted action is recorded as the BC target."""
    import gymnasium as gym
    import metaworld  # noqa: F401  registers the Meta-World namespace
    env = gym.make("Meta-World/MT1", env_name=task)
    policy = scripted_policy_for(task)
    rng = np.random.default_rng(seed)
    obs_buf, act_buf = [], []
    for ep in range(n_traj):
        o, _ = env.reset(seed=seed + ep)
        obs_ep, act_ep = [], []
        for _ in range(demo_horizon):
            a_exp = np.asarray(policy.get_action(o), dtype=np.float32)
            a = gain * a_exp + noise * rng.standard_normal(a_exp.shape).astype(np.float32)
            if random_prob > 0 and rng.random() < random_prob:
                a = rng.uniform(-1, 1, size=a_exp.shape).astype(np.float32)
            a = np.clip(a, -1.0, 1.0)
            obs_ep.append(np.asarray(o, dtype=np.float32)); act_ep.append(a)
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                o, _ = env.reset(seed=seed + ep + 10_000)  # keep fixed length
        obs_buf.append(np.stack(obs_ep)); act_buf.append(np.stack(act_ep))
    env.close()
    return (torch.from_numpy(np.stack(obs_buf)), torch.from_numpy(np.stack(act_buf)))


def make_metaworld_multisource(task="reach-v3", sources=None, n_per_source=40,
                               demo_horizon=150, seed=0):
    """Build a multi-fidelity DemoDataset from corrupted scripted policies (fidelity=source)."""
    from smor.envs.demos import DemoDataset
    sources = sources if sources is not None else METAWORLD_SOURCES
    obs_l, act_l, fid_l, names = [], [], [], []
    for i, s in enumerate(sources):
        o, a = collect_metaworld_demos(
            task, int(s.get("n_traj", n_per_source)),
            gain=float(s.get("gain", 1.0)), noise=float(s.get("noise", 0.0)),
            random_prob=float(s.get("random_prob", 0.0)),
            demo_horizon=demo_horizon, seed=seed + 100 * (i + 1))
        obs_l.append(o); act_l.append(a)
        fid_l.append(torch.full((o.shape[0],), i, dtype=torch.int64))
        names.append(str(s.get("name", f"source{i}")))
    ds = DemoDataset(obs=torch.cat(obs_l), act=torch.cat(act_l), fidelity=torch.cat(fid_l))
    return ds, names


def make_metaworld_datasets(task="reach-v3", sources=None, n_per_source=40, n_sources=None,
                            demo_horizon=150, n_val=20, seed=0, cache_dir="data/metaworld"):
    """Generate (train multisource, val target, source names) ONCE, with a disk cache.

    Avoids regenerating demonstrations for every method/seed. ``n_sources`` optionally trims the
    default source list. Cache key includes task/sources/sizes/horizon/seed.
    """
    import hashlib
    from pathlib import Path
    from smor.envs.demos import DemoDataset

    src = sources if sources is not None else METAWORLD_SOURCES
    if n_sources is not None:
        src = src[:n_sources]
    key = f"{task}|{[s['name'] for s in src]}|nps{n_per_source}|h{demo_horizon}|v{n_val}|s{seed}"
    tag = hashlib.md5(key.encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"mw_{task}_{tag}.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        return (DemoDataset(d["tr_obs"], d["tr_act"], d["tr_fid"]),
                DemoDataset(d["va_obs"], d["va_act"], d["va_fid"]), d["names"])

    train, names = make_metaworld_multisource(task=task, sources=src, n_per_source=n_per_source,
                                              demo_horizon=demo_horizon, seed=seed)
    val = make_metaworld_target(task=task, n=n_val, demo_horizon=demo_horizon, seed=seed + 4242)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tr_obs": train.obs, "tr_act": train.act, "tr_fid": train.fidelity,
                "va_obs": val.obs, "va_act": val.act, "va_fid": val.fidelity,
                "names": names}, cache)
    return train, val, names


def make_metaworld_target(task="reach-v3", n=20, demo_horizon=150, seed=0):
    """Clean scripted (proficient) demonstrations = deployment target / validation set."""
    from smor.envs.demos import DemoDataset
    o, a = collect_metaworld_demos(task, n, gain=1.0, noise=0.0, random_prob=0.0,
                                   demo_horizon=demo_horizon, seed=seed)
    return DemoDataset(obs=o, act=a, fidelity=torch.zeros(n, dtype=torch.int64))
