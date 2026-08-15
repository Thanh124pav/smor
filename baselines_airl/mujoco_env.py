"""Batched seals/HalfCheetah-v1 wrapper with the PointMassEnv-compatible interface,
so smor's rollout_policy / BCLearner.evaluate work unchanged. Reward-based (no success flag).
Runs in the smor-airl env (gymnasium 0.29)."""

from __future__ import annotations

import numpy as np
import torch


class MujocoVecEnv:
    def __init__(self, env_id="seals/HalfCheetah-v1", horizon=1000, device="cpu", seed=0):
        import gymnasium as gym
        import seals  # noqa: F401  registers seals/ envs
        self.env_id = env_id
        self.cfg_horizon = int(horizon)
        self.device = torch.device(device)
        self._seed = seed
        self._make = lambda: gym.make(env_id)
        probe = self._make()
        self.obs_dim = int(probe.observation_space.shape[0])
        self.act_dim = int(probe.action_space.shape[0])
        self._alow = probe.action_space.low
        self._ahigh = probe.action_space.high
        probe.close()
        self._envs: list = []

    @property
    def horizon(self):
        return self.cfg_horizon

    def _ensure(self, b):
        if len(self._envs) != b:
            for e in self._envs:
                e.close()
            self._envs = [self._make() for _ in range(b)]

    def reset(self, batch_size):
        self._ensure(batch_size)
        obs = [np.asarray(e.reset(seed=self._seed + 1000 + i)[0], dtype=np.float32)
               for i, e in enumerate(self._envs)]
        self._last = np.stack(obs)
        self._done = np.zeros(batch_size, dtype=bool)
        return torch.from_numpy(self._last).to(self.device)

    def step(self, action):
        a = np.clip(action.detach().cpu().numpy().astype(np.float32), self._alow, self._ahigh)
        obs, rew = [], []
        for i, e in enumerate(self._envs):
            if self._done[i]:
                obs.append(self._last[i]); rew.append(0.0); continue
            o, r, term, trunc, _ = e.step(a[i])
            self._done[i] = bool(term or trunc)
            obs.append(np.asarray(o, dtype=np.float32)); rew.append(float(r))
        self._last = np.stack(obs)
        info = {"success": torch.zeros(len(self._envs), device=self.device)}
        return (torch.from_numpy(self._last).to(self.device),
                torch.tensor(rew, dtype=torch.float32, device=self.device),
                torch.from_numpy(self._done.copy()).to(self.device), info)

    def close(self):
        for e in self._envs:
            e.close()
        self._envs = []
