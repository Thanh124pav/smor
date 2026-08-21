"""Optional robosuite rollout env for RoboMimic tasks (real closed-loop success-rate eval).

This is the "option 2" evaluator: it reconstructs the exact robosuite/MuJoCo environment a
RoboMimic dataset was collected in (from the ``env_args`` metadata stored in the HDF5) and rolls
the BC policy in it, reporting a real task success rate. It exposes the same duck-typed interface
as :class:`smor.envs.metaworld_env.MetaWorldVecEnv` / ``PointMassEnv`` so ``rollout_policy`` and
``BCLearner.evaluate`` work unchanged.

The env is built with **robosuite directly** from the dataset's ``env_kwargs`` (no ``robomimic``
dependency, so the fiddly ``egl_probe`` build is avoided), forced fully headless — state-based
low-dim observations need no MuJoCo rendering / OpenGL at all. Install with::

    pip install "robosuite>=1.4,<1.5" mujoco

The observation vector is built by concatenating exactly the ``obs_keys`` used for training, in
the same order. The live robosuite obs dict names the flattened object state ``object-state``
whereas the RoboMimic HDF5 stored it as ``object``; the two are aliased here so training and
rollout see the same vector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from smor.data.robomimic.loader import DEFAULT_OBS_KEYS
from smor.data.robomimic.registry import local_path

_MISSING = (
    "RobosuiteVecEnv needs the optional 'robosuite' (+ 'mujoco') packages. Install with "
    "`pip install \"robosuite>=1.4,<1.5\" mujoco`, or run evaluation with rollout disabled "
    "(held-out validation loss, the default)."
)

# Live robosuite obs-dict key -> RoboMimic HDF5 obs key. Only the object state differs in name.
_OBS_ALIASES = {"object": "object-state"}


def _read_env_args(dataset_path: Path) -> dict:
    import h5py

    with h5py.File(str(dataset_path), "r") as f:
        return json.loads(f["data"].attrs["env_args"])


class RobosuiteVecEnv:
    """Batched robosuite env for a RoboMimic task, reconstructed from dataset metadata."""

    def __init__(
        self,
        task: str = "lift",
        dtype: str = "ph",
        obs_keys: Sequence[str] = DEFAULT_OBS_KEYS,
        horizon: int = 400,
        device: str = "cpu",
        seed: int = 0,
        root: str | Path | None = None,
        reward_shaping: bool = False,
    ):
        try:
            import robosuite  # noqa: F401
        except Exception as e:  # pragma: no cover - optional heavy dep
            raise ImportError(_MISSING) from e

        path = local_path(task, dtype, root)
        if not path.exists():
            raise FileNotFoundError(
                f"dataset {path} not found; download it first (needed for env metadata)."
            )
        env_args = _read_env_args(path)
        kw = dict(env_args["env_kwargs"])
        # reward_shaping=True gives a DENSE reward (reach/grasp/lift components) so rollout returns
        # vary continuously even before full success — important signal for a REINFORCE outer
        # objective (sparse success reward gives no gradient until the policy already succeeds).
        kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                  reward_shaping=bool(reward_shaping))
        self._env_name = env_args["env_name"]
        self._kw = kw
        self.task = task
        self.obs_keys = tuple(obs_keys)
        self.cfg_horizon = int(horizon)
        self.device = torch.device(device)
        self._seed = int(seed)
        self._envs: list = []

        probe = self._make()
        obs = probe.reset()
        self.obs_dim = int(self._obs_vec(obs).shape[-1])
        self.act_dim = int(probe.action_dim)
        self._probe = probe
        self._last_obs: Optional[np.ndarray] = None
        self._done: Optional[np.ndarray] = None
        self._success: Optional[np.ndarray] = None

    def _make(self):
        import robosuite
        return robosuite.make(self._env_name, **self._kw)

    @property
    def horizon(self) -> int:
        return self.cfg_horizon

    def _obs_vec(self, obs_dict) -> np.ndarray:
        parts = []
        for k in self.obs_keys:
            key = k if k in obs_dict else _OBS_ALIASES.get(k)
            if key is None or key not in obs_dict:
                raise KeyError(f"obs key '{k}' not in robosuite obs {list(obs_dict.keys())}")
            parts.append(np.asarray(obs_dict[key], dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=-1)

    def _ensure(self, batch_size: int):
        if len(self._envs) != batch_size:
            self._envs = [self._probe] + [self._make() for _ in range(batch_size - 1)]

    def reset(self, batch_size: int) -> torch.Tensor:
        self._ensure(batch_size)
        obs = [self._obs_vec(e.reset()) for e in self._envs]
        self._last_obs = np.stack(obs)
        self._done = np.zeros(batch_size, dtype=bool)
        self._success = np.zeros(batch_size, dtype=bool)
        return torch.from_numpy(self._last_obs).to(self.device)

    def step(self, action: torch.Tensor):
        a = np.clip(action.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
        obs, rew = [], []
        for i, e in enumerate(self._envs):
            if self._done[i]:
                obs.append(self._last_obs[i]); rew.append(0.0); continue
            o, r, done, _ = e.step(a[i])
            succ = bool(e._check_success())
            self._success[i] = self._success[i] or succ
            self._done[i] = bool(done) or succ
            obs.append(self._obs_vec(o)); rew.append(float(r))
        self._last_obs = np.stack(obs)
        obs_t = torch.from_numpy(self._last_obs).to(self.device)
        rew_t = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done_t = torch.from_numpy(self._done.copy()).to(self.device)
        info = {"success": torch.from_numpy(self._success.copy()).to(self.device)}
        return obs_t, rew_t, done_t, info

    def close(self):
        for e in self._envs:
            try:
                e.close()
            except Exception:
                pass
        self._envs = []
