"""Real (batched) env rollouts producing episodic return + success rate."""

from __future__ import annotations

from typing import Callable, Dict

import torch


@torch.no_grad()
def rollout_policy(
    env,
    policy: Callable[[torch.Tensor], torch.Tensor],
    n_episodes: int = 64,
    device: str | torch.device | None = None,
) -> Dict[str, float]:
    """Run ``n_episodes`` parallel episodes and report return + success.

    Args:
        env:    a :class:`PointMassEnv`-like object with ``reset``/``step``/``horizon``.
        policy: maps observation ``(B, obs_dim)`` -> action ``(B, act_dim)``.
        n_episodes: number of parallel episodes (batched).
    """
    obs = env.reset(n_episodes)
    total_reward = torch.zeros(n_episodes, device=obs.device)
    last_success = torch.zeros(n_episodes, dtype=torch.bool, device=obs.device)
    for _ in range(env.horizon):
        action = policy(obs)
        obs, reward, _done, info = env.step(action)
        total_reward = total_reward + reward
        last_success = info["success"]
    return {
        "return_mean": float(total_reward.mean()),
        "return_std": float(total_reward.std()),
        "success_rate": float(last_success.float().mean()),
        "n_episodes": int(n_episodes),
    }
