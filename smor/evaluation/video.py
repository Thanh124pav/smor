"""Render a trained policy on a Meta-World task to a video file (mp4 via cv2, gif fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def render_metaworld_video(
    task: str,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    path: str,
    n_episodes: int = 3,
    horizon: int = 200,
    seed: int = 0,
    fps: int = 30,
) -> str:
    """Roll ``policy_fn`` (obs->action) on ``task`` and write frames to ``path``.

    Returns the path actually written (``.mp4`` if OpenCV succeeds, else ``.gif``).
    """
    import gymnasium as gym
    import metaworld  # noqa: F401  registers namespace

    from smor.envs.metaworld_env import scripted_policy_for  # noqa: F401 (keeps task valid)

    env = gym.make("Meta-World/MT1", env_name=task, render_mode="rgb_array")
    frames = []
    successes = 0
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep)
        ok = 0
        for _ in range(horizon):
            a = np.clip(np.asarray(policy_fn(o), dtype=np.float32), -1.0, 1.0)
            o, r, term, trunc, info = env.step(a)
            frames.append(env.render())
            ok = ok or int(info.get("success", 0))
            if term or trunc:
                break
        successes += ok
    env.close()

    out = _write(frames, path, fps)
    print(f"video: {out}  ({len(frames)} frames, {n_episodes} eps, "
          f"{successes}/{n_episodes} success)")
    return out


def _write(frames, path: str, fps: int) -> str:
    path = str(path)
    frames = [np.asarray(f, dtype=np.uint8) for f in frames if f is not None]
    if not frames:
        raise ValueError("no frames rendered")
    # try mp4 via OpenCV (no ffmpeg python pkg required)
    try:
        import cv2

        if not path.endswith(".mp4"):
            path = str(Path(path).with_suffix(".mp4"))
        h, w = frames[0].shape[:2]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter failed to open")
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        if Path(path).stat().st_size > 1024:
            return path
        raise RuntimeError("mp4 too small")
    except Exception:
        # gif fallback (imageio + pillow, no ffmpeg)
        import imageio

        gif = str(Path(path).with_suffix(".gif"))
        Path(gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif, frames[::2], duration=1.0 / max(1, fps // 2))
        return gif
