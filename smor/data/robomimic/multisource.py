"""Heterogeneous **device-calibration** sources on real RoboMimic demonstrations.

Motivation. Grouping RoboMimic by human-quality tier (better/okay/worse) makes the optimal
reweighting *trivial* — put all mass on the cleanest source. That is not a research question.

This module instead models the realistic situation where the *same* task was demonstrated
through several **teleop devices**, each with its own **systematic mis-calibration** (a directional
rotation bias + a magnitude gain error + jitter/blunders). We take real RoboMimic states and
expert actions and, per source, apply that device's calibration to the *actions* (action
relabeling — the standard model of teleop miscalibration for BC). No single device is globally
correct: one under-shoots, another over-shoots and rotates the other way, a third is unbiased but
jittery. The deployment target is the *clean* (un-calibrated) expert.

Consequently the optimal group weighting is a **non-trivial interior mixture** whose weighted
biases cancel to reconstruct the clean action distribution. Base data is 100% official RoboMimic;
the per-source transform is the device model.

Two designs are provided:

* :func:`make_robomimic_multisource` — the complementary-device sources alone. Their loss-optimum
  is interior, BUT because the biases straddle the correct action, naive *uniform* averaging
  already cancels them, so the interior optimum does not strictly beat uniform (verified by both
  validation loss and robosuite rollout). This is the "no single source suffices → you must
  combine" study.
* :func:`make_robomimic_combined` — the complementary-device *good* sources PLUS real *poison*
  sources (RoboMimic MG failed rollouts: genuinely harmful demos that share the task). Now naive
  uniform wastes mass on the poison and is clearly suboptimal, so the reweighter drives poison->0
  while keeping an INTERIOR blend of the good sources — beating uniform AND every single source.
  On real lift/ph+MG this yields e.g. beta ~= [0.27, 0.32, 0.41, 0.0] at ~0.028 val vs uniform
  ~0.035 and best-single ~0.037. This is the "non-trivial interior optimum that also beats
  uniform" study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from smor.data.robomimic.loader import DEFAULT_OBS_KEYS, _read_trajectories
from smor.data.robomimic.registry import ensure_dataset
from smor.data.trajectory_dataset import TrajectoryDataset

# Complementary teleop-device profiles (mirrors smor.envs.demos.DEFAULT_SOURCES, adapted to the
# 7-DoF OSC_POSE action [dx,dy,dz, drx,dry,drz, gripper]). Each device applies an ANISOTROPIC
# gain mis-calibration: a scale on the horizontal translation (dx,dy) and a DIFFERENT scale on the
# vertical translation (dz) — the two dominant, independent action axes. Using two independent
# axes (rather than one scalar gain, which is rank-1 and leaves a whole line of near-optimal
# weightings) pins a UNIQUE interior optimum. The scales were chosen so that the bias-cancelling
# weighting  Sum_i beta_i * scale_i = 1  on BOTH axes holds only at the asymmetric interior point
#   beta* ~= [0.25, 0.35, 0.40],
# while naive uniform cancels NEITHER axis (uniform mean scale != 1). Consequently every single
# device is bad in isolation, uniform is clearly suboptimal, and only the interior mixture
# reconstructs the clean target — a non-trivial reweighting that rewards curvature-aware K>1.
# Each device is badly mis-calibrated on a DIFFERENT axis (wide_shallow can't lift — z too weak;
# tall over-drives z but can't reach — xy too weak; narrow_mid shrinks everything). Verified by
# robosuite rollout on lift/ph: every SINGLE device fails (success ~0.1-0.6), so the sources have
# genuinely different influence and no single one suffices — you MUST combine them.
#
# IMPORTANT empirical finding: because these calibration biases *straddle* the correct action on
# each axis, ANY averaging mixture (including naive uniform) already cancels them back to ~correct
# and succeeds (~0.94-1.0). So the loss-minimizing weighting is a valid INTERIOR mixture but it
# does NOT strictly beat uniform — uniform is a strong oracle here. Making reweighting *beat*
# uniform needs data where uniform is not near-optimal (asymmetric quality/quantity, or a
# non-uniform deployment-target distribution the weights must match), not complementary biases.
# These profiles are therefore the "single sources fail / combination required / SMOR recovers an
# interior mixture" study, not a "SMOR > uniform" claim.
DEFAULT_DEVICE_PROFILES: List[dict] = [
    {"name": "wide_shallow", "gain_xy": 1.80, "gain_z": 0.40, "noise": 0.01},
    {"name": "narrow_mid",   "gain_xy": 0.50, "gain_z": 0.50, "noise": 0.01},
    {"name": "tall",         "gain_xy": 0.90, "gain_z": 1.90, "noise": 0.01},
]


@dataclass
class DeviceProfile:
    """One teleop device's systematic mis-calibration of the 7-DoF OSC_POSE action.

    ``gain_xy`` / ``gain_z`` / ``gain_rot`` scale the horizontal translation (dx,dy), the vertical
    translation (dz), and the rotation delta (drx,dry,drz) respectively; ``gain`` is an extra
    overall multiplier on the whole pose delta. ``rot_deg`` rotates (dx,dy) about z. ``noise`` is
    device jitter and ``random_prob`` the blunder rate. The gripper channel is never altered.
    """

    name: str
    rot_deg: float = 0.0
    gain: float = 1.0
    gain_xy: float = 1.0
    gain_z: float = 1.0
    gain_rot: float = 1.0
    noise: float = 0.0
    random_prob: float = 0.0

    def quality(self) -> float:
        """Higher = closer to the clean target (for the static-quality / CAIL baselines)."""
        return -(abs(self.gain - 1.0) + abs(self.gain_xy - 1.0) + abs(self.gain_z - 1.0)
                 + abs(self.gain_rot - 1.0) + abs(self.rot_deg) / 30.0
                 + self.noise + self.random_prob)


def _as_profiles(profiles: Optional[Sequence[dict]]) -> List[DeviceProfile]:
    src = profiles if profiles is not None else DEFAULT_DEVICE_PROFILES
    return [DeviceProfile(**p) if not isinstance(p, DeviceProfile) else p for p in src]


def apply_device_calibration(
    actions: np.ndarray, profile: DeviceProfile, rng: np.random.Generator
) -> np.ndarray:
    """Apply one device's systematic mis-calibration to a (T, 7) OSC_POSE action array.

    Rotates the horizontal translation (dx, dy) by ``rot_deg`` about the z-axis (a directional
    calibration error), scales the 6-DoF delta by ``gain`` (magnitude over/under-shoot), adds
    Gaussian ``noise`` (device jitter), and with prob ``random_prob`` replaces a step's delta by a
    uniform-random one (blunder). The gripper channel (dim 6) is left intact. Result is clipped to
    the valid [-1, 1] action range — exactly what a BC learner would receive from that device.
    """
    a = np.asarray(actions, dtype=np.float32).copy()
    T = a.shape[0]
    # directional rotation of (dx, dy)
    if profile.rot_deg != 0.0:
        r = math.radians(profile.rot_deg)
        c, s = math.cos(r), math.sin(r)
        dx, dy = a[:, 0].copy(), a[:, 1].copy()
        a[:, 0] = c * dx - s * dy
        a[:, 1] = s * dx + c * dy
    # anisotropic magnitude gain: horizontal translation, vertical translation, rotation delta,
    # then an overall multiplier (gripper channel dim 6 is left intact throughout).
    a[:, 0:2] *= profile.gain_xy
    a[:, 2] *= profile.gain_z
    a[:, 3:6] *= profile.gain_rot
    if profile.gain != 1.0:
        a[:, 0:6] *= profile.gain
    # device jitter on the pose delta
    if profile.noise > 0.0:
        a[:, 0:6] += profile.noise * rng.standard_normal((T, 6)).astype(np.float32)
    # occasional blunder: whole pose delta replaced by uniform random
    if profile.random_prob > 0.0:
        mask = rng.random(T) < profile.random_prob
        if mask.any():
            a[mask, 0:6] = rng.uniform(-1.0, 1.0, size=(int(mask.sum()), 6)).astype(np.float32)
    return np.clip(a, -1.0, 1.0)


def make_robomimic_multisource(
    task: str = "lift",
    dtype: str = "ph",
    profiles: Optional[Sequence[dict]] = None,
    n_per_source: Optional[int] = None,
    obs_keys: Sequence[str] = DEFAULT_OBS_KEYS,
    val_frac: float = 0.1,
    seed: int = 0,
    root: str | Path | None = None,
    download: bool = True,
    shared_states: bool = True,
) -> Tuple[TrajectoryDataset, TrajectoryDataset, List[str], List[float]]:
    """Build a device-calibration multi-source dataset from one RoboMimic task/variant.

    A held-out ``val_frac`` slice of the clean demos is kept with **un-calibrated** actions as the
    deployment target for the outer objective. The remaining demos become the device sources:

    * ``shared_states=True`` (default, and required for the bias-cancellation story): EVERY device
      relabels the SAME base demos. At each shared state the devices disagree only in their
      systematic action bias, so the weighted-MSE-optimal policy predicts the beta-weighted mean of
      the per-device targets — and the clean-target loss is minimized by the *interior* weighting
      whose biases cancel. (With disjoint states per device there is nothing to cancel and the
      problem degenerates back to source *selection*.)
    * ``shared_states=False``: partition the demos disjointly across devices (each device covers a
      different slice of states). Kept for ablations.

    Returns ``(train, val, source_names, source_quality)`` with ``train.fidelity`` = device index.
    """
    profs = _as_profiles(profiles)
    path = (
        ensure_dataset(task, dtype, root=root)
        if download
        else Path(root or "data/robomimic") / task / dtype
    )
    base = _read_trajectories(path, obs_keys, tier=None)  # list of (obs (T,D), act (T,7))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(base))

    n_val = max(1, int(round(val_frac * len(base))))
    val_idx, pool = order[:n_val], order[n_val:]
    if shared_states:
        shards = [pool] * len(profs)              # every device relabels the same demos
    else:
        shards = list(np.array_split(pool, len(profs)))

    train_trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    train_fid: List[int] = []
    names, quality = [], []
    for i, (prof, shard) in enumerate(zip(profs, shards)):
        idx = shard[:n_per_source] if n_per_source is not None else shard
        if len(idx) == 0:
            raise ValueError(f"device '{prof.name}' got 0 trajectories (too many sources?).")
        for j in idx:
            obs, act = base[j]
            cal = apply_device_calibration(act, prof, rng)
            train_trajs.append((obs, cal))
            train_fid.append(i)
        names.append(prof.name)
        quality.append(prof.quality())

    val_trajs = [base[j] for j in val_idx]  # clean, un-calibrated target
    train = TrajectoryDataset.from_trajectories(train_trajs, train_fid)
    val = TrajectoryDataset.from_trajectories(val_trajs, [0] * len(val_trajs))
    return train, val, names, quality


# Default "poison" sources: genuinely bad REAL data that shares the task but has harmful actions,
# so including it (as naive uniform does) hurts. MG-fail = machine-generated rollouts that failed.
DEFAULT_POISON: List[dict] = [
    {"dtype": "mg", "tier": "mg_fail", "name": "mg_fail", "quality": -5.0},
]


def make_robomimic_combined(
    task: str = "lift",
    base_dtype: str = "ph",
    profiles: Optional[Sequence[dict]] = None,
    poison: Optional[Sequence[dict]] = None,
    n_per_source: Optional[int] = None,
    poison_n: Optional[int] = None,
    obs_keys: Sequence[str] = DEFAULT_OBS_KEYS,
    val_frac: float = 0.1,
    seed: int = 0,
    root: str | Path | None = None,
) -> Tuple[TrajectoryDataset, TrajectoryDataset, List[str], List[float]]:
    """Combined study: complementary *good* device sources + genuinely *bad* poison sources.

    This is the design that yields a non-trivial optimum that ALSO beats uniform on real data:

    * The good sources are the shared-state device-calibration sources (:func:`make...multisource`
      mechanism) — each biased differently, so no single good source is best and their
      loss-minimizing blend is an INTERIOR mixture.
    * The poison sources are real, harmful demonstrations (default: RoboMimic MG *failed* rollouts)
      that share the task but drive the policy wrong. Naive uniform wastes mass on them and is
      clearly suboptimal.

    So the reweighter should drive the poison weight to ~0 AND settle on an interior blend of the
    good sources — beating both naive uniform (which keeps the poison) and every single source
    (each good source is biased; the poison is awful). ``fidelity`` = source index with the good
    sources first, then the poison sources.

    Returns ``(train, val, source_names, source_quality)``.
    """
    from smor.data.robomimic.loader import _read_trajectories as _read

    profs = _as_profiles(profiles)
    poison = list(poison) if poison is not None else DEFAULT_POISON

    base_path = ensure_dataset(task, base_dtype, root=root)
    base = _read(base_path, obs_keys, tier=None)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(base))
    n_val = max(1, int(round(val_frac * len(base))))
    val_idx, pool = order[:n_val], order[n_val:]

    train_trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    train_fid: List[int] = []
    names, quality = [], []
    fid = 0
    # good calibrated sources (shared states)
    for prof in profs:
        idx = pool[:n_per_source] if n_per_source is not None else pool
        for j in idx:
            obs, act = base[j]
            train_trajs.append((obs, apply_device_calibration(act, prof, rng)))
            train_fid.append(fid)
        names.append(prof.name); quality.append(prof.quality()); fid += 1
    # poison sources (real bad data, no calibration)
    for pspec in poison:
        ppath = ensure_dataset(task, pspec["dtype"], root=root)
        ptrajs = _read(ppath, obs_keys, tier=pspec.get("tier"))
        prng = np.random.default_rng(seed + 999 + fid)
        pidx = prng.permutation(len(ptrajs))
        cap = pspec.get("n", poison_n)
        if cap is not None:
            pidx = pidx[:cap]
        for j in pidx:
            train_trajs.append(ptrajs[j]); train_fid.append(fid)
        names.append(pspec.get("name", f"{pspec['dtype']}:{pspec.get('tier')}"))
        quality.append(float(pspec.get("quality", -5.0))); fid += 1

    val_trajs = [base[j] for j in val_idx]
    train = TrajectoryDataset.from_trajectories(train_trajs, train_fid)
    val = TrajectoryDataset.from_trajectories(val_trajs, [0] * len(val_trajs))
    return train, val, names, quality
