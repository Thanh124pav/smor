"""Small, GPU-friendly environments for training + evaluation."""

from smor.envs.point_mass import PointMassEnv, expert_action

__all__ = ["PointMassEnv", "expert_action"]
