"""Official-dataset loaders for SMOR.

The reweighting core depends only on the ``(obs, act, fidelity)`` trajectory contract, never on
where the demonstrations come from. This package adapts *real* imitation-learning datasets to
that contract so Module 1 can reweight demonstrations of genuinely varying optimality instead of
synthetic corrupted-scripted sources.

* :class:`smor.data.trajectory_dataset.TrajectoryDataset` — a variable-length ("ragged")
  trajectory container that duck-types the fixed-horizon ``DemoDataset`` interface the
  ``BCLearner`` consumes.
* :mod:`smor.data.robomimic` — RoboMimic (PH / MH / MG) loader; fidelity = real operator-quality
  tier. See that module for the multi-variant mixing API.
"""

from smor.data.trajectory_dataset import TrajectoryDataset

__all__ = ["TrajectoryDataset"]
