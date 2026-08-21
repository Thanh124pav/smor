"""RoboMimic official-dataset loader for SMOR.

RoboMimic (Mandlekar et al., CoRL 2021) is the one widely-used imitation-learning benchmark that
ships demonstrations of *labelled, varying human quality* — exactly the multi-fidelity structure
Module 1 reweights. Each task comes in several variants:

* ``ph``  — proficient-human: 200 clean teleop demos (single high-quality source).
* ``mh``  — multi-human: 300 demos from 6 operators, grouped by skill into the filter keys
            ``better`` / ``okay`` / ``worse`` (100 each). These tiers are the real fidelity
            labels SMOR groups on.
* ``mg``  — machine-generated: SAC rollouts of mixed success (a noisy, lower-quality source);
            per-demo success is recoverable from the sparse reward.

The loader reads the low-dim HDF5 files (no robosuite/MuJoCo needed for training), concatenates a
configurable set of low-dim observation keys, and returns a
:class:`smor.data.trajectory_dataset.TrajectoryDataset` whose ``fidelity`` label is the source /
quality-tier index. Arbitrary mixes across tasks and variants (PH+MH, MH+MG, ...) are supported;
see :func:`load_robomimic_mix` and :func:`smor.data.robomimic.spec.parse_mix`.
"""

from smor.data.robomimic.loader import (
    DEFAULT_OBS_KEYS,
    Component,
    load_robomimic_mix,
)
from smor.data.robomimic.registry import (
    dataset_url,
    ensure_dataset,
    local_path,
)
from smor.data.robomimic.spec import PRESETS, parse_mix

__all__ = [
    "DEFAULT_OBS_KEYS",
    "Component",
    "load_robomimic_mix",
    "dataset_url",
    "ensure_dataset",
    "local_path",
    "PRESETS",
    "parse_mix",
]
