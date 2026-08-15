"""CAIL baselines.

* ``common_backbone`` — the apples-to-apples CAIL-style confidence baseline (K=1 one-step +
  optional CAIL ranking loss) on the shared BC backbone (implemented, used vs SMOR).
* ``adapter.CAILAIRLAdapter`` — original AIRL-CAIL adapter STUB (Stage A, needs MuJoCo RL).
"""

from smor.baselines.cail.adapter import CAILAIRLAdapter
from smor.baselines.cail.common_backbone import (
    cail_style_config, group_quality_from_sources,
)

__all__ = ["CAILAIRLAdapter", "cail_style_config", "group_quality_from_sources"]
