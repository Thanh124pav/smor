"""ReweightingEvidence: the stable output object consumed by a future acquisition module.

See PLAN.md §16. Everything a downstream acquisition/scaling module needs to reason about
data utility is captured here, without exposing the learner implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class ReweightingEvidence:
    """Time-series evidence of data utility produced by ``OnlineReweighter.fit``.

    Arrays use shape conventions:
        beta_history          (T_updates, M)      beta after each update
        hypergradient_history (T_updates, M)      raw group hypergradient per update
    where M is the number of groups and T_updates the number of beta updates.
    """

    beta_history: np.ndarray                     # (T, M)
    hypergradient_history: np.ndarray            # (T, M)
    group_assignments: np.ndarray                # (N,) group id per trajectory
    group_fidelity: np.ndarray                   # (M,) fidelity label per group
    group_stats: dict = field(default_factory=dict)
    eval_history: dict = field(default_factory=dict)
    compute_history: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    final_policy_path: Optional[str] = None

    # ---- convenience ----------------------------------------------------
    @property
    def final_beta(self) -> np.ndarray:
        """Terminal learned weight vector (PLAN.md §3.3 ``final_beta_only``)."""
        return np.asarray(self.beta_history)[-1]

    @property
    def n_groups(self) -> int:
        return int(np.asarray(self.group_fidelity).shape[0])

    def save(self, path: str | Path) -> Path:
        """Save arrays + a JSON sidecar of scalar metadata to ``path`` (``.npz``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            beta_history=np.asarray(self.beta_history),
            hypergradient_history=np.asarray(self.hypergradient_history),
            group_assignments=np.asarray(self.group_assignments),
            group_fidelity=np.asarray(self.group_fidelity),
        )
        sidecar = path.with_suffix(".meta.json")
        sidecar.write_text(json.dumps(_jsonify({
            "group_stats": self.group_stats,
            "eval_history": self.eval_history,
            "compute_history": self.compute_history,
            "config": self.config,
            "final_policy_path": self.final_policy_path,
        }), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReweightingEvidence":
        path = Path(path)
        arrs = np.load(path, allow_pickle=False)
        sidecar = path.with_suffix(".meta.json")
        meta: dict[str, Any] = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        return cls(
            beta_history=arrs["beta_history"],
            hypergradient_history=arrs["hypergradient_history"],
            group_assignments=arrs["group_assignments"],
            group_fidelity=arrs["group_fidelity"],
            group_stats=meta.get("group_stats", {}),
            eval_history=meta.get("eval_history", {}),
            compute_history=meta.get("compute_history", {}),
            config=meta.get("config", {}),
            final_policy_path=meta.get("final_policy_path"),
        )

    def summary(self) -> dict:
        beta = np.asarray(self.beta_history)
        return {
            "n_updates": int(beta.shape[0]),
            "n_groups": self.n_groups,
            "final_beta": self.final_beta.tolist(),
            "final_policy_path": self.final_policy_path,
        }


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
