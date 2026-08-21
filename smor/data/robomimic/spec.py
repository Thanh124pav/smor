"""Parse a RoboMimic mix specification (CLI-friendly) into :class:`Component` lists.

Two ways to specify a mix:

* **Preset** — a named, curated mix (see :data:`PRESETS`), e.g. ``mh-tiers`` or ``ph-plus-mg``.
* **DSL string** — comma-separated ``task:dtype[:tier][:n]`` components, e.g.
  ``lift:mh:better,lift:mh:okay,lift:mh:worse`` or ``lift:ph,lift:mg:mg_fail:150``.

The first ``target``-eligible source (or the highest-quality one) becomes the clean validation
target; override per-component by appending ``*`` to a DSL component (``lift:ph*``).
"""

from __future__ import annotations

from typing import List

from smor.data.robomimic.loader import Component

# Curated multi-fidelity mixes. Each value is a list of Component kwargs dicts.
PRESETS = {
    # The canonical multi-human quality-tier reweighting problem (single task, 3 real tiers).
    "mh-tiers": [
        {"task": "lift", "dtype": "mh", "tier": "better", "target": True},
        {"task": "lift", "dtype": "mh", "tier": "okay"},
        {"task": "lift", "dtype": "mh", "tier": "worse"},
    ],
    # Proficient target + the two weaker human tiers (should downweight okay/worse).
    "ph-plus-mh": [
        {"task": "lift", "dtype": "ph", "target": True},
        {"task": "lift", "dtype": "mh", "tier": "okay"},
        {"task": "lift", "dtype": "mh", "tier": "worse"},
    ],
    # Human proficient target vs machine-generated successes vs machine-generated failures.
    "ph-plus-mg": [
        {"task": "lift", "dtype": "ph", "target": True},
        {"task": "lift", "dtype": "mg", "tier": "mg_success"},
        {"task": "lift", "dtype": "mg", "tier": "mg_fail"},
    ],
    # Full quality spectrum on one task: proficient + 3 human tiers + machine failures.
    "full-spectrum": [
        {"task": "lift", "dtype": "ph", "target": True},
        {"task": "lift", "dtype": "mh", "tier": "better"},
        {"task": "lift", "dtype": "mh", "tier": "okay"},
        {"task": "lift", "dtype": "mh", "tier": "worse"},
        {"task": "lift", "dtype": "mg", "tier": "mg_fail"},
    ],
}


def _retask(components: List[dict], task: str) -> List[dict]:
    return [{**c, "task": task} for c in components]


def parse_mix(spec: str, task: str | None = None) -> List[Component]:
    """Parse a preset name or a DSL string into a list of :class:`Component`.

    ``task`` (if given) overrides the task of every component — lets a preset defined on ``lift``
    be reused on ``can``/``square`` without editing the preset.
    """
    spec = spec.strip()
    if spec in PRESETS:
        raw = PRESETS[spec]
        if task:
            raw = _retask(raw, task)
        return [Component(**c) for c in raw]

    components: List[Component] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        is_target = token.endswith("*")
        token = token.rstrip("*")
        parts = token.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"bad mix component '{token}': expected task:dtype[:tier][:n] (or a preset name "
                f"from {sorted(PRESETS)})."
            )
        t, dtype = parts[0], parts[1]
        tier = parts[2] if len(parts) >= 3 and parts[2] != "" else None
        n = int(parts[3]) if len(parts) >= 4 and parts[3] != "" else None
        components.append(
            Component(task=task or t, dtype=dtype, tier=tier, n=n, target=is_target)
        )
    if not components:
        raise ValueError(f"empty mix spec '{spec}'.")
    return components
