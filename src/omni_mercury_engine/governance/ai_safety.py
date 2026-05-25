"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""AI-assurance governance scalars, each kept or dropped on a per-family **signal vet**.

The three AI-assurance frameworks were treated as genuinely open and vetted against this
engine's *actual* observable surface (not against their reputation as "just checklists").
The verdicts (recorded in :data:`omni_mercury_engine.governance.contract.GOVERNANCE_FAMILY_VET`):

* **NIST AI RMF 1.0 -- UNAVAILABLE-capable (kept).**  GOVERN/MAP/MANAGE are governance
  *processes* with no runtime signal, but the **MEASURE** function is quantitative and maps
  onto genuine runtime trustworthiness metrics this engine computes: fairness
  (``ml/bias_detection.py:49`` ``FairnessResult.overall_score``), performance
  (``evaluation/metrics.py:33`` ``AnomalyMetrics.auc_roc``), and drift stability
  (``ml/drift.py:63`` ``DriftResult``).  So the family is grounded *through MEASURE*.

* **MITRE ATLAS -- UNAVAILABLE-capable (kept).**  The engine observes adversary activity
  against its surface at runtime: ``security/threat_detection.py:135``
  ``ThreatDetector.detect_all`` (wired live at ``engine.py:2759``) emits
  ``threat_type``/``confidence``.  Honestly scoped: this covers the conventional/input-layer
  tactics the web-payload detector surfaces (initial_access, discovery), **not** the
  adversarial-ML tactics (evasion/poisoning/model-extraction) -- the engine has no detector
  for those.

* **OWASP Top 10 for LLM Applications (2025) -- UNDECIDABLE (dropped).**  The engine *does*
  run an LLM (``models/foundation/llm_adapter.py:153``), but ships **no** prompt-injection /
  output-handling / system-prompt-leakage / jailbreak detector, no per-category guardrail
  output, and no token-consumption accounting (the API rate-limiter is generic).  No runtime
  signal can produce a per-OWASP-category value; the only possible input is an operator
  checklist, which is not a runtime signal -- so the family would be UNDECIDABLE-in-disguise
  and is **not built**.  Its verdict is recorded in the vet table and asserted by tests.

Both kept families are **metric-only**, so they never perturb the σ_Immutable gate, and each
abstains UNAVAILABLE when its signal is absent this run rather than fabricating a value.
"""

from typing import TYPE_CHECKING

from omni_mercury_engine.governance.contract import GovernanceScalar, grounded, unavailable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_NIST_FAMILY = "nist_ai_rmf"
_ATLAS_FAMILY = "mitre_atlas"

# NIST AI RMF 1.0 MEASURE-function trustworthiness characteristics that map to a *real*
# runtime metric in this engine.  Each key cites its in-engine source; values are unit
# trustworthiness scores in [0, 1] (1.0 = best).  (Robustness is deliberately excluded:
# ``core/ethical_config.py:209 omni_adversarial_robustness`` is a static default constant,
# not a runtime-computed measurement.)
_NIST_MEASURE_METRICS: dict[str, str] = {
    "fairness": "ml/bias_detection.py:49 FairnessResult.overall_score",
    "performance": "evaluation/metrics.py:33 AnomalyMetrics.auc_roc",
    "drift_stability": "ml/drift.py:63 DriftResult (1 - drifting-feature fraction)",
}

# The ATLAS tactics this engine's threat detectors can actually surface, keyed by the
# ``threat_type`` strings emitted by ``security/threat_detection.py`` ``detect_*`` methods.
# This is an honest *subset* of the 14-tactic ATLAS matrix.
_THREAT_TYPE_TO_ATLAS_TACTIC: dict[str, str] = {
    "sql_injection": "initial_access",
    "xss": "initial_access",
    "path_traversal": "discovery",
}
_ATLAS_OBSERVABLE_TACTICS: tuple[str, ...] = tuple(
    sorted(set(_THREAT_TYPE_TO_ATLAS_TACTIC.values()))
)


def _is_unit(value: object) -> bool:
    """Whether ``value`` is a real number in the closed unit interval ``[0, 1]``."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0


def nist_ai_rmf_measure_scalar(
    *, measurements: Mapping[str, float] | None = None
) -> GovernanceScalar:
    """Ground NIST AI RMF (via the MEASURE function) in live trustworthiness metrics.

    Args:
        measurements: Mapping of MEASURE metric name (see :data:`_NIST_MEASURE_METRICS`) ->
            unit score in ``[0, 1]``, sourced from the engine's runtime signals.  ``None``
            or no recognised/valid metric this run abstains UNAVAILABLE (the capability is
            real; the metric simply was not produced this execution).

    Returns:
        GROUNDED with the mean of the present MEASURE metrics, else an UNAVAILABLE abstention.
    """
    name = "omni_nist_airmf_measure"
    if not measurements:
        return unavailable(
            name,
            family=_NIST_FAMILY,
            reason="NIST AI RMF MEASURE: no runtime trustworthiness metric this run",
            missing_inputs=("measurements",),
        )
    valid = {
        k: float(v) for k, v in measurements.items() if k in _NIST_MEASURE_METRICS and _is_unit(v)
    }
    if not valid:
        return unavailable(
            name,
            family=_NIST_FAMILY,
            reason="NIST AI RMF MEASURE: no recognised metric in [0,1] supplied",
            missing_inputs=tuple(sorted(_NIST_MEASURE_METRICS)),
        )
    value = sum(valid.values()) / len(valid)
    return grounded(
        name,
        value,
        family=_NIST_FAMILY,
        reason=f"NIST AI RMF MEASURE = mean over {sorted(valid)} ({len(valid)} metric(s))",
        provenance={"measurements": valid, "function": "MEASURE"},
    )


def mitre_atlas_scalar(
    *, observed_events: Sequence[Mapping[str, object]] | None = None
) -> GovernanceScalar:
    """Ground MITRE ATLAS coverage in observed runtime threat-detection events.

    Args:
        observed_events: Sequence of threat-detection outputs, each a mapping with a
            ``threat_type`` key (as emitted by ``ThreatDetector.detect_*``).  ``None`` or an
            empty sequence abstains UNAVAILABLE (the live detector exists at
            ``engine.py:2759``; it simply observed nothing this run).

    Returns:
        GROUNDED with observed-tactic coverage = (distinct observable ATLAS tactics seen) /
        (engine's observable ATLAS surface), else an UNAVAILABLE abstention.
    """
    name = "omni_mitre_atlas_coverage"
    if not observed_events:
        return unavailable(
            name,
            family=_ATLAS_FAMILY,
            reason="MITRE ATLAS: no threat-detection event observed this run",
            missing_inputs=("observed_events",),
        )
    observed_tactics = {
        _THREAT_TYPE_TO_ATLAS_TACTIC[str(ev.get("threat_type"))]
        for ev in observed_events
        if str(ev.get("threat_type")) in _THREAT_TYPE_TO_ATLAS_TACTIC
    }
    if not observed_tactics:
        return unavailable(
            name,
            family=_ATLAS_FAMILY,
            reason="MITRE ATLAS: no event maps to an observable ATLAS tactic",
            missing_inputs=("observed_events",),
        )
    coverage = len(observed_tactics) / len(_ATLAS_OBSERVABLE_TACTICS)
    return grounded(
        name,
        coverage,
        family=_ATLAS_FAMILY,
        reason=(
            f"MITRE ATLAS observed-tactic coverage = {len(observed_tactics)}/"
            f"{len(_ATLAS_OBSERVABLE_TACTICS)} ({sorted(observed_tactics)})"
        ),
        provenance={
            "observed_tactics": sorted(observed_tactics),
            "observable_surface": list(_ATLAS_OBSERVABLE_TACTICS),
        },
    )


def ai_safety_scalars(
    *,
    nist_measurements: Mapping[str, float] | None = None,
    atlas_events: Sequence[Mapping[str, object]] | None = None,
) -> list[GovernanceScalar]:
    """Build the two kept AI-assurance scalars (NIST AI RMF MEASURE, MITRE ATLAS).

    OWASP LLM Top 10 is intentionally absent: it vets UNDECIDABLE (no runtime signal in this
    engine) and is therefore not built -- see this module's docstring and the vet table.

    Args:
        nist_measurements: Live MEASURE metrics for :func:`nist_ai_rmf_measure_scalar`.
        atlas_events: Observed threat-detection events for :func:`mitre_atlas_scalar`.

    Returns:
        ``[nist_scalar, atlas_scalar]`` -- each GROUNDED only where its real signal is present.
    """
    return [
        nist_ai_rmf_measure_scalar(measurements=nist_measurements),
        mitre_atlas_scalar(observed_events=atlas_events),
    ]
