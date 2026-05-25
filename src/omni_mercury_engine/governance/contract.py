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

"""Three-state honesty contract for descriptive (metric-only) governance scalars.

Governance, medical, and AI-assurance frameworks *describe* a system; they do not drive
the σ_Immutable decision boundary.  Every scalar defined under this contract is therefore
**metric-only** -- excluded from the operational vector by
:meth:`GlobalOmniScalarNetwork._is_metric_only_scalar` -- so the trained ethical gate is
never perturbed.

The contract has **two orthogonal axes**, not one:

* A *runtime* axis -- were the formula and its inputs both present **this execution**?
* A *design-time* axis -- can a runtime signal for the scalar **ever exist at all** in
  this engine's observable surface?

Their product gives exactly three states (mirroring the cross-repo invariant in
:mod:`omni_mercury_engine.verifiers.three_state`, PR #244):

* :attr:`ScalarState.GROUNDED` -- formula and inputs both present this run.  Registers a
  metric-only scalar.
* :attr:`ScalarState.UNAVAILABLE` -- the formula/oracle is real **and a runtime signal for
  it genuinely exists in this engine**, but the input is absent this run.  Registers
  nothing this run; the capability is real and fires when the signal appears.  A *kept*
  scalar (e.g. SOFA when a lab is missing, MELD-Na when sodium has not flowed yet).
* :attr:`ScalarState.UNDECIDABLE` -- no oracle/formula exists, or **no runtime signal for
  it can ever exist in this engine**.  Registers nothing, *ever*.  A *dropped* scalar.

The distinction between the last two is a per-family **vetting judgment made once at
design time** (see :class:`SignalClass` / :data:`GOVERNANCE_FAMILY_VET`), not a runtime
computation.  Mapping the old two-state ``unavailable`` mechanically onto
:attr:`ScalarState.UNAVAILABLE` would collapse three states back into two; a family is only
honest-by-default if its abstention is UNAVAILABLE *because a real signal exists*.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
)

logger = logging.getLogger(__name__)


class ScalarState(Enum):
    """The three-state verdict for a governance scalar: grounded, deferred, or impossible.

    Member-and-value-identical to :class:`omni_mercury_engine.verifiers.three_state.ThreeState`
    (PR #244), so a serialised governance state round-trips against the verifier side over
    the same wire vocabulary.  It is kept as a separate symbol only because #243 and #244 are
    parallel un-merged drafts; once both land, unify to a single import.
    """

    #: Formula and inputs both present this run.  Registers a metric-only scalar.
    GROUNDED = "grounded"

    #: Decidable in principle and a real runtime signal exists in this engine, but the
    #: input was absent this run.  Registers nothing THIS run; re-running with the signal
    #: present grounds it.  This is a *kept* scalar.
    UNAVAILABLE = "unavailable"

    #: No oracle/formula exists, or no runtime signal for it can ever exist in this engine.
    #: Registers nothing, EVER.  This is a *dropped* scalar; the family must not be built.
    UNDECIDABLE = "undecidable"


class SignalClass(Enum):
    """Design-time vetting verdict for a whole governance *family* (made once, recorded).

    The verdict is a judgment about the engine's observable surface, not about any single
    execution: does a runtime signal that could ground this family plausibly exist anywhere
    (telemetry, request/log patterns, resource curves, security events, detector outputs,
    clinical inputs)?  It is the design-time axis that separates a *kept* UNAVAILABLE family
    from a *dropped* UNDECIDABLE one.
    """

    #: A runtime signal genuinely exists in this engine -- keep and build the family.  Its
    #: scalars are GROUNDED when the signal is present and UNAVAILABLE when it is absent.
    UNAVAILABLE_CAPABLE = "unavailable_capable"

    #: No runtime signal can ever exist here (or there is no formula) -- drop the family,
    #: do not build it.  Any scalar it would emit is UNDECIDABLE.
    UNDECIDABLE = "undecidable"

    #: A tier gate / tag by design (e.g. the EU AI Act risk tier): a real classification,
    #: but never expressed as a scalar and never registered into the operational surface.
    TAG_ONLY = "tag_only"


@dataclass(frozen=True)
class FamilyVet:
    """The recorded per-family signal vet: formula, signal, verdict, and one-line rationale.

    Attributes:
        family: Family key (matches :attr:`GovernanceScalar.family`).
        classification: The design-time :class:`SignalClass` verdict.
        standard: The published formula/oracle, cited.
        runtime_signal: The in-engine signal that would ground it (``file:line``) or
            ``"none"`` when no signal can exist.
        rationale: One line citing what signal would or would not exist.
    """

    family: str
    classification: SignalClass
    standard: str
    runtime_signal: str
    rationale: str


# ----------------------------------------------------------------------------------------
# The per-family signal vet -- the single machine-readable source of truth behind the PR
# body's vet table.  Every family the package considers appears here exactly once, kept
# or dropped on codebase evidence (file:line), never on a framework's reputation.
# ----------------------------------------------------------------------------------------
GOVERNANCE_FAMILY_VET: dict[str, FamilyVet] = {
    # --- Clinical / medical-device: real formulas, real input signal (kept). ---
    "sofa": FamilyVet(
        family="sofa",
        classification=SignalClass.UNAVAILABLE_CAPABLE,
        standard="SOFA -- Vincent et al., Intensive Care Med 1996;22:707-710",
        runtime_signal="medical/critical_care/sepsis_detector.py SOFACalculator; "
        "core/domain_feature_extractors.py:313-318 (clinical vitals/labs flow)",
        rationale="Clinical labs/vitals flow through the engine's patient_data surface; "
        "absent labs are UNAVAILABLE, never a healthy organ.",
    ),
    "ews": FamilyVet(
        family="ews",
        classification=SignalClass.UNAVAILABLE_CAPABLE,
        standard="NEWS2 -- Royal College of Physicians, 2017 (SpO2 Scale 1)",
        runtime_signal="core/domain_feature_extractors.py:311-318 (VITAL_RANGES: "
        "respiratory_rate, systolic_bp, temperature, ...)",
        rationale="The seven NEWS2 vitals are members of the engine's vital-sign surface; "
        "a missing vital abstains (UNAVAILABLE).",
    ),
    "mews": FamilyVet(
        family="mews",
        classification=SignalClass.UNAVAILABLE_CAPABLE,
        standard="MEWS -- Subbe et al., QJM 2001;94:521-526",
        runtime_signal="core/domain_feature_extractors.py:311-318 (same vital-sign surface "
        "as NEWS2)",
        rationale="MEWS reads the same vitals channel as NEWS2; the signal exists, a missing "
        "vital abstains (UNAVAILABLE).",
    ),
    "meld": FamilyVet(
        family="meld",
        classification=SignalClass.UNAVAILABLE_CAPABLE,
        standard="MELD-Na -- Kim et al., NEJM 2008; OPTN/UNOS allocation policy",
        runtime_signal="medical patient_data channel (bilirubin/creatinine flow via SOFA; "
        "INR/sodium are the same channel, not yet consumed elsewhere)",
        rationale="Two of four labs already flow (SOFA); INR/sodium are ordinary members of "
        "the same clinical channel, so the family abstains (UNAVAILABLE) until they appear.",
    ),
    "iso14971": FamilyVet(
        family="iso14971",
        classification=SignalClass.UNAVAILABLE_CAPABLE,
        standard="ISO 14971:2019 -- risk = f(severity, probability of harm)",
        runtime_signal="core/types.py:92 ThreatLevel; "
        "security/realtime_threat_detection.py:50 ThreatSignature.severity/confidence",
        rationale="The engine emits runtime severity/confidence coordinates; an absent "
        "severity/probability pair abstains (UNAVAILABLE).",
    ),
    # --- AI assurance: vetted on codebase evidence, not on checklist reputation. ---
    "nist_ai_rmf": FamilyVet(
        family="nist_ai_rmf",
        classification=SignalClass.UNAVAILABLE_CAPABLE,
        standard="NIST AI RMF 1.0 -- MEASURE function (quantitative trustworthiness)",
        runtime_signal="ml/drift.py:63 DriftResult; ml/bias_detection.py:49 "
        "FairnessResult.overall_score; evaluation/metrics.py:33 AnomalyMetrics.auc_roc",
        rationale="MEASURE maps to genuine runtime trustworthiness metrics (drift, "
        "fairness, performance); absent metrics this run abstain (UNAVAILABLE).",
    ),
    "mitre_atlas": FamilyVet(
        family="mitre_atlas",
        classification=SignalClass.UNAVAILABLE_CAPABLE,
        standard="MITRE ATLAS -- adversary tactic coverage over a security-adjacent surface",
        runtime_signal="security/threat_detection.py:135 ThreatDetector.detect_all "
        "(wired live at engine.py:2759); security/realtime_threat_detection.py:170",
        rationale="The engine observes attacks against its surface at runtime "
        "(threat_type/confidence); zero observed events abstains (UNAVAILABLE).",
    ),
    # --- Dropped: no runtime signal can ever exist here (UNDECIDABLE). ---
    "owasp_llm": FamilyVet(
        family="owasp_llm",
        classification=SignalClass.UNDECIDABLE,
        standard="OWASP Top 10 for LLM Applications (2025)",
        runtime_signal="none (models/foundation/llm_adapter.py:153 uses an LLM but ships no "
        "injection/output-handling/leakage detector; api rate-limit is generic; no token "
        "accounting)",
        rationale="The engine runs an LLM but emits no per-category mitigation signal; the "
        "only input is an operator checklist, which is not a runtime signal.",
    ),
    "imdrf_samd": FamilyVet(
        family="imdrf_samd",
        classification=SignalClass.UNDECIDABLE,
        standard="IMDRF SaMD risk categorization (N12, 2014)",
        runtime_signal="none (a design-time categorization of what the software *is*)",
        rationale="SaMD category is a static device classification, not a runtime "
        "measurement; no signal can ever exist.",
    ),
    "iso_42001": FamilyVet(
        family="iso_42001",
        classification=SignalClass.UNDECIDABLE,
        standard="ISO/IEC 42001:2023 -- AI management system",
        runtime_signal="none (management-system process conformance)",
        rationale="A management-system conformance checklist with no runtime signal.",
    ),
    "iso_23894": FamilyVet(
        family="iso_23894",
        classification=SignalClass.UNDECIDABLE,
        standard="ISO/IEC 23894:2023 -- AI risk management guidance",
        runtime_signal="none (process guidance)",
        rationale="Process guidance with no runtime signal; risk activities are not "
        "quantities the engine emits.",
    ),
    "nist_sp_1270": FamilyVet(
        family="nist_sp_1270",
        classification=SignalClass.UNDECIDABLE,
        standard="NIST SP 1270 -- managing bias in AI",
        runtime_signal="bias signal exists (ml/bias_detection.py:49) but is surfaced under "
        "nist_ai_rmf MEASURE; SP 1270 itself has no distinct scoring oracle",
        rationale="The bias runtime signal is folded into NIST AI RMF MEASURE; as a "
        "standalone conformance family SP 1270 is a checklist (UNDECIDABLE).",
    ),
    "ieee_7000": FamilyVet(
        family="ieee_7000",
        classification=SignalClass.UNDECIDABLE,
        standard="IEEE 7000-2021 series (ethics/transparency by design)",
        runtime_signal="none (design-process standards; leveled assessment)",
        rationale="Process/ethics standards yield a design-time leveled assessment, not a "
        "runtime score.",
    ),
    # --- A tier gate / tag by design, never a scalar. ---
    "eu_ai_act": FamilyVet(
        family="eu_ai_act",
        classification=SignalClass.TAG_ONLY,
        standard="EU AI Act (Reg. (EU) 2024/1689) -- risk tiers",
        runtime_signal="n/a (a declared-use-case tier tag, never a scalar)",
        rationale="Risk tier is a categorical gate on the declared use case; it registers "
        "nothing into the operational surface by design.",
    ),
}


@dataclass(frozen=True)
class GovernanceScalar:
    """A single descriptive governance measurement, or an honest abstention.

    Attributes:
        name: Metric-only scalar key (must match a ``_METRIC_ONLY_PREFIXES`` entry).
        family: Framework family this scalar belongs to (e.g. ``"sofa"``).
        state: One of :class:`ScalarState` (GROUNDED / UNAVAILABLE / UNDECIDABLE).
        value: Measurement in ``[0, 1]`` when GROUNDED, else ``None``.
        reason: Human-readable provenance or abstention reason.
        missing_inputs: For UNAVAILABLE, the input signals that were absent this run.
        provenance: JSON-friendly structured context (inputs used, formula identifier).
    """

    name: str
    family: str
    state: ScalarState
    value: float | None
    reason: str
    missing_inputs: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def is_grounded(self) -> bool:
        """Whether this scalar was computed from a present input signal (registers)."""
        return self.state is ScalarState.GROUNDED

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this scalar."""
        return {
            "name": self.name,
            "family": self.family,
            "state": self.state.value,
            "value": self.value,
            "reason": self.reason,
            "missing_inputs": list(self.missing_inputs),
            **self.provenance,
        }


def grounded(
    name: str,
    value: float,
    *,
    family: str,
    reason: str,
    provenance: dict[str, object] | None = None,
) -> GovernanceScalar:
    """Build a ``GROUNDED`` scalar, clamping ``value`` into ``[0, 1]``."""
    clamped = max(0.0, min(1.0, float(value)))
    return GovernanceScalar(
        name=name,
        family=family,
        state=ScalarState.GROUNDED,
        value=clamped,
        reason=reason,
        provenance=provenance or {},
    )


def unavailable(
    name: str,
    *,
    family: str,
    reason: str,
    missing_inputs: tuple[str, ...] = (),
    provenance: dict[str, object] | None = None,
) -> GovernanceScalar:
    """Build an ``UNAVAILABLE`` scalar: a kept abstention (signal exists, absent this run)."""
    return GovernanceScalar(
        name=name,
        family=family,
        state=ScalarState.UNAVAILABLE,
        value=None,
        reason=reason,
        missing_inputs=tuple(missing_inputs),
        provenance=provenance or {},
    )


def undecidable(
    name: str,
    *,
    family: str,
    reason: str,
    provenance: dict[str, object] | None = None,
) -> GovernanceScalar:
    """Build an ``UNDECIDABLE`` scalar: a dropped family's would-be scalar (never registers)."""
    return GovernanceScalar(
        name=name,
        family=family,
        state=ScalarState.UNDECIDABLE,
        value=None,
        reason=reason,
        provenance=provenance or {},
    )


@dataclass(frozen=True)
class GovernanceLedgerEntry:
    """Provenance record for one adjudicated governance scalar."""

    name: str
    family: str
    state: str
    value: float | None
    reason: str
    registered: bool
    missing_inputs: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for this ledger entry."""
        return {
            "name": self.name,
            "family": self.family,
            "state": self.state,
            "value": self.value,
            "reason": self.reason,
            "registered": self.registered,
            "missing_inputs": list(self.missing_inputs),
        }


class GovernanceRegistry:
    """Registers only *GROUNDED*, *metric-only* governance scalars into the GOSNN.

    The registry enforces the structural guarantee so the upgrade can never perturb the
    trained σ_Immutable gate:

    * **Abstention:** an UNAVAILABLE or UNDECIDABLE scalar registers nothing (ledger-only).
    * **Metric-only:** a GROUNDED scalar whose key is not recognised as metric-only by
      :meth:`GlobalOmniScalarNetwork._is_metric_only_scalar` is rejected with a
      :class:`ValueError` rather than silently inflating the operational vector.
    * **No UNDECIDABLE leakage:** a GROUNDED scalar from a family the design-time vet marks
      UNDECIDABLE is a contract violation and is refused -- a dropped family can never
      ground a value, even by mistake.
    """

    def __init__(self, gosnn: GlobalOmniScalarNetwork) -> None:
        """Bind the registry to a GOSNN singleton instance."""
        self.gosnn = gosnn
        self.ledger: list[GovernanceLedgerEntry] = []

    def register(
        self,
        scalar: GovernanceScalar,
        *,
        group: ScalarGroup,
        component_name: str,
    ) -> GovernanceLedgerEntry:
        """Register one scalar iff GROUNDED and metric-only; else record an abstention."""
        registered = False
        if scalar.is_grounded and scalar.value is not None:
            if not GlobalOmniScalarNetwork._is_metric_only_scalar(scalar.name):
                raise ValueError(
                    f"governance scalar {scalar.name!r} is not metric-only; registering it "
                    "would inflate the σ_Immutable operational vector"
                )
            vet = GOVERNANCE_FAMILY_VET.get(scalar.family)
            if vet is not None and vet.classification is SignalClass.UNDECIDABLE:
                raise ValueError(
                    f"family {scalar.family!r} is classified UNDECIDABLE; it must never "
                    "ground a scalar (no runtime signal can exist for it in this engine)"
                )
            self.gosnn.register_scalars(
                component_name=component_name,
                scalars={scalar.name: scalar.value},
                group=group,
                metadata={"governance_family": scalar.family, **scalar.provenance},
            )
            registered = True
        else:
            logger.info(
                "governance scalar %s %s (%s); nothing registered",
                scalar.name,
                scalar.state.value,
                scalar.reason,
            )
        entry = GovernanceLedgerEntry(
            name=scalar.name,
            family=scalar.family,
            state=scalar.state.value,
            value=scalar.value if registered else None,
            reason=scalar.reason,
            registered=registered,
            missing_inputs=scalar.missing_inputs,
        )
        self.ledger.append(entry)
        return entry

    def register_all(
        self,
        scalars: list[GovernanceScalar],
        *,
        group: ScalarGroup,
        component_name: str,
    ) -> list[GovernanceLedgerEntry]:
        """Register every GROUNDED scalar in ``scalars`` (abstentions stay ledger-only)."""
        return [self.register(s, group=group, component_name=component_name) for s in scalars]

    def summary(self) -> dict[str, object]:
        """Return registration/abstention counts and the current operational vector size."""
        by_state: dict[str, int] = {}
        by_family: dict[str, int] = {}
        for entry in self.ledger:
            by_state[entry.state] = by_state.get(entry.state, 0) + 1
            by_family[entry.family] = by_family.get(entry.family, 0) + 1
        return {
            "total": len(self.ledger),
            "registered": sum(1 for e in self.ledger if e.registered),
            "abstained": sum(1 for e in self.ledger if not e.registered),
            "by_state": by_state,
            "by_family": by_family,
            "operational_scalar_count": len(self.gosnn._collect_all_scalars()),
        }
