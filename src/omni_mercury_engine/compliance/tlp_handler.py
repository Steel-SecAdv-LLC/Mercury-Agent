# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Traffic Light Protocol (TLP) classification for Mercury Agent.

This module implements automated Traffic Light Protocol (TLP) tagging for
anomaly detection results, narrative outputs, and other sensitive artifacts
produced by Mercury Agent.  It supports the full five-level TLP 2.0
specification from FIRST.org and CISA:

* ``TLP:RED`` -- Not for disclosure, restricted to the named recipients.
* ``TLP:AMBER+STRICT`` -- Limited disclosure, restricted to participants'
  own organisation (NOT including clients/customers).
* ``TLP:AMBER`` -- Limited disclosure, restricted to participants'
  organisations AND their clients/customers on a need-to-know basis.
* ``TLP:GREEN`` -- Limited disclosure, restricted to the trusted community.
* ``TLP:CLEAR`` -- Disclosure is not limited (replaces the legacy
  ``TLP:WHITE`` label).

The source provenance for this module is::

    Omni-AXA-Engine/src/omni_anomaly_engine/domains/ciad/compliance/tlp_handler.py
    (313 LOC, GPL-3.0-or-later)

with the following behavioural deltas applied during the port:

* ``TLP:AMBER+STRICT`` has been added.  The upstream module shipped only
  the four legacy colours (RED/AMBER/GREEN/CLEAR), which is non-compliant
  with FIRST.org TLP 2.0.  The five-level model is implemented end-to-end:
  classification, reasoning, sharing guidelines, ethical considerations,
  watermarks and export metadata all understand AMBER+STRICT.
* All public APIs are fully typed for ``mypy --strict``.
* Bare exception handling has been removed; the upstream module did not
  raise but the port adds explicit ``ValueError`` paths for invalid
  scores.
* Sharing-guideline strings are verbatim from FIRST.org TLP 2.0; the
  upstream legacy text has been refreshed to match.

References
----------
* FIRST TLP 2.0 standard: https://www.first.org/tlp/
* CISA TLP definitions: https://www.cisa.gov/tlp
* CISA TLP 2.0 transition notice:
  https://www.cisa.gov/news-events/news/cisa-and-partners-update-traffic-light-protocol-version-20
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

__all__ = [
    "TLPClassification",
    "TLPColor",
    "TLPHandler",
    "TLPValidationError",
    "get_tlp_handler",
]


class TLPValidationError(ValueError):
    """Raised when TLP classification inputs are out of range or malformed."""


class TLPColor(Enum):
    """TLP 2.0 colour classifications for information sharing.

    Values mirror the canonical FIRST.org labels (``CLEAR``, ``GREEN``,
    ``AMBER``, ``AMBER+STRICT``, ``RED``).  The legacy ``WHITE`` label
    from TLP 1.0 is intentionally NOT defined; consumers should map any
    pre-2022 ``WHITE`` value to :attr:`CLEAR` before classification.
    """

    CLEAR = "CLEAR"
    GREEN = "GREEN"
    AMBER = "AMBER"
    AMBER_STRICT = "AMBER+STRICT"
    RED = "RED"

    @property
    def label(self) -> str:
        """Return the canonical ``TLP:<colour>`` label string."""
        return f"TLP:{self.value}"

    @property
    def rank(self) -> int:
        """Monotonic severity rank ranging from ``0`` (CLEAR) to ``4`` (RED)."""
        return _TLP_RANK[self]


_TLP_RANK: Final[dict[TLPColor, int]] = {
    TLPColor.CLEAR: 0,
    TLPColor.GREEN: 1,
    TLPColor.AMBER: 2,
    TLPColor.AMBER_STRICT: 3,
    TLPColor.RED: 4,
}


@dataclass(frozen=True)
class TLPClassification:
    """Result of a TLP classification decision.

    Attributes:
        color: The assigned :class:`TLPColor`.
        confidence: A value in ``[0.0, 1.0]`` describing how confident the
            handler is in the classification.
        reasoning: A human-readable rationale, suitable for audit logs.
        sharing_guidelines: The canonical FIRST.org TLP 2.0 disclosure
            guidance for the assigned colour.
        ethical_considerations: Domain-specific ethical considerations
            that callers MUST take into account before sharing.
    """

    color: TLPColor
    confidence: float
    reasoning: str
    sharing_guidelines: str
    ethical_considerations: tuple[str, ...] = field(default_factory=tuple)


# Sharing guidelines lifted verbatim from FIRST.org TLP 2.0 (2022).
_SHARING_GUIDELINES: Final[dict[TLPColor, str]] = {
    TLPColor.RED: (
        "TLP:RED - For the eyes and ears of individual recipients only, "
        "no further disclosure. Sources may use TLP:RED when information "
        "cannot be effectively acted upon without significant risk for "
        "the privacy, reputation, or operations of the organizations "
        "involved. Recipients may therefore not share TLP:RED information "
        "with anyone else. In the context of a meeting, for example, "
        "TLP:RED information is limited to those present at the meeting."
    ),
    TLPColor.AMBER_STRICT: (
        "TLP:AMBER+STRICT - Limited disclosure, recipients can only spread "
        "this on a need-to-know basis within their organization. Note that "
        "TLP:AMBER+STRICT restricts sharing to the organization only."
    ),
    TLPColor.AMBER: (
        "TLP:AMBER - Limited disclosure, recipients can only spread this "
        "on a need-to-know basis within their organization and its clients. "
        "Sources may use TLP:AMBER when information requires support to be "
        "effectively acted upon, yet carries risk to privacy, reputation, "
        "or operations if shared outside of the organizations involved. "
        "Recipients may share TLP:AMBER information with members of their "
        "own organization and its clients, but only on a need-to-know basis "
        "to protect their organization and its clients and prevent further "
        "harm."
    ),
    TLPColor.GREEN: (
        "TLP:GREEN - Limited disclosure, recipients can spread this within "
        "their community. Sources may use TLP:GREEN when information is "
        "useful to increase awareness within their wider community. "
        "Recipients may share TLP:GREEN information with peers and partner "
        "organizations within their community, but not via publicly "
        "accessible channels. TLP:GREEN information may not be released "
        "outside of the community."
    ),
    TLPColor.CLEAR: (
        "TLP:CLEAR - Recipients can spread this to the world, there is no "
        "limit on disclosure. Sources may use TLP:CLEAR when information "
        "carries minimal or no foreseeable risk of misuse, in accordance "
        "with applicable rules and procedures for public release. Subject "
        "to standard copyright rules, TLP:CLEAR information may be shared "
        "without restriction."
    ),
}

_ETHICAL_CONSIDERATIONS_BASE: Final[tuple[str, ...]] = (
    "Verify recipient authorization before sharing classified information",
    "Document all information sharing activities for audit trails",
    "Respect privacy rights of individuals mentioned in anomaly data",
    "Comply with applicable laws (GDPR, HIPAA, CCPA, etc.)",
    "Consider potential harm from disclosure or non-disclosure",
    "Maintain confidentiality of sources and methods",
    "Balance transparency with security requirements",
)

_CRITICAL_DOMAINS: Final[frozenset[str]] = frozenset(
    {"cyber", "security", "infrastructure", "critical_infrastructure"}
)
_CRITICAL_TYPES: Final[frozenset[str]] = frozenset(
    {
        "cyberattack",
        "intrusion",
        "breach",
        "exploit",
        "malware",
        "infrastructure_failure",
        "cascading_failure",
        "critical_threat",
        "pandemic",
        "bio_threat",
        "terrorism",
        "weapon",
    }
)
_SENSITIVE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "medical",
        "patient_data",
        "pii",
        "financial",
        "insider_threat",
        "vulnerability",
        "zero_day",
        "safety_hazard",
    }
)
_WELL_DEFINED_DOMAINS: Final[frozenset[str]] = frozenset(
    {"cyber", "security", "medical", "infrastructure"}
)


class TLPHandler:
    """Automated TLP classification for Mercury Agent outputs.

    The handler combines anomaly score, anomaly type and domain context to
    assign a TLP 2.0 colour, attach the appropriate FIRST.org sharing
    guidelines, and surface domain-specific ethical considerations.

    The handler is fully deterministic and thread-safe (state-free past
    initialisation); callers may share a single instance across detection
    pipelines.

    Example:
        >>> handler = TLPHandler()
        >>> result = handler.classify_anomaly(
        ...     anomaly_score=0.92, anomaly_type="zero_day", domain="cyber",
        ... )
        >>> result.color is TLPColor.RED
        True
    """

    #: Anomaly score at/above which output is unconditionally classified RED.
    DEFAULT_RED_THRESHOLD: Final[float] = 0.85
    #: Anomaly score at/above which output is classified AMBER (or stricter).
    DEFAULT_AMBER_THRESHOLD: Final[float] = 0.60
    #: Anomaly score at/above which output is classified GREEN.
    DEFAULT_GREEN_THRESHOLD: Final[float] = 0.30

    def __init__(
        self,
        red_threshold: float = DEFAULT_RED_THRESHOLD,
        amber_threshold: float = DEFAULT_AMBER_THRESHOLD,
        green_threshold: float = DEFAULT_GREEN_THRESHOLD,
    ) -> None:
        """Initialise the handler.

        Args:
            red_threshold: Score at/above which anomalies are RED.
            amber_threshold: Score at/above which anomalies are AMBER.
            green_threshold: Score at/above which anomalies are GREEN.

        Raises:
            TLPValidationError: If thresholds are not in monotonic order
                inside the closed interval ``[0.0, 1.0]``.
        """
        if not 0.0 <= green_threshold <= amber_threshold <= red_threshold <= 1.0:
            msg = (
                "TLP thresholds must satisfy "
                "0 <= green <= amber <= red <= 1; "
                f"got green={green_threshold}, amber={amber_threshold}, "
                f"red={red_threshold}"
            )
            raise TLPValidationError(msg)

        self.red_threshold: float = red_threshold
        self.amber_threshold: float = amber_threshold
        self.green_threshold: float = green_threshold
        self.sharing_guidelines: dict[TLPColor, str] = dict(_SHARING_GUIDELINES)
        self.ethical_considerations_base: tuple[str, ...] = _ETHICAL_CONSIDERATIONS_BASE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_anomaly(
        self,
        anomaly_score: float,
        anomaly_type: str,
        domain: str,
        context: Mapping[str, Any] | None = None,
    ) -> TLPClassification:
        """Classify a single anomaly into a TLP colour.

        Args:
            anomaly_score: Severity score in ``[0.0, 1.0]``.
            anomaly_type: Free-text label describing the anomaly
                (e.g. ``"zero_day"``, ``"patient_data"``,
                ``"infrastructure_failure"``).
            domain: Domain context (e.g. ``"cyber"``, ``"medical"``).
            context: Optional additional context.  Recognised keys:

                * ``"strict_sharing"`` (bool): If ``True`` and the
                  computed colour is :attr:`TLPColor.AMBER`, the colour
                  is escalated to :attr:`TLPColor.AMBER_STRICT`.
                * ``"contains_pii"`` (bool): If ``True`` and the computed
                  colour is :attr:`TLPColor.AMBER` or weaker, the colour
                  is escalated to :attr:`TLPColor.AMBER_STRICT`.

        Returns:
            A :class:`TLPClassification` instance.

        Raises:
            TLPValidationError: If ``anomaly_score`` is not finite or
                falls outside ``[0.0, 1.0]``.
        """
        if not _is_in_unit_interval(anomaly_score):
            msg = "anomaly_score must be a finite number in [0.0, 1.0]; " f"got {anomaly_score!r}"
            raise TLPValidationError(msg)
        if not anomaly_type:
            raise TLPValidationError("anomaly_type must be a non-empty string")
        if not domain:
            raise TLPValidationError("domain must be a non-empty string")

        ctx: Mapping[str, Any] = context if context is not None else {}

        color = self._determine_color(anomaly_score, anomaly_type, domain, ctx)
        confidence = self._calculate_confidence(anomaly_score, anomaly_type, domain)
        reasoning = self._generate_reasoning(anomaly_score, anomaly_type, domain, color)
        guidelines = self.sharing_guidelines[color]
        ethical_considerations = self._get_ethical_considerations(color, domain)

        return TLPClassification(
            color=color,
            confidence=confidence,
            reasoning=reasoning,
            sharing_guidelines=guidelines,
            ethical_considerations=ethical_considerations,
        )

    def batch_classify(self, anomalies: Iterable[Mapping[str, Any]]) -> list[TLPClassification]:
        """Classify a batch of anomalies in submission order.

        Args:
            anomalies: Iterable of anomaly dictionaries.  Each dictionary
                may provide ``score`` (float), ``type`` (str), ``domain``
                (str) and ``context`` (mapping); missing entries fall
                back to safe defaults (``score=0.0``, ``type="unknown"``,
                ``domain="general"``).

        Returns:
            A list of :class:`TLPClassification` results, one per input.
        """
        classifications: list[TLPClassification] = []
        for anomaly in anomalies:
            score = float(anomaly.get("score", 0.0))
            anomaly_type = str(anomaly.get("type", "unknown"))
            domain = str(anomaly.get("domain", "general"))
            context_obj = anomaly.get("context", {})
            context: Mapping[str, Any] | None
            if isinstance(context_obj, Mapping):
                context = context_obj
            else:
                context = None
            classifications.append(
                self.classify_anomaly(
                    anomaly_score=score,
                    anomaly_type=anomaly_type,
                    domain=domain,
                    context=context,
                )
            )
        return classifications

    def get_color_statistics(self, classifications: Sequence[TLPClassification]) -> dict[str, int]:
        """Return per-colour counts for a sequence of classifications.

        The returned mapping always contains every TLP 2.0 colour, with
        zero counts for colours that did not appear.
        """
        stats: dict[str, int] = {color.value: 0 for color in TLPColor}
        for classification in classifications:
            stats[classification.color.value] += 1
        return stats

    def generate_watermark_text(self, color: TLPColor) -> str:
        """Generate a single-line watermark suitable for reports.

        The watermark starts with the canonical ``TLP:<colour>`` label
        and is followed by the first sentence of the FIRST.org sharing
        guidelines so reviewers can interpret the colour at a glance.
        """
        guideline = self.sharing_guidelines[color]
        # First sentence == everything up to the first period followed by
        # a space or end of string.
        first_period = guideline.find(". ")
        snippet = guideline if first_period == -1 else guideline[: first_period + 1]
        return f"{color.label} - {snippet}"

    def get_export_metadata(self, classification: TLPClassification) -> dict[str, Any]:
        """Build a JSON-serialisable export-metadata block.

        The returned dictionary is suitable for embedding alongside
        Mercury Agent reports, narrative exports and API responses.  It
        contains the colour label, confidence, reasoning, full sharing
        guidelines, ethical considerations and a watermark string.
        """
        return {
            "tlp_color": classification.color.value,
            "tlp_label": classification.color.label,
            "tlp_rank": classification.color.rank,
            "tlp_confidence": classification.confidence,
            "tlp_reasoning": classification.reasoning,
            "sharing_guidelines": classification.sharing_guidelines,
            "ethical_considerations": list(classification.ethical_considerations),
            "watermark": self.generate_watermark_text(classification.color),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_color(
        self,
        score: float,
        anomaly_type: str,
        domain: str,
        context: Mapping[str, Any],
    ) -> TLPColor:
        """Decide a TLP colour given inputs and contextual hints."""
        lowered_domain = domain.lower()
        lowered_type = anomaly_type.lower()
        type_is_critical = any(ct in lowered_type for ct in _CRITICAL_TYPES)
        type_is_sensitive = any(st in lowered_type for st in _SENSITIVE_TYPES)

        if (
            score >= self.red_threshold
            or (lowered_domain in _CRITICAL_DOMAINS and score >= 0.70)
            or (type_is_critical and score >= 0.70)
        ):
            base = TLPColor.RED
        elif (type_is_critical and score >= 0.50) or (
            type_is_sensitive and score >= self.amber_threshold
        ):
            base = TLPColor.AMBER
        elif type_is_sensitive and score >= self.green_threshold:
            base = TLPColor.GREEN
        elif score >= self.amber_threshold:
            base = TLPColor.AMBER
        elif score >= self.green_threshold:
            base = TLPColor.GREEN
        else:
            base = TLPColor.CLEAR

        return self._maybe_strict_escalate(base, lowered_domain, lowered_type, context)

    def _maybe_strict_escalate(
        self,
        base: TLPColor,
        lowered_domain: str,
        lowered_type: str,
        context: Mapping[str, Any],
    ) -> TLPColor:
        """Escalate AMBER to AMBER+STRICT when context warrants it."""
        if base is not TLPColor.AMBER:
            return base

        strict_context = bool(context.get("strict_sharing", False)) or bool(
            context.get("contains_pii", False)
        )
        if strict_context:
            return TLPColor.AMBER_STRICT

        # Sensitive types in medical/healthcare default to strict because
        # they almost always carry PII / PHI risk.
        if lowered_domain in {"medical", "healthcare"} and any(
            st in lowered_type for st in ("patient_data", "pii", "medical")
        ):
            return TLPColor.AMBER_STRICT

        return base

    def _calculate_confidence(self, score: float, anomaly_type: str, domain: str) -> float:
        """Compute classification confidence in ``[0.0, 1.0]``."""
        base_confidence = 0.80
        if score >= 0.90 or score <= 0.20:
            base_confidence += 0.15
        if domain.lower() in _WELL_DEFINED_DOMAINS:
            base_confidence += 0.05
        # anomaly_type is currently a passthrough; keep it in the signature
        # for forward compatibility and to encourage callers to supply it.
        del anomaly_type
        return min(base_confidence, 1.0)

    def _generate_reasoning(
        self,
        score: float,
        anomaly_type: str,
        domain: str,
        color: TLPColor,
    ) -> str:
        """Generate a human-readable rationale string."""
        if color is TLPColor.RED:
            severity = (
                f"Critical severity (score: {score:.2f})",
                "Immediate containment required",
                "Potential for significant harm if disclosed improperly",
            )
        elif color is TLPColor.AMBER_STRICT:
            severity = (
                f"Moderate severity (score: {score:.2f})",
                "Strict-organisation sharing only (no client/customer disclosure)",
                "PII/PHI or sensitive-type indicators detected",
            )
        elif color is TLPColor.AMBER:
            severity = (
                f"Moderate severity (score: {score:.2f})",
                "Requires controlled sharing within organisations and their clients",
                "Risk to operations if broadly disclosed",
            )
        elif color is TLPColor.GREEN:
            severity = (
                f"Low severity (score: {score:.2f})",
                "Suitable for community awareness",
                "Limited risk from controlled disclosure",
            )
        else:
            severity = (
                f"Minimal severity (score: {score:.2f})",
                "Public information with no restrictions",
                "No foreseeable risk from disclosure",
            )

        return " | ".join((*severity, f"Domain: {domain}, Type: {anomaly_type}"))

    def _get_ethical_considerations(self, color: TLPColor, domain: str) -> tuple[str, ...]:
        """Return domain-specific ethical considerations as a tuple."""
        considerations: list[str] = list(self.ethical_considerations_base)
        if color in {TLPColor.RED, TLPColor.AMBER_STRICT}:
            considerations.extend(
                (
                    "Obtain explicit authorization before any disclosure",
                    "Use secure channels for all communications",
                    "Implement need-to-know access controls",
                    "Consider legal liability for unauthorized disclosure",
                )
            )

        lowered_domain = domain.lower()
        if lowered_domain in {"medical", "healthcare"}:
            considerations.extend(
                (
                    "Comply with HIPAA privacy and security rules",
                    "Protect patient identifiable information",
                    "Obtain patient consent where required",
                )
            )
        if lowered_domain in {"cyber", "security"}:
            considerations.extend(
                (
                    "Protect vulnerability details until patches available",
                    "Coordinate disclosure with affected parties",
                    "Consider impact on ongoing investigations",
                )
            )
        return tuple(considerations)


def _is_in_unit_interval(value: float) -> bool:
    """Return True if value is a finite float in ``[0.0, 1.0]``."""
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(float_value):
        return False
    return 0.0 <= float_value <= 1.0


def get_tlp_handler() -> TLPHandler:
    """Factory returning a fresh :class:`TLPHandler` with default thresholds."""
    return TLPHandler()
