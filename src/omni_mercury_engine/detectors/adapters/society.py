# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engine and registry adapters for the economic, energy-grid and payload-threat detectors.

Each adapter serves one domain detector through ``fit`` / ``detect`` /
``extract_features`` / ``predict`` over that detector's true input and maps
it one-to-one onto the detector's own entry point; scores and decisions are
the detector's own, or its own ladder over the ladder's top (named per
adapter).  All three are defensive: they grade what they are given and act
on nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.hazard_adapter import (
    Assessment,
    HazardRecordAdapter,
    HazardTextAdapter,
    nan_if_none,
    ordinal_score,
)
from omni_mercury_engine.detectors.provenance import WrappedDecisionSource

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["EMPRecordAdapter", "FinancialCrisisRecordAdapter", "ThreatPayloadAdapter"]


def _domain_refusal(error: ValueError) -> ValueError:
    """The detector's own refusal under the adapters' typed errors."""
    if isinstance(error, (InputModalityError, InvalidObservationError)):
        return error
    return InvalidObservationError(str(error))


class ThreatPayloadAdapter(HazardTextAdapter):
    """Injection payloads (SQL injection, XSS, path traversal) in request text.

    Each payload is graded by ``ThreatDetector.detect_all`` (linear time).
    The decision is the detector's own ``is_threat`` (any category's pattern
    matched); the score is the highest category ``confidence`` -- the
    detector's own fraction of that category's patterns matched -- so the
    decision is exactly score > 0.  A payload above ``MAX_CHARS`` is graded
    on its first ``MAX_CHARS`` characters and the verdict is marked partial,
    naming the characters set aside.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.security.threat_detection:ThreatDetector"
    )

    FEATURE_NAMES = (
        "max_confidence",
        "threat_count",
        "sql_injection",
        "xss",
        "path_traversal",
    )
    _CATEGORIES = ("sql_injection", "xss", "path_traversal")

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.security.threat_detection import ThreatDetector

        if self._detector is None:
            self._detector = ThreatDetector()
        result = self._detector.detect_all(unit)
        confidence = {t["threat_type"]: float(t["confidence"]) for t in result["threats"]}
        return Assessment(
            score=max(confidence.values(), default=0.0),
            flag=bool(result["is_threat"]),
            features=(
                max(confidence.values(), default=0.0),
                float(result["threat_count"]),
                *(confidence.get(c, 0.0) for c in self._CATEGORIES),
            ),
            result=result,
        )


class FinancialCrisisRecordAdapter(HazardRecordAdapter):
    """Financial crisis risk from market, banking, network and currency sections.

    Each record is ``FinancialCrisisDetector.predict_financial_crisis``'s own
    input: ``market_data``, ``banking_data``, ``network_data`` (systemic
    interconnection), ``currency_data``, ``trading_data`` (graded once fraud
    weights load) and ``sector_data`` (sector labels only).  The decision is
    the detector's own ``crisis_imminent`` (two or more crisis indicators);
    the score is its severity ladder stable < stress < crisis <
    severe_crisis < systemic_collapse over 4, so the decision is exactly
    score >= 1/2.  The ``financial_stress`` entry serves the market and
    banking sections as named rows; this adapter serves the whole record.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.detectors.economic.financial_crisis_detector:FinancialCrisisDetector"
    )

    ALLOWED_KEYS = frozenset(
        {
            "market_data",
            "banking_data",
            "network_data",
            "currency_data",
            "trading_data",
            "sector_data",
        }
    )
    FEATURE_NAMES = (
        "severity_level",
        "confidence",
        "market_crash_detected",
        "banking_stress",
        "currency_instability",
        "systemic_risk_score",
        "vix_level",
        "credit_default_swap_spread",
    )
    _SEVERITY = ("stable", "stress", "crisis", "severe_crisis", "systemic_collapse")

    def _parse(self, record: Mapping[str, Any]) -> Any:
        return dict(record)

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.economic.financial_crisis_detector import (
            FinancialCrisisDetector,
        )

        if self._detector is None:
            self._detector = FinancialCrisisDetector()
        try:
            r = self._detector.predict_financial_crisis(unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        systemic = "network_data" in unit
        return Assessment(
            score=ordinal_score(r.severity_level, self._SEVERITY),
            flag=bool(r.crisis_imminent),
            features=(
                float(self._SEVERITY.index(r.severity_level)),
                float(r.confidence),
                float(r.market_crash_detected),
                float(r.banking_stress),
                float(r.currency_instability),
                # Unscored without network data: NaN, not the dataclass 0.0.
                float(r.systemic_risk_score) if systemic else float("nan"),
                nan_if_none(r.vix_level),
                nan_if_none(r.credit_default_swap_spread),
            ),
            result=r,
        )


class EMPRecordAdapter(HazardRecordAdapter):
    """Electromagnetic pulse and geomagnetic threat from field and magnetometer records.

    Each record is ``EMPDetector.predict_emp``'s own input: ``sensor_data``
    (an E1 field waveform or its summary values, and E2 values),
    ``magnetometer_data`` (dB/dt for E3 and grid impact), ``signature_data``
    and ``sensor_array`` (TDOA localization).  The score is the detector's
    threat ladder benign < anomalous < suspicious < threat < critical over
    4.  The decision is that ladder's THREAT level, the one the detector
    assigns a detected EMP or a confirmed intentional-EMI attack (and the
    floor a critical grid impact sets); ``emp_detected`` -- EMP proper, false
    for a geomagnetic storm or an IEMI attack -- is a feature.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.detectors.energy.emp_detector:EMPDetector"
    )

    ALLOWED_KEYS = frozenset({"sensor_data", "magnetometer_data", "signature_data", "sensor_array"})
    FEATURE_NAMES = (
        "threat_level",
        "emp_detected",
        "e1_component_detected",
        "e3_component_detected",
        "field_strength_vm",
        "peak_dbdt_nt_per_s",
        "intentional_attack_probability",
    )
    _LADDER = ("benign", "anomalous", "suspicious", "threat", "critical")

    def _parse(self, record: Mapping[str, Any]) -> Any:
        return dict(record)

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.detectors.energy.emp_detector import EMPDetector

        if self._detector is None:
            self._detector = EMPDetector()
        try:
            r = self._detector.predict_emp(unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        level = self._LADDER.index(r.threat_level)
        return Assessment(
            score=ordinal_score(r.threat_level, self._LADDER),
            flag=level >= self._LADDER.index("threat"),
            features=(
                float(level),
                float(r.emp_detected),
                float(r.e1_component_detected),
                float(r.e3_component_detected),
                nan_if_none(r.field_strength_vm),
                nan_if_none(r.peak_dbdt_nt_per_s),
                nan_if_none(r.intentional_attack_probability),
            ),
            result=r,
        )
