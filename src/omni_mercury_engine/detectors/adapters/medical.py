# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engine and registry adapters for the medical domain detectors.

``SepsisRecordAdapter`` serves ``SepsisDetector.detect_sepsis`` -- the
Sepsis-3 qSOFA and SOFA instruments -- through the shared record contract,
and :func:`physionet_2019_record` maps one hourly row of the open PhysioNet /
Computing in Cardiology Challenge 2019 data (CC BY 4.0) onto that record.
Decision support only: a flag is a possible-sepsis screen, not a diagnosis
(see ``SepsisDetector.decision_source``).

``CaseSurgeWindowAdapter`` serves ``CaseSurgeDetector.detect_case_surge``
(disease surveillance: a week-on-week rise in daily case counts) over one
window of daily counts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.hazard_adapter import (
    Assessment,
    HazardRecordAdapter,
    HazardWindowAdapter,
    nan_if_none,
)
from omni_mercury_engine.detectors.provenance import WrappedDecisionSource

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "PHYSIONET_2019_COLUMNS",
    "CaseSurgeWindowAdapter",
    "SepsisRecordAdapter",
    "physionet_2019_record",
]

#: PhysioNet Challenge 2019 column -> (section, detector key).  The units
#: match: Resp breaths/min, SBP and MAP mmHg, Platelets 10^3/uL, Bilirubin_total
#: and Creatinine mg/dL (challenge data description).  GCS, PaO2/FiO2, urine
#: output and vasopressor doses are not in the challenge files, so those
#: criteria and organs are reported unassessed, never scored as normal.
PHYSIONET_2019_COLUMNS: dict[str, tuple[str, str]] = {
    "Resp": ("vital_signs", "respiratory_rate_bpm"),
    "SBP": ("vital_signs", "systolic_bp_mmhg"),
    "MAP": ("laboratory_values", "mean_arterial_pressure"),
    "Platelets": ("laboratory_values", "platelets_k_ul"),
    "Bilirubin_total": ("laboratory_values", "bilirubin_mg_dl"),
    "Creatinine": ("laboratory_values", "creatinine_mg_dl"),
}


def _domain_refusal(error: ValueError) -> ValueError:
    """The detector's own refusal under the adapters' typed errors."""
    if isinstance(error, (InputModalityError, InvalidObservationError)):
        return error
    return InvalidObservationError(str(error))


def physionet_2019_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """One PhysioNet 2019 hourly row as a ``detect_sepsis`` record.

    The challenge files write ``NaN`` for a quantity not measured that hour;
    such a column is left out (unmeasured), not passed on as a value.

    Args:
        row: Column name -> value, as read from a ``p*.psv`` file.

    Returns:
        ``{"vital_signs": {...}, "laboratory_values": {...}}`` holding the
        measured columns only (a section with none is omitted).
    """
    record: dict[str, dict[str, float]] = {}
    for column, (section, key) in PHYSIONET_2019_COLUMNS.items():
        raw = row.get(column)
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "nan")):
            continue
        value = float(raw)
        if math.isnan(value):
            continue
        record.setdefault(section, {})[key] = value
    return record


class SepsisRecordAdapter(HazardRecordAdapter):
    """Possible-sepsis screening from vital signs and laboratory values.

    Each record is ``SepsisDetector.detect_sepsis``'s own input:
    ``vital_signs`` (qSOFA) and ``laboratory_values`` (SOFA).  The decision
    is the detector's ``sepsis_detected`` (qSOFA >= 2 or SOFA >= 2 with a
    zero baseline); the score is its ``risk_score``.  The trained
    progression model is not consulted here (it ships untrained and
    abstains), so ``temporal_sequence`` is not accepted.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.medical.critical_care.sepsis_detector:SepsisDetector"
    )

    ALLOWED_KEYS = frozenset({"vital_signs", "laboratory_values"})
    FEATURE_NAMES = (
        "risk_score",
        "sepsis_detected",
        "qsofa_score",
        "sofa_score",
        "sofa_is_lower_bound",
        "organ_dysfunction_count",
    )

    def _parse(self, record: Mapping[str, Any]) -> Any:
        return dict(record)

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.medical.critical_care.sepsis_detector import SepsisDetector

        if self._detector is None:
            self._detector = SepsisDetector(enable_ml_prediction=False)
        try:
            r = self._detector.detect_sepsis(unit)
        except ValueError as e:
            raise _domain_refusal(e) from e
        return Assessment(
            score=float(min(max(r.risk_score, 0.0), 1.0)),
            flag=bool(r.sepsis_detected),
            features=(
                float(r.risk_score),
                float(r.sepsis_detected),
                nan_if_none(r.qsofa_score),
                nan_if_none(r.sofa_score),
                float(r.sofa_is_lower_bound),
                float(len(r.organ_dysfunctions)),
            ),
            result=r,
        )


class CaseSurgeWindowAdapter(HazardWindowAdapter):
    """Disease-surveillance case surge over one window of daily counts.

    The window is ``[n, 1]`` over ``daily_cases`` (new cases per day, oldest
    first); NaN marks an unreported day.  ``config`` may set
    ``serial_interval_days`` (default 5).  ``CaseSurgeDetector`` compares
    the mean of the latest 7 days with the 7 before (surge above 1.5x, an
    experimental ratio; see its ``decision_source``), so a window needs 14
    usable days, and a window whose gap-free run is shorter is refused.
    The decision is the detector's ``surge_detected``; the score is the
    week-on-week ratio ``q`` as ``q / (q + 1.5)``, monotone and exactly 1/2
    at the surge threshold, so the decision is score > 1/2.
    """

    decision_source = WrappedDecisionSource(
        "omni_mercury_engine.medical.pandemic.pandemic_detector:CaseSurgeDetector"
    )

    COLUMNS = ("daily_cases",)
    OPTIONAL_COLUMNS = frozenset({"daily_cases"})
    MIN_SAMPLES = 14
    FEATURE_NAMES = (
        "week_on_week_ratio",
        "re_estimate",
        "doubling_time_days",
        "surge_detected",
    )

    def _assess(self, unit: Any) -> Assessment:
        from omni_mercury_engine.medical.pandemic.pandemic_detector import (
            SURGE_RATIO,
            CaseSurgeDetector,
        )

        if self._detector is None:
            self._detector = CaseSurgeDetector()
        case_data: dict[str, Any] = {"daily_cases": self.column(unit, "daily_cases")}
        if "serial_interval_days" in self._config:
            case_data["serial_interval_days"] = self._config["serial_interval_days"]
        try:
            r = self._detector.detect_case_surge(case_data)
        except ValueError as e:
            raise _domain_refusal(e) from e
        if r["re_method"] == "insufficient_history":
            reasons = "; ".join(entry["reason"] for entry in r.get("set_aside", []))
            raise InvalidObservationError(f"{self._name}: nothing assessable ({reasons})")
        ratio = r["week_on_week_ratio"]
        if ratio is None:
            score = 0.0  # no case in either week
        elif math.isinf(ratio):
            score = 1.0  # emergence from a zero baseline
        else:
            score = ratio / (ratio + SURGE_RATIO)
        return Assessment(
            score=float(score),
            flag=bool(r["surge_detected"]),
            features=(
                nan_if_none(ratio),
                nan_if_none(r["re_estimate"]),
                nan_if_none(r["doubling_time_days"]),
                float(r["surge_detected"]),
            ),
            result=r,
            # The detector's result is a dict: carry its gaps explicitly.
            set_aside=tuple(r.get("set_aside", ())),
        )
