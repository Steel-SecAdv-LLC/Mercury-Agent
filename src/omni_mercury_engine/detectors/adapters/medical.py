# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engine and registry adapters for the medical domain detectors.

``SepsisRecordAdapter`` serves ``SepsisDetector.detect_sepsis`` -- the
Sepsis-3 qSOFA and SOFA instruments -- through the shared record contract,
and :func:`physionet_2019_record` maps one hourly row of the open PhysioNet /
Computing in Cardiology Challenge 2019 data (CC BY 4.0) onto that record.
Decision support only: a flag is a possible-sepsis screen, not a diagnosis
(see ``SepsisDetector.decision_source``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.hazard_adapter import (
    Assessment,
    HazardRecordAdapter,
    nan_if_none,
)
from omni_mercury_engine.detectors.provenance import WrappedDecisionSource

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["PHYSIONET_2019_COLUMNS", "SepsisRecordAdapter", "physionet_2019_record"]

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
