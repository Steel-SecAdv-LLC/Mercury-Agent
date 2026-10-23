# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sepsis screening on recorded ICU data, checked against the published tables.

Two recorded ICU stays from the open PhysioNet/CinC Challenge 2019 data (see
``tests/fixtures/medical/physionet_2019/README.md``) are scored row by row,
and every qSOFA and SOFA value the detector reports is recomputed here,
independently, from the published thresholds: qSOFA (Seymour et al. 2016)
and the SOFA table (Vincent et al. 1996).  The challenge's own
``SepsisLabel`` is not the reference: it leads onset by six hours and also
needs suspected infection, which these files do not record.

Also pinned: the input defects behind that mapping -- PhysioNet writes NaN
for "not measured", and a NaN passed straight to the detector used to fall
through the SOFA ladders to their worst band (an unmeasured platelet count
read as organ failure).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.adapters.medical import (
    SepsisRecordAdapter,
    physionet_2019_record,
)
from omni_mercury_engine.medical.critical_care.sepsis_detector import SepsisDetector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "medical" / "physionet_2019"
NAN = float("nan")


def _rows(name: str) -> list[dict[str, str]]:
    with (FIXTURES / f"{name}.psv").open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="|"))


def _value(row: dict[str, str], column: str) -> float | None:
    raw = row[column]
    return None if raw == "NaN" else float(raw)


def _band(value: float, ladder: tuple[tuple[float, int], ...], worst: int) -> int:
    for bound, points in ladder:
        if value < bound:
            return points
    return worst


def _expected(row: dict[str, str]) -> tuple[int, int | None]:
    """qSOFA (GCS unrecorded) and the SOFA total over the recorded organs."""
    resp, sbp = _value(row, "Resp"), _value(row, "SBP")
    qsofa = int(resp is not None and resp >= 22) + int(sbp is not None and sbp <= 100)
    organs: list[int] = []
    platelets = _value(row, "Platelets")  # 10^3/uL; lower is worse
    if platelets is not None:
        organs.append(_band(platelets, ((20.0, 4), (50.0, 3), (100.0, 2), (150.0, 1)), 0))
    bilirubin = _value(row, "Bilirubin_total")  # mg/dL
    if bilirubin is not None:
        organs.append(_band(bilirubin, ((1.2, 0), (2.0, 1), (6.0, 2), (12.0, 3)), 4))
    creatinine = _value(row, "Creatinine")  # mg/dL; no urine output recorded
    if creatinine is not None:
        organs.append(_band(creatinine, ((1.2, 0), (2.0, 1), (3.5, 2), (5.0, 3)), 4))
    mean_ap = _value(row, "MAP")  # mmHg; no vasopressor doses recorded
    if mean_ap is not None:
        organs.append(1 if mean_ap < 70 else 0)
    return qsofa, (sum(organs) if organs else None)


@pytest.fixture(scope="module")
def detector() -> SepsisDetector:
    return SepsisDetector(enable_ml_prediction=False)


class TestRecordedStays:
    @pytest.mark.parametrize("patient", ["p000001", "p000011"])
    def test_every_row_matches_the_published_tables(
        self, detector: SepsisDetector, patient: str
    ) -> None:
        graded = 0
        for row in _rows(patient):
            record = physionet_2019_record(row)
            if not record:
                continue  # an hour with none of the instruments' inputs measured
            graded += 1
            result = detector.detect_sepsis(record)
            qsofa, sofa = _expected(row)
            if "vital_signs" in record:
                assert result.qsofa_score == qsofa, row
            assert result.sofa_score == sofa, row
            if sofa is not None:
                assert result.sofa_is_lower_bound  # respiration and CNS are unrecorded
            assert result.sepsis_detected is (qsofa >= 2 or (sofa or 0) >= 2), row
            # A NaN in the file is "not measured": it is left out, never set aside.
            assert result.partial is False
        assert graded > 10

    def test_labelled_stay_below_threshold_is_not_flagged(self, detector: SepsisDetector) -> None:
        """p000011 carries SepsisLabel=1 from hour 24, yet no measured hour
        reaches qSOFA >= 2 or SOFA >= 2 (peak Resp 22 with SBP 146, platelets
        >= 184, creatinine <= 1.0).  The challenge sets the label six hours
        ahead of an onset that also requires suspected infection (Reyna et al.
        2020), neither of which the instruments see, so the screen must not
        fire: a positive here would be fabricated.
        """
        rows = _rows("p000011")
        assert any(row["SepsisLabel"] == "1" for row in rows)
        for row in rows:
            record = physionet_2019_record(row)
            if record:
                assert detector.detect_sepsis(record).sepsis_detected is False, row

    def test_adapter_serves_the_detector_row_by_row(self, detector: SepsisDetector) -> None:
        records = [r for r in map(physionet_2019_record, _rows("p000011")) if r]
        # The stay never screens positive; one renal SOFA-2 hour keeps the
        # comparison from passing on a constant output.
        records.append({"laboratory_values": {"creatinine_mg_dl": 2.4}})
        out = SepsisRecordAdapter().detect(records)
        direct = [detector.detect_sepsis(r).sepsis_detected for r in records]
        assert out["flags"].tolist() == direct
        assert direct[-1] is True
        assert np.all((out["scores"] >= 0.0) & (out["scores"] <= 1.0))
        assert "partial" not in out


class TestInputDefects:
    """Regressions for what recorded data exposed."""

    def test_unmeasured_labs_are_not_organ_failure(self, detector: SepsisDetector) -> None:
        """All-NaN labs scored SOFA 8, 'severe_sepsis', two fabricated failing organs."""
        labs: dict[str, Any] = {
            "platelets_k_ul": NAN,
            "bilirubin_mg_dl": NAN,
            "creatinine_mg_dl": NAN,
            "mean_arterial_pressure": 82.0,
        }
        result = detector.detect_sepsis({"laboratory_values": labs})
        assert result.sofa_score == 0
        assert result.organ_dysfunctions == []
        assert result.sepsis_detected is False
        assert {"coagulation", "liver", "renal"} <= set(result.unassessed_organs)
        assert result.partial is True
        assert len(result.set_aside) == 3

    def test_unmeasured_vitals_are_unassessed_not_normal(self, detector: SepsisDetector) -> None:
        result = detector.detect_sepsis(
            {"vital_signs": {"respiratory_rate_bpm": NAN, "systolic_bp_mmhg": 92.0}}
        )
        assert result.qsofa_score == 1
        assert "vital_signs.respiratory_rate_bpm" in result.safety.unassessed_inputs
        assert result.partial is True

    def test_unit_error_and_misspelt_key_are_set_aside(self, detector: SepsisDetector) -> None:
        result = detector.detect_sepsis(
            {
                "laboratory_values": {"platelets_k_ul": 150000.0, "creatinine_mg_dl": 2.4},
                "vital_signs": {"resp_rate": 30.0, "systolic_bp_mmhg": 95.0},
            }
        )
        where = [entry["where"] for entry in result.set_aside]
        assert "laboratory_values.platelets_k_ul" in where  # per uL, not 10^3/uL
        assert "vital_signs.resp_rate" in where
        assert result.sofa_score == 2  # creatinine 2.4 mg/dL alone
        assert result.sepsis_detected is True

    def test_empty_record_is_refused(self, detector: SepsisDetector) -> None:
        """An empty record used to return an unscored negative screen."""
        with pytest.raises(InvalidObservationError, match="nothing usable"):
            detector.detect_sepsis({})

    def test_nan_cell_in_a_physionet_row_is_left_out(self) -> None:
        row = {"Resp": "NaN", "SBP": "98", "MAP": "nan", "Platelets": "", "Creatinine": "1.4"}
        assert physionet_2019_record(row) == {
            "vital_signs": {"systolic_bp_mmhg": 98.0},
            "laboratory_values": {"creatinine_mg_dl": 1.4},
        }
        assert not math.isnan(physionet_2019_record(row)["laboratory_values"]["creatinine_mg_dl"])
