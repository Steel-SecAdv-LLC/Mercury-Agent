# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Payload-threat, financial-crisis and EMP adapters serve their detector's own verdict.

Every adapter is checked against a direct call to the detector it wraps (the
adapter adds no judgement of its own), on threatening and quiet input, and
must refuse malformed input instead of grading it.  The EMP adapter is also
run on the recorded USGS Boulder magnetometer of the May 2024 (Gannon)
storm: a real geomagnetic storm is a grid threat, not an EMP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.adapters import (
    EMPRecordAdapter,
    FinancialCrisisRecordAdapter,
    ThreatPayloadAdapter,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(
    "cls", [EMPRecordAdapter, FinancialCrisisRecordAdapter, ThreatPayloadAdapter]
)
def test_every_adapter_is_a_base_detector(cls: Any) -> None:
    assert issubclass(cls, BaseDetector)
    for method in ("fit", "detect", "extract_features", "predict"):
        assert callable(getattr(cls, method))


class TestThreatPayload:
    PAYLOADS = [
        "GET /index.html",
        "id=1' OR '1'='1",
        "<script>alert(1)</script>",
        "../../etc/passwd",
    ]

    def test_matches_the_detector(self) -> None:
        from omni_mercury_engine.security.threat_detection import ThreatDetector

        out = ThreatPayloadAdapter().detect(self.PAYLOADS)
        detector = ThreatDetector()
        for payload, score, flag in zip(self.PAYLOADS, out["scores"], out["flags"], strict=True):
            direct = detector.detect_all(payload)
            assert flag == direct["is_threat"]
            assert score == pytest.approx(
                max((t["confidence"] for t in direct["threats"]), default=0.0)
            )
        assert out["flags"].tolist() == [False, True, True, True]

    def test_decision_is_exactly_a_positive_score(self) -> None:
        out = ThreatPayloadAdapter().detect(self.PAYLOADS)
        assert out["flags"].tolist() == (out["scores"] > 0).tolist()

    def test_features_name_the_category(self) -> None:
        features = ThreatPayloadAdapter().extract_features(["../../etc/passwd"])[0]
        named = dict(zip(ThreatPayloadAdapter.FEATURE_NAMES, features.tolist(), strict=True))
        assert named["path_traversal"] > 0.0
        assert named["sql_injection"] == named["xss"] == 0.0
        assert named["threat_count"] == 1.0

    def test_oversized_payload_is_graded_up_to_the_limit_and_marked_partial(self) -> None:
        """Regression: one character over the limit refused the whole payload.

        The payload's first MAX_CHARS characters are graded; the rest is set
        aside with its length and the verdict is marked partial, so an attack
        inside the graded stretch is still caught and padding past the limit
        can never read as a full all-clear.
        """
        limit = ThreatPayloadAdapter.MAX_CHARS
        attack = "x' OR 1=1 --" + "a" * limit
        out = ThreatPayloadAdapter().detect([attack, "plain text"])
        assert out["flags"].tolist() == [True, False]
        assert out["partial"] is True
        assert set(out["set_aside"]) == {0}
        (entry,) = out["set_aside"][0]
        assert entry["where"] == f"payload[0][{limit}:]"
        assert entry["reason"].startswith(f"{len(attack) - limit} of {len(attack)} characters")
        within = ThreatPayloadAdapter().detect("x" * limit)
        assert "partial" not in within

    def test_non_text_payload_is_refused(self) -> None:
        with pytest.raises(InputModalityError):
            ThreatPayloadAdapter().detect(b"' OR 1=1")
        with pytest.raises(InputModalityError):
            ThreatPayloadAdapter().detect(["ok", 42])


class TestFinancialCrisis:
    CRISIS: dict[str, Any] = {
        "market_data": {"price_change_pct": -7.0, "vix": 45.0},
        "banking_data": {"cds_spread_bps": 420.0},
    }
    CALM: dict[str, Any] = {"market_data": {"price_change_pct": 0.5, "vix": 14.0}}
    COLLAPSE: dict[str, Any] = {
        "market_data": {"price_change_pct": -9.0, "vix": 80.0},
        "banking_data": {"cds_spread_bps": 600.0, "liquidity_ratio": 0.4},
        "network_data": {
            "interconnectedness_score": 0.95,
            "concentration_ratio": 0.9,
            "cross_border_exposure": 0.9,
        },
    }
    LADDER = ("stable", "stress", "crisis", "severe_crisis", "systemic_collapse")

    @pytest.mark.parametrize("record", [CRISIS, CALM, COLLAPSE])
    def test_matches_the_detector_on_its_severity_ladder(self, record: dict[str, Any]) -> None:
        from omni_mercury_engine.detectors.economic.financial_crisis_detector import (
            FinancialCrisisDetector,
        )

        direct = FinancialCrisisDetector().predict_financial_crisis(record)
        out = FinancialCrisisRecordAdapter().detect(record)
        assert out["scores"][0] == pytest.approx(self.LADDER.index(direct.severity_level) / 4)
        assert bool(out["flags"][0]) is direct.crisis_imminent

    def test_decision_is_exactly_score_at_least_half(self) -> None:
        out = FinancialCrisisRecordAdapter().detect([self.CRISIS, self.CALM, self.COLLAPSE])
        assert out["flags"].tolist() == [True, False, True]
        assert out["flags"].tolist() == (out["scores"] >= 0.5).tolist()
        assert out["scores"][2] == 1.0  # systemic collapse

    def test_unscored_systemic_risk_is_nan_not_zero(self) -> None:
        names = FinancialCrisisRecordAdapter.FEATURE_NAMES
        features = FinancialCrisisRecordAdapter().extract_features([self.CALM, self.COLLAPSE])
        assert np.isnan(features[0, names.index("systemic_risk_score")])
        assert features[1, names.index("systemic_risk_score")] > 0.7
        assert np.isnan(features[0, names.index("credit_default_swap_spread")])

    @pytest.mark.parametrize(
        "record",
        [
            {},
            {"market": {"vix": 60.0}},  # unknown section
            {"market_data": {"VIX": 60.0}},  # misspelt measurement
            {"trading_data": {"volume_anomaly_score": 0.9}},  # unconsulted network only
        ],
    )
    def test_unmeasured_or_misspelt_record_is_refused(self, record: dict[str, Any]) -> None:
        with pytest.raises(InvalidObservationError):
            FinancialCrisisRecordAdapter().detect(record)


@pytest.fixture(scope="module")
def gannon_peak_dbdt_nt_s() -> float:
    """Peak horizontal 1-minute dB/dt at Boulder on 2024-05-10 (Gannon storm)."""
    data = json.loads((FIXTURES / "space_weather" / "usgs_geomag_bou_2024_05_10.json").read_text())
    series = {v["id"]: v["values"] for v in data["data"]["values"]}

    def arr(values: list[Any]) -> np.ndarray[Any, Any]:
        return np.array([np.nan if v is None else float(v) for v in values])

    rate = np.hypot(np.diff(arr(series["X"])), np.diff(arr(series["Y"]))) / 60.0
    return float(np.nanmax(rate))


class TestEMP:
    IEC_E1: dict[str, Any] = {
        "sensor_data": {
            "field_strength_vm": 50_000.0,
            "rise_time_ns": 2.47,
            "peak_frequency_mhz": 100.0,
        }
    }
    LADDER = ("benign", "anomalous", "suspicious", "threat", "critical")

    @staticmethod
    def _direct(record: dict[str, Any]) -> Any:
        from omni_mercury_engine.detectors.energy.emp_detector import EMPDetector

        return EMPDetector().predict_emp(record)

    def test_recorded_gannon_storm_is_a_grid_threat_not_an_emp(
        self, gannon_peak_dbdt_nt_s: float
    ) -> None:
        record = {
            "magnetometer_data": {
                "db_dt_nt_s": gannon_peak_dbdt_nt_s,
                "duration_seconds": 60.0,
                "dbdt_cadence_s": 60.0,
            }
        }
        direct = self._direct(record)
        out = EMPRecordAdapter().detect(record)
        result = out["metadata"]["results"][0]
        assert gannon_peak_dbdt_nt_s > 3.0  # ~250 nT/min: a G5 storm at mid latitude
        assert result.emp_type == "solar_geomagnetic_current"
        assert not result.emp_detected
        assert result.threat_level == direct.threat_level
        assert out["scores"][0] == pytest.approx(self.LADDER.index(direct.threat_level) / 4)
        assert not out["flags"][0]  # below THREAT: a storm calls for grid actions, not EMP ones

    def test_iec_e1_is_a_threat_and_e1_plus_e3_is_critical(self) -> None:
        burst = {
            **self.IEC_E1,
            "magnetometer_data": {"db_dt_nt_s": 5000.0, "duration_seconds": 120.0},
        }
        out = EMPRecordAdapter().detect([self.IEC_E1, burst])
        assert out["scores"].tolist() == pytest.approx([0.75, 1.0])
        assert out["flags"].tolist() == [True, True]
        for record, result in zip([self.IEC_E1, burst], out["metadata"]["results"], strict=True):
            assert result.threat_level == self._direct(record).threat_level

    def test_quiet_field_is_benign(self) -> None:
        out = EMPRecordAdapter().detect({"sensor_data": {"field_strength_vm": 5.0}})
        assert out["scores"][0] == 0.0
        assert not out["flags"][0]

    @pytest.mark.parametrize(
        "record",
        [
            {},
            {"sensor": {"field_strength_vm": 5.0}},  # unknown section
            {"sensor_data": {"field_strength": 50_000.0}},  # misspelt measurement
            {"signature_data": {"repetition_rate_hz": 100.0}},  # nothing measured
        ],
    )
    def test_unmeasured_or_misspelt_record_is_refused(self, record: dict[str, Any]) -> None:
        with pytest.raises(InvalidObservationError):
            EMPRecordAdapter().detect(record)
