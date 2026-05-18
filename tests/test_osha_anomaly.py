"""Tests for the OSHA compliance anomaly detector."""

from __future__ import annotations

import io
import json
from typing import Any, cast
from unittest.mock import patch

import pytest

from omni_mercury_engine.compliance.osha_anomaly import (
    ComplianceLevel,
    ECFRClient,
    ECFRClientError,
    HazardCategory,
    OSHAComplianceDetector,
    OSHAHazard,
    OSHASector,
    compute_heat_index_fahrenheit,
    get_osha_compliance_detector,
)


class TestRothfuszHeatIndex:
    """Tests for the NWS Rothfusz heat-index regression."""

    def test_low_temperature_uses_simple_steadman(self) -> None:
        """At T < 80F the simple Steadman form is returned."""
        # Steadman formula at T=70 F, RH=50%
        value = compute_heat_index_fahrenheit(70.0, 50.0)
        assert value == pytest.approx(0.5 * (70.0 + 61.0 + 2.4 + 4.7), rel=1e-9)

    def test_published_nws_reference_point(self) -> None:
        """NWS published example: 96 F at 65% RH ~ 121 F."""
        # https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
        value = compute_heat_index_fahrenheit(96.0, 65.0)
        assert value == pytest.approx(121.0, abs=1.0)

    def test_low_humidity_adjustment_engaged(self) -> None:
        """The low-RH adjustment (RH<13, 80<=T<=112) lowers the HI."""
        adjusted = compute_heat_index_fahrenheit(100.0, 10.0)
        # Re-run without adjustment by passing RH=13.0
        baseline = compute_heat_index_fahrenheit(100.0, 13.0)
        assert adjusted < baseline

    def test_low_temperature_high_humidity_adjustment_engaged(self) -> None:
        """The low-T/high-RH adjustment (RH>85, 80<=T<=87) raises the HI."""
        adjusted = compute_heat_index_fahrenheit(82.0, 95.0)
        baseline = compute_heat_index_fahrenheit(82.0, 85.0)
        assert adjusted > baseline

    def test_regression_vs_simplified_formula(self) -> None:
        """Rothfusz must materially differ from the original ``T + 0.5*RH``.

        The simplified formula heavily under-reports at moderate-RH
        conditions; this test pins the difference at T=95 F, RH=70%.
        """
        rothfusz = compute_heat_index_fahrenheit(95.0, 70.0)
        simplified = 95.0 + 0.5 * 70.0  # 130 F
        # Rothfusz at T=95, RH=70% is ~122 F per NWS table; not 130 F
        assert abs(rothfusz - simplified) > 5.0
        assert rothfusz == pytest.approx(122.0, abs=2.0)

    def test_rejects_invalid_humidity(self) -> None:
        """RH outside [0, 100] raises ValueError."""
        with pytest.raises(ValueError):
            compute_heat_index_fahrenheit(80.0, 150.0)
        with pytest.raises(ValueError):
            compute_heat_index_fahrenheit(80.0, -1.0)

    def test_rejects_non_finite_temperature(self) -> None:
        """Non-finite temperature raises ValueError."""
        with pytest.raises(ValueError):
            compute_heat_index_fahrenheit(float("inf"), 50.0)


class TestECFRClient:
    """Tests for the eCFR client (verifies without making real HTTP calls)."""

    def test_parse_citation_numeric(self) -> None:
        """parse_citation extracts numeric Title/Part/Section tuples."""
        parsed = ECFRClient.parse_citation("29 CFR 1910.95")
        assert parsed == ("29", "1910", "95")

    def test_parse_citation_non_numeric_returns_none(self) -> None:
        """parse_citation returns None for OSHA guidance citations."""
        assert ECFRClient.parse_citation("OSHA Heat Illness Prevention") is None

    def test_parse_citation_part_only(self) -> None:
        """parse_citation accepts citations with no section component."""
        parsed = ECFRClient.parse_citation("29 CFR 1926")
        assert parsed == ("29", "1926", "")

    def test_verify_citation_caches_result(self) -> None:
        """Verified results are cached in-process and not re-fetched."""
        client = ECFRClient()
        payload = {
            "children": [{"type": "chapter", "children": [{"type": "part", "identifier": "1910"}]}]
        }
        body = json.dumps(payload).encode("utf-8")

        class _Response:
            status = 200

            def __init__(self, data: bytes) -> None:
                self._buf = io.BytesIO(data)

            def read(self) -> bytes:
                return self._buf.read()

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        with patch(
            "omni_mercury_engine.compliance.osha_anomaly.urlopen",
            return_value=_Response(body),
        ) as mocked:
            assert client.verify_citation("29 CFR 1910.95") is True
            assert client.verify_citation("29 CFR 1910.95") is True
            assert mocked.call_count == 1

    def test_verify_citation_unparseable_returns_false(self) -> None:
        """Unparseable citations don't hit the network."""
        client = ECFRClient()
        with patch("omni_mercury_engine.compliance.osha_anomaly.urlopen") as mocked:
            assert client.verify_citation("OSHA Guidelines") is False
            mocked.assert_not_called()

    def test_verify_citation_raises_on_network_error(self) -> None:
        """Non-404 HTTP errors are wrapped as ECFRClientError."""
        from urllib.error import URLError

        client = ECFRClient()
        with (
            patch(
                "omni_mercury_engine.compliance.osha_anomaly.urlopen",
                side_effect=URLError("boom"),
            ),
            pytest.raises(ECFRClientError),
        ):
            client.verify_citation("29 CFR 1910.95")

    def test_verify_citation_404_returns_false(self) -> None:
        """404 responses are cached as False without raising."""
        from urllib.error import HTTPError

        client = ECFRClient()
        with patch(
            "omni_mercury_engine.compliance.osha_anomaly.urlopen",
            side_effect=HTTPError(
                "https://www.ecfr.gov/", 404, "Not Found", {}, None  # type: ignore[arg-type]
            ),
        ):
            assert client.verify_citation("29 CFR 9999.99") is False


class TestOSHAComplianceDetectorConstruction:
    """Construction-sector hazard detection."""

    def test_fall_hazard_critical(self) -> None:
        """Worker above 6 feet without protection emits a critical fall hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.CONSTRUCTION)
        hazards = detector.detect_hazards(
            {"height_above_ground": 15.0, "fall_protection_active": False}
        )
        assert any(h.category is HazardCategory.FALL for h in hazards)
        fall = next(h for h in hazards if h.category is HazardCategory.FALL)
        assert fall.compliance_level is ComplianceLevel.CRITICAL_VIOLATION
        assert fall.osha_standard == "29 CFR 1926.501"

    def test_fall_protection_active_suppresses_hazard(self) -> None:
        """fall_protection_active=True must suppress the fall hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.CONSTRUCTION)
        hazards = detector.detect_hazards(
            {"height_above_ground": 15.0, "fall_protection_active": True}
        )
        assert all(h.category is not HazardCategory.FALL for h in hazards)

    def test_electrical_hazard_detected(self) -> None:
        """Voltage > 50 V without protection emits an electrical hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.CONSTRUCTION)
        hazards = detector.detect_hazards(
            {"electrical_voltage": 480.0, "electrical_protection": False}
        )
        assert any(h.category is HazardCategory.ELECTRICAL for h in hazards)


class TestOSHAComplianceDetectorAgriculture:
    """Agriculture-sector hazard detection."""

    def test_machinery_proximity_hazard(self) -> None:
        """Worker within 3m of active machinery emits a machinery hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.AGRICULTURE)
        hazards = detector.detect_hazards(
            {"machinery_proximity_meters": 1.0, "machinery_active": True}
        )
        assert any(h.category is HazardCategory.MACHINERY for h in hazards)

    def test_heat_stress_uses_rothfusz(self) -> None:
        """Heat stress fires using the Rothfusz HI, not T + 0.5*RH."""
        detector = OSHAComplianceDetector(sector=OSHASector.AGRICULTURE)
        # T=95, RH=70% -> Rothfusz HI ~ 122 F, > 103 threshold
        hazards = detector.detect_hazards(
            {"temperature_fahrenheit": 95.0, "humidity_percent": 70.0}
        )
        assert any(h.category is HazardCategory.HEAT_STRESS for h in hazards)

    def test_low_heat_no_hazard(self) -> None:
        """Low temperatures yield no heat-stress hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.AGRICULTURE)
        hazards = detector.detect_hazards(
            {"temperature_fahrenheit": 70.0, "humidity_percent": 50.0}
        )
        assert all(h.category is not HazardCategory.HEAT_STRESS for h in hazards)


class TestOSHAComplianceDetectorHealthcare:
    """Healthcare-sector hazard detection."""

    def test_violence_hazard(self) -> None:
        """High violence risk score emits a violence hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.HEALTHCARE)
        hazards = detector.detect_hazards({"violence_risk_score": 0.9})
        assert any(h.category is HazardCategory.VIOLENCE for h in hazards)

    def test_bloodborne_pathogen_hazard(self) -> None:
        """High pathogen exposure risk emits a biological hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.HEALTHCARE)
        hazards = detector.detect_hazards({"pathogen_exposure_risk": 0.95})
        assert any(h.category is HazardCategory.BIOLOGICAL for h in hazards)


class TestOSHAComplianceDetectorManufacturing:
    """Manufacturing-sector hazard detection."""

    def test_noise_hazard(self) -> None:
        """Noise > 85 dB emits a noise hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.MANUFACTURING)
        hazards = detector.detect_hazards({"noise_level_db": 100.0})
        assert any(h.category is HazardCategory.NOISE for h in hazards)

    def test_ergonomic_hazard(self) -> None:
        """High repetitive-motion score emits an ergonomic hazard."""
        detector = OSHAComplianceDetector(sector=OSHASector.MANUFACTURING)
        hazards = detector.detect_hazards({"repetitive_motion_score": 0.85})
        assert any(h.category is HazardCategory.ERGONOMIC for h in hazards)


class TestOSHATraining:
    """Training recommendation behaviour."""

    def test_critical_hazards_get_osha_30(self) -> None:
        """Critical hazards trigger the OSHA 30-Hour Outreach Training."""
        detector = OSHAComplianceDetector(sector=OSHASector.CONSTRUCTION)
        hazards = detector.detect_hazards(
            {"height_above_ground": 20.0, "fall_protection_active": False}
        )
        training = detector.recommend_training(hazards)
        assert any("OSHA 30" in t.program_name for t in training)
        assert any("Fall Protection" in t.program_name for t in training)


class TestOSHAReport:
    """generate_compliance_report integration."""

    def test_report_includes_counts_and_score(self) -> None:
        """The report includes hazard counts and a compliance score."""
        detector = OSHAComplianceDetector(sector=OSHASector.MANUFACTURING)
        hazards = detector.detect_hazards({"noise_level_db": 100.0})
        report = detector.generate_compliance_report(hazards, {"facility_name": "Test Facility"})
        assert report["sector"] == "MANUFACTURING"
        assert report["total_hazards"] >= 1
        assert "overall_compliance_score" in report
        assert isinstance(report["overall_compliance_score"], float)
        assert "next_steps" in report

    def test_report_empty_hazards(self) -> None:
        """No hazards yields a compliance score of 1.0."""
        detector = OSHAComplianceDetector()
        report = detector.generate_compliance_report([], {"facility_name": "Clean"})
        assert report["total_hazards"] == 0
        assert report["overall_compliance_score"] == pytest.approx(1.0)


class TestECFRIntegration:
    """detect_hazards wires citation_verified when an eCFR client is present."""

    def test_citation_verified_set_when_client_provided(self) -> None:
        """citation_verified is populated when an eCFR client is passed."""

        class _FakeClient:
            def verify_citation(self, citation: str) -> bool:
                return citation.startswith("29 CFR")

        detector = OSHAComplianceDetector(
            sector=OSHASector.MANUFACTURING,
            ecfr_client=cast("ECFRClient", _FakeClient()),
        )
        hazards = detector.detect_hazards({"noise_level_db": 100.0})
        noise = next(h for h in hazards if h.category is HazardCategory.NOISE)
        assert noise.citation_verified is True

    def test_citation_verified_none_on_client_error(self) -> None:
        """citation_verified is None when the eCFR client raises."""

        class _BrokenClient:
            def verify_citation(self, citation: str) -> bool:
                raise ECFRClientError("network down")

        detector = OSHAComplianceDetector(
            sector=OSHASector.MANUFACTURING,
            ecfr_client=cast("ECFRClient", _BrokenClient()),
        )
        hazards = detector.detect_hazards({"noise_level_db": 100.0})
        for h in hazards:
            assert h.citation_verified is None


class TestFactory:
    """Module-level factory."""

    def test_factory_returns_detector(self) -> None:
        """get_osha_compliance_detector returns a configured detector."""
        detector = get_osha_compliance_detector(sector=OSHASector.HEALTHCARE)
        assert isinstance(detector, OSHAComplianceDetector)
        assert detector.sector is OSHASector.HEALTHCARE


class TestComplianceLevel:
    """Severity-to-compliance-level mapping."""

    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            (0.95, ComplianceLevel.CRITICAL_VIOLATION),
            (0.80, ComplianceLevel.SERIOUS_VIOLATION),
            (0.65, ComplianceLevel.OTHER_THAN_SERIOUS),
            (0.45, ComplianceLevel.DE_MINIMIS),
            (0.10, ComplianceLevel.COMPLIANT),
        ],
    )
    def test_severity_brackets(self, severity: float, expected: ComplianceLevel) -> None:
        """Severity values map to the correct ComplianceLevel."""
        assert OSHAComplianceDetector._determine_compliance_level(severity) is expected


class TestOSHAHazardDataclass:
    """OSHAHazard equality and field behaviour."""

    def test_hazard_dataclass(self) -> None:
        """OSHAHazard exposes the expected fields."""
        detector = OSHAComplianceDetector(sector=OSHASector.MANUFACTURING)
        hazards = detector.detect_hazards({"noise_level_db": 90.0})
        for h in hazards:
            assert isinstance(h, OSHAHazard)
            assert h.osha_standard
            assert h.recommendations
            assert h.severity >= 0.0
