# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for the OSHA compliance anomaly detector."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest
import requests

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
from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError


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

        At moderate humidity the simplified formula **over-reports**
        apparent temperature relative to the NWS Rothfusz regression.
        This test pins that gap at T=95 F / RH=70%, where the heuristic
        returns 130 F while Rothfusz returns ~122 F per the NWS heat
        index table.  The mirror failure mode - the simplified formula
        *under-reporting* at low humidity because it skips the
        low-humidity adjustment - is covered by
        :meth:`test_low_humidity_adjustment_engaged`.
        """
        rothfusz = compute_heat_index_fahrenheit(95.0, 70.0)
        simplified = 95.0 + 0.5 * 70.0  # 130 F
        # Rothfusz at T=95, RH=70% is ~122 F per NWS table, so the
        # simplified heuristic over-reports by ~8 F at this operating
        # point.
        assert simplified > rothfusz
        assert simplified - rothfusz > 5.0
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

    @pytest.mark.parametrize(
        ("temperature_f", "relative_humidity", "expected_hi", "tol"),
        [
            # Task 5 NWS reference points (https://www.wpc.ncep.noaa.gov/html/heatindex.shtml)
            # 1. Steadman / low-temp path: HI ~ T at moderate RH and low T.
            (80.0, 40.0, 80.0, 2.0),
            # 2. Rothfusz, no adjustment: T=95, RH=70 -> ~122 F.
            (95.0, 70.0, 122.0, 1.0),
            # 3. Rothfusz + low-humidity adjustment: T=100, RH=10 -> ~95 F.
            (100.0, 10.0, 95.0, 1.0),
        ],
    )
    def test_heat_index_known_values(
        self,
        temperature_f: float,
        relative_humidity: float,
        expected_hi: float,
        tol: float,
    ) -> None:
        """Three NWS reference points pinning the Rothfusz / Steadman branches.

        Source: NWS Weather Prediction Center, heat-index table
        (https://www.wpc.ncep.noaa.gov/html/heatindex.shtml).  Reference
        points cover (a) the Steadman / low-temp branch, (b) the
        unadjusted Rothfusz branch, and (c) the low-humidity adjustment
        branch so a regression that disables either branch is caught.
        """
        value = compute_heat_index_fahrenheit(temperature_f, relative_humidity)
        assert value == pytest.approx(expected_hi, abs=tol)


class TestECFRClient:
    """Tests for the eCFR client (verifies without making real HTTP calls).

    Every test patches :class:`SafeHTTPClient` at its public boundary so
    the scheme allowlist, IP/private-network gate, DNS-rebinding pin,
    and redirect refusal sit in front of every eCFR call - the unit
    tests focus on the client's caching, parsing, and exception-mapping
    logic.
    """

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

    def test_init_rejects_unlisted_base_url(self) -> None:
        """Any base_url outside ALLOWED_BASE_URLS is refused at construction."""
        with pytest.raises(ValueError, match="ALLOWED_BASE_URLS"):
            ECFRClient(base_url="https://attacker.example/api")
        # The default (https://www.ecfr.gov) is always accepted.
        ECFRClient(base_url="https://www.ecfr.gov")

    def test_init_canonicalises_trailing_slash(self) -> None:
        """A trailing slash is allowed and stripped during validation."""
        client = ECFRClient(base_url="https://www.ecfr.gov/")
        # Internal attribute is intentionally read here to confirm
        # canonicalisation - the trailing slash is removed before
        # being concatenated into request URLs.
        assert client._base_url == "https://www.ecfr.gov"

    def test_verify_citation_caches_result(self) -> None:
        """Verified results are cached in-process and not re-fetched."""
        client = ECFRClient()
        payload = {
            "children": [{"type": "chapter", "children": [{"type": "part", "identifier": "1910"}]}]
        }
        call_count = {"n": 0}

        def _fake_get_json(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            call_count["n"] += 1
            return payload

        with patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)):
            assert client.verify_citation("29 CFR 1910.95") is True
            assert client.verify_citation("29 CFR 1910.95") is True
        assert call_count["n"] == 1, "Second verify must reuse the cached result"

    def test_verify_citation_unparseable_returns_false(self) -> None:
        """Unparseable citations don't hit the network."""
        client = ECFRClient()
        call_count = {"n": 0}

        def _fake_get_json(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            call_count["n"] += 1
            return {}

        with patch.object(SafeHTTPClient, "get_json", staticmethod(_fake_get_json)):
            assert client.verify_citation("OSHA Guidelines") is False
        assert call_count["n"] == 0

    def test_verify_citation_raises_on_network_error(self) -> None:
        """Non-404 HTTP errors are wrapped as ECFRClientError."""
        client = ECFRClient()

        def _raise_unsafe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise UnsafeURLError("scheme allowlist denial")

        with (
            patch.object(SafeHTTPClient, "get_json", staticmethod(_raise_unsafe)),
            pytest.raises(ECFRClientError),
        ):
            client.verify_citation("29 CFR 1910.95")

    def test_verify_citation_500_wrapped(self) -> None:
        """5xx HTTP errors are wrapped as ECFRClientError, not silently cached."""
        client = ECFRClient()

        def _raise_http_500(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            response = requests.Response()
            response.status_code = 500
            response.reason = "Internal Server Error"
            raise requests.HTTPError("500 Internal Server Error", response=response)

        with (
            patch.object(SafeHTTPClient, "get_json", staticmethod(_raise_http_500)),
            pytest.raises(ECFRClientError, match="500"),
        ):
            client.verify_citation("29 CFR 1910.95")

    def test_verify_citation_404_returns_false(self) -> None:
        """404 responses are cached as False without raising."""
        client = ECFRClient()

        def _raise_http_404(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            response = requests.Response()
            response.status_code = 404
            response.reason = "Not Found"
            raise requests.HTTPError("404 Not Found", response=response)

        with patch.object(SafeHTTPClient, "get_json", staticmethod(_raise_http_404)):
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
