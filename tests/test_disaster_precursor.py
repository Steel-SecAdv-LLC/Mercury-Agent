# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Disaster Precursor Detector."""

from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from omni_mercury_engine.space.disaster_precursor_detector import (
    DisasterPrecursorDetector,
    DisasterPrecursorResult,
    EarthquakePrecursorAnalyzer,
    GeomageticCorrelator,
    IonosphericDisturbanceDetector,
    SeismicCorrelator,
)


class TestDisasterPrecursorResult:
    """Tests for DisasterPrecursorResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = DisasterPrecursorResult(
            precursor_detected=False,
            confidence=0.0,
            disaster_type="none",
            risk_level="low",
        )
        assert result.precursor_detected is False
        assert result.time_to_event_hours is None
        assert result.ionospheric_disturbance is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        result = DisasterPrecursorResult(
            precursor_detected=True,
            confidence=0.8,
            disaster_type="earthquake",
            risk_level="high",
            time_to_event_hours=24.0,
            estimated_magnitude=6.5,
        )
        assert result.precursor_detected is True
        assert result.estimated_magnitude == 6.5


class TestEarthquakePrecursorAnalyzer:
    """Tests for EarthquakePrecursorAnalyzer class."""

    def test_init(self) -> None:
        """Test initialization."""
        analyzer = EarthquakePrecursorAnalyzer()
        assert isinstance(analyzer, torch.nn.Module)

    def test_init_custom_dim(self) -> None:
        """Test initialization with custom input dimension."""
        analyzer = EarthquakePrecursorAnalyzer(input_dim=256)
        assert isinstance(analyzer, torch.nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        analyzer = EarthquakePrecursorAnalyzer(input_dim=128)
        features = torch.randn(4, 128)
        magnitude, time_to_event, confidence = analyzer(features)
        assert magnitude.shape == (4, 1)
        assert time_to_event.shape == (4, 1)
        assert confidence.shape == (4, 1)

    def test_output_ranges(self) -> None:
        """Test output value ranges (sigmoid outputs)."""
        analyzer = EarthquakePrecursorAnalyzer(input_dim=128)
        features = torch.randn(4, 128)
        magnitude, time_to_event, confidence = analyzer(features)
        assert (magnitude >= 0).all() and (magnitude <= 1).all()
        assert (time_to_event >= 0).all() and (time_to_event <= 1).all()
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_batch_sizes(self) -> None:
        """Test different batch sizes."""
        analyzer = EarthquakePrecursorAnalyzer(input_dim=128)
        analyzer.eval()
        for batch_size in [2, 8, 16]:
            features = torch.randn(batch_size, 128)
            magnitude, time_to_event, confidence = analyzer(features)
            assert magnitude.shape == (batch_size, 1)


class TestGeomageticCorrelator:
    """Tests for GeomageticCorrelator class."""

    def test_init(self) -> None:
        """Test initialization."""
        correlator = GeomageticCorrelator()
        assert correlator.kp_thresholds is not None

    def test_correlate_default_data(self) -> None:
        """Test correlation with default geomagnetic data."""
        correlator = GeomageticCorrelator()
        schumann_anomaly = {"frequency_anomaly": False, "amplitude_anomaly": False}
        result = correlator.correlate_geomagnetic(schumann_anomaly)
        assert "correlation_strength" in result
        assert "geomagnetic_status" in result

    def test_correlate_with_storm(self) -> None:
        """Test correlation during geomagnetic storm."""
        correlator = GeomageticCorrelator()
        schumann_anomaly = {
            "frequency_anomaly": True,
            "amplitude_anomaly": True,
            "power_spectrum_shift": True,
        }
        geomagnetic_data = {"kp_index": 7.0, "dst_index": -100}
        result = correlator.correlate_geomagnetic(schumann_anomaly, geomagnetic_data)
        assert result["correlation_strength"] > 0
        assert result["geomagnetic_status"] == "severe_storm"

    def test_correlate_quiet_conditions(self) -> None:
        """Test correlation during quiet conditions."""
        correlator = GeomageticCorrelator()
        schumann_anomaly = {"frequency_anomaly": False}
        geomagnetic_data = {"kp_index": 1.0, "dst_index": -10}
        result = correlator.correlate_geomagnetic(schumann_anomaly, geomagnetic_data)
        assert result["geomagnetic_status"] == "quiet"

    def test_classify_geomagnetic_activity(self) -> None:
        """Test geomagnetic activity classification."""
        correlator = GeomageticCorrelator()
        assert correlator._classify_geomagnetic_activity(1.0) == "quiet"
        assert correlator._classify_geomagnetic_activity(3.5) == "unsettled"
        assert correlator._classify_geomagnetic_activity(4.5) == "active"
        assert correlator._classify_geomagnetic_activity(5.5) == "minor_storm"
        assert correlator._classify_geomagnetic_activity(6.5) == "major_storm"
        assert correlator._classify_geomagnetic_activity(8.0) == "severe_storm"

    def test_space_weather_factor(self) -> None:
        """Test space weather factor calculation."""
        correlator = GeomageticCorrelator()
        schumann_anomaly: dict[str, Any] = {}
        result_low = correlator.correlate_geomagnetic(
            schumann_anomaly, {"kp_index": 3.0, "dst_index": -20}
        )
        result_high = correlator.correlate_geomagnetic(
            schumann_anomaly, {"kp_index": 6.0, "dst_index": -20}
        )
        assert result_high["space_weather_factor"] > result_low["space_weather_factor"]


class TestIonosphericDisturbanceDetector:
    """Tests for IonosphericDisturbanceDetector class."""

    def test_init(self) -> None:
        """Test initialization."""
        detector = IonosphericDisturbanceDetector()
        assert detector is not None

    def test_detect_no_disturbance(self) -> None:
        """Test detection with no disturbance."""
        detector = IonosphericDisturbanceDetector()
        schumann_data = {"fundamental_deviation": 0.1}
        result = detector.detect_ionospheric_disturbance(schumann_data)
        assert result["disturbance_detected"] is False

    def test_detect_frequency_shift(self) -> None:
        """Test detection with significant frequency shift."""
        detector = IonosphericDisturbanceDetector()
        schumann_data = {"fundamental_deviation": 2.0}
        result = detector.detect_ionospheric_disturbance(schumann_data)
        assert result["disturbance_detected"] is True
        assert "significant_frequency_shift" in result["indicators"]

    def test_detect_with_tec_data(self) -> None:
        """Test detection with TEC data."""
        detector = IonosphericDisturbanceDetector()
        schumann_data = {"fundamental_deviation": 0.5}
        tec_data = np.array([50.0, 55.0, 60.0, 40.0, 35.0])
        result = detector.detect_ionospheric_disturbance(schumann_data, tec_data)
        assert "disturbance_level" in result

    def test_detect_high_tec_variability(self) -> None:
        """Test detection with high TEC variability."""
        detector = IonosphericDisturbanceDetector()
        schumann_data = {"fundamental_deviation": 0.5}
        tec_data = np.array([50.0, 70.0, 30.0, 80.0, 20.0])
        result = detector.detect_ionospheric_disturbance(schumann_data, tec_data)
        assert result["disturbance_detected"] is True

    def test_detect_tec_depletion(self) -> None:
        """Test detection of TEC depletion."""
        detector = IonosphericDisturbanceDetector()
        schumann_data = {"fundamental_deviation": 0.5}
        tec_data = np.array([60.0, 55.0, 50.0, 45.0, 40.0])
        result = detector.detect_ionospheric_disturbance(schumann_data, tec_data)
        assert "indicators" in result


class TestSeismicCorrelator:
    """Tests for SeismicCorrelator class."""

    def test_init(self) -> None:
        """Test initialization."""
        correlator = SeismicCorrelator()
        assert correlator is not None

    def test_correlate_no_data(self) -> None:
        """Test correlation with no seismic data."""
        correlator = SeismicCorrelator()
        schumann_anomaly = {"risk_score": 0.5}
        result = correlator.correlate_seismic(schumann_anomaly, None)
        assert result["correlation"] == 0.0
        assert result["significant"] is False

    def test_correlate_empty_data(self) -> None:
        """Test correlation with empty seismic data."""
        correlator = SeismicCorrelator()
        schumann_anomaly = {"risk_score": 0.5}
        result = correlator.correlate_seismic(schumann_anomaly, np.array([]))
        assert result["correlation"] == 0.0

    def test_correlate_low_activity(self) -> None:
        """Test correlation with low seismic activity."""
        correlator = SeismicCorrelator()
        schumann_anomaly = {"risk_score": 0.3}
        seismic_data = np.array([2.0, 2.5, 3.0, 2.8, 2.2])
        result = correlator.correlate_seismic(schumann_anomaly, seismic_data)
        assert result["significant"] is False

    def test_correlate_high_activity(self) -> None:
        """Test correlation with high seismic activity."""
        correlator = SeismicCorrelator()
        schumann_anomaly = {"risk_score": 0.8}
        seismic_data = np.array([5.0, 5.5, 6.0, 5.8, 5.2])
        result = correlator.correlate_seismic(schumann_anomaly, seismic_data)
        assert result["correlation"] > 0
        assert "recent_seismic_activity" in result

    def test_correlate_significant(self) -> None:
        """Test significant correlation detection."""
        correlator = SeismicCorrelator()
        schumann_anomaly = {"risk_score": 0.9}
        seismic_data = np.array([6.0, 6.5, 7.0, 6.8, 6.2])
        result = correlator.correlate_seismic(schumann_anomaly, seismic_data)
        assert result["significant"]
        assert result["precursor_likelihood"] > 0


class TestDisasterPrecursorDetector:
    """Tests for DisasterPrecursorDetector class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        detector = DisasterPrecursorDetector()
        assert detector.enable_earthquake is True
        assert detector.enable_tsunami is True
        assert detector.enable_geomagnetic is True

    def test_init_disabled_components(self) -> None:
        """Test initialization with disabled components."""
        detector = DisasterPrecursorDetector(
            enable_earthquake=False,
            enable_tsunami=False,
            enable_geomagnetic=False,
        )
        assert detector.earthquake_analyzer is None
        assert detector.geomagnetic_correlator is None

    def test_detect_no_signal(self) -> None:
        """Test detection with no ELF signal."""
        detector = DisasterPrecursorDetector()
        result = detector.detect_disaster_precursor({})
        assert result.precursor_detected is False
        assert result.disaster_type == "none"

    def test_detect_with_elf_signal(self) -> None:
        """Test detection with ELF signal."""
        detector = DisasterPrecursorDetector()
        elf_signal = np.sin(2 * np.pi * 7.83 * np.linspace(0, 1, 100))
        precursor_data = {"elf_signal": elf_signal}
        result = detector.detect_disaster_precursor(precursor_data)
        assert result.schumann_anomaly is not None

    def test_detect_with_geomagnetic_data(self) -> None:
        """Test detection with geomagnetic data."""
        detector = DisasterPrecursorDetector()
        elf_signal = np.sin(2 * np.pi * 7.83 * np.linspace(0, 1, 100))
        precursor_data = {
            "elf_signal": elf_signal,
            "geomagnetic_data": {"kp_index": 5.0, "dst_index": -50},
        }
        result = detector.detect_disaster_precursor(precursor_data)
        assert result.geomagnetic_indicators is not None

    def test_detect_with_seismic_data(self) -> None:
        """Test detection with seismic data."""
        detector = DisasterPrecursorDetector()
        elf_signal = np.sin(2 * np.pi * 7.83 * np.linspace(0, 1, 100))
        precursor_data = {
            "elf_signal": elf_signal,
            "seismic_data": np.array([4.0, 4.5, 5.0, 4.8, 4.2]),
        }
        result = detector.detect_disaster_precursor(precursor_data)
        assert result.seismic_correlation is not None

    def test_detect_with_tec_data(self) -> None:
        """Test detection with TEC data."""
        detector = DisasterPrecursorDetector()
        elf_signal = np.sin(2 * np.pi * 7.83 * np.linspace(0, 1, 100))
        precursor_data = {
            "elf_signal": elf_signal,
            "tec_data": np.array([50.0, 55.0, 60.0, 45.0, 40.0]),
        }
        result = detector.detect_disaster_precursor(precursor_data)
        assert isinstance(result.ionospheric_disturbance, bool)

    def test_assess_risk_level(self) -> None:
        """Test risk level assessment."""
        detector = DisasterPrecursorDetector()

        result_low = DisasterPrecursorResult(
            precursor_detected=False,
            confidence=0.2,
            disaster_type="none",
            risk_level="low",
        )
        assert detector._assess_risk_level(result_low) == "low"

        result_high = DisasterPrecursorResult(
            precursor_detected=True,
            confidence=0.7,
            disaster_type="earthquake",
            risk_level="high",
        )
        assert detector._assess_risk_level(result_high) == "high"

    def test_generate_early_warning_actions(self) -> None:
        """Test early warning action generation."""
        detector = DisasterPrecursorDetector()
        result = DisasterPrecursorResult(
            precursor_detected=True,
            confidence=0.9,
            disaster_type="earthquake",
            risk_level="critical",
        )
        actions = detector._generate_early_warning_actions(result)
        assert len(actions) > 0

    def test_generate_tsunami_warning(self) -> None:
        """Test tsunami warning generation."""
        detector = DisasterPrecursorDetector()
        result = DisasterPrecursorResult(
            precursor_detected=True,
            confidence=0.9,
            disaster_type="earthquake_tsunami_risk",
            risk_level="critical",
        )
        actions = detector._generate_early_warning_actions(result)
        assert any("TSUNAMI" in action for action in actions)

    def test_estimate_time_to_event(self) -> None:
        """Test time to event estimation."""
        detector = DisasterPrecursorDetector()
        time_low = detector._estimate_time_to_event(0.3, 0.3)
        time_high = detector._estimate_time_to_event(0.9, 0.9)
        assert time_high < time_low
        assert time_high >= 2.0
