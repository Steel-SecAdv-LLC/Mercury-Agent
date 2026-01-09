"""
Tests for TEMPEST Detection module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC
"""

import numpy as np
import torch

from omni_mercury_engine.security.tempest_detection import (
    EmanationType,
    EMSECCountermeasureGenerator,
    RFSpectrumAnalyzer,
    SideChannelVulnerabilityAssessor,
    TEMPESTAnalysisResult,
    TEMPESTDetector,
    TEMPESTThreatLevel,
    VideoEmanationDetector,
)


class TestEmanationType:
    """Tests for EmanationType enum."""

    def test_enum_values(self):
        """Test enum values exist."""
        assert EmanationType.VIDEO_DISPLAY.value == "video_display_emanation"
        assert EmanationType.KEYBOARD.value == "keyboard_emanation"
        assert EmanationType.PROCESSOR.value == "processor_emanation"
        assert EmanationType.NETWORK_CABLE.value == "network_cable_emanation"
        assert EmanationType.POWER_LINE.value == "power_line_emanation"
        assert EmanationType.ACOUSTIC.value == "acoustic_emanation"
        assert EmanationType.OPTICAL.value == "optical_emanation"


class TestTEMPESTThreatLevel:
    """Tests for TEMPESTThreatLevel enum."""

    def test_enum_values(self):
        """Test enum values exist."""
        assert TEMPESTThreatLevel.NO_THREAT.value == "no_threat"
        assert TEMPESTThreatLevel.LOW.value == "low_risk"
        assert TEMPESTThreatLevel.MODERATE.value == "moderate_risk"
        assert TEMPESTThreatLevel.HIGH.value == "high_risk"
        assert TEMPESTThreatLevel.CRITICAL.value == "critical_risk"


class TestTEMPESTAnalysisResult:
    """Tests for TEMPESTAnalysisResult dataclass."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        result = TEMPESTAnalysisResult(
            emanation_detected=False,
            confidence=0.0,
            threat_level="no_threat",
            risk_score=0.0,
        )
        assert not result.emanation_detected
        assert result.confidence == 0.0
        assert result.emanation_types == []
        assert result.countermeasures == []

    def test_init_full(self):
        """Test initialization with all parameters."""
        result = TEMPESTAnalysisResult(
            emanation_detected=True,
            confidence=0.85,
            threat_level="high_risk",
            risk_score=0.75,
            emanation_types=["video_display_emanation"],
            frequency_bands=[{"equipment": "vga_video"}],
            signal_strength_dbm=-50.0,
            compromising_potential=0.8,
            reconstruction_feasibility=0.7,
            vulnerable_equipment=["unshielded_cables"],
            shielding_effectiveness=40.0,
            countermeasures=["Deploy TEMPEST-certified displays"],
            compliance_status={"zone1_shielding": False},
        )
        assert result.emanation_detected
        assert result.confidence == 0.85
        assert len(result.emanation_types) == 1
        assert result.signal_strength_dbm == -50.0


class TestRFSpectrumAnalyzer:
    """Tests for RFSpectrumAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = RFSpectrumAnalyzer()
        assert "vga_video" in analyzer.tempest_frequency_bands
        assert "keyboard" in analyzer.tempest_frequency_bands
        assert "processor" in analyzer.tempest_frequency_bands

    def test_analyze_spectrum_empty(self):
        """Test analysis with empty data."""
        analyzer = RFSpectrumAnalyzer()
        result = analyzer.analyze_spectrum({})
        assert not result["emanation_detected"]

    def test_analyze_spectrum_no_emanation(self):
        """Test analysis with no emanation detected."""
        analyzer = RFSpectrumAnalyzer()
        spectrum_data = {
            "frequencies": [1e6, 2e6, 3e6],
            "power_dbm": [-110.0, -115.0, -120.0],
        }
        result = analyzer.analyze_spectrum(spectrum_data)
        assert not result["emanation_detected"]

    def test_analyze_spectrum_with_emanation(self):
        """Test analysis with emanation detected."""
        analyzer = RFSpectrumAnalyzer()
        spectrum_data = {
            "frequencies": [50e6, 100e6, 150e6],  # VGA video band
            "power_dbm": [-50.0, -40.0, -60.0],  # Strong signals
        }
        result = analyzer.analyze_spectrum(spectrum_data)
        assert "emanation_detected" in result
        assert "max_signal_strength_dbm" in result

    def test_detect_band_emanation(self):
        """Test band emanation detection."""
        analyzer = RFSpectrumAnalyzer()
        frequencies = [50e6, 100e6, 150e6]
        power_levels = [-50.0, -40.0, -60.0]
        result = analyzer._detect_band_emanation(frequencies, power_levels, 25e6, 200e6)
        assert "detected" in result
        assert "peak_power" in result
        assert "compromising_potential" in result

    def test_detect_band_emanation_no_match(self):
        """Test band detection with no matching frequencies."""
        analyzer = RFSpectrumAnalyzer()
        frequencies = [1e6, 2e6, 3e6]
        power_levels = [-50.0, -40.0, -60.0]
        result = analyzer._detect_band_emanation(frequencies, power_levels, 100e6, 200e6)
        assert not result["detected"]


class TestVideoEmanationDetector:
    """Tests for VideoEmanationDetector class."""

    def test_init(self):
        """Test initialization."""
        detector = VideoEmanationDetector(input_dim=128)
        assert detector.emanation_encoder is not None
        assert detector.reconstruction_predictor is not None
        assert detector.resolution_estimator is not None

    def test_forward(self):
        """Test forward pass."""
        detector = VideoEmanationDetector(input_dim=64)
        detector.eval()
        x = torch.randn(4, 64)
        with torch.no_grad():
            reconstruction_score, resolution_probs = detector.forward(x)
        assert reconstruction_score.shape == (4, 1)
        assert resolution_probs.shape == (4, 3)
        assert torch.all(reconstruction_score >= 0) and torch.all(reconstruction_score <= 1)
        assert torch.allclose(resolution_probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        detector = VideoEmanationDetector(input_dim=64)
        detector.eval()
        for batch_size in [2, 4, 8, 16]:
            x = torch.randn(batch_size, 64)
            with torch.no_grad():
                reconstruction_score, resolution_probs = detector.forward(x)
            assert reconstruction_score.shape == (batch_size, 1)
            assert resolution_probs.shape == (batch_size, 3)


class TestSideChannelVulnerabilityAssessor:
    """Tests for SideChannelVulnerabilityAssessor class."""

    def test_init(self):
        """Test initialization."""
        assessor = SideChannelVulnerabilityAssessor()
        assert assessor.logger is not None

    def test_assess_vulnerabilities_no_issues(self):
        """Test assessment with no vulnerabilities."""
        assessor = SideChannelVulnerabilityAssessor()
        equipment_data = {
            "em_shielding_db": 80.0,
            "power_line_filtering": True,
            "cable_shielding": True,
            "distance_to_boundary_m": 25.0,
        }
        result = assessor.assess_vulnerabilities(equipment_data)
        assert not result["vulnerabilities_detected"]
        assert len(result["vulnerabilities"]) == 0

    def test_assess_vulnerabilities_with_issues(self):
        """Test assessment with vulnerabilities."""
        assessor = SideChannelVulnerabilityAssessor()
        equipment_data = {
            "em_shielding_db": 20.0,
            "power_line_filtering": False,
            "cable_shielding": False,
            "distance_to_boundary_m": 5.0,
        }
        result = assessor.assess_vulnerabilities(equipment_data)
        assert result["vulnerabilities_detected"]
        assert "insufficient_em_shielding" in result["vulnerabilities"]
        assert "unfiltered_power_lines" in result["vulnerabilities"]
        assert "unshielded_cables" in result["vulnerabilities"]
        assert "insufficient_control_zone" in result["vulnerabilities"]

    def test_check_compliance(self):
        """Test compliance checking."""
        assessor = SideChannelVulnerabilityAssessor()
        equipment_data = {
            "em_shielding_db": 70.0,
            "power_line_filtering": True,
            "cable_shielding": True,
            "distance_to_boundary_m": 25.0,
        }
        compliance = assessor._check_compliance(equipment_data)
        assert not compliance["zone1_shielding"]  # 70 < 80
        assert compliance["zone2_shielding"]  # 70 >= 60
        assert compliance["zone3_shielding"]  # 70 >= 40
        assert compliance["power_line_filtering"]
        assert compliance["control_zone"]


class TestEMSECCountermeasureGenerator:
    """Tests for EMSECCountermeasureGenerator class."""

    def test_init(self):
        """Test initialization."""
        generator = EMSECCountermeasureGenerator()
        assert generator.logger is not None

    def test_generate_countermeasures_empty(self):
        """Test countermeasure generation with no issues."""
        generator = EMSECCountermeasureGenerator()
        countermeasures = generator.generate_countermeasures({}, [])
        assert len(countermeasures) >= 2  # Always includes testing and inventory

    def test_generate_countermeasures_video(self):
        """Test countermeasures for video emanation."""
        generator = EMSECCountermeasureGenerator()
        analysis_result = {"emanation_sources": ["video_display_emanation"]}
        countermeasures = generator.generate_countermeasures(analysis_result, [])
        assert any("TEMPEST-certified displays" in c for c in countermeasures)

    def test_generate_countermeasures_keyboard(self):
        """Test countermeasures for keyboard emanation."""
        generator = EMSECCountermeasureGenerator()
        analysis_result = {"emanation_sources": ["keyboard_emanation"]}
        countermeasures = generator.generate_countermeasures(analysis_result, [])
        assert any("keyboard" in c.lower() for c in countermeasures)

    def test_generate_countermeasures_vulnerabilities(self):
        """Test countermeasures for vulnerabilities."""
        generator = EMSECCountermeasureGenerator()
        vulnerabilities = ["insufficient_em_shielding", "unfiltered_power_lines"]
        countermeasures = generator.generate_countermeasures({}, vulnerabilities)
        assert any("shielding" in c.lower() for c in countermeasures)
        assert any("power" in c.lower() for c in countermeasures)


class TestTEMPESTDetector:
    """Tests for TEMPESTDetector class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        detector = TEMPESTDetector()
        assert detector.enable_rf_analysis
        assert detector.enable_video_detection
        assert detector.enable_vulnerability_assessment
        assert detector.rf_analyzer is not None
        assert detector.video_detector is not None
        assert detector.vulnerability_assessor is not None

    def test_init_disabled_components(self):
        """Test initialization with disabled components."""
        detector = TEMPESTDetector(
            enable_rf_analysis=False,
            enable_video_detection=False,
            enable_vulnerability_assessment=False,
        )
        assert detector.rf_analyzer is None
        assert detector.video_detector is None
        assert detector.vulnerability_assessor is None

    def test_detect_tempest_threats_empty(self):
        """Test detection with empty data."""
        detector = TEMPESTDetector()
        result = detector.detect_tempest_threats({})
        assert isinstance(result, TEMPESTAnalysisResult)
        assert not result.emanation_detected

    def test_detect_tempest_threats_with_spectrum(self):
        """Test detection with spectrum data."""
        detector = TEMPESTDetector()
        tempest_data = {
            "spectrum_data": {
                "frequencies": [50e6, 100e6, 150e6],
                "power_dbm": [-50.0, -40.0, -60.0],
            }
        }
        result = detector.detect_tempest_threats(tempest_data)
        assert isinstance(result, TEMPESTAnalysisResult)

    def test_detect_tempest_threats_with_equipment(self):
        """Test detection with equipment data."""
        detector = TEMPESTDetector()
        tempest_data = {
            "equipment_data": {
                "em_shielding_db": 30.0,
                "power_line_filtering": False,
                "cable_shielding": False,
                "distance_to_boundary_m": 5.0,
            }
        }
        result = detector.detect_tempest_threats(tempest_data)
        assert isinstance(result, TEMPESTAnalysisResult)
        assert len(result.vulnerable_equipment) > 0
        assert len(result.countermeasures) > 0

    def test_detect_tempest_threats_with_video(self):
        """Test detection with video emanation features."""
        detector = TEMPESTDetector()
        tempest_data = {"video_emanation_features": np.random.randn(128).astype(np.float32)}
        result = detector.detect_tempest_threats(tempest_data)
        assert isinstance(result, TEMPESTAnalysisResult)
        assert result.reconstruction_feasibility >= 0

    def test_detect_tempest_threats_comprehensive(self):
        """Test comprehensive detection with all data types."""
        detector = TEMPESTDetector()
        tempest_data = {
            "spectrum_data": {
                "frequencies": [50e6, 100e6, 150e6],
                "power_dbm": [-50.0, -40.0, -60.0],
            },
            "equipment_data": {
                "em_shielding_db": 30.0,
                "power_line_filtering": False,
                "cable_shielding": False,
                "distance_to_boundary_m": 5.0,
            },
            "video_emanation_features": np.random.randn(128).astype(np.float32),
        }
        result = detector.detect_tempest_threats(tempest_data)
        assert isinstance(result, TEMPESTAnalysisResult)
        assert result.threat_level in [
            "no_threat",
            "low_risk",
            "moderate_risk",
            "high_risk",
            "critical_risk",
        ]
        assert len(result.countermeasures) > 0
