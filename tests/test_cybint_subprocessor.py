"""
Tests for CYBINT Sub-Processor module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC
"""

import numpy as np
import pytest
import torch

from omni_mercury_engine.security.cybint_subprocessor import (
    APTGroup,
    APTPatternRecognizer,
    C2InfrastructureDetector,
    CyberKillChainStage,
    CYBINTAnalysisResult,
    CYBINTSubProcessor,
    MalwareFamily,
    MalwareTaxonomyClassifier,
    ThreatActorType,
    ZeroDayIndicatorAnalyzer,
)


class TestAPTGroup:
    """Tests for APTGroup enum."""

    def test_enum_values(self):
        """Test enum values exist."""
        assert APTGroup.APT1.value == "apt1_comment_crew"
        assert APTGroup.APT28.value == "apt28_fancy_bear"
        assert APTGroup.APT29.value == "apt29_cozy_bear"
        assert APTGroup.APT38.value == "apt38_lazarus"
        assert APTGroup.UNKNOWN.value == "unknown_apt"


class TestMalwareFamily:
    """Tests for MalwareFamily enum."""

    def test_enum_values(self):
        """Test enum values exist."""
        assert MalwareFamily.RANSOMWARE_WANNACRY.value == "wannacry"
        assert MalwareFamily.TROJAN_EMOTET.value == "emotet"
        assert MalwareFamily.RAT_COBALT_STRIKE.value == "cobalt_strike"
        assert MalwareFamily.UNKNOWN.value == "unknown_malware"


class TestCyberKillChainStage:
    """Tests for CyberKillChainStage enum."""

    def test_enum_values(self):
        """Test enum values exist."""
        assert CyberKillChainStage.RECONNAISSANCE.value == "reconnaissance"
        assert CyberKillChainStage.WEAPONIZATION.value == "weaponization"
        assert CyberKillChainStage.DELIVERY.value == "delivery"
        assert CyberKillChainStage.EXPLOITATION.value == "exploitation"
        assert CyberKillChainStage.COMMAND_CONTROL.value == "command_and_control"


class TestThreatActorType:
    """Tests for ThreatActorType enum."""

    def test_enum_values(self):
        """Test enum values exist."""
        assert ThreatActorType.NATION_STATE.value == "nation_state"
        assert ThreatActorType.CYBERCRIME.value == "cybercriminal"
        assert ThreatActorType.HACKTIVIST.value == "hacktivist"
        assert ThreatActorType.INSIDER.value == "insider_threat"


class TestCYBINTAnalysisResult:
    """Tests for CYBINTAnalysisResult dataclass."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        result = CYBINTAnalysisResult(
            threat_detected=False,
            confidence=0.0,
            threat_severity="low",
            risk_score=0.0,
        )
        assert result.threat_detected == False
        assert result.apt_group is None
        assert result.ttps_detected == []

    def test_init_full(self):
        """Test initialization with all parameters."""
        result = CYBINTAnalysisResult(
            threat_detected=True,
            confidence=0.9,
            threat_severity="critical",
            risk_score=0.95,
            apt_group="apt28_fancy_bear",
            malware_family="emotet",
            kill_chain_stage="command_and_control",
            threat_actor_type="nation_state",
            ttps_detected=["T1059", "T1071"],
            iocs={"ip": ["192.168.1.1"]},
            c2_indicators=["beaconing"],
            zero_day_likelihood=0.8,
            attribution_confidence=0.85,
            recommended_actions=["isolate"],
            defensive_measures=["block"],
        )
        assert result.threat_detected == True
        assert result.apt_group == "apt28_fancy_bear"
        assert len(result.ttps_detected) == 2


class TestAPTPatternRecognizer:
    """Tests for APTPatternRecognizer class."""

    def test_init(self):
        """Test initialization."""
        recognizer = APTPatternRecognizer(input_dim=256)
        assert recognizer.pattern_encoder is not None
        assert recognizer.apt_classifier is not None
        assert recognizer.confidence_head is not None

    def test_forward(self):
        """Test forward pass."""
        recognizer = APTPatternRecognizer(input_dim=128)
        recognizer.eval()
        x = torch.randn(4, 128)
        with torch.no_grad():
            apt_logits, confidence = recognizer.forward(x)
        assert apt_logits.shape == (4, len(APTGroup))
        assert confidence.shape == (4, 1)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        recognizer = APTPatternRecognizer(input_dim=128)
        recognizer.eval()
        for batch_size in [2, 4, 8, 16]:
            x = torch.randn(batch_size, 128)
            with torch.no_grad():
                apt_logits, confidence = recognizer.forward(x)
            assert apt_logits.shape == (batch_size, len(APTGroup))
            assert confidence.shape == (batch_size, 1)


class TestMalwareTaxonomyClassifier:
    """Tests for MalwareTaxonomyClassifier class."""

    def test_init(self):
        """Test initialization."""
        classifier = MalwareTaxonomyClassifier(input_dim=128)
        assert classifier.feature_extractor is not None
        assert classifier.family_classifier is not None

    def test_forward(self):
        """Test forward pass."""
        classifier = MalwareTaxonomyClassifier(input_dim=64)
        classifier.eval()
        x = torch.randn(4, 64)
        with torch.no_grad():
            classification = classifier.forward(x)
        assert classification.shape == (4, len(MalwareFamily))

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        classifier = MalwareTaxonomyClassifier(input_dim=64)
        classifier.eval()
        for batch_size in [2, 4, 8, 16]:
            x = torch.randn(batch_size, 64)
            with torch.no_grad():
                classification = classifier.forward(x)
            assert classification.shape == (batch_size, len(MalwareFamily))


class TestC2InfrastructureDetector:
    """Tests for C2InfrastructureDetector class."""

    def test_init(self):
        """Test initialization."""
        detector = C2InfrastructureDetector()
        assert "http_beacon" in detector.c2_signatures
        assert "dns_tunneling" in detector.c2_signatures

    def test_detect_c2_empty(self):
        """Test C2 detection with empty data."""
        detector = C2InfrastructureDetector()
        result = detector.detect_c2({})
        assert result["c2_detected"] == False
        assert result["c2_indicators"] == []

    def test_detect_c2_with_beaconing(self):
        """Test C2 detection with beaconing pattern."""
        detector = C2InfrastructureDetector()
        network_data = {
            "connection_intervals": [60, 60, 60, 60, 60, 60],  # Regular intervals
        }
        result = detector.detect_c2(network_data)
        assert result["c2_detected"] == True
        assert "beaconing_activity" in result["c2_indicators"]

    def test_detect_c2_with_dga(self):
        """Test C2 detection with DGA domains."""
        detector = C2InfrastructureDetector()
        network_data = {
            "queried_domains": [
                "xyzabc123def.com",
                "qwerty789xyz.net",
                "abcdefghijkl.org",
            ],
        }
        result = detector.detect_c2(network_data)
        assert "c2_detected" in result

    def test_detect_c2_with_signature_match(self):
        """Test C2 detection with signature match."""
        detector = C2InfrastructureDetector()
        network_data = {
            "protocol": "HTTP",
            "patterns": ["periodic_callback"],
        }
        result = detector.detect_c2(network_data)
        assert result["c2_detected"] == True
        assert "http_beacon" in result["c2_indicators"]

    def test_calculate_entropy(self):
        """Test entropy calculation."""
        detector = C2InfrastructureDetector()
        entropy = detector._calculate_entropy("aaaaaa")
        assert entropy == 0.0  # All same characters

        entropy = detector._calculate_entropy("abcdef")
        assert entropy > 0  # Different characters

    def test_generate_c2_recommendations(self):
        """Test C2 recommendation generation."""
        detector = C2InfrastructureDetector()
        indicators = ["beaconing_activity", "domain_generation_algorithm"]
        recs = detector._generate_c2_recommendations(indicators)
        assert len(recs) > 0
        assert any("beacon" in r.lower() for r in recs)
        assert any("dga" in r.lower() for r in recs)


class TestZeroDayIndicatorAnalyzer:
    """Tests for ZeroDayIndicatorAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = ZeroDayIndicatorAnalyzer()
        assert analyzer.logger is not None

    def test_analyze_zero_day_no_indicators(self):
        """Test analysis with no zero-day indicators."""
        analyzer = ZeroDayIndicatorAnalyzer()
        exploit_data = {
            "cve_id": "CVE-2024-1234",
            "technique_novelty_score": 0.3,
            "signature_matches": 5,
            "exploitation_successful": False,
        }
        result = analyzer.analyze_zero_day_likelihood(exploit_data)
        assert result["zero_day_likelihood"] == 0.0
        assert len(result["indicators"]) == 0

    def test_analyze_zero_day_high_likelihood(self):
        """Test analysis with high zero-day likelihood."""
        analyzer = ZeroDayIndicatorAnalyzer()
        exploit_data = {
            "cve_id": None,  # Unknown vulnerability
            "technique_novelty_score": 0.9,
            "signature_matches": 0,
            "exploitation_successful": True,
        }
        result = analyzer.analyze_zero_day_likelihood(exploit_data)
        assert result["zero_day_likelihood"] > 0.7
        assert result["priority"] == "critical"
        assert "unknown_vulnerability" in result["indicators"]
        assert "novel_exploitation_technique" in result["indicators"]

    def test_generate_zero_day_recommendations(self):
        """Test zero-day recommendation generation."""
        analyzer = ZeroDayIndicatorAnalyzer()
        recs_critical = analyzer._generate_zero_day_recommendations(0.8)
        assert len(recs_critical) > 0
        assert any("critical" in r.lower() for r in recs_critical)

        recs_high = analyzer._generate_zero_day_recommendations(0.6)
        assert len(recs_high) > 0


class TestCYBINTSubProcessor:
    """Tests for CYBINTSubProcessor class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        processor = CYBINTSubProcessor()
        assert processor.enable_apt_attribution == True
        assert processor.enable_malware_classification == True
        assert processor.enable_c2_detection == True
        assert processor.enable_zero_day_analysis == True
        assert processor.apt_recognizer is not None
        assert processor.malware_classifier is not None
        assert processor.c2_detector is not None
        assert processor.zero_day_analyzer is not None

    def test_init_disabled_components(self):
        """Test initialization with disabled components."""
        processor = CYBINTSubProcessor(
            enable_apt_attribution=False,
            enable_malware_classification=False,
            enable_c2_detection=False,
            enable_zero_day_analysis=False,
        )
        assert processor.apt_recognizer is None
        assert processor.malware_classifier is None
        assert processor.c2_detector is None
        assert processor.zero_day_analyzer is None

    def test_process_cybint_empty(self):
        """Test processing with empty data."""
        processor = CYBINTSubProcessor()
        result = processor.process_cybint({})
        assert isinstance(result, CYBINTAnalysisResult)
        assert result.threat_detected == False

    def test_process_cybint_with_threat_features(self):
        """Test processing with threat features."""
        processor = CYBINTSubProcessor()
        threat_data = {
            "threat_features": np.random.randn(256).astype(np.float32),
        }
        result = processor.process_cybint(threat_data)
        assert isinstance(result, CYBINTAnalysisResult)
        assert result.apt_group is not None

    def test_process_cybint_with_malware_features(self):
        """Test processing with malware features."""
        processor = CYBINTSubProcessor()
        threat_data = {
            "malware_features": np.random.randn(128).astype(np.float32),
        }
        result = processor.process_cybint(threat_data)
        assert isinstance(result, CYBINTAnalysisResult)
        assert result.malware_family is not None

    def test_process_cybint_with_network_data(self):
        """Test processing with network data."""
        processor = CYBINTSubProcessor()
        threat_data = {
            "network_data": {
                "connection_intervals": [60, 60, 60, 60, 60, 60],
            },
        }
        result = processor.process_cybint(threat_data)
        assert isinstance(result, CYBINTAnalysisResult)
        assert len(result.c2_indicators) > 0

    def test_process_cybint_with_exploit_data(self):
        """Test processing with exploit data."""
        processor = CYBINTSubProcessor()
        threat_data = {
            "exploit_data": {
                "cve_id": None,
                "technique_novelty_score": 0.9,
                "signature_matches": 0,
                "exploitation_successful": True,
            },
        }
        result = processor.process_cybint(threat_data)
        assert isinstance(result, CYBINTAnalysisResult)
        assert result.zero_day_likelihood > 0

    def test_process_cybint_with_ttps(self):
        """Test processing with TTPs."""
        processor = CYBINTSubProcessor()
        threat_data = {
            "ttps": ["T1059", "T1071", "T1082"],
        }
        result = processor.process_cybint(threat_data)
        assert result.ttps_detected == ["T1059", "T1071", "T1082"]

    def test_process_cybint_with_iocs(self):
        """Test processing with IOCs."""
        processor = CYBINTSubProcessor()
        threat_data = {
            "iocs": {
                "ip": ["192.168.1.1", "10.0.0.1"],
                "domain": ["malware.com"],
            },
        }
        result = processor.process_cybint(threat_data)
        assert result.iocs == threat_data["iocs"]

    def test_process_cybint_comprehensive(self):
        """Test comprehensive processing with all data types."""
        processor = CYBINTSubProcessor()
        threat_data = {
            "threat_features": np.random.randn(256).astype(np.float32),
            "malware_features": np.random.randn(128).astype(np.float32),
            "network_data": {
                "connection_intervals": [60, 60, 60, 60, 60, 60],
            },
            "exploit_data": {
                "cve_id": None,
                "technique_novelty_score": 0.9,
                "signature_matches": 0,
                "exploitation_successful": True,
            },
            "ttps": ["T1059", "T1071"],
            "iocs": {"ip": ["192.168.1.1"]},
        }
        result = processor.process_cybint(threat_data)
        assert isinstance(result, CYBINTAnalysisResult)
        assert result.apt_group is not None
        assert result.malware_family is not None
        assert len(result.c2_indicators) > 0
        assert result.zero_day_likelihood > 0
        assert len(result.defensive_measures) > 0
