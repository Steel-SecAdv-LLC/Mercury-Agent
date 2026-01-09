"""
Tests for Traffic Analysis module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC
"""

import numpy as np
import pytest
import torch

from omni_mercury_engine.security.traffic_analysis import (
    CommunicationGraphAnalyzer,
    CovertChannelDetector,
    EncryptedTrafficFingerprinter,
    NetworkFlowAnalyzer,
    TrafficAnalysisEngine,
    TrafficAnalysisResult,
    TrafficAnomalyType,
)


class TestTrafficAnomalyType:
    """Tests for TrafficAnomalyType enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert TrafficAnomalyType.NORMAL.value == "normal_traffic"
        assert TrafficAnomalyType.PORT_SCAN.value == "port_scanning"
        assert TrafficAnomalyType.DDoS.value == "distributed_denial_of_service"
        assert TrafficAnomalyType.DATA_EXFILTRATION.value == "data_exfiltration"
        assert TrafficAnomalyType.COVERT_CHANNEL.value == "covert_channel"


class TestTrafficAnalysisResult:
    """Tests for TrafficAnalysisResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = TrafficAnalysisResult(
            anomaly_detected=False,
            confidence=0.5,
            anomaly_type="normal_traffic",
            risk_score=0.0,
        )
        assert result.anomaly_detected is False
        assert result.flow_statistics == {}
        assert result.covert_channels == []

    def test_custom_values(self) -> None:
        """Test custom values."""
        result = TrafficAnalysisResult(
            anomaly_detected=True,
            confidence=0.9,
            anomaly_type="port_scanning",
            risk_score=0.8,
            protocol_anomalies=["suspicious_protocol"],
        )
        assert result.anomaly_detected is True
        assert "suspicious_protocol" in result.protocol_anomalies


class TestNetworkFlowAnalyzer:
    """Tests for NetworkFlowAnalyzer class."""

    def test_init(self) -> None:
        """Test initialization."""
        analyzer = NetworkFlowAnalyzer()
        assert analyzer is not None

    def test_analyze_empty_flows(self) -> None:
        """Test analysis with empty flow data."""
        analyzer = NetworkFlowAnalyzer()
        result = analyzer.analyze_flows([])
        assert result["anomaly_detected"] is False

    def test_analyze_normal_flows(self) -> None:
        """Test analysis with normal flow data."""
        analyzer = NetworkFlowAnalyzer()
        flow_data = [
            {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.1", "dst_port": 80, "protocol": "TCP", "bytes": 1000},
            {"src_ip": "192.168.1.2", "dst_ip": "10.0.0.1", "dst_port": 443, "protocol": "TCP", "bytes": 2000},
        ]
        result = analyzer.analyze_flows(flow_data)
        assert "statistics" in result
        assert result["statistics"]["total_flows"] == 2

    def test_detect_port_scanning(self) -> None:
        """Test port scanning detection."""
        analyzer = NetworkFlowAnalyzer()
        flow_data = []
        for port in range(1, 100):
            flow_data.append({
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "dst_port": port,
                "protocol": "TCP",
                "bytes": 100,
            })
        result = analyzer.analyze_flows(flow_data)
        assert "port_scanning" in result.get("anomalies", [])

    def test_detect_ddos(self) -> None:
        """Test DDoS detection."""
        analyzer = NetworkFlowAnalyzer()
        flow_data = []
        for i in range(200):
            flow_data.append({
                "src_ip": f"192.168.1.{i % 255}",
                "dst_ip": "10.0.0.1",
                "dst_port": 80,
                "protocol": "TCP",
                "bytes": 1000,
            })
        result = analyzer.analyze_flows(flow_data)
        assert "statistics" in result

    def test_detect_data_exfiltration(self) -> None:
        """Test data exfiltration detection."""
        analyzer = NetworkFlowAnalyzer()
        flow_data = []
        for i in range(20):
            flow_data.append({
                "src_ip": "192.168.1.1",
                "dst_ip": "10.0.0.1",
                "dst_port": 443,
                "protocol": "TCP",
                "bytes": 10000000,
                "direction": "outbound",
            })
        result = analyzer.analyze_flows(flow_data)
        assert "statistics" in result

    def test_identify_suspicious_sources(self) -> None:
        """Test suspicious source identification."""
        analyzer = NetworkFlowAnalyzer()
        flow_data = []
        for port in range(1, 150):
            flow_data.append({
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "dst_port": port,
                "protocol": "TCP",
                "bytes": 100,
            })
        result = analyzer.analyze_flows(flow_data)
        assert "suspicious_sources" in result


class TestCommunicationGraphAnalyzer:
    """Tests for CommunicationGraphAnalyzer class."""

    def test_init(self) -> None:
        """Test initialization."""
        analyzer = CommunicationGraphAnalyzer()
        assert isinstance(analyzer, torch.nn.Module)

    def test_init_custom_dims(self) -> None:
        """Test initialization with custom dimensions."""
        analyzer = CommunicationGraphAnalyzer(node_feature_dim=128, hidden_dim=256)
        assert isinstance(analyzer, torch.nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        analyzer = CommunicationGraphAnalyzer(node_feature_dim=64, hidden_dim=128)
        node_features = torch.randn(10, 64)
        adjacency_matrix = torch.eye(10)
        output = analyzer(node_features, adjacency_matrix)
        assert output.shape == (10, 1)

    def test_output_range(self) -> None:
        """Test output is in [0, 1] range (sigmoid)."""
        analyzer = CommunicationGraphAnalyzer(node_feature_dim=64, hidden_dim=128)
        node_features = torch.randn(10, 64)
        adjacency_matrix = torch.eye(10)
        output = analyzer(node_features, adjacency_matrix)
        assert (output >= 0).all() and (output <= 1).all()


class TestEncryptedTrafficFingerprinter:
    """Tests for EncryptedTrafficFingerprinter class."""

    def test_init(self) -> None:
        """Test initialization."""
        fingerprinter = EncryptedTrafficFingerprinter()
        assert fingerprinter is not None

    def test_fingerprint_tls_basic(self) -> None:
        """Test basic TLS fingerprinting."""
        fingerprinter = EncryptedTrafficFingerprinter()
        tls_handshake = {
            "tls_version": "TLS1.3",
            "cipher_suites": ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"],
            "extensions": [0, 10, 11, 13, 43, 51],
            "elliptic_curves": [29, 23, 24],
        }
        result = fingerprinter.fingerprint_tls(tls_handshake)
        assert "fingerprint_hash" in result
        assert "tls_version" in result
        assert result["tls_version"] == "TLS1.3"

    def test_fingerprint_suspicious_tls(self) -> None:
        """Test fingerprinting suspicious TLS."""
        fingerprinter = EncryptedTrafficFingerprinter()
        tls_handshake = {
            "tls_version": "SSLv3",
            "cipher_suites": ["NULL_WITH_NULL_NULL", "EXPORT_RSA"],
            "extensions": list(range(30)),
        }
        result = fingerprinter.fingerprint_tls(tls_handshake)
        assert result["is_suspicious"] is True

    def test_identify_risk_indicators(self) -> None:
        """Test risk indicator identification."""
        fingerprinter = EncryptedTrafficFingerprinter()
        tls_handshake = {
            "tls_version": "TLS1.0",
            "cipher_suites": ["NULL_CIPHER"],
            "sni_mismatch": True,
        }
        result = fingerprinter.fingerprint_tls(tls_handshake)
        assert len(result["risk_indicators"]) > 0


class TestCovertChannelDetector:
    """Tests for CovertChannelDetector class."""

    def test_init(self) -> None:
        """Test initialization."""
        detector = CovertChannelDetector()
        assert detector is not None

    def test_detect_no_covert_channels(self) -> None:
        """Test detection with no covert channels."""
        detector = CovertChannelDetector()
        traffic_sample = {
            "packet_timestamps": [1.0, 2.0, 3.0, 4.0, 5.0],
            "packet_sizes": [100, 100, 100, 100, 100],
        }
        result = detector.detect_covert_channels(traffic_sample)
        assert "covert_channels_detected" in result

    def test_detect_timing_channel(self) -> None:
        """Test timing channel detection."""
        detector = CovertChannelDetector()
        traffic_sample = {
            "packet_timestamps": list(np.cumsum(np.random.uniform(0.1, 0.2, 20))),
            "packet_sizes": [100] * 20,
        }
        result = detector.detect_covert_channels(traffic_sample)
        assert "channels" in result

    def test_detect_storage_channel(self) -> None:
        """Test storage channel detection."""
        detector = CovertChannelDetector()
        traffic_sample = {
            "packet_timestamps": list(range(20)),
            "packet_sizes": list(np.random.randint(100, 10000, 20)),
        }
        result = detector.detect_covert_channels(traffic_sample)
        assert "channels" in result

    def test_detect_protocol_field_manipulation(self) -> None:
        """Test protocol field manipulation detection."""
        detector = CovertChannelDetector()
        traffic_sample = {
            "packet_timestamps": list(range(20)),
            "packet_sizes": [100] * 20,
            "protocol_fields": {
                "ip_id": list(np.random.randint(0, 65535, 20)),
                "tcp_seq": list(range(1000, 1020)),
            },
        }
        result = detector.detect_covert_channels(traffic_sample)
        assert "channels" in result

    def test_calculate_entropy(self) -> None:
        """Test entropy calculation."""
        detector = CovertChannelDetector()
        uniform_data = [1, 2, 3, 4, 5, 6, 7, 8]
        entropy = detector._calculate_entropy(uniform_data)
        assert entropy > 0

    def test_calculate_entropy_empty(self) -> None:
        """Test entropy calculation with empty data."""
        detector = CovertChannelDetector()
        entropy = detector._calculate_entropy([])
        assert entropy == 0.0


class TestTrafficAnalysisEngine:
    """Tests for TrafficAnalysisEngine class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        engine = TrafficAnalysisEngine()
        assert engine.enable_flow_analysis is True
        assert engine.enable_graph_analysis is True
        assert engine.enable_tls_fingerprinting is True
        assert engine.enable_covert_detection is True

    def test_init_disabled_components(self) -> None:
        """Test initialization with disabled components."""
        engine = TrafficAnalysisEngine(
            enable_flow_analysis=False,
            enable_graph_analysis=False,
            enable_tls_fingerprinting=False,
            enable_covert_detection=False,
        )
        assert engine.flow_analyzer is None
        assert engine.graph_analyzer is None
        assert engine.tls_fingerprinter is None
        assert engine.covert_detector is None

    def test_analyze_empty_traffic(self) -> None:
        """Test analysis with empty traffic data."""
        engine = TrafficAnalysisEngine()
        result = engine.analyze_traffic({})
        assert result.anomaly_detected is False
        assert result.anomaly_type == "normal_traffic"

    def test_analyze_with_flow_records(self) -> None:
        """Test analysis with flow records."""
        engine = TrafficAnalysisEngine()
        traffic_data = {
            "flow_records": [
                {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.1", "dst_port": 80, "protocol": "TCP", "bytes": 1000},
            ]
        }
        result = engine.analyze_traffic(traffic_data)
        assert result.flow_statistics is not None

    def test_analyze_with_tls_handshakes(self) -> None:
        """Test analysis with TLS handshakes."""
        engine = TrafficAnalysisEngine()
        traffic_data = {
            "tls_handshakes": [
                {
                    "tls_version": "TLS1.3",
                    "cipher_suites": ["TLS_AES_256_GCM_SHA384"],
                    "extensions": [0, 10, 11],
                }
            ]
        }
        result = engine.analyze_traffic(traffic_data)
        assert isinstance(result, TrafficAnalysisResult)

    def test_analyze_with_raw_traffic(self) -> None:
        """Test analysis with raw traffic."""
        engine = TrafficAnalysisEngine()
        traffic_data = {
            "raw_traffic": {
                "packet_timestamps": list(range(20)),
                "packet_sizes": [100] * 20,
            }
        }
        result = engine.analyze_traffic(traffic_data)
        assert isinstance(result, TrafficAnalysisResult)
