"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Traffic Analysis Module - Network Flow & Communication Pattern Analysis

Advanced network traffic intelligence for SIGINT/COMINT fusion:
- Network flow graph analysis
- Communication pattern detection
- Protocol anomaly identification
- Encrypted traffic fingerprinting
- Covert channel detection
- Traffic correlation and attribution

⚠️ SIMULATION-BASED: Research/development tool for traffic analysis patterns.
Operational deployment requires legal authorization and privacy compliance.

Research sources:
- NetFlow/IPFIX analysis methodologies
- TLS fingerprinting research (JA3/JA4)
- Covert channel detection literature
- Graph theory for network analysis

"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class TrafficAnomalyType(Enum):
    """Network traffic anomaly classifications"""

    NORMAL = "normal_traffic"
    PORT_SCAN = "port_scanning"
    DDoS = "distributed_denial_of_service"
    DATA_EXFILTRATION = "data_exfiltration"
    COVERT_CHANNEL = "covert_channel"
    PROTOCOL_ANOMALY = "protocol_anomaly"
    ENCRYPTED_TUNNEL = "encrypted_tunnel"
    BOTNET_C2 = "botnet_command_control"


@dataclass
class TrafficAnalysisResult:
    """Traffic analysis results"""

    anomaly_detected: bool
    confidence: float
    anomaly_type: str
    risk_score: float

    flow_statistics: dict[str, Any] = field(default_factory=dict)
    communication_graph: dict[str, Any] = field(default_factory=dict)
    protocol_anomalies: list[str] = field(default_factory=list)
    encrypted_flows: list[dict] = field(default_factory=list)
    covert_channels: list[str] = field(default_factory=list)

    attribution_indicators: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)


class NetworkFlowAnalyzer:
    """
    Network flow statistical analysis.

    Analyzes NetFlow/IPFIX data for anomalous patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_flows(self, flow_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze network flows for anomalies.

        Args:
            flow_data: List of network flow records

        Returns:
            Flow analysis with anomaly detection
        """
        if not flow_data:
            return {"anomaly_detected": False, "statistics": {}}

        flows_per_src = defaultdict(int)
        flows_per_dst = defaultdict(int)
        ports_accessed = defaultdict(set)
        protocols = defaultdict(int)
        byte_volumes = []

        for flow in flow_data:
            src = flow.get("src_ip")
            dst = flow.get("dst_ip")
            dst_port = flow.get("dst_port")
            protocol = flow.get("protocol")
            bytes_transferred = flow.get("bytes", 0)

            flows_per_src[src] += 1
            flows_per_dst[dst] += 1
            ports_accessed[src].add(dst_port)
            protocols[protocol] += 1
            byte_volumes.append(bytes_transferred)

        port_scan_detected = self._detect_port_scanning(flows_per_src, ports_accessed)
        ddos_detected = self._detect_ddos(flows_per_dst)
        exfiltration_detected = self._detect_data_exfiltration(byte_volumes, flow_data)

        statistics = {
            "total_flows": len(flow_data),
            "unique_sources": len(flows_per_src),
            "unique_destinations": len(flows_per_dst),
            "total_bytes": sum(byte_volumes),
            "avg_bytes_per_flow": np.mean(byte_volumes) if byte_volumes else 0,
            "protocol_distribution": dict(protocols),
        }

        anomalies = []
        if port_scan_detected:
            anomalies.append("port_scanning")
        if ddos_detected:
            anomalies.append("ddos_attack")
        if exfiltration_detected:
            anomalies.append("data_exfiltration")

        return {
            "anomaly_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "statistics": statistics,
            "suspicious_sources": self._identify_suspicious_sources(flows_per_src, ports_accessed),
        }

    def _detect_port_scanning(self, flows_per_src: dict, ports_accessed: dict[str, set]) -> bool:
        """Detect port scanning activity"""
        for src_ip, port_set in ports_accessed.items():
            if len(port_set) > 20 and flows_per_src[src_ip] > 50:
                return True
        return False

    def _detect_ddos(self, flows_per_dst: dict) -> bool:
        """Detect DDoS attack patterns"""
        if not flows_per_dst:
            return False

        max_flows = max(flows_per_dst.values())
        avg_flows = np.mean(list(flows_per_dst.values()))

        return max_flows > avg_flows * 10 and max_flows > 100

    def _detect_data_exfiltration(self, byte_volumes: list[int], flow_data: list[dict]) -> bool:
        """Detect data exfiltration patterns"""
        if not byte_volumes:
            return False

        outbound_volumes = []
        for flow in flow_data:
            if flow.get("direction") == "outbound":
                outbound_volumes.append(flow.get("bytes", 0))

        if outbound_volumes:
            total_outbound = sum(outbound_volumes)
            total_volume = sum(byte_volumes)

            outbound_ratio = total_outbound / total_volume if total_volume > 0 else 0

            return outbound_ratio > 0.8 and total_outbound > 10**7

        return False

    def _identify_suspicious_sources(
        self, flows_per_src: dict, ports_accessed: dict[str, set]
    ) -> list[str]:
        """Identify suspicious source IPs"""
        suspicious = []

        for src_ip, flow_count in flows_per_src.items():
            if flow_count > 100 and len(ports_accessed.get(src_ip, set())) > 10:
                suspicious.append(src_ip)

        return suspicious


class CommunicationGraphAnalyzer(nn.Module):
    """
    Graph neural network for communication pattern analysis.

    Models network communications as a graph and detects anomalous patterns.
    """

    def __init__(self, node_feature_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.graph_conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_conv2 = nn.Linear(hidden_dim, hidden_dim)

        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for graph-based anomaly detection.

        Args:
            node_features: Node feature matrix (N, feature_dim)
            adjacency_matrix: Adjacency matrix (N, N)

        Returns:
            Anomaly scores for each node
        """
        x = self.node_encoder(node_features)

        x = torch.relu(self.graph_conv1(torch.mm(adjacency_matrix, x)))
        x = torch.relu(self.graph_conv2(torch.mm(adjacency_matrix, x)))

        anomaly_scores = self.anomaly_detector(x)

        return anomaly_scores


class EncryptedTrafficFingerprinter:
    """
    Encrypted traffic fingerprinting (JA3/JA4-style analysis).

    Identifies encrypted traffic patterns without decryption.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fingerprint_tls(self, tls_handshake: dict[str, Any]) -> dict[str, Any]:
        """
        Generate TLS fingerprint from handshake.

        Args:
            tls_handshake: TLS handshake parameters

        Returns:
            TLS fingerprint and analysis
        """
        tls_version = tls_handshake.get("tls_version", "")
        cipher_suites = tls_handshake.get("cipher_suites", [])
        extensions = tls_handshake.get("extensions", [])
        curves = tls_handshake.get("elliptic_curves", [])

        cipher_str = ",".join(map(str, cipher_suites))
        ext_str = ",".join(map(str, extensions))
        curve_str = ",".join(map(str, curves))
        fingerprint = f"{tls_version}_{cipher_str}_{ext_str}_{curve_str}"

        fingerprint_hash = hash(fingerprint) % (10**8)

        is_suspicious = self._analyze_tls_parameters(tls_handshake)

        return {
            "fingerprint_hash": fingerprint_hash,
            "fingerprint": fingerprint[:100],
            "tls_version": tls_version,
            "cipher_suite_count": len(cipher_suites),
            "extension_count": len(extensions),
            "is_suspicious": is_suspicious,
            "risk_indicators": self._identify_risk_indicators(tls_handshake),
        }

    def _analyze_tls_parameters(self, handshake: dict[str, Any]) -> bool:
        """Analyze TLS parameters for suspicious patterns"""
        cipher_suites = handshake.get("cipher_suites", [])
        extensions = handshake.get("extensions", [])

        weak_ciphers = [c for c in cipher_suites if "NULL" in str(c) or "EXPORT" in str(c)]
        if weak_ciphers:
            return True

        if len(extensions) > 20:
            return True

        return False

    def _identify_risk_indicators(self, handshake: dict[str, Any]) -> list[str]:
        """Identify TLS risk indicators"""
        indicators = []

        if handshake.get("tls_version") in ["SSLv2", "SSLv3", "TLS1.0"]:
            indicators.append("outdated_tls_version")

        cipher_suites = handshake.get("cipher_suites", [])
        if any("NULL" in str(c) or "EXPORT" in str(c) for c in cipher_suites):
            indicators.append("weak_cipher_suites")

        if handshake.get("sni_mismatch", False):
            indicators.append("sni_certificate_mismatch")

        return indicators


class CovertChannelDetector:
    """
    Covert channel detection in network traffic.

    Identifies hidden communication channels in seemingly benign protocols.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_covert_channels(self, traffic_sample: dict[str, Any]) -> dict[str, Any]:
        """
        Detect covert channels in traffic.

        Args:
            traffic_sample: Network traffic sample

        Returns:
            Covert channel detection results
        """
        channels_detected = []
        confidence_scores = []

        timing_channel = self._detect_timing_channel(traffic_sample)
        if timing_channel["detected"]:
            channels_detected.append("timing_channel")
            confidence_scores.append(timing_channel["confidence"])

        storage_channel = self._detect_storage_channel(traffic_sample)
        if storage_channel["detected"]:
            channels_detected.append("storage_channel")
            confidence_scores.append(storage_channel["confidence"])

        protocol_field_channel = self._detect_protocol_field_manipulation(traffic_sample)
        if protocol_field_channel["detected"]:
            channels_detected.append("protocol_field_manipulation")
            confidence_scores.append(protocol_field_channel["confidence"])

        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {
            "covert_channels_detected": len(channels_detected) > 0,
            "channels": channels_detected,
            "confidence": avg_confidence,
            "recommendations": self._generate_covert_channel_recommendations(channels_detected),
        }

    def _detect_timing_channel(self, traffic: dict[str, Any]) -> dict[str, bool]:
        """Detect timing-based covert channels"""
        packet_times = traffic.get("packet_timestamps", [])

        if len(packet_times) < 10:
            return {"detected": False, "confidence": 0.0}

        inter_packet_intervals = np.diff(packet_times)

        entropy = self._calculate_entropy(inter_packet_intervals)

        timing_channel_detected = entropy > 0.8 and entropy < 2.5

        return {
            "detected": timing_channel_detected,
            "confidence": 0.7 if timing_channel_detected else 0.0,
        }

    def _detect_storage_channel(self, traffic: dict[str, Any]) -> dict[str, bool]:
        """Detect storage-based covert channels"""
        packet_sizes = traffic.get("packet_sizes", [])

        if len(packet_sizes) < 10:
            return {"detected": False, "confidence": 0.0}

        size_variance = np.var(packet_sizes)
        size_entropy = self._calculate_entropy(packet_sizes)

        storage_channel_detected = size_entropy > 3.0 and size_variance > 1000

        return {
            "detected": storage_channel_detected,
            "confidence": 0.6 if storage_channel_detected else 0.0,
        }

    def _detect_protocol_field_manipulation(self, traffic: dict[str, Any]) -> dict[str, bool]:
        """Detect protocol field manipulation for covert channels"""
        protocol_fields = traffic.get("protocol_fields", {})

        suspicious_fields = []

        if "ip_id" in protocol_fields:
            ip_ids = protocol_fields["ip_id"]
            if len(ip_ids) > 5:
                id_entropy = self._calculate_entropy(ip_ids)
                if id_entropy > 3.5:
                    suspicious_fields.append("ip_id")

        if "tcp_seq" in protocol_fields:
            tcp_seqs = protocol_fields["tcp_seq"]
            if len(tcp_seqs) > 5:
                seq_pattern = self._detect_pattern_in_sequence(tcp_seqs)
                if seq_pattern:
                    suspicious_fields.append("tcp_seq")

        detected = len(suspicious_fields) > 0

        return {"detected": detected, "confidence": 0.8 if detected else 0.0}

    def _calculate_entropy(self, data: list) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0

        data_array = np.array(data)
        unique, counts = np.unique(data_array, return_counts=True)
        probabilities = counts / len(data_array)

        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        return entropy

    def _detect_pattern_in_sequence(self, sequence: list[int]) -> bool:
        """Detect non-random patterns in sequence"""
        if len(sequence) < 5:
            return False

        diffs = np.diff(sequence)
        diff_std = np.std(diffs)

        return diff_std < np.mean(diffs) * 0.1

    def _generate_covert_channel_recommendations(self, channels: list[str]) -> list[str]:
        """Generate recommendations for covert channel mitigation"""
        recs = []

        if "timing_channel" in channels:
            recs.append("Implement traffic padding to normalize timing patterns")
            recs.append("Monitor for unusual inter-packet timing distributions")

        if "storage_channel" in channels:
            recs.append("Inspect payload entropy for hidden data")
            recs.append("Normalize packet sizes where possible")

        if "protocol_field_manipulation" in channels:
            recs.append("Validate protocol field values against RFC standards")
            recs.append("Implement strict protocol field sanitization")

        return recs


class TrafficAnalysisEngine:
    """
    Comprehensive traffic analysis engine integrating flow analysis,
    graph-based detection, encrypted traffic fingerprinting, and covert channel detection.
    """

    def __init__(
        self,
        enable_flow_analysis: bool = True,
        enable_graph_analysis: bool = True,
        enable_tls_fingerprinting: bool = True,
        enable_covert_detection: bool = True,
    ):
        self.enable_flow_analysis = enable_flow_analysis
        self.enable_graph_analysis = enable_graph_analysis
        self.enable_tls_fingerprinting = enable_tls_fingerprinting
        self.enable_covert_detection = enable_covert_detection

        self.flow_analyzer = NetworkFlowAnalyzer() if enable_flow_analysis else None
        self.graph_analyzer = CommunicationGraphAnalyzer() if enable_graph_analysis else None
        self.tls_fingerprinter = (
            EncryptedTrafficFingerprinter() if enable_tls_fingerprinting else None
        )
        self.covert_detector = CovertChannelDetector() if enable_covert_detection else None

        self.logger = logging.getLogger(__name__)

    def analyze_traffic(self, traffic_data: dict[str, Any]) -> TrafficAnalysisResult:
        """
        Comprehensive traffic analysis.

        Args:
            traffic_data: Network traffic data including:
                - flow_records: NetFlow/IPFIX records
                - graph_data: Communication graph (nodes, edges)
                - tls_handshakes: TLS handshake data
                - raw_traffic: Raw packet data

        Returns:
            Traffic analysis result with anomaly detection
        """
        result = TrafficAnalysisResult(
            anomaly_detected=False,
            confidence=0.0,
            anomaly_type="normal_traffic",
            risk_score=0.0,
        )

        if self.enable_flow_analysis and "flow_records" in traffic_data:
            flow_result = self.flow_analyzer.analyze_flows(traffic_data["flow_records"])
            result.flow_statistics = flow_result["statistics"]

            if flow_result["anomaly_detected"]:
                result.anomaly_detected = True
                result.anomaly_type = flow_result["anomalies"][0]
                result.confidence = 0.8
                result.recommended_actions.append("Investigate suspicious network flows")

        if self.enable_tls_fingerprinting and "tls_handshakes" in traffic_data:
            for handshake in traffic_data["tls_handshakes"]:
                fingerprint = self.tls_fingerprinter.fingerprint_tls(handshake)
                result.encrypted_flows.append(fingerprint)

                if fingerprint["is_suspicious"]:
                    result.anomaly_detected = True
                    result.protocol_anomalies.extend(fingerprint["risk_indicators"])

        if self.enable_covert_detection and "raw_traffic" in traffic_data:
            covert_result = self.covert_detector.detect_covert_channels(traffic_data["raw_traffic"])

            if covert_result["covert_channels_detected"]:
                result.anomaly_detected = True
                result.covert_channels = covert_result["channels"]
                result.anomaly_type = "covert_channel"
                result.confidence = max(result.confidence, covert_result["confidence"])
                result.recommended_actions.extend(covert_result["recommendations"])

        result.risk_score = self._calculate_risk_score(result)
        result.attribution_indicators = self._extract_attribution_indicators(traffic_data, result)

        self.logger.info(
            f"Traffic analysis: {result.anomaly_type}, " f"confidence={result.confidence:.2f}"
        )

        return result

    def _calculate_risk_score(self, result: TrafficAnalysisResult) -> float:
        """Calculate overall traffic risk score"""
        base_score = result.confidence

        if result.anomaly_type in ["data_exfiltration", "covert_channel"]:
            base_score *= 1.5
        elif result.anomaly_type == "botnet_c2":
            base_score *= 1.4

        if result.covert_channels:
            base_score *= 1.3

        if len(result.protocol_anomalies) > 3:
            base_score *= 1.2

        return min(base_score, 1.0)

    def _extract_attribution_indicators(
        self, traffic_data: dict[str, Any], result: TrafficAnalysisResult
    ) -> list[str]:
        """Extract indicators for threat attribution"""
        indicators = []

        if result.flow_statistics.get("unique_sources", 0) == 1:
            indicators.append("single_source_attack")

        if result.anomaly_type == "port_scanning":
            indicators.append("reconnaissance_phase")

        if result.covert_channels:
            indicators.append("sophisticated_tradecraft")

        if len(result.encrypted_flows) > 10:
            indicators.append("extensive_encryption_usage")

        return indicators
