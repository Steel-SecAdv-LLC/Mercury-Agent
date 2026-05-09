"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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

from __future__ import annotations

"""Real-data tests for Cyber Fortress module using simulated datasets."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from omni_mercury_engine.security.cyber_fortress import CyberFortress, ResonanceHashIntegrityChecker

try:
    from assets.loaders import generate_pcap_data

    ASSETS_AVAILABLE = True
except ImportError:
    ASSETS_AVAILABLE = False
    generate_pcap_data = None

pytestmark = pytest.mark.skipif(
    not ASSETS_AVAILABLE, reason="assets module not available (requires real data loaders)"
)


class TestRealDataValidation:
    def test_hash_integrity_with_tampering(self):
        """Test hash integrity detection on simulated PCAP with tampering."""
        normal_data = generate_pcap_data(num_packets=500, inject_tampering=False)
        tampered_data = generate_pcap_data(
            num_packets=500, inject_tampering=True, tampering_ratio=0.15
        )

        checker = ResonanceHashIntegrityChecker(threshold_std=10.0)

        normal_result = checker.check_integrity(normal_data["hash_chain"])
        assert normal_result["integrity_verified"] is True

        tampered_result = checker.check_integrity(tampered_data["hash_chain"])
        assert tampered_result["resonance_anomalies"] > 0

    def test_threshold_sensitivity(self):
        """Test different thresholds on simulated data."""
        data = generate_pcap_data(num_packets=300, inject_tampering=True, tampering_ratio=0.05)

        checker = ResonanceHashIntegrityChecker()

        thresholds = [3.0, 5.0, 7.0, 10.0]
        anomaly_counts = []

        for threshold in thresholds:
            result = checker.check_integrity(data["hash_chain"], threshold_std=threshold)
            anomaly_counts.append(result["resonance_anomalies"])

        assert (
            anomaly_counts[0] >= anomaly_counts[-1]
        ), "Lower threshold should detect more anomalies"

    def test_encrypted_traffic_detection(self):
        """Test encrypted traffic anomaly detection on simulated PCAP."""
        fortress = CyberFortress()

        results = []
        for _ in range(10):
            data = generate_pcap_data(num_packets=200)
            result = fortress.fortress_scan(
                {"hash_chain": data["hash_chain"], "network_traffic": data["network_traffic"]}
            )
            results.append(result)

        assert all(0.0 <= r.threat_score <= 1.0 for r in results)
        assert all(r.hash_integrity_verified is not None for r in results)

    def test_pcap_benchmark_accuracy(self):
        """Benchmark accuracy on simulated PCAP data."""
        checker = ResonanceHashIntegrityChecker(threshold_std=10.0)

        true_positives = 0
        true_negatives = 0

        for _ in range(25):
            normal = generate_pcap_data(num_packets=200, inject_tampering=False)
            result = checker.check_integrity(normal["hash_chain"])
            if result["integrity_verified"]:
                true_negatives += 1

        for _ in range(25):
            tampered = generate_pcap_data(
                num_packets=200, inject_tampering=True, tampering_ratio=0.1
            )
            result = checker.check_integrity(tampered["hash_chain"])
            if not result["integrity_verified"]:
                true_positives += 1

        accuracy = (true_positives + true_negatives) / 50
        assert accuracy > 0.6, f"Accuracy {accuracy:.2f} should be > 60% on simulated data"
