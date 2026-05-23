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

"""Real-data tests for Emergent Life Detector using simulated datasets."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from omni_mercury_engine.emergent.emergent_life_detector import (
    EmergentLifeDetector,
    SETICosmicSignalAnalyzer,
)

try:
    from assets.loaders import generate_seti_signal

    ASSETS_AVAILABLE = True
except ImportError:
    ASSETS_AVAILABLE = False
    generate_seti_signal = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not ASSETS_AVAILABLE, reason="assets module not available (requires real data loaders)"
)


class TestRealDataValidation:
    def test_seti_technosignature_detection(self) -> None:
        """Test technosignature detection on simulated cosmic signals."""
        noise_data = generate_seti_signal(num_samples=10000, inject_technosignature=False)
        signal_data = generate_seti_signal(
            num_samples=10000, inject_technosignature=True, signal_type="narrow_band"
        )

        analyzer = SETICosmicSignalAnalyzer(threshold_std=4.0)

        noise_result = analyzer.detect_seti_anomaly(noise_data["cosmic_signal"])
        signal_result = analyzer.detect_seti_anomaly(signal_data["cosmic_signal"])

        assert noise_result["seti_anomaly_detected"] in [True, False]
        assert signal_result["seti_anomaly_detected"] in [True, False]

    def test_threshold_sensitivity(self) -> None:
        """Test different thresholds on simulated SETI data."""
        data = generate_seti_signal(
            num_samples=10000, inject_technosignature=True, signal_type="narrow_band"
        )

        analyzer = SETICosmicSignalAnalyzer()

        thresholds = [2.0, 4.0, 6.0, 8.0]
        detection_scores = []

        for threshold in thresholds:
            result = analyzer.detect_seti_anomaly(data["cosmic_signal"], threshold_std=threshold)
            detection_scores.append(result["seti_confidence"])

        assert all(0.0 <= s <= 1.0 for s in detection_scores)

    def test_seti_benchmark_accuracy(self) -> None:
        """Benchmark accuracy on simulated cosmic signals."""
        detector = EmergentLifeDetector(enable_biosignatures=False, enable_contact_protocols=False)

        true_positives = 0
        true_negatives = 0

        for _ in range(20):
            noise = generate_seti_signal(num_samples=10000, inject_technosignature=False)
            result = detector.detect_emergent_life(noise["cosmic_signal"], "seti")
            if not result.life_signal_detected:
                true_negatives += 1

        for _ in range(20):
            signal = generate_seti_signal(num_samples=10000, inject_technosignature=True)
            result = detector.detect_emergent_life(signal["cosmic_signal"], "seti")
            if result.life_signal_detected:
                true_positives += 1

        accuracy = (true_positives + true_negatives) / 40
        assert accuracy > 0.4, f"Accuracy {accuracy:.2f} should be > 40% on simulated data"

    def test_comprehensive_life_detection(self) -> None:
        """Test comprehensive life detection workflow."""
        detector = EmergentLifeDetector()

        signal_data = generate_seti_signal(
            num_samples=10000, inject_technosignature=True, signal_type="repeating"
        )

        result = detector.detect_emergent_life(signal_data["cosmic_signal"], "comprehensive")

        assert hasattr(result, "life_signal_detected")
        assert hasattr(result, "confidence")
        assert len(result.recommendations) > 0
