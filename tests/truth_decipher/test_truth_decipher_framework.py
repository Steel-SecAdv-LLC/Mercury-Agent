"""
Mercury Agent ♱
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

from __future__ import annotations

"""
Integration tests for Truth Deciphering Framework.

Tests all 5 phases (Discovery, Cognitive Analysis, Identification, Ethics, Resolution)
independently and as an integrated pipeline.

Five-Phase Architecture (Enhanced with Cognitive Layer):
1. Discovery: Multi-dimensional anomaly detection + novel class discovery
2. Cognitive Analysis: Uncertainty quantification, causal discovery, reasoning
3. Identification: Classification by type/severity with detailed analysis
4. Ethical Course: Evaluation against 8 ethical principles
5. Resolution: Automated fixes with self-healing and autonomous execution
"""

import numpy as np

from omni_mercury_engine.truth_decipher import TruthDecipherFramework, TruthDecipherResult


class TestTruthDecipherFramework:
    """Test Truth Deciphering Framework orchestration."""

    def test_initialization(self):
        """Test framework initialization with all components."""
        framework = TruthDecipherFramework()

        assert framework.anomaly_engine is not None
        assert framework.ethics_governor is not None
        assert framework.autonomy is not None
        assert framework.enable_novel_discovery is True
        assert framework.enable_self_healing is True

    def test_phase1_discovery_with_anomaly(self):
        """Test Phase 1: Discovery on synthetic anomaly data."""
        framework = TruthDecipherFramework()

        normal_data = np.random.randn(100, 10) * 0.5
        anomaly_data = np.random.randn(20, 10) * 5.0
        data_stream = np.vstack([normal_data, anomaly_data])

        result = framework.detect_anomalies(data_stream)

        assert "anomaly_detected" in result
        assert "anomaly_score" in result
        assert "novel_classes" in result
        assert isinstance(result["anomaly_score"], float)

    def test_phase1_discovery_without_anomaly(self):
        """Test Phase 1: Discovery on normal data."""
        framework = TruthDecipherFramework()

        normal_data = np.random.randn(100, 10) * 0.3

        result = framework.detect_anomalies(normal_data)

        assert "anomaly_detected" in result
        assert "anomaly_score" in result

    def test_phase2_identification_critical(self):
        """Test Phase 2: Identification of critical anomaly."""
        framework = TruthDecipherFramework()

        discovery_result = {
            "anomaly_detected": True,
            "anomaly_score": 0.95,
            "severity": 0.9,
            "novel_classes": ["novel_class_0"],
        }

        result = framework.classify_and_identify(discovery_result)

        assert result["issue_type"] == "CRITICAL"
        assert "Immediate investigation required" in result["recommendations"]
        assert len(result["recommendations"]) >= 3

    def test_phase2_identification_medium(self):
        """Test Phase 2: Identification of medium severity anomaly."""
        framework = TruthDecipherFramework()

        discovery_result = {
            "anomaly_detected": True,
            "anomaly_score": 0.6,
            "severity": 0.5,
            "novel_classes": [],
        }

        result = framework.classify_and_identify(discovery_result)

        assert result["issue_type"] == "MEDIUM"
        assert "Schedule detailed analysis" in result["recommendations"]

    def test_phase3_ethics_evaluation_pass(self):
        """Test Phase 3: Ethical evaluation that passes."""
        framework = TruthDecipherFramework()

        identification_result = {
            "issue_type": "MEDIUM",
            "severity": 0.5,
            "anomaly_score": 0.6,
            "recommendations": ["Monitor for escalation"],
        }

        result = framework.determine_ethics(identification_result)

        assert result.passed is True
        assert result.overall_score >= 0.7
        assert len(result.principle_scores) == 8

    def test_phase3_ethics_evaluation_with_context(self):
        """Test Phase 3: Ethical evaluation with context."""
        framework = TruthDecipherFramework()

        identification_result = {
            "issue_type": "CRITICAL",
            "severity": 0.9,
            "anomaly_score": 0.95,
            "recommendations": ["Immediate action"],
        }

        context = {"bias_checked": True, "verified_claims": True}

        result = framework.determine_ethics(identification_result, context)

        assert result.passed is True
        assert "compassion" in result.principle_scores
        assert "justice" in result.principle_scores

    def test_phase4_resolution_with_self_healing(self):
        """Test Phase 4: Resolution with self-healing."""
        framework = TruthDecipherFramework(enable_self_healing=True)

        identification_result = {"issue_type": "HIGH", "severity": 0.8, "anomaly_score": 0.85}

        data_stream = np.random.randn(50, 10) * 3.0

        result = framework.resolve_with_measures(identification_result, data_stream)

        assert result["applied"] is True
        assert result["type"] == "autonomous_with_self_healing"
        assert len(result["actions"]) > 0
        assert result["signature_id"] is not None

    def test_phase4_resolution_without_self_healing(self):
        """Test Phase 4: Resolution without self-healing."""
        framework = TruthDecipherFramework(enable_self_healing=False)

        identification_result = {"issue_type": "MEDIUM", "severity": 0.6, "anomaly_score": 0.65}

        data_stream = np.random.randn(50, 10) * 2.0

        result = framework.resolve_with_measures(identification_result, data_stream)

        assert result["applied"] is True
        assert len(result["actions"]) > 0
        assert result["signature_id"] is None

    def test_full_pipeline_with_anomaly(self):
        """Test complete pipeline: All 5 phases with anomaly."""
        framework = TruthDecipherFramework()

        normal_data = np.random.randn(80, 10) * 0.5
        anomaly_data = np.random.randn(20, 10) * 4.0
        data_stream = np.vstack([normal_data, anomaly_data])

        result = framework.decipher_truth(data_stream)

        assert isinstance(result, TruthDecipherResult)
        assert result.phase_completed >= 1

        if result.anomaly_detected:
            assert result.phase_completed >= 2
            assert result.issue_type is not None
            assert len(result.recommendations) > 0

            if result.ethics_passed:
                assert result.phase_completed == 5  # Updated for 5-phase architecture
                assert result.resolution_applied is True

    def test_full_pipeline_without_anomaly(self):
        """Test complete pipeline: Early exit when no anomaly."""
        np.random.seed(999)
        framework = TruthDecipherFramework()

        normal_data = np.random.randn(100, 10) * 0.1

        result = framework.decipher_truth(normal_data)

        assert isinstance(result, TruthDecipherResult)
        # Phase 1 = early exit (no anomaly), Phase 5 = full pipeline completed
        assert result.phase_completed in [1, 5]
        if result.phase_completed == 1:
            assert result.resolution_applied is False

    def test_full_pipeline_energy_grid_scenario(self):
        """Test pipeline on realistic energy grid anomaly scenario."""
        framework = TruthDecipherFramework()

        normal_voltage = np.random.normal(120.0, 2.0, (200, 5))
        spike_voltage = np.random.normal(150.0, 10.0, (30, 5))
        data_stream = np.vstack([normal_voltage, spike_voltage])

        context = {
            "infrastructure": "energy_grid",
            "sensor_type": "voltage",
            "critical_threshold": 140.0,
        }

        result = framework.decipher_truth(data_stream, context)

        assert result.phase_completed >= 1
        if result.anomaly_detected:
            assert result.severity is not None
            assert len(result.recommendations) > 0

    def test_statistics_collection(self):
        """Test statistics collection from framework."""
        framework = TruthDecipherFramework()

        data = np.random.randn(50, 10)
        framework.detect_anomalies(data)

        stats = framework.get_statistics()

        assert "ethics_stats" in stats
        assert "autonomy_metrics" in stats
        assert "self_healing_signatures" in stats
        assert isinstance(stats["ethics_stats"], dict)

    def test_torch_tensor_input(self):
        """Test framework with PyTorch tensor input."""
        import torch

        framework = TruthDecipherFramework()

        data_tensor = torch.randn(50, 10) * 2.0

        result = framework.decipher_truth(data_tensor)

        assert isinstance(result, TruthDecipherResult)
        assert result.phase_completed >= 1

    def test_dict_input(self):
        """Test framework with dictionary input."""
        framework = TruthDecipherFramework()

        data_dict = {
            "sensor_1": np.random.randn(50),
            "sensor_2": np.random.randn(50) * 3.0,
            "sensor_3": np.random.randn(50),
        }

        result = framework.decipher_truth(data_dict)

        assert isinstance(result, TruthDecipherResult)
        assert result.phase_completed >= 1


class TestTruthDecipherIntegration:
    """Integration tests for cross-domain scenarios."""

    def test_space_infrastructure_anomaly(self):
        """Test framework on space infrastructure anomaly."""
        framework = TruthDecipherFramework()

        earth_radius = 6371.0
        normal_positions = np.random.randn(100, 3) * 5 + [earth_radius + 400, 0, 0]
        anomalous_positions = np.random.randn(20, 3) * 50 + [earth_radius + 450, 50, 30]
        data_stream = np.vstack([normal_positions, anomalous_positions])

        context = {"infrastructure": "space", "type": "satellite_position"}

        result = framework.decipher_truth(data_stream, context)

        assert result.phase_completed >= 1

    def test_healthcare_emergency_scenario(self):
        """Test framework on healthcare emergency scenario."""
        framework = TruthDecipherFramework()

        normal_vitals = np.random.normal([120, 80, 98.6, 70], [5, 3, 0.5, 5], (150, 4))
        critical_vitals = np.random.normal([180, 110, 104, 120], [10, 5, 1, 10], (20, 4))
        data_stream = np.vstack([normal_vitals, critical_vitals])

        context = {"infrastructure": "healthcare", "type": "vital_signs"}

        result = framework.decipher_truth(data_stream, context)

        assert result.phase_completed >= 1
        if result.anomaly_detected and result.issue_type == "CRITICAL":
            assert "Immediate investigation required" in result.recommendations
