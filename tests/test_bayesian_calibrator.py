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

"""
Unit tests for BayesianConfidenceCalibrator.

Tests the learned confidence model that replaces the fixed 0.76 heuristic.
Verifies:
- Novel contexts start at ~0.76
- Confidence climbs to 0.95+ after >=5 successes
- Failures appropriately reduce confidence
- Serialization/deserialization works correctly
"""

import tempfile
from pathlib import Path

from omni_mercury_engine.agentic.bayesian_calibrator import (
    BayesianConfidenceCalibrator,
    CalibrationConfig,
    ContextStats,
)


class TestContextStats:
    """Tests for ContextStats dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        stats = ContextStats()
        assert stats.alpha == 0.76
        assert stats.beta == 0.24
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.total_observations == 0

    def test_posterior_mean_no_observations(self):
        """Test posterior mean with no observations equals prior mean."""
        stats = ContextStats(alpha=0.76, beta=0.24)
        assert abs(stats.posterior_mean - 0.76) < 0.01

    def test_posterior_mean_with_successes(self):
        """Test posterior mean increases with successes."""
        stats = ContextStats(alpha=0.76, beta=0.24, successes=5, failures=0)
        # posterior_mean = (0.76 + 5) / (0.76 + 0.24 + 5) = 5.76 / 6 ≈ 0.96
        assert stats.posterior_mean > 0.95

    def test_posterior_mean_with_failures(self):
        """Test posterior mean decreases with failures."""
        stats = ContextStats(alpha=0.76, beta=0.24, successes=0, failures=5)
        # posterior_mean = 0.76 / (0.76 + 0.24 + 5) = 0.76 / 6 ≈ 0.127
        assert stats.posterior_mean < 0.2

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict preserve data."""
        stats = ContextStats(
            alpha=0.8,
            beta=0.2,
            successes=10,
            failures=2,
            last_updated=12345.0,
        )
        data = stats.to_dict()
        restored = ContextStats.from_dict(data)

        assert restored.alpha == stats.alpha
        assert restored.beta == stats.beta
        assert restored.successes == stats.successes
        assert restored.failures == stats.failures
        assert restored.last_updated == stats.last_updated


class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CalibrationConfig()
        assert config.prior_mean == 0.76
        assert config.prior_kappa == 1.0
        assert config.familiarity_target == 5
        assert config.min_confidence == 0.5
        assert config.max_confidence == 0.999


class TestBayesianConfidenceCalibrator:
    """Tests for BayesianConfidenceCalibrator."""

    def test_novel_context_returns_prior(self):
        """Test that novel contexts return approximately the prior mean."""
        calibrator = BayesianConfidenceCalibrator()
        confidence = calibrator.get_confidence("medical", "Analyze patient data")

        # Should be close to 0.76 for novel context
        assert abs(confidence - 0.76) < 0.01

    def test_confidence_increases_with_successes(self):
        """Test that confidence increases after successful executions."""
        calibrator = BayesianConfidenceCalibrator()
        domain = "medical"
        goal = "Analyze patient data"

        initial_confidence = calibrator.get_confidence(domain, goal)

        # Add 5 successes
        for _ in range(5):
            calibrator.update(domain, goal, success=True)

        final_confidence = calibrator.get_confidence(domain, goal)

        # Should have increased significantly
        assert final_confidence > initial_confidence
        assert final_confidence > 0.95

    def test_confidence_decreases_with_failures(self):
        """Test that confidence decreases after failures."""
        calibrator = BayesianConfidenceCalibrator()
        domain = "security"
        goal = "Detect intrusion patterns"

        # First add some successes to establish baseline
        for _ in range(3):
            calibrator.update(domain, goal, success=True)

        confidence_after_successes = calibrator.get_confidence(domain, goal)

        # Now add failures
        for _ in range(5):
            calibrator.update(domain, goal, success=False)

        confidence_after_failures = calibrator.get_confidence(domain, goal)

        # Should have decreased
        assert confidence_after_failures < confidence_after_successes

    def test_rapid_climb_to_high_confidence(self):
        """Test that confidence rapidly climbs to 0.95+ after 5 successes."""
        calibrator = BayesianConfidenceCalibrator()
        domain = "humanitarian"
        goal = "Monitor crisis indicators"

        confidences = [calibrator.get_confidence(domain, goal)]

        for i in range(10):
            calibrator.update(domain, goal, success=True)
            confidences.append(calibrator.get_confidence(domain, goal))

        # Should start at ~0.76
        assert abs(confidences[0] - 0.76) < 0.01

        # Should be > 0.95 after 5 successes
        assert confidences[5] > 0.95

        # Should approach max after 10 successes
        assert confidences[10] > 0.97

        # Should be monotonically increasing
        for i in range(len(confidences) - 1):
            assert confidences[i + 1] >= confidences[i]

    def test_different_contexts_are_independent(self):
        """Test that different contexts maintain independent statistics."""
        calibrator = BayesianConfidenceCalibrator()

        # Update medical context with successes
        for _ in range(5):
            calibrator.update("medical", "Analyze data", success=True)

        # Update security context with failures
        for _ in range(5):
            calibrator.update("security", "Detect threats", success=False)

        medical_confidence = calibrator.get_confidence("medical", "Analyze data")
        security_confidence = calibrator.get_confidence("security", "Detect threats")

        # Medical should be high, security should be low
        assert medical_confidence > 0.95
        # Note: min_confidence is 0.5, so security can't go below that
        assert security_confidence <= 0.5

    def test_goal_type_classification(self):
        """Test that goal types are correctly classified."""
        calibrator = BayesianConfidenceCalibrator()

        assert calibrator.classify_goal_type("Analyze patient data") == "analysis"
        assert calibrator.classify_goal_type("Detect anomalies") == "analysis"
        assert calibrator.classify_goal_type("Monitor network traffic") == "monitoring"
        assert calibrator.classify_goal_type("Track resource usage") == "monitoring"
        assert calibrator.classify_goal_type("Respond to incident") == "response"
        assert calibrator.classify_goal_type("Take action on alert") == "response"
        assert calibrator.classify_goal_type("Process data batch") == "generic"

    def test_same_goal_type_shares_context(self):
        """Test that goals with same type share context statistics."""
        calibrator = BayesianConfidenceCalibrator()
        domain = "medical"

        # Update with one analysis goal
        for _ in range(5):
            calibrator.update(domain, "Analyze patient vitals", success=True)

        # Check confidence for different analysis goal in same domain
        confidence = calibrator.get_confidence(domain, "Detect anomalies in ECG")

        # Should be high because both are "analysis" type in "medical" domain
        assert confidence > 0.95

    def test_confidence_bounds(self):
        """Test that confidence stays within configured bounds."""
        config = CalibrationConfig(min_confidence=0.5, max_confidence=0.999)
        calibrator = BayesianConfidenceCalibrator(config)

        # Add many successes
        for _ in range(100):
            calibrator.update("test", "Analyze data", success=True)

        confidence = calibrator.get_confidence("test", "Analyze data")
        assert confidence <= 0.999

        # Add many failures to a different context
        for _ in range(100):
            calibrator.update("test", "Monitor system", success=False)

        confidence = calibrator.get_confidence("test", "Monitor system")
        assert confidence >= 0.5

    def test_memory_evidence_boosts_familiarity(self):
        """Test that memory evidence count increases familiarity."""
        calibrator = BayesianConfidenceCalibrator()
        domain = "scientific"
        goal = "Analyze experimental data"

        # No observations, no memory evidence
        conf_no_evidence = calibrator.get_confidence(domain, goal, memory_evidence_count=0)

        # No observations, but memory evidence
        conf_with_evidence = calibrator.get_confidence(domain, goal, memory_evidence_count=5)

        # Memory evidence should increase familiarity, moving toward posterior
        # Since posterior starts at prior (0.76), the effect is subtle but present
        assert conf_with_evidence >= conf_no_evidence

    def test_get_stats(self):
        """Test getting statistics for a context."""
        calibrator = BayesianConfidenceCalibrator()
        domain = "energy"
        goal = "Analyze grid data"

        # Initially no stats
        assert calibrator.get_stats(domain, goal) is None

        # After getting confidence, stats should exist
        calibrator.get_confidence(domain, goal)
        stats = calibrator.get_stats(domain, goal)
        assert stats is not None
        assert stats.total_observations == 0

        # After update, stats should reflect it
        calibrator.update(domain, goal, success=True)
        stats = calibrator.get_stats(domain, goal)
        assert stats.successes == 1
        assert stats.total_observations == 1

    def test_get_summary(self):
        """Test getting calibrator summary."""
        calibrator = BayesianConfidenceCalibrator()

        # Empty summary
        summary = calibrator.get_summary()
        assert summary["total_contexts"] == 0
        assert summary["total_observations"] == 0

        # Add some data
        for _ in range(3):
            calibrator.update("medical", "Analyze data", success=True)
        for _ in range(2):
            calibrator.update("security", "Detect threats", success=True)

        summary = calibrator.get_summary()
        assert summary["total_contexts"] == 2
        assert summary["total_observations"] == 5
        assert "medical:analysis" in summary["contexts"]
        assert "security:analysis" in summary["contexts"]

    def test_save_and_load(self):
        """Test saving and loading calibrator state."""
        calibrator = BayesianConfidenceCalibrator()

        # Add some data
        for _ in range(5):
            calibrator.update("medical", "Analyze data", success=True)
        for _ in range(3):
            calibrator.update("security", "Detect threats", success=False)

        original_medical_conf = calibrator.get_confidence("medical", "Analyze data")
        original_security_conf = calibrator.get_confidence("security", "Detect threats")

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibrator.json"
            calibrator.save(path)

            # Create new calibrator and load
            new_calibrator = BayesianConfidenceCalibrator()
            new_calibrator.load(path)

            # Should have same confidences
            loaded_medical_conf = new_calibrator.get_confidence("medical", "Analyze data")
            loaded_security_conf = new_calibrator.get_confidence("security", "Detect threats")

            assert abs(loaded_medical_conf - original_medical_conf) < 0.001
            assert abs(loaded_security_conf - original_security_conf) < 0.001

    def test_reset(self):
        """Test resetting calibrator state."""
        calibrator = BayesianConfidenceCalibrator()

        # Add some data
        for _ in range(5):
            calibrator.update("medical", "Analyze data", success=True)

        assert len(calibrator.contexts) > 0

        # Reset
        calibrator.reset()

        assert len(calibrator.contexts) == 0
        # Novel context should return prior
        assert abs(calibrator.get_confidence("medical", "Analyze data") - 0.76) < 0.01

    def test_custom_config(self):
        """Test calibrator with custom configuration."""
        config = CalibrationConfig(
            prior_mean=0.5,
            prior_kappa=2.0,
            familiarity_target=10,
        )
        calibrator = BayesianConfidenceCalibrator(config)

        # Novel context should return custom prior
        confidence = calibrator.get_confidence("test", "Analyze data")
        assert abs(confidence - 0.5) < 0.01

        # Should take longer to reach high confidence with higher familiarity_target
        for _ in range(5):
            calibrator.update("test", "Analyze data", success=True)

        confidence = calibrator.get_confidence("test", "Analyze data")
        # With familiarity_target=10, 5 observations gives familiarity=0.5
        # So confidence should be between prior (0.5) and posterior
        assert confidence > 0.5
        assert confidence < 0.95  # Not yet at full posterior


class TestIntegrationScenarios:
    """Integration tests simulating real Mercury Agent scenarios."""

    def test_training_scenario_confidence_growth(self):
        """Test confidence growth across training epochs like Mercury Agent."""
        calibrator = BayesianConfidenceCalibrator()

        # Simulate 8 scenarios across 100 epochs (like mercury_agent_training.py)
        scenarios = [
            ("medical", "Analyze patient vital signs for anomalies"),
            ("security", "Detect potential intrusion patterns in network traffic"),
            ("humanitarian", "Monitor humanitarian crisis indicators"),
            ("infrastructure", "Assess critical infrastructure health"),
            ("energy", "Optimize energy distribution and detect anomalies"),
            ("scientific", "Analyze experimental data for significant findings"),
            ("financial", "Detect potential fraudulent transactions"),
            ("general", "Perform general anomaly detection on mixed data"),
        ]

        epoch_confidences = []

        for epoch in range(100):
            epoch_conf = []
            for domain, goal in scenarios:
                # All scenarios succeed (like current benchmark)
                calibrator.update(domain, goal, success=True)
                conf = calibrator.get_confidence(domain, goal)
                epoch_conf.append(conf)

            avg_conf = sum(epoch_conf) / len(epoch_conf)
            epoch_confidences.append(avg_conf)

        # First epoch should be around 0.76 (prior)
        assert epoch_confidences[0] < 0.85

        # After 5 epochs, should be climbing
        assert epoch_confidences[4] > epoch_confidences[0]

        # After 100 epochs, should be very high
        assert epoch_confidences[-1] > 0.98

        # Should be monotonically non-decreasing (all successes)
        for i in range(len(epoch_confidences) - 1):
            assert epoch_confidences[i + 1] >= epoch_confidences[i] - 0.001  # Small tolerance

    def test_mixed_success_failure_scenario(self):
        """Test confidence behavior with mixed success/failure."""
        calibrator = BayesianConfidenceCalibrator()
        domain = "security"
        goal = "Detect intrusion patterns"

        confidences = []

        # 10 successes
        for _ in range(10):
            calibrator.update(domain, goal, success=True)
            confidences.append(calibrator.get_confidence(domain, goal))

        # 5 failures (simulating distribution shift)
        for _ in range(5):
            calibrator.update(domain, goal, success=False)
            confidences.append(calibrator.get_confidence(domain, goal))

        # 5 more successes (recovery)
        for _ in range(5):
            calibrator.update(domain, goal, success=True)
            confidences.append(calibrator.get_confidence(domain, goal))

        # Should have peaked around observation 10
        peak_idx = confidences.index(max(confidences[:11]))
        assert peak_idx >= 5

        # Should have dropped after failures
        assert confidences[14] < confidences[9]

        # Should have recovered somewhat after more successes
        assert confidences[-1] > confidences[14]
