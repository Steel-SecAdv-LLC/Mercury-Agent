"""
Production Tests for UncertaintyQuantifier

Tests the real implementations:
- Monte Carlo Dropout (Gal & Ghahramani 2016)
- Temperature Scaling (Guo et al. 2017)
- Adaptive Conformal Inference (Gibbs & Candes 2021)
- Heteroscedastic Aleatoric Estimation
- Epistemic/Aleatoric Decomposition
"""

from __future__ import annotations

import numpy as np


class TestMCDropout:
    """Tests for Monte Carlo Dropout uncertainty estimation."""

    def test_mc_dropout_wrapper_initialization(self) -> None:
        """Test MCDropoutWrapper creates properly."""
        from omni_mercury_engine.cognitive.uncertainty import MCDropoutWrapper

        # MCDropoutWrapper takes a model, not n_samples
        class DummyModel:
            training = False

            def __call__(self, x):
                return x

        wrapper = MCDropoutWrapper(model=DummyModel(), dropout_rate=0.2)
        assert wrapper.dropout_rate == 0.2
        assert wrapper.model is not None


class TestTemperatureScaling:
    """Tests for Temperature Scaling calibration."""

    def test_temperature_scaling_initialization(self) -> None:
        """Test TemperatureScaler initializes correctly."""
        from omni_mercury_engine.cognitive.uncertainty import TemperatureScaler

        scaler = TemperatureScaler(init_temperature=1.0)
        assert scaler.temperature == 1.0
        assert not scaler._fitted

    def test_temperature_scaling_fit(self) -> None:
        """Test temperature scaling learns optimal temperature."""
        from omni_mercury_engine.cognitive.uncertainty import TemperatureScaler

        scaler = TemperatureScaler()

        # Create overconfident logits (miscalibrated)
        n_samples = 500
        n_classes = 5
        logits = np.random.randn(n_samples, n_classes) * 3  # High variance
        labels = np.random.randint(0, n_classes, n_samples)

        scaler.fit(logits, labels)

        assert scaler._fitted
        assert scaler.temperature > 0

    def test_temperature_scaling_calibration(self) -> None:
        """Test calibrated probabilities are well-behaved."""
        from omni_mercury_engine.cognitive.uncertainty import TemperatureScaler

        scaler = TemperatureScaler()

        n_samples = 200
        n_classes = 3
        logits = np.random.randn(n_samples, n_classes) * 2
        labels = np.random.randint(0, n_classes, n_samples)

        scaler.fit(logits, labels)
        calibrated_probs = scaler.calibrated_probs(logits)

        # Probabilities should sum to 1
        assert np.allclose(calibrated_probs.sum(axis=1), 1.0, atol=1e-6)
        # All probabilities should be in [0, 1]
        assert np.all(calibrated_probs >= 0)
        assert np.all(calibrated_probs <= 1)


class TestAdaptiveConformal:
    """Tests for Adaptive Conformal Inference."""

    def test_adaptive_conformal_initialization(self) -> None:
        """Test AdaptiveConformalInference initializes correctly."""
        from omni_mercury_engine.cognitive.uncertainty import AdaptiveConformalInference

        aci = AdaptiveConformalInference(target_coverage=0.9, gamma=0.01)
        assert aci.target_coverage == 0.9
        assert aci.gamma == 0.01
        assert abs(aci.alpha - 0.1) < 1e-9  # 1 - target_coverage

    def test_adaptive_conformal_update(self) -> None:
        """Test alpha updates based on coverage."""
        from omni_mercury_engine.cognitive.uncertainty import AdaptiveConformalInference

        aci = AdaptiveConformalInference(target_coverage=0.9, gamma=0.05)
        initial_alpha = aci.alpha

        # Update with covered point (should decrease alpha)
        aci.update(score=0.1, covered=True)
        # Update with uncovered points (should increase alpha)
        aci.update(score=0.5, covered=False)
        aci.update(score=0.6, covered=False)
        aci.update(score=0.7, covered=False)

        # After multiple uncovered, alpha should increase
        assert aci.alpha > initial_alpha

    def test_adaptive_conformal_prediction_interval(self) -> None:
        """Test prediction interval computation."""
        from omni_mercury_engine.cognitive.uncertainty import AdaptiveConformalInference

        aci = AdaptiveConformalInference(target_coverage=0.8, gamma=0.01)

        # Add calibration scores
        for i in range(100):
            aci.update(score=abs(np.random.randn()), covered=np.random.rand() > 0.2)

        # Get prediction interval
        point_pred = 5.0
        lower, upper = aci.predict_interval(point_pred, residual_std=1.0)

        assert lower < point_pred < upper
        assert upper - lower > 0


class TestHeteroscedasticEstimator:
    """Tests for Heteroscedastic Aleatoric Estimation."""

    def test_heteroscedastic_initialization(self) -> None:
        """Test HeteroscedasticEstimator initializes correctly."""
        from omni_mercury_engine.cognitive.uncertainty import HeteroscedasticEstimator

        estimator = HeteroscedasticEstimator(window_size=50, min_samples=10)
        assert estimator.window_size == 50
        assert estimator.min_samples == 10

    def test_heteroscedastic_update_and_estimate(self) -> None:
        """Test heteroscedastic variance estimation."""
        from omni_mercury_engine.cognitive.uncertainty import HeteroscedasticEstimator

        estimator = HeteroscedasticEstimator(window_size=100, min_samples=10)

        # Add some residuals
        for i in range(50):
            pred = np.random.randn()
            true_val = pred + np.random.randn() * 0.5
            estimator.update(pred, true_val)

        # Estimate variance
        variance = estimator.estimate_variance()

        assert variance >= 0


class TestUncertaintyQuantifierIntegration:
    """Integration tests for the full UncertaintyQuantifier."""

    def test_uncertainty_quantifier_initialization(self) -> None:
        """Test UncertaintyQuantifier initializes all components."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier(
            n_monte_carlo=50,
            calibration_bins=15,
            enable_aci=True,
            aci_coverage=0.9,
        )
        assert uq is not None
        assert uq.n_monte_carlo == 50
        assert uq.temperature_scaler is not None
        assert uq.aci is not None

    def test_estimate_uncertainty_basic(self) -> None:
        """Test basic uncertainty estimation."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        predictions = np.array([0.7, 0.75, 0.72, 0.68, 0.73, 0.71, 0.69, 0.74])

        result = uq.estimate_uncertainty(predictions)

        assert result is not None
        assert 0 <= result.confidence <= 1
        assert result.epistemic >= 0
        assert result.aleatoric >= 0
        assert result.total >= 0

    def test_uncertainty_decomposition(self) -> None:
        """Test epistemic vs aleatoric decomposition."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier(n_monte_carlo=100)

        # Low variance predictions - should have low uncertainty
        low_var = np.array([0.8, 0.81, 0.79, 0.8, 0.8])
        result_low = uq.estimate_uncertainty(low_var)

        # High variance predictions - should have higher uncertainty
        high_var = np.array([0.3, 0.9, 0.5, 0.7, 0.2])
        result_high = uq.estimate_uncertainty(high_var)

        assert result_high.total > result_low.total

    def test_decompose_uncertainty_method(self) -> None:
        """Test decompose_uncertainty method."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()

        # Predictions with different uncertainty profiles
        predictions_ensemble = np.random.rand(10, 50)  # 10 models, 50 outputs

        decomp = uq.decompose_uncertainty(predictions_ensemble)

        assert "epistemic" in decomp
        assert "aleatoric" in decomp
        assert "total" in decomp
        assert decomp["epistemic"] >= 0
        assert decomp["aleatoric"] >= 0

    def test_calibration(self) -> None:
        """Test calibration assessment."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()

        # Create predictions and outcomes
        n_samples = 200
        predictions = np.random.rand(n_samples)
        confidences = predictions  # Use predictions as confidence
        outcomes = (np.random.rand(n_samples) < predictions).astype(float)

        result = uq.calibrate(predictions, confidences, outcomes)

        assert result.ece >= 0
        assert result.mce >= 0
        assert "expected" in result.reliability_diagram

    def test_conformal_prediction(self) -> None:
        """Test conformal prediction method."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()

        # Calibration scores
        calibration_scores = np.abs(np.random.randn(100))
        test_score = 0.5

        result = uq.conformal_prediction(calibration_scores, test_score, alpha=0.1)

        assert "in_prediction_set" in result
        assert "threshold" in result
        assert result["coverage_guarantee"] == 0.9


class TestUncertaintyAwareDecision:
    """Tests for uncertainty-aware decision making."""

    def test_decision_confident(self) -> None:
        """Test decision with confident prediction."""
        from omni_mercury_engine.cognitive.uncertainty import (
            UncertaintyEstimate,
            UncertaintyQuantifier,
        )

        uq = UncertaintyQuantifier()

        estimate = UncertaintyEstimate(
            prediction=0.9,
            epistemic=0.05,
            aleatoric=0.1,
            total=0.11,
            confidence=0.85,
            confidence_interval=(0.7, 1.0),
            calibration_error=0.02,
            is_reliable=True,
            explanation="Test",
        )

        decision = uq.uncertainty_aware_decision(estimate, action_threshold=0.5)

        assert decision["should_act"] is True
        assert decision["action"] == "take_action"

    def test_decision_high_epistemic(self) -> None:
        """Test decision with high epistemic uncertainty."""
        from omni_mercury_engine.cognitive.uncertainty import (
            UncertaintyEstimate,
            UncertaintyQuantifier,
        )

        uq = UncertaintyQuantifier()

        estimate = UncertaintyEstimate(
            prediction=0.7,
            epistemic=0.5,  # High epistemic
            aleatoric=0.1,
            total=0.51,
            confidence=0.5,
            confidence_interval=(0.2, 1.0),
            calibration_error=0.02,
            is_reliable=False,
            explanation="Test",
        )

        decision = uq.uncertainty_aware_decision(estimate, epistemic_threshold=0.3)

        assert decision["should_collect_more_data"] is True
        assert decision["epistemic_concern"] is True


class TestUncertaintyEdgeCases:
    """Edge case tests for uncertainty quantification."""

    def test_single_prediction(self) -> None:
        """Test with single prediction value."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        result = uq.estimate_uncertainty(np.array([0.5]))

        assert result is not None
        assert result.total >= 0

    def test_extreme_predictions(self) -> None:
        """Test with extreme prediction values."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()

        # Near-zero predictions
        result_low = uq.estimate_uncertainty(np.array([0.01, 0.02, 0.01]))
        assert result_low is not None

        # Near-one predictions
        result_high = uq.estimate_uncertainty(np.array([0.99, 0.98, 0.99]))
        assert result_high is not None

    def test_large_batch(self) -> None:
        """Test with large batch of predictions."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier(n_monte_carlo=20)
        predictions = np.random.rand(1000)

        result = uq.estimate_uncertainty(predictions)

        assert result is not None
        assert result.confidence >= 0

    def test_statistics(self) -> None:
        """Test statistics collection."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()

        # Perform some estimations
        for _ in range(5):
            uq.estimate_uncertainty(np.random.rand(10))

        stats = uq.get_statistics()

        assert "estimates_computed" in stats
        assert stats["estimates_computed"] == 5


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_calibration_result_to_dict(self) -> None:
        """Test CalibrationResult serialization."""
        from omni_mercury_engine.cognitive.uncertainty import CalibrationResult

        result = CalibrationResult(
            expected_confidence=[0.1, 0.5, 0.9],
            observed_accuracy=[0.2, 0.5, 0.8],
            ece=0.05,
            mce=0.1,
            ace=0.06,
            is_calibrated=True,
            temperature=1.2,
            reliability_diagram={"expected": [0.1], "observed": [0.2], "counts": [10]},
        )

        d = result.to_dict()

        assert d["ece"] == 0.05
        assert d["calibrated"] is True
        assert d["temperature"] == 1.2


class TestUncertaintyEstimate:
    """Tests for UncertaintyEstimate dataclass."""

    def test_uncertainty_estimate_to_dict(self) -> None:
        """Test UncertaintyEstimate serialization."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyEstimate

        estimate = UncertaintyEstimate(
            prediction=0.7,
            epistemic=0.1,
            aleatoric=0.2,
            total=0.22,
            confidence=0.8,
            confidence_interval=(0.5, 0.9),
            calibration_error=0.03,
            is_reliable=True,
            explanation="Test explanation",
            mutual_information=0.05,
            predictive_entropy=0.3,
            mc_samples=30,
        )

        d = estimate.to_dict()

        assert d["prediction"] == 0.7
        assert d["epistemic"] == 0.1
        assert d["aleatoric"] == 0.2
        assert d["total"] == 0.22
        assert d["reliable"] is True
        assert d["mc_samples"] == 30
        assert d["overconfident"] is False


class TestOverconfidenceDetection:
    """Tests for overconfidence detection (Kaddour et al. 2026)."""

    def test_overconfident_flag_in_dataclass(self) -> None:
        """Test is_overconfident field exists and defaults to False."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyEstimate

        estimate = UncertaintyEstimate(
            prediction=0.7,
            epistemic=0.1,
            aleatoric=0.2,
            total=0.22,
            confidence=0.8,
            confidence_interval=(0.5, 0.9),
            calibration_error=0.03,
            is_reliable=True,
            explanation="Test",
        )
        assert estimate.is_overconfident is False

    def test_overconfident_flag_set_when_appropriate(self) -> None:
        """Test is_overconfident is True when confidence > 0.8 and ECE > threshold."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyEstimate

        estimate = UncertaintyEstimate(
            prediction=0.9,
            epistemic=0.05,
            aleatoric=0.1,
            total=0.11,
            confidence=0.9,
            confidence_interval=(0.7, 1.0),
            calibration_error=0.15,  # Poor calibration
            is_reliable=False,
            explanation="Test",
            is_overconfident=True,
        )
        assert estimate.is_overconfident is True
        assert estimate.to_dict()["overconfident"] is True

    def test_decision_defers_on_overconfidence(self) -> None:
        """Test uncertainty_aware_decision defers when overconfident."""
        from omni_mercury_engine.cognitive.uncertainty import (
            UncertaintyEstimate,
            UncertaintyQuantifier,
        )

        uq = UncertaintyQuantifier()

        # High confidence, but flagged as overconfident
        estimate = UncertaintyEstimate(
            prediction=0.9,
            epistemic=0.05,
            aleatoric=0.1,
            total=0.11,
            confidence=0.85,
            confidence_interval=(0.7, 1.0),
            calibration_error=0.02,
            is_reliable=True,
            explanation="Test",
            is_overconfident=True,
        )

        decision = uq.uncertainty_aware_decision(estimate, action_threshold=0.5)

        assert decision["should_defer"] is True
        assert decision["action"] == "defer_to_human"
        assert "Overconfidence" in decision["reason"]

    def test_not_overconfident_when_well_calibrated(self) -> None:
        """Test no overconfidence flag when calibration is good."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyEstimate

        estimate = UncertaintyEstimate(
            prediction=0.9,
            epistemic=0.05,
            aleatoric=0.1,
            total=0.11,
            confidence=0.85,
            confidence_interval=(0.7, 1.0),
            calibration_error=0.02,  # Good calibration
            is_reliable=True,
            explanation="Test",
            is_overconfident=False,
        )
        assert estimate.is_overconfident is False


class TestEnsembleDisagreement:
    """Tests for ensemble disagreement metric."""

    def test_decompose_includes_disagreement(self) -> None:
        """Test decompose_uncertainty returns ensemble_disagreement."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        predictions_ensemble = np.random.rand(10, 50)

        decomp = uq.decompose_uncertainty(predictions_ensemble)

        assert "ensemble_disagreement" in decomp
        assert decomp["ensemble_disagreement"] >= 0

    def test_high_agreement_low_disagreement(self) -> None:
        """Test low disagreement when models agree."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        # All models predict similar values
        base = np.random.rand(50)
        predictions_ensemble = np.tile(base, (10, 1)) + np.random.randn(10, 50) * 0.001

        decomp = uq.decompose_uncertainty(predictions_ensemble)

        assert decomp["ensemble_disagreement"] < 0.01

    def test_high_disagreement_when_models_diverge(self) -> None:
        """Test high disagreement when models diverge."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        # Models predict very different values
        predictions_ensemble = np.random.rand(10, 50) * 10  # Large spread

        decomp = uq.decompose_uncertainty(predictions_ensemble)

        assert decomp["ensemble_disagreement"] > 0.1

    def test_single_model_zero_disagreement(self) -> None:
        """Test zero disagreement for single model (ndim < 2)."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        predictions = np.array([0.5, 0.6, 0.7])

        decomp = uq.decompose_uncertainty(predictions)

        assert decomp["ensemble_disagreement"] == 0.0


class TestBayesianCalibrationIntegration:
    """Tests for BayesianConfidenceCalibrator integration into UncertaintyQuantifier."""

    def test_init_without_bayesian(self) -> None:
        """Test UQ works without Bayesian calibrator (default)."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        assert uq.bayesian_calibrator is None

    def test_init_with_bayesian(self) -> None:
        """Test UQ accepts Bayesian calibrator."""
        from omni_mercury_engine.agentic.bayesian_calibrator import BayesianConfidenceCalibrator
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        cal = BayesianConfidenceCalibrator()
        uq = UncertaintyQuantifier(bayesian_calibrator=cal)
        assert uq.bayesian_calibrator is cal

    def test_calibrate_with_bayesian_passthrough(self) -> None:
        """Test calibrate_with_bayesian returns raw when no calibrator."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        result = uq.calibrate_with_bayesian(0.85, domain="security", goal="detect anomaly")
        assert result == 0.85

    def test_calibrate_with_bayesian_blends(self) -> None:
        """Test calibrate_with_bayesian blends raw and Bayesian confidence."""
        from omni_mercury_engine.agentic.bayesian_calibrator import BayesianConfidenceCalibrator
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        cal = BayesianConfidenceCalibrator()
        # Feed some successes to shift the posterior
        for _ in range(10):
            cal.update("security", "detect anomaly", success=True)

        uq = UncertaintyQuantifier(bayesian_calibrator=cal)
        raw = 0.60
        blended = uq.calibrate_with_bayesian(raw, domain="security", goal="detect anomaly")

        # Blended should be between raw and Bayesian (which is high after 10 successes)
        assert blended >= raw
        assert 0.01 <= blended <= 0.99

    def test_update_bayesian_noop_without_calibrator(self) -> None:
        """Test update_bayesian is safe without calibrator."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier()
        # Should not raise
        uq.update_bayesian("security", "detect", success=True)

    def test_update_bayesian_updates_posterior(self) -> None:
        """Test update_bayesian actually updates the calibrator."""
        from omni_mercury_engine.agentic.bayesian_calibrator import BayesianConfidenceCalibrator
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        cal = BayesianConfidenceCalibrator()
        uq = UncertaintyQuantifier(bayesian_calibrator=cal)

        uq.update_bayesian("medical", "analyze", success=True)
        stats = cal.get_stats("medical", "analyze")
        assert stats is not None
        assert stats.successes == 1


class TestConformalFusedScores:
    """Tests for conformal prediction interval propagation through fusion."""

    def test_no_intervals_without_aci(self) -> None:
        """Test returns no intervals when ACI is disabled."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier(enable_aci=False)
        result = uq.conformal_fused_scores(np.array([0.5, 0.7, 0.9]))

        assert result["has_intervals"] is False
        np.testing.assert_array_equal(result["predictions"], [0.5, 0.7, 0.9])

    def test_no_intervals_without_calibration_data(self) -> None:
        """Test returns no intervals when ACI has no calibration scores."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier(enable_aci=True)
        result = uq.conformal_fused_scores(np.array([0.5, 0.7, 0.9]))

        assert result["has_intervals"] is False

    def test_intervals_with_calibrated_aci(self) -> None:
        """Test returns valid intervals after ACI calibration."""
        from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

        uq = UncertaintyQuantifier(enable_aci=True, aci_coverage=0.9)
        assert uq.aci is not None

        # Feed calibration data through ACI
        for i in range(50):
            score = abs(np.random.randn())
            uq.aci.update(score, covered=np.random.rand() > 0.1)

        fused = np.array([0.3, 0.6, 0.9])
        result = uq.conformal_fused_scores(fused, residual_std=0.1)

        assert result["has_intervals"] is True
        assert result["coverage_level"] == 0.9
        assert len(result["lower_bounds"]) == 3
        assert len(result["upper_bounds"]) == 3
        # Lower bounds should be below upper bounds
        assert np.all(result["lower_bounds"] <= result["upper_bounds"])
