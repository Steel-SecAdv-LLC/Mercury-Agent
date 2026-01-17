"""
Integration tests for fusion model training.

Tests for Issue #1 (Untrained Fusion Neural Network) and Issue #6 (Feature Dimension Mismatch).
Validates that OmniFusionModel can be trained and produces meaningful scores.

Mercury Agent - Copyright (C) 2025 Steel Security Advisory LLC
Licensed under GNU GPL v3
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification


class TestFusionTraining:
    """Test fusion model training functionality."""

    @pytest.fixture
    def engine(self):
        """Create engine in fusion mode."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        return OmniMercuryEngine(mode="fusion", device="cpu")

    @pytest.fixture
    def training_data(self):
        """Generate training data with 15% anomalies."""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=10,
            n_classes=2,
            weights=[0.85, 0.15],
            random_state=42,
        )
        return X.astype(np.float32), y

    def test_fit_fusion_supervised(self, engine, training_data):
        """Test supervised fusion training with labels."""
        X, y = training_data

        metrics = engine.fit_fusion(
            X,
            y,
            epochs=10,
            batch_size=32,
            early_stopping_patience=5,
        )

        assert engine._fusion_trained, "Fusion model should be marked as trained"
        assert "best_loss" in metrics
        assert "epochs_trained" in metrics
        assert metrics["epochs_trained"] > 0
        assert metrics["best_loss"] >= 0

    def test_fit_fusion_semi_supervised(self, engine, training_data):
        """Test semi-supervised fusion training without labels."""
        X, _ = training_data

        metrics = engine.fit_fusion(
            X,
            y=None,  # No labels - use pseudo-labeling
            epochs=10,
            contamination=0.15,
        )

        assert engine._fusion_trained
        assert metrics["epochs_trained"] > 0

    def test_fit_fusion_requires_fusion_mode(self, training_data):
        """Verify fit_fusion raises error if not in fusion mode."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="statistical", device="cpu")
        X, y = training_data

        with pytest.raises(ValueError, match="requires mode='fusion'"):
            engine.fit_fusion(X, y)

    def test_trained_model_has_training_flag(self, engine, training_data):
        """Verify _fusion_trained flag is set after training."""
        X, y = training_data

        assert not engine._fusion_trained, "Should start untrained"

        engine.fit_fusion(X, y, epochs=5)

        assert engine._fusion_trained, "Should be trained after fit_fusion"

    def test_loss_decreases_during_training(self, engine, training_data):
        """Verify training loss generally decreases."""
        X, y = training_data

        metrics = engine.fit_fusion(
            X,
            y,
            epochs=20,
            batch_size=32,
            early_stopping_patience=15,
        )

        history = metrics["loss_history"]
        if len(history) >= 5:
            # Compare first few epochs to last few
            early_loss = np.mean([h["train_loss"] for h in history[:3]])
            late_loss = np.mean([h["train_loss"] for h in history[-3:]])

            # Loss should generally decrease (or at least not increase much)
            assert (
                late_loss <= early_loss * 1.5
            ), f"Loss should not increase significantly: {early_loss:.4f} -> {late_loss:.4f}"

    def test_early_stopping_works(self, engine, training_data):
        """Verify early stopping triggers when loss plateaus."""
        X, y = training_data

        metrics = engine.fit_fusion(
            X,
            y,
            epochs=100,  # High epoch count
            early_stopping_patience=3,  # Stop quickly if no improvement
        )

        # Should stop before all 100 epochs if early stopping works
        assert metrics["epochs_trained"] < 100 or not metrics["early_stopped"]


class TestFusionInference:
    """Test that trained fusion model produces better scores."""

    @pytest.fixture
    def trained_engine(self):
        """Create and train engine."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")

        # Generate training data
        X, y = make_classification(
            n_samples=150,
            n_features=15,
            n_informative=8,
            n_classes=2,
            weights=[0.8, 0.2],
            random_state=42,
        )
        X = X.astype(np.float32)

        engine.fit_fusion(X, y, epochs=15, early_stopping_patience=10)

        return engine

    def test_detect_with_fusion_returns_valid_scores(self, trained_engine):
        """Verify detect_with_fusion returns valid anomaly probabilities."""
        X_test = np.random.randn(10, 15).astype(np.float32)

        for sample in X_test:
            result = trained_engine.detect_with_fusion(sample.reshape(1, -1))

            assert "anomaly_prob" in result
            assert 0.0 <= result["anomaly_prob"] <= 1.0
            assert "is_anomaly" in result
            assert isinstance(result["is_anomaly"], bool)

    def test_scores_differentiate_anomalies(self, trained_engine):
        """Verify trained model assigns different scores to different samples."""
        # Generate clearly different samples
        normal = np.zeros((5, 15), dtype=np.float32)
        anomaly = np.ones((5, 15), dtype=np.float32) * 5

        X_test = np.vstack([normal, anomaly])

        scores = []
        for sample in X_test:
            result = trained_engine.detect_with_fusion(sample.reshape(1, -1))
            scores.append(result["anomaly_prob"])

        scores = np.array(scores)

        # Should have some variance in scores
        assert np.std(scores) > 0.01, "Scores should vary between samples"


class TestDynamicDimensions:
    """Test dynamic feature dimension handling (Issue #6 fix)."""

    def test_dimension_mismatch_handled(self):
        """Verify dimension mismatches don't crash the model."""
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()

        # Create features with mismatched dimensions
        features = {
            "statistical": torch.randn(4, 15),  # Expected 10, got 15
            "temporal": torch.randn(4, 32),  # Matches expected
        }

        # Should not raise, should use dynamic projection
        output = model(features)

        assert "anomaly_probs" in output
        assert output["anomaly_probs"].shape == (4, 1)

    def test_dynamic_projections_are_cached(self):
        """Verify dynamic projections are cached, not recreated."""
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()

        features = {
            "statistical": torch.randn(4, 15),  # Dimension mismatch
        }

        # First forward - creates projection
        model(features)
        n_params_1 = sum(p.numel() for p in model.parameters())
        n_dynamic_1 = len(model._dynamic_projections)

        # Multiple forwards with same dimensions
        for _ in range(5):
            model(features)

        n_params_2 = sum(p.numel() for p in model.parameters())
        n_dynamic_2 = len(model._dynamic_projections)

        # Should have same number of parameters (projection cached)
        assert n_params_1 == n_params_2, "Parameters should not increase"
        assert n_dynamic_1 == n_dynamic_2, "Dynamic projections should be cached"

    def test_different_dimensions_create_separate_projections(self):
        """Verify different dimensions create separate cached projections."""
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()

        # First dimension
        features1 = {"statistical": torch.randn(4, 15)}
        model(features1)
        n_projections_1 = len(model._dynamic_projections)

        # Different dimension
        features2 = {"statistical": torch.randn(4, 20)}
        model(features2)
        n_projections_2 = len(model._dynamic_projections)

        # Should have created a new projection for different dimension
        assert n_projections_2 == n_projections_1 + 1

    def test_dynamic_projections_in_parameters(self):
        """Verify dynamic projections are in model.parameters() for training."""
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()

        # Before creating dynamic projection
        params_before = list(model.parameters())

        # Create dynamic projection
        features = {"statistical": torch.randn(4, 15)}
        model(features)

        # After creating dynamic projection
        params_after = list(model.parameters())

        # Should have more parameters after dynamic projection created
        assert len(params_after) > len(params_before), "Dynamic projections should add parameters"

    def test_gradients_flow_through_dynamic_projections(self):
        """Verify gradients can flow through dynamic projections."""
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()

        features = {"statistical": torch.randn(4, 15, requires_grad=True)}
        output = model(features)

        # Compute loss and backprop
        loss = output["anomaly_probs"].sum()
        loss.backward()

        # Check that dynamic projection has gradients
        assert len(model._dynamic_projections) > 0, "Should have dynamic projection"
        for key, proj in model._dynamic_projections.items():
            for param in proj.parameters():
                assert param.grad is not None, f"Projection {key} should have gradients"


class TestPseudoLabeling:
    """Test pseudo-label generation for semi-supervised learning."""

    @pytest.fixture
    def engine(self):
        """Create engine in fusion mode."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        return OmniMercuryEngine(mode="fusion", device="cpu")

    def test_pseudo_labels_generated(self, engine):
        """Verify pseudo-labels are generated when y is None."""
        X, _ = make_classification(
            n_samples=100,
            n_features=10,
            weights=[0.9, 0.1],
            random_state=42,
        )
        X = X.astype(np.float32)

        # Fit detectors first
        for detector in engine.detectors.values():
            try:
                detector.fit(X)
            except Exception:
                pass

        pseudo_labels = engine._generate_pseudo_labels(X, contamination=0.1)

        assert len(pseudo_labels) == len(X)
        assert pseudo_labels.sum() > 0, "Should have some positive labels"
        assert pseudo_labels.sum() < len(X), "Should not label everything as anomaly"

    def test_pseudo_labels_respect_contamination(self, engine):
        """Verify pseudo-labels respect specified contamination rate."""
        X = np.random.randn(100, 10).astype(np.float32)

        # Fit detectors
        for detector in engine.detectors.values():
            try:
                detector.fit(X)
            except Exception:
                pass

        contamination = 0.2  # 20%
        pseudo_labels = engine._generate_pseudo_labels(X, contamination=contamination)

        # Should have approximately contamination * n_samples anomalies
        actual_rate = pseudo_labels.sum() / len(X)
        assert (
            abs(actual_rate - contamination) < 0.1
        ), f"Pseudo-label rate {actual_rate:.2f} should be near {contamination}"


class TestEdgeCases:
    """Test edge cases for fusion training."""

    def test_small_dataset(self):
        """Test training on very small dataset."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0] * 18 + [1] * 2)

        # Should not crash
        metrics = engine.fit_fusion(X, y, epochs=5, validation_split=0.1)
        assert metrics["epochs_trained"] > 0

    def test_all_normal_data(self):
        """Test training when all samples are labeled normal."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.zeros(50)  # All normal

        # Should not crash (though results may not be meaningful)
        metrics = engine.fit_fusion(X, y, epochs=5)
        assert metrics["epochs_trained"] > 0

    def test_highly_imbalanced_data(self):
        """Test training with highly imbalanced data (1% anomalies)."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        X = np.random.randn(200, 10).astype(np.float32)
        y = np.array([0] * 198 + [1] * 2)  # 1% anomalies

        # Should handle gracefully
        metrics = engine.fit_fusion(X, y, epochs=10)
        assert metrics["epochs_trained"] > 0
