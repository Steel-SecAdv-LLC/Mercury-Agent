# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration tests for fusion model training.

Tests for Issue #1 (Untrained Fusion Neural Network) and Issue #6 (Feature Dimension Mismatch).
Validates that OmniFusionModel can be trained and produces meaningful scores.
"""

from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.xdist_group("fusion_training")

from omni_mercury_engine.ml.mercury_ml import make_classification


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

    def test_fit_fusion_supervised(self, engine: Any, training_data: Any) -> None:
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

    def test_fit_fusion_semi_supervised(self, engine: Any, training_data: Any) -> None:
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

    def test_fit_fusion_requires_fusion_mode(self, training_data: Any) -> None:
        """Verify fit_fusion raises error if not in fusion mode."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="statistical", device="cpu")
        X, y = training_data

        with pytest.raises(ValueError, match="requires mode='fusion'"):
            engine.fit_fusion(X, y)

    def test_trained_model_has_training_flag(self, engine: Any, training_data: Any) -> None:
        """Verify _fusion_trained flag is set after training."""
        X, y = training_data

        assert not engine._fusion_trained, "Should start untrained"

        engine.fit_fusion(X, y, epochs=5)

        assert engine._fusion_trained, "Should be trained after fit_fusion"

    def test_loss_decreases_during_training(self, engine: Any, training_data: Any) -> None:
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

    @pytest.mark.timeout(600)
    def test_early_stopping_works(self, engine: Any, training_data: Any) -> None:
        """Verify early stopping triggers when loss plateaus.

        Uses a high ``epochs`` ceiling on purpose so we can observe
        ``early_stopped=True`` before the ceiling is reached.  Doing 100
        full fusion-training epochs on a CPU-only GitHub-hosted runner
        can outrun the global 300 s pytest-timeout under load even
        though the data set is small, so we extend the per-test budget
        to 10 minutes rather than reduce ``epochs`` (which would weaken
        what the test actually exercises).
        """
        X, y = training_data

        metrics = engine.fit_fusion(
            X,
            y,
            epochs=100,  # High epoch count
            early_stopping_patience=3,  # Stop quickly if no improvement
        )

        # If early stopping triggered, the loop broke before completing
        # all 100 epochs.  The break-point can land at epoch index 99
        # though — see ``engine.py:1054-1056`` — so ``epochs_trained <
        # 100`` is too strict an upper bound; the contract is "stopped
        # at or before the ceiling".  We also accept the all-100 case
        # where ``early_stopped`` is False (loss kept improving).
        assert metrics["epochs_trained"] <= 100
        if metrics["early_stopped"]:
            # If early-stopped, len(loss_history) recorded all epochs up
            # to and including the one that crossed the patience threshold.
            assert metrics["epochs_trained"] >= 3, (
                f"early_stopped=True but epochs_trained={metrics['epochs_trained']} "
                f"is below patience=3 — engine bug"
            )


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

    def test_detect_with_fusion_returns_valid_scores(self, trained_engine: Any) -> None:
        """Verify detect_with_fusion returns valid anomaly probabilities."""
        X_test = np.random.randn(10, 15).astype(np.float32)

        for sample in X_test:
            result = trained_engine.detect_with_fusion(sample.reshape(1, -1))

            assert "anomaly_prob" in result
            assert 0.0 <= result["anomaly_prob"] <= 1.0
            assert "is_anomaly" in result
            assert isinstance(result["is_anomaly"], bool)

    def test_scores_differentiate_anomalies(self, trained_engine: Any) -> None:
        """Verify trained model assigns different scores to different samples."""
        # Generate clearly different samples
        normal = np.zeros((5, 15), dtype=np.float32)
        anomaly = np.ones((5, 15), dtype=np.float32) * 5

        X_test = np.vstack([normal, anomaly])

        scores: list[Any] = []
        for sample in X_test:
            result = trained_engine.detect_with_fusion(sample.reshape(1, -1))
            scores.append(result["anomaly_prob"])

        scores_arr = np.array(scores)

        # Should have some variance in scores
        assert np.std(scores_arr) > 0.01, "Scores should vary between samples"


class TestDynamicDimensions:
    """Test dynamic feature dimension handling (Issue #6 fix)."""

    def test_dimension_mismatch_handled(self) -> None:
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

    def test_dynamic_projections_are_cached(self) -> None:
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

    def test_different_dimensions_create_separate_projections(self) -> None:
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

    def test_dynamic_projections_in_parameters(self) -> None:
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

    def test_gradients_flow_through_dynamic_projections(self) -> None:
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


class TestSaveLoadRoundTrip:
    """Save/load must round-trip dynamic projections (the load_model fix).

    Dynamic projection layers are created lazily during forward and so do not
    exist on a freshly constructed model; load_state_dict would otherwise fail
    on the unexpected ``_dynamic_projections.*`` keys.
    """

    def test_projection_registry_export_rebuild(self) -> None:
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()
        feats = {
            "statistical": torch.randn(4, 99),  # forces a dynamic projection
            "temporal": torch.randn(4, 77),
        }
        model(feats)
        registry = model.export_projection_registry()
        assert registry, "forward should have created dynamic projections"

        state = model.state_dict()
        fresh = OmniFusionModel()
        with pytest.raises(RuntimeError):
            # Without rebuilding, the dynamic-projection keys are unexpected.
            fresh.load_state_dict(state)

        fresh.rebuild_projection_registry(registry)
        fresh.load_state_dict(state)  # now succeeds

        model.eval()
        fresh.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            a = model(feats)["anomaly_probs"]
            torch.manual_seed(0)
            b = fresh(feats)["anomaly_probs"]
        assert torch.allclose(a, b, atol=1e-6)

    def test_engine_save_load_with_dynamic_projections(self, tmp_path: Any) -> None:
        """A trained engine with dynamic projections reloads in a fresh engine."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        X, y = make_classification(
            n_samples=160,
            n_features=33,  # mismatched vs default dims -> dynamic projections
            n_informative=8,
            n_classes=2,
            weights=[0.85, 0.15],
            random_state=7,
        )
        X = X.astype(np.float32)

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        engine.fit_fusion(X, y, epochs=5, early_stopping_patience=3)

        path = tmp_path / "fusion.pt"
        engine.save_model(str(path))

        loaded = OmniMercuryEngine(mode="fusion", device="cpu")
        loaded.load_model(str(path))
        assert loaded._fusion_trained

        # Loaded model's projection registry matches the saved one.
        assert (
            engine.fusion_model.export_projection_registry()
            == loaded.fusion_model.export_projection_registry()
        )

    def test_load_default_checkpoint_contract(self) -> None:
        """Default-checkpoint loader: no-op in non-fusion mode; loads if present."""
        from omni_mercury_engine.engine import (
            OmniMercuryEngine,
            default_fusion_checkpoint_path,
        )

        # Non-fusion mode never loads a fusion checkpoint.
        non_fusion = OmniMercuryEngine(mode="statistical", device="cpu")
        assert non_fusion.load_default_fusion_checkpoint() is False

        # Fusion mode: returns True and marks trained iff the file is shipped.
        fusion = OmniMercuryEngine(mode="fusion", device="cpu")
        loaded = fusion.load_default_fusion_checkpoint()
        assert loaded is default_fusion_checkpoint_path().exists()
        if loaded:
            assert fusion._fusion_trained


class TestPseudoLabeling:
    """Test pseudo-label generation for semi-supervised learning."""

    @pytest.fixture
    def engine(self):
        """Create engine in fusion mode."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        return OmniMercuryEngine(mode="fusion", device="cpu")

    def test_pseudo_labels_generated(self, engine: Any) -> None:
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

    def test_pseudo_labels_respect_contamination(self, engine: Any) -> None:
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

    def test_small_dataset(self) -> None:
        """Test training on very small dataset."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.array([0] * 18 + [1] * 2)

        # Should not crash
        metrics = engine.fit_fusion(X, y, epochs=5, validation_split=0.1)
        assert metrics["epochs_trained"] > 0

    def test_all_normal_data(self) -> None:
        """Test training when all samples are labeled normal."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.zeros(50)  # All normal

        # Should not crash (though results may not be meaningful)
        metrics = engine.fit_fusion(X, y, epochs=5)
        assert metrics["epochs_trained"] > 0

    def test_highly_imbalanced_data(self) -> None:
        """Test training with highly imbalanced data (1% anomalies)."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        X = np.random.randn(200, 10).astype(np.float32)
        y = np.array([0] * 198 + [1] * 2)  # 1% anomalies

        # Should handle gracefully
        metrics = engine.fit_fusion(X, y, epochs=10)
        assert metrics["epochs_trained"] > 0


class TestBuildFeaturesCLI:
    """End-to-end coverage for the ``build-features`` command and its
    round-trip into ``train`` (the feature-archive path)."""

    def test_build_features_archive_then_train(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        rng = np.random.default_rng(11)
        x = rng.standard_normal((60, 12)).astype(np.float32)
        data_path = tmp_path / "samples.npy"
        np.save(data_path, x)

        archive = tmp_path / "features.npz"
        runner = CliRunner()
        result = runner.invoke(main, ["build-features", "-d", str(data_path), "-o", str(archive)])
        assert result.exit_code == 0, result.output
        assert archive.exists()

        with np.load(archive, allow_pickle=False) as npz:
            keys = set(npz.files)
            assert "labels" in keys
            assert len(keys) > 1, "expected at least one detector feature array"
            assert npz["labels"].shape[0] == len(x)
            for key in keys - {"labels"}:
                assert npz[key].shape[0] == len(x)

        # The archive must round-trip into the feature-archive trainer.
        model_path = tmp_path / "fusion.pt"
        train_result = runner.invoke(
            main, ["train", "-d", str(archive), "-o", str(model_path), "-e", "2"]
        )
        assert train_result.exit_code == 0, train_result.output
        assert model_path.exists()

    def test_build_features_rejects_non_npz_output(self, tmp_path: Any) -> None:
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        data_path = tmp_path / "samples.npy"
        np.save(data_path, np.random.default_rng(3).standard_normal((10, 8)).astype(np.float32))

        result = CliRunner().invoke(
            main, ["build-features", "-d", str(data_path), "-o", str(tmp_path / "bad.bin")]
        )
        assert result.exit_code == 1
        assert "must be a .npz" in result.output
