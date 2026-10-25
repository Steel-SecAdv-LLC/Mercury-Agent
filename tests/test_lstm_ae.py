# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for LSTM Autoencoder anomaly detection module."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from omni_mercury_engine.models.lstm_ae import (
    AnomalyDetector,
    LSTMAutoencoder,
    evaluate_detector,
)


class TestLSTMAutoencoder:
    """Tests for LSTMAutoencoder class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        model = LSTMAutoencoder(input_dim=10)
        assert model.input_dim == 10
        assert model.hidden_dim == 64
        assert model.latent_dim == 32
        assert model.num_layers == 2
        assert model.seq_len == 100

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        model = LSTMAutoencoder(
            input_dim=20,
            hidden_dim=128,
            latent_dim=64,
            num_layers=3,
            dropout=0.2,
            seq_len=50,
        )
        assert model.input_dim == 20
        assert model.hidden_dim == 128
        assert model.latent_dim == 64
        assert model.num_layers == 3
        assert model.seq_len == 50

    def test_encode(self) -> None:
        """Test encoding input to latent representation."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16, seq_len=20)
        model.eval()
        x = torch.randn(4, 20, 10)  # batch=4, seq_len=20, input_dim=10
        with torch.no_grad():
            latent = model.encode(x)
        assert latent.shape == (4, 16)  # batch, latent_dim

    def test_decode(self) -> None:
        """Test decoding latent representation to sequence."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16, seq_len=20)
        model.eval()
        z = torch.randn(4, 16)  # batch=4, latent_dim=16
        with torch.no_grad():
            output = model.decode(z, seq_len=20)
        assert output.shape == (4, 20, 10)  # batch, seq_len, input_dim

    def test_forward(self) -> None:
        """Test forward pass returning reconstruction and latent."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16, seq_len=20)
        model.eval()
        x = torch.randn(4, 20, 10)
        with torch.no_grad():
            recon, latent = model.forward(x)
        assert recon.shape == x.shape
        assert latent.shape == (4, 16)

    def test_reconstruction_error(self) -> None:
        """Test reconstruction error computation."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16, seq_len=20)
        model.eval()
        x = torch.randn(4, 20, 10)
        with torch.no_grad():
            error = model.reconstruction_error(x)
        assert error.shape == (4,)  # one error per sample
        assert torch.all(error >= 0)  # MSE is non-negative

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16, seq_len=20)
        model.train()
        x = torch.randn(4, 20, 10, requires_grad=True)
        recon, latent = model.forward(x)
        loss = torch.mean((x - recon) ** 2)
        loss.backward()
        assert x.grad is not None

    def test_different_batch_sizes(self) -> None:
        """Test model with different batch sizes."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16, seq_len=20)
        model.eval()
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 20, 10)
            with torch.no_grad():
                recon, latent = model.forward(x)
            assert recon.shape == (batch_size, 20, 10)
            assert latent.shape == (batch_size, 16)


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    def test_init_default_device(self) -> None:
        """Test initialization with auto device selection."""
        detector = AnomalyDetector(input_dim=10)
        assert detector.input_dim == 10
        assert detector.seq_len == 100
        assert detector.threshold is None
        assert detector.train_errors is None

    def test_init_cpu_device(self) -> None:
        """Test initialization with CPU device."""
        detector = AnomalyDetector(input_dim=10, device="cpu")
        assert detector.device == torch.device("cpu")

    def test_create_sequences(self) -> None:
        """Test sequence creation from data."""
        detector = AnomalyDetector(input_dim=5, seq_len=10)
        data = np.random.randn(100, 5)
        sequences = detector._create_sequences(data)
        assert sequences.shape == (91, 10, 5)  # 100 - 10 + 1 = 91 sequences

    def test_create_sequences_short_data(self) -> None:
        """Test sequence creation with short data."""
        detector = AnomalyDetector(input_dim=5, seq_len=10)
        data = np.random.randn(10, 5)
        sequences = detector._create_sequences(data)
        assert sequences.shape == (1, 10, 5)

    def test_fit_basic(self) -> None:
        """Test basic training."""
        detector = AnomalyDetector(input_dim=5, seq_len=10, hidden_dim=16, latent_dim=8)
        train_data = np.random.randn(200, 5).astype(np.float32)
        history = detector.fit(train_data, epochs=2, batch_size=16, verbose=False, early_stopping=5)
        assert "train_loss" in history
        assert "val_loss" in history
        assert detector.threshold is not None
        assert detector.train_errors is not None

    def test_predict(self) -> None:
        """Test anomaly score prediction."""
        detector = AnomalyDetector(input_dim=5, seq_len=10, hidden_dim=16, latent_dim=8)
        train_data = np.random.randn(200, 5).astype(np.float32)
        detector.fit(train_data, epochs=2, batch_size=16, verbose=False)

        test_data = np.random.randn(50, 5).astype(np.float32)
        scores = detector.predict(test_data)
        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    def test_detect(self) -> None:
        """Test anomaly detection."""
        detector = AnomalyDetector(input_dim=5, seq_len=10, hidden_dim=16, latent_dim=8)
        train_data = np.random.randn(200, 5).astype(np.float32)
        detector.fit(train_data, epochs=2, batch_size=16, verbose=False)

        test_data = np.random.randn(50, 5).astype(np.float32)
        labels = detector.detect(test_data)
        assert labels.shape == (50,)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_detect_custom_threshold(self) -> None:
        """Test detection with custom threshold."""
        detector = AnomalyDetector(input_dim=5, seq_len=10, hidden_dim=16, latent_dim=8)
        train_data = np.random.randn(200, 5).astype(np.float32)
        detector.fit(train_data, epochs=2, batch_size=16, verbose=False)

        test_data = np.random.randn(50, 5).astype(np.float32)
        labels = detector.detect(test_data, threshold=0.0)
        assert np.all(labels == 1)  # All should be anomalies with threshold=0

    def test_save_load(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Test model save and load."""
        detector = AnomalyDetector(input_dim=5, seq_len=10, hidden_dim=16, latent_dim=8)
        train_data = np.random.randn(200, 5).astype(np.float32)
        detector.fit(train_data, epochs=2, batch_size=16, verbose=False)

        save_path = str(tmp_path / "model.pt")
        detector.save(save_path)

        # Patch torch.load to use weights_only=False for this test
        original_load = torch.load

        def patched_load(path, **kwargs):
            kwargs["weights_only"] = False
            return original_load(path, **kwargs)

        monkeypatch.setattr(torch, "load", patched_load)

        loaded = AnomalyDetector.load(save_path, device="cpu")
        assert loaded.input_dim == detector.input_dim
        assert loaded.seq_len == detector.seq_len
        assert loaded.threshold == detector.threshold

        test_data = np.random.randn(50, 5).astype(np.float32)
        original_scores = detector.predict(test_data)
        loaded_scores = loaded.predict(test_data)
        np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)

    @pytest.mark.timeout(600)
    def test_early_stopping(self) -> None:
        """Test early stopping during training.

        Uses a high ``epochs`` ceiling on purpose so we can observe early
        stopping firing before the ceiling is reached.  Doing 100 LSTM-AE
        training epochs on a CPU-only GitHub-hosted runner under
        ``pytest -n 4`` parallelism can outrun the global 300 s
        pytest-timeout (see ``tests/test_fusion_training.py``'s sibling
        ``test_early_stopping_works`` for the same pattern).  Extending
        the per-test budget to 10 minutes preserves what the test
        actually exercises rather than weakening it by reducing
        ``epochs``.
        """
        detector = AnomalyDetector(input_dim=5, seq_len=10, hidden_dim=16, latent_dim=8)
        train_data = np.random.randn(200, 5).astype(np.float32)
        history = detector.fit(
            train_data, epochs=100, batch_size=16, verbose=False, early_stopping=2
        )
        assert len(history["train_loss"]) < 100


class TestEvaluateDetector:
    """Tests for evaluate_detector function."""

    def test_basic_evaluation(self) -> None:
        """Test basic evaluation metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.2, 0.3, 0.6, 0.1])
        result = evaluate_detector(y_true, y_scores)

        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "auc_roc" in result
        assert "auc_pr" in result
        assert "best_threshold" in result
        assert "best_f1" in result

    def test_perfect_predictions(self) -> None:
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        result = evaluate_detector(y_true, y_scores, y_pred)

        assert result["auc_roc"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_single_class(self) -> None:
        """Test with only one class in y_true."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = evaluate_detector(y_true, y_scores)

        assert "error" in result
        assert result["error"] == "Only one class in y_true"

    def test_with_predictions(self) -> None:
        """Test with provided predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        result = evaluate_detector(y_true, y_scores, y_pred)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_random_predictions(self) -> None:
        """Test with random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        result = evaluate_detector(y_true, y_scores)

        assert 0 <= result["auc_roc"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1


class TestLSTMAutoencoderEdgeCases:
    """Edge case tests for LSTM Autoencoder."""

    def test_single_layer(self) -> None:
        """Test with single LSTM layer (no dropout)."""
        model = LSTMAutoencoder(input_dim=10, num_layers=1, dropout=0.5)
        model.eval()
        x = torch.randn(4, 20, 10)
        with torch.no_grad():
            recon, latent = model.forward(x)
        assert recon.shape == x.shape

    def test_large_latent_dim(self) -> None:
        """Test with latent dim larger than hidden dim."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=64)
        model.eval()
        x = torch.randn(4, 20, 10)
        with torch.no_grad():
            recon, latent = model.forward(x)
        assert latent.shape == (4, 64)

    def test_variable_sequence_length(self) -> None:
        """Test decoding with different sequence lengths."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16)
        model.eval()
        z = torch.randn(4, 16)
        for seq_len in [10, 50, 100, 200]:
            with torch.no_grad():
                output = model.decode(z, seq_len=seq_len)
            assert output.shape == (4, seq_len, 10)


class TestHeldOutCalibrationAndGuards:
    """Short-input guard, chronological hold-out and the registry adapter."""

    def test_short_input_is_rejected_clearly(self) -> None:
        detector = AnomalyDetector(input_dim=3, seq_len=20, hidden_dim=8, latent_dim=4)
        with pytest.raises(ValueError, match="seq_len=20"):
            detector.fit(np.random.randn(10, 3), epochs=1, verbose=False)

    def test_tiny_series_trains_without_nan(self) -> None:
        detector = AnomalyDetector(input_dim=2, seq_len=10, hidden_dim=8, latent_dim=4)
        history = detector.fit(np.random.randn(25, 2), epochs=2, verbose=False)
        assert all(np.isfinite(history["val_loss"]))
        assert detector.threshold is not None and np.isfinite(detector.threshold)

    def test_threshold_is_cut_on_pooled_out_of_fold_point_scores(self) -> None:
        """The cut covers the whole series, one block at a time.

        Each of the five blocks is scored by a model trained on the other
        four, so every pooled score is out-of-sample and the pool spans the
        series. Cutting on a single contiguous tail understated the quantile
        on autocorrelated data.
        """
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        detector = AnomalyDetector(
            input_dim=2, seq_len=10, hidden_dim=8, latent_dim=4, device="cpu", seed=0
        )
        detector.fit(rng.normal(size=(400, 2)).astype(np.float32), epochs=3, verbose=False)
        assert detector.calibration_scores is not None
        # Five blocks of 80 rows, each scored point-wise out of fold.
        assert len(detector.calibration_scores) == 400
        ranked = np.sort(detector.calibration_scores)
        k = int(np.ceil(401 * 0.95))
        assert detector.threshold == pytest.approx(float(ranked[k - 1]))

    def test_a_fold_model_trains_without_calibrating_a_threshold(self) -> None:
        """``calibration_folds=0`` is the internal path for fold models."""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        detector = AnomalyDetector(
            input_dim=2, seq_len=10, hidden_dim=8, latent_dim=4, device="cpu", seed=0
        )

        detector.fit(
            rng.normal(size=(200, 2)).astype(np.float32),
            epochs=2,
            calibration_folds=0,
            verbose=False,
        )

        assert detector.threshold is None
        assert detector.calibration_scores is None
        # Still a usable model: it just has no decision threshold of its own.
        assert np.all(np.isfinite(detector.predict(rng.normal(size=(50, 2)).astype(np.float32))))

    def test_evaluation_reports_threshold_source(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        assert evaluate_detector(y, s)["threshold_source"].startswith("oracle")
        assert evaluate_detector(y, s, threshold=0.5)["threshold_source"] == "provided_threshold"

    def test_registry_adapter(self) -> None:
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
        from omni_mercury_engine.models.lstm_ae import LSTMAutoencoderDetector

        entry = next(e for e in DETECTOR_MANIFEST if e.name == "lstm_autoencoder")
        assert entry.class_name == "LSTMAutoencoderDetector"
        rng = np.random.default_rng(1)
        t = np.arange(400) * 2 * np.pi / 40
        X = np.column_stack([np.sin(t), np.cos(t)]) + rng.normal(0, 0.05, (400, 2))
        det = LSTMAutoencoderDetector(seq_len=20, epochs=3)
        assert not det.is_fitted
        det.fit(X)
        out = det.predict(X[:100])
        assert out["scores"].shape == (100,)
        assert det.extract_features(X[:100]).shape == (100, 1)
        with pytest.raises(ValueError, match="seq_len"):
            det.predict(X[:5])


class TestInputScaling:
    """Inputs are standardised by the training rows, so the unit of a series is irrelevant."""

    @staticmethod
    def _series(seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.arange(400)
        return np.stack(
            [np.sin(t / 7.0) + 0.05 * rng.normal(size=400), np.cos(t / 11.0)], axis=1
        ).astype(np.float32)

    def _fitted(self, data: np.ndarray) -> AnomalyDetector:
        torch.manual_seed(0)
        detector = AnomalyDetector(input_dim=2, seq_len=10, hidden_dim=16, latent_dim=8, seed=0)
        detector.fit(data, epochs=3, batch_size=32, verbose=False)
        return detector

    def test_statistics_come_from_the_training_rows_only(self) -> None:
        data = self._series()
        detector = self._fitted(data)
        n = len(data)
        n_holdout = int(n * 0.1)
        fit_rows = data[: n - n_holdout - 10]
        assert detector.mean_ is not None and detector.scale_ is not None
        np.testing.assert_allclose(detector.mean_, fit_rows.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(detector.scale_, fit_rows.std(axis=0), rtol=1e-4)

    def test_the_unit_of_the_series_does_not_change_its_scores(self) -> None:
        """Millions and units of the same signal score alike (raw, they differed by 1e12)."""
        data = self._series()
        shifted = (data * np.float32(1e6) + np.float32(5e6)).astype(np.float32)
        small, large = self._fitted(data), self._fitted(shifted)
        np.testing.assert_allclose(
            small.predict(data), large.predict(shifted), rtol=1e-2, atol=1e-3
        )

    def test_a_constant_feature_is_not_divided_by_zero(self) -> None:
        data = self._series()
        data[:, 1] = 3.0
        detector = self._fitted(data)
        assert detector.scale_ is not None
        assert detector.scale_[1] == 1.0
        assert np.isfinite(detector.predict(data)).all()

    def test_the_scaling_round_trips_through_save_and_load(self, tmp_path: Path) -> None:
        data = self._series()
        detector = self._fitted(data)
        detector.save(str(tmp_path / "ae.pt"))
        restored = AnomalyDetector.load(str(tmp_path / "ae.pt"), device="cpu")
        assert restored.mean_ is not None and restored.scale_ is not None
        assert detector.mean_ is not None and detector.scale_ is not None
        np.testing.assert_allclose(restored.mean_, detector.mean_)
        np.testing.assert_allclose(restored.scale_, detector.scale_)
        np.testing.assert_allclose(restored.predict(data), detector.predict(data), rtol=1e-5)

    def test_a_checkpoint_saved_before_scaling_is_served_on_raw_values(
        self, tmp_path: Path
    ) -> None:
        data = self._series()
        detector = self._fitted(data)
        path = str(tmp_path / "old.pt")
        # What an older build wrote: no statistics.
        detector.mean_ = detector.scale_ = None
        detector.save(path)
        restored = AnomalyDetector.load(path, device="cpu")
        assert restored.mean_ is None and restored.scale_ is None
        np.testing.assert_allclose(restored.predict(data), detector.predict(data), rtol=1e-5)
