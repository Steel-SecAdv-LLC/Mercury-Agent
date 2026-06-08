# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for LSTM Autoencoder anomaly detection module."""

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
