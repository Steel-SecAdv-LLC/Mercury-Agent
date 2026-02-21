"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

LSTM-Autoencoder for Time-Series Anomaly Detection

A working anomaly detector that actually trains and detects.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from omni_mercury_engine.utils.logging import LoggerMixin

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection.

    Architecture:
        Encoder: LSTM layers that compress the input sequence
        Decoder: LSTM layers that reconstruct the input

    Anomaly Detection:
        High reconstruction error = anomaly
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation."""
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.encoder_lstm(x)
        # Use last layer hidden state
        latent = self.encoder_fc(h_n[-1])
        return latent

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent representation back to sequence."""
        # z: (batch, latent_dim)
        h = self.decoder_fc(z)
        # Repeat for sequence length
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(h)
        output = self.output_fc(decoded)
        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent."""
        z = self.encode(x)
        recon = self.decode(z, x.size(1))
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error."""
        recon, _ = self.forward(x)
        # MSE per sample
        error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error


class AnomalyDetector(LoggerMixin):
    """
    Complete anomaly detection pipeline using LSTM-Autoencoder.

    Usage:
        detector = AnomalyDetector(input_dim=38)
        detector.fit(train_data)
        scores = detector.predict(test_data)
        labels = detector.detect(test_data, threshold=0.95)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        seq_len: int = 100,
        device: str = "auto",
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
        ).to(self.device)

        self.threshold = None
        self.train_errors = None

    def _create_sequences(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Create overlapping sequences from data."""
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i : i + self.seq_len])
        return np.array(sequences)

    def fit(
        self,
        train_data: np.ndarray[Any, Any],
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        val_split: float = 0.1,
        early_stopping: int = 10,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Train the autoencoder on normal data.

        Args:
            train_data: Training data (n_samples, n_features) - should be mostly normal
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            val_split: Validation split ratio
            early_stopping: Patience for early stopping
            verbose: Print training progress

        Returns:
            Training history dict
        """
        # Create sequences
        sequences = self._create_sequences(train_data)
        if verbose:
            self.logger.info("Created %d sequences of length %d", len(sequences), self.seq_len)

        # Train/val split
        n_val = int(len(sequences) * val_split)
        indices = np.random.permutation(len(sequences))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_seqs = torch.FloatTensor(sequences[train_idx]).to(self.device)
        val_seqs = torch.FloatTensor(sequences[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(train_seqs),
            batch_size=batch_size,
            shuffle=True,
        )

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            for (batch,) in train_loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_recon, _ = self.model(val_seqs)
                val_loss = criterion(val_recon, val_seqs).item()

            train_loss = np.mean(train_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                self.logger.info(
                    "Epoch %d/%d - Train: %.6f, Val: %.6f", epoch + 1, epochs, train_loss, val_loss
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    if verbose:
                        self.logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Load best model
        if best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # Compute threshold from training data reconstruction errors
        self.model.eval()
        with torch.no_grad():
            train_errors = self.model.reconstruction_error(train_seqs).cpu().numpy()

        self.train_errors = train_errors  # type: ignore[assignment, unused-ignore]
        # Set threshold at 95th percentile of training errors
        self.threshold = np.percentile(train_errors, 95)

        if verbose:
            self.logger.info("Training complete. Threshold set to %.6f", self.threshold)

        return history

    def predict(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute anomaly scores for data.

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        sequences = self._create_sequences(data)
        seq_tensor = torch.FloatTensor(sequences).to(self.device)

        self.model.eval()
        with torch.no_grad():
            errors = self.model.reconstruction_error(seq_tensor).cpu().numpy()

        # Map sequence errors back to point-level
        # Each point appears in multiple sequences
        point_scores = np.zeros(len(data))
        point_counts = np.zeros(len(data))

        for i, error in enumerate(errors):
            point_scores[i : i + self.seq_len] += error
            point_counts[i : i + self.seq_len] += 1

        point_scores = point_scores / np.maximum(point_counts, 1)
        return point_scores

    def detect(
        self, data: np.ndarray[Any, Any], threshold: float | None = None
    ) -> np.ndarray[Any, Any]:
        """
        Detect anomalies in data.

        Args:
            data: Input data (n_samples, n_features)
            threshold: Anomaly threshold (uses training threshold if None)

        Returns:
            Binary labels (1 = anomaly, 0 = normal)
        """
        scores = self.predict(data)
        thresh = threshold if threshold is not None else self.threshold
        return (scores > thresh).astype(int)

    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "threshold": self.threshold,
                "input_dim": self.input_dim,
                "seq_len": self.seq_len,
                "hidden_dim": self.model.hidden_dim,
                "latent_dim": self.model.latent_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "auto") -> AnomalyDetector:
        """Load model from file."""
        checkpoint = torch.load(
            path, map_location="cpu"
        )  # nosec B614 - loading trusted model checkpoints
        detector = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
            seq_len=checkpoint["seq_len"],
            device=device,
        )
        detector.model.load_state_dict(checkpoint["model_state"])
        detector.threshold = checkpoint["threshold"]
        return detector


def evaluate_detector(
    y_true: np.ndarray[Any, Any],
    y_scores: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate anomaly detection performance.

    Args:
        y_true: Ground truth labels (1 = anomaly)
        y_scores: Anomaly scores
        y_pred: Predicted labels (optional, will compute at best threshold)

    Returns:
        Dictionary with precision, recall, f1, auc_roc, auc_pr
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {"error": "Only one class in y_true"}

    # AUC scores
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except (ValueError, TypeError):
        auc_roc = 0.5

    try:
        auc_pr = average_precision_score(y_true, y_scores)
    except (ValueError, TypeError):
        auc_pr = np.mean(y_true)

    # Find best F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    if y_pred is None:
        y_pred = (y_scores > best_threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "best_threshold": best_threshold,
        "best_f1": f1_scores[best_idx],
    }
