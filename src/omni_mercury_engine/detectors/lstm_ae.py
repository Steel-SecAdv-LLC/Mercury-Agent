# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LSTM-Autoencoder for Time-Series Anomaly Detection.

A working anomaly detector that actually trains and detects.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.security.safe_torch import safe_torch_load
from omni_mercury_engine.utils.logging import LoggerMixin

logger = logging.getLogger(__name__)

_FIT_EVENTS = itertools.count(1)


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for time-series anomaly detection.

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
        """Initialize the instance."""
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
    """Complete anomaly detection pipeline using LSTM-Autoencoder.

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
        seed: int | None = None,
    ):
        """Initialize the instance."""
        self.input_dim = input_dim
        self.seq_len = seq_len
        self._rng: np.random.Generator = np.random.default_rng(seed)

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

        self.threshold: float | None = None
        #: Per-feature mean and scale of the training windows' rows; inputs are
        #: standardised by them.  ``None`` for a checkpoint saved before this
        #: existed, which was trained on raw values and is served as such.
        self.mean_: np.ndarray[Any, Any] | None = None
        self.scale_: np.ndarray[Any, Any] | None = None
        self.train_errors: np.ndarray[Any, Any] | None = None
        #: Point scores of the held-out tail the threshold was cut on.
        self.calibration_scores: np.ndarray[Any, Any] | None = None

    def _check_length(self, data: np.ndarray[Any, Any], what: str) -> None:
        """Reject input with fewer rows than one sequence (no window fits)."""
        if len(data) < self.seq_len:
            raise ValueError(
                f"{what} needs at least seq_len={self.seq_len} rows to form one "
                f"sequence; got {len(data)}"
            )

    def _standardise(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """``data`` in the training rows' units (identity for a pre-scaling checkpoint)."""
        if self.mean_ is None or self.scale_ is None:
            return data
        return ((data - self.mean_) / self.scale_).astype(np.float32)

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
        """Train the autoencoder on normal data.

        The split is chronological: the last ``val_split`` of the series is
        held out, separated from the training windows by ``seq_len`` rows so
        no held-out point appears in a training window (a random split of
        overlapping windows shares almost every point between the two
        sides). Early stopping watches the held-out loss, and the decision
        threshold is the one-class conformal 95% cut of the held-out
        *point* scores -- the quantity :meth:`detect` compares against.
        With too little data to hold anything out, training loss drives
        early stopping and the threshold falls back to the in-sample
        point scores (logged).

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

        Raises:
            ValueError: If *train_data* has fewer than ``seq_len`` rows.
        """
        train_data = np.asarray(train_data, dtype=np.float32)
        self._check_length(train_data, "fit")
        n = len(train_data)
        n_holdout = int(n * val_split)
        # The held-out tail must hold at least one window after the gap.
        if n_holdout < self.seq_len or n - n_holdout - self.seq_len < self.seq_len:
            n_holdout = 0
        if n_holdout:
            raw_fit_part = train_data[: n - n_holdout - self.seq_len]
            raw_holdout = train_data[n - n_holdout :]
        else:
            raw_fit_part, raw_holdout = train_data, None

        # Raw values of any scale (a series of 1e7 or of 1e-3) cannot be
        # reconstructed by a network with bounded activations: on a recorded
        # road-traffic file the unscaled model ranked its anomaly windows at
        # AUROC 0.10, below chance, and 0.95 once standardised.  The statistics
        # come from the training windows only.
        self.mean_ = raw_fit_part.mean(axis=0)
        spread = raw_fit_part.std(axis=0)
        self.scale_ = np.where(spread > 1e-12, spread, 1.0)
        fit_part = self._standardise(raw_fit_part)
        holdout = None if raw_holdout is None else self._standardise(raw_holdout)

        sequences = self._create_sequences(fit_part)
        if verbose:
            self.logger.info("Created %d sequences of length %d", len(sequences), self.seq_len)
        order = self._rng.permutation(len(sequences))
        train_seqs = torch.FloatTensor(sequences[order]).to(self.device)
        val_seqs = (
            torch.FloatTensor(self._create_sequences(holdout)).to(self.device)
            if holdout is not None
            else None
        )

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

            train_loss = float(np.mean(train_losses))

            # Validation (held-out tail; training loss when nothing is held out)
            self.model.eval()
            if val_seqs is not None:
                with torch.no_grad():
                    val_recon, _ = self.model(val_seqs)
                    val_loss = criterion(val_recon, val_seqs).item()
            else:
                val_loss = train_loss

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

        self.model.eval()
        with torch.no_grad():
            self.train_errors = self.model.reconstruction_error(train_seqs).cpu().numpy()

        # Threshold: one-class conformal 95% cut of point scores (the unit
        # detect() compares), on the held-out tail when there is one.
        if raw_holdout is not None:
            calibration = self.predict(raw_holdout)
        else:
            self.logger.warning(
                "fit: series too short to hold out a calibration tail; the "
                "threshold uses in-sample point scores (optimistic)"
            )
            calibration = self.predict(raw_fit_part)
        self.calibration_scores = calibration
        ranked = np.sort(calibration)
        k = min(max(int(np.ceil((len(ranked) + 1) * 0.95)), 1), len(ranked))
        self.threshold = float(ranked[k - 1])

        if verbose:
            self.logger.info("Training complete. Threshold set to %.6f", self.threshold)

        return history

    def predict(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute anomaly scores for data.

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Anomaly scores (higher = more anomalous)

        Raises:
            ValueError: If *data* has fewer than ``seq_len`` rows.
        """
        data = np.asarray(data, dtype=np.float32)
        self._check_length(data, "predict")
        sequences = self._create_sequences(self._standardise(data))
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
        """Detect anomalies in data.

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
                "mean": None if self.mean_ is None else self.mean_.tolist(),
                "scale": None if self.scale_ is None else self.scale_.tolist(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "auto") -> AnomalyDetector:
        """Load model from file.

        Hard-pinned to ``weights_only=True``; checkpoints saved by an
        older code path that need to round-trip must be re-saved by
        the operator.
        """
        checkpoint = safe_torch_load(path, map_location="cpu")
        detector = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
            seq_len=checkpoint["seq_len"],
            device=device,
        )
        detector.model.load_state_dict(checkpoint["model_state"])
        detector.threshold = checkpoint["threshold"]
        # Absent in a checkpoint saved before inputs were standardised.
        mean, scale = checkpoint.get("mean"), checkpoint.get("scale")
        if mean is not None and scale is not None:
            detector.mean_ = np.asarray(mean, dtype=np.float32)
            detector.scale_ = np.asarray(scale, dtype=np.float32)
        return detector


class LSTMAutoencoderDetector:
    """Detector-registry adapter for the LSTM-autoencoder.

    The only trained multivariate sequence-reconstruction detector in the
    registry. On multivariate series with injected bursts, level shifts and
    frequency changes it reached AUROC 0.85 against 0.79 for the statistical
    tier, and 0.94 on frequency changes the statistical tier misses (0.54).
    Like every reconstruction model it does not see a broken correlation
    between channels whose individual shapes stay plausible (0.47).

    Args:
        seq_len: Window length (rows); inputs need at least this many rows.
        hidden_dim: LSTM hidden size.
        latent_dim: Bottleneck size.
        epochs: Training epochs (early stopping on a held-out tail).
        batch_size: Training batch size.
        lr: Learning rate.
        seed: Seed for weight init and window shuffling (deterministic fit).
        device: Torch device (``"cpu"`` by default for reproducibility).
    """

    decision_source = DecisionSource(
        kind="trained",
        description=(
            "Per-row LSTM-autoencoder reconstruction scores decided at a "
            "threshold selected when the model is trained: the one-class "
            "conformal 95% cut of point scores on a chronologically held-out "
            "calibration tail (in-sample, with a logged warning, when the series "
            "is too short to hold one out)."
        ),
        operating_point=(
            "No fixed shipped checkpoint: the model and its cut are re-derived "
            "at every fit. Selection record: AnomalyDetector.fit in this module "
            "(early stopping on held-out loss, best state restored, threshold = "
            "one-class conformal 95% quantile of held-out point scores, kept in "
            "calibration_scores), plus the benchmark record in this class "
            "docstring (AUROC 0.85 vs 0.79 for the statistical tier on injected "
            "multivariate faults)."
        ),
    )

    def __init__(
        self,
        seq_len: int = 30,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        """Store the configuration; the model is built at fit time."""
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device
        self._detector: AnomalyDetector | None = None
        self._fit_event: int = 0

    def feature_cache_token(self) -> tuple[Any, ...] | None:
        """Unique per fit (process-wide), for the registry's feature cache."""
        return None if self._detector is None else ("lstm_ae", self._fit_event)

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has trained a model."""
        return self._detector is not None

    @staticmethod
    def _as_matrix(data: Any) -> np.ndarray[Any, Any]:
        arr = np.asarray(
            data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data,
            dtype=np.float32,
        )
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    def fit(self, data: Any) -> LSTMAutoencoderDetector:
        """Train on (mostly normal) rows ordered in time."""
        X = self._as_matrix(data)
        torch.manual_seed(self.seed)
        detector = AnomalyDetector(
            input_dim=X.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
            device=self.device,
            seed=self.seed,
        )
        detector.fit(X, epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, verbose=False)
        self._detector = detector
        self._fit_event = next(_FIT_EVENTS)
        return self

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        """Per-row reconstruction score as an ``(n, 1)`` feature column."""
        return self.predict(data)["scores"].reshape(-1, 1).astype(np.float32)

    def predict(self, data: Any) -> dict[str, Any]:
        """Per-row reconstruction scores and decisions at the held-out threshold.

        Raises:
            ValueError: If unfitted, or *data* has fewer than ``seq_len`` rows.
        """
        if self._detector is None:
            raise ValueError("LSTMAutoencoderDetector must be fitted before predict")
        scores = self._detector.predict(self._as_matrix(data))
        threshold = float(self._detector.threshold or 0.0)
        return {"scores": scores, "is_anomaly": scores > threshold, "threshold": threshold}

    detect = predict


def evaluate_detector(
    y_true: np.ndarray[Any, Any],
    y_scores: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate anomaly detection performance.

    Precision / recall / F1 are reported at, in order of preference: the
    caller's *y_pred*, the deployable *threshold* (e.g. the detector's own),
    or -- only when neither is given -- the best-F1 threshold chosen ON THE
    EVALUATION LABELS. That last choice peeks at the answers; the result
    says so in ``threshold_source`` and is a diagnostic, never a claim.

    Args:
        y_true: Ground truth labels (1 = anomaly)
        y_scores: Anomaly scores
        y_pred: Predicted labels (optional)
        threshold: Deployable decision threshold (optional)

    Returns:
        Dictionary with precision, recall, f1, auc_roc, auc_pr,
        best_threshold / best_f1 (oracle diagnostics) and threshold_source
    """
    from omni_mercury_engine.ml.mercury_ml import (
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

    if y_pred is not None:
        threshold_source = "provided_predictions"
    elif threshold is not None:
        y_pred = (y_scores > threshold).astype(int)
        threshold_source = "provided_threshold"
    else:
        y_pred = (y_scores > best_threshold).astype(int)
        threshold_source = "oracle_best_f1_on_evaluation_labels"

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
        "threshold_source": threshold_source,
    }
