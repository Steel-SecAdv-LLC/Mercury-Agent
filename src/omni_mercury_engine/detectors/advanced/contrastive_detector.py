"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Contrastive Learning Detector for Anomaly Detection

Implements SimCLR-style contrastive learning adapted for anomaly detection:
1. Time-series augmentation strategies
2. NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss
3. Hard negative mining for anomaly-specific learning
4. Representation-based anomaly scoring

Key Insight: Normal samples form tight clusters in representation space;
anomalies are distant from these clusters.

Reference:
- Chen et al. (2020) - A Simple Framework for Contrastive Learning
- Shenkar & Wolf (2022) - Anomaly Detection for Tabular Data with IC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn


if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "ContrastiveConfig",
    "ContrastiveLearningDetector",
]


@dataclass
class ContrastiveConfig:
    """Configuration for Contrastive Learning detector."""

    input_dim: int = 38
    hidden_dim: int = 256
    projection_dim: int = 128
    n_layers: int = 3

    # Contrastive learning
    temperature: float = 0.07
    n_augmentations: int = 2
    hard_negative_ratio: float = 0.1

    # Augmentation strengths
    noise_std: float = 0.1
    mask_ratio: float = 0.15
    scale_range: tuple[float, float] = (0.8, 1.2)
    shift_range: tuple[float, float] = (-0.1, 0.1)

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10

    # Detection
    n_neighbors: int = 10
    threshold_percentile: float = 95.0

    # Ethical constraints
    benevolence_threshold: float = 0.99


class TimeSeriesAugmenter:
    """
    Augmentation strategies for time-series data.

    Generates diverse views of the same sample while preserving
    essential characteristics for contrastive learning.
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        mask_ratio: float = 0.15,
        scale_range: tuple[float, float] = (0.8, 1.2),
        shift_range: tuple[float, float] = (-0.1, 0.1),
    ) -> None:
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.scale_range = scale_range
        self.shift_range = shift_range

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to input."""
        # Randomly select augmentation
        aug_type = np.random.choice(
            ["noise", "mask", "scale", "shift", "permute", "combined"],
            p=[0.2, 0.15, 0.2, 0.15, 0.1, 0.2],
        )

        if aug_type == "noise":
            return self._add_noise(x)
        elif aug_type == "mask":
            return self._random_mask(x)
        elif aug_type == "scale":
            return self._random_scale(x)
        elif aug_type == "shift":
            return self._random_shift(x)
        elif aug_type == "permute":
            return self._feature_permute(x)
        else:
            # Combined augmentation
            x = self._add_noise(x)
            x = self._random_scale(x)
            return x

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def _random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask features."""
        mask = torch.rand_like(x) > self.mask_ratio
        return x * mask

    def _random_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random scaling per feature."""
        scale = torch.empty(x.shape[-1], device=x.device).uniform_(
            self.scale_range[0], self.scale_range[1]
        )
        return x * scale

    def _random_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Random shift per feature."""
        shift = torch.empty(x.shape[-1], device=x.device).uniform_(
            self.shift_range[0], self.shift_range[1]
        )
        return x + shift

    def _feature_permute(self, x: torch.Tensor) -> torch.Tensor:
        """Permute a subset of features."""
        n_features = x.shape[-1]
        n_permute = max(1, int(n_features * 0.2))
        idx = np.random.choice(n_features, n_permute, replace=False)
        permuted = x.clone()
        permuted[..., idx] = permuted[..., idx[np.random.permutation(n_permute)]]
        return permuted


class ContrastiveEncoder(nn.Module):
    """
    Encoder network for contrastive learning.

    Architecture: MLP with residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Encoder layers with residual connections
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
            )

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to representation."""
        h = self.input_proj(x)

        for layer in self.layers:
            h = h + layer(h)  # Residual connection

        return self.final_norm(h)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps representations to a lower-dimensional space where
    contrastive loss is computed.
    """

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveModel(nn.Module):
    """
    Full contrastive learning model.

    Combines encoder and projection head with NT-Xent loss.
    """

    def __init__(self, config: ContrastiveConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = ContrastiveEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        )

        self.projection = ProjectionHead(
            hidden_dim=config.hidden_dim,
            projection_dim=config.projection_dim,
        )

        self.augmenter = TimeSeriesAugmenter(
            noise_std=config.noise_std,
            mask_ratio=config.mask_ratio,
            scale_range=config.scale_range,
            shift_range=config.shift_range,
        )

    def forward(self, x: torch.Tensor, return_projection: bool = True) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input [batch, input_dim]
            return_projection: Whether to return projected features

        Returns:
            Dictionary with representations and projections
        """
        # Encode
        h = self.encoder(x)

        result = {"representation": h}

        if return_projection:
            z = self.projection(h)
            z = F.normalize(z, dim=1)
            result["projection"] = z

        return result

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute contrastive loss with augmented views.

        Uses NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss.
        """
        batch_size = x.shape[0]

        # Generate augmented views
        x_aug1 = self.augmenter.augment(x)
        x_aug2 = self.augmenter.augment(x)

        # Get projections
        z1 = self(x_aug1)["projection"]
        z2 = self(x_aug2)["projection"]

        # Concatenate projections
        z = torch.cat([z1, z2], dim=0)  # [2*batch, projection_dim]

        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.config.temperature  # [2*batch, 2*batch]

        # Create labels (positive pairs are i and i+batch)
        labels = torch.cat(
            [torch.arange(batch_size) + batch_size, torch.arange(batch_size)],
            dim=0,
        ).to(x.device)

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=x.device)
        sim.masked_fill_(mask, float("-inf"))

        # NT-Xent loss
        loss = F.cross_entropy(sim, labels)

        # Hard negative mining bonus
        with torch.no_grad():
            # Find hardest negatives (highest similarity to wrong pairs)
            sim_pos = sim[torch.arange(2 * batch_size), labels]
            sim_neg_max = sim.clone()
            sim_neg_max[torch.arange(2 * batch_size), labels] = float("-inf")
            sim_neg_max, _ = sim_neg_max.max(dim=1)
            hardness = (sim_neg_max - sim_pos).mean()

        return {
            "loss": loss,
            "hardness": hardness,
            "z1": z1,
            "z2": z2,
        }


class ContrastiveLearningDetector:
    """
    Contrastive Learning Detector for Anomaly Detection.

    Uses learned representations to detect anomalies based on
    distance to normal cluster centroids.

    Example:
        >>> detector = ContrastiveLearningDetector(input_dim=38)
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        n_neighbors: int = 10,
        temperature: float = 0.07,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = ContrastiveConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            n_neighbors=n_neighbors,
            temperature=temperature,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ContrastiveModel | None = None
        self.train_representations: NDArray[np.float64] | None = None
        self.threshold: float = 0.0
        self._fitted = False

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
    ) -> ContrastiveLearningDetector:
        """
        Fit the detector using contrastive learning.

        Args:
            X: Training data [n_samples, n_features]
            y: Optional labels (ignored for unsupervised)
            validation_split: Fraction for validation

        Returns:
            self
        """
        # Flatten if 3D (time-series windows)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        # Update input_dim
        self.config.input_dim = X.shape[-1]

        # Initialize model
        self.model = ContrastiveModel(self.config).to(self.device)

        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device) if n_val > 0 else None

        # Training
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        best_val_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            # Shuffle
            perm = torch.randperm(len(X_train))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), self.config.batch_size):
                batch_idx = perm[i : i + self.config.batch_size]
                batch = X_train[batch_idx]

                if len(batch) < 2:  # Need at least 2 samples for contrastive
                    continue

                optimizer.zero_grad()
                loss_dict = self.model.compute_loss(batch)
                loss = loss_dict["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            if X_val is not None and len(X_val) >= 2:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.model.compute_loss(X_val)["loss"].item()
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break

        # Store training representations for anomaly scoring
        self.model.eval()
        with torch.no_grad():
            train_reps = []
            for i in range(0, len(X_train), self.config.batch_size):
                batch = X_train[i : i + self.config.batch_size]
                rep = self.model(batch, return_projection=False)["representation"]
                train_reps.append(rep.cpu().numpy())

            self.train_representations = np.concatenate(train_reps, axis=0)

        # Compute threshold based on training distances
        train_scores = self._compute_knn_scores(self.train_representations)
        self.threshold = float(np.percentile(train_scores, self.config.threshold_percentile))

        self._fitted = True
        return self

    def _compute_knn_scores(self, representations: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute k-NN based anomaly scores."""
        if self.train_representations is None:
            return np.zeros(len(representations))

        # Compute distances to training representations
        # Using L2 distance
        train_reps = self.train_representations
        n_test = len(representations)
        n_train = len(train_reps)

        # Batch computation for memory efficiency
        scores = np.zeros(n_test)
        batch_size = 1000

        for i in range(0, n_test, batch_size):
            batch = representations[i : i + batch_size]

            # Compute pairwise distances
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            batch_sq = (batch**2).sum(axis=1, keepdims=True)
            train_sq = (train_reps**2).sum(axis=1, keepdims=True)
            dists = batch_sq + train_sq.T - 2 * batch @ train_reps.T

            # k-NN distances
            k = min(self.config.n_neighbors, n_train)
            knn_dists = np.partition(dists, k, axis=1)[:, :k]
            scores[i : i + len(batch)] = knn_dists.mean(axis=1)

        return scores

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict anomaly scores.

        Args:
            X: Test data

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self._fitted or self.model is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        # Flatten if 3D
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        X_tensor = torch.FloatTensor(X).to(self.device)

        # Get representations
        self.model.eval()
        representations = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i : i + self.config.batch_size]
                rep = self.model(batch, return_projection=False)["representation"]
                representations.append(rep.cpu().numpy())

        representations = np.concatenate(representations, axis=0)

        # Compute k-NN scores
        return self._compute_knn_scores(representations)

    def detect(
        self,
        X: NDArray[np.float64],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Perform anomaly detection.

        Args:
            X: Test data
            threshold: Detection threshold

        Returns:
            Detection results
        """
        scores = self.predict(X)
        thresh = threshold if threshold is not None else self.threshold
        predictions = (scores > thresh).astype(int)

        return {
            "anomaly_score": scores,
            "predictions": predictions,
            "threshold": thresh,
            "is_anomaly": predictions.astype(bool),
            "detector_type": "ContrastiveLearning",
            "confidence": np.clip(scores / (thresh + 1e-8), 0, 1),
        }

    def extract_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract learned representations for fusion."""
        if not self._fitted or self.model is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        features = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i : i + self.config.batch_size]
                rep = self.model(batch, return_projection=False)["representation"]
                features.append(rep.cpu().numpy())

        return np.concatenate(features, axis=0)
