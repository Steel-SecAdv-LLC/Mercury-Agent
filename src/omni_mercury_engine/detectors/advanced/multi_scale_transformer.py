"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Multi-Scale Transformer for Time-Series Anomaly Detection

Addresses the time-series gap (F1 0.15-0.25 → target 0.70+) by:
1. Multi-scale temporal pattern extraction (local + global)
2. Cross-scale attention for pattern fusion
3. Reconstruction + forecasting dual-objective
4. Adaptive threshold calibration with point-adjustment

Architecture inspired by:
- MAAT (Mamba Adaptive Anomaly Transformer, 2025)
- TranAD (VLDB 2022)
- Anomaly Transformer (ICLR 2022)

Performance Target: SMD F1 > 0.70, SMAP/MSL F1 > 0.85
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "MultiScaleTransformerConfig",
    "MultiScaleTransformerDetector",
]


@dataclass
class MultiScaleTransformerConfig:
    """Configuration for Multi-Scale Transformer detector."""

    # Model dimensions
    input_dim: int = 38  # SMD default
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1

    # Multi-scale configuration
    window_sizes: list[int] = field(default_factory=lambda: [10, 25, 50])
    use_cross_scale_attention: bool = True

    # Detection configuration
    reconstruction_weight: float = 0.5
    forecasting_weight: float = 0.3
    association_weight: float = 0.2

    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10

    # Threshold calibration
    threshold_percentile: float = 95.0
    use_point_adjustment: bool = True

    # Ethical constraints
    benevolence_threshold: float = 0.99
    sigma_immutable: float = 0.96


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learnable scale."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:, : x.size(1), :]  # type: ignore[index, unused-ignore]
        x = x + self.scale * pe
        return self.dropout(x)


class TemporalConvBlock(nn.Module):
    """Temporal convolution block for local pattern extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, channels] -> [batch, channels, seq]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder extracting patterns at different temporal resolutions."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        window_sizes: list[int],
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_sizes = window_sizes
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList()
        for ws in window_sizes:
            # Temporal conv for local patterns
            conv_block = TemporalConvBlock(d_model, d_model, kernel_size=min(ws, 7))

            # Transformer for global patterns
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.scale_encoders.append(
                nn.ModuleDict({"conv": conv_block, "transformer": transformer})
            )

        # Positional encoding per scale
        self.pos_encodings = nn.ModuleList(
            [PositionalEncoding(d_model, max_len=ws * 2, dropout=dropout) for ws in window_sizes]
        )

        # Scale fusion (simple concatenation + projection)
        self.scale_fusion = nn.Linear(d_model * len(window_sizes), d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Multi-scale encoding.

        Args:
            x: Input [batch, seq, input_dim]

        Returns:
            Fused representation and list of scale-specific representations
        """
        x = self.input_projection(x)

        scale_outputs = []
        for i, encoder in enumerate(self.scale_encoders):
            # Apply positional encoding
            x_scale = self.pos_encodings[i](x)

            # Local patterns via convolution
            x_local = encoder["conv"](x_scale)  # type: ignore[index, unused-ignore]

            # Global patterns via transformer
            x_global = encoder["transformer"](x_local)  # type: ignore[index, unused-ignore]

            scale_outputs.append(x_global)

        # Fuse scales
        fused = torch.cat(scale_outputs, dim=-1)
        fused = self.scale_fusion(fused)

        return fused, scale_outputs


class CrossScaleAttention(nn.Module):
    """Cross-scale attention for learning inter-scale dependencies."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        attn_out, _ = self.cross_attn(query, key, value)
        x = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class AssociationDiscrepancy(nn.Module):
    """
    Association Discrepancy module from Anomaly Transformer.

    Measures the discrepancy between prior-association and series-association
    to identify anomalies that break normal temporal patterns.
    """

    def __init__(self, d_model: int, n_heads: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Prior-association (learnable Gaussian kernel)
        self.prior_scale = nn.Parameter(torch.ones(n_heads, 1, 1))

        # Series-association (Q, K projections)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute association discrepancy.

        Args:
            x: Input [batch, seq, d_model]

        Returns:
            Prior association and series association matrices
        """
        batch_size, seq_len, _ = x.shape

        # Prior association (Gaussian kernel)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        prior_assoc = torch.exp(-distances / (2 * self.prior_scale**2 + 1e-8))
        prior_assoc = prior_assoc / (prior_assoc.sum(dim=-1, keepdim=True) + 1e-8)

        # Series association (attention-based)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Attention scores
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        series_assoc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        series_assoc = F.softmax(series_assoc, dim=-1)

        return prior_assoc, series_assoc

    def compute_discrepancy(self, prior: torch.Tensor, series: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between prior and series associations."""
        # Average over heads: series is [batch, heads, seq, seq]
        series_avg = series.mean(dim=1)  # [batch, seq, seq]
        batch_size = series_avg.shape[0]

        # Prior is [n_heads, seq, seq] - take mean over heads first
        if prior.dim() == 3 and prior.shape[0] == self.n_heads:
            prior_avg = prior.mean(dim=0)  # [seq, seq]
        else:
            prior_avg = prior

        # Expand prior for batch
        prior_expanded = prior_avg.unsqueeze(0).expand(batch_size, -1, -1)

        # Symmetric KL divergence
        kl_forward = F.kl_div(
            torch.log(series_avg + 1e-8),
            prior_expanded,
            reduction="none",
        ).sum(dim=-1)
        kl_backward = F.kl_div(
            torch.log(prior_expanded + 1e-8),
            series_avg,
            reduction="none",
        ).sum(dim=-1)

        discrepancy = (kl_forward + kl_backward) / 2
        return discrepancy


class MultiScaleTransformerModel(nn.Module):
    """
    Multi-Scale Transformer for Time-Series Anomaly Detection.

    Combines:
    1. Multi-scale temporal encoding
    2. Cross-scale attention
    3. Association discrepancy
    4. Reconstruction + forecasting decoders
    """

    def __init__(self, config: MultiScaleTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Multi-scale encoder
        self.encoder = MultiScaleEncoder(
            input_dim=config.input_dim,
            d_model=config.d_model,
            window_sizes=config.window_sizes,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Cross-scale attention (optional)
        self.cross_scale_attn = (
            CrossScaleAttention(config.d_model, config.n_heads, config.dropout)
            if config.use_cross_scale_attention
            else None
        )

        # Association discrepancy
        self.association = AssociationDiscrepancy(config.d_model, config.n_heads)

        # Reconstruction decoder
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.input_dim),
        )

        # Forecasting decoder (predicts next timestep)
        self.forecasting_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.input_dim),
        )

    def forward(self, x: torch.Tensor, return_all: bool = False) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input [batch, seq, input_dim]
            return_all: Return all intermediate outputs

        Returns:
            Dictionary with anomaly scores and reconstructions
        """
        # Multi-scale encoding
        encoded, scale_outputs = self.encoder(x)

        # Cross-scale attention
        if self.cross_scale_attn is not None and len(scale_outputs) > 1:
            # Attend from finest to coarsest scale
            for i in range(len(scale_outputs) - 1):
                encoded = self.cross_scale_attn(
                    encoded,
                    scale_outputs[i + 1],
                    scale_outputs[i + 1],
                )

        # Association discrepancy
        prior_assoc, series_assoc = self.association(encoded)
        discrepancy = self.association.compute_discrepancy(prior_assoc, series_assoc)

        # Reconstruction
        reconstruction = self.reconstruction_decoder(encoded)

        # Forecasting (shift by 1)
        forecast = self.forecasting_decoder(encoded[:, :-1, :])

        # Compute anomaly scores
        recon_error = ((x - reconstruction) ** 2).mean(dim=-1)

        # Forecasting error (shifted target)
        target = x[:, 1:, :]
        forecast_error = ((target - forecast) ** 2).mean(dim=-1)
        # Pad to match sequence length
        forecast_error = F.pad(forecast_error, (0, 1), value=0.0)

        # Combined anomaly score
        anomaly_score = (
            self.config.reconstruction_weight * recon_error
            + self.config.forecasting_weight * forecast_error
            + self.config.association_weight * discrepancy
        )

        result = {
            "anomaly_score": anomaly_score,
            "reconstruction": reconstruction,
            "reconstruction_error": recon_error,
            "forecast_error": forecast_error,
            "discrepancy": discrepancy,
        }

        if return_all:
            result["encoded"] = encoded
            result["prior_association"] = prior_assoc
            result["series_association"] = series_assoc
            result["scale_outputs"] = scale_outputs

        return result


class MultiScaleTransformerDetector:
    """
    Multi-Scale Transformer Detector for Time-Series Anomaly Detection.

    Provides Mercury-compatible interface with fit/predict methods.

    Example:
        >>> detector = MultiScaleTransformerDetector(input_dim=38)
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
        >>> predictions = detector.detect(X_test, threshold=0.95)
    """

    def __init__(
        self,
        input_dim: int = 38,
        window_sizes: list[int] | None = None,
        d_model: int = 128,
        n_heads: int = 8,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        device: str | None = None,
        use_point_adjustment: bool = True,
        **kwargs: Any,
    ) -> None:
        self.config = MultiScaleTransformerConfig(
            input_dim=input_dim,
            window_sizes=window_sizes or [10, 25, 50],
            d_model=d_model,
            n_heads=n_heads,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_point_adjustment=use_point_adjustment,
            **kwargs,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MultiScaleTransformerModel | None = None
        self.threshold: float = 0.0
        self._fitted = False

    def _create_windows(self, data: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
        """Create sliding windows from time series."""
        n_samples = len(data) - window_size + 1
        windows = np.zeros((n_samples, window_size, data.shape[1]))
        for i in range(n_samples):
            windows[i] = data[i : i + window_size]
        return windows

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
    ) -> MultiScaleTransformerDetector:
        """
        Fit the detector on training data.

        Args:
            X: Training data [n_samples, n_features] or [n_samples, seq_len, n_features]
            y: Optional labels (ignored for unsupervised)
            validation_split: Fraction for validation

        Returns:
            self
        """
        # Handle 2D input (convert to windows)
        if X.ndim == 2:
            window_size = max(self.config.window_sizes)
            X = self._create_windows(X, window_size)

        # Update input_dim from data
        self.config.input_dim = X.shape[-1]

        # Initialize model
        self.model = MultiScaleTransformerModel(self.config).to(self.device)

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
            # Shuffle training data
            perm = torch.randperm(len(X_train))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), self.config.batch_size):
                batch_idx = perm[i : i + self.config.batch_size]
                batch = X_train[batch_idx]

                optimizer.zero_grad()
                result = self.model(batch)

                # Combined loss
                loss = result["reconstruction_error"].mean()
                loss += 0.5 * result["forecast_error"].mean()
                loss += 0.1 * result["discrepancy"].mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_result = self.model(X_val)
                    val_loss = val_result["reconstruction_error"].mean().item()
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break

        # Compute threshold on training data
        self.model.eval()
        with torch.no_grad():
            train_scores = []
            for i in range(0, len(X_train), self.config.batch_size):
                batch = X_train[i : i + self.config.batch_size]
                result = self.model(batch)
                train_scores.append(result["anomaly_score"].cpu().numpy())

            train_scores = np.concatenate([s.flatten() for s in train_scores])  # type: ignore[assignment, unused-ignore]
            self.threshold = float(np.percentile(train_scores, self.config.threshold_percentile))

        self._fitted = True
        return self

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

        # Handle 2D input
        if X.ndim == 2:
            window_size = max(self.config.window_sizes)
            X = self._create_windows(X, window_size)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i : i + self.config.batch_size]
                result = self.model(batch)
                scores.append(result["anomaly_score"].cpu().numpy())

        return np.concatenate([s.flatten() for s in scores])

    def detect(
        self,
        X: NDArray[np.float64],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Perform anomaly detection.

        Args:
            X: Test data
            threshold: Detection threshold (uses fitted threshold if None)

        Returns:
            Detection results with scores, predictions, and metadata
        """
        scores = self.predict(X)
        thresh = threshold if threshold is not None else self.threshold

        predictions = (scores > thresh).astype(int)

        # Apply point-adjustment if enabled
        if self.config.use_point_adjustment:
            predictions = _point_adjust(predictions, scores, thresh)

        return {
            "anomaly_score": scores,
            "predictions": predictions,
            "threshold": thresh,
            "is_anomaly": predictions.astype(bool),
            "detector_type": "MultiScaleTransformer",
            "confidence": np.clip(1.0 - np.exp(-scores / thresh), 0, 1),
        }

    def extract_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract learned features for fusion."""
        if not self._fitted or self.model is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        if X.ndim == 2:
            window_size = max(self.config.window_sizes)
            X = self._create_windows(X, window_size)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        features = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i : i + self.config.batch_size]
                result = self.model(batch, return_all=True)
                # Use mean pooled encoded representation
                feat = result["encoded"].mean(dim=1).cpu().numpy()
                features.append(feat)

        return np.concatenate(features, axis=0)


def _point_adjust(
    predictions: NDArray[np.int64],
    scores: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.int64]:
    """
    Point-adjustment for time-series evaluation.

    If any point in an anomaly segment is detected, mark the entire segment.
    This is standard practice for time-series anomaly detection evaluation.
    """
    adjusted = predictions.copy()

    # Find anomaly segments (consecutive 1s in predictions or high scores)
    in_segment = False
    segment_start = 0

    for i in range(len(predictions)):
        if scores[i] > threshold * 0.8:  # Lower threshold for segment detection
            if not in_segment:
                in_segment = True
                segment_start = i
        elif in_segment:
            # Check if any point in segment was predicted as anomaly
            if predictions[segment_start:i].sum() > 0:
                adjusted[segment_start:i] = 1
            in_segment = False

    # Handle last segment
    if in_segment and predictions[segment_start:].sum() > 0:
        adjusted[segment_start:] = 1

    return adjusted
