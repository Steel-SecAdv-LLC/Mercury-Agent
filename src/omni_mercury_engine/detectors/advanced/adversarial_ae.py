"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Adversarial Autoencoder for Industrial Control System Anomaly Detection

Addresses the industrial control gap (F1 0.30-0.45 → target 0.80+) by:
1. Adversarial regularization for distribution matching
2. Sensor correlation modeling via covariance-aware encoding
3. Multi-scale reconstruction for capturing process dynamics
4. Temporal consistency constraints

Architecture inspired by:
- Makhzani et al. (2015) - Adversarial Autoencoders
- Li et al. (2021) - MAD-GAN for multivariate anomaly detection
- AE+GWO (2025) - Grey Wolf optimized autoencoders

Performance Target: BATADAL F1 > 0.80, SWaT F1 > 0.90
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "AdversarialAEConfig",
    "AdversarialAutoencoderDetector",
]


@dataclass
class AdversarialAEConfig:
    """Configuration for Adversarial Autoencoder detector."""

    input_dim: int = 51  # SWaT default
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    latent_dim: int = 16

    # Adversarial configuration
    discriminator_dims: list[int] = field(default_factory=lambda: [32, 16])
    adversarial_weight: float = 0.5
    prior_type: str = "gaussian"  # "gaussian", "mixture", "uniform"

    # Reconstruction
    reconstruction_weight: float = 1.0
    use_temporal_consistency: bool = True
    temporal_weight: float = 0.1

    # Sensor correlation
    use_covariance_loss: bool = True
    covariance_weight: float = 0.1

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    early_stopping_patience: int = 15

    # Detection
    threshold_percentile: float = 95.0

    # Ethical constraints
    benevolence_threshold: float = 0.99


class Encoder(nn.Module):
    """Encoder network with sensor correlation modeling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Latent projections (mean and log-variance for VAE-style)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Sensor correlation attention
        # Find largest divisor of input_dim that is <= 4
        num_heads = 1
        for h in [4, 3, 2, 1]:
            if input_dim % h == 0:
                num_heads = h
                break
        self.sensor_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> dict[str, torch.Tensor]:
        """Encode input to latent space."""
        # Sensor correlation attention (if 3D input)
        if x.ndim == 3:
            # x: [batch, seq, features]
            attn_out, attn_weights = self.sensor_attention(x, x, x)
            x = x + 0.1 * attn_out  # Residual
            x = x.mean(dim=1)  # Pool over sequence
        elif x.ndim == 2:
            attn_weights = None
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        result = {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "hidden": h,
        }

        if return_attention and attn_weights is not None:
            result["attention"] = attn_weights

        return result


class Decoder(nn.Module):
    """Decoder network with multi-scale reconstruction."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Build decoder (reverse of encoder)
        layers = []
        prev_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Multi-scale heads
        self.scale_heads = nn.ModuleList([nn.Linear(prev_dim, output_dim) for _ in range(3)])

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode latent representation."""
        h = self.decoder(z)
        reconstruction = self.output_layer(h)

        # Multi-scale outputs
        scales = [head(h) for head in self.scale_heads]

        return {
            "reconstruction": reconstruction,
            "scales": scales,
            "hidden": h,
        }


class Discriminator(nn.Module):
    """Discriminator for adversarial regularization."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Discriminate real (prior) from fake (encoded) samples."""
        return self.net(z)


class AdversarialAutoencoder(nn.Module):
    """
    Adversarial Autoencoder for Industrial Control Systems.

    Combines:
    1. Encoder with sensor correlation attention
    2. Decoder with multi-scale reconstruction
    3. Discriminator for adversarial regularization
    """

    def __init__(self, config: AdversarialAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
        )

        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.input_dim,
        )

        self.discriminator = Discriminator(
            latent_dim=config.latent_dim,
            hidden_dims=config.discriminator_dims,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through autoencoder."""
        # Encode
        enc_result = self.encoder(x)
        z = enc_result["z"]
        mu = enc_result["mu"]
        logvar = enc_result["logvar"]

        # Decode
        dec_result = self.decoder(z)
        reconstruction = dec_result["reconstruction"]

        # Compute reconstruction error
        if x.ndim == 3:
            x_flat = x.mean(dim=1)
        else:
            x_flat = x
        recon_error = ((x_flat - reconstruction) ** 2).mean(dim=-1)

        return {
            "reconstruction": reconstruction,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "recon_error": recon_error,
            "scales": dec_result["scales"],
        }

    def sample_prior(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from prior distribution."""
        if self.config.prior_type == "gaussian":
            return torch.randn(n_samples, self.config.latent_dim, device=device)
        elif self.config.prior_type == "mixture":
            # Gaussian mixture with 5 components
            n_components = 5
            component = torch.randint(0, n_components, (n_samples,), device=device)
            means = torch.linspace(-2, 2, n_components, device=device)
            z = torch.randn(n_samples, self.config.latent_dim, device=device)
            z += means[component].unsqueeze(1)
            return z
        else:  # uniform
            return torch.rand(n_samples, self.config.latent_dim, device=device) * 4 - 2

    def compute_losses(
        self, x: torch.Tensor, train_discriminator: bool = True
    ) -> dict[str, torch.Tensor]:
        """Compute all losses for training."""
        result = self(x)
        batch_size = result["z"].shape[0]
        device = x.device

        losses = {}

        # 1. Reconstruction loss
        if x.ndim == 3:
            x_target = x.mean(dim=1)
        else:
            x_target = x

        recon_loss = F.mse_loss(result["reconstruction"], x_target)
        losses["reconstruction"] = recon_loss

        # 2. Multi-scale reconstruction
        for i, scale_recon in enumerate(result["scales"]):
            scale_loss = F.mse_loss(scale_recon, x_target)
            losses[f"scale_{i}"] = scale_loss

        # 3. KL divergence (VAE regularization)
        kl_loss = -0.5 * torch.mean(
            1 + result["logvar"] - result["mu"].pow(2) - result["logvar"].exp()
        )
        losses["kl"] = kl_loss

        # 4. Adversarial loss
        z_fake = result["z"]
        z_real = self.sample_prior(batch_size, device)

        if train_discriminator:
            # Discriminator loss
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z_fake.detach())

            d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
            d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            losses["discriminator"] = (d_loss_real + d_loss_fake) / 2

        # Generator adversarial loss (fool discriminator)
        d_fake_for_g = self.discriminator(z_fake)
        g_adv_loss = F.binary_cross_entropy_with_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))
        losses["adversarial"] = g_adv_loss

        # 5. Covariance loss (preserve sensor correlations)
        if self.config.use_covariance_loss and x_target.shape[0] > 1:
            # Compute covariance matrices
            x_cov = torch.cov(x_target.T)
            recon_cov = torch.cov(result["reconstruction"].T)
            cov_loss = F.mse_loss(recon_cov, x_cov)
            losses["covariance"] = cov_loss

        # 6. Temporal consistency (if sequential input)
        if self.config.use_temporal_consistency and x.ndim == 3:
            # Latent space should be smooth over time
            z_expanded = result["z"].unsqueeze(1).expand(-1, x.shape[1], -1)
            temporal_loss = ((z_expanded[:, 1:] - z_expanded[:, :-1]) ** 2).mean()
            losses["temporal"] = temporal_loss

        # Total loss
        total = (
            self.config.reconstruction_weight * recon_loss
            + 0.01 * kl_loss
            + self.config.adversarial_weight * g_adv_loss
        )

        if "covariance" in losses:
            total += self.config.covariance_weight * losses["covariance"]
        if "temporal" in losses:
            total += self.config.temporal_weight * losses["temporal"]

        losses["total"] = total

        return losses


class AdversarialAutoencoderDetector:
    """
    Adversarial Autoencoder Detector for Industrial Control Systems.

    Provides sklearn-compatible interface with fit/predict methods.

    Example:
        >>> detector = AdversarialAutoencoderDetector(input_dim=51)
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 16,
        adversarial_weight: float = 0.5,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = AdversarialAEConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims or [128, 64, 32],
            latent_dim=latent_dim,
            adversarial_weight=adversarial_weight,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: AdversarialAutoencoder | None = None
        self.threshold: float = 0.0
        self._fitted = False

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
        validation_split: float = 0.1,
    ) -> AdversarialAutoencoderDetector:
        """
        Fit the detector on training data.

        Args:
            X: Training data [n_samples, n_features] or [n_samples, seq, n_features]
            y: Optional labels (ignored for unsupervised)
            validation_split: Fraction for validation

        Returns:
            self
        """
        # Update input_dim from data
        if X.ndim == 3:
            self.config.input_dim = X.shape[-1]
        else:
            self.config.input_dim = X.shape[-1]

        # Initialize model
        self.model = AdversarialAutoencoder(self.config).to(self.device)

        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device) if n_val > 0 else None

        # Optimizers (separate for generator and discriminator)
        params_g = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        params_d = list(self.model.discriminator.parameters())

        optimizer_g = torch.optim.AdamW(params_g, lr=self.config.learning_rate)
        optimizer_d = torch.optim.AdamW(params_d, lr=self.config.learning_rate)

        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_g, T_max=self.config.epochs
        )

        best_val_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            perm = torch.randperm(len(X_train))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), self.config.batch_size):
                batch_idx = perm[i : i + self.config.batch_size]
                batch = X_train[batch_idx]

                # Train discriminator every 2nd step
                train_disc = n_batches % 2 == 0

                if train_disc:
                    optimizer_d.zero_grad()
                    losses = self.model.compute_losses(batch, train_discriminator=True)
                    if "discriminator" in losses:
                        losses["discriminator"].backward(retain_graph=True)
                        optimizer_d.step()

                # Train generator
                optimizer_g.zero_grad()
                losses = self.model.compute_losses(batch, train_discriminator=False)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(params_g, 1.0)
                optimizer_g.step()

                total_loss += losses["total"].item()
                n_batches += 1

            scheduler_g.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = self.model.compute_losses(X_val, train_discriminator=False)
                    val_loss = val_losses["reconstruction"].item()
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
                train_scores.append(result["recon_error"].cpu().numpy())

            train_scores = np.concatenate(train_scores)
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

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i : i + self.config.batch_size]
                result = self.model(batch)
                scores.append(result["recon_error"].cpu().numpy())

        return np.concatenate(scores)

    def detect(
        self,
        X: NDArray[np.float64],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Perform anomaly detection."""
        scores = self.predict(X)
        thresh = threshold if threshold is not None else self.threshold
        predictions = (scores > thresh).astype(int)

        return {
            "anomaly_score": scores,
            "predictions": predictions,
            "threshold": thresh,
            "is_anomaly": predictions.astype(bool),
            "detector_type": "AdversarialAutoencoder",
            "confidence": np.clip(scores / (thresh + 1e-8), 0, 1),
        }

    def extract_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract latent representations for fusion."""
        if not self._fitted or self.model is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        features = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.config.batch_size):
                batch = X_tensor[i : i + self.config.batch_size]
                result = self.model.encoder(batch)
                features.append(result["z"].cpu().numpy())

        return np.concatenate(features, axis=0)
