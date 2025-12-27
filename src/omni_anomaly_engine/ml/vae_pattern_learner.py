"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""
from __future__ import annotations

"""
Variational Autoencoder (VAE) for Unsupervised Pattern Learning

Learns latent representations of normal patterns for anomaly detection
through reconstruction error and KL divergence.

⚠️ SIMULATION-BASED: Trained on simulated data. Real-world validation required.

"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    """Variational Autoencoder for pattern learning."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: list[int] | None = None) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                ]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                ]
            )
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Input data
            recon: Reconstructed data
            mu: Latent mean
            logvar: Latent log variance
            beta: Weight for KL divergence term (beta-VAE)

        Returns:
            Dict with total loss and components
        """
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score based on reconstruction error."""
        with torch.no_grad():
            recon, _mu, _logvar = self.forward(x)
            score = F.mse_loss(recon, x, reduction="none").mean(dim=1)
            return score


class VAEPatternLearner:
    """Wrapper for VAE-based unsupervised pattern learning."""

    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        self.vae = VAE(input_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.threshold = None

    def fit(
        self, X_train: torch.Tensor, epochs: int = 50, batch_size: int = 32, beta: float = 1.0
    ) -> VAEPatternLearner:
        """Fit VAE on normal training data."""
        self.vae.train()

        for _epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch = X_train[i : i + batch_size]

                self.optimizer.zero_grad()
                recon, mu, logvar = self.vae(batch)
                losses = self.vae.compute_loss(batch, recon, mu, logvar, beta)

                losses["total_loss"].backward()
                self.optimizer.step()

        self.vae.eval()
        with torch.no_grad():
            train_scores = self.vae.anomaly_score(X_train)
            self.threshold = train_scores.mean() + 3 * train_scores.std()

        return self

    def predict(self, X: torch.Tensor) -> dict[str, Any]:
        """Predict anomalies using VAE reconstruction error."""
        self.vae.eval()

        with torch.no_grad():
            scores = self.vae.anomaly_score(X)
            is_anomaly = scores > self.threshold if self.threshold is not None else scores > 0.5

            return {
                "anomaly_scores": scores.cpu().numpy(),
                "is_anomaly": is_anomaly.cpu().numpy(),
                "threshold": self.threshold.item() if self.threshold is not None else None,
            }
