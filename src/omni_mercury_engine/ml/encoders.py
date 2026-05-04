"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Feature encoders for different detector types

Each encoder transforms domain-specific features into fixed-size embeddings
for neural fusion.
"""

import torch
from torch import nn


class StatisticalEncoder(nn.Module):
    """Encodes statistical features (z-scores, IQR, distributions)"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x: [batch_size, input_dim] - Statistical features

        Returns:
            embedding: [batch_size, output_dim]
        """
        return self.encoder(x)


class TemporalEncoder(nn.Module):
    """LSTM-based encoder for time series features (handles both sequential and pre-extracted)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.embedding_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x: [batch_size, seq_len, input_dim] - Time series data OR
               [batch_size, input_dim] - Pre-extracted embeddings

        Returns:
            embedding: [batch_size, output_dim]
        """
        if x.dim() == 3:
            _lstm_out, (hidden, _) = self.lstm(x)
            last_hidden = hidden[-1]
            embedding = self.projection(last_hidden)
        elif x.dim() == 2:
            embedding = self.embedding_projection(x)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        return embedding


class BiometricEncoder(nn.Module):
    """Encoder for biometric features (handles both images and pre-extracted embeddings)"""

    def __init__(
        self,
        input_channels: int = 3,
        image_size: int = 224,
        output_dim: int = 128,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

        self.embedding_projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x: [batch_size, channels, height, width] - Face images OR
               [batch_size, embedding_dim] - Pre-extracted embeddings

        Returns:
            embedding: [batch_size, output_dim]
        """
        if x.dim() == 4:
            conv_out = self.conv_layers(x)
            embedding = self.image_fc(conv_out)
        elif x.dim() == 2:
            embedding = self.embedding_projection(x)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.dim()}D")

        return embedding


class QuantumEncoder(nn.Module):
    """Encodes quantum state vectors and observables."""

    def __init__(self, state_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.complex_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x: [batch_size, state_dim, 2] - Complex quantum states (real, imag) OR
               [batch_size, state_dim] - Pre-extracted quantum embeddings

        Returns:
            embedding: [batch_size, output_dim]
        """
        if x.dim() == 3 and x.shape[2] == 2:
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            return self.complex_encoder(x_flat)
        elif x.dim() == 2:
            return self.embedding_projection(x)
        else:
            raise ValueError(f"Expected 2D or 3D input with last dim=2, got shape {x.shape}")


class AstrophysicalEncoder(nn.Module):
    """Encodes astrophysical features (gravitational fields, event horizons)"""

    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x: [batch_size, input_dim] - Astrophysical features

        Returns:
            embedding: [batch_size, output_dim]
        """
        return self.encoder(x)


class AffectiveEncoder(nn.Module):
    """BiLSTM encoder for emotional sequences (handles both sequential and pre-extracted)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        self.embedding_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x: [batch_size, seq_len, input_dim] - Emotional feature sequences OR
               [batch_size, input_dim] - Pre-extracted embeddings

        Returns:
            embedding: [batch_size, output_dim]
        """
        if x.dim() == 3:
            _lstm_out, (hidden, _) = self.bilstm(x)
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            combined = torch.cat([forward_hidden, backward_hidden], dim=1)
            embedding = self.projection(combined)
        elif x.dim() == 2:
            embedding = self.embedding_projection(x)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        return embedding
