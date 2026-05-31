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

"""Differentiable domain encoders (WS-B / Target 2).

The production :class:`~omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector`
computes three *static* (non-learnable) feature families:

* **spectral / resonance** -- per-feature FFT harmonic ratios, scored at
  inference as ``mean(1 - h_train * exp(-dev * noise_ratio))``;
* **kinematic** -- finite-difference velocity/acceleration/jerk z-scores;
* **Fisher / info-geometry** -- Mahalanobis distance under a fitted precision
  matrix, plus entropy-like normalisation.

This module reimplements each as a small, principled, *jointly-trainable*
``nn.Module`` so the same physics can be co-trained with the fusion head
instead of frozen. They are **opt-in**: nothing here runs unless a caller
explicitly enables it (see ``engine.fit_fusion(domain_encoder=True)``), so the
default neural path stays byte-for-byte identical.

Design principles (so this is a faithful differentiable analog, not an
arbitrary MLP):

* :class:`SpectralEncoder` uses :func:`torch.fft.rfft` -- a differentiable
  FFT -- mirroring the static resonance extractor's spectral basis, then learns
  a filter bank over the magnitude spectrum.
* :class:`KinematicEncoder` uses ``Conv1d`` kernels *initialised* to the exact
  finite-difference operators (velocity ``[-1, 1]``, acceleration
  ``[1, -2, 1]``, jerk ``[-1, 3, -3, 1]``) but left learnable, so it starts at
  the static extractor and adapts.
* :class:`FisherEntropyEncoder` learns a whitening map (Mahalanobis analog) and
  emits the squared whitened norm plus a differentiable softmax-entropy.

Every encoder maps ``(batch, n_features) -> (batch, output_dim)`` and is
deterministic given its weights and a fixed seed.
"""

from typing import Any

import torch
from torch import nn

# Finite-difference stencils used to *initialise* the (learnable) kinematic
# kernels. These are the exact operators the static extractor applies via
# ``np.diff`` of orders 1/2/3.
_DIFF_STENCILS: dict[int, list[float]] = {
    1: [-1.0, 1.0],  # velocity
    2: [1.0, -2.0, 1.0],  # acceleration
    3: [-1.0, 3.0, -3.0, 1.0],  # jerk
}


class SpectralEncoder(nn.Module):
    """Differentiable spectral (resonance) encoder built on ``torch.fft``.

    Treats each sample's feature vector as a 1-D signal, takes its real FFT
    magnitude spectrum (differentiable), and learns a filter bank + MLP over
    it. Generalises the static per-feature harmonic-ratio extractor into a
    learnable spectral representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        magnitude_transform: str = "log1p",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        if magnitude_transform not in ("log1p", "sqrt", "none"):
            raise ValueError(f"unknown magnitude_transform: {magnitude_transform!r}")
        self.magnitude_transform = magnitude_transform
        # rfft of a length-n real signal yields floor(n/2)+1 complex bins.
        n_bins = self.input_dim // 2 + 1
        self.spectral_filter = nn.Linear(n_bins, hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode feature vectors into spectral embeddings."""
        # Magnitude spectrum along the feature axis; abs() of a complex tensor
        # is differentiable w.r.t. the real input.
        spectrum = torch.fft.rfft(x, dim=-1)
        magnitude = torch.abs(spectrum)
        # Dynamic-range compression (default log1p; sweepable for WS-B design
        # search). log1p matches the static extractor's ratio-of-energies
        # intuition without exploding on large bins.
        if self.magnitude_transform == "log1p":
            magnitude = torch.log1p(magnitude)
        elif self.magnitude_transform == "sqrt":
            magnitude = torch.sqrt(magnitude + 1e-9)
        return self.mlp(self.spectral_filter(magnitude))


class KinematicEncoder(nn.Module):
    """Differentiable kinematic encoder via learnable finite-difference convs.

    Three ``Conv1d`` channels are initialised to the velocity/acceleration/jerk
    stencils (so the module *starts* as the static extractor) and then learn.
    Each difference signal is pooled to (mean, std, max|.|) summary stats, which
    feed a small MLP -- mirroring the static z-score-of-jerk scoring.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        kernel_widths: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.kernel_widths = tuple(kernel_widths)
        self.convs = nn.ModuleList()
        for width in self.kernel_widths:
            conv = nn.Conv1d(1, 1, kernel_size=width, bias=False)
            # Initialise to the exact finite-difference stencil when one exists
            # for this width (velocity/accel/jerk); otherwise leave the default
            # learnable init. Either way the kernel is trained.
            stencil = _DIFF_STENCILS.get(width - 1)
            if stencil is not None and len(stencil) == width:
                with torch.no_grad():
                    conv.weight.copy_(torch.tensor(stencil).view(1, 1, -1))
            self.convs.append(conv)
        # n difference orders x 3 summary stats (mean, std, max-abs).
        self.mlp = nn.Sequential(
            nn.Linear(len(self.kernel_widths) * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode feature vectors into kinematic embeddings."""
        signal = x.unsqueeze(1)  # (batch, 1, n_features)
        feats: list[torch.Tensor] = []
        for conv in self.convs:
            diff = conv(signal).squeeze(1)  # (batch, n_features - k + 1)
            feats.append(diff.mean(dim=-1, keepdim=True))
            feats.append(diff.std(dim=-1, keepdim=True))
            feats.append(diff.abs().amax(dim=-1, keepdim=True))
        summary = torch.cat(feats, dim=-1)  # (batch, 9)
        return self.mlp(summary)


class FisherEntropyEncoder(nn.Module):
    """Differentiable Fisher/info-geometry encoder.

    Learns a whitening map ``W`` (a Mahalanobis analog of the static precision
    matrix) and emits, per sample: the whitened vector, its squared norm (the
    Mahalanobis-distance analog of the static info-geometry score), and the
    Shannon entropy of ``softmax`` over the whitened coordinates (a
    differentiable information-geometry summary). An MLP fuses these.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.whiten = nn.Linear(input_dim, input_dim, bias=True)
        # whitened coords (input_dim) + squared norm (1) + entropy (1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode feature vectors into Fisher-entropy embeddings."""
        z = self.whiten(x)  # (batch, input_dim)
        mahal_sq = (z * z).sum(dim=-1, keepdim=True)  # Mahalanobis analog
        probs = torch.softmax(z, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)
        summary = torch.cat([z, mahal_sq, entropy], dim=-1)
        return self.mlp(summary)


class DomainEncoderStack(nn.Module):
    """Joint differentiable domain encoder = spectral + kinematic + Fisher.

    Concatenates the three encoders' embeddings and projects to ``output_dim``.
    This is the single ``nn.Module`` wired (opt-in) into the fusion path as a
    ``differentiable_domain`` feature, jointly trained with the fusion head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        per_encoder_dim: int = 64,
        output_dim: int = 128,
        domains: tuple[str, ...] = ("spectral", "kinematic", "fisher"),
        encoder_kwargs: dict[str, dict[str, Any]] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.per_encoder_dim = int(per_encoder_dim)
        self.output_dim = int(output_dim)
        self.domains = tuple(domains)
        self.normalize = bool(normalize)
        if not self.domains:
            raise ValueError("DomainEncoderStack needs at least one domain")
        encoder_kwargs = encoder_kwargs or {}
        self.encoder_kwargs = {name: dict(values) for name, values in encoder_kwargs.items()}
        self.encoders = nn.ModuleDict()
        builders: dict[str, Any] = {
            "spectral": SpectralEncoder,
            "kinematic": KinematicEncoder,
            "fisher": FisherEntropyEncoder,
        }
        for name in self.domains:
            if name not in builders:
                raise ValueError(f"unknown domain encoder: {name!r}")
            self.encoders[name] = builders[name](
                input_dim, hidden_dim, per_encoder_dim, **encoder_kwargs.get(name, {})
            )
        # Optional LayerNorm before projection (WS-B normalization design axis;
        # default off keeps the stack byte-identical to the original).
        self.norm: nn.Module = (
            nn.LayerNorm(per_encoder_dim * len(self.domains)) if normalize else nn.Identity()
        )
        self.project = nn.Sequential(
            nn.Linear(per_encoder_dim * len(self.domains), output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode feature vectors with all active domain encoders."""
        parts = [self.encoders[name](x) for name in self.domains]
        return self.project(self.norm(torch.cat(parts, dim=-1)))


__all__ = [
    "DomainEncoderStack",
    "FisherEntropyEncoder",
    "KinematicEncoder",
    "SpectralEncoder",
]
