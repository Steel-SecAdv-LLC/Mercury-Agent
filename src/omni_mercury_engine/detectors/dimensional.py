"""
Mercury Agent ♱
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
Dimensional analyzer using PCA, t-SNE, and neural projection
Enhanced with DB term (dimensional code-breaking via Fourier analysis)
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.fft import fft
from sklearn.decomposition import PCA
from torch import nn

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException


if TYPE_CHECKING:
    from sklearn.manifold import TSNE


class NeuralProjection(nn.Module):
    """Neural network autoencoder for dimensionality reduction"""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        hidden_dim = max(input_dim // 2, latent_dim * 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class DimensionalAnalyzer(BaseDetector):
    """
    Multi-dimensional analysis and projection:
    - PCA for linear projection
    - t-SNE for non-linear visualization
    - Neural autoencoder for learned projection
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.n_components = self.config.get("n_components", 10)
        self.reconstruction_threshold = self.config.get("reconstruction_threshold", 2.0)
        self.use_db_term = self.config.get("use_db_term", True)

        self.pca: PCA | None = None
        self.tsne: TSNE | None = None
        self.autoencoder: NeuralProjection | None = None

        self.input_dim: int | None = None
        self.baseline_spectral_signature: np.ndarray[Any, Any] | None = None

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> DimensionalAnalyzer:
        """Fit dimensional analyzers to data"""
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        self.input_dim = data_np.shape[1]
        n_comp = min(self.n_components, data_np.shape[1])

        self.pca = PCA(n_components=n_comp)
        self.pca.fit(data_np)

        self.autoencoder = NeuralProjection(
            input_dim=self.input_dim,
            latent_dim=n_comp,
        )

        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)

        for _ in range(100):
            _, reconstructed = self.autoencoder(data_tensor)
            loss = nn.functional.mse_loss(reconstructed, data_tensor)

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        if self.use_db_term:
            self.baseline_spectral_signature = self._compute_spectral_signature(data_np)

        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect dimensional anomalies"""
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            data_tensor = data
        else:
            data_np = data
            data_tensor = torch.tensor(data, dtype=torch.float32)

        assert self.pca is not None, "PCA must be fitted before detection"
        assert self.autoencoder is not None, "Autoencoder must be fitted before detection"

        pca_components = self.pca.transform(data_np)
        pca_reconstructed = self.pca.inverse_transform(pca_components)
        pca_errors = np.linalg.norm(data_np - pca_reconstructed, axis=1)

        with torch.no_grad():
            _, ae_reconstructed = self.autoencoder(data_tensor)
            ae_errors = torch.norm(data_tensor - ae_reconstructed, dim=1).cpu().numpy()

        combined_scores = (pca_errors + ae_errors) / 2.0

        db_scores = None
        if self.use_db_term and self.baseline_spectral_signature is not None:
            db_scores = self._dimensional_code_breaking(data_np)
            combined_scores = combined_scores * 0.7 + db_scores * 0.3

        # Fix for P0: Safe normalization handling NaN/Inf and constant arrays
        # Replace NaN/Inf values before normalization
        if np.any(~np.isfinite(combined_scores)):
            combined_scores = np.nan_to_num(combined_scores, nan=0.5, posinf=1.0, neginf=0.0)

        score_max = combined_scores.max()
        if score_max < 1e-10:
            # All scores are near-zero: return neutral 0.5
            normalized_scores = np.full_like(combined_scores, 0.5)
        else:
            normalized_scores = combined_scores / score_max
            normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

        is_anomaly = normalized_scores > self.threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": normalized_scores,
            "pca_errors": pca_errors,
            "autoencoder_errors": ae_errors,
            "db_scores": db_scores,
            "detector_type": "dimensional",
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract dimensional features for ML fusion"""
        if not self._is_fitted:
            if isinstance(data, torch.Tensor):
                self.fit(data.cpu().numpy())
            else:
                self.fit(data)

        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            data_tensor = data
        else:
            data_np = data
            data_tensor = torch.tensor(data, dtype=torch.float32)

        assert self.pca is not None, "PCA must be fitted before feature extraction"
        assert self.autoencoder is not None, "Autoencoder must be fitted before feature extraction"

        pca_components = self.pca.transform(data_np)

        with torch.no_grad():
            ae_components, _ = self.autoencoder(data_tensor)
            ae_components_np = ae_components.cpu().numpy()

        features = np.column_stack([pca_components, ae_components_np])

        if features.shape[1] < 50:
            padding = np.zeros((features.shape[0], 50 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

    def _compute_spectral_signature(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute baseline spectral signature using Fourier transform
        DB term: Dimensional Code-Breaking via frequency analysis
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        signatures = []
        for i in range(data.shape[1]):
            column = data[:, i]
            fft_result = fft(column)
            power_spectrum = np.abs(fft_result) ** 2
            signatures.append(power_spectrum[: len(power_spectrum) // 2])

        mean_signature: np.ndarray[Any, Any] = np.asarray(np.mean(signatures, axis=0))
        return mean_signature

    def _dimensional_code_breaking(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        DB Term: Dimensional Code-Breaking Detection
        Detects anomalies via spectral divergence in Fourier space
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        scores = np.zeros(data.shape[0])

        assert (
            self.baseline_spectral_signature is not None
        ), "Baseline spectral signature must be computed"

        for idx in range(data.shape[0]):
            sample = data[idx : idx + 1, :]
            sample_signature = self._compute_spectral_signature(sample)

            min_len = min(len(self.baseline_spectral_signature), len(sample_signature))
            baseline_truncated = self.baseline_spectral_signature[:min_len]
            sample_truncated = sample_signature[:min_len]

            spectral_divergence = np.linalg.norm(baseline_truncated - sample_truncated) / (
                np.linalg.norm(baseline_truncated) + 1e-10
            )

            phase_coherence = self._compute_phase_coherence(data[idx, :])

            harmonic_distortion = self._compute_harmonic_distortion(data[idx, :])

            db_score = (
                spectral_divergence * 0.5
                + (1.0 - phase_coherence) * 0.3
                + harmonic_distortion * 0.2
            )

            scores[idx] = min(db_score, 1.0)

        return scores

    def _compute_phase_coherence(self, signal: np.ndarray[Any, Any]) -> float:
        """Compute phase coherence for DB term"""
        if len(signal) < 4:
            return 1.0

        fft_result = fft(signal)
        phases = np.angle(fft_result)

        phase_diffs = np.diff(phases)
        phase_diffs = np.abs(phase_diffs)

        coherence = 1.0 - np.mean(phase_diffs) / np.pi

        return float(max(0.0, min(1.0, coherence)))

    def _compute_harmonic_distortion(self, signal: np.ndarray[Any, Any]) -> float:
        """Compute total harmonic distortion for DB term"""
        if len(signal) < 8:
            return 0.0

        fft_result = fft(signal)
        power_spectrum = np.abs(fft_result) ** 2

        fundamental_idx: int = int(np.argmax(power_spectrum[: len(power_spectrum) // 2]))
        if fundamental_idx == 0:
            fundamental_idx = 1

        fundamental_power = power_spectrum[fundamental_idx]

        harmonic_powers: list[float] = []
        max_harmonic = int(min(8, len(power_spectrum) // (2 * fundamental_idx)))
        for n in range(2, max_harmonic):
            harmonic_idx = n * fundamental_idx
            if harmonic_idx < len(power_spectrum):
                harmonic_powers.append(float(power_spectrum[harmonic_idx]))

        if not harmonic_powers or fundamental_power == 0:
            return 0.0

        total_harmonic_power = sum(harmonic_powers)
        thd = float(np.sqrt(total_harmonic_power / (fundamental_power + 1e-10)))

        return float(min(thd, 1.0))
