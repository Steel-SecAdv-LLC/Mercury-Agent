# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dimensional analyzer using PCA and neural projection.

Enhanced with DB term (dimensional code-breaking via Fourier analysis)
for detecting subtle anomalies in high-dimensional data representations.

This module provides multi-modal dimensionality analysis for anomaly detection,
combining linear (PCA) and non-linear (autoencoder) projection methods with
spectral analysis in Fourier space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        from torch import nn

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

from scipy.fft import fft

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _NativePCA:
    """Minimal PCA via truncated SVD (no sklearn dependency).

    Supports fit / transform / inverse_transform with the same API surface that DimensionalAnalyzer
    requires.
    """

    def __init__(self, n_components: int) -> None:
        """Initialize the instance."""
        self.n_components = n_components
        self.components_: np.ndarray[Any, Any] | None = None
        self.mean_: np.ndarray[Any, Any] | None = None

    def fit(self, X: np.ndarray[Any, Any]) -> _NativePCA:
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        # Economy SVD - only compute first min(n, d) singular vectors
        _U, _s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self.mean_ is not None and self.components_ is not None
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X_reduced: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self.mean_ is not None and self.components_ is not None
        return X_reduced @ self.components_ + self.mean_


@dataclass(frozen=True)
class DimensionalWeights:
    """Configurable weights for dimensional score combination.

    Attributes:
        pca_weight: Weight for PCA reconstruction error (default: 0.5).
        autoencoder_weight: Weight for autoencoder reconstruction error (default: 0.5).
        db_blend: Blend factor for DB term when enabled (default: 0.3).
            When DB term is enabled, combined = base * (1 - db_blend) + db * db_blend.

    Example:
        >>> weights = DimensionalWeights(pca_weight=0.6, autoencoder_weight=0.4)
        >>> analyzer = DimensionalAnalyzer({"weights": weights})
    """

    pca_weight: float = 0.5
    autoencoder_weight: float = 0.5
    db_blend: float = 0.3

    def __post_init__(self) -> None:
        """Validate weights configuration."""
        base_sum = self.pca_weight + self.autoencoder_weight
        if abs(base_sum - 1.0) > 1e-6:
            raise ValueError(f"pca_weight + autoencoder_weight must equal 1.0, got {base_sum:.4f}")
        if not 0.0 <= self.db_blend <= 1.0:
            raise ValueError(f"db_blend must be in [0, 1], got {self.db_blend}")


if TYPE_CHECKING or TORCH_AVAILABLE:

    class NeuralProjection(nn.Module):
        """Neural network autoencoder for dimensionality reduction.

        A symmetric encoder-decoder architecture that learns compressed
        representations of input data. Reconstruction error serves as
        an anomaly indicator.

        Attributes:
            encoder: Sequential network mapping input to latent space.
            decoder: Sequential network reconstructing input from latent.

        Args:
            input_dim: Dimensionality of input features.
            latent_dim: Dimensionality of compressed latent space.
        """

        def __init__(self, input_dim: int, latent_dim: int) -> None:
            """Initialize autoencoder architecture.

            Args:
                input_dim: Input feature dimension.
                latent_dim: Latent space dimension (compression target).
            """
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
            """Forward pass through encoder and decoder.

            Args:
                x: Input tensor of shape (batch_size, input_dim).

            Returns:
                Tuple of (latent_representation, reconstructed_input).
            """
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return latent, reconstructed

else:

    class NeuralProjection:
        """Stub: NeuralProjection requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the instance."""
            raise ImportError("NeuralProjection requires PyTorch. Install with: pip install torch")


class DimensionalAnalyzer(BaseDetector):
    """Multi-dimensional analysis and projection for anomaly detection.

    Combines multiple projection methods to detect anomalies based on
    reconstruction error and spectral characteristics:

    - **PCA**: Linear projection capturing maximum variance directions.
      High reconstruction error indicates deviation from principal subspace.
    - **Neural Autoencoder**: Non-linear learned compression detecting
      samples that don't fit learned manifold structure.
    - **DB Term** (optional): Dimensional Code-Breaking via Fourier analysis
      detecting spectral anomalies invisible in spatial domain.

    Attributes:
        n_components: Number of PCA/latent components (default: 10).
        reconstruction_threshold: Threshold multiplier for errors (default: 2.0).
        use_db_term: Whether to enable spectral analysis (default: True).
        weights: DimensionalWeights for configurable score combination.
        min_samples_for_pca: Minimum samples required for PCA fitting.

    Example:
        >>> analyzer = DimensionalAnalyzer({
        ...     "n_components": 5,
        ...     "use_db_term": True,
        ...     "weights": DimensionalWeights(pca_weight=0.6, autoencoder_weight=0.4),
        ... })
        >>> analyzer.fit(training_data)
        >>> result = analyzer.detect(test_data)
        >>> anomalies = result["is_anomaly"]
    """

    # Minimum samples required for stable PCA estimation
    MIN_SAMPLES_FOR_PCA = 2

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize DimensionalAnalyzer with configuration.

        Args:
            config: Configuration dictionary with optional keys:
                - n_components: Number of PCA/latent dimensions (default: 10)
                - reconstruction_threshold: Error threshold multiplier (default: 2.0)
                - use_db_term: Enable DB term spectral analysis (default: True)
                - weights: DimensionalWeights instance or dict
                - autoencoder_epochs: Training epochs for autoencoder (default: 100)
                - autoencoder_lr: Learning rate for autoencoder (default: 0.001)

        Raises:
            ValueError: If weights configuration is invalid.
        """
        super().__init__(config)
        self.n_components: int = self.config.get("n_components", 10)
        self.reconstruction_threshold: float = self.config.get("reconstruction_threshold", 2.0)
        self.use_db_term: bool = self.config.get("use_db_term", True)
        # Real semantics for the DB term's spectral-divergence component
        # (operator-approved 2026-06-11). Default True: the pre-registered
        # ablation gate (benchmarks/db_spectral_ablation.py,
        # artifacts/db_spectral_ablation.json) cleared decisively — mean
        # paired detector dAUC +0.071 over 5 ADBench datasets x 3 seeds
        # (bar +0.002), seed agreement 0.93, and the term alone moved from
        # chance (0.460) to 0.811 AUC. Set False for the legacy
        # identically-zero term (pre-2026-06-11 shipped scores).
        self.db_spectral_divergence: bool = bool(
            self.config.get("db_spectral_divergence", True)
        )
        self.autoencoder_epochs: int = self.config.get("autoencoder_epochs", 100)
        self.autoencoder_lr: float = self.config.get("autoencoder_lr", 0.001)

        # Configurable weights (addresses hardcoded magic numbers issue)
        weights_config = self.config.get("weights", None)
        if weights_config is None:
            self.weights = DimensionalWeights()
        elif isinstance(weights_config, DimensionalWeights):
            self.weights = weights_config
        elif isinstance(weights_config, dict):
            self.weights = DimensionalWeights(**weights_config)
        else:
            raise ValueError(
                f"weights must be DimensionalWeights or dict, got {type(weights_config)}"
            )

        self.pca: _NativePCA | None = None
        self.autoencoder: NeuralProjection | None = None

        self.input_dim: int | None = None
        self.baseline_spectral_signature: NDArray[np.float64] | None = None
        # Mean per-row power spectrum of the training data (length d // 2):
        # the baseline the opt-in spectral-divergence semantics compare
        # against. Always computed at fit when the DB term is enabled, so
        # the flag can be evaluated without refitting.
        self.baseline_row_spectrum: NDArray[np.float64] | None = None

    def fit(self, data: NDArray[np.float64] | torch.Tensor) -> DimensionalAnalyzer:
        """Fit dimensional analyzers to training data.

        Fits both PCA and neural autoencoder to learn the normal data
        distribution. Optionally computes baseline spectral signature
        for DB term analysis.

        Args:
            data: Training data array of shape (n_samples, n_features) or tensor.
                Should contain representative normal/non-anomalous samples.

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty, has insufficient samples for PCA,
                or contains only NaN/Inf values.

        Note:
            PCA requires n_samples >= n_components. If fewer samples are provided,
            n_components is automatically reduced to n_samples - 1.

        Example:
            >>> analyzer = DimensionalAnalyzer({"n_components": 5})
            >>> analyzer.fit(training_data)  # Must have at least 2 samples
        """
        data_np = data.cpu().numpy() if TORCH_AVAILABLE and isinstance(data, torch.Tensor) else data
        assert isinstance(data_np, np.ndarray)

        # Validate data shape
        if data_np.size == 0:
            raise DetectorException("Cannot fit DimensionalAnalyzer with empty data.")

        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        n_samples, n_features = data_np.shape

        # Validate minimum samples for PCA (fixes PCA minimum samples issue)
        if n_samples < self.MIN_SAMPLES_FOR_PCA:
            raise DetectorException(
                f"DimensionalAnalyzer requires at least {self.MIN_SAMPLES_FOR_PCA} samples "
                f"for PCA fitting, got {n_samples}. Provide more training data."
            )

        # Validate finite values
        finite_mask = np.isfinite(data_np).all(axis=1)
        if not np.any(finite_mask):
            raise DetectorException(
                "Cannot fit DimensionalAnalyzer: all data values are NaN or Inf."
            )
        if not np.all(finite_mask):
            data_np = data_np[finite_mask]
            n_samples = data_np.shape[0]

        self.input_dim = n_features

        # Ensure n_components doesn't exceed data dimensions
        # PCA requires: n_components <= min(n_samples, n_features)
        max_components = min(n_samples - 1, n_features)
        n_comp = min(self.n_components, max_components)

        n_comp = max(n_comp, 1)

        self.pca = _NativePCA(n_components=n_comp)
        self.pca.fit(data_np)

        self.autoencoder = NeuralProjection(
            input_dim=self.input_dim,
            latent_dim=n_comp,
        )

        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.autoencoder_lr)

        for _ in range(self.autoencoder_epochs):
            _, reconstructed = self.autoencoder(data_tensor)
            loss = nn.functional.mse_loss(reconstructed, data_tensor)

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            optimizer.step()

        if self.use_db_term:
            self.baseline_spectral_signature = self._compute_spectral_signature(data_np)
            row_spectra = self._row_power_spectrum(data_np)
            self.baseline_row_spectrum = (
                row_spectra.mean(axis=0) if row_spectra.size else None
            )

        self._is_fitted = True
        return self

    def detect(self, data: NDArray[np.float64] | torch.Tensor) -> dict[str, Any]:
        """Detect dimensional anomalies using reconstruction error analysis.

        Computes anomaly scores based on how well each sample can be reconstructed
        by the learned projection models. High reconstruction error indicates
        the sample deviates from the normal data manifold.

        Auto-Calibration:
            When auto_calibrate=True (via enable_auto_calibration()), the
            threshold is automatically calibrated based on the score
            distribution, solving the F1=0 problem where good ROC-AUC
            is achieved but fixed threshold produces no positive predictions.

        Args:
            data: Input data array of shape (n_samples, n_features) or tensor.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Normalized combined scores in [0, 1] range
                - pca_errors: Raw PCA reconstruction errors
                - autoencoder_errors: Raw autoencoder reconstruction errors
                - db_scores: DB term spectral scores (None if disabled)
                - detector_type: "dimensional"
                - threshold: Effective threshold used (may be calibrated)
                - calibration_diagnostics: CalibrationDiagnostics if auto-calibrated

        Raises:
            DetectorException: If detector has not been fitted.

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> analyzer.fit(train_data).enable_auto_calibration(contamination=0.05)
            >>> result = analyzer.detect(test_data)
            >>> print(f"Found {result['is_anomaly'].sum()} anomalies")
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        data_np: np.ndarray[Any, Any]
        data_tensor: torch.Tensor | None
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            data_tensor = data
        else:
            data_np = np.asarray(data)
            data_tensor = torch.tensor(data_np, dtype=torch.float32) if TORCH_AVAILABLE else None

        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
            if data_tensor is not None:
                data_tensor = data_tensor.reshape(-1, 1)

        if self.pca is None:
            raise DetectorException("PCA must be fitted before detection")
        if self.autoencoder is None:
            raise DetectorException("Autoencoder must be fitted before detection")

        # Compute PCA reconstruction error
        pca_components = self.pca.transform(data_np)
        pca_reconstructed = self.pca.inverse_transform(pca_components)
        pca_errors = np.linalg.norm(data_np - pca_reconstructed, axis=1)

        # Compute autoencoder reconstruction error
        with torch.no_grad():
            assert data_tensor is not None
            _, ae_reconstructed = self.autoencoder(data_tensor)
            ae_errors = torch.norm(data_tensor - ae_reconstructed, dim=1).cpu().numpy()

        # Normalize errors to comparable scales before combining
        pca_errors_norm = self._safe_normalize(pca_errors)
        ae_errors_norm = self._safe_normalize(ae_errors)

        # Combine using configurable weights (fixes hardcoded magic numbers)
        combined_scores = (
            pca_errors_norm * self.weights.pca_weight
            + ae_errors_norm * self.weights.autoencoder_weight
        )

        # Optionally blend in DB term spectral analysis
        db_scores: NDArray[np.float64] | None = None
        if self.use_db_term and self.baseline_spectral_signature is not None:
            db_scores = self._dimensional_code_breaking(data_np)
            blend = self.weights.db_blend
            combined_scores = combined_scores * (1 - blend) + db_scores * blend

        # Ensure scores are finite and in valid range
        if np.any(~np.isfinite(combined_scores)):
            combined_scores = np.nan_to_num(combined_scores, nan=0.5, posinf=1.0, neginf=0.0)
        combined_scores = np.clip(combined_scores, 0.0, 1.0)

        # Auto-calibration: compute optimal threshold from score distribution
        effective_threshold = self.threshold
        calibration_diagnostics = None

        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = combined_scores > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "pca_errors": pca_errors,
            "autoencoder_errors": ae_errors,
            "db_scores": db_scores,
            "detector_type": "dimensional",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def _safe_normalize(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Safely normalize scores to [0, 1] range.

        Handles edge cases including constant arrays and NaN/Inf values.

        Args:
            scores: Raw score array.

        Returns:
            Normalized scores in [0, 1] range.
        """
        if np.any(~np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)

        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min

        if score_range < 1e-10:
            return np.full_like(scores, 0.5)

        return np.clip((scores - score_min) / score_range, 0.0, 1.0)

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract dimensional features for ML fusion."""
        if not self._is_fitted:
            if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                self.fit(data.cpu().numpy())
            else:
                self.fit(data)

        data_np: np.ndarray[Any, Any]
        data_tensor: torch.Tensor | None
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            data_tensor = data
        else:
            data_np = np.asarray(data)
            data_tensor = torch.tensor(data_np, dtype=torch.float32) if TORCH_AVAILABLE else None

        assert self.pca is not None, "PCA must be fitted before feature extraction"
        assert self.autoencoder is not None, "Autoencoder must be fitted before feature extraction"

        pca_components = self.pca.transform(data_np)

        with torch.no_grad():
            assert data_tensor is not None
            ae_components, _ = self.autoencoder(data_tensor)
            ae_components_np = ae_components.cpu().numpy()

        features = np.column_stack([pca_components, ae_components_np])

        if features.shape[1] < 50:
            padding = np.zeros((features.shape[0], 50 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

    def _compute_spectral_signature(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute baseline spectral signature using Fourier transform.

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
        """DB term: dimensional code-breaking — detect anomalies via spectral divergence in Fourier space.

        Vectorized (2026-06-11): one batched FFT over all rows replaces the
        former per-row loop (two FFT calls per sample). Outputs are
        equivalence-pinned to the loop semantics by
        ``tests/test_native_acceleration.py``.

        Note on the spectral-divergence component: the original loop computed
        it per *single row*, where each column's FFT has length 1, so the
        half-spectrum slice ``[:0]`` is empty and the divergence evaluates to
        ``0.0 / (0.0 + 1e-10) == 0.0`` for every sample — identically zero.
        By default that legacy zero is preserved (changing a live detector's
        shipped scores requires a measured gate). With the opt-in
        ``db_spectral_divergence`` config flag the term carries real,
        coherent semantics: each row's feature-axis power spectrum is
        compared against the fit-time mean training-row spectrum with the
        original normalized-distance formula. The flag's effect on real
        labels is measured by ``benchmarks/db_spectral_ablation.py``; the
        default flips only if that pre-registered gate clears.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        assert (
            self.baseline_spectral_signature is not None
        ), "Baseline spectral signature must be computed"

        spectral_divergence: NDArray[np.float64] | float
        if (
            self.db_spectral_divergence
            and self.baseline_row_spectrum is not None
            and self.baseline_row_spectrum.size > 0
        ):
            sample_spectra = self._row_power_spectrum(data)
            min_len = min(self.baseline_row_spectrum.shape[0], sample_spectra.shape[1])
            baseline = self.baseline_row_spectrum[:min_len]
            diff = sample_spectra[:, :min_len] - baseline[None, :]
            spectral_divergence = np.linalg.norm(diff, axis=1) / (
                np.linalg.norm(baseline) + 1e-10
            )
        else:
            spectral_divergence = 0.0  # legacy dead term (see docstring)

        phase_coherence = self._phase_coherence_rows(data)
        harmonic_distortion = self._harmonic_distortion_rows(data)

        db_scores = (
            spectral_divergence * 0.5
            + (1.0 - phase_coherence) * 0.3
            + harmonic_distortion * 0.2
        )

        return np.minimum(db_scores, 1.0)

    @staticmethod
    def _row_power_spectrum(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Per-row half power spectrum along the feature axis, shape (n, d // 2)."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        d = data.shape[1]
        power = np.abs(fft(data, axis=1)) ** 2
        return np.asarray(power[:, : d // 2], dtype=np.float64)

    @staticmethod
    def _phase_coherence_rows(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Row-wise phase coherence (vectorized form of the per-row DB term)."""
        n, d = data.shape
        if d < 4:
            return np.ones(n)

        fft_result = fft(data, axis=1)
        phases = np.angle(fft_result)
        phase_diffs = np.abs(np.diff(phases, axis=1))
        coherence = 1.0 - phase_diffs.mean(axis=1) / np.pi
        return np.clip(coherence, 0.0, 1.0)

    @staticmethod
    def _harmonic_distortion_rows(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Row-wise total harmonic distortion (vectorized DB term).

        Mirrors the scalar logic exactly: per row, the fundamental is the
        argmax of the first half-spectrum (index 0 promoted to 1), harmonics
        ``n = 2 .. min(8, d // (2 * fundamental)) - 1`` are accumulated in
        ascending order, and rows with no harmonics or zero fundamental
        power score 0.0.
        """
        n, d = data.shape
        if d < 8:
            return np.zeros(n)

        power = np.abs(fft(data, axis=1)) ** 2
        rows = np.arange(n)

        fundamental_idx = np.argmax(power[:, : d // 2], axis=1)
        fundamental_idx = np.where(fundamental_idx == 0, 1, fundamental_idx)
        fundamental_power = power[rows, fundamental_idx]

        max_harmonic = np.minimum(8, d // (2 * fundamental_idx))
        total_harmonic_power = np.zeros(n)
        has_harmonics = np.zeros(n, dtype=bool)
        for harmonic_n in range(2, 8):
            harmonic_idx = harmonic_n * fundamental_idx
            mask = (harmonic_n < max_harmonic) & (harmonic_idx < d)
            gathered = power[rows, np.minimum(harmonic_idx, d - 1)]
            total_harmonic_power += np.where(mask, gathered, 0.0)
            has_harmonics |= mask

        thd = np.sqrt(total_harmonic_power / (fundamental_power + 1e-10))
        thd = np.where(has_harmonics & (fundamental_power != 0), thd, 0.0)
        return np.minimum(thd, 1.0)

    def _compute_phase_coherence(self, signal: np.ndarray[Any, Any]) -> float:
        """Compute phase coherence for DB term."""
        if len(signal) < 4:
            return 1.0

        fft_result = fft(signal)
        phases = np.angle(fft_result)

        phase_diffs = np.diff(phases)
        phase_diffs = np.abs(phase_diffs)

        coherence = 1.0 - np.mean(phase_diffs) / np.pi

        return float(max(0.0, min(1.0, coherence)))

    def _compute_harmonic_distortion(self, signal: np.ndarray[Any, Any]) -> float:
        """Compute total harmonic distortion for DB term."""
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
