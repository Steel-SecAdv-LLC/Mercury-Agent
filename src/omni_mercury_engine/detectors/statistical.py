"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Statistical anomaly detector using three Mercury-original mathematical frameworks.

Ensemble composition (replaces prior z-score + IQR + IsolationForest):
    ResonanceScore    (40%) — FFT spectral density profiling
    KinematicScore    (30%) — Physics-based derivative analysis
    InfoGeometryScore (30%) — Fisher Information Mahalanobis OOD

No sklearn anomaly detectors are used.  numpy is the only numerical dependency.
"""

from typing import Any

import numpy as np
import torch

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException


class StatisticalAnomalyDetector(BaseDetector):
    """
    Statistical anomaly detection using three Mercury-original frameworks:

    1. **ResonanceScore** (weight 0.4): At *fit time*, runs FFT on each feature
       column to build spectral density profiles (mean amplitude, noise ratio).
       At *inference time*, scores each sample by how far its per-feature values
       deviate from the training mean, attenuated by the precomputed noise ratio.
       Complexity: O(n*d*log n) fit, O(n*d) inference.

    2. **KinematicScore** (weight 0.3): Computes first-order (velocity),
       second-order (acceleration), and third-order (jerk) derivatives via
       ``np.diff``.  Aggregates via L2 norm.  Effective on temporally-ordered
       data; near-random on shuffled tabular data.
       Complexity: O(n*d) fit and inference.

    3. **InfoGeometryScore** (weight 0.3): Computes the empirical covariance
       matrix at fit time and inverts it (Cholesky or pseudo-inverse for
       singular matrices) to obtain a precision matrix.  At inference,
       computes Mahalanobis distance for each sample — a Fisher Information
       metric on the empirical distribution.
       Complexity: O(n*d^2 + d^3) fit, O(n*d^2) inference.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        # Training statistics
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        # Resonance (FFT) profiles
        self._resonance_mean: np.ndarray | None = None
        self._resonance_std: np.ndarray | None = None
        self._noise_ratio: np.ndarray | None = None

        # Kinematic (derivative) baselines
        self._kinematic_mean: float = 0.0
        self._kinematic_std: float = 1.0

        # Info geometry (precision matrix)
        self._precision_matrix: np.ndarray | None = None
        self._mahal_mean: float = 0.0
        self._mahal_std: float = 1.0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, data: np.ndarray | torch.Tensor) -> StatisticalAnomalyDetector:
        """Fit the detector on training data.

        Precomputes FFT spectral profiles, kinematic derivative baselines,
        and the precision matrix for Mahalanobis scoring.

        Complexity:
            O(n * d) for statistics and kinematic derivatives (np.diff),
            O(n * d * log n) for FFT spectral profiles (once at fit time),
            O(d^3) for covariance inversion.

        Args:
            data: Training data array or tensor.

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty or contains only NaN/Inf values.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.size == 0:
            raise DetectorException(
                "Cannot fit StatisticalAnomalyDetector with empty data. "
                "Provide at least one sample for statistical baseline computation."
            )

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Filter to finite rows
        finite_mask = np.isfinite(data).all(axis=1)
        if not np.any(finite_mask):
            raise DetectorException(
                "Cannot fit StatisticalAnomalyDetector: all data values are NaN or Inf. "
                "Provide data with at least some finite values."
            )
        if not np.all(finite_mask):
            data = data[finite_mask]

        # Core statistics
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8

        # 1. Resonance: FFT spectral profiles  — O(n * d * log n)
        self._precompute_resonance_profiles(data)

        # 2. Kinematic: derivative baselines   — O(n * d)
        self._precompute_kinematic_baselines(data)

        # 3. InfoGeometry: precision matrix     — O(n * d^2 + d^3)
        self._precompute_precision_matrix(data)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # detect
    # ------------------------------------------------------------------
    def detect(self, data: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using the three-component ensemble.

        Returns continuous scores in [0, 1] for each component and
        a weighted combined score.

        Args:
            data: Input data array or tensor.

        Returns:
            Dictionary with keys:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Combined anomaly scores in [0, 1]
                - resonance_scores: ResonanceScore per sample
                - kinematic_scores: KinematicScore per sample
                - info_geometry_scores: InfoGeometryScore per sample
                - detector_type: "statistical"
                - threshold: Effective threshold used
                - calibration_diagnostics: If auto-calibrated
                - isolation_forest_scores: DEPRECATED alias for scores
                - isolation_forest_flags: DEPRECATED alias for is_anomaly
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Component scores
        resonance_scores = self._compute_resonance_score(data)
        kinematic_scores = self._compute_kinematic_score(data)
        info_geometry_scores = self._compute_info_geometry_score(data)

        # Ensemble: 40% resonance, 30% kinematic, 30% info geometry
        combined_scores = (
            resonance_scores * 0.4
            + kinematic_scores * 0.3
            + info_geometry_scores * 0.3
        )

        # Auto-calibration
        effective_threshold = self.threshold
        calibration_diagnostics = None
        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = combined_scores > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "resonance_scores": resonance_scores,
            "kinematic_scores": kinematic_scores,
            "info_geometry_scores": info_geometry_scores,
            # DEPRECATED: will be removed in v2.0 — use "scores" instead
            "isolation_forest_scores": combined_scores,
            # DEPRECATED: will be removed in v2.0 — use "is_anomaly" instead
            "isolation_forest_flags": is_anomaly,
            # Legacy keys mapped to nearest new component
            "z_scores": self._compute_z_scores(data),
            "z_score_continuous": resonance_scores,
            "iqr_scores": kinematic_scores,
            "iqr_flags": is_anomaly,
            "detector_type": "statistical",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    # ------------------------------------------------------------------
    # ResonanceScore — FFT spectral density profiling
    # ------------------------------------------------------------------
    def _precompute_resonance_profiles(self, data: np.ndarray) -> None:
        """Precompute FFT spectral profiles at fit time.

        For each feature column, runs real FFT and records:
        - mean spectral amplitude
        - std spectral amplitude
        - noise ratio (fraction of energy in upper half of spectrum)

        Complexity: O(n * d * log n)
        """
        _n_samples, n_features = data.shape
        self._resonance_mean = np.zeros(n_features)
        self._resonance_std = np.zeros(n_features)
        self._noise_ratio = np.zeros(n_features)

        for j in range(n_features):
            col = data[:, j] - self.mean[j]  # center
            spectrum = np.abs(np.fft.rfft(col))
            self._resonance_mean[j] = np.mean(spectrum)
            self._resonance_std[j] = np.std(spectrum) + 1e-8
            # Noise ratio: fraction of spectral energy in upper half
            mid = max(1, len(spectrum) // 2)
            total_energy = spectrum.sum()
            if total_energy > 1e-8:
                self._noise_ratio[j] = spectrum[mid:].sum() / total_energy
            else:
                self._noise_ratio[j] = 0.5

    def _compute_resonance_score(self, data: np.ndarray) -> np.ndarray:
        """Spectral-profile anomaly score using precomputed FFT statistics.

        At **inference time**, no FFT is performed.  Instead, each sample is
        scored by how far its per-feature values deviate from the training
        mean, attenuated by the precomputed noise ratio (noisier features
        carry less weight).

        Complexity: O(n * d) — element-wise operations, no FFT at inference.
        """
        if self.mean is None or self._noise_ratio is None:
            return np.full(data.shape[0], 0.5)

        deviation = np.abs(data - self.mean) / (self.std + 1e-8)
        # Weight by signal quality: cleaner features matter more
        signal_weight = 1.0 - self._noise_ratio
        signal_weight = signal_weight / (signal_weight.sum() + 1e-8)
        # Exponential decay scoring
        weighted_dev = deviation * signal_weight[np.newaxis, :]
        raw_scores = np.mean(weighted_dev, axis=1)
        scores = 1.0 - np.exp(-raw_scores)
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # KinematicScore — physics-based derivative analysis
    # ------------------------------------------------------------------
    def _precompute_kinematic_baselines(self, data: np.ndarray) -> None:
        """Precompute derivative baselines at fit time.

        Computes first-, second-, and third-order derivatives and records
        the mean and std of the combined kinematic energy on training data.

        Complexity: O(n * d)
        """
        kin = self._raw_kinematic_energy(data)
        self._kinematic_mean = float(np.mean(kin))
        self._kinematic_std = float(np.std(kin)) + 1e-8

    def _raw_kinematic_energy(self, data: np.ndarray) -> np.ndarray:
        """Compute combined kinematic energy per sample.

        Uses np.diff for velocity (1st), acceleration (2nd), jerk (3rd).
        Pads shorter derivative arrays with zeros to match original length.
        """
        n = data.shape[0]
        if n < 2:
            return np.zeros(n)

        velocity = np.diff(data, n=1, axis=0)
        v_norm = (
            np.linalg.norm(velocity, axis=1)
            if velocity.ndim > 1
            else np.abs(velocity.ravel())
        )

        v_padded = np.zeros(n)
        v_padded[1:] = v_norm

        if n < 3:
            return v_padded

        acceleration = np.diff(data, n=2, axis=0)
        a_norm = (
            np.linalg.norm(acceleration, axis=1)
            if acceleration.ndim > 1
            else np.abs(acceleration.ravel())
        )
        a_padded = np.zeros(n)
        a_padded[2:] = a_norm

        if n < 4:
            return v_padded + a_padded

        jerk = np.diff(data, n=3, axis=0)
        j_norm = (
            np.linalg.norm(jerk, axis=1)
            if jerk.ndim > 1
            else np.abs(jerk.ravel())
        )
        j_padded = np.zeros(n)
        j_padded[3:] = j_norm

        return v_padded + a_padded + j_padded

    def _compute_kinematic_score(self, data: np.ndarray) -> np.ndarray:
        """Score using kinematic derivatives (velocity, acceleration, jerk).

        Computes deviation of per-sample kinematic energy from training
        baseline.  Effective on temporally-ordered data; near-random on
        shuffled tabular data (where np.diff is meaningless).

        Complexity: O(n * d)
        """
        kin = self._raw_kinematic_energy(data)
        z = np.abs(kin - self._kinematic_mean) / (self._kinematic_std + 1e-8)
        scores = 1.0 - np.exp(-z / 3.0)
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # InfoGeometryScore — Fisher Information Mahalanobis OOD
    # ------------------------------------------------------------------
    def _precompute_precision_matrix(self, data: np.ndarray) -> None:
        """Compute the precision matrix (inverse covariance) at fit time.

        Uses Cholesky decomposition when possible, falls back to
        pseudo-inverse for singular matrices (d >= n or degenerate features).

        Complexity: O(n * d^2 + d^3)
        """
        n_samples, n_features = data.shape
        if n_features == 0:
            self._precision_matrix = None
            return

        centered = data - self.mean
        # Empirical covariance with regularization
        cov = (centered.T @ centered) / max(n_samples - 1, 1)
        # Tikhonov regularization for numerical stability
        cov += np.eye(n_features) * 1e-6

        try:
            # Cholesky decomposition for well-conditioned matrices
            L = np.linalg.cholesky(cov)
            L_inv = np.linalg.inv(L)
            self._precision_matrix = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular matrices
            self._precision_matrix = np.linalg.pinv(cov)

        # Precompute training Mahalanobis distances for normalization
        mahal = self._raw_mahalanobis(data)
        self._mahal_mean = float(np.mean(mahal))
        self._mahal_std = float(np.std(mahal)) + 1e-8

    def _raw_mahalanobis(self, data: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each sample."""
        if self._precision_matrix is None or self.mean is None:
            return np.zeros(data.shape[0])
        centered = data - self.mean
        mahal_sq = np.sum((centered @ self._precision_matrix) * centered, axis=1)
        return np.sqrt(np.maximum(mahal_sq, 0.0))

    def _compute_info_geometry_score(self, data: np.ndarray) -> np.ndarray:
        """Fisher Information Mahalanobis OOD score.

        Scores each sample by how far it is from the training distribution
        in Mahalanobis distance (a Fisher Information metric).

        Complexity: O(n * d^2)
        """
        mahal = self._raw_mahalanobis(data)
        if self._mahal_std < 1e-8:
            return np.full(data.shape[0], 0.5)
        z = (mahal - self._mahal_mean) / self._mahal_std
        scores = 1.0 - np.exp(-np.maximum(z, 0.0) / 3.0)
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Legacy helpers (preserved for backward compatibility)
    # ------------------------------------------------------------------
    def _compute_z_scores(self, data: np.ndarray) -> np.ndarray:
        """Compute z-scores (diagnostic helper, not a detection component)."""
        if self.std is None or self.mean is None:
            return np.zeros_like(data)
        return (data - self.mean) / self.std

    # ------------------------------------------------------------------
    # Feature extraction for ML fusion
    # ------------------------------------------------------------------
    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract statistical features for ML fusion."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not self._is_fitted:
            self.fit(data)

        z_scores = self._compute_z_scores(data)

        features = np.column_stack(
            [
                np.mean(data, axis=1) if data.shape[1] > 1 else data.flatten(),
                np.std(data, axis=1) if data.shape[1] > 1 else np.zeros(data.shape[0]),
                np.max(np.abs(z_scores), axis=1),
                np.mean(np.abs(z_scores), axis=1),
            ]
        )

        if features.shape[1] < 10:
            padding = np.zeros((features.shape[0], 10 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)
