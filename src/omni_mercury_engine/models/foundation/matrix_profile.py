# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Matrix Profile Integration using STUMPY.

Provides O(n log n) time complexity for pattern discovery and
anomaly detection in time series data.

Matrix Profile is a data structure and algorithm that enables:
    - Discord detection (anomalies)
    - Motif discovery (repeated patterns)
    - Semantic segmentation
    - Time series chains

Reference:
    STUMPY: https://github.com/TDAmeritrade/stumpy
    Matrix Profile: https://www.cs.ucr.edu/~eamonn/MatrixProfile.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.models.foundation.base_foundation import (
    BaseFoundationModel,
    FoundationModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class MatrixProfileConfig(FoundationModelConfig):
    """Configuration for Matrix Profile detector.

    Attributes:
        window_size: Subsequence window size (m)
        n_discords: Number of discords (anomalies) to find
        n_motifs: Number of motifs to find
        normalize: Whether to z-normalize subsequences
        discord_threshold: Threshold for discord detection (z-score)
        use_gpu: Use GPU acceleration (requires stumpy.gpu)
    """

    window_size: int = 50
    n_discords: int = 10
    n_motifs: int = 3
    normalize: bool = True
    discord_threshold: float = 2.0
    use_gpu: bool = True
    model_name: str = "matrix_profile"


class MatrixProfileDetector(BaseFoundationModel):
    """Matrix Profile-based anomaly detector using STUMPY.

    Uses the Matrix Profile algorithm for efficient time series
    anomaly detection via discord discovery.

    Features:
        - O(n log n) complexity with GPU acceleration
        - Discord detection (anomalies = subsequences with no close match)
        - Motif discovery (repeated patterns)
        - No training required

    Example:
        >>> detector = MatrixProfileDetector(window_size=50)
        >>> results = detector.detect_anomalies(time_series)
        >>> discords = results['discord_indices']
        >>> motifs = results['motif_indices']
    """

    def __init__(self, config: MatrixProfileConfig | dict[str, Any] | None = None) -> None:
        """Initialize Matrix Profile detector.

        Args:
            config: Detector configuration
        """
        if config is None:
            config = MatrixProfileConfig()
        elif isinstance(config, dict):
            config = MatrixProfileConfig(**config)

        # Set typed config BEFORE calling super().__init__() to avoid AttributeError
        # when the base class accesses self.config property
        self.mp_config: MatrixProfileConfig = config
        super().__init__(config)

        self._stumpy_available = False
        self._gpu_available = False

    @property
    def config(self) -> MatrixProfileConfig:
        """Return the typed Matrix Profile configuration."""
        return self.mp_config

    @config.setter
    def config(self, value: dict[str, Any] | MatrixProfileConfig) -> None:
        """Store the underlying config object (required for base class compatibility).

        The base class sets self.config to a dict during __init__. We intercept this and store it,
        but always return the typed config.
        """
        if isinstance(value, MatrixProfileConfig):
            self.mp_config = value
        # If dict, it's from base class init - we already have typed config set

    def _initialize_model(self) -> None:
        """Check STUMPY availability."""
        import importlib

        try:
            importlib.import_module("stumpy")
        except Exception as exc:
            raise NotImplementedError(
                "STUMPY not installed or not importable "
                f"({type(exc).__name__}: {exc}). "
                "Install with: pip install stumpy. "
                "Silent mock degradation is not permitted."
            ) from exc

        self._stumpy_available = True
        logger.info("STUMPY library loaded")

        if self.mp_config.use_gpu:
            try:
                importlib.import_module("stumpy.gpu")
            except Exception as exc:
                logger.info(
                    "STUMPY GPU not available (%s: %s); using CPU",
                    type(exc).__name__,
                    exc,
                )
            else:
                self._gpu_available = True
                logger.info("STUMPY GPU acceleration available")

    def compute_matrix_profile(
        self,
        series: np.ndarray[Any, Any],
        window_size: int | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Compute the Matrix Profile for a time series.

        Args:
            series: Input time series [T]
            window_size: Subsequence window size (default from config)

        Returns:
            Dict containing:
                - matrix_profile: Distance to nearest neighbor [T-m+1]
                - profile_index: Index of nearest neighbor [T-m+1]
        """
        self._ensure_initialized()

        window_size = window_size or self.mp_config.window_size

        series = np.asarray(series)
        if series.ndim != 1:
            raise ValueError(
                f"Matrix Profile expects a 1-D time series; got shape {tuple(series.shape)}. "
                "Pass a single univariate series [T]."
            )
        if series.shape[0] <= window_size:
            raise ValueError(
                "Matrix Profile expects a 1-D time series longer than "
                f"window_size={window_size}; got {series.shape[0]} points. "
                "Provide a longer series or configure a smaller window_size."
            )

        if not self._stumpy_available:
            raise RuntimeError(
                "STUMPY is not available for Matrix Profile computation. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            )

        import stumpy

        try:
            if self._gpu_available and self.mp_config.use_gpu:
                import stumpy.gpu

                result = stumpy.gpu_stump(series, m=window_size)
            else:
                result = stumpy.stump(series, m=window_size)

            return {
                "matrix_profile": result[:, 0].astype(float),
                "profile_index": result[:, 1].astype(int),
                "left_index": result[:, 2].astype(int),
                "right_index": result[:, 3].astype(int),
            }

        except Exception as e:
            raise RuntimeError(
                f"Matrix Profile computation failed: {e}. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            ) from e

    def find_discords(
        self,
        series_or_mp: np.ndarray[Any, Any] | torch.Tensor,
        top_k: int | None = None,
        exclusion_zone: int | None = None,
        is_matrix_profile: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Find discords (anomalies) in a time series or Matrix Profile.

        Discords are subsequences with the largest Matrix Profile values,
        meaning they have no close neighbors (unusual patterns).

        Args:
            series_or_mp: Time series or pre-computed Matrix Profile array
            top_k: Number of discords to find (alias for n_discords)
            exclusion_zone: Exclusion zone size (default: window_size // 2)
            is_matrix_profile: Whether ``series_or_mp`` is a pre-computed
                Matrix Profile. When None (default), the input is treated as a
                RAW SERIES and its matrix profile is computed; pass
                ``is_matrix_profile=True`` to supply a pre-computed profile
                directly (otherwise a profile-of-a-profile would be computed)

        Behaviour change (2026-07, PR #339): ``is_matrix_profile=None``
        previously guessed via a length heuristic (``len <= 2*window`` =>
        profile), which misclassified realistic pre-computed profiles
        (length ~ n - window + 1) as raw series and silently computed a
        profile-of-a-profile. The undeclared default is now always RAW
        SERIES; callers passing a pre-computed profile must say so
        explicitly with ``is_matrix_profile=True``.

        Returns:
            List of discord info dicts with index and score
        """
        self._ensure_initialized()

        # Convert torch tensor if needed (detach so a grad-tracking tensor
        # does not raise on .numpy()).
        if isinstance(series_or_mp, torch.Tensor):
            series_or_mp = series_or_mp.detach().cpu().numpy()

        if is_matrix_profile is None:
            # Treat an undeclared input as a RAW SERIES (compute its matrix
            # profile). The previous length heuristic misclassified real
            # matrix profiles -- whose length is ~ n - window + 1, typically
            # >> 2*window -- as raw series and silently computed a
            # profile-of-a-profile; a precomputed profile must be declared
            # explicitly with is_matrix_profile=True.
            is_matrix_profile = False

        if is_matrix_profile:
            matrix_profile = series_or_mp
        else:
            # Compute matrix profile from series
            mp_result = self.compute_matrix_profile(series_or_mp)
            matrix_profile = mp_result["matrix_profile"]

        n_discords = top_k or self.mp_config.n_discords
        exclusion_zone = exclusion_zone or self.mp_config.window_size // 2

        discords = []
        mp_copy = matrix_profile.copy()

        for _ in range(n_discords):
            if np.all(np.isinf(mp_copy)):
                break

            # Find max (most anomalous)
            idx = np.argmax(mp_copy)
            score = mp_copy[idx]

            if np.isinf(score):
                break

            discords.append(
                {
                    "index": int(idx),
                    "score": float(score),
                }
            )

            # Apply exclusion zone
            start = max(0, idx - exclusion_zone)
            end = min(len(mp_copy), idx + exclusion_zone + 1)
            mp_copy[start:end] = -np.inf  # type: ignore[misc, unused-ignore]

        return discords

    def find_motifs(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        top_k: int | None = None,
        matrix_profile: np.ndarray[Any, Any] | None = None,
        profile_index: np.ndarray[Any, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find motifs (repeated patterns) in the time series.

        Motifs are pairs of subsequences with the smallest Matrix Profile
        values, meaning they are very similar to each other.

        Args:
            series: Original time series
            top_k: Number of motifs to find (alias for n_motifs)
            matrix_profile: Pre-computed Matrix Profile array (optional)
            profile_index: Pre-computed Profile index array (optional)

        Returns:
            List of motif info dicts with index1, index2, and distance
        """
        self._ensure_initialized()

        # Convert torch tensor if needed
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim > 1:
            series = series.flatten()

        # Compute matrix profile if not provided
        if matrix_profile is None or profile_index is None:
            mp_result = self.compute_matrix_profile(series)
            matrix_profile = mp_result["matrix_profile"]
            profile_index = mp_result["profile_index"]

        n_motifs = top_k or self.mp_config.n_motifs

        try:
            motifs = []
            mp_copy = matrix_profile.copy()
            exclusion_zone = self.mp_config.window_size // 2

            for _ in range(n_motifs):
                # Find minimum (most similar pair)
                idx = np.argmin(mp_copy)
                partner_idx = profile_index[idx]

                if np.isinf(mp_copy[idx]):
                    break

                motifs.append(
                    {
                        "index1": int(idx),
                        "index2": int(partner_idx),
                        "distance": float(mp_copy[idx]),
                    }
                )

                # Exclude both motif locations
                for loc in [idx, partner_idx]:
                    start = max(0, loc - exclusion_zone)
                    end = min(len(mp_copy), loc + exclusion_zone + 1)
                    mp_copy[start:end] = np.inf

            return motifs

        except Exception as e:
            logger.warning(f"Motif discovery failed: {e}")
            return []

    def forecast(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Generate forecasts using motif-based prediction.

        Uses discovered motifs to predict future values based on
        similar historical patterns.

        Args:
            series: Input time series
            horizon: Forecast horizon

        Returns:
            Forecast dict
        """
        self._ensure_initialized()

        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim > 1:
            series = series.flatten()

        horizon = horizon or self.foundation_config.prediction_length

        # Compute matrix profile
        mp_result = self.compute_matrix_profile(series)

        # Find most similar historical pattern to recent data
        recent_start = len(series) - self.mp_config.window_size
        if recent_start < 0:
            # Series too short, use naive forecast
            return {
                "forecast": np.full(horizon, series[-1]),
                "lower": np.full(horizon, series[-1] * 0.9),
                "upper": np.full(horizon, series[-1] * 1.1),
            }

        # Find nearest neighbor of recent pattern
        mp = mp_result["matrix_profile"]
        pi = mp_result["profile_index"]

        # Get the match index for the end of the series
        match_idx = pi[recent_start] if recent_start < len(pi) else len(series) - 1

        # Forecast = what happened after the historical match
        forecast_start = match_idx + self.mp_config.window_size
        forecast_end = min(forecast_start + horizon, len(series))

        if forecast_end > forecast_start:
            forecast = series[forecast_start:forecast_end]
            # Pad if needed
            if len(forecast) < horizon:
                forecast = np.pad(forecast, (0, horizon - len(forecast)), mode="edge")
        else:
            forecast = np.full(horizon, series[-1])

        # Confidence intervals based on MP value (higher = less confident)
        confidence = 1.0 / (1.0 + mp[recent_start])
        margin = np.std(series) * (1 - confidence) * 2

        return {
            "forecast": forecast[:horizon],
            "lower": forecast[:horizon] - margin,
            "upper": forecast[:horizon] + margin,
        }

    def detect_anomalies(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies using Matrix Profile discords.

        Args:
            series: Input time series

        Returns:
            Dict with scores, is_anomaly, discord details
        """
        self._ensure_initialized()

        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim > 1:
            series = series.flatten()

        # Compute Matrix Profile
        mp_result = self.compute_matrix_profile(series)
        mp = mp_result["matrix_profile"]

        # Find discords in the already-computed profile
        discords = self.find_discords(mp, is_matrix_profile=True)

        # Create scores array (normalized MP values)
        scores = np.zeros(len(series))

        # Map MP scores to original series indices
        for i, val in enumerate(mp):
            # Each MP value covers indices [i, i + window_size)
            end = min(i + self.mp_config.window_size, len(series))
            scores[i:end] = np.maximum(scores[i:end], val)

        # Normalize scores to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        # Threshold based on percentile
        threshold = np.percentile(scores, self.foundation_config.anomaly_threshold * 100)
        is_anomaly = scores > threshold

        # Find motifs too
        motifs = self.find_motifs(
            series,
            matrix_profile=mp_result["matrix_profile"],
            profile_index=mp_result["profile_index"],
        )

        return {
            "scores": scores,
            "is_anomaly": is_anomaly,
            "threshold": threshold,
            "discords": discords,
            "motifs": motifs,
            "matrix_profile": mp,
            "discord_indices": [d["index"] for d in discords],
        }

    def detect(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies in time series data.

        This is the primary detection interface that wraps detect_anomalies
        for a consistent API across all foundation model adapters.

        Args:
            series: Input time series [T] or [B, T]

        Returns:
            Dict with scores, is_anomaly flags, discords, and threshold
        """
        return self.detect_anomalies(series)
