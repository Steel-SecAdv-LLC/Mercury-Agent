# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Abstract base class and shared constants for all equation probes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Centralized constants
# ---------------------------------------------------------------------------
PHI: float = 1.618033988749895
EPSILON: float = 1e-10
MIN_SAMPLES: int = 8
CATALAN_G: float = 0.9159655941772190


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Immutable result from a single equation probe."""

    probe_name: str
    deviation_scores: npt.NDArray[np.float64]
    confidence: float
    trajectory_fit_quality: float
    anomaly_geometry: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseEquationProbe(ABC):
    """
    Abstract base class for all Anomaly Math Arrest equation probes.

    Every probe must implement ``fit_trajectory`` and ``deviation_score``. Utility methods handle
    input validation, score normalization, and dimensionality reduction so that individual probes
    stay focused on their core mathematics.
    """

    def __init__(self, *, min_samples: int = MIN_SAMPLES) -> None:
        self._min_samples = min_samples
        self._is_fitted: bool = False
        self._fit_quality: float = 0.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """
        Learn normal evolution parameters from training data.

        Args:
            data: Training data, shape ``(n_samples,)`` or
                ``(n_samples, n_features)``.

        Raises:
            ValueError: If *data* has fewer than ``min_samples`` samples.
        """

    @abstractmethod
    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """
        Compute per-sample deviation scores.

        Args:
            data: Evaluation data, same shape convention as
                ``fit_trajectory``.

        Returns:
            A :class:`ProbeResult` with scores in ``[0, 1]``.

        Raises:
            RuntimeError: If the probe has not been fitted.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit_trajectory` has been called successfully."""
        return self._is_fitted

    # ------------------------------------------------------------------
    # Guard utilities
    # ------------------------------------------------------------------

    def _validate_fitted(self) -> None:
        """Raise ``RuntimeError`` if the probe has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. Call fit_trajectory() first."
            )

    def _validate_data(
        self,
        data: npt.NDArray[np.float64],
        min_n: int | None = None,
    ) -> None:
        """
        Validate that *data* is non-empty and has enough samples.

        Args:
            data: Input array.
            min_n: Minimum number of samples.  Defaults to
                ``self._min_samples``.

        Raises:
            ValueError: If *data* is empty or too short.
        """
        if data.size == 0:
            raise ValueError("Input data is empty.")
        n = data.shape[0]
        required = min_n if min_n is not None else self._min_samples
        if n < required:
            raise ValueError(
                f"{type(self).__name__} requires at least {required} samples, got {n}."
            )

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_1d(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Reduce 2-D input to 1-D via column-wise mean.

        Args:
            data: Array of shape ``(n,)`` or ``(n, m)``.

        Returns:
            1-D array of shape ``(n,)``.
        """
        if data.ndim == 2:
            return np.mean(data, axis=1).astype(np.float64)
        return data.astype(np.float64)

    @staticmethod
    def _normalize_scores(
        raw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Normalize raw scores to ``[0, 1]`` using the 99th percentile.

        Args:
            raw: Raw deviation values (non-negative expected).

        Returns:
            Clipped scores in ``[0, 1]``.
        """
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        p99 = float(np.percentile(raw, 99))
        if p99 < EPSILON:
            return np.zeros_like(raw, dtype=np.float64)
        normalized: npt.NDArray[np.float64] = np.clip(raw / p99, 0.0, 1.0)
        return normalized

    @staticmethod
    def _r_squared(
        actual: npt.NDArray[np.float64],
        predicted: npt.NDArray[np.float64],
    ) -> float:
        """
        Compute R-squared, clamped to ``[0, 1]``.

        Returns ``1.0`` for constant data when the residual is negligible.
        """
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        if ss_tot < EPSILON:
            return 1.0 if ss_res < EPSILON else 0.0
        r2 = 1.0 - ss_res / ss_tot
        return float(np.clip(r2, 0.0, 1.0))
