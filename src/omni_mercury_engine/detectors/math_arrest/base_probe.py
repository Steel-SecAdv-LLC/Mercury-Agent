# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
    """Abstract base class for all Anomaly Math Arrest equation probes.

    Every probe must implement ``fit_trajectory`` and ``deviation_score``. Utility methods handle
    input validation, score normalization, and dimensionality reduction so that individual probes
    stay focused on their core mathematics.
    """

    def __init__(self, *, min_samples: int = MIN_SAMPLES) -> None:
        """Initialize the instance."""
        self._min_samples = min_samples
        self._is_fitted: bool = False
        self._fit_quality: float = 0.0
        #: Fit-time 99th percentile of this probe's raw deviations, set by
        #: :meth:`calibrate_normalizer`. ``None`` falls back to the scored
        #: array's own percentile -- see :meth:`_normalize_scores`.
        self._norm_scale: float | None = None
        self._capturing_norm: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn normal evolution parameters from training data.

        Args:
            data: Training data, shape ``(n_samples,)`` or
                ``(n_samples, n_features)``.

        Raises:
            ValueError: If *data* has fewer than ``min_samples`` samples.
        """

    @abstractmethod
    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Compute per-sample deviation scores.

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
        """Validate that *data* is non-empty and has enough samples.

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
        """Reduce 2-D input to 1-D via column-wise mean.

        Args:
            data: Array of shape ``(n,)`` or ``(n, m)``.

        Returns:
            1-D array of shape ``(n,)``.
        """
        if data.ndim == 2:
            return np.mean(data, axis=1).astype(np.float64)
        return data.astype(np.float64)

    def _normalize_scores(
        self,
        raw: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Normalize raw deviations to ``[0, 1]`` against the **fit-time** scale.

        The divisor is the 99th percentile of the raw deviations measured on the
        fit series (:meth:`calibrate_normalizer`), not of the array being
        scored.

        Scoring against the scored array's own percentile is what this replaced,
        and it made a score mean different things on different calls: a quiet
        batch's deviations were stretched to fill ``[0, 1]`` exactly as a violent
        batch's were, so the absolute value carried no information about how
        anomalous the batch was -- only about a point's rank *within its batch*.
        Within one call, and *per probe*, the ranking is the same either way,
        which is why the defect was invisible to every ranking measurement.

        Per probe is the necessary qualifier, and this docstring used to omit
        it. The 21 probes are averaged by the fusion step, so each probe's
        scale sets its implicit weight in that mean, and changing the scale
        does move the fused ranking: the repair itself cost 0.0087 of NAB
        AUROC. ``research/math_arrest/harness_norm_scale.py`` sweeps that
        choice over ten estimators and records the result -- the scale is a
        dial along an AUROC / NAB-score trade-off, and every alternative
        bought its aggregate gain by regressing individual files past
        tolerance, so this one stays.

        What it broke is the alarm decision, which compares an absolute score
        against a fixed conformal cut. Measured on the recorded NAB files with a
        cut promising alpha = 0.05, the false-alarm rate was a function of the
        caller's batch size rather than of the data:

        ===========  ===================  ===================
        batch         realAWSCloudwatch    realAdExchange
        ===========  ===================  ===================
        whole series  0.0520               0.0176
        256           0.1769               0.0535
        128           0.2176               0.0711
        64            0.3037               0.0986
        32            0.4612               0.1269
        ===========  ===================  ===================

        So a streaming caller feeding 32-point chunks saw a 46% false-alarm rate
        from a detector promising 5%, and the published 0.0615 stationary pin
        held only for the whole-series geometry it was measured in. Carrying the
        fit-time scale makes the rate a property of the data again.

        This is the same defect class as the fit-time intercept in
        ``AdditiveProbe`` and the fitted basis in ``SVDProjectionProbe``:
        fit-time coordinates not carried to score time. Those two were one probe
        each; this one sat in the normalizer all 21 share.

        **What the repair costs, measured.** The released normalizer stretched
        every batch to fill ``[0, 1]``, which made the detector more sensitive
        than its cut intended -- raising true detections along with false ones.
        Carrying the fit-time scale removes that inflation, and the
        whole-series figures fall with it:

        ==========================  ==========  ==========  ============
        variant                      NAB AUROC   NAB score   batch-stable
        ==========================  ==========  ==========  ============
        released (per-call p99)      0.7581      40.4        no
        fit-time scale (shipped)     0.7494      33.67       yes
        ==========================  ==========  ==========  ============

        The released 40.4 is not a figure the repair could have kept: it was
        produced by the same inflation that made the alarm rate a function of
        batch size, and it was only ever valid for whole-series scoring, the one
        geometry it was measured in. A detector that reports 46% false alarms to
        a streaming caller is not usable at any ranking score.

        **A monotone variant was measured and refused.** Clipping at the
        fit-time percentile ties every deviation above it at 1.0, which costs
        top-end ordering, so ``1 - exp(-raw/scale)`` -- strictly increasing,
        same ``[0, 1]`` codomain, no ties -- was tried. It recovers the ranking
        (AUROC 0.7575, within 0.0006 of the released value) and is equally
        batch-stable, but its NAB standard score is **29.69** against this
        repair's 33.67. The ledger's own precedent decides it: the NAB standard
        score charges for a false alarm where AUROC is blind to one
        (``docs/DORMANCY_LEDGER.md`` §2e, where the decorrelator was kept on
        exactly that reasoning), so the variant that scores better on the
        purpose-built streaming metric is the one that ships, and the ranking
        loss is recorded rather than optimised away.

        An uncalibrated probe (``_norm_scale is None``) keeps the old per-call
        behaviour, so a probe used directly without
        :meth:`calibrate_normalizer` is unchanged.

        Args:
            raw: Raw deviation values (non-negative expected).

        Returns:
            Clipped scores in ``[0, 1]``.
        """
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self._capturing_norm:
            self._norm_scale = float(np.percentile(raw, 99))
        scale = self._norm_scale
        if scale is None:
            scale = float(np.percentile(raw, 99))
        if scale < EPSILON:
            return np.zeros_like(raw, dtype=np.float64)
        normalized: npt.NDArray[np.float64] = np.clip(raw / scale, 0.0, 1.0)
        return normalized

    def calibrate_normalizer(self, data: npt.NDArray[np.float64]) -> None:
        """Record the fit-time score scale by scoring the fit series once.

        Called by :meth:`AnomalyMathArrest.fit` immediately after
        ``fit_trajectory``, so every probe's published scores are expressed
        against its own fit distribution. Each probe derives ``raw``
        differently, so the scale is captured through the probe's real
        ``deviation_score`` path rather than recomputed here.

        Args:
            data: The series the probe was just fitted on.
        """
        self._validate_fitted()
        self._norm_scale = None
        self._capturing_norm = True
        try:
            self.deviation_score(data)
        finally:
            self._capturing_norm = False

    @staticmethod
    def _r_squared(
        actual: npt.NDArray[np.float64],
        predicted: npt.NDArray[np.float64],
    ) -> float:
        """Compute R-squared, clamped to ``[0, 1]``.

        Returns ``1.0`` for constant data when the residual is negligible.
        """
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        if ss_tot < EPSILON:
            return 1.0 if ss_res < EPSILON else 0.0
        r2 = 1.0 - ss_res / ss_tot
        return float(np.clip(r2, 0.0, 1.0))
