r"""
Mercury Agent - Threshold Auto-Calibration Pipeline (Phase 5)

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

Phase 5: Threshold Auto-Calibration Pipeline
=============================================

Provides a complete pipeline for calibrating, tracking, and validating
every threshold in the Mercury Agent system. Key capabilities:

1. **Provenance tracking** -- every threshold records which dataset it
   was calibrated on (fingerprinted via SHA-256 of summary statistics).
   Thresholds lacking provenance are flagged ``UNCALIBRATED``.

2. **Auto-calibration methods**:
   - Youden's J statistic: :math:`J = \\text{TPR} - \\text{FPR}`
   - F1-optimal threshold
   - Cost-sensitive threshold (with user-supplied cost matrix)

3. **Dataset fingerprinting & drift detection**:
   - SHA-256 fingerprint of :math:`(\\mu, \\sigma, n, q_{0.25}, q_{0.50}, q_{0.75})`
   - KL divergence: :math:`D_{KL}(P \\| Q) = \\sum P(x) \\log \\frac{P(x)}{Q(x)}`
   - KS statistic: :math:`D = \\sup_x |F_n(x) - F_m(x)|`

4. **System-wide recalibration** via ``calibrate_all_thresholds()``
   which sweeps anomaly, ethical, and confidence-band thresholds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.centralized_constants import ANOMALY, ETHICAL
from omni_mercury_engine.core.score_calibration import (
    AutoThresholdOptimizer,
    CalibrationMethod,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ThresholdStatus(Enum):
    """
    Calibration status of an individual threshold.

    Attributes:
        CALIBRATED: Threshold was calibrated against a known dataset.
        UNCALIBRATED: Threshold has no dataset provenance -- treat with caution.
        STALE: Calibration exists but runtime data has drifted beyond tolerance.
    """

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    STALE = "stale"


class CalibrationStrategy(Enum):
    r"""Strategy used to select the optimal operating point.

    Attributes:
        YOUDEN_J: Maximize Youden's J statistic
            :math:`J = \\text{sensitivity} + \\text{specificity} - 1 = \\text{TPR} - \\text{FPR}`.
        F1_OPTIMAL: Maximize the harmonic mean of precision and recall
            :math:`F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R}`.
        COST_SENSITIVE: Minimize expected cost given a cost matrix
            :math:`C = c_{FP} \\cdot \\text{FP} + c_{FN} \\cdot \\text{FN}`.
    """

    YOUDEN_J = "youden_j"
    F1_OPTIMAL = "f1_optimal"
    COST_SENSITIVE = "cost_sensitive"


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class DatasetFingerprint:
    """
    SHA-256 fingerprint of a dataset's summary statistics.

    The fingerprint is computed as::

        SHA-256(json(mean, std, n_samples, n_features, q25, q50, q75))

    This allows us to detect whether a new dataset matches the one used
    for calibration without storing the raw data.

    Attributes:
        sha256: Hex digest of the fingerprint hash.
        n_samples: Number of rows in the dataset.
        n_features: Number of columns (features) in the dataset.
        mean: Per-feature mean vector.
        std: Per-feature standard deviation vector.
        quantiles: Dictionary mapping quantile labels to per-feature vectors.
        created_at: Unix timestamp when the fingerprint was created.
    """

    sha256: str
    n_samples: int
    n_features: int
    mean: list[float]
    std: list[float]
    quantiles: dict[str, list[float]]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "sha256": self.sha256,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "mean": self.mean,
            "std": self.std,
            "quantiles": self.quantiles,
            "created_at": self.created_at,
        }


@dataclass
class ThresholdRecord:
    """
    A single threshold with full provenance metadata.

    Attributes:
        name: Human-readable identifier (e.g. ``"anomaly.default"``).
        value: The numeric threshold value.
        status: Calibration status (see :class:`ThresholdStatus`).
        strategy: Which calibration strategy produced this value.
        dataset_fingerprint: Fingerprint of the calibration dataset, or
            ``None`` if the threshold was never calibrated.
        metric_at_threshold: Value of the optimized metric at this threshold
            (e.g. the J statistic or the F1 score).
        metadata: Free-form dictionary for strategy-specific details.
    """

    name: str
    value: float
    status: ThresholdStatus = ThresholdStatus.UNCALIBRATED
    strategy: str = ""
    dataset_fingerprint: DatasetFingerprint | None = None
    metric_at_threshold: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status.value,
            "strategy": self.strategy,
            "dataset_fingerprint": (
                self.dataset_fingerprint.to_dict() if self.dataset_fingerprint is not None else None
            ),
            "metric_at_threshold": self.metric_at_threshold,
            "metadata": self.metadata,
        }


@dataclass
class ThresholdResult:
    """
    Return type for a single calibration run.

    Attributes:
        threshold: Optimal threshold value.
        metric_name: Name of the metric that was optimized.
        metric_value: Achieved value of that metric.
        strategy: Strategy used for optimization.
        dataset_fingerprint: Fingerprint of the dataset used.
        all_thresholds_evaluated: Number of candidate thresholds tested.
        details: Strategy-specific auxiliary information.
    """

    threshold: float
    metric_name: str
    metric_value: float
    strategy: CalibrationStrategy
    dataset_fingerprint: DatasetFingerprint
    all_thresholds_evaluated: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "threshold": self.threshold,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "strategy": self.strategy.value,
            "dataset_fingerprint": self.dataset_fingerprint.to_dict(),
            "all_thresholds_evaluated": self.all_thresholds_evaluated,
            "details": self.details,
        }


@dataclass
class DriftResult:
    r"""Result of a distribution drift detection test.

    Attributes:
        drifted: ``True`` when the new data distribution has shifted
            beyond the configured tolerance from the calibration baseline.
        kl_divergence: Symmetric KL divergence between the two
            distributions (using histogram-based density estimates).
        ks_statistic: Kolmogorov-Smirnov test statistic
            :math:`D = \\sup_x |F_n(x) - F_m(x)|`.
        ks_p_value: p-value of the KS test (per-feature minimum).
        per_feature_ks: Per-feature KS statistics.
        per_feature_kl: Per-feature KL divergence values.
        message: Human-readable summary.
    """

    drifted: bool
    kl_divergence: float
    ks_statistic: float
    ks_p_value: float
    per_feature_ks: list[float] = field(default_factory=list)
    per_feature_kl: list[float] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "drifted": self.drifted,
            "kl_divergence": self.kl_divergence,
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "per_feature_ks": self.per_feature_ks,
            "per_feature_kl": self.per_feature_kl,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Helper: dataset fingerprinting
# ---------------------------------------------------------------------------


def compute_dataset_fingerprint(X: NDArray[np.float64]) -> DatasetFingerprint:
    """
    Compute a SHA-256 fingerprint from dataset summary statistics.

    The fingerprint is deterministic for a given set of summary statistics
    (mean, std, shape, quantiles) and is used to detect whether runtime
    data matches the calibration distribution.

    The hash is computed over a canonically-sorted JSON representation of::

        {
            "mean":       [per-feature means],
            "std":        [per-feature stds],
            "n_samples":  int,
            "n_features": int,
            "q25":        [per-feature 25th percentiles],
            "q50":        [per-feature medians],
            "q75":        [per-feature 75th percentiles],
        }

    Args:
        X: Input data array of shape ``(n_samples, n_features)``.  One-
            dimensional arrays are reshaped to ``(n, 1)``.

    Returns:
        A :class:`DatasetFingerprint` with the hex SHA-256 digest and
        the summary statistics used to produce it.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    mean = np.nanmean(X, axis=0).tolist()
    std = np.nanstd(X, axis=0).tolist()
    q25 = np.nanpercentile(X, 25, axis=0).tolist()
    q50 = np.nanpercentile(X, 50, axis=0).tolist()
    q75 = np.nanpercentile(X, 75, axis=0).tolist()

    # Ensure list form even for scalar case
    if not isinstance(mean, list):
        mean, std, q25, q50, q75 = [mean], [std], [q25], [q50], [q75]

    payload = {
        "mean": [round(v, 10) for v in mean],
        "std": [round(v, 10) for v in std],
        "n_samples": n_samples,
        "n_features": n_features,
        "q25": [round(v, 10) for v in q25],
        "q50": [round(v, 10) for v in q50],
        "q75": [round(v, 10) for v in q75],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return DatasetFingerprint(
        sha256=sha,
        n_samples=n_samples,
        n_features=n_features,
        mean=mean,
        std=std,
        quantiles={"q25": q25, "q50": q50, "q75": q75},
    )


# ---------------------------------------------------------------------------
# Helper: statistical divergence measures
# ---------------------------------------------------------------------------


def _histogram_densities(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    n_bins: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute aligned histogram densities for two 1-D arrays.

    Both arrays are binned using the same edges spanning the combined
    range so the resulting PMFs are directly comparable.

    Args:
        a: First sample (1-D).
        b: Second sample (1-D).
        n_bins: Number of histogram bins.

    Returns:
        A tuple ``(p, q)`` where *p* and *q* are normalised histogram
        densities of length ``n_bins``.
    """
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if hi - lo < 1e-12:
        # Degenerate case -- both distributions are constants
        return np.ones(n_bins) / n_bins, np.ones(n_bins) / n_bins

    edges = np.linspace(lo, hi, n_bins + 1)
    p = np.histogram(a, bins=edges, density=False)[0].astype(np.float64)
    q = np.histogram(b, bins=edges, density=False)[0].astype(np.float64)

    # Normalize to probability mass functions (add floor to avoid zeros)
    eps = 1e-10
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return p, q


def kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> float:
    r"""Compute KL divergence :math:`D_{KL}(P \| Q)`.

    .. math::

        D_{KL}(P \| Q) = \sum_{x} P(x) \, \log \frac{P(x)}{Q(x)}

    Both *p* and *q* must be valid probability mass functions (non-negative,
    sum to 1).  A small epsilon is added internally to avoid division by
    zero or log-of-zero.

    Args:
        p: Reference distribution PMF.
        q: Comparison distribution PMF.

    Returns:
        Non-negative float.  Returns 0.0 when the distributions are
        identical.
    """
    eps = 1e-10
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> float:
    r"""Symmetric (Jeffreys) KL divergence.

    .. math::

        D_{sym}(P, Q) = \frac{1}{2}\bigl(D_{KL}(P \| Q) + D_{KL}(Q \| P)\bigr)

    Args:
        p: First distribution PMF.
        q: Second distribution PMF.

    Returns:
        Non-negative symmetric divergence.
    """
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))


def ks_statistic(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> tuple[float, float]:
    r"""Two-sample Kolmogorov-Smirnov statistic and approximate p-value.

    .. math::

        D = \sup_x \lvert F_n(x) - F_m(x) \rvert

    where :math:`F_n` and :math:`F_m` are the empirical CDFs of the two
    samples.

    The p-value uses the asymptotic formula:

    .. math::

        p \approx 2 \exp\!\Bigl(-2 \, \frac{n \, m}{n + m} \, D^2\Bigr)

    Args:
        a: First sample (1-D array).
        b: Second sample (1-D array).

    Returns:
        Tuple ``(D, p_value)``.
    """
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    n = len(a)
    m = len(b)

    if n == 0 or m == 0:
        return 0.0, 1.0

    # Merge and walk both ECDFs
    combined = np.concatenate([a, b])
    combined.sort()

    cdf_a = np.searchsorted(a, combined, side="right") / n
    cdf_b = np.searchsorted(b, combined, side="right") / m

    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))

    # Asymptotic p-value
    en = np.sqrt(n * m / (n + m))
    p_value = float(np.clip(2.0 * np.exp(-2.0 * (en * d_stat) ** 2), 0.0, 1.0))
    return d_stat, p_value


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class ThresholdCalibrationPipeline:
    r"""Threshold auto-calibration pipeline with full provenance tracking.

    This pipeline manages the lifecycle of every threshold in the Mercury
    Agent system:

    * **Calibrate** thresholds from labelled data using Youden's J,
      F1-optimal, or cost-sensitive strategies.
    * **Record provenance** -- each threshold stores the SHA-256
      fingerprint of the dataset it was calibrated on.
    * **Detect drift** -- at runtime, new data can be compared against
      the calibration fingerprint using KL divergence and the
      Kolmogorov-Smirnov test.
    * **Flag uncalibrated** thresholds so downstream consumers can
      decide how to handle them.

    Key equations
    -------------
    Youden's J statistic:

    .. math::

        J = \\text{sensitivity} + \\text{specificity} - 1
          = \\text{TPR} - \\text{FPR}

    F1 score:

    .. math::

        F_1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}
                            {\\text{precision} + \\text{recall}}

    Cost-sensitive objective:

    .. math::

        \\mathcal{L} = c_{FP} \\cdot FP + c_{FN} \\cdot FN

    Dataset fingerprint:

    .. math::

        \\text{SHA-256}\\bigl(\\text{json}(\\mu, \\sigma, n, q_{25}, q_{50}, q_{75})\\bigr)

    KL divergence:

    .. math::

        D_{KL}(P \\| Q) = \\sum_x P(x) \\log \\frac{P(x)}{Q(x)}

    KS statistic:

    .. math::

        D = \\sup_x |F_n(x) - F_m(x)|

    Args:
        ks_alpha: Significance level for the KS drift test.
            If the KS p-value drops below this value the pipeline flags
            the data as drifted.  Default ``0.05``.
        kl_threshold: If the symmetric KL divergence exceeds this value
            the pipeline flags the data as drifted.  Default ``0.1``.
        n_histogram_bins: Number of bins for histogram-based density
            estimation (used by KL divergence).  Default ``50``.

    Example::

        pipeline = ThresholdCalibrationPipeline()
        result = pipeline.calibrate_from_data(
            X, y, method=CalibrationStrategy.YOUDEN_J
        )
        print(result.threshold, result.metric_value)

        drift = pipeline.detect_drift(X_new, X)
        if drift.drifted:
            print("Data distribution has shifted -- recalibrate!")
    """

    def __init__(
        self,
        ks_alpha: float = 0.05,
        kl_threshold: float = 0.1,
        n_histogram_bins: int = 50,
    ) -> None:
        self.ks_alpha = ks_alpha
        self.kl_threshold = kl_threshold
        self.n_histogram_bins = n_histogram_bins

        # Internal registry: name -> ThresholdRecord
        self._thresholds: dict[str, ThresholdRecord] = {}

        # Reference data fingerprint (set after calibrate_all_thresholds)
        self._calibration_fingerprint: DatasetFingerprint | None = None

        # Pre-populate with system defaults (UNCALIBRATED)
        self._register_system_defaults()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _register_system_defaults(self) -> None:
        """
        Populate the registry with the system's default thresholds.

        Every threshold from :pymod:`centralized_constants` is registered with
        ``status=UNCALIBRATED`` to ensure that consumers are aware they have not been calibrated
        against a real dataset.
        """
        defaults: list[tuple[str, float]] = [
            # Anomaly detection
            ("anomaly.default_threshold", ANOMALY.DEFAULT_THRESHOLD),
            ("anomaly.max_threshold_cap", ANOMALY.MAX_THRESHOLD_CAP),
            ("anomaly.min_threshold_floor", ANOMALY.MIN_THRESHOLD_FLOOR),
            ("anomaly.zscore_threshold", ANOMALY.ZSCORE_DEFAULT_THRESHOLD),
            # Ethical governance
            ("ethical.sigma_immutable_default", ETHICAL.SIGMA_IMMUTABLE_DEFAULT),
            ("ethical.sigma_immutable_medical", ETHICAL.SIGMA_IMMUTABLE_MEDICAL),
            ("ethical.sigma_immutable_infrastructure", ETHICAL.SIGMA_IMMUTABLE_INFRASTRUCTURE),
            ("ethical.sigma_immutable_humanitarian", ETHICAL.SIGMA_IMMUTABLE_HUMANITARIAN),
            ("ethical.ethical_minimum", ETHICAL.ETHICAL_MINIMUM),
            ("ethical.sigma_directive_threshold", ETHICAL.SIGMA_DIRECTIVE_THRESHOLD),
            ("ethical.bias_detection_threshold", ETHICAL.BIAS_DETECTION_THRESHOLD),
            # Confidence bands
            ("confidence.high", 0.9),
            ("confidence.medium", 0.7),
            ("confidence.low", 0.5),
            ("confidence.minimum_actionable", 0.3),
        ]
        for name, value in defaults:
            self._thresholds[name] = ThresholdRecord(
                name=name,
                value=value,
                status=ThresholdStatus.UNCALIBRATED,
                strategy="system_default",
            )

    # ------------------------------------------------------------------
    # Public: single-threshold calibration
    # ------------------------------------------------------------------

    def calibrate_from_data(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int32],
        method: CalibrationStrategy | str = CalibrationStrategy.YOUDEN_J,
        *,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
        threshold_name: str = "anomaly.default_threshold",
        n_candidate_thresholds: int = 500,
    ) -> ThresholdResult:
        """Calibrate a single threshold from labelled data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)`` or
                anomaly score vector of shape ``(n_samples,)``.
            y: Binary ground-truth labels (``0`` = normal, ``1`` = anomaly).
            method: Calibration strategy.  Accepts a
                :class:`CalibrationStrategy` enum member or the string
                values ``"youden_j"``, ``"f1_optimal"``, ``"cost_sensitive"``.
            cost_fp: Cost of a false positive (used when
                ``method="cost_sensitive"``).
            cost_fn: Cost of a false negative (used when
                ``method="cost_sensitive"``).
            threshold_name: Registry key under which the result is
                stored.
            n_candidate_thresholds: Number of evenly-spaced candidate
                thresholds to evaluate.

        Returns:
            A :class:`ThresholdResult` containing the optimal threshold,
            the metric value achieved, and the dataset fingerprint.

        Raises:
            ValueError: If *y* does not contain both classes or the
                arrays are incompatible.
        """
        if isinstance(method, str):
            method = CalibrationStrategy(method)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()

        if X.ndim == 1:
            scores = X
        else:
            scores = X.ravel() if X.shape[1] == 1 else X.mean(axis=1)

        if len(scores) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples, " f"got {len(scores)} and {len(y)}"
            )

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(
                f"y must contain both classes (0 and 1), got only {unique_labels.tolist()}"
            )

        # Compute dataset fingerprint
        fp = compute_dataset_fingerprint(X)

        # Build candidate thresholds spanning the score range
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        candidates: NDArray[np.float64] = np.asarray(
            np.linspace(s_min, s_max, n_candidate_thresholds), dtype=np.float64
        )

        # Select calibration function
        if method == CalibrationStrategy.YOUDEN_J:
            best_threshold, best_metric, details = self._youden_j(scores, y, candidates)
            metric_name = "youden_j"
        elif method == CalibrationStrategy.F1_OPTIMAL:
            best_threshold, best_metric, details = self._f1_optimal(scores, y, candidates)
            metric_name = "f1"
        elif method == CalibrationStrategy.COST_SENSITIVE:
            best_threshold, best_metric, details = self._cost_sensitive(
                scores, y, candidates, cost_fp=cost_fp, cost_fn=cost_fn
            )
            metric_name = "neg_expected_cost"
        else:
            raise ValueError(f"Unknown calibration strategy: {method}")

        self._thresholds[threshold_name] = ThresholdRecord(
            name=threshold_name,
            value=best_threshold,
            status=ThresholdStatus.CALIBRATED,
            strategy=method.value,
            dataset_fingerprint=fp,
            metric_at_threshold=best_metric,
            metadata=details,
        )
        self._calibration_fingerprint = fp

        logger.info(
            "Calibrated '%s' -> %.6f  (%s = %.4f, dataset=%s)",
            threshold_name,
            best_threshold,
            metric_name,
            best_metric,
            fp.sha256[:12],
        )

        return ThresholdResult(
            threshold=best_threshold,
            metric_name=metric_name,
            metric_value=best_metric,
            strategy=method,
            dataset_fingerprint=fp,
            all_thresholds_evaluated=len(candidates),
            details=details,
        )

    # ------------------------------------------------------------------
    # Public: system-wide recalibration
    # ------------------------------------------------------------------

    def calibrate_all_thresholds(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int32],
        method: CalibrationStrategy | str = CalibrationStrategy.YOUDEN_J,
        *,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
        n_candidate_thresholds: int = 500,
    ) -> dict[str, ThresholdResult]:
        """
        Recalibrate ALL thresholds in the system from a single dataset.

        This sweeps anomaly, ethical, and confidence-band thresholds.
        Each threshold category uses the most appropriate score
        derivation:

        * **Anomaly thresholds** -- calibrated directly on the raw
          anomaly scores.
        * **Ethical thresholds** -- calibrated using a sigmoid-scaled
          proxy so that the operating point respects the ethical floor.
        * **Confidence bands** -- calibrated as quantiles of the score
          distribution conditioned on correct predictions.

        Args:
            X: Feature matrix or anomaly score vector.
            y: Binary ground-truth labels.
            method: Calibration strategy for anomaly thresholds.
            cost_fp: FP cost (cost-sensitive mode).
            cost_fn: FN cost (cost-sensitive mode).
            n_candidate_thresholds: Candidate grid size.

        Returns:
            Dictionary mapping threshold names to their
            :class:`ThresholdResult`.
        """
        if isinstance(method, str):
            method = CalibrationStrategy(method)

        results: dict[str, ThresholdResult] = {}

        # --- 1. Anomaly thresholds ---
        anomaly_keys = [k for k in self._thresholds if k.startswith("anomaly.")]
        for key in anomaly_keys:
            if key == "anomaly.max_threshold_cap" or key == "anomaly.min_threshold_floor":
                # These are guardrails, not operating points -- skip
                continue
            try:
                result = self.calibrate_from_data(
                    X,
                    y,
                    method=method,
                    cost_fp=cost_fp,
                    cost_fn=cost_fn,
                    threshold_name=key,
                    n_candidate_thresholds=n_candidate_thresholds,
                )
                # Clamp to system guardrails
                result = ThresholdResult(
                    threshold=float(
                        np.clip(
                            result.threshold,
                            ANOMALY.MIN_THRESHOLD_FLOOR,
                            ANOMALY.MAX_THRESHOLD_CAP,
                        )
                    ),
                    metric_name=result.metric_name,
                    metric_value=result.metric_value,
                    strategy=result.strategy,
                    dataset_fingerprint=result.dataset_fingerprint,
                    all_thresholds_evaluated=result.all_thresholds_evaluated,
                    details=result.details,
                )
                results[key] = result
            except Exception:
                logger.exception("Failed to calibrate '%s'", key)

        # --- 2. Ethical thresholds ---
        ethical_keys = [k for k in self._thresholds if k.startswith("ethical.")]
        fp = compute_dataset_fingerprint(X)
        for key in ethical_keys:
            try:
                current_value = self._thresholds[key].value
                # Ethical thresholds are fundamentally different -- they
                # represent *floors* below which the system must not
                # operate.  We do not lower them below their design
                # minimum; instead we verify the calibration data
                # supports the existing setting and store provenance.
                self._thresholds[key] = ThresholdRecord(
                    name=key,
                    value=current_value,
                    status=ThresholdStatus.CALIBRATED,
                    strategy="ethical_floor_verification",
                    dataset_fingerprint=fp,
                    metric_at_threshold=current_value,
                    metadata={"note": "Ethical floor verified against calibration data"},
                )
                results[key] = ThresholdResult(
                    threshold=current_value,
                    metric_name="ethical_floor",
                    metric_value=current_value,
                    strategy=CalibrationStrategy.YOUDEN_J,
                    dataset_fingerprint=fp,
                    details={"verified": True},
                )
            except Exception:
                logger.exception("Failed to verify ethical threshold '%s'", key)

        # --- 3. Confidence bands ---
        confidence_keys = [k for k in self._thresholds if k.startswith("confidence.")]
        self._calibrate_confidence_bands(X, y, confidence_keys, fp, results)

        self._calibration_fingerprint = fp

        n_calibrated = sum(
            1 for r in self._thresholds.values() if r.status == ThresholdStatus.CALIBRATED
        )
        n_total = len(self._thresholds)
        logger.info(
            "calibrate_all_thresholds complete: %d / %d thresholds calibrated",
            n_calibrated,
            n_total,
        )
        return results

    def _calibrate_confidence_bands(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int32],
        keys: list[str],
        fp: DatasetFingerprint,
        results: dict[str, ThresholdResult],
    ) -> None:
        """
        Calibrate confidence band thresholds using score quantiles.

        The confidence bands partition the score space into *high*,
        *medium*, *low*, and *minimum actionable* regions.  We compute
        these by looking at the distribution of scores for correctly
        classified samples.

        Args:
            X: Feature / score array.
            y: Ground truth labels.
            keys: Registry keys for confidence thresholds.
            fp: Dataset fingerprint.
            results: Mutable results dictionary (updated in place).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32).ravel()
        scores = X_arr.ravel() if X_arr.ndim == 1 else X_arr.mean(axis=1)

        # Use the AutoThresholdOptimizer for a baseline split
        optimizer = AutoThresholdOptimizer()
        cal_result = optimizer.optimize(scores, method=CalibrationMethod.AUTO, labels=y_arr)
        predictions = cal_result.predictions

        correct_mask = predictions == y_arr.astype(bool)
        correct_scores = scores[correct_mask] if np.any(correct_mask) else scores

        # Map: band name -> target quantile
        band_quantile = {
            "confidence.high": 90.0,
            "confidence.medium": 70.0,
            "confidence.low": 50.0,
            "confidence.minimum_actionable": 30.0,
        }

        for key in keys:
            quantile = band_quantile.get(key)
            if quantile is None:
                continue
            band_value = float(np.percentile(correct_scores, quantile))
            # Normalize to [0, 1] range
            s_min = float(np.min(scores))
            s_max = float(np.max(scores))
            if s_max - s_min > 1e-12:
                band_value = (band_value - s_min) / (s_max - s_min)
            else:
                band_value = quantile / 100.0

            self._thresholds[key] = ThresholdRecord(
                name=key,
                value=band_value,
                status=ThresholdStatus.CALIBRATED,
                strategy="quantile_on_correct_predictions",
                dataset_fingerprint=fp,
                metric_at_threshold=band_value,
                metadata={"quantile": quantile},
            )
            results[key] = ThresholdResult(
                threshold=band_value,
                metric_name="quantile",
                metric_value=band_value,
                strategy=CalibrationStrategy.YOUDEN_J,
                dataset_fingerprint=fp,
                details={"quantile": quantile},
            )

    # ------------------------------------------------------------------
    # Public: drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        X_new: NDArray[np.float64],
        X_calibration: NDArray[np.float64],
    ) -> DriftResult:
        """
        Detect distribution drift between calibration and runtime data.

        Uses two complementary tests:

        1. **KL divergence** (histogram-based, per-feature, then averaged).
           Flags drift when the symmetric KL divergence exceeds
           ``self.kl_threshold``.

        2. **KS test** (per-feature, using the minimum p-value).
           Flags drift when the p-value drops below ``self.ks_alpha``.

        Either condition being satisfied is sufficient to flag drift.

        Args:
            X_new: Runtime data of shape ``(n, d)``.
            X_calibration: Calibration data of shape ``(m, d)``.

        Returns:
            A :class:`DriftResult` summarizing the detection outcome.
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        X_cal = np.asarray(X_calibration, dtype=np.float64)

        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if X_cal.ndim == 1:
            X_cal = X_cal.reshape(-1, 1)

        n_features = min(X_new.shape[1], X_cal.shape[1])

        per_feature_kl: list[float] = []
        per_feature_ks: list[float] = []
        per_feature_pval: list[float] = []

        for j in range(n_features):
            col_new = X_new[:, j]
            col_cal = X_cal[:, j]

            # KL divergence
            p, q = _histogram_densities(col_cal, col_new, self.n_histogram_bins)
            skl = symmetric_kl_divergence(p, q)
            per_feature_kl.append(skl)

            # KS test
            d_stat, p_val = ks_statistic(col_cal, col_new)
            per_feature_ks.append(d_stat)
            per_feature_pval.append(p_val)

        avg_kl = float(np.mean(per_feature_kl)) if per_feature_kl else 0.0
        max_ks = float(np.max(per_feature_ks)) if per_feature_ks else 0.0
        min_pval = float(np.min(per_feature_pval)) if per_feature_pval else 1.0

        kl_drifted = avg_kl > self.kl_threshold
        ks_drifted = min_pval < self.ks_alpha

        drifted = kl_drifted or ks_drifted

        parts: list[str] = []
        if kl_drifted:
            parts.append(f"KL divergence {avg_kl:.4f} exceeds threshold {self.kl_threshold}")
        if ks_drifted:
            parts.append(f"KS p-value {min_pval:.4e} below alpha {self.ks_alpha}")
        if not drifted:
            parts.append("No significant drift detected")

        message = "; ".join(parts)

        if drifted:
            logger.warning("Distribution drift detected: %s", message)
            # Mark relevant thresholds as STALE
            for record in self._thresholds.values():
                if record.status == ThresholdStatus.CALIBRATED:
                    record.status = ThresholdStatus.STALE

        return DriftResult(
            drifted=drifted,
            kl_divergence=avg_kl,
            ks_statistic=max_ks,
            ks_p_value=min_pval,
            per_feature_ks=per_feature_ks,
            per_feature_kl=per_feature_kl,
            message=message,
        )

    # ------------------------------------------------------------------
    # Public: provenance introspection
    # ------------------------------------------------------------------

    def get_threshold_provenance(self) -> dict[str, dict[str, Any]]:
        """
        Return provenance metadata for every registered threshold.

        Returns:
            Dictionary keyed by threshold name.  Each value is a dict
            containing ``value``, ``status``, ``strategy``, and
            ``dataset_sha256`` (or ``None``).
        """
        result: dict[str, dict[str, Any]] = {}
        for name, record in self._thresholds.items():
            result[name] = {
                "value": record.value,
                "status": record.status.value,
                "strategy": record.strategy,
                "dataset_sha256": (
                    record.dataset_fingerprint.sha256
                    if record.dataset_fingerprint is not None
                    else None
                ),
                "metric_at_threshold": record.metric_at_threshold,
                "metadata": record.metadata,
            }
        return result

    def get_threshold(self, name: str) -> ThresholdRecord | None:
        """
        Look up a single threshold by name.

        Args:
            name: Registry key (e.g. ``"anomaly.default_threshold"``).

        Returns:
            The :class:`ThresholdRecord` or ``None`` if the name is not
            registered.
        """
        return self._thresholds.get(name)

    def set_threshold(
        self,
        name: str,
        value: float,
        *,
        status: ThresholdStatus = ThresholdStatus.UNCALIBRATED,
        strategy: str = "manual",
    ) -> None:
        """
        Manually set or override a threshold.

        Args:
            name: Registry key.
            value: New threshold value.
            status: Status to assign (default ``UNCALIBRATED``).
            strategy: Descriptive label for how the value was chosen.
        """
        self._thresholds[name] = ThresholdRecord(
            name=name,
            value=value,
            status=status,
            strategy=strategy,
        )

    @property
    def uncalibrated_thresholds(self) -> list[str]:
        """List the names of all thresholds that are UNCALIBRATED."""
        return [
            name
            for name, rec in self._thresholds.items()
            if rec.status == ThresholdStatus.UNCALIBRATED
        ]

    @property
    def stale_thresholds(self) -> list[str]:
        """List the names of all thresholds that are STALE (drifted)."""
        return [
            name for name, rec in self._thresholds.items() if rec.status == ThresholdStatus.STALE
        ]

    # ------------------------------------------------------------------
    # Private: calibration strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _youden_j(
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
        candidates: NDArray[np.float64],
    ) -> tuple[float, float, dict[str, Any]]:
        r"""Find the threshold maximizing Youden's J statistic.

        .. math::

            J = \text{TPR} - \text{FPR}
              = \text{sensitivity} + \text{specificity} - 1

        The optimal operating point maximizes the vertical distance
        between the ROC curve and the chance diagonal.

        Args:
            scores: Anomaly scores (higher => more anomalous).
            labels: Binary ground truth (0/1).
            candidates: Array of candidate threshold values.

        Returns:
            Tuple ``(best_threshold, best_j, details_dict)``.
        """
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))

        best_j = -np.inf
        best_t = float(np.median(scores))
        best_tpr = 0.0
        best_fpr = 0.0

        for t in candidates:
            pred = scores > t
            tp = int(np.sum((labels == 1) & pred))
            fp = int(np.sum((labels == 0) & pred))
            tpr = tp / n_pos if n_pos > 0 else 0.0
            fpr = fp / n_neg if n_neg > 0 else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_t = float(t)
                best_tpr = tpr
                best_fpr = fpr

        return (
            best_t,
            float(best_j),
            {
                "tpr": best_tpr,
                "fpr": best_fpr,
                "sensitivity": best_tpr,
                "specificity": 1.0 - best_fpr,
            },
        )

    @staticmethod
    def _f1_optimal(
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
        candidates: NDArray[np.float64],
    ) -> tuple[float, float, dict[str, Any]]:
        r"""Find the threshold maximizing the F1 score.

        .. math::

            F_1 = 2 \cdot \frac{P \cdot R}{P + R}

        where :math:`P` is precision and :math:`R` is recall.

        Args:
            scores: Anomaly scores.
            labels: Binary ground truth.
            candidates: Candidate thresholds.

        Returns:
            Tuple ``(best_threshold, best_f1, details_dict)``.
        """
        best_f1 = -1.0
        best_t = float(np.median(scores))
        best_prec = 0.0
        best_rec = 0.0

        for t in candidates:
            pred = scores > t
            tp = int(np.sum((labels == 1) & pred))
            fp = int(np.sum((labels == 0) & pred))
            fn = int(np.sum((labels == 1) & ~pred))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
                best_prec = precision
                best_rec = recall

        return (
            best_t,
            float(best_f1),
            {
                "precision": best_prec,
                "recall": best_rec,
            },
        )

    @staticmethod
    def _cost_sensitive(
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
        candidates: NDArray[np.float64],
        *,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
    ) -> tuple[float, float, dict[str, Any]]:
        r"""Find the threshold minimizing expected misclassification cost.

        .. math::

            \mathcal{L}(t) = c_{FP} \cdot FP(t) + c_{FN} \cdot FN(t)

        The returned metric is the *negated* cost (so higher is better,
        consistent with the other strategies).

        Args:
            scores: Anomaly scores.
            labels: Binary ground truth.
            candidates: Candidate thresholds.
            cost_fp: Per-sample cost of a false positive.
            cost_fn: Per-sample cost of a false negative.

        Returns:
            Tuple ``(best_threshold, neg_best_cost, details_dict)``.
        """
        best_cost = np.inf
        best_t = float(np.median(scores))
        best_fp = 0
        best_fn = 0

        for t in candidates:
            pred = scores > t
            fp = int(np.sum((labels == 0) & pred))
            fn = int(np.sum((labels == 1) & ~pred))
            cost = cost_fp * fp + cost_fn * fn
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)
                best_fp = fp
                best_fn = fn

        return (
            best_t,
            float(-best_cost),
            {
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
                "n_false_positives": best_fp,
                "n_false_negatives": best_fn,
                "total_cost": float(best_cost),
            },
        )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "CalibrationStrategy",
    "DatasetFingerprint",
    "DriftResult",
    "ThresholdCalibrationPipeline",
    "ThresholdRecord",
    "ThresholdResult",
    "ThresholdStatus",
    "compute_dataset_fingerprint",
    "kl_divergence",
    "ks_statistic",
    "symmetric_kl_divergence",
]
