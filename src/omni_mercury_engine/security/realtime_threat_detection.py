# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-Time Threat Detection using Mercury-native anomaly detection.

Implements real-time threat detection using Mercury's own ensemble anomaly
detection methods.  All detection is performed with numpy/scipy — no sklearn.

Detection ensemble:
  - Isolation-style random-projection detector (tree-free)
  - Local density estimator (scipy.spatial.cKDTree)
  - Robust covariance (Mahalanobis distance)

The sub-detector scores are combined *scale-invariantly*: each detector's
training scores are robustly standardized (median / ``1.4826·MAD``) before the
ensemble mean, so a detector whose raw scores are large (kNN distances) cannot
drown out one whose scores are small (robust z-scores). The decision threshold
and per-sample anomaly probability are calibrated against the training
distribution (an absolute, training-referenced quantile + empirical CDF), so a
single sample can be judged, an all-normal batch stays near the configured
false-positive budget, and an all-anomalous batch is flagged in full — none of
which a batch-relative percentile threshold can do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from omni_mercury_engine.utils.logging import LoggerMixin

logger = logging.getLogger(__name__)


@dataclass
class ThreatSignature:
    """Threat signature with metadata."""

    threat_id: str
    feature_vector: np.ndarray[Any, Any]
    threat_type: str
    severity: float
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.95


# ---------------------------------------------------------------------------
# Mercury-native detector components (no sklearn)
# ---------------------------------------------------------------------------


class _RandomProjectionDetector:
    """Isolation-style anomaly detector using random projections (no trees)."""

    def __init__(
        self, contamination: float = 0.1, n_projections: int = 100, random_state: int = 42
    ) -> None:
        """Initialize the instance."""
        self.contamination = contamination
        self.n_projections = n_projections
        self._rng = np.random.default_rng(random_state)
        self._projections: np.ndarray[Any, Any] | None = None
        self._medians: np.ndarray[Any, Any] | None = None
        self._mads: np.ndarray[Any, Any] | None = None

    def fit(self, X: np.ndarray[Any, Any]) -> None:
        n_features = X.shape[1]
        self._projections = self._rng.standard_normal((self.n_projections, n_features))
        norms = np.linalg.norm(self._projections, axis=1, keepdims=True)
        self._projections /= np.where(norms > 1e-10, norms, 1.0)
        projected = X @ self._projections.T  # (n_samples, n_projections)
        self._medians = np.median(projected, axis=0)
        self._mads = np.median(np.abs(projected - self._medians), axis=0)
        self._mads = np.where(self._mads > 1e-10, self._mads, 1.0)

    def score_samples(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self._projections is not None
        projected = X @ self._projections.T
        z = np.abs(projected - self._medians) / self._mads
        result: np.ndarray[Any, Any] = np.mean(z, axis=1)
        return result

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores >= threshold, -1, 1)


class _LocalDensityDetector:
    """KDTree-based local density anomaly detector (LOF-style, no sklearn)."""

    def __init__(self, contamination: float = 0.1, n_neighbors: int = 20) -> None:
        """Initialize the instance."""
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self._tree: cKDTree | None = None

    def fit(self, X: np.ndarray[Any, Any]) -> None:
        self._tree = cKDTree(X)

    def score_samples(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self._tree is not None
        k = min(self.n_neighbors, self._tree.n)
        dists, _ = self._tree.query(X, k=max(k, 1))
        if dists.ndim == 1:
            dists = dists[:, np.newaxis]
        result: np.ndarray[Any, Any] = np.mean(dists, axis=1)
        return result

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores >= threshold, -1, 1)

    def decision_function(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return -self.score_samples(X)


class _RobustCovarianceDetector:
    """Mahalanobis-distance detector with robust covariance (no sklearn)."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42) -> None:
        """Initialize the instance."""
        self.contamination = contamination
        self._mean: np.ndarray[Any, Any] | None = None
        self._cov_inv: np.ndarray[Any, Any] | None = None

    def fit(self, X: np.ndarray[Any, Any]) -> None:
        n_samples, n_features = X.shape
        median = np.median(X, axis=0)
        dists = np.sqrt(np.sum((X - median) ** 2, axis=1))
        n_support = max(int(n_samples * 0.9), n_features + 1)
        idx = np.argsort(dists)[:n_support]
        X_s = X[idx]
        self._mean = np.mean(X_s, axis=0)
        centered = X_s - self._mean
        cov = centered.T @ centered / max(len(X_s) - 1, 1)
        cov += 1e-6 * np.eye(n_features)
        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self._cov_inv = np.linalg.pinv(cov)

    def score_samples(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self._mean is not None and self._cov_inv is not None
        centered = X - self._mean
        left = centered @ self._cov_inv
        result: np.ndarray[Any, Any] = np.sqrt(np.maximum(np.sum(left * centered, axis=1), 0.0))
        return result

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores >= threshold, -1, 1)

    def decision_function(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return -self.score_samples(X)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RealTimeThreatDetector(LoggerMixin):
    """Real-time threat detection using Mercury-native ensemble anomaly detection.

    Combines multiple Mercury-native detection algorithms:
    - Random-projection isolation detector
    - Local density estimator (cKDTree)
    - Robust covariance (Mahalanobis)

    All detection is numpy/scipy only — zero sklearn dependency.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        enable_isolation_forest: bool = True,
        enable_lof: bool = True,
        enable_elliptic: bool = True,
    ):
        """Initialize real-time threat detector.

        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
            n_estimators: Number of random projections for isolation detector
            enable_isolation_forest: Enable random-projection detector
            enable_lof: Enable local density detector
            enable_elliptic: Enable robust covariance detector
        """
        self.contamination = contamination
        self.n_estimators = n_estimators

        self.detectors: dict[str, Any] = {}

        if enable_isolation_forest:
            self.detectors["isolation_forest"] = _RandomProjectionDetector(
                contamination=contamination,
                n_projections=n_estimators,
            )

        if enable_lof:
            self.detectors["lof"] = _LocalDensityDetector(
                contamination=contamination,
            )

        if enable_elliptic:
            self.detectors["elliptic"] = _RobustCovarianceDetector(
                contamination=contamination,
            )

        self.is_fitted = False
        self.threat_history: list[ThreatSignature] = []
        # Per-detector robust standardization stats from training (median /
        # 1.4826·MAD) so heterogeneous score scales are made commensurate
        # before they are combined -- see ``fit``/``_standardized_ensemble``.
        self._detector_center: dict[str, float] = {}
        self._detector_scale: dict[str, float] = {}
        # Reference distribution of the *standardized* ensemble score on
        # training data: an absolute (training-referenced) decision threshold,
        # the sorted vector for an empirical-CDF anomaly probability, and the
        # p90/95/99 quantiles that grade the threat level.
        self._ref_threshold: float = 0.0
        self._ref_ensemble_sorted: np.ndarray[Any, Any] = np.zeros(0)
        self._ref_p90: float = 0.0
        self._ref_p95: float = 0.0
        self._ref_p99: float = 0.0

    _MAD_TO_SIGMA = 1.4826  # MAD → σ for a normal distribution.

    def fit(self, X: np.ndarray[Any, Any]) -> RealTimeThreatDetector:
        """Fit detectors on normal (non-threatening) data.

        Beyond fitting each sub-detector, this learns the calibration the
        ensemble needs to be *scale-invariant* and to threshold *absolutely*:

        * Per-detector robust location/scale (median, ``1.4826·MAD``) of each
          detector's training scores, so a detector whose raw scores happen to
          be large (e.g. kNN Euclidean distances) can no longer dominate the
          average over one whose scores are small (e.g. robust z-scores).
        * The distribution of the standardized ensemble score on training data,
          which yields an absolute decision threshold at the configured
          contamination and an empirical CDF for a calibrated per-sample
          anomaly probability. The previous implementation thresholded against
          the *current inference batch*, which cannot flag a single sample,
          always flags ``contamination`` fraction of an all-normal batch, and
          under-flags an all-anomalous batch.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            Self
        """
        fitted_count = 0
        for name, detector in self.detectors.items():
            try:
                detector.fit(X)
                fitted_count += 1
            except Exception as e:
                self.logger.warning("Failed to fit %s: %s", name, e)

        if fitted_count == 0:
            # Fail closed: entering a fitted state with zero working detectors
            # would make detect_threat() report "no threat / LOW" for every
            # input — a security detector silently blind. Refuse instead.
            raise RuntimeError(
                f"RealTimeThreatDetector.fit: all {len(self.detectors)} sub-detectors "
                "failed to fit; refusing to enter a fitted state that would report "
                "'no threat' for every input. Check the training data shape/dtype "
                "and detector dependencies."
            )

        # Per-detector robust standardization stats from the training scores.
        raw_ref: dict[str, np.ndarray[Any, Any]] = {}
        for name, detector in self.detectors.items():
            try:
                s = self._raw_score(detector, X)
            except Exception:
                continue
            if s is None:
                continue
            raw_ref[name] = s
            center = float(np.median(s))
            mad = float(np.median(np.abs(s - center)))
            scale = self._MAD_TO_SIGMA * mad
            if scale <= 1e-12:  # near-constant training score → std fallback
                scale = float(np.std(s)) or 1.0
            self._detector_center[name] = center
            self._detector_scale[name] = scale

        # Reference distribution of the standardized ensemble on training data.
        if raw_ref:
            ref_ensemble = self._combine_standardized(
                {name: self._standardize(name, s) for name, s in raw_ref.items()}
            )
            self._ref_ensemble_sorted = np.sort(ref_ensemble)
            self._ref_threshold = float(np.quantile(ref_ensemble, 1.0 - self.contamination))
            self._ref_p90 = float(np.quantile(ref_ensemble, 0.90))
            self._ref_p95 = float(np.quantile(ref_ensemble, 0.95))
            self._ref_p99 = float(np.quantile(ref_ensemble, 0.99))

        self.is_fitted = True
        return self

    @staticmethod
    def _raw_score(detector: Any, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
        """A detector's per-sample anomaly score (higher = more anomalous)."""
        if hasattr(detector, "score_samples"):
            return np.asarray(detector.score_samples(X), dtype=float)
        if hasattr(detector, "decision_function"):
            # decision_function is oriented lower = more anomalous; flip it.
            return -np.asarray(detector.decision_function(X), dtype=float)
        return None

    def _standardize(self, name: str, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Robust z-standardize one detector's scores with its training stats."""
        center = self._detector_center.get(name, 0.0)
        scale = self._detector_scale.get(name, 1.0)
        return (scores - center) / scale

    @staticmethod
    def _combine_standardized(
        standardized: dict[str, np.ndarray[Any, Any]],
    ) -> np.ndarray[Any, Any]:
        """Mean of the per-detector standardized scores (scale-invariant)."""
        return np.asarray(np.mean(np.vstack(list(standardized.values())), axis=0))

    def detect_threat(self, X: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect threats in real-time data.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Dictionary with per-sample results. In addition to the historical
            keys it exposes ``anomaly_probabilities`` (calibrated [0, 1] via the
            training empirical CDF) and ``detector_agreement`` (fraction of
            sub-detectors flagging each sample) for downstream analysis.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")

        predictions: dict[str, Any] = {}
        standardized: dict[str, np.ndarray[Any, Any]] = {}

        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, "predict"):
                    predictions[name] = detector.predict(X)
                raw = self._raw_score(detector, X)
                if raw is not None:
                    standardized[name] = self._standardize(name, raw)
            except Exception as e:
                self.logger.warning("Failed to predict with %s: %s", name, e)

        if not standardized:
            # Fitted, but every sub-detector errored at inference. This is a
            # detection FAILURE, not a "no threat" result — returning LOW here
            # would let a blind detector assert safety. Fail closed.
            raise RuntimeError(
                "RealTimeThreatDetection failed: all sub-detectors errored during "
                "prediction, so threat cannot be assessed. This is NOT a 'no threat' "
                "result — treat it as a detector outage."
            )

        # Scale-invariant ensemble of standardized scores.
        ensemble_score = self._combine_standardized(standardized)

        # Absolute, training-referenced decision (works for a single sample and
        # gives correct false-positive control on all-normal / all-anomalous
        # batches, unlike a batch-relative percentile).
        is_threat = ensemble_score > self._ref_threshold
        anomaly_prob = self._anomaly_probability(ensemble_score)

        # Per-sample detector agreement: fraction of sub-detectors flagging it.
        agreement = self._detector_agreement(predictions, len(ensemble_score))

        threat_indices = np.where(is_threat)[0]
        threat_level = self._calculate_threat_level(ensemble_score)

        return {
            "is_threat": bool(np.any(is_threat)),
            "threat_indices": threat_indices.tolist(),
            "ensemble_scores": ensemble_score.tolist(),
            "anomaly_probabilities": anomaly_prob.tolist(),
            "detector_agreement": agreement.tolist(),
            "individual_predictions": {k: v.tolist() for k, v in predictions.items()},
            "threat_level": threat_level,
            "num_threats": int(np.sum(is_threat)),
            "timestamp": datetime.now().isoformat(),
        }

    def _anomaly_probability(self, ensemble_score: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Calibrated [0, 1] anomaly probability via the training empirical CDF.

        ``P(x)`` is the fraction of training ensemble scores at or below ``x``:
        a normal-looking sample lands near the bulk (low P), an outlier lands in
        the right tail (P → 1). Distribution-free — no Gaussian assumption.
        """
        ref = self._ref_ensemble_sorted
        if ref.size == 0:
            return np.zeros(len(ensemble_score))
        ranks = np.searchsorted(ref, ensemble_score, side="right")
        return ranks.astype(float) / float(ref.size)

    @staticmethod
    def _detector_agreement(
        predictions: dict[str, np.ndarray[Any, Any]], n: int
    ) -> np.ndarray[Any, Any]:
        """Fraction of sub-detectors that flag each sample as an outlier (-1)."""
        if not predictions:
            return np.zeros(n)
        flags = np.vstack([np.asarray(p) == -1 for p in predictions.values()])
        return np.asarray(flags.mean(axis=0), dtype=float)

    def _calculate_threat_level(self, scores: np.ndarray[Any, Any]) -> str:
        """Grade the batch by its peak standardized ensemble score vs training."""
        max_score = float(np.max(scores))

        if max_score > self._ref_p99:
            return "CRITICAL"
        elif max_score > self._ref_p95:
            return "HIGH"
        elif max_score > self._ref_p90:
            return "MEDIUM"
        else:
            return "LOW"

    def record_threat(
        self, threat_data: np.ndarray[Any, Any], threat_type: str, severity: float
    ) -> ThreatSignature:
        """Record detected threat for future analysis.

        Args:
            threat_data: Threat feature vector
            threat_type: Type of threat (e.g., 'ddos', 'intrusion', 'malware')
            severity: Threat severity (0.0 to 1.0)

        Returns:
            ThreatSignature object
        """
        threat_id = f"threat_{datetime.now().timestamp()}"

        signature = ThreatSignature(
            threat_id=threat_id,
            feature_vector=threat_data,
            threat_type=threat_type,
            severity=severity,
        )

        self.threat_history.append(signature)

        if len(self.threat_history) > 10000:
            self.threat_history = self.threat_history[-10000:]

        return signature

    def get_threat_statistics(self) -> dict[str, Any]:
        """Get statistics about detected threats."""
        if not self.threat_history:
            return {"total_threats": 0, "threat_types": {}, "avg_severity": 0.0}

        threat_types: dict[str, int] = {}
        for threat in self.threat_history:
            threat_types[threat.threat_type] = threat_types.get(threat.threat_type, 0) + 1

        avg_severity = np.mean([t.severity for t in self.threat_history])

        return {
            "total_threats": len(self.threat_history),
            "threat_types": threat_types,
            "avg_severity": float(avg_severity),
            "most_recent": (
                self.threat_history[-1].timestamp.isoformat() if self.threat_history else None
            ),
        }


class AdaptiveThreatDetector(RealTimeThreatDetector):
    """Adaptive threat detector that updates based on new threats.

    Implements online learning for continuous adaptation to evolving threats. All detection is
    Mercury-native (numpy/scipy only).
    """

    def __init__(self, *args: Any, update_frequency: int = 100, **kwargs: Any) -> None:
        """Initialize adaptive threat detector.

        Args:
            update_frequency: Number of samples between model updates
            *args, **kwargs: Arguments for RealTimeThreatDetector
        """
        super().__init__(*args, **kwargs)
        self.update_frequency = update_frequency
        self.samples_since_update = 0
        self.training_buffer: list[np.ndarray[Any, Any]] = []
        self.max_buffer_size = 1000

    def detect_and_adapt(
        self, X: np.ndarray[Any, Any], is_normal: bool | None = None
    ) -> dict[str, Any]:
        """Detect threats and adapt model based on feedback.

        Args:
            X: Input data
            is_normal: Optional feedback on whether data is normal

        Returns:
            Detection results
        """
        result = self.detect_threat(X)

        if is_normal is not None and is_normal:
            self.training_buffer.append(X)

            if len(self.training_buffer) > self.max_buffer_size:
                self.training_buffer = self.training_buffer[-self.max_buffer_size :]

            self.samples_since_update += len(X)

            if self.samples_since_update >= self.update_frequency:
                self._update_models()
                self.samples_since_update = 0

        return result

    def _update_models(self) -> None:
        """Update models with new training data."""
        if not self.training_buffer:
            return

        X_new = np.vstack(self.training_buffer)

        for name, detector in self.detectors.items():
            try:
                detector.fit(X_new)
            except Exception as e:
                self.logger.warning("Failed to update %s: %s", name, e)
