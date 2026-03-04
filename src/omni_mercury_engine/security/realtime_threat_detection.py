"""
Mercury Agent
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

"""Real-Time Threat Detection using Mercury-native anomaly detection.

Implements real-time threat detection using Mercury's own ensemble anomaly
detection methods.  All detection is performed with numpy/scipy — no sklearn.

Detection ensemble:
  - Isolation-style random-projection detector (tree-free)
  - Local density estimator (scipy.spatial.cKDTree)
  - Robust covariance (Mahalanobis distance)
"""

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
        self.contamination = contamination
        self.n_projections = n_projections
        self._rng = np.random.default_rng(random_state)
        self._projections: np.ndarray | None = None
        self._medians: np.ndarray | None = None
        self._mads: np.ndarray | None = None

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
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self._tree: cKDTree | None = None

    def fit(self, X: np.ndarray[Any, Any]) -> None:
        self._tree = cKDTree(X)

    def score_samples(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self._tree is not None
        k = min(self.n_neighbors, self._tree.n)
        dists, _ = self._tree.query(X, k=max(k, 1))
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
        self.contamination = contamination
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None

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
    """
    Real-time threat detection using Mercury-native ensemble anomaly detection.

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
        """
        Initialize real-time threat detector.

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

    def fit(self, X: np.ndarray[Any, Any]) -> RealTimeThreatDetector:
        """
        Fit detectors on normal (non-threatening) data.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            Self
        """
        for name, detector in self.detectors.items():
            try:
                detector.fit(X)
            except Exception as e:
                self.logger.warning("Failed to fit %s: %s", name, e)

        self.is_fitted = True
        return self

    def detect_threat(self, X: np.ndarray[Any, Any]) -> dict[str, Any]:
        """
        Detect threats in real-time data.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Dictionary with threat detection results
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")

        predictions: dict[str, Any] = {}
        scores: dict[str, Any] = {}

        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, "predict"):
                    pred = detector.predict(X)
                    predictions[name] = pred

                if hasattr(detector, "score_samples"):
                    score = detector.score_samples(X)
                    scores[name] = score
                elif hasattr(detector, "decision_function"):
                    score = detector.decision_function(X)
                    scores[name] = score
            except Exception as e:
                self.logger.warning("Failed to predict with %s: %s", name, e)

        if not scores:
            return {
                "is_threat": False,
                "threat_indices": [],
                "ensemble_scores": [],
                "individual_predictions": {},
                "threat_level": "LOW",
                "num_threats": 0,
                "timestamp": datetime.now().isoformat(),
            }

        ensemble_score = np.mean(list(scores.values()), axis=0)

        is_threat = ensemble_score < np.percentile(ensemble_score, self.contamination * 100)

        threat_indices = np.where(is_threat)[0]

        threat_level = self._calculate_threat_level(ensemble_score)

        return {
            "is_threat": bool(np.any(is_threat)),
            "threat_indices": threat_indices.tolist(),
            "ensemble_scores": ensemble_score.tolist(),
            "individual_predictions": {k: v.tolist() for k, v in predictions.items()},
            "threat_level": threat_level,
            "num_threats": int(np.sum(is_threat)),
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_threat_level(self, scores: np.ndarray[Any, Any]) -> str:
        """Calculate threat level based on scores."""
        min_score = np.min(scores)

        if min_score < np.percentile(scores, 1):
            return "CRITICAL"
        elif min_score < np.percentile(scores, 5):
            return "HIGH"
        elif min_score < np.percentile(scores, 10):
            return "MEDIUM"
        else:
            return "LOW"

    def record_threat(
        self, threat_data: np.ndarray[Any, Any], threat_type: str, severity: float
    ) -> ThreatSignature:
        """
        Record detected threat for future analysis.

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
    """
    Adaptive threat detector that updates based on new threats.

    Implements online learning for continuous adaptation to evolving threats.
    All detection is Mercury-native (numpy/scipy only).
    """

    def __init__(self, *args: Any, update_frequency: int = 100, **kwargs: Any) -> None:
        """
        Initialize adaptive threat detector.

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
        """
        Detect threats and adapt model based on feedback.

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
