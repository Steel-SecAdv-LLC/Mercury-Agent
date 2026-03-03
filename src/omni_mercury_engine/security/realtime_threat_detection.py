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

import warnings

warnings.warn(
    f"{__name__} is deprecated. Use MercuryAnomalyDetector.",
    DeprecationWarning,
    stacklevel=2,
)

"""Real-time threat detection using Mercury-native anomaly detectors.

Original: Real-Time Threat Detection with PyOD-Compatible Anomaly Detection

Implements real-time threat detection using ensemble anomaly detection methods
compatible with PyOD (Python Outlier Detection) framework.

Reference: Zhao et al., "PyOD: A Python Toolbox for Scalable Outlier Detection" (2019)
https://github.com/yzhao062/pyod

MIT-compatible implementation using Mercury-native ML primitives and numpy.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

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


class RealTimeThreatDetector(LoggerMixin):
    """
    Real-time threat detection using Mercury-native anomaly detection.

    Uses MercuryAnomalyDetector (ResonanceScore, KinematicScore,
    InfoGeometryScore probe ensemble) for all detection.
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
            n_estimators: Unused, kept for API compatibility.
            enable_isolation_forest: Unused, kept for API compatibility.
            enable_lof: Unused, kept for API compatibility.
            enable_elliptic: Unused, kept for API compatibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators

        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        self._mercury_detector = MercuryAnomalyDetector()
        self.detectors: dict[str, Any] = {}

        self.is_fitted = False
        self.threat_history: list[ThreatSignature] = []

    def fit(self, X: np.ndarray[Any, Any]) -> RealTimeThreatDetector:
        """
        Fit detector on normal (non-threatening) data.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            Self
        """
        self._mercury_detector.fit(X)
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

        result = self._mercury_detector.detect(X)
        anomaly_scores = np.asarray(result["scores"])
        is_anomaly = np.asarray(result["is_anomaly"])

        is_threat = is_anomaly.astype(bool)
        threat_indices = np.where(is_threat)[0]
        threat_level = self._calculate_threat_level(-anomaly_scores)

        return {
            "is_threat": bool(np.any(is_threat)),
            "threat_indices": threat_indices.tolist(),
            "ensemble_scores": (-anomaly_scores).tolist(),
            "individual_predictions": {},
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

        try:
            self._mercury_detector.fit(X_new)
        except Exception as e:
            self.logger.warning("Failed to update Mercury detector: %s", e)
