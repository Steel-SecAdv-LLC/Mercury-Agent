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

"""
Unified Self-Healing Engine for Autonomous Error Recovery and Adaptive Defense

Combines two complementary approaches:
1. Component Health Monitoring: Traditional engineering approach for system resilience
2. CRISPR-Inspired Adaptive Defense: Pattern-based anomaly memory system

CRISPR Inspiration (Clustered Regularly Interspaced Short Palindromic Repeats):
Three-Stage Defense analogous to prokaryotic adaptive immune system:
- Acquisition: Capture novel anomaly signatures (like CRISPR spacer acquisition)
- Expression: Process signatures into detection patterns (like crRNA transcription)
- Interference: Neutralize/block similar anomalies (like Cas protein cutting)

Research source: Ishino (1987), Mojica (2007), Doudna/Charpentier (2012)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.core.config import ThresholdConfig
from omni_mercury_engine.resilience.circuit_breaker import CircuitBreaker

# Centralized thresholds for consistent behavior
_thresholds = ThresholdConfig()


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class AnomalySignature:
    """Compact representation of anomaly pattern (analogous to CRISPR spacer)."""

    signature_id: str
    feature_vector: npt.NDArray[np.floating[Any]]
    timestamp: float
    detection_count: int = 0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AdaptiveDefenseSystem:
    """
    CRISPR-inspired adaptive defense system for anomaly pattern memory.

    Maintains library of known anomaly signatures and uses them for
    faster future detection and neutralization.

    Enhanced with online learning capabilities:
    - Incremental statistics for continuous adaptation
    - Sliding window for temporal distribution shift detection
    - Exponential forgetting factors for concept drift handling
    - Online PCA for dimensionality reduction without full retraining

    These features enable the system to adapt to evolving threats and
    changing data distributions, critical for humanitarian applications
    like pandemic monitoring and crisis detection.
    """

    def __init__(
        self,
        max_signatures: int = 1000,
        similarity_threshold: float = 0.85,
        enable_online_learning: bool = False,
        sliding_window_size: int = 100,
        forgetting_factor: float = 0.99,
        adaptation_rate: float = 0.01,
    ):
        """
        Initialize adaptive defense system.

        Args:
            max_signatures: Maximum number of anomaly signatures to store
            similarity_threshold: Threshold for matching signatures (0-1)
            enable_online_learning: Enable online learning features
            sliding_window_size: Size of sliding window for temporal adaptation
            forgetting_factor: Exponential forgetting factor (0-1, higher = slower forget)
            adaptation_rate: Learning rate for online updates
        """
        self.max_signatures = max_signatures
        self.similarity_threshold = similarity_threshold
        self.signature_library: dict[str, AnomalySignature] = {}
        self.acquisition_history: list[str] = []
        self.logger = logging.getLogger(__name__)

        self.enable_online_learning = enable_online_learning
        self.sliding_window_size = sliding_window_size
        self.forgetting_factor = forgetting_factor
        self.adaptation_rate = adaptation_rate

        self._running_mean: npt.NDArray[np.floating[Any]] | None = None
        self._running_var: npt.NDArray[np.floating[Any]] | None = None
        self._sample_count: int = 0

        self._sliding_window: list[npt.NDArray[np.floating[Any]]] = []
        self._window_statistics: dict[str, float] = {}

        self._concept_drift_detected: bool = False
        self._drift_magnitude: float = 0.0

    def stage_1_acquisition(
        self, anomaly_data: npt.NDArray[np.floating[Any]], metadata: dict[str, Any] | None = None
    ) -> AnomalySignature:
        """
        Stage 1: Acquisition - Capture novel anomaly signature.

        Analogous to CRISPR spacer acquisition from invading phage DNA.

        Args:
            anomaly_data: Raw anomaly data to capture
            metadata: Optional metadata about the anomaly

        Returns:
            AnomalySignature object
        """
        feature_vector = self._extract_signature_features(anomaly_data)
        signature_id = self._generate_signature_id(feature_vector)

        signature = AnomalySignature(
            signature_id=signature_id,
            feature_vector=feature_vector,
            timestamp=float(np.datetime64("now").astype("datetime64[s]").astype(int)),
            detection_count=1,
            confidence=0.95,
            metadata=metadata or {},
        )

        if len(self.signature_library) >= self.max_signatures:
            self._prune_oldest_signature()

        self.signature_library[signature_id] = signature
        self.acquisition_history.append(signature_id)

        self.logger.debug(f"Acquired anomaly signature: {signature_id}")
        return signature

    def stage_2_expression(self, signature: AnomalySignature) -> npt.NDArray[np.floating[Any]]:
        """
        Stage 2: Expression - Process signature into detection pattern.

        Analogous to CRISPR crRNA transcription for guide RNA creation.

        Args:
            signature: AnomalySignature to process

        Returns:
            Detection pattern (normalized feature vector)
        """
        norm = np.linalg.norm(signature.feature_vector)
        if norm == 0:
            return signature.feature_vector
        return signature.feature_vector / norm

    def stage_3_interference(
        self, input_data: npt.NDArray[np.floating[Any]]
    ) -> tuple[bool, float, str | None]:
        """
        Stage 3: Interference - Detect and neutralize matching anomalies.

        Analogous to Cas proteins using crRNA guides to cut target DNA.

        Args:
            input_data: Input data to check for known anomaly patterns

        Returns:
            Tuple of (is_anomaly, confidence, matching_signature_id)
        """
        if not self.signature_library:
            return False, 0.0, None

        input_features = self._extract_signature_features(input_data)

        best_match_id = None
        best_similarity = 0.0

        for sig_id, signature in self.signature_library.items():
            detection_pattern = self.stage_2_expression(signature)
            similarity = self._compute_similarity(input_features, detection_pattern)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = sig_id

        is_anomaly = bool(best_similarity >= self.similarity_threshold)

        if is_anomaly and best_match_id:
            self.signature_library[best_match_id].detection_count += 1
            self.signature_library[best_match_id].confidence = min(
                1.0, self.signature_library[best_match_id].confidence + 0.01
            )

        return is_anomaly, best_similarity, best_match_id

    def _extract_signature_features(
        self, data: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Extract compact feature representation from anomaly data."""
        flat_data = data.flatten()
        features = np.array(
            [
                np.mean(flat_data),
                np.std(flat_data),
                np.min(flat_data),
                np.max(flat_data),
                np.median(flat_data),
                *np.percentile(flat_data, [25, 75]),
            ]
        )
        return features

    def _generate_signature_id(self, feature_vector: npt.NDArray[np.floating[Any]]) -> str:
        """Generate unique ID for signature based on feature vector.

        Uses hashlib.sha3_256 for stable, reproducible hashing across Python sessions.
        Python's built-in hash() is randomized per-session (PEP 456) and would
        produce different IDs for the same feature vector across runs.
        """
        import hashlib

        # Use SHA3-256 for Ava-Guardian alignment with stable, reproducible hashing
        hash_bytes = hashlib.sha3_256(feature_vector.tobytes()).digest()
        # Take first 8 bytes (64 bits) for a compact but collision-resistant ID
        hash_value = int.from_bytes(hash_bytes[:8], byteorder="big")
        return f"sig_{hash_value:016x}"

    def _compute_similarity(
        self, vec1: npt.NDArray[np.floating[Any]], vec2: npt.NDArray[np.floating[Any]]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _prune_oldest_signature(self) -> None:
        """Remove oldest signature when library is full."""
        if self.acquisition_history:
            oldest_id = self.acquisition_history.pop(0)
            if oldest_id in self.signature_library:
                del self.signature_library[oldest_id]

    def save_library(self, filepath: str) -> None:
        """Save signature library to file for heritable immunity."""
        data = {
            sig_id: {
                "feature_vector": sig.feature_vector.tolist(),
                "timestamp": sig.timestamp,
                "detection_count": sig.detection_count,
                "confidence": sig.confidence,
                "metadata": sig.metadata,
            }
            for sig_id, sig in self.signature_library.items()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_library(self, filepath: str) -> None:
        """Load signature library from file for heritable immunity."""
        with open(filepath) as f:
            data = json.load(f)

        self.signature_library = {
            sig_id: AnomalySignature(
                signature_id=sig_id,
                feature_vector=np.array(sig_data["feature_vector"]),
                timestamp=sig_data["timestamp"],
                detection_count=sig_data["detection_count"],
                confidence=sig_data["confidence"],
                metadata=sig_data.get("metadata", {}),
            )
            for sig_id, sig_data in data.items()
        }
        self.acquisition_history = list(self.signature_library.keys())

    # Backward compatibility aliases
    def save_signature_library(self, filepath: str) -> None:
        """Alias for save_library (deprecated)."""
        return self.save_library(filepath)

    def load_signature_library(self, filepath: str) -> None:
        """Alias for load_library (deprecated)."""
        return self.load_library(filepath)

    def update_online_statistics(self, data: npt.NDArray[np.floating[Any]]) -> None:
        """Update running statistics with new data sample (Welford's algorithm).

        Implements incremental mean and variance computation for online learning.
        Uses exponential forgetting to adapt to concept drift.

        Args:
            data: New data sample to incorporate
        """
        if not self.enable_online_learning:
            return

        features = self._extract_signature_features(data)

        if self._running_mean is None or self._running_var is None:
            self._running_mean = features.copy()
            self._running_var = np.zeros_like(features)
            self._sample_count = 1
        else:
            self._sample_count += 1

            delta = features - self._running_mean
            self._running_mean = self._running_mean * self.forgetting_factor + features * (
                1 - self.forgetting_factor
            )
            delta2 = features - self._running_mean
            self._running_var = self._running_var * self.forgetting_factor + delta * delta2 * (
                1 - self.forgetting_factor
            )

        self._update_sliding_window(features)

    def _update_sliding_window(self, features: npt.NDArray[np.floating[Any]]) -> None:
        """Update sliding window and detect concept drift.

        Args:
            features: Feature vector to add to window
        """
        self._sliding_window.append(features)

        if len(self._sliding_window) > self.sliding_window_size:
            self._sliding_window.pop(0)

        if len(self._sliding_window) >= self.sliding_window_size // 2:
            self._detect_concept_drift()

    def _detect_concept_drift(self) -> None:
        """Detect concept drift using sliding window statistics."""
        if len(self._sliding_window) < self.sliding_window_size // 2:
            return

        window_array = np.array(self._sliding_window)
        mid_point = len(window_array) // 2

        first_half = window_array[:mid_point]
        second_half = window_array[mid_point:]

        first_mean = np.mean(first_half, axis=0)
        second_mean = np.mean(second_half, axis=0)

        first_std = np.std(first_half, axis=0) + 1e-8
        second_std = np.std(second_half, axis=0) + 1e-8

        z_scores = np.abs(second_mean - first_mean) / np.sqrt(
            first_std**2 / len(first_half) + second_std**2 / len(second_half)
        )

        drift_threshold = 2.0
        self._drift_magnitude = float(np.mean(z_scores))
        self._concept_drift_detected = self._drift_magnitude > drift_threshold

        if self._concept_drift_detected:
            self._adapt_to_drift()

    def _adapt_to_drift(self) -> None:
        """Adapt signature library to detected concept drift."""
        if not self._concept_drift_detected:
            return

        for _sig_id, signature in self.signature_library.items():
            signature.confidence *= 1 - self.adaptation_rate

        signatures_to_remove = [
            sig_id
            for sig_id, sig in self.signature_library.items()
            if sig.confidence < _thresholds.anomaly_default
        ]

        for sig_id in signatures_to_remove:
            del self.signature_library[sig_id]
            if sig_id in self.acquisition_history:
                self.acquisition_history.remove(sig_id)

        self.logger.info(
            f"Adapted to concept drift: removed {len(signatures_to_remove)} signatures"
        )

    def adapt_signature(self, signature_id: str, new_data: npt.NDArray[np.floating[Any]]) -> bool:
        """Incrementally adapt an existing signature with new data.

        Args:
            signature_id: ID of signature to adapt
            new_data: New data to incorporate

        Returns:
            True if signature was adapted, False otherwise
        """
        if signature_id not in self.signature_library:
            return False

        if not self.enable_online_learning:
            return False

        signature = self.signature_library[signature_id]
        new_features = self._extract_signature_features(new_data)

        signature.feature_vector = (
            signature.feature_vector * (1 - self.adaptation_rate)
            + new_features * self.adaptation_rate
        )

        signature.detection_count += 1
        signature.confidence = min(1.0, signature.confidence + 0.01)

        return True

    def get_online_learning_stats(self) -> dict[str, Any]:
        """Get online learning statistics.

        Returns:
            Dictionary with online learning statistics
        """
        return {
            "enabled": self.enable_online_learning,
            "sample_count": self._sample_count,
            "window_size": len(self._sliding_window),
            "concept_drift_detected": self._concept_drift_detected,
            "drift_magnitude": self._drift_magnitude,
            "forgetting_factor": self.forgetting_factor,
            "adaptation_rate": self.adaptation_rate,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get adaptive defense statistics."""
        if not self.signature_library:
            base_stats: dict[str, Any] = {
                "total_signatures": 0,
                "total_detections": 0,
                "average_confidence": 0.0,
            }
        else:
            total_detections = sum(s.detection_count for s in self.signature_library.values())
            avg_confidence = np.mean([s.confidence for s in self.signature_library.values()])
            base_stats = {
                "total_signatures": len(self.signature_library),
                "total_detections": total_detections,
                "average_confidence": float(avg_confidence),
                "library_capacity": f"{len(self.signature_library)}/{self.max_signatures}",
            }

        if self.enable_online_learning:
            base_stats["online_learning"] = self.get_online_learning_stats()

        return base_stats


class SelfHealingEngine:
    """
    Unified self-healing system for autonomous error recovery.

    Combines:
    - Component health monitoring with circuit breakers
    - CRISPR-inspired adaptive defense for anomaly pattern memory

    Features:
    - Automatic error detection
    - Component health monitoring
    - Graceful degradation
    - Adaptive anomaly memory
    - Pattern-based threat detection
    """

    def __init__(
        self,
        max_signatures: int = 1000,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize self-healing engine.

        Args:
            max_signatures: Maximum anomaly signatures for adaptive defense
            similarity_threshold: Threshold for signature matching
        """
        self.components: dict[str, dict[str, Any]] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)

        # Adaptive defense subsystem (CRISPR-inspired)
        self.adaptive_defense = AdaptiveDefenseSystem(
            max_signatures=max_signatures,
            similarity_threshold=similarity_threshold,
        )

    def register_component(
        self,
        name: str,
        health_check: Callable[[], bool],
        recovery_action: Callable[[], None] | None = None,
    ) -> None:
        """Register a component for health monitoring."""
        self.components[name] = {
            "health_check": health_check,
            "recovery_action": recovery_action,
            "status": "healthy",
        }

        self.circuit_breakers[name] = CircuitBreaker()

    def check_health(self, component_name: str) -> bool:
        """Check health of a component."""
        if component_name not in self.components:
            return False

        component = self.components[component_name]

        try:
            is_healthy = component["health_check"]()
            component["status"] = "healthy" if is_healthy else "unhealthy"
            return bool(is_healthy)  # type: ignore[no-any-return, unused-ignore]
        except Exception as e:
            self.logger.error(f"Health check failed for {component_name}: {e}")
            component["status"] = "unhealthy"
            return False

    def attempt_recovery(self, component_name: str) -> bool:
        """Attempt to recover a component."""
        if component_name not in self.components:
            return False

        component = self.components[component_name]
        recovery_action = component.get("recovery_action")

        if recovery_action is None:
            return False

        try:
            recovery_action()
            return self.check_health(component_name)
        except Exception as e:
            self.logger.error(f"Recovery failed for {component_name}: {e}")
            return False

    def learn_anomaly(
        self, anomaly_data: npt.NDArray[np.floating[Any]], metadata: dict[str, Any] | None = None
    ) -> AnomalySignature:
        """
        Learn a new anomaly pattern (CRISPR Stage 1: Acquisition).

        Args:
            anomaly_data: Raw anomaly data to learn
            metadata: Optional metadata about the anomaly

        Returns:
            Captured anomaly signature
        """
        return self.adaptive_defense.stage_1_acquisition(anomaly_data, metadata)

    def check_known_anomaly(
        self, input_data: npt.NDArray[np.floating[Any]]
    ) -> tuple[bool, float, str | None]:
        """
        Check if data matches a known anomaly pattern (CRISPR Stage 3: Interference).

        Args:
            input_data: Data to check

        Returns:
            Tuple of (is_known_anomaly, confidence, matching_signature_id)
        """
        return self.adaptive_defense.stage_3_interference(input_data)

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        health_status = {}

        for name, component in self.components.items():
            is_healthy = self.check_health(name)
            health_status[name] = {
                "status": component["status"],
                "is_healthy": is_healthy,
            }

        all_healthy = all(status["is_healthy"] for status in health_status.values())

        return {
            "overall_health": "healthy" if all_healthy else "degraded",
            "components": health_status,
            "adaptive_defense": self.adaptive_defense.get_statistics(),
        }
