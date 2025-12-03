"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

"""
CRISPR-Inspired Self-Healing Module for Adaptive Anomaly Detection

Inspired by CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats),
the prokaryotic adaptive immune system discovered by Ishino (1987), with mechanism
clarified by Mojica (2007) and gene editing breakthrough by Doudna/Charpentier (2012).

Three-Stage Defense (analogous to CRISPR):
1. Acquisition: Capture novel anomaly signatures (like CRISPR spacer acquisition)
2. Expression: Process signatures into detection patterns (like crRNA transcription)
3. Interference: Neutralize/block similar anomalies (like Cas protein cutting)

Research source: Wikipedia - CRISPR (https://en.wikipedia.org/wiki/CRISPR)
Verified: October 2025

Attribution: Integrated concept from CRISPR biological mechanism
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalySignature:
    """Compact representation of anomaly pattern (analogous to CRISPR spacer)."""

    signature_id: str
    feature_vector: np.ndarray
    timestamp: float
    detection_count: int = 0
    confidence: float = 0.0


class CRISPRInspiredSelfHealing:
    """
    Adaptive self-healing system inspired by CRISPR immune mechanism.

    Maintains library of known anomaly signatures and uses them for
    faster future detection and neutralization.
    """

    def __init__(self, max_signatures: int = 1000, similarity_threshold: float = 0.85):
        """
        Initialize self-healing system.

        Args:
            max_signatures: Maximum number of anomaly signatures to store
            similarity_threshold: Threshold for matching signatures (0-1)
        """
        self.max_signatures = max_signatures
        self.similarity_threshold = similarity_threshold
        self.signature_library: dict[str, AnomalySignature] = {}
        self.acquisition_history: list[str] = []

    def stage_1_acquisition(self, anomaly_data: np.ndarray) -> AnomalySignature:
        """
        Stage 1: Acquisition - Capture novel anomaly signature.

        Analogous to CRISPR spacer acquisition from invading phage DNA.

        Args:
            anomaly_data: Raw anomaly data to capture

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
        )

        if len(self.signature_library) >= self.max_signatures:
            self._prune_oldest_signature()

        self.signature_library[signature_id] = signature
        self.acquisition_history.append(signature_id)

        return signature

    def stage_2_expression(self, signature: AnomalySignature) -> np.ndarray:
        """
        Stage 2: Expression - Process signature into detection pattern.

        Analogous to CRISPR crRNA transcription for guide RNA creation.

        Args:
            signature: AnomalySignature to process

        Returns:
            Detection pattern (processed feature vector)
        """
        detection_pattern: np.ndarray = signature.feature_vector / (
            np.linalg.norm(signature.feature_vector) + 1e-8
        )
        return detection_pattern

    def stage_3_interference(self, input_data: np.ndarray) -> tuple[bool, float, str | None]:
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

    def _extract_signature_features(self, data: np.ndarray) -> np.ndarray:
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

    def _generate_signature_id(self, feature_vector: np.ndarray) -> str:
        """Generate unique ID for signature based on feature vector."""
        hash_value = hash(feature_vector.tobytes())
        return f"sig_{abs(hash_value):016x}"

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
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

    def save_signature_library(self, filepath: str) -> None:
        """Save signature library to file for heritable immunity."""
        data = {
            sig_id: {
                "feature_vector": sig.feature_vector.tolist(),
                "timestamp": sig.timestamp,
                "detection_count": sig.detection_count,
                "confidence": sig.confidence,
            }
            for sig_id, sig in self.signature_library.items()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_signature_library(self, filepath: str) -> None:
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
            )
            for sig_id, sig_data in data.items()
        }
        self.acquisition_history = list(self.signature_library.keys())
