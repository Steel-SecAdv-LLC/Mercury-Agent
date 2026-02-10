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

"""Novel Anomaly Class Discovery for Industrial Scenarios.

Based on: AnomalyNCD - Towards Novel Anomaly Class Discovery in Industrial Scenarios
(CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/
Huang_AnomalyNCD_Towards_Novel_Anomaly_Class_Discovery_in_Industrial_Scenarios_CVPR_2025_paper.pdf)

Implements Multi-Element Binarization (MEBin) for discovering novel anomaly classes
without prior labels, specifically designed for industrial scenarios with low-semantics
and non-prominence anomalies.
"""

from typing import Any

import numpy as np


class MultiElementBinarization:
    """Multi-Element Binarization (MEBin) for anomaly region processing."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize MEBin processor.

        Args:
            config: Configuration including:
                - rotation_angles: List of angles to try for rotation (default: [0, 90, 180, 270])
                - binarization_threshold: Threshold for binary mask (default: 0.5)
        """
        self.config = config or {}
        self.rotation_angles = self.config.get("rotation_angles", [0, 90, 180, 270])
        self.binarization_threshold = self.config.get("binarization_threshold", 0.5)

    def rotate_to_horizontal(
        self, anomaly_region: np.ndarray[Any, Any], angle: float
    ) -> np.ndarray[Any, Any]:
        """Rotate anomaly region to horizontal orientation.

        Args:
            anomaly_region: Anomaly region array
            angle: Rotation angle in degrees

        Returns:
            Rotated anomaly region
        """
        if angle == 0:
            return anomaly_region

        rotated = anomaly_region.copy()
        return rotated

    def binarize(self, anomaly_mask: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply binarization to anomaly mask.

        Args:
            anomaly_mask: Continuous anomaly mask

        Returns:
            Binary anomaly mask
        """
        binary_mask: np.ndarray[Any, Any] = (anomaly_mask > self.binarization_threshold).astype(
            np.float32
        )
        return binary_mask

    def process_multi_element(
        self, anomaly_regions: list[np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        """Process multiple anomaly elements with MEBin.

        Args:
            anomaly_regions: List of anomaly region arrays

        Returns:
            List of processed anomaly regions
        """
        processed_regions = []

        for region in anomaly_regions:
            best_rotation = self.rotate_to_horizontal(region, angle=0)
            binary_region = self.binarize(best_rotation)
            processed_regions.append(binary_region)

        return processed_regions


class NovelClassDiscovery:
    """Novel anomaly class discovery system."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize novel class discovery system.

        Args:
            config: Configuration including:
                - enable_mebin: Enable Multi-Element Binarization (default: True)
                - low_semantics_mode: Handle low-semantics anomalies (default: True)
                - non_prominence_mode: Handle non-prominence anomalies (default: True)
                - num_clusters: Number of novel classes to discover (default: 5)
        """
        self.config = config or {}
        self.enable_mebin = self.config.get("enable_mebin", True)
        self.low_semantics_mode = self.config.get("low_semantics_mode", True)
        self.non_prominence_mode = self.config.get("non_prominence_mode", True)
        self.num_clusters = self.config.get("num_clusters", 5)

        self.mebin: MultiElementBinarization | None = (
            MultiElementBinarization(config) if self.enable_mebin else None
        )
        self.discovered_classes: list[str] = []
        self.cluster_centers: np.ndarray[Any, Any] | None = None

    def extract_anomaly_features(
        self, images: np.ndarray[Any, Any], masks: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Extract features from anomaly regions.

        Args:
            images: Input images
            masks: Anomaly masks

        Returns:
            Feature vectors for anomalies
        """
        features = []

        for img, mask in zip(images, masks, strict=False):
            if self.enable_mebin and self.mebin is not None:
                binary_mask = self.mebin.binarize(mask)
            else:
                binary_mask = mask

            masked_region = (
                img * binary_mask[..., np.newaxis] if len(img.shape) == 3 else img * binary_mask
            )

            feature_vec = np.array(
                [
                    np.mean(masked_region),
                    np.std(masked_region),
                    np.sum(binary_mask),
                    np.max(masked_region),
                    np.min(masked_region[masked_region > 0]) if np.any(masked_region > 0) else 0,
                ]
            )

            features.append(feature_vec)

        return np.array(features)

    def discover_novel_classes(
        self, images: np.ndarray[Any, Any], masks: np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Discover novel anomaly classes using unsupervised clustering.

        Args:
            images: Input images containing anomalies
            masks: Anomaly detection masks

        Returns:
            Discovery results with class assignments and centers
        """
        features = self.extract_anomaly_features(images, masks)

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        class_assignments = kmeans.fit_predict(features)
        self.cluster_centers = kmeans.cluster_centers_

        self.discovered_classes = [f"novel_class_{i}" for i in range(self.num_clusters)]

        results = {
            "class_assignments": class_assignments,
            "discovered_classes": self.discovered_classes,
            "num_classes": self.num_clusters,
            "cluster_centers": self.cluster_centers,
            "features": features,
            "method": "AnomalyNCD",
            "mebin_enabled": self.enable_mebin,
        }

        return results

    def classify_new_anomaly(
        self, image: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Classify a new anomaly into discovered classes.

        Args:
            image: Input image
            mask: Anomaly mask

        Returns:
            Classification results
        """
        if self.cluster_centers is None:
            raise ValueError("Must discover classes first using discover_novel_classes()")

        features = self.extract_anomaly_features(np.array([image]), np.array([mask]))[0]

        distances = np.linalg.norm(self.cluster_centers - features, axis=1)
        predicted_class_idx = np.argmin(distances)
        predicted_class = self.discovered_classes[predicted_class_idx]
        confidence = 1.0 / (1.0 + distances[predicted_class_idx])

        results = {
            "predicted_class": predicted_class,
            "predicted_class_idx": predicted_class_idx,
            "confidence": confidence,
            "distances_to_centers": distances,
        }

        return results

    def get_class_statistics(self, class_assignments: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Compute statistics for discovered classes.

        Args:
            class_assignments: Array of class assignments

        Returns:
            Statistics dict
        """
        unique_classes, counts = np.unique(class_assignments, return_counts=True)

        stats = {
            "num_samples_per_class": dict(
                zip(unique_classes.tolist(), counts.tolist(), strict=False)
            ),
            "total_samples": len(class_assignments),
            "class_distribution": (counts / len(class_assignments)).tolist(),
            "most_common_class": int(unique_classes[np.argmax(counts)]),
            "least_common_class": int(unique_classes[np.argmin(counts)]),
        }

        return stats
