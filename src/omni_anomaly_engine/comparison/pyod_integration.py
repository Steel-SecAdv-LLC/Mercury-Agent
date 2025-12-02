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
PyOD Integration and Comparison

Compare OMNI ♱ AVA with PyOD's 40+ anomaly detection algorithms.

Research sources:
- PyOD GitHub (github.com/yzhao062/pyod)
- Zhao et al. "PyOD: A Python Toolbox" (JMLR 2019)

Note: This compares approaches, doesn't copy PyOD code
"""

from typing import Dict, List, Optional, Any
import numpy as np
from enum import Enum


class PyODAlgorithm(Enum):
    """PyOD algorithms for comparison."""

    ISOLATION_FOREST = "isolation_forest"
    LOF = "local_outlier_factor"
    COPOD = "copod"
    ECOD = "ecod"
    OCSVM = "one_class_svm"
    KNN = "knn"
    AUTOENCODER = "autoencoder"
    PCA = "pca"


class CombinationMethod(Enum):
    """Ensemble combination methods from PyOD."""

    AVERAGE = "average"
    MAXIMUM = "maximum"
    AOM = "average_of_maximum"
    MOA = "maximum_of_average"


class PyODComparison:
    """
    Compare OMNI ♱ AVA with PyOD algorithms.

    Enables:
    - Benchmarking Omni-AXA's 13 engines against PyOD's 40+ algorithms
    - Learning from PyOD's ensemble combination methods
    - Identifying complementary detection approaches
    - Algorithm selection guidance
    """

    def __init__(self):
        self.algorithm_characteristics = self._init_algorithm_profiles()
        self.benchmark_results = {}

    def _init_algorithm_profiles(self) -> Dict:
        """Initialize algorithm characteristics for selection guidance."""
        return {
            PyODAlgorithm.ISOLATION_FOREST: {
                "type": "tree_based",
                "strengths": [
                    "Efficient",
                    "No assumptions on data distribution",
                    "Handles high dimensions",
                ],
                "weaknesses": ["May struggle with local anomalies", "Not interpretable"],
                "best_for": ["High-dimensional data", "Global anomalies", "Large datasets"],
                "complexity": "O(n log n)",
                "parameters": ["n_estimators", "max_samples", "contamination"],
            },
            PyODAlgorithm.LOF: {
                "type": "density_based",
                "strengths": [
                    "Captures local density deviations",
                    "Intuitive concept",
                    "Works well for clusters",
                ],
                "weaknesses": [
                    "Sensitive to k parameter",
                    "Computationally expensive O(n²)",
                    "Memory intensive",
                ],
                "best_for": ["Datasets with varying density", "Local anomalies", "Clustered data"],
                "complexity": "O(n²)",
                "parameters": ["n_neighbors", "contamination"],
            },
            PyODAlgorithm.COPOD: {
                "type": "statistical",
                "strengths": ["Fast O(n)", "Parameter-free", "Empirical distribution", "Scalable"],
                "weaknesses": ["May miss complex patterns", "Assumes feature independence"],
                "best_for": ["Quick baseline", "Large-scale data", "When speed is critical"],
                "complexity": "O(n)",
                "parameters": ["contamination"],
            },
            PyODAlgorithm.ECOD: {
                "type": "statistical",
                "strengths": ["No hyperparameters", "Fast", "Works on raw data", "Interpretable"],
                "weaknesses": ["Assumes feature independence", "May not capture dependencies"],
                "best_for": ["Quick deployment", "No tuning budget", "Interpretability needed"],
                "complexity": "O(n log n)",
                "parameters": [],
            },
        }

    def recommend_algorithm(
        self, data_characteristics: Dict[str, Any], constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Recommend best algorithm(s) based on data characteristics.

        Args:
            data_characteristics: Data properties dict
            constraints: Optional constraints dict

        Returns:
            Recommended algorithms with rationale
        """
        recommendations = []

        num_samples = data_characteristics.get("num_samples", 1000)
        max_time = (
            constraints.get("max_time_seconds", float("inf")) if constraints else float("inf")
        )

        if num_samples > 100000 and max_time < 60:
            recommendations.append(
                {
                    "algorithm": PyODAlgorithm.COPOD,
                    "rationale": "Fast O(n), parameter-free, scales to large datasets",
                    "priority": 1,
                }
            )
        else:
            recommendations.append(
                {
                    "algorithm": PyODAlgorithm.ISOLATION_FOREST,
                    "rationale": "General-purpose, robust default",
                    "priority": 1,
                }
            )

        recommendations.sort(key=lambda x: x["priority"])

        return {
            "recommendations": recommendations,
            "data_summary": data_characteristics,
            "constraints": constraints,
        }

    def combine_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        method: CombinationMethod = CombinationMethod.AVERAGE,
    ) -> np.ndarray:
        """
        Combine predictions from multiple detectors using PyOD-inspired methods.

        Args:
            predictions: {detector_name: anomaly_scores} for multiple detectors
            method: Combination method (Average, Maximum, AOM, MOA)

        Returns:
            Combined anomaly scores
        """
        scores_matrix = np.array(list(predictions.values()))

        if method == CombinationMethod.AVERAGE:
            return np.mean(scores_matrix, axis=0)

        elif method == CombinationMethod.MAXIMUM:
            return np.max(scores_matrix, axis=0)

        elif method == CombinationMethod.AOM:
            num_detectors = len(predictions)
            k = max(1, num_detectors // 2)

            partitions = np.array_split(scores_matrix, num_detectors // k)

            max_scores = [np.max(partition, axis=0) for partition in partitions]

            return np.mean(max_scores, axis=0)

        elif method == CombinationMethod.MOA:
            num_detectors = len(predictions)
            k = max(1, num_detectors // 2)

            partitions = np.array_split(scores_matrix, num_detectors // k)

            avg_scores = [np.mean(partition, axis=0) for partition in partitions]

            return np.max(avg_scores, axis=0)

        return np.mean(scores_matrix, axis=0)

    def benchmark_against_pyod(
        self,
        omni_engine,
        test_data: np.ndarray,
        ground_truth: np.ndarray,
        pyod_algorithms: List[PyODAlgorithm],
    ) -> Dict:
        """
        Benchmark OMNI ♱ AVA against PyOD algorithms.

        Args:
            omni_engine: OMNI ♱ AVA instance
            test_data: Test dataset
            ground_truth: True anomaly labels
            pyod_algorithms: PyOD algorithms to compare

        Returns:
            Benchmark results with metrics for each algorithm
        """
        results = {
            "omni_ava": self._evaluate_detector(omni_engine, test_data, ground_truth),
            "pyod_algorithms": {},
        }

        for algo in pyod_algorithms:
            results["pyod_algorithms"][algo.value] = {
                "characteristics": self.algorithm_characteristics.get(algo, {}),
                "note": "Would run actual PyOD algorithm here if library installed",
            }

        results["comparison_summary"] = self._generate_comparison_summary(results)

        return results

    def _evaluate_detector(self, detector, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate detector performance."""
        try:
            scores = detector.predict(data)

            threshold = np.percentile(scores, 95)
            predictions = (scores > threshold).astype(int)

            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
        except Exception as e:
            return {"error": str(e)}

    def _generate_comparison_summary(self, results: Dict) -> Dict:
        """Generate summary comparing Omni-AXA with PyOD algorithms."""
        return {
            "omni_ava_strengths": [
                "Multi-domain fusion (mathematical, physical, quantum, biometric)",
                "Interdisciplinary approach",
                "13 integrated engines",
                "Quantum-inspired methods",
                "Ethical scalars framework",
            ],
            "pyod_strengths": [
                "40+ algorithms (comprehensive)",
                "Classical statistical methods",
                "Well-documented",
                "Easy to use API",
                "Active maintenance",
            ],
            "recommendation": "Use Omni-AXA for STEM-specific anomaly detection with domain fusion; use PyOD for general-purpose anomaly detection with classical methods",
        }
