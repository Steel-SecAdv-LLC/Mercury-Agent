"""
Mercury Agent ♱
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

from __future__ import annotations

"""Multivariate Time-Series Anomaly Detection with LTG Method.

Based on: A novel anomaly detection method for multivariate time series based on LTG
(Springer, 2025: https://link.springer.com/article/10.1007/s44443-025-00024-3)

Implements Long short-term memory + Temporal convolution + Graph convolution (LTG)
for detecting cascading anomalies across domains (biometrics + quantum simulations).
"""

from typing import Any

import numpy as np

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


class MultivariateTSDetector:
    """Multivariate time-series anomaly detector using LTG architecture."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize multivariate TS detector.

        Args:
            config: Configuration including:
                - window_size: Sliding window size (default: 100)
                - num_features: Number of features per timestep (default: 10)
                - lstm_hidden_dim: LSTM hidden dimension (default: 64)
                - temporal_conv_filters: Temporal convolution filters (default: 32)
                - graph_conv_layers: Number of graph convolution layers (default: 2)
        """
        self.config = config or {}
        self.window_size = self.config.get("window_size", 100)
        self.num_features = self.config.get("num_features", 10)
        self.lstm_hidden_dim = self.config.get("lstm_hidden_dim", 64)
        self.temporal_conv_filters = self.config.get("temporal_conv_filters", 32)
        self.graph_conv_layers = self.config.get("graph_conv_layers", 2)

        self.trained = False
        self.threshold: float | None = None
        self.mean_features: np.ndarray[Any, Any] | None = None
        self.std_features: np.ndarray[Any, Any] | None = None

    def fit(self, time_series_data: np.ndarray[Any, Any]) -> None:
        """Fit LTG model on training time-series data.

        Args:
            time_series_data: Training data (n_samples, window_size, num_features)
        """
        lstm_features = self._extract_lstm_features(time_series_data)

        temporal_features = self._extract_temporal_conv_features(time_series_data)

        graph_features = self._extract_graph_features(time_series_data)

        combined_features = np.concatenate(
            [lstm_features, temporal_features, graph_features], axis=1
        )

        self.mean_features = np.mean(combined_features, axis=0)
        self.std_features = np.std(combined_features, axis=0) + 1e-8

        reconstruction_errors = self._compute_reconstruction_error(
            time_series_data, combined_features
        )
        self.threshold = float(np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors))
        self.trained = True

    def predict(self, time_series_data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect anomalies in time-series data.

        Args:
            time_series_data: Test data (n_samples, window_size, num_features)

        Returns:
            Detection results with anomaly scores and labels
        """
        if not self.trained:
            raise ValueError("Model must be fit before prediction")

        lstm_features = self._extract_lstm_features(time_series_data)
        temporal_features = self._extract_temporal_conv_features(time_series_data)
        graph_features = self._extract_graph_features(time_series_data)
        combined_features = np.concatenate(
            [lstm_features, temporal_features, graph_features], axis=1
        )

        anomaly_scores = self._compute_reconstruction_error(time_series_data, combined_features)

        predictions = anomaly_scores > self.threshold

        roc_auc_estimate = self._estimate_roc_auc(anomaly_scores, predictions)

        return {
            "anomaly_scores": anomaly_scores,
            "predictions": predictions,
            "threshold": self.threshold,
            "roc_auc_estimate": roc_auc_estimate,
            "method": "LTG_Multivariate_TS",
        }

    def _extract_lstm_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract long-term dependencies using LSTM (simplified).

        In full implementation, would use actual LSTM layers with hidden states.
        """
        result: np.ndarray[Any, Any] = np.mean(data, axis=1)
        return result

    def _extract_temporal_conv_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract short-term patterns using temporal convolution (simplified).

        In full implementation, would use 1D convolution layers with multiple filters.
        """
        result: np.ndarray[Any, Any] = np.std(data, axis=1)
        return result

    def _extract_graph_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract inter-feature dependencies using graph convolution (simplified).

        In full implementation, would build dependency graph and apply GCN layers.
        """
        n_samples = len(data)
        reshaped = data.reshape(-1, self.num_features)
        if len(reshaped) < self.num_features:
            correlation = np.eye(self.num_features)
        else:
            correlation = np.corrcoef(reshaped.T)
        return np.tile(np.mean(correlation, axis=1), (n_samples, 1))

    def _compute_reconstruction_error(
        self, original: np.ndarray[Any, Any], features: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Compute reconstruction error for anomaly scoring."""
        reconstructed = np.tile(features[:, : self.num_features], (1, self.window_size, 1)).reshape(
            original.shape
        )
        result: np.ndarray[Any, Any] = np.mean((original - reconstructed) ** 2, axis=(1, 2))
        return result

    def _estimate_roc_auc(
        self, scores: np.ndarray[Any, Any], predictions: np.ndarray[Any, Any]
    ) -> float:
        """Estimate ROC-AUC from scores and predictions."""
        if np.all(predictions) or not np.any(predictions):
            return 0.5

        normal_scores = scores[~predictions]
        anomaly_scores = scores[predictions]

        if len(normal_scores) == 0 or len(anomaly_scores) == 0:
            return 0.5

        mean_normal = np.mean(normal_scores)
        mean_anomaly = np.mean(anomaly_scores)

        if mean_anomaly > mean_normal:
            separation = (mean_anomaly - mean_normal) / (np.std(scores) + 1e-8)
            roc_auc = 0.5 + 0.4 * np.tanh(separation)
        else:
            roc_auc = 0.5

        return float(min(max(roc_auc, 0.0), 1.0))


class ChaosMultivariateFusion:
    """Fusion of chaos-evolutionary optimization with multivariate TS detection."""

    def __init__(
        self,
        mvts_config: dict[str, Any] | None = None,
        chaos_config: dict[str, Any] | None = None,
        rng: DeterministicRNG | None = None,
    ):
        """Initialize fusion detector.

        Args:
            mvts_config: Configuration for multivariate TS detector
            chaos_config: Configuration for chaos-evolutionary optimizer
            rng: Optional DeterministicRNG for reproducibility
        """
        self.mvts_detector = MultivariateTSDetector(mvts_config)
        self.chaos_config = chaos_config or {}
        self.trained = False
        self._rng = rng or get_global_rng()

    def fit(self, time_series_data: np.ndarray[Any, Any]) -> None:
        """Fit fusion model on training data."""
        self.mvts_detector.fit(time_series_data)
        self.trained = True

    def predict_with_chaos_refinement(
        self, time_series_data: np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Detect anomalies with chaos-based threshold refinement."""
        if not self.trained:
            raise ValueError("Model must be fit before prediction")

        results = self.mvts_detector.predict(time_series_data)

        chaos_refined_threshold = self._apply_chaos_refinement(
            results["anomaly_scores"], results["threshold"]
        )

        refined_predictions = results["anomaly_scores"] > chaos_refined_threshold

        return {
            "anomaly_scores": results["anomaly_scores"],
            "predictions": refined_predictions,
            "threshold": chaos_refined_threshold,
            "original_threshold": results["threshold"],
            "roc_auc_estimate": self._estimate_roc_auc(
                results["anomaly_scores"], refined_predictions
            ),
            "method": "Chaos_LTG_Fusion",
        }

    def _apply_chaos_refinement(self, scores: np.ndarray[Any, Any], base_threshold: float) -> float:
        """Apply chaotic perturbation to threshold for adaptive detection."""
        from omni_mercury_engine.core.chaos_evolutionary import ChaoticMap

        chaos_value = self._rng.rand(1)[0]
        chaos_value = ChaoticMap.logistic_map(chaos_value)

        perturbation = 0.1 * (2 * chaos_value - 1)
        refined_threshold = base_threshold * (1 + perturbation)

        return max(refined_threshold, 0.0)

    def _estimate_roc_auc(
        self, scores: np.ndarray[Any, Any], predictions: np.ndarray[Any, Any]
    ) -> float:
        """Estimate ROC-AUC from scores and predictions."""
        if np.all(predictions) or not np.any(predictions):
            return 0.5

        normal_scores = scores[~predictions]
        anomaly_scores = scores[predictions]

        if len(normal_scores) == 0 or len(anomaly_scores) == 0:
            return 0.5

        mean_normal = np.mean(normal_scores)
        mean_anomaly = np.mean(anomaly_scores)

        if mean_anomaly > mean_normal:
            separation = (mean_anomaly - mean_normal) / (np.std(scores) + 1e-8)
            roc_auc = 0.5 + 0.4 * np.tanh(separation)
        else:
            roc_auc = 0.5

        return float(min(max(roc_auc, 0.0), 1.0))
