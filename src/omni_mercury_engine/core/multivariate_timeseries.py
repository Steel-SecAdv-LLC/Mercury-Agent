# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Multivariate time-series anomaly detection — statistical LTG-shaped baseline.

Motivated by the LTG method (Springer, 2025:
https://link.springer.com/article/10.1007/s44443-025-00024-3), which combines Long
short-term memory, Temporal convolution and Graph convolution.

**This module does not implement that architecture.** It implements a
deterministic statistical baseline with the same three-branch *shape*: the
"LSTM" branch is a per-window mean, the "temporal convolution" branch is a
per-window standard deviation, and the "graph" branch is a feature-correlation
summary. There are no learned parameters, no recurrent state and no convolution
kernels anywhere in this file. The names are kept because they map onto the
paper's branches and the reconstruction-error score they feed; the docstrings
say plainly what each one computes.

The previous docstring described this as an LTG implementation and the detector
returned a ``roc_auc_estimate`` computed as ``0.5 + 0.4 * tanh(separation)`` from
its *own* scores and its *own* thresholded predictions -- no labels were involved
at any point, so the number could not be an AUC of anything and rose whenever the
detector was merely self-consistent. It has been removed rather than renamed: a
fabricated metric is worse than no metric, because it invites comparison with
real ones. Callers that want a real AUC must supply ground-truth labels and use
``sklearn.metrics.roc_auc_score`` (or Mercury's evaluation harness) over the
returned ``anomaly_scores``.
"""

from __future__ import annotations

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

        return {
            "anomaly_scores": anomaly_scores,
            "predictions": predictions,
            "threshold": self.threshold,
            # No ``roc_auc_estimate``: this method never sees a label, so it
            # cannot report a ranking metric. See the module docstring.
            "method": "statistical_multivariate_ts",
            "is_learned": False,
        }

    def _extract_lstm_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the per-window mean — the long-horizon branch.

        Named for the LTG paper's LSTM branch, which it stands in for. It is a
        mean, not a recurrent network: there is no hidden state and nothing
        learned.
        """
        result: np.ndarray[Any, Any] = np.mean(data, axis=1)
        return result

    def _extract_temporal_conv_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the per-window standard deviation — the short-horizon branch.

        Named for the LTG paper's temporal-convolution branch. It is a standard
        deviation, not a convolution: there are no filters and nothing learned.
        """
        result: np.ndarray[Any, Any] = np.std(data, axis=1)
        return result

    def _extract_graph_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return a feature-correlation summary — the inter-feature branch.

        Named for the LTG paper's graph-convolution branch. It is the row-mean of
        the Pearson correlation matrix, not a GCN: no graph is built and nothing
        is learned.
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
            # No ``roc_auc_estimate``: no labels reach this method either.
            "method": "chaos_refined_statistical_multivariate_ts",
            "is_learned": False,
        }

    def _apply_chaos_refinement(self, scores: np.ndarray[Any, Any], base_threshold: float) -> float:
        """Apply chaotic perturbation to threshold for adaptive detection."""
        from omni_mercury_engine.core.chaos_evolutionary import ChaoticMap

        chaos_value = self._rng.rand(1)[0]
        chaos_value = ChaoticMap.logistic_map(chaos_value)

        perturbation = 0.1 * (2 * chaos_value - 1)
        refined_threshold = base_threshold * (1 + perturbation)

        return max(refined_threshold, 0.0)
