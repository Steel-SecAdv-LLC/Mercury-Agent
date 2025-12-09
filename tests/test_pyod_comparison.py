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
from __future__ import annotations

"""
Tests for PyOD comparison module.
"""

import numpy as np

from omni_anomaly_engine.comparison import CombinationMethod, PyODAlgorithm, PyODComparison


class TestPyODComparison:
    """Tests for PyOD comparison functionality."""

    def test_initialization(self):
        """Test comparison module initialization."""
        comparison = PyODComparison()
        assert len(comparison.algorithm_characteristics) > 0

    def test_algorithm_recommendation(self):
        """Test algorithm recommendation."""
        comparison = PyODComparison()

        data_char = {"num_samples": 1000, "num_features": 10, "has_clusters": False}

        result = comparison.recommend_algorithm(data_char)

        assert "recommendations" in result
        assert len(result["recommendations"]) > 0
        assert "algorithm" in result["recommendations"][0]

    def test_algorithm_recommendation_large_dataset(self):
        """Test recommendation for large dataset."""
        comparison = PyODComparison()

        data_char = {"num_samples": 200000, "num_features": 50}

        constraints = {"max_time_seconds": 30}

        result = comparison.recommend_algorithm(data_char, constraints)

        assert result["recommendations"][0]["algorithm"] == PyODAlgorithm.COPOD

    def test_combine_predictions_average(self):
        """Test prediction combination with average method."""
        comparison = PyODComparison()

        predictions = {
            "detector1": np.array([0.1, 0.2, 0.3, 0.4]),
            "detector2": np.array([0.2, 0.3, 0.4, 0.5]),
            "detector3": np.array([0.15, 0.25, 0.35, 0.45]),
        }

        combined = comparison.combine_predictions(predictions, CombinationMethod.AVERAGE)

        assert combined.shape == (4,)
        assert np.allclose(combined, [0.15, 0.25, 0.35, 0.45])

    def test_combine_predictions_maximum(self):
        """Test prediction combination with maximum method."""
        comparison = PyODComparison()

        predictions = {
            "detector1": np.array([0.1, 0.2, 0.3, 0.4]),
            "detector2": np.array([0.2, 0.3, 0.4, 0.5]),
        }

        combined = comparison.combine_predictions(predictions, CombinationMethod.MAXIMUM)

        assert combined.shape == (4,)
        assert np.allclose(combined, [0.2, 0.3, 0.4, 0.5])

    def test_combine_predictions_aom(self):
        """Test prediction combination with AOM method."""
        comparison = PyODComparison()

        predictions = {
            "detector1": np.array([0.1, 0.2, 0.3, 0.4]),
            "detector2": np.array([0.2, 0.3, 0.4, 0.5]),
            "detector3": np.array([0.15, 0.25, 0.35, 0.45]),
            "detector4": np.array([0.25, 0.35, 0.45, 0.55]),
        }

        combined = comparison.combine_predictions(predictions, CombinationMethod.AOM)

        assert combined.shape == (4,)

    def test_comparison_summary(self):
        """Test comparison summary generation."""
        comparison = PyODComparison()

        results = {"omni_ava": {"f1": 0.85}, "pyod_algorithms": {}}

        summary = comparison._generate_comparison_summary(results)

        assert "omni_ava_strengths" in summary
        assert "pyod_strengths" in summary
        assert "recommendation" in summary
