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


"""
Tests for Foundation Model Adapters.

Tests TimeGPT, Chronos, Matrix Profile, and Foundation Ensemble adapters.
"""

import importlib.util

import pytest


HAS_TORCH = importlib.util.find_spec("torch") is not None


pytestmark = pytest.mark.foundation


class TestTimeGPTAdapter:
    """Tests for TimeGPT API adapter."""

    def test_timegpt_initialization(self):
        """Test TimeGPT adapter can be initialized."""
        from omni_mercury_engine.models.foundation import TimeGPTAdapter

        adapter = TimeGPTAdapter()
        assert adapter is not None

    def test_timegpt_config(self):
        """Test TimeGPT with custom config."""
        from omni_mercury_engine.models.foundation import TimeGPTAdapter
        from omni_mercury_engine.models.foundation.timegpt_adapter import TimeGPTConfig

        config = TimeGPTConfig(
            model="timegpt-1-long-horizon",
            freq="D",
            fh=14,
        )
        adapter = TimeGPTAdapter(config=config)
        assert adapter.config.fh == 14

    def test_timegpt_detect_mock(self, univariate_data):
        """Test TimeGPT anomaly detection with mock mode."""
        from omni_mercury_engine.models.foundation import TimeGPTAdapter

        # Without API key, should use mock mode
        adapter = TimeGPTAdapter()
        result = adapter.detect(univariate_data)

        assert "scores" in result
        assert "is_anomaly" in result

    def test_timegpt_forecast(self, univariate_data):
        """Test TimeGPT forecasting."""
        from omni_mercury_engine.models.foundation import TimeGPTAdapter

        adapter = TimeGPTAdapter()
        forecast = adapter.forecast(univariate_data, horizon=10)

        assert forecast is not None
        assert len(forecast) == 10


class TestChronosAdapter:
    """Tests for Amazon Chronos adapter."""

    def test_chronos_initialization(self):
        """Test Chronos adapter can be initialized."""
        from omni_mercury_engine.models.foundation import ChronosAdapter

        adapter = ChronosAdapter()
        assert adapter is not None

    def test_chronos_config(self):
        """Test Chronos with custom config."""
        from omni_mercury_engine.models.foundation import ChronosAdapter
        from omni_mercury_engine.models.foundation.chronos_adapter import ChronosConfig

        config = ChronosConfig(
            model_name="amazon/chronos-t5-small",
            prediction_length=24,
            num_samples=10,
        )
        adapter = ChronosAdapter(config=config)
        assert adapter.config.prediction_length == 24

    def test_chronos_detect_mock(self, univariate_data):
        """Test Chronos anomaly detection with mock mode."""
        from omni_mercury_engine.models.foundation import ChronosAdapter

        adapter = ChronosAdapter()
        result = adapter.detect(univariate_data)

        assert "scores" in result
        assert "is_anomaly" in result
        assert "forecasts" in result

    def test_chronos_forecast(self, univariate_data):
        """Test Chronos forecasting."""
        from omni_mercury_engine.models.foundation import ChronosAdapter

        adapter = ChronosAdapter()
        forecast = adapter.forecast(univariate_data, horizon=10)

        assert forecast is not None


class TestMatrixProfileAdapter:
    """Tests for Matrix Profile (STUMPY) adapter."""

    def test_matrix_profile_initialization(self):
        """Test Matrix Profile adapter can be initialized."""
        from omni_mercury_engine.models.foundation import MatrixProfileAdapter

        adapter = MatrixProfileAdapter()
        assert adapter is not None

    def test_matrix_profile_config(self):
        """Test Matrix Profile with custom config."""
        from omni_mercury_engine.models.foundation import MatrixProfileAdapter
        from omni_mercury_engine.models.foundation.matrix_profile import MatrixProfileConfig

        config = MatrixProfileConfig(
            window_size=50,
            normalize=True,
            discord_threshold=2.0,
        )
        adapter = MatrixProfileAdapter(config=config)
        assert adapter.config.window_size == 50

    def test_matrix_profile_detect(self, time_series_with_anomaly):
        """Test Matrix Profile anomaly detection."""
        from omni_mercury_engine.models.foundation import MatrixProfileAdapter

        adapter = MatrixProfileAdapter()
        result = adapter.detect(time_series_with_anomaly)

        assert "scores" in result
        assert "is_anomaly" in result
        assert "discords" in result

    def test_matrix_profile_find_motifs(self, univariate_data):
        """Test motif discovery."""
        from omni_mercury_engine.models.foundation import MatrixProfileAdapter

        adapter = MatrixProfileAdapter()
        motifs = adapter.find_motifs(univariate_data, top_k=3)

        assert motifs is not None
        assert isinstance(motifs, list)

    def test_matrix_profile_find_discords(self, time_series_with_anomaly):
        """Test discord (anomaly) discovery."""
        from omni_mercury_engine.models.foundation import MatrixProfileAdapter

        adapter = MatrixProfileAdapter()
        discords = adapter.find_discords(time_series_with_anomaly, top_k=5)

        assert discords is not None
        assert isinstance(discords, list)


class TestFoundationEnsemble:
    """Tests for Foundation Model Ensemble."""

    def test_ensemble_initialization(self):
        """Test Foundation Ensemble can be initialized."""
        from omni_mercury_engine.models.foundation import FoundationEnsemble

        ensemble = FoundationEnsemble()
        assert ensemble is not None

    def test_ensemble_config(self):
        """Test Foundation Ensemble with custom config."""
        from omni_mercury_engine.models.foundation import FoundationEnsemble
        from omni_mercury_engine.models.foundation.ensemble import EnsembleConfig

        config = EnsembleConfig(
            adapters=["matrix_profile"],
            weights={"matrix_profile": 1.0},
            aggregation="mean",
        )
        ensemble = FoundationEnsemble(config=config)
        assert ensemble.config.aggregation == "mean"

    def test_ensemble_detect(self, time_series_with_anomaly):
        """Test Foundation Ensemble anomaly detection."""
        from omni_mercury_engine.models.foundation import FoundationEnsemble

        ensemble = FoundationEnsemble()
        result = ensemble.detect(time_series_with_anomaly)

        assert "scores" in result
        assert "is_anomaly" in result
        assert "adapter_scores" in result

    def test_ensemble_aggregation_methods(self, univariate_data):
        """Test different aggregation methods."""
        from omni_mercury_engine.models.foundation import FoundationEnsemble
        from omni_mercury_engine.models.foundation.ensemble import EnsembleConfig

        for method in ["mean", "max", "voting"]:
            config = EnsembleConfig(
                adapters=["matrix_profile"],
                aggregation=method,
            )
            ensemble = FoundationEnsemble(config=config)
            result = ensemble.detect(univariate_data)
            assert "scores" in result


class TestBaseFoundationAdapter:
    """Tests for base foundation model adapter class."""

    def test_base_adapter_initialization(self):
        """Test BaseFoundationAdapter can be initialized."""
        from omni_mercury_engine.models.foundation.base_foundation import BaseFoundationAdapter

        adapter = BaseFoundationAdapter()
        assert adapter is not None

    def test_base_adapter_interface(self):
        """Test base adapter has required interface."""
        from omni_mercury_engine.models.foundation.base_foundation import BaseFoundationAdapter

        adapter = BaseFoundationAdapter()

        # Check interface methods exist
        assert hasattr(adapter, "detect")
        assert hasattr(adapter, "forecast")
        assert hasattr(adapter, "fit")
