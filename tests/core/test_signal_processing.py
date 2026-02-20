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

"""Tests for Signal Processing module."""

import numpy as np
import pytest

from omni_mercury_engine.core.signal_processing import (
    AdaptiveNoiseFilter,
    FilterConfig,
    FilterType,
    MultiStageFilter,
    compute_interaction_features,
    compute_rolling_statistics,
    compute_temporal_lag_features,
)


class TestFilterType:
    """Tests for FilterType enumeration."""

    def test_filter_type_values(self):
        """Test FilterType enum values."""
        assert FilterType.FFT_LOWPASS.value == "fft_lowpass"
        assert FilterType.WAVELET.value == "wavelet"
        assert FilterType.KALMAN.value == "kalman"
        assert FilterType.SAVITZKY_GOLAY.value == "savitzky_golay"
        assert FilterType.ADAPTIVE_BANDPASS.value == "adaptive_bandpass"
        assert FilterType.MEDIAN.value == "median"
        assert FilterType.EMA.value == "ema"


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FilterConfig()

        assert config.filter_type == FilterType.FFT_LOWPASS
        assert config.window_size == 5
        assert config.poly_order == 2
        assert config.cutoff_freq == 0.5
        assert config.kalman_process_noise == 1e-5
        assert config.kalman_measurement_noise == 1e-2
        assert config.wavelet_level == 3
        assert config.wavelet_threshold == 1.0
        assert config.ema_alpha == 0.3
        assert config.extra_params == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FilterConfig(
            filter_type=FilterType.KALMAN,
            window_size=10,
            kalman_process_noise=1e-4,
        )

        assert config.filter_type == FilterType.KALMAN
        assert config.window_size == 10
        assert config.kalman_process_noise == 1e-4


class TestAdaptiveNoiseFilter:
    """Tests for AdaptiveNoiseFilter."""

    @pytest.fixture
    def noisy_signal(self, deterministic_rng):
        """Create noisy signal for testing."""
        t = np.linspace(0, 1, 200)
        clean = np.sin(2 * np.pi * 5 * t)
        noise = deterministic_rng.randn(len(t)) * 0.5
        return clean + noise

    def test_default_initialization(self):
        """Test default initialization."""
        filter_obj = AdaptiveNoiseFilter()
        assert filter_obj.config.filter_type == FilterType.FFT_LOWPASS

    def test_custom_initialization(self):
        """Test initialization with custom config."""
        config = FilterConfig(filter_type=FilterType.KALMAN)
        filter_obj = AdaptiveNoiseFilter(config=config)
        assert filter_obj.config.filter_type == FilterType.KALMAN

    def test_short_signal_passthrough(self):
        """Test that very short signals are passed through unchanged."""
        filter_obj = AdaptiveNoiseFilter()
        short_data = np.array([1.0, 2.0])
        result = filter_obj.apply(short_data)

        np.testing.assert_array_equal(result, short_data)

    def test_fft_lowpass_filter(self, noisy_signal):
        """Test FFT lowpass filter."""
        config = FilterConfig(filter_type=FilterType.FFT_LOWPASS, cutoff_freq=0.3)
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)
        # Filtered signal should be smoother (lower std of differences)
        assert np.std(np.diff(filtered)) < np.std(np.diff(noisy_signal))

    def test_wavelet_filter(self, noisy_signal):
        """Test wavelet denoising filter."""
        config = FilterConfig(filter_type=FilterType.WAVELET, wavelet_level=3)
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)

    def test_wavelet_filter_short_signal(self):
        """Test wavelet filter with very short signal."""
        config = FilterConfig(filter_type=FilterType.WAVELET, wavelet_level=5)
        filter_obj = AdaptiveNoiseFilter(config=config)
        short_signal = np.array([1.0, 2.0, 3.0, 4.0])
        filtered = filter_obj.apply(short_signal)

        assert len(filtered) == len(short_signal)

    def test_kalman_filter(self, noisy_signal):
        """Test Kalman filter."""
        config = FilterConfig(
            filter_type=FilterType.KALMAN,
            kalman_process_noise=1e-5,
            kalman_measurement_noise=1e-2,
        )
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)
        assert filter_obj._kalman_state is not None

    def test_savitzky_golay_filter(self, noisy_signal):
        """Test Savitzky-Golay filter."""
        config = FilterConfig(
            filter_type=FilterType.SAVITZKY_GOLAY,
            window_size=11,
            poly_order=3,
        )
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)

    def test_savitzky_golay_even_window(self, noisy_signal):
        """Test Savitzky-Golay with even window size (auto-corrected)."""
        config = FilterConfig(
            filter_type=FilterType.SAVITZKY_GOLAY,
            window_size=10,  # Even, should be corrected to 11
        )
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)

    def test_adaptive_bandpass_filter(self, noisy_signal):
        """Test adaptive bandpass filter."""
        config = FilterConfig(
            filter_type=FilterType.ADAPTIVE_BANDPASS,
            cutoff_freq=0.3,
        )
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)

    def test_median_filter(self, noisy_signal):
        """Test median filter."""
        config = FilterConfig(
            filter_type=FilterType.MEDIAN,
            window_size=5,
        )
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)

    def test_ema_filter(self, noisy_signal):
        """Test exponential moving average filter."""
        config = FilterConfig(
            filter_type=FilterType.EXPONENTIAL_MOVING_AVERAGE,
            ema_alpha=0.3,
        )
        filter_obj = AdaptiveNoiseFilter(config=config)
        filtered = filter_obj.apply(noisy_signal)

        assert len(filtered) == len(noisy_signal)

    def test_reset_state(self, noisy_signal):
        """Test reset state for Kalman filter."""
        config = FilterConfig(filter_type=FilterType.KALMAN)
        filter_obj = AdaptiveNoiseFilter(config=config)

        filter_obj.apply(noisy_signal)
        assert filter_obj._kalman_state is not None

        filter_obj.reset_state()
        assert filter_obj._kalman_state is None
        assert filter_obj._kalman_covariance == 1.0


class TestMultiStageFilter:
    """Tests for MultiStageFilter."""

    @pytest.fixture
    def noisy_signal(self, deterministic_rng):
        """Create noisy signal for testing."""
        return deterministic_rng.randn(100) + 5.0

    def test_empty_pipeline(self, noisy_signal):
        """Test empty filter pipeline."""
        pipeline = MultiStageFilter()
        result = pipeline.apply(noisy_signal)

        np.testing.assert_array_equal(result, noisy_signal)

    def test_single_stage(self, noisy_signal):
        """Test single stage filter."""
        stages = [FilterConfig(filter_type=FilterType.MEDIAN)]
        pipeline = MultiStageFilter(stages=stages)
        result = pipeline.apply(noisy_signal)

        assert len(result) == len(noisy_signal)

    def test_multi_stage(self, noisy_signal):
        """Test multi-stage filter pipeline."""
        stages = [
            FilterConfig(filter_type=FilterType.MEDIAN, window_size=3),
            FilterConfig(filter_type=FilterType.SAVITZKY_GOLAY, window_size=5),
        ]
        pipeline = MultiStageFilter(stages=stages)
        result = pipeline.apply(noisy_signal)

        assert len(result) == len(noisy_signal)

    def test_add_stage(self, noisy_signal):
        """Test adding a stage to pipeline."""
        pipeline = MultiStageFilter()
        assert len(pipeline.stages) == 0

        pipeline.add_stage(FilterConfig(filter_type=FilterType.KALMAN))
        assert len(pipeline.stages) == 1
        assert len(pipeline.filters) == 1

        result = pipeline.apply(noisy_signal)
        assert len(result) == len(noisy_signal)


class TestRollingStatistics:
    """Tests for compute_rolling_statistics function."""

    def test_basic_rolling_stats(self):
        """Test basic rolling statistics computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stats = compute_rolling_statistics(data, window_size=3)

        assert "rolling_mean" in stats
        assert "rolling_std" in stats
        assert "rolling_min" in stats
        assert "rolling_max" in stats

        assert len(stats["rolling_mean"]) == len(data)
        assert len(stats["rolling_std"]) == len(data)
        assert len(stats["rolling_min"]) == len(data)
        assert len(stats["rolling_max"]) == len(data)

    def test_rolling_stats_window_larger_than_data(self):
        """Test rolling stats when window is larger than data."""
        data = np.array([1.0, 2.0, 3.0])
        stats = compute_rolling_statistics(data, window_size=10)

        assert len(stats["rolling_mean"]) == 3

    def test_rolling_stats_single_element(self):
        """Test rolling stats with single element."""
        data = np.array([5.0])
        stats = compute_rolling_statistics(data, window_size=3)

        assert stats["rolling_mean"][0] == 5.0


class TestTemporalLagFeatures:
    """Tests for compute_temporal_lag_features function."""

    def test_basic_lag_features(self):
        """Test basic lag feature computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        features = compute_temporal_lag_features(data)

        assert "lag_1" in features
        assert "lag_2" in features
        assert "diff_1" in features
        assert "diff_2" in features

    def test_custom_lags(self):
        """Test with custom lag values."""
        data = np.arange(20, dtype=float)
        features = compute_temporal_lag_features(data, lags=[1, 3, 7])

        assert "lag_1" in features
        assert "lag_3" in features
        assert "lag_7" in features
        assert "lag_2" not in features

    def test_lag_larger_than_data(self):
        """Test when lag is larger than data length."""
        data = np.array([1.0, 2.0, 3.0])
        features = compute_temporal_lag_features(data, lags=[1, 5, 10])

        assert "lag_1" in features
        assert "lag_5" not in features
        assert "lag_10" not in features


class TestInteractionFeatures:
    """Tests for compute_interaction_features function."""

    def test_basic_interactions(self):
        """Test basic interaction feature computation."""
        features_dict = {
            "feature_a": np.array([1.0, 2.0, 3.0]),
            "feature_b": np.array([4.0, 5.0, 6.0]),
        }
        interactions = compute_interaction_features(features_dict)

        assert "feature_a_x_feature_b" in interactions
        assert "feature_a_corr_feature_b" in interactions

    def test_interactions_product(self):
        """Test that product interaction is correct."""
        features_dict = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 4.0]),
        }
        interactions = compute_interaction_features(features_dict)

        expected = np.array([3.0, 8.0])
        np.testing.assert_array_almost_equal(interactions["a_x_b"], expected)

    def test_interactions_mismatched_shapes(self):
        """Test interactions with mismatched shapes."""
        features_dict = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([4.0, 5.0]),
        }
        interactions = compute_interaction_features(features_dict)

        # Should skip mismatched shapes
        assert "a_x_b" not in interactions

    def test_interactions_single_feature(self):
        """Test interactions with single feature."""
        features_dict = {
            "a": np.array([1.0, 2.0, 3.0]),
        }
        interactions = compute_interaction_features(features_dict)

        # No interactions possible with single feature
        assert len(interactions) == 0

    def test_interactions_multiple_features(self):
        """Test interactions with multiple features."""
        features_dict = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([4.0, 5.0, 6.0]),
            "c": np.array([7.0, 8.0, 9.0]),
        }
        interactions = compute_interaction_features(features_dict)

        # Should have interactions for all pairs
        assert "a_x_b" in interactions
        assert "a_x_c" in interactions
        assert "b_x_c" in interactions
