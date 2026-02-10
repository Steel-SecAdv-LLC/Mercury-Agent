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

"""Tests for Oceanography Patterns module"""

import numpy as np

from omni_mercury_engine.ocean.oceanography_patterns import OceanographyPatterns, WavePattern


def test_oceanography_initialization():
    """Test oceanography patterns system initialization"""
    system = OceanographyPatterns(
        depth_levels=4, acoustic_sensitivity=0.8, wave_detection_threshold=0.6
    )
    assert system.depth_levels == 4
    assert system.acoustic_sensitivity == 0.8
    assert system.wave_detection_threshold == 0.6
    assert len(system.pattern_history) == 0


def test_acoustic_sensing():
    """Test acoustic sensing analogous to sonar"""
    system = OceanographyPatterns()

    data = np.random.randn(100)

    result = system.acoustic_sensing(data, pulse_frequency=2.0)

    assert "reflection_pattern" in result
    assert "depth_profile" in result
    assert "time_of_flight" in result
    assert "signal_strength" in result
    assert isinstance(result["reflection_pattern"], np.ndarray)


def test_wave_pattern_analysis():
    """Test wave pattern analysis for time-series"""
    system = OceanographyPatterns(wave_detection_threshold=0.3)

    t = np.linspace(0, 10, 200)
    time_series = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 3.0 * t)

    patterns = system.wave_pattern_analysis(time_series)

    assert isinstance(patterns, list)
    if len(patterns) > 0:
        assert isinstance(patterns[0], WavePattern)
        assert patterns[0].frequency > 0
        assert patterns[0].amplitude > 0


def test_depth_based_stratification():
    """Test depth-based hierarchical analysis"""
    system = OceanographyPatterns(depth_levels=3)

    data = np.random.randn(300)

    stratified = system.depth_based_stratification(data)

    assert "surface" in stratified
    assert "mid_level" in stratified
    assert "deep" in stratified
    assert all(isinstance(v, np.ndarray) for v in stratified.values())
    assert all(len(v) == 4 for v in stratified.values())


def test_multi_sensor_fusion():
    """Test multi-sensor fusion inspired by interdisciplinary oceanography"""
    system = OceanographyPatterns()

    sensor_data = {
        "temperature": np.random.randn(100) * 2.0 + 20.0,
        "salinity": np.random.randn(100) * 0.5 + 35.0,
        "pressure": np.random.randn(100) * 5.0 + 100.0,
    }

    fused = system.multi_sensor_fusion(sensor_data)

    assert isinstance(fused, np.ndarray)
    assert len(fused) == 12


def test_multi_sensor_fusion_with_weights():
    """Test multi-sensor fusion with custom weights"""
    system = OceanographyPatterns()

    sensor_data = {"sensor_a": np.array([1.0, 2.0, 3.0]), "sensor_b": np.array([4.0, 5.0, 6.0])}

    weights = {"sensor_a": 0.7, "sensor_b": 0.3}

    fused = system.multi_sensor_fusion(sensor_data, weights)

    assert isinstance(fused, np.ndarray)
    assert len(fused) == 8


def test_systematic_sampling_challenger():
    """Test systematic sampling inspired by HMS Challenger (492 soundings)"""
    system = OceanographyPatterns()

    data = np.random.randn(10000)

    indices, values = system.systematic_sampling(data, num_samples=492)

    assert len(indices) <= 492
    assert len(values) <= 492
    assert len(indices) == len(values)


def test_tidal_pattern_detection():
    """Test tidal pattern detection for periodic signals"""
    system = OceanographyPatterns()

    t = np.linspace(0, 100, 1000)
    tidal_signal = np.sin(2 * np.pi * 0.1 * t)

    result = system.tidal_pattern_detection(tidal_signal)

    assert "detected_period" in result
    assert "pattern_strength" in result
    assert "is_periodic" in result
    assert isinstance(result["is_periodic"], bool)


def test_climate_drift_detection():
    """Test climate drift detection for gradual changes"""
    system = OceanographyPatterns()

    historical = np.random.randn(500) * 1.0 + 10.0
    recent = np.random.randn(500) * 1.2 + 12.0

    drift = system.climate_drift_detection(historical, recent)

    assert "mean_drift" in drift
    assert "std_ratio" in drift
    assert "trend_slope" in drift
    assert "drift_detected" in drift
    assert "distribution_shift" in drift
    assert isinstance(drift["drift_detected"], bool)
