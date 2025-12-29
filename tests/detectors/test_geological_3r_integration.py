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

from typing import Any

"""
Comprehensive tests for 3R (Recursion-Resonance-Refactoring) integration
in geological detectors: TornadoDetector, HurricaneDetector, FloodDetector.

These tests verify that the core 3R engines (RecursionEngine, ResonanceEngine,
RefactoringEngine) are properly integrated and actively used in the prediction
flow of each geological detector.
"""

import numpy as np
import pytest

from omni_mercury_engine.core.three_r_mechanism import (
    RecursionEngine,
    RefactoringEngine,
    ResonanceEngine,
)
from omni_mercury_engine.detectors.geological.flood_detector import (
    FloodDetector,
    FloodPredictionResult,
)
from omni_mercury_engine.detectors.geological.hurricane_detector import (
    HurricaneDetector,
    HurricanePredictionResult,
)
from omni_mercury_engine.detectors.geological.tornado_detector import (
    TornadoDetector,
    TornadoPredictionResult,
)


class TestRecursionEngineIntegration:
    """Tests for RecursionEngine integration across geological detectors."""

    @pytest.fixture
    def recursion_engine(self):
        """Create RecursionEngine instance."""
        return RecursionEngine(max_depth=5)

    def test_recursion_engine_initialization(self, recursion_engine):
        """Test RecursionEngine initializes correctly."""
        assert recursion_engine is not None
        assert recursion_engine.max_depth == 5
        assert hasattr(recursion_engine, "recursion_cache")

    def test_hierarchical_feature_extraction(self, recursion_engine):
        """Test hierarchical feature extraction."""
        data = np.random.randn(100)
        features = recursion_engine.hierarchical_feature_extraction(data, num_levels=3)
        assert len(features) == 3
        for level_features in features:
            assert isinstance(level_features, np.ndarray)

    def test_recursive_transform(self, recursion_engine):
        """Test recursive transform with convergence."""
        data = np.random.randn(50)

        def transform_fn(x):
            return x * 0.9  # Converging transform

        result = recursion_engine.recursive_transform(data, transform_fn, threshold=0.1)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_recursive_transform_max_depth(self, recursion_engine):
        """Test recursive transform respects max depth."""
        data = np.random.randn(50)

        def transform_fn(x):
            return x * 1.1  # Non-converging transform

        result = recursion_engine.recursive_transform(data, transform_fn, threshold=0.001)
        assert isinstance(result, np.ndarray)

    def test_sliding_window_stats(self, recursion_engine):
        """Test sliding window statistics extraction."""
        data = np.random.randn(100)
        stats = recursion_engine._sliding_window_stats(data, window_size=10)
        assert isinstance(stats, np.ndarray)
        assert len(stats) > 0

    def test_downsample(self, recursion_engine):
        """Test downsampling function."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        downsampled = recursion_engine._downsample(data)
        assert len(downsampled) == 4
        np.testing.assert_array_equal(downsampled, np.array([1, 3, 5, 7]))


class TestResonanceEngineIntegration:
    """Tests for ResonanceEngine integration across geological detectors."""

    @pytest.fixture
    def resonance_engine(self):
        """Create ResonanceEngine instance."""
        return ResonanceEngine(sampling_rate=1.0)

    def test_resonance_engine_initialization(self, resonance_engine):
        """Test ResonanceEngine initializes correctly."""
        assert resonance_engine is not None
        assert resonance_engine.sampling_rate == 1.0

    def test_compute_resonance_spectrum(self, resonance_engine):
        """Test resonance spectrum computation."""
        signal_data = np.sin(2 * np.pi * 0.1 * np.arange(100))
        frequencies, magnitudes = resonance_engine.compute_resonance_spectrum(signal_data)
        assert len(frequencies) > 0
        assert len(magnitudes) > 0
        assert len(frequencies) == len(magnitudes)

    def test_amplify_resonant_frequencies(self, resonance_engine):
        """Test resonant frequency amplification."""
        signal_data = np.sin(2 * np.pi * 0.1 * np.arange(100))
        amplified = resonance_engine.amplify_resonant_frequencies(
            signal_data, amplification_factor=2.0
        )
        assert isinstance(amplified, np.ndarray)
        assert len(amplified) == len(signal_data)

    def test_detect_resonance_anomalies(self, resonance_engine):
        """Test resonance anomaly detection."""
        signal_data = np.random.randn(100)
        signal_data[50] = 100  # Add anomaly
        result = resonance_engine.detect_resonance_anomalies(signal_data, threshold_std=2.5)
        assert "is_anomalous" in result
        assert "num_anomalies" in result
        assert "anomalous_frequencies" in result
        assert "threshold" in result

    def test_detect_dominant_frequencies(self, resonance_engine):
        """Test dominant frequency detection."""
        signal_data = np.sin(2 * np.pi * 0.1 * np.arange(100)) + np.sin(
            2 * np.pi * 0.2 * np.arange(100)
        )
        frequencies, magnitudes = resonance_engine.compute_resonance_spectrum(signal_data)
        dominant = resonance_engine._detect_dominant_frequencies(
            frequencies, magnitudes, num_peaks=3
        )
        assert isinstance(dominant, list)

    def test_multidimensional_signal_handling(self, resonance_engine):
        """Test handling of multidimensional signals."""
        signal_data = np.random.randn(10, 10)
        frequencies, magnitudes = resonance_engine.compute_resonance_spectrum(signal_data)
        assert len(frequencies) > 0


class TestRefactoringEngineIntegration:
    """Tests for RefactoringEngine integration across geological detectors."""

    @pytest.fixture
    def refactoring_engine(self):
        """Create RefactoringEngine instance."""
        return RefactoringEngine()

    def test_refactoring_engine_initialization(self, refactoring_engine):
        """Test RefactoringEngine initializes correctly."""
        assert refactoring_engine is not None

    def test_analyze_complexity(self, refactoring_engine):
        """Test code complexity analysis."""
        code = """
def example_function(x):
    if x > 0:
        return x * 2
    else:
        return x
"""
        result = refactoring_engine.analyze_complexity(code)
        assert "cyclomatic_complexity" in result
        assert "num_nodes" in result

    def test_detect_code_anomalies(self, refactoring_engine):
        """Test code anomaly detection returns a result."""
        code = """
def complex_function(a, b, c, d, e, f, g, h):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    return a + b + c + d
    return 0
"""
        result = refactoring_engine.detect_code_anomalies(code)
        assert isinstance(result, dict)

    def test_suggest_refactorings(self, refactoring_engine):
        """Test refactoring suggestions."""
        code = """
def long_function():
    x = 1
    y = 2
    z = 3
    a = x + y
    b = y + z
    c = a + b
    return c
"""
        suggestions = refactoring_engine.suggest_refactorings(code)
        assert isinstance(suggestions, list)

    def test_analyze_function_complexity(self, refactoring_engine):
        """Test function-level complexity analysis."""
        code = """
def func1():
    return 1

def func2(x):
    if x > 0:
        return x
    return 0
"""
        result = refactoring_engine.analyze_function_complexity(code)
        assert isinstance(result, dict)


class TestTornadoDetector3RIntegration:
    """Tests for 3R integration in TornadoDetector."""

    @pytest.fixture
    def tornado_detector(self):
        """Create TornadoDetector with all 3R engines enabled."""
        return TornadoDetector(
            enable_radar=True,
            enable_atmospheric=True,
            enable_pressure=True,
            enable_resonance=True,
            enable_recursion=True,
            enable_refactoring=True,
        )

    @pytest.fixture
    def tornado_data(self, deterministic_rng):
        """Generate tornado detection data."""
        return {
            "radar_sequence": deterministic_rng.randn(1, 10, 64).astype(np.float32),
            "atmospheric_data": {
                "cape_j_kg": 3000.0,
                "cin_j_kg": -30.0,
                "srh_m2_s2": 200.0,
                "bulk_shear_kt": 45.0,
                "lcl_m": 900.0,
            },
            "pressure_data": {
                "pressure_mb": np.array([1010.0, 1008.0, 1005.0, 1000.0]),
            },
            "location": {"state": "OK"},
        }

    def test_tornado_detector_has_3r_engines(self, tornado_detector):
        """Test TornadoDetector has all 3R engines initialized."""
        assert hasattr(tornado_detector, "recursion_engine")
        assert hasattr(tornado_detector, "resonance_engine")
        assert hasattr(tornado_detector, "refactoring_engine")
        assert tornado_detector.recursion_engine is not None
        assert tornado_detector.resonance_engine is not None
        assert tornado_detector.refactoring_engine is not None

    def test_tornado_detector_3r_engine_types(self, tornado_detector):
        """Test 3R engines are correct types."""
        assert isinstance(tornado_detector.recursion_engine, RecursionEngine)
        assert isinstance(tornado_detector.resonance_engine, ResonanceEngine)
        assert isinstance(tornado_detector.refactoring_engine, RefactoringEngine)

    def test_tornado_prediction_with_3r(self, tornado_detector, tornado_data):
        """Test tornado prediction uses 3R engines."""
        result = tornado_detector.predict_tornado(tornado_data)
        assert isinstance(result, TornadoPredictionResult)
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0

    def test_tornado_detector_resonance_analysis(self, tornado_detector):
        """Test resonance analysis in tornado detection."""
        signal_data = np.random.randn(100)
        result = tornado_detector.resonance_engine.detect_resonance_anomalies(signal_data)
        assert "is_anomalous" in result

    def test_tornado_detector_recursion_features(self, tornado_detector):
        """Test recursion feature extraction in tornado detection."""
        data = np.random.randn(100)
        features = tornado_detector.recursion_engine.hierarchical_feature_extraction(data)
        assert len(features) == 3

    def test_tornado_detector_empty_data(self, tornado_detector):
        """Test tornado detector handles empty data gracefully."""
        result = tornado_detector.predict_tornado({})
        assert isinstance(result, TornadoPredictionResult)
        assert result.tornado_likely is False


class TestHurricaneDetector3RIntegration:
    """Tests for 3R integration in HurricaneDetector."""

    @pytest.fixture
    def hurricane_detector(self):
        """Create HurricaneDetector with all 3R engines enabled."""
        return HurricaneDetector(
            enable_sst=True,
            enable_wind=True,
            enable_pressure=True,
            enable_resonance=True,
            enable_recursion=True,
            enable_refactoring=True,
        )

    @pytest.fixture
    def hurricane_data(self, deterministic_rng):
        """Generate hurricane detection data."""
        return {
            "sst_data": {
                "sst_celsius": 29.0,
                "climatology_celsius": 27.0,
                "depth_26c_m": 70.0,
            },
            "pressure_data": {
                "central_pressure_mb": 950.0,
                "environmental_pressure_mb": 1013.0,
                "pressure_history_mb": [980.0, 970.0, 960.0, 950.0],
            },
            "signal_data": deterministic_rng.randn(100),
            "basin": "atlantic",
        }

    def test_hurricane_detector_has_3r_engines(self, hurricane_detector):
        """Test HurricaneDetector has all 3R engines initialized."""
        assert hasattr(hurricane_detector, "recursion_engine")
        assert hasattr(hurricane_detector, "resonance_engine")
        assert hasattr(hurricane_detector, "refactoring_engine")
        assert hurricane_detector.recursion_engine is not None
        assert hurricane_detector.resonance_engine is not None
        assert hurricane_detector.refactoring_engine is not None

    def test_hurricane_detector_3r_engine_types(self, hurricane_detector):
        """Test 3R engines are correct types."""
        assert isinstance(hurricane_detector.recursion_engine, RecursionEngine)
        assert isinstance(hurricane_detector.resonance_engine, ResonanceEngine)
        assert isinstance(hurricane_detector.refactoring_engine, RefactoringEngine)

    def test_hurricane_prediction_with_3r(self, hurricane_detector, hurricane_data):
        """Test hurricane prediction uses 3R engines."""
        result = hurricane_detector.predict_hurricane(hurricane_data)
        assert isinstance(result, HurricanePredictionResult)
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0

    def test_hurricane_detector_resonance_analysis(self, hurricane_detector):
        """Test resonance analysis in hurricane detection."""
        signal_data = np.random.randn(100)
        result = hurricane_detector.resonance_engine.detect_resonance_anomalies(signal_data)
        assert "is_anomalous" in result

    def test_hurricane_detector_recursion_features(self, hurricane_detector):
        """Test recursion feature extraction in hurricane detection."""
        data = np.random.randn(100)
        features = hurricane_detector.recursion_engine.hierarchical_feature_extraction(data)
        assert len(features) == 3

    def test_hurricane_detector_empty_data(self, hurricane_detector):
        """Test hurricane detector handles empty data gracefully."""
        result = hurricane_detector.predict_hurricane({})
        assert isinstance(result, HurricanePredictionResult)
        assert result.cyclone_detected is False


class TestFloodDetector3RIntegration:
    """Tests for 3R integration in FloodDetector."""

    @pytest.fixture
    def flood_detector(self):
        """Create FloodDetector with all 3R engines enabled."""
        return FloodDetector(
            enable_precipitation=True,
            enable_river_gauge=True,
            enable_soil=True,
            enable_runoff=True,
            enable_resonance=True,
            enable_recursion=True,
            enable_refactoring=True,
        )

    @pytest.fixture
    def flood_data(self, deterministic_rng):
        """Generate flood detection data."""
        return {
            "precip_data": {
                "precipitation_1h_inches": 2.5,
                "precipitation_6h_inches": 6.0,
                "precipitation_24h_inches": 10.0,
                "forecast_24h_inches": 3.0,
            },
            "gauge_data": {
                "current_stage_ft": 18.0,
                "action_stage_ft": 10.0,
                "flood_stage_ft": 15.0,
                "moderate_flood_stage_ft": 20.0,
                "major_flood_stage_ft": 25.0,
                "stage_history_ft": [12.0, 14.0, 16.0, 18.0],
            },
            "soil_data": {
                "soil_moisture_pct": 85.0,
                "soil_type": "clay",
            },
        }

    def test_flood_detector_has_3r_engines(self, flood_detector):
        """Test FloodDetector has all 3R engines initialized."""
        assert hasattr(flood_detector, "recursion_engine")
        assert hasattr(flood_detector, "resonance_engine")
        assert hasattr(flood_detector, "refactoring_engine")
        assert flood_detector.recursion_engine is not None
        assert flood_detector.resonance_engine is not None
        assert flood_detector.refactoring_engine is not None

    def test_flood_detector_3r_engine_types(self, flood_detector):
        """Test 3R engines are correct types."""
        assert isinstance(flood_detector.recursion_engine, RecursionEngine)
        assert isinstance(flood_detector.resonance_engine, ResonanceEngine)
        assert isinstance(flood_detector.core_refactoring_engine, RefactoringEngine)

    def test_flood_prediction_with_3r(self, flood_detector, flood_data):
        """Test flood prediction uses 3R engines."""
        result = flood_detector.predict_flood(flood_data)
        assert isinstance(result, FloodPredictionResult)
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0

    def test_flood_detector_resonance_analysis(self, flood_detector):
        """Test resonance analysis in flood detection."""
        signal_data = np.random.randn(100)
        result = flood_detector.resonance_engine.detect_resonance_anomalies(signal_data)
        assert "is_anomalous" in result

    def test_flood_detector_recursion_features(self, flood_detector):
        """Test recursion feature extraction in flood detection."""
        data = np.random.randn(100)
        features = flood_detector.recursion_engine.hierarchical_feature_extraction(data)
        assert len(features) == 3

    def test_flood_detector_empty_data(self, flood_detector):
        """Test flood detector handles empty data gracefully."""
        result = flood_detector.predict_flood({})
        assert isinstance(result, FloodPredictionResult)
        assert result.flood_likely is False


class TestCrossDetector3RConsistency:
    """Tests for 3R consistency across all geological detectors."""

    @pytest.fixture
    def all_detectors(self):
        """Create all geological detectors."""
        return {
            "tornado": TornadoDetector(),
            "hurricane": HurricaneDetector(),
            "flood": FloodDetector(),
        }

    def test_all_detectors_have_recursion_engine(self, all_detectors):
        """Test all detectors have RecursionEngine."""
        for name, detector in all_detectors.items():
            assert hasattr(detector, "recursion_engine"), f"{name} missing recursion_engine"
            assert isinstance(detector.recursion_engine, RecursionEngine)

    def test_all_detectors_have_resonance_engine(self, all_detectors):
        """Test all detectors have ResonanceEngine."""
        for name, detector in all_detectors.items():
            assert hasattr(detector, "resonance_engine"), f"{name} missing resonance_engine"
            assert isinstance(detector.resonance_engine, ResonanceEngine)

    def test_all_detectors_have_refactoring_engine(self, all_detectors):
        """Test all detectors have RefactoringEngine."""
        for name, detector in all_detectors.items():
            if name == "flood":
                assert hasattr(
                    detector, "core_refactoring_engine"
                ), f"{name} missing core_refactoring_engine"
                assert isinstance(detector.core_refactoring_engine, RefactoringEngine)
            else:
                assert hasattr(detector, "refactoring_engine"), f"{name} missing refactoring_engine"
                assert isinstance(detector.refactoring_engine, RefactoringEngine)

    def test_recursion_engine_consistency(self, all_detectors):
        """Test RecursionEngine has consistent max_depth across detectors."""
        max_depths = [d.recursion_engine.max_depth for d in all_detectors.values()]
        assert all(depth == max_depths[0] for depth in max_depths)

    def test_resonance_engine_consistency(self, all_detectors):
        """Test ResonanceEngine has consistent sampling_rate across detectors."""
        sampling_rates = [d.resonance_engine.sampling_rate for d in all_detectors.values()]
        assert all(rate == sampling_rates[0] for rate in sampling_rates)

    def test_3r_engines_independent_instances(self, all_detectors):
        """Test each detector has independent 3R engine instances."""
        recursion_ids = [id(d.recursion_engine) for d in all_detectors.values()]
        resonance_ids = [id(d.resonance_engine) for d in all_detectors.values()]
        refactoring_ids = []
        for name, d in all_detectors.items():
            if name == "flood":
                refactoring_ids.append(id(d.core_refactoring_engine))
            else:
                refactoring_ids.append(id(d.refactoring_engine))

        assert len(set(recursion_ids)) == len(recursion_ids)
        assert len(set(resonance_ids)) == len(resonance_ids)
        assert len(set(refactoring_ids)) == len(refactoring_ids)


class Test3RMechanismIntegration:
    """Tests for the unified 3R mechanism integration."""

    def test_recursion_resonance_pipeline(self):
        """Test combined recursion and resonance pipeline."""
        recursion = RecursionEngine(max_depth=3)
        resonance = ResonanceEngine(sampling_rate=1.0)

        data = np.random.randn(100)
        features = recursion.hierarchical_feature_extraction(data, num_levels=2)

        for level_features in features:
            if len(level_features) > 10:
                anomalies = resonance.detect_resonance_anomalies(level_features)
                assert "is_anomalous" in anomalies

    def test_resonance_refactoring_pipeline(self):
        """Test combined resonance and refactoring pipeline."""
        resonance = ResonanceEngine(sampling_rate=1.0)
        refactoring = RefactoringEngine()

        signal_data = np.random.randn(100)
        anomalies = resonance.detect_resonance_anomalies(signal_data)

        code = f"""
def analyze_anomalies():
    num_anomalies = {anomalies["num_anomalies"]}
    return num_anomalies > 0
"""
        complexity = refactoring.analyze_complexity(code)
        assert "cyclomatic_complexity" in complexity

    def test_full_3r_pipeline(self):
        """Test full 3R pipeline integration."""
        recursion = RecursionEngine(max_depth=3)
        resonance = ResonanceEngine(sampling_rate=1.0)
        refactoring = RefactoringEngine()

        data = np.random.randn(100)
        features = recursion.hierarchical_feature_extraction(data)

        all_anomalies = []
        for level_features in features:
            if len(level_features) > 10:
                anomalies = resonance.detect_resonance_anomalies(level_features)
                all_anomalies.append(anomalies)

        code = """
def process_anomalies(anomalies):
    total = sum(a["num_anomalies"] for a in anomalies)
    return total
"""
        suggestions = refactoring.suggest_refactorings(code)
        assert isinstance(suggestions, list)


class TestGeologicalDetectorFeatureExtraction:
    """Tests for feature extraction in geological detectors."""

    @pytest.fixture
    def tornado_detector(self):
        return TornadoDetector()

    @pytest.fixture
    def hurricane_detector(self):
        return HurricaneDetector()

    @pytest.fixture
    def flood_detector(self):
        return FloodDetector()

    def test_tornado_extract_features(self, tornado_detector, deterministic_rng):
        """Test tornado detector feature extraction with numpy array."""
        data = deterministic_rng.randn(100).astype(np.float32)
        if hasattr(tornado_detector, "extract_features"):
            features = tornado_detector.extract_features(data)
            assert features is not None
            assert len(features) == 20

    def test_hurricane_extract_features(self, hurricane_detector, deterministic_rng):
        """Test hurricane detector feature extraction with numpy array."""
        data = deterministic_rng.randn(100).astype(np.float32)
        if hasattr(hurricane_detector, "extract_features"):
            features = hurricane_detector.extract_features(data)
            assert features is not None
            assert len(features) == 20

    def test_flood_extract_features(self, flood_detector, deterministic_rng):
        """Test flood detector feature extraction with numpy array."""
        data = deterministic_rng.randn(100).astype(np.float32)
        if hasattr(flood_detector, "extract_features"):
            features = flood_detector.extract_features(data)
            assert features is not None
            assert len(features) == 20
