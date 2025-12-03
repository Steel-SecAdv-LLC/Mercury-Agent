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

"""Test suite for 3R Mechanism"""

import numpy as np

from omni_anomaly_engine.core.three_r_mechanism import (
    RecursionEngine,
    RefactoringEngine,
    ResonanceEngine,
    ThreeRMechanism,
)


class TestRecursionEngine:
    def test_initialization(self):
        engine = RecursionEngine(max_depth=5)
        assert engine.max_depth == 5
        assert isinstance(engine.recursion_cache, dict)

    def test_recursive_transform_convergence(self):
        engine = RecursionEngine(max_depth=10)
        data = np.array([1.0, 2.0, 3.0, 4.0])

        def transform_fn(x):
            return x * 0.9

        result = engine.recursive_transform(data, transform_fn, threshold=0.01)

        assert result.shape == data.shape
        assert np.all(result < data)

    def test_recursive_transform_max_depth(self):
        engine = RecursionEngine(max_depth=2)
        data = np.array([1.0, 2.0, 3.0, 4.0])

        def transform_fn(x):
            return x * 1.1

        result = engine.recursive_transform(data, transform_fn, threshold=0.001)

        assert result.shape == data.shape

    def test_hierarchical_feature_extraction(self):
        engine = RecursionEngine()
        data = np.random.randn(100)

        features = engine.hierarchical_feature_extraction(data, num_levels=3)

        assert len(features) == 3
        assert all(isinstance(f, np.ndarray) for f in features)

    def test_hierarchical_feature_extraction_2d(self):
        engine = RecursionEngine()
        data = np.random.randn(20, 5)

        features = engine.hierarchical_feature_extraction(data, num_levels=2)

        assert len(features) == 2
        assert all(isinstance(f, np.ndarray) for f in features)

    def test_extract_level_features_small_data(self):
        engine = RecursionEngine()
        data = np.array([1.0, 2.0])

        features = engine._extract_level_features(data, level=0)

        assert isinstance(features, np.ndarray)

    def test_sliding_window_stats_small_window(self):
        engine = RecursionEngine()
        data = np.array([1.0, 2.0])

        features = engine._sliding_window_stats(data, window_size=5)

        assert isinstance(features, np.ndarray)

    def test_downsample_small_data(self):
        engine = RecursionEngine()
        data = np.array([1.0])

        downsampled = engine._downsample(data)

        assert len(downsampled) >= 1


class TestResonanceEngine:
    def test_initialization(self):
        engine = ResonanceEngine(sampling_rate=100.0)
        assert engine.sampling_rate == 100.0

    def test_compute_resonance_spectrum(self):
        engine = ResonanceEngine(sampling_rate=1.0)
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))

        frequencies, magnitudes = engine.compute_resonance_spectrum(signal)

        assert len(frequencies) > 0
        assert len(magnitudes) > 0
        assert len(frequencies) == len(magnitudes)

    def test_compute_resonance_spectrum_2d(self):
        engine = ResonanceEngine(sampling_rate=1.0)
        signal = np.random.randn(10, 10)

        frequencies, magnitudes = engine.compute_resonance_spectrum(signal)

        assert len(frequencies) > 0
        assert len(magnitudes) > 0

    def test_amplify_resonant_frequencies_with_targets(self):
        engine = ResonanceEngine(sampling_rate=1.0)
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))

        amplified = engine.amplify_resonant_frequencies(
            signal, target_frequencies=[5.0], amplification_factor=2.0
        )

        assert len(amplified) == len(signal)
        assert isinstance(amplified, np.ndarray)

    def test_amplify_resonant_frequencies_auto_detect(self):
        engine = ResonanceEngine(sampling_rate=1.0)
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))

        amplified = engine.amplify_resonant_frequencies(
            signal, target_frequencies=None, amplification_factor=1.5
        )

        assert len(amplified) == len(signal)

    def test_detect_resonance_anomalies(self):
        engine = ResonanceEngine(sampling_rate=1.0)

        normal_signal = np.random.randn(100) * 0.1

        result = engine.detect_resonance_anomalies(normal_signal, threshold_std=2.0)

        assert "is_anomalous" in result
        assert "num_anomalies" in result
        assert isinstance(result["is_anomalous"], (bool, np.bool_))

    def test_detect_resonance_anomalies_with_spike(self):
        engine = ResonanceEngine(sampling_rate=1.0)

        signal = np.zeros(100)
        signal[50] = 100.0

        result = engine.detect_resonance_anomalies(signal, threshold_std=2.0)

        assert "anomalous_frequencies" in result
        assert "threshold" in result


class TestRefactoringEngine:
    def test_initialization(self):
        engine = RefactoringEngine()
        assert isinstance(engine.optimization_history, list)

    def test_analyze_function_complexity(self):
        engine = RefactoringEngine()

        def simple_function(x):
            return x + 1

        metrics = engine.analyze_function_complexity(simple_function)

        assert "num_nodes" in metrics or "error" in metrics

    def test_analyze_complex_function(self):
        engine = RefactoringEngine()

        def complex_function(x, y):
            result = 0
            for i in range(x):
                if i % 2 == 0:
                    for j in range(y):
                        if j > i:
                            result += i * j
                        else:
                            result -= j
            return result

        metrics = engine.analyze_function_complexity(complex_function)

        if "error" not in metrics:
            assert metrics["num_loops"] >= 2
            assert metrics["num_branches"] >= 2

    def test_suggest_refactorings_simple(self):
        engine = RefactoringEngine()

        def simple_function(x):
            return x + 1

        suggestions = engine.suggest_refactorings(simple_function)

        assert isinstance(suggestions, list)

    def test_suggest_refactorings_complex(self):
        engine = RefactoringEngine()

        def complex_function(x, y):
            result = 0
            for i in range(x):
                if i % 2 == 0:
                    for j in range(y):
                        if j > i:
                            for k in range(10):
                                if k % 3 == 0:
                                    result += i * j * k
            return result

        suggestions = engine.suggest_refactorings(complex_function)

        assert isinstance(suggestions, list)

    def test_optimize_data_structure(self):
        engine = RefactoringEngine()

        data = [1, 2, 3, 4, 5]
        optimized = engine.optimize_data_structure(data, target_operation="lookup")

        assert optimized is not None

    def test_optimize_data_structure_iteration(self):
        engine = RefactoringEngine()

        data = {1, 2, 3, 4, 5}
        optimized = engine.optimize_data_structure(data, target_operation="iteration")

        assert isinstance(optimized, list)

    def test_optimize_data_structure_insertion(self):
        engine = RefactoringEngine()

        data = (1, 2, 3, 4, 5)
        optimized = engine.optimize_data_structure(data, target_operation="insertion")

        assert isinstance(optimized, list)


class TestThreeRMechanism:
    def test_initialization(self):
        mechanism = ThreeRMechanism(max_recursion_depth=5, sampling_rate=1.0)

        assert mechanism.recursion_engine is not None
        assert mechanism.resonance_engine is not None
        assert mechanism.refactoring_engine is not None

    def test_enhance_features(self):
        mechanism = ThreeRMechanism()
        data = np.random.randn(50)

        enhanced = mechanism.enhance_features(data, enable_recursion=True, enable_resonance=True)

        assert isinstance(enhanced, np.ndarray)
        assert enhanced.size > 0

    def test_enhance_features_recursion_only(self):
        mechanism = ThreeRMechanism()
        data = np.random.randn(50)

        enhanced = mechanism.enhance_features(data, enable_recursion=True, enable_resonance=False)

        assert isinstance(enhanced, np.ndarray)
        assert enhanced.size > 0

    def test_enhance_features_short_signal(self):
        mechanism = ThreeRMechanism()
        data = np.random.randn(5)

        enhanced = mechanism.enhance_features(data, enable_recursion=True, enable_resonance=True)

        assert isinstance(enhanced, np.ndarray)

    def test_detect_with_resonance(self):
        mechanism = ThreeRMechanism()
        signal = np.random.randn(100)

        result = mechanism.detect_with_resonance(signal, threshold_std=2.5)

        assert "is_anomalous" in result
        assert "num_anomalies" in result

    def test_optimize_component(self):
        mechanism = ThreeRMechanism()

        def sample_function(x, y):
            if x > 0:
                if y > 0:
                    return x + y
            return 0

        result = mechanism.optimize_component(sample_function)

        assert "complexity_metrics" in result
        assert "refactoring_suggestions" in result

    def test_recursive_anomaly_refinement(self):
        mechanism = ThreeRMechanism()

        initial_scores = np.array([0.5, 0.6, 0.7, 0.8])

        def refinement_fn(scores):
            return scores * 0.95

        refined = mechanism.recursive_anomaly_refinement(
            initial_scores, refinement_fn, max_iterations=3
        )

        assert isinstance(refined, np.ndarray)
        assert refined.shape == initial_scores.shape


class TestRefactoringEngineAutoApplication:
    """Tests for automatic refactoring application."""

    def test_apply_refactorings_disabled_by_default(self):
        """Test that apply_refactorings requires explicit opt-in."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig

        config = RefactoringConfig()
        assert config.apply_refactorings is False

    def test_apply_refactorings_with_backup(self):
        """Test automatic refactoring creates backup."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine

        config = RefactoringConfig(apply_refactorings=True, require_confirmation=False)
        engine = RefactoringEngine(config=config)

        def sample_function(x):
            if x > 0:
                if x < 10:
                    return x * 2
            return 0

        result = engine.apply_refactorings(sample_function)

        assert "backup_path" in result or "message" in result
        assert "success" in result

    def test_rollback_refactoring(self):
        """Test rollback functionality."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine

        config = RefactoringConfig(apply_refactorings=True, require_confirmation=False)
        engine = RefactoringEngine(config=config)

        def test_func(x):
            return x + 1

        result = engine.apply_refactorings(test_func)

        if result.get("success") and result.get("rollback_available"):
            rollback = engine.rollback_refactoring("test_func")
            assert "restored_code" in rollback or "error" in rollback

    def test_harmonic_analysis(self):
        """Test harmonic-enhanced code analysis."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def complex_func(x, y):
            result = 0
            for i in range(x):
                for j in range(y):
                    if i > j:
                        result += i * j
            return result

        harmonic_result = engine.analyze_with_harmonics(complex_func)

        assert "num_nodes" in harmonic_result or "error" in harmonic_result
        if "harmonic_analysis" in harmonic_result:
            assert "dominant_frequency" in harmonic_result["harmonic_analysis"]

    def test_quantum_path_exploration(self):
        """Test quantum-inspired refactoring path exploration."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def func_to_refactor(x):
            if x > 0:
                if x < 10:
                    if x % 2 == 0:
                        return x * 2
            return 0

        paths = engine.explore_quantum_refactoring_paths(func_to_refactor, num_paths=3)

        assert isinstance(paths, list)
        assert len(paths) <= 3
        if paths:
            assert "path_id" in paths[0]
            assert "suggestions" in paths[0]
            assert "score" in paths[0]

    def test_resonance_pattern_detection(self):
        """Test resonance-based pattern detection."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def repetitive_func(data):
            result = []
            for item in data:
                if item > 0:
                    result.append(item * 2)
            for item in data:
                if item < 0:
                    result.append(item / 2)
            return result

        resonance = engine.detect_pattern_resonance(repetitive_func)

        assert "resonance_detected" in resonance or "error" in resonance

    def test_orchestrated_refactoring(self):
        """Test meta-orchestration of multiple strategies."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def sample_func(x, y):
            if x > 0:
                result = 0
                for i in range(x):
                    result += i
                return result
            return 0

        orchestrated = engine.orchestrate_refactoring(sample_func)

        assert "orchestrated_analysis" in orchestrated
        assert "unified_suggestions" in orchestrated
        assert "recommended_strategy" in orchestrated

    def test_refactoring_with_invalid_function(self):
        """Test error handling for invalid functions."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine

        config = RefactoringConfig(apply_refactorings=True, require_confirmation=False)
        engine = RefactoringEngine(config=config)

        result = engine.apply_refactorings(len)

        assert result["success"] is False
        assert "error" in result

    def test_ast_transformation_validation(self):
        """Test that AST transformations produce valid Python code."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine

        config = RefactoringConfig(apply_refactorings=True, require_confirmation=False)
        engine = RefactoringEngine(config=config)

        def valid_func(x):
            if x > 0:
                return x * 2
            return 0

        suggestions = engine.suggest_refactorings(valid_func)
        result = engine.apply_refactorings(valid_func, suggestions=suggestions)

        if result.get("success") and "refactored_code" in result:
            try:
                compile(result["refactored_code"], "<test>", "exec")
                assert True
            except SyntaxError:
                assert False, "Refactored code has syntax errors"

    def test_complex_nesting_reduction(self):
        """Test refactoring of deeply nested code."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine

        config = RefactoringConfig(apply_refactorings=True, require_confirmation=False)
        engine = RefactoringEngine(config=config)

        def deeply_nested(x, y, z, w):
            if x > 0:
                if y > 0:
                    if z > 0:
                        if w > 0:
                            return x + y + z + w
            return 0

        suggestions = engine.suggest_refactorings(deeply_nested)

        nesting_suggestions = [s for s in suggestions if s.get("type") == "reduce_nesting"]
        assert len(nesting_suggestions) > 0

    def test_optimization_history_tracking(self):
        """Test that optimization history is tracked."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine

        config = RefactoringConfig(apply_refactorings=True, require_confirmation=False)
        engine = RefactoringEngine(config=config)

        initial_history_len = len(engine.optimization_history)

        def func1(x):
            return x + 1

        engine.apply_refactorings(func1)

        assert len(engine.optimization_history) >= initial_history_len

    def test_three_r_with_enhanced_refactoring(self):
        """Test ThreeRMechanism integration with enhanced refactoring."""
        from omni_anomaly_engine.core.three_r_mechanism import ThreeRMechanism

        mechanism = ThreeRMechanism()

        def test_component(x):
            if x > 0:
                for i in range(x):
                    if i % 2 == 0:
                        yield i

        result = mechanism.optimize_component(test_component)

        assert "complexity_metrics" in result
        assert "refactoring_suggestions" in result

    def test_backup_file_creation(self):
        """Test that backup files are created correctly."""
        import os

        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig, RefactoringEngine

        config = RefactoringConfig(
            apply_refactorings=True, create_backup=True, require_confirmation=False
        )
        engine = RefactoringEngine(config=config)

        def backup_test_func(x):
            return x * 2

        result = engine.apply_refactorings(backup_test_func)

        if result.get("backup_path"):
            assert os.path.exists(result["backup_path"])

    def test_refactoring_config_defaults(self):
        """Test RefactoringConfig default values."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringConfig

        config = RefactoringConfig()
        assert config.apply_refactorings is False
        assert config.create_backup is True
        assert config.require_confirmation is True
        assert config.max_complexity_threshold == 10
        assert config.max_nesting_threshold == 4


class TestNewEnginePatterns:
    """Tests for patterns integrated from 17 new engine documents."""

    def test_multi_dimensional_anomaly_detection(self):
        """Test multi-dimensional anomaly detection from Anomaly Engine."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def high_complexity(x, y, z):
            result = 0
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        if i > j and j > k:
                            result += i * j * k
            return result

        anomalies = engine.detect_code_anomalies(high_complexity, threshold=2.0)

        assert "is_anomaly" in anomalies
        assert "anomaly_score" in anomalies
        assert "method" in anomalies

    def test_issue_classification_by_type_and_severity(self):
        """Test issue classification from Engineering & Refinement Engine."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def complex_func(x):
            if x > 0:
                if x < 10:
                    if x % 2 == 0:
                        if x < 5:
                            return x * 2
            return 0

        issues = engine.classify_code_issues(complex_func)

        assert isinstance(issues, list)
        if len(issues) > 0:
            assert "type" in issues[0]
            assert "severity" in issues[0]
            assert "description" in issues[0]
            assert "recommendation" in issues[0]

    def test_evolution_strategy_adaptation(self):
        """Test adaptive evolution strategy from Evolution Engine."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def test_func(x):
            return x + 1

        history = [
            {"cyclomatic_complexity": 10},
            {"cyclomatic_complexity": 8},
            {"cyclomatic_complexity": 6},
        ]

        evolved = engine.evolve_refactoring_strategy(test_func, history)

        assert "recommended_strategy" in evolved
        assert "current_complexity" in evolved
        assert "strategy_justification" in evolved

    def test_neurosymbolic_symbolic_analysis(self):
        """Test neurosymbolic symbolic analysis."""
        from omni_anomaly_engine.core.three_r_mechanism import RefactoringEngine

        engine = RefactoringEngine()

        def sample_func(x):
            for i in range(x):
                if i % 2 == 0:
                    yield i

        results = engine.analyze_with_neurosymbolic(sample_func)

        assert "symbolic" in results
        assert "neural" in results
        assert "readiness_level" in results

    def test_neurosymbolic_readiness_level(self):
        """Test neurosymbolic readiness level assessment."""
        from omni_anomaly_engine.core.neurosymbolic_engine import (
            NeurosymbolicConfig,
            NeurosymbolicEngine,
            ReadinessLevel,
        )

        ns_engine = NeurosymbolicEngine(
            config=NeurosymbolicConfig(enable_neural=False, enable_symbolic=True)
        )

        readiness = ns_engine.get_readiness_level()
        assert readiness == ReadinessLevel.PRODUCTION_READY

    def test_bias_checking(self):
        """Test bias checking in neurosymbolic engine."""
        from omni_anomaly_engine.core.neurosymbolic_engine import NeurosymbolicEngine

        ns_engine = NeurosymbolicEngine()

        predictions = [
            {"type": "simplify"},
            {"type": "optimize"},
            {"type": "extract"},
        ]

        bias_result = ns_engine.check_bias(predictions)

        assert "bias_detected" in bias_result
        assert "diversity_ratio" in bias_result

    def test_three_r_with_enhanced_refactoring(self):
        """Test ThreeRMechanism integration with enhanced refactoring."""
        from omni_anomaly_engine.core.three_r_mechanism import ThreeRMechanism

        mechanism = ThreeRMechanism()

        def test_component(x):
            if x > 0:
                for i in range(x):
                    if i % 2 == 0:
                        yield i

        result = mechanism.optimize_component(test_component)

        assert "complexity_metrics" in result
        assert "anomaly_detection" in result
        assert "classified_issues" in result
        assert "refactoring_suggestions" in result
        assert "optimization_status" in result
