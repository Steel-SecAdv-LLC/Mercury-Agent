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
Tests for new modules added in Claude's branch:
- drift.py: Data/model drift detection
- fairness.py: Fairlearn-compatible bias detection
- optimization.py: Efficiency optimizations
- llm_adapter.py: LLM integration for zero-shot anomaly detection
- engine_config.py: Pydantic-based configuration
"""

import numpy as np
import pytest


class TestDriftDetection:
    """Tests for drift detection module."""

    def test_ks_drift_no_drift_identical_distributions(self) -> None:
        """KS detector should report no drift for identical distributions."""
        from omni_mercury_engine.ml.drift import KolmogorovSmirnovDriftDetector

        detector = KolmogorovSmirnovDriftDetector(p_value_threshold=0.05)
        reference = np.random.randn(1000, 5)
        detector.fit(reference)

        result = detector.detect(reference)
        assert not result.is_drift, "Should not detect drift on identical data"
        assert result.p_value > 0.05, "P-value should be high for identical distributions"

    def test_ks_drift_detects_shifted_distribution(self) -> None:
        """KS detector should detect drift when distribution shifts."""
        from omni_mercury_engine.ml.drift import KolmogorovSmirnovDriftDetector

        detector = KolmogorovSmirnovDriftDetector(p_value_threshold=0.05)
        reference = np.random.randn(1000, 3)
        detector.fit(reference)

        shifted = reference + 5.0
        result = detector.detect(shifted)
        assert result.is_drift, "Should detect drift on shifted data"
        assert result.p_value < 0.05, "P-value should be low for shifted distributions"

    def test_psi_drift_no_drift(self) -> None:
        """PSI detector should report no drift for similar distributions."""
        from omni_mercury_engine.ml.drift import PopulationStabilityIndexDetector

        detector = PopulationStabilityIndexDetector()
        reference = np.random.randn(1000, 3)
        detector.fit(reference)

        result = detector.detect(reference)
        assert not result.is_drift, "Should not detect drift on identical data"

    def test_create_drift_detector_factory(self) -> None:
        """Factory function should create appropriate detector."""
        from omni_mercury_engine.ml.drift import create_drift_detector

        detector = create_drift_detector("ks")
        assert detector is not None

        detector = create_drift_detector("psi")
        assert detector is not None

    def test_drift_result_to_dict(self) -> None:
        """DriftResult should serialize to dictionary."""
        from omni_mercury_engine.ml.drift import (
            DriftResult,
            DriftSeverity,
            DriftType,
        )

        result = DriftResult(
            is_drift=True,
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.MEDIUM,
            p_value=0.01,
            test_statistic=0.5,
            threshold=0.05,
        )
        d = result.to_dict()
        assert d["is_drift"] is True
        assert d["drift_type"] == "data_drift"
        assert d["severity"] == "medium"


class TestFairnessAuditing:
    """Tests for fairness auditing module."""

    def test_demographic_parity_equal_groups(self) -> None:
        """Demographic parity should be high for equal prediction rates."""
        from omni_mercury_engine.ml.fairness import FairnessAuditor

        auditor = FairnessAuditor()
        predictions = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = auditor.compute_demographic_parity(predictions, sensitive)
        assert result["parity_score"] == 1.0, "Equal rates should have perfect parity"
        assert result["max_disparity"] == 0.0

    def test_demographic_parity_unequal_groups(self) -> None:
        """Demographic parity should detect disparity in prediction rates."""
        from omni_mercury_engine.ml.fairness import FairnessAuditor

        auditor = FairnessAuditor()
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = auditor.compute_demographic_parity(predictions, sensitive)
        assert result["parity_score"] < 1.0, "Unequal rates should have lower parity"
        assert result["max_disparity"] > 0.0

    def test_disparate_impact_four_fifths_rule(self) -> None:
        """Disparate impact should check 4/5ths rule."""
        from omni_mercury_engine.ml.fairness import FairnessAuditor

        auditor = FairnessAuditor()
        predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

        result = auditor.compute_disparate_impact(predictions, sensitive)
        assert "passes_four_fifths" in result
        assert result["min_ratio"] == 0.0

    def test_fairness_audit_returns_report(self) -> None:
        """Full audit should return FairnessReport."""
        from omni_mercury_engine.ml.fairness import FairnessAuditor

        auditor = FairnessAuditor()
        predictions = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = auditor.audit(predictions, labels, sensitive)
        assert hasattr(report, "overall_fairness_score")
        assert hasattr(report, "is_fair")
        assert 0.0 <= report.overall_fairness_score <= 1.0

    def test_compute_fairness_score_convenience(self) -> None:
        """Convenience function should return score in [0, 1]."""
        from omni_mercury_engine.ml.fairness import compute_fairness_score

        predictions = np.array([1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        score = compute_fairness_score(predictions, None, sensitive)
        assert 0.0 <= score <= 1.0


class TestOptimization:
    """Tests for optimization utilities module."""

    def test_memory_efficient_cache_basic(self) -> None:
        """Cache should store and retrieve values."""
        from omni_mercury_engine.ml.optimization import MemoryEfficientCache

        cache = MemoryEfficientCache(max_size_mb=1.0, max_entries=10)
        cache.put("key1", np.array([1, 2, 3]))
        result = cache.get("key1")
        assert result is not None
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_memory_efficient_cache_eviction(self) -> None:
        """Cache should evict entries when limit reached."""
        from omni_mercury_engine.ml.optimization import MemoryEfficientCache

        cache = MemoryEfficientCache(max_size_mb=0.001, max_entries=2)
        cache.put("key1", np.zeros(100))
        cache.put("key2", np.zeros(100))
        cache.put("key3", np.zeros(100))

        assert cache.get("key3") is not None

    def test_memory_efficient_cache_stats(self) -> None:
        """Cache should track hit/miss statistics."""
        from omni_mercury_engine.ml.optimization import MemoryEfficientCache

        cache = MemoryEfficientCache()
        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("key2")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_parallel_executor_sequential_fallback(self) -> None:
        """ParallelExecutor should work in sequential mode."""
        from omni_mercury_engine.ml.optimization import ParallelExecutor

        executor = ParallelExecutor(n_jobs=1)
        results = executor.map(lambda x: x * 2, [1, 2, 3, 4])
        assert results == [2, 4, 6, 8]

    def test_apply_all_optimizations(self) -> None:
        """apply_all_optimizations should return components dict."""
        from omni_mercury_engine.ml.optimization import (
            OptimizationConfig,
            apply_all_optimizations,
        )

        config = OptimizationConfig(enable_ddp=False, enable_torch_compile=False)
        components = apply_all_optimizations(config=config)

        assert "memory_manager" in components
        assert "cache" in components


class TestLLMAdapter:
    """Tests for LLM adapter module."""

    def test_mock_adapter_available(self) -> None:
        """Mock adapter should always be available."""
        from omni_mercury_engine.models.foundation.llm_adapter import MockLLMAdapter

        adapter = MockLLMAdapter()
        assert adapter.is_available()

    def test_mock_adapter_generates_valid_json(self) -> None:
        """Mock adapter should generate valid JSON response."""
        import json

        from omni_mercury_engine.models.foundation.llm_adapter import MockLLMAdapter

        adapter = MockLLMAdapter()
        response = adapter.generate("test prompt")

        parsed = json.loads(response)
        assert "is_anomaly" in parsed
        assert "anomaly_score" in parsed
        assert "confidence" in parsed

    def test_zero_shot_detector_returns_dict(self) -> None:
        """ZeroShotAnomalyDetector should return fusion-compatible dict."""
        from omni_mercury_engine.models.foundation.llm_adapter import (
            ZeroShotAnomalyDetector,
        )

        detector = ZeroShotAnomalyDetector()
        result = detector.detect("test data")

        assert "anomaly_score" in result
        assert "is_anomaly" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_llm_anomaly_result_to_dict(self) -> None:
        """LLMAnomalyResult should convert to fusion-compatible dict."""
        from omni_mercury_engine.models.foundation.llm_adapter import LLMAnomalyResult

        result = LLMAnomalyResult(
            is_anomaly=True,
            anomaly_score=0.85,
            confidence=0.9,
            explanation="Test",
            category="test",
            raw_response="{}",
        )
        d = result.to_dict()

        assert d["anomaly_score"] == 0.85
        assert d["anomaly_prob"] == 0.85
        assert d["is_anomaly"] is True
        assert d["confidence"] == 0.9

    def test_create_llm_detector_factory(self) -> None:
        """Factory should create detector with mock provider."""
        from omni_mercury_engine.models.foundation.llm_adapter import create_llm_detector

        detector = create_llm_detector(provider="mock")
        assert detector is not None
        assert detector.adapter.is_available()


class TestEngineConfig:
    """Tests for engine configuration module."""

    def test_default_config_creation(self) -> None:
        """Default config should be created with valid defaults."""
        from omni_mercury_engine.core.engine_config import MercuryEngineConfig

        config = MercuryEngineConfig()
        assert config.ethical.benevolence_threshold == 0.99
        assert config.three_r.lyapunov_lambda == 0.25

    def test_domain_specific_thresholds(self) -> None:
        """Domain-specific configs should adjust thresholds."""
        from omni_mercury_engine.core.engine_config import (
            DomainType,
            MercuryEngineConfig,
        )

        cyber_config = MercuryEngineConfig(domain=DomainType.CYBER)
        assert cyber_config.ethical.sigma_sacred_threshold == 0.93

        medical_config = MercuryEngineConfig(domain=DomainType.MEDICAL)
        assert medical_config.ethical.sigma_sacred_threshold == 0.93

    def test_get_ethical_threshold_for_domain(self) -> None:
        """get_ethical_threshold_for_domain should return correct values."""
        from omni_mercury_engine.core.engine_config import (
            DomainType,
            MercuryEngineConfig,
        )

        config = MercuryEngineConfig()
        assert config.get_ethical_threshold_for_domain(DomainType.CYBER) == 0.93
        assert config.get_ethical_threshold_for_domain(DomainType.MEDICAL) == 0.93
        assert config.get_ethical_threshold_for_domain(DomainType.GENERAL) == 0.96

    def test_ethical_weights_normalize(self) -> None:
        """Ethical weights should normalize to sum to 1.0."""
        from omni_mercury_engine.core.engine_config import EthicalConfig

        config = EthicalConfig(fairness_weight=0.8, safety_weight=0.8)
        assert abs(config.fairness_weight + config.safety_weight - 1.0) < 1e-6

    def test_fusion_weights_normalize(self) -> None:
        """Fusion weights should normalize correctly."""
        from omni_mercury_engine.core.engine_config import FusionWeightConfig

        config = FusionWeightConfig()
        weights = config.get_normalized_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_config_to_dict_and_back(self) -> None:
        """Config should serialize and deserialize correctly."""
        from omni_mercury_engine.core.engine_config import MercuryEngineConfig

        config = MercuryEngineConfig()
        d = config.to_dict()
        restored = MercuryEngineConfig.from_dict(d)

        assert restored.ethical.benevolence_threshold == config.ethical.benevolence_threshold
        assert restored.three_r.lyapunov_lambda == config.three_r.lyapunov_lambda

    def test_factory_functions(self) -> None:
        """Factory functions should create domain-specific configs."""
        from omni_mercury_engine.core.engine_config import (
            create_cyber_config,
            create_infrastructure_config,
            create_medical_config,
        )

        cyber = create_cyber_config()
        assert cyber.ethical.sigma_sacred_threshold == 0.93

        medical = create_medical_config()
        assert medical.ethical.sigma_sacred_threshold == 0.93

        infra = create_infrastructure_config()
        assert infra.ethical.benevolence_threshold == 0.995


class TestLazyImports:
    """Tests for lazy import wrappers in __init__.py files."""

    def test_ml_lazy_imports(self) -> None:
        """ML module lazy imports should work."""
        from omni_mercury_engine.ml import (
            FairnessAuditor,
            MemoryEfficientCache,
            ParallelExecutor,
            apply_all_optimizations,
            compute_fairness_score,
            create_drift_detector,
        )

        assert create_drift_detector is not None
        assert compute_fairness_score is not None
        assert FairnessAuditor is not None
        assert apply_all_optimizations is not None
        assert MemoryEfficientCache is not None
        assert ParallelExecutor is not None

    def test_core_lazy_imports(self) -> None:
        """Core module lazy imports should work."""
        from omni_mercury_engine.core import (
            DomainType,
            EthicalConfig,
            FusionWeightConfig,
            MercuryEngineConfig,
            ThreeRConfig,
            get_default_config,
        )

        assert get_default_config is not None
        assert MercuryEngineConfig is not None
        assert EthicalConfig is not None
        assert FusionWeightConfig is not None
        assert ThreeRConfig is not None
        assert DomainType is not None
