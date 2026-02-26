"""
Mercury Agent - Comprehensive Tests for Enhanced Anomaly Detection Modules
Copyright (C) 2025 Steel Security Advisors LLC

Tests for:
- Enhanced Statistical Methods (MAD, LOF, DBSCAN, MCD, CUSUM, GESD)
- Cross-Platform Hub
- Ensemble Coordinator
- Distributed Processor
- Visualization Dashboard
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest


def _plotly_available() -> bool:
    """Check if plotly is available."""
    try:
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================================
# Test Enhanced Statistical Detectors
# ============================================================================


class TestEnhancedStatisticalDetectors:
    """Tests for enhanced statistical anomaly detection methods."""

    @pytest.fixture
    def normal_data(self) -> np.ndarray:
        """Generate normal distribution data."""
        np.random.seed(42)
        return np.random.randn(1000, 5)

    @pytest.fixture
    def data_with_anomalies(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate data with injected anomalies."""
        np.random.seed(42)
        data = np.random.randn(1000, 5)

        # Inject anomalies
        anomaly_indices = [50, 100, 200, 500, 750]
        for idx in anomaly_indices:
            data[idx] = data[idx] * 10  # Make outliers

        labels = np.zeros(1000)
        labels[anomaly_indices] = 1

        return data, labels

    def test_mad_detector_fit(self, normal_data: np.ndarray) -> None:
        """Test MAD detector fitting."""
        from omni_mercury_engine.detectors.enhanced_statistical import MADDetector

        detector = MADDetector(threshold_multiplier=3.5)
        detector.fit(normal_data)

        assert detector.median_ is not None
        assert detector.mad_ is not None
        assert detector._fitted is True

    def test_mad_detector_detect(self, data_with_anomalies: tuple) -> None:
        """Test MAD detector anomaly detection."""
        from omni_mercury_engine.detectors.enhanced_statistical import MADDetector

        data, labels = data_with_anomalies

        detector = MADDetector(threshold_multiplier=3.0)
        detector.fit(data)
        result = detector.detect(data)

        assert "is_anomaly" in result.__dict__
        assert "scores" in result.__dict__
        assert len(result.scores) == len(data)
        assert result.method == "mad"

        # Should detect at least some anomalies
        assert np.sum(result.is_anomaly) > 0

    def test_lof_detector(self, data_with_anomalies: tuple) -> None:
        """Test LOF detector."""
        from omni_mercury_engine.detectors.enhanced_statistical import LOFDetector

        data, labels = data_with_anomalies

        detector = LOFDetector(n_neighbors=20, contamination=0.05)
        detector.fit(data)
        result = detector.detect(data)

        assert result.method == "lof_mercury_native"
        assert len(result.scores) == len(data)
        assert np.all((result.scores >= 0) & (result.scores <= 1))

    def test_dbscan_detector(self, data_with_anomalies: tuple) -> None:
        """Test DBSCAN detector."""
        from omni_mercury_engine.detectors.enhanced_statistical import DBSCANDetector

        data, labels = data_with_anomalies

        detector = DBSCANDetector(min_samples=5, auto_eps=True)
        detector.fit(data)
        result = detector.detect(data)

        assert result.method == "dbscan_mercury_native"
        assert "n_clusters" in result.details
        assert "n_noise" in result.details

    def test_mcd_detector(self, data_with_anomalies: tuple) -> None:
        """Test MCD detector."""
        from omni_mercury_engine.detectors.enhanced_statistical import MCDDetector

        data, labels = data_with_anomalies

        detector = MCDDetector(contamination=0.1)
        detector.fit(data)
        result = detector.detect(data)

        assert result.method == "mcd_mercury_native"
        assert "mahalanobis_distances" in result.details
        assert "p_values" in result.details

    def test_cusum_detector(self, normal_data: np.ndarray) -> None:
        """Test CUSUM detector for sequential data."""
        from omni_mercury_engine.detectors.enhanced_statistical import CUSUMDetector

        # Create time series with shift
        ts_data = normal_data[:, 0].copy()
        ts_data[500:] += 2.0  # Introduce mean shift

        detector = CUSUMDetector(threshold_h=5.0, two_sided=True)
        detector.fit(ts_data[:200])  # Fit on stable portion
        result = detector.detect(ts_data)

        assert result.method == "cusum"
        assert "c_plus" in result.details
        assert "c_minus" in result.details

        # Should detect shift after index 500
        assert np.any(result.is_anomaly[500:])

    def test_gesd_test(self, data_with_anomalies: tuple) -> None:
        """Test GESD outlier detection."""
        from omni_mercury_engine.detectors.enhanced_statistical import GESDTest

        data, labels = data_with_anomalies

        test = GESDTest(max_outliers=10, alpha=0.05)
        result = test.detect(data[:, 0])  # Test on univariate

        assert result.method == "gesd"
        assert "n_outliers" in result.details
        assert "R_values" in result.details

    def test_grubbs_test(self, data_with_anomalies: tuple) -> None:
        """Test Grubbs outlier test."""
        from omni_mercury_engine.detectors.enhanced_statistical import GrubbsTest

        data, labels = data_with_anomalies

        test = GrubbsTest(alpha=0.05, max_outliers=5)
        result = test.detect(data[:, 0])

        assert result.method == "grubbs"
        assert "n_outliers_detected" in result.details

    def test_dynamic_threshold_adapter(self) -> None:
        """Test dynamic threshold adaptation."""
        from omni_mercury_engine.detectors.enhanced_statistical import DynamicThresholdAdapter

        adapter = DynamicThresholdAdapter(
            initial_threshold=0.5,
            target_anomaly_rate=0.05,
        )

        # Simulate score stream
        np.random.seed(42)
        scores = np.random.beta(2, 5, 500)  # Most scores low

        for score in scores:
            adapter.update(score)

        stats = adapter.get_statistics()

        assert "threshold" in stats
        assert "ema_score" in stats
        assert stats["history_size"] == 500

    def test_enhanced_statistical_detector_ensemble(self, data_with_anomalies: tuple) -> None:
        """Test unified enhanced statistical detector."""
        from omni_mercury_engine.detectors.enhanced_statistical import (
            EnhancedStatisticalDetector,
            StatisticalMethod,
        )

        data, labels = data_with_anomalies

        detector = EnhancedStatisticalDetector(
            methods=[
                StatisticalMethod.MAD,
                StatisticalMethod.LOF,
                StatisticalMethod.CUSUM,
            ],
            ensemble_strategy="weighted_average",
            use_dynamic_threshold=True,
        )

        detector.fit(data)
        result = detector.detect(data)

        assert "scores" in result
        assert "method_results" in result
        assert "methods_used" in result
        assert len(result["methods_used"]) > 0


# ============================================================================
# Test Cross-Platform Hub
# ============================================================================


class TestCrossPlatformHub:
    """Tests for cross-platform anomaly detection hub."""

    def test_anomaly_event_creation(self) -> None:
        """Test AnomalyEvent creation from detection result."""
        from omni_mercury_engine.integrations.cross_platform_hub import AnomalyEvent

        result = {
            "scores": np.array([0.8, 0.3, 0.9]),
            "is_anomaly": np.array([True, False, True]),
            "detector_type": "statistical",
        }

        event = AnomalyEvent.from_detection_result(result, source="test", index=0)

        assert event.score == 0.8
        assert event.is_anomaly is True
        assert event.source == "test"
        assert event.severity == "high"
        assert event.detector_type == "statistical"

    def test_data_transformer_prometheus(self) -> None:
        """Test Prometheus format transformation."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            AnomalyEvent,
            DataTransformer,
        )

        event = AnomalyEvent(
            event_id="test123",
            timestamp=datetime.now(UTC),
            source="mercury-agent",
            severity="high",
            score=0.85,
            is_anomaly=True,
        )

        prometheus_output = DataTransformer.to_prometheus(event)

        assert "mercury_anomaly_score" in prometheus_output
        assert "mercury_anomaly_detected" in prometheus_output
        assert "0.85" in prometheus_output

    def test_data_transformer_elastic(self) -> None:
        """Test Elasticsearch format transformation."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            AnomalyEvent,
            DataTransformer,
        )

        event = AnomalyEvent(
            event_id="test123",
            timestamp=datetime.now(UTC),
            source="mercury-agent",
            severity="critical",
            score=0.95,
            is_anomaly=True,
        )

        elastic_output = DataTransformer.to_elastic(event)

        assert "@timestamp" in elastic_output
        assert "mercury" in elastic_output
        assert elastic_output["mercury"]["score"] == 0.95

    def test_data_transformer_opentelemetry(self) -> None:
        """Test OpenTelemetry format transformation."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            AnomalyEvent,
            DataTransformer,
        )

        event = AnomalyEvent(
            event_id="test123",
            timestamp=datetime.now(UTC),
            source="mercury-agent",
            severity="medium",
            score=0.65,
            is_anomaly=True,
        )

        otlp_output = DataTransformer.to_opentelemetry(event)

        assert "resourceMetrics" in otlp_output
        assert len(otlp_output["resourceMetrics"]) > 0

    def test_cross_platform_hub_init(self) -> None:
        """Test CrossPlatformHub initialization."""
        from omni_mercury_engine.integrations.cross_platform_hub import CrossPlatformHub

        hub = CrossPlatformHub(enable_correlation=True, buffer_size=500)

        stats = hub.get_statistics()

        assert stats["registered_platforms"] == 0
        assert stats["buffer_size"] == 500 or "buffered_events" in stats

    def test_platform_registration(self) -> None:
        """Test platform adapter registration."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            CrossPlatformHub,
            PlatformConfig,
            PlatformType,
            ProtocolType,
        )

        hub = CrossPlatformHub()

        config = PlatformConfig(
            platform_type=PlatformType.ELASTIC,
            name="test-elastic",
            endpoint="http://localhost:9200",
            protocol=ProtocolType.REST,
        )

        hub.register_platform("elastic", config)

        status = hub.get_platform_status()

        assert "elastic" in status
        assert status["elastic"]["platform_type"] == PlatformType.ELASTIC


# ============================================================================
# Test Ensemble Coordinator
# ============================================================================


class TestEnsembleCoordinator:
    """Tests for advanced ensemble coordinator."""

    class MockDetector:
        """Mock detector for testing."""

        def __init__(self, bias: float = 0.0):
            self.bias = bias
            self._fitted = False

        def fit(self, data: np.ndarray) -> TestEnsembleCoordinator.MockDetector:
            self._fitted = True
            return self

        def detect(self, data: np.ndarray) -> dict:
            np.random.seed(42)
            scores = np.clip(np.random.rand(len(data)) + self.bias, 0, 1)
            return {"scores": scores, "is_anomaly": scores > 0.5}

    def test_ensemble_coordinator_init(self) -> None:
        """Test ensemble coordinator initialization."""
        from omni_mercury_engine.ml.ensemble_coordinator import (
            EnsembleCoordinator,
            EnsembleStrategy,
        )

        coordinator = EnsembleCoordinator(
            strategy=EnsembleStrategy.DYNAMIC,
            enable_meta_learning=True,
            enable_cascading=True,
        )

        summary = coordinator.get_ensemble_summary()

        assert summary["strategy"] == "dynamic"
        assert summary["meta_learning_enabled"] is True
        assert summary["cascading_enabled"] is True

    def test_detector_registration(self) -> None:
        """Test detector registration with ensemble."""
        from omni_mercury_engine.ml.ensemble_coordinator import EnsembleCoordinator

        coordinator = EnsembleCoordinator()

        detector1 = self.MockDetector(bias=0.1)
        detector2 = self.MockDetector(bias=-0.1)

        coordinator.register_detector("detector1", detector1, weight=1.0, cost_tier=1)
        coordinator.register_detector("detector2", detector2, weight=0.8, cost_tier=2)

        summary = coordinator.get_ensemble_summary()

        assert summary["total_detectors"] == 2

    def test_ensemble_detection(self) -> None:
        """Test ensemble detection."""
        from omni_mercury_engine.ml.ensemble_coordinator import EnsembleCoordinator

        coordinator = EnsembleCoordinator()

        detector1 = self.MockDetector(bias=0.1)
        detector2 = self.MockDetector(bias=0.2)

        coordinator.register_detector("detector1", detector1)
        coordinator.register_detector("detector2", detector2)

        data = np.random.randn(100, 5)
        coordinator.fit(data)

        result = coordinator.detect(data)

        assert len(result.scores) == 100
        assert len(result.is_anomaly) == 100
        assert len(result.active_detectors) > 0
        assert result.strategy_used in [
            "voting",
            "averaging",
            "stacking",
            "cascading",
            "boosting",
            "dynamic",
            "mixture_of_experts",
        ]

    def test_bayesian_weight_optimizer(self) -> None:
        """Test Bayesian weight optimization."""
        from omni_mercury_engine.ml.ensemble_coordinator import BayesianWeightOptimizer

        optimizer = BayesianWeightOptimizer()

        detector_scores = {
            "det1": np.random.rand(100),
            "det2": np.random.rand(100),
        }
        labels = (np.random.rand(100) > 0.5).astype(int)
        current_weights = {"det1": 0.5, "det2": 0.5}

        new_weights = optimizer.optimize(detector_scores, labels, current_weights)

        assert "det1" in new_weights
        assert "det2" in new_weights
        assert abs(sum(new_weights.values()) - 1.0) < 0.01

    def test_gradient_weight_optimizer(self) -> None:
        """Test gradient-based weight optimization."""
        from omni_mercury_engine.ml.ensemble_coordinator import GradientWeightOptimizer

        optimizer = GradientWeightOptimizer(learning_rate=0.1)

        detector_scores = {
            "det1": np.random.rand(100),
            "det2": np.random.rand(100),
        }
        labels = (np.random.rand(100) > 0.5).astype(int)
        current_weights = {"det1": 0.5, "det2": 0.5}

        new_weights = optimizer.optimize(detector_scores, labels, current_weights)

        assert sum(new_weights.values()) > 0

    def test_meta_learner(self) -> None:
        """Test meta-learner for detector selection."""
        from omni_mercury_engine.ml.ensemble_coordinator import MetaLearner

        meta_learner = MetaLearner(n_features=10)

        data = np.random.randn(100, 5)
        detector_names = ["det1", "det2", "det3"]

        weights = meta_learner.predict_weights(data, detector_names)

        assert len(weights) == 3
        assert all(w >= 0 for w in weights.values())


# ============================================================================
# Test Distributed Processor
# ============================================================================


class TestDistributedProcessor:
    """Tests for distributed processing module."""

    class MockDetector:
        """Mock detector for testing."""

        def detect(self, data: np.ndarray) -> dict:
            scores = np.random.rand(len(data))
            return {"scores": scores, "is_anomaly": scores > 0.5}

    def test_chunk_generator(self) -> None:
        """Test chunk generation."""
        from omni_mercury_engine.scaling.distributed_processor import ChunkGenerator

        data = np.random.randn(1000, 5)
        chunk_gen = ChunkGenerator(data, chunk_size=100)

        chunks = list(chunk_gen)

        assert len(chunks) == 10
        assert all(len(chunk_data) <= 100 for _, _, chunk_data in chunks)

    def test_chunk_generator_with_overlap(self) -> None:
        """Test chunk generation with overlap."""
        from omni_mercury_engine.scaling.distributed_processor import ChunkGenerator

        data = np.random.randn(500, 3)
        chunk_gen = ChunkGenerator(data, chunk_size=100, overlap=10)

        chunks = list(chunk_gen)

        # Check overlap handling
        assert len(chunks) >= 5

    def test_distributed_processor_sequential(self) -> None:
        """Test sequential distributed processing."""
        from omni_mercury_engine.scaling.distributed_processor import (
            DistributedProcessor,
            ProcessingConfig,
            ProcessingStrategy,
        )

        detector = self.MockDetector()
        config = ProcessingConfig(
            strategy=ProcessingStrategy.SEQUENTIAL,
            chunk_size=100,
        )

        processor = DistributedProcessor(detector, config)

        data = np.random.randn(500, 5)
        scores, is_anomaly, stats = processor.process(data)

        assert len(scores) == 500
        assert len(is_anomaly) == 500
        assert stats.processed_samples == 500

    def test_distributed_processor_threaded(self) -> None:
        """Test threaded distributed processing."""
        from omni_mercury_engine.scaling.distributed_processor import (
            DistributedProcessor,
            ProcessingConfig,
            ProcessingStrategy,
        )

        detector = self.MockDetector()
        config = ProcessingConfig(
            strategy=ProcessingStrategy.THREADED,
            num_workers=2,
            chunk_size=100,
        )

        processor = DistributedProcessor(detector, config)

        data = np.random.randn(500, 5)
        scores, is_anomaly, stats = processor.process(data)

        assert len(scores) == 500
        assert stats.total_chunks > 0

    def test_processing_stats(self) -> None:
        """Test processing statistics collection."""
        from omni_mercury_engine.scaling.distributed_processor import (
            DistributedProcessor,
            ProcessingConfig,
        )

        detector = self.MockDetector()
        processor = DistributedProcessor(detector, ProcessingConfig(chunk_size=50))

        data = np.random.randn(200, 5)
        _, _, stats = processor.process(data)

        assert stats.total_samples == 200
        assert stats.processed_samples > 0
        assert stats.throughput_samples_per_sec > 0
        assert stats.total_time_seconds > 0

    @pytest.mark.asyncio
    async def test_async_processor(self) -> None:
        """Test async distributed processing."""
        from omni_mercury_engine.scaling.distributed_processor import (
            AsyncProcessor,
            ProcessingConfig,
        )

        detector = self.MockDetector()
        processor = AsyncProcessor(detector, ProcessingConfig(chunk_size=100))

        data = np.random.randn(300, 5)
        scores, is_anomaly, stats = await processor.process(data)

        assert len(scores) == 300
        assert stats.processed_samples > 0

    def test_stream_processor(self) -> None:
        """Test stream processor."""
        from omni_mercury_engine.scaling.distributed_processor import StreamProcessor

        detector = self.MockDetector()
        processor = StreamProcessor(detector, queue_size=100)

        processor.start()

        # Submit some data
        data = np.random.randn(50, 5)
        submitted = processor.submit(data)

        assert submitted is True

        # Get result
        import time

        time.sleep(0.5)  # Wait for processing

        result = processor.get_result(timeout=2.0)

        processor.stop()

        if result is not None:
            scores, is_anomaly, metadata = result
            assert len(scores) == 50


# ============================================================================
# Test Visualization Dashboard
# ============================================================================


class TestVisualizationDashboard:
    """Tests for visualization dashboard module."""

    @pytest.fixture
    def sample_scores(self) -> np.ndarray:
        """Generate sample anomaly scores."""
        np.random.seed(42)
        return np.random.beta(2, 5, 100)

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Generate sample feature data."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    @pytest.mark.skipif(
        not _plotly_available(),
        reason="Plotly not installed",
    )
    def test_time_series_plot(self, sample_scores: np.ndarray) -> None:
        """Test time series plot generation."""
        from omni_mercury_engine.gui.visualization_dashboard import AnomalyVisualizer

        visualizer = AnomalyVisualizer()

        timestamps = [datetime.now(UTC) for _ in range(len(sample_scores))]

        fig = visualizer.time_series_plot(timestamps, sample_scores, threshold=0.5)

        assert fig is not None
        assert len(fig.data) >= 2  # Normal, anomaly traces

    @pytest.mark.skipif(
        not _plotly_available(),
        reason="Plotly not installed",
    )
    def test_feature_importance_plot(self) -> None:
        """Test feature importance plot."""
        from omni_mercury_engine.gui.visualization_dashboard import AnomalyVisualizer

        visualizer = AnomalyVisualizer()

        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        importances = np.array([0.3, 0.5, 0.1, 0.05, 0.05])

        fig = visualizer.feature_importance_plot(feature_names, importances, top_k=5)

        assert fig is not None

    @pytest.mark.skipif(
        not _plotly_available(),
        reason="Plotly not installed",
    )
    def test_correlation_heatmap(self, sample_data: np.ndarray) -> None:
        """Test correlation heatmap."""
        from omni_mercury_engine.gui.visualization_dashboard import AnomalyVisualizer

        visualizer = AnomalyVisualizer()

        fig = visualizer.correlation_heatmap(sample_data)

        assert fig is not None

    @pytest.mark.skipif(
        not _plotly_available(),
        reason="Plotly not installed",
    )
    def test_distribution_plot(self, sample_scores: np.ndarray) -> None:
        """Test score distribution plot."""
        from omni_mercury_engine.gui.visualization_dashboard import AnomalyVisualizer

        visualizer = AnomalyVisualizer()

        fig = visualizer.distribution_plot(sample_scores, threshold=0.3)

        assert fig is not None

    @pytest.mark.skipif(
        not _plotly_available(),
        reason="Plotly not installed",
    )
    def test_3d_visualization(self, sample_data: np.ndarray, sample_scores: np.ndarray) -> None:
        """Test 3D anomaly visualization."""
        from omni_mercury_engine.gui.visualization_dashboard import AnomalyVisualizer

        visualizer = AnomalyVisualizer()

        fig = visualizer.anomaly_scatter_3d(sample_data, sample_scores)

        assert fig is not None

    @pytest.mark.skipif(
        not _plotly_available(),
        reason="Plotly not installed",
    )
    def test_dashboard_builder(self, sample_data: np.ndarray, sample_scores: np.ndarray) -> None:
        """Test dashboard builder."""
        from omni_mercury_engine.gui.visualization_dashboard import DashboardBuilder

        builder = DashboardBuilder()

        builder.add_distribution("Score Distribution", sample_scores)

        figures = builder.build()

        assert "Score Distribution" in figures

    @pytest.mark.skipif(
        not _plotly_available(),
        reason="Plotly not installed",
    )
    def test_quick_dashboard(self, sample_data: np.ndarray, sample_scores: np.ndarray) -> None:
        """Test quick dashboard creation."""
        from omni_mercury_engine.gui.visualization_dashboard import create_quick_dashboard

        dashboard = create_quick_dashboard(
            scores=sample_scores,
            data=sample_data,
            title="Test Dashboard",
        )

        figures = dashboard.build()

        assert len(figures) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the enhanced anomaly detection system."""

    def test_full_pipeline(self) -> None:
        """Test complete detection pipeline."""
        from omni_mercury_engine.detectors.enhanced_statistical import (
            EnhancedStatisticalDetector,
            StatisticalMethod,
        )
        from omni_mercury_engine.ml.ensemble_coordinator import (
            EnsembleCoordinator,
            EnsembleStrategy,
        )

        # Create data
        np.random.seed(42)
        data = np.random.randn(500, 5)
        data[100:110] *= 5  # Inject anomalies

        # Create detector
        detector = EnhancedStatisticalDetector(
            methods=[StatisticalMethod.MAD, StatisticalMethod.LOF],
            ensemble_strategy="weighted_average",
        )
        detector.fit(data)

        # Create ensemble coordinator
        coordinator = EnsembleCoordinator(strategy=EnsembleStrategy.AVERAGING)
        coordinator.register_detector("enhanced_stats", detector)
        coordinator.fit(data)

        # Run detection
        result = coordinator.detect(data)

        assert len(result.scores) == 500
        assert np.sum(result.is_anomaly) > 0  # Should detect anomalies

    def test_cross_platform_integration(self) -> None:
        """Test cross-platform event publishing."""
        from omni_mercury_engine.integrations.cross_platform_hub import (
            AnomalyEvent,
            CrossPlatformHub,
        )

        # Verify hub can be instantiated
        hub = CrossPlatformHub()
        assert hub is not None

        # Create mock detection result
        result = {
            "scores": np.array([0.9, 0.3, 0.8, 0.2]),
            "is_anomaly": np.array([True, False, True, False]),
            "detector_type": "ensemble",
        }

        # Create events
        events = []
        for i in range(len(result["scores"])):
            event = AnomalyEvent.from_detection_result(result, "test", i)
            events.append(event)

        # Verify events
        assert len(events) == 4
        assert events[0].is_anomaly is True
        assert events[1].is_anomaly is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
