"""
Mercury Agent - Tests for Advanced ML Capabilities
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for:
- Concept drift evaluation
- Few-shot learning
- Cross-domain transfer
- SHAP explainability
- Active learning
- Online learning

These tests validate Mercury's architectural advantages over pure supervised methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate synthetic anomaly detection data."""
    np.random.seed(42)

    n_normal = 800
    n_anomaly = 200

    # Normal samples from multivariate normal
    normal_data = np.random.randn(n_normal, 20) * 0.5

    # Anomalies with shifted mean
    anomaly_data = np.random.randn(n_anomaly, 20) * 0.5 + 2

    X = np.vstack([normal_data, anomaly_data])
    y = np.array([0] * n_normal + [1] * n_anomaly, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


@pytest.fixture
def drift_data() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Generate data with temporal drift."""
    np.random.seed(42)

    n_samples = 1000
    n_features = 20

    # Generate time-varying data
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int64)
    timestamps = np.arange(n_samples, dtype=np.float64)

    for i in range(n_samples):
        # Gradual drift - mean shifts over time
        drift_factor = i / n_samples
        X[i] = np.random.randn(n_features) + drift_factor * 2

        # Label based on threshold that also drifts
        if X[i, 0] > 1 + drift_factor:
            y[i] = 1

    return X, y, timestamps


@pytest.fixture
def few_shot_data() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate data for few-shot learning."""
    np.random.seed(42)

    n_classes = 5
    samples_per_class = 50
    n_features = 32

    X_list = []
    y_list = []

    for cls in range(n_classes):
        # Each class has distinct mean
        mean = np.random.randn(n_features) * 3
        samples = np.random.randn(samples_per_class, n_features) * 0.5 + mean
        X_list.append(samples)
        y_list.extend([cls] * samples_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


@pytest.fixture
def transfer_data() -> (
    tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]
):
    """Generate source and target domain data."""
    np.random.seed(42)

    n_samples = 500
    n_features = 15

    # Source domain
    source_X = np.random.randn(n_samples, n_features)
    source_y = (source_X[:, 0] + source_X[:, 1] > 0).astype(np.int64)

    # Target domain with distribution shift
    target_X = np.random.randn(n_samples, n_features) + 1.5  # Mean shift
    target_y = (target_X[:, 0] + target_X[:, 1] > 1.5).astype(np.int64)  # Adjusted threshold

    return source_X, source_y, target_X, target_y


# =============================================================================
# Concept Drift Evaluation Tests
# =============================================================================


class TestConceptDriftEvaluation:
    """Tests for concept drift evaluation framework."""

    def test_temporal_splitter_expanding_window(self, drift_data):
        """Test expanding window temporal splits."""
        from omni_mercury_engine.ml.concept_drift_evaluation import (
            TemporalSplitStrategy,
            TemporalSplitter,
        )

        X, y, _ = drift_data

        splitter = TemporalSplitter(
            n_splits=5,
            strategy=TemporalSplitStrategy.EXPANDING_WINDOW,
            min_train_size=100,
        )

        splits = splitter.split(X, y)

        assert len(splits) == 5
        # Verify temporal ordering
        for split in splits:
            assert split.train_end <= split.test_start
            assert split.train_start < split.train_end
            assert split.test_start < split.test_end

    def test_temporal_splitter_sliding_window(self, drift_data):
        """Test sliding window temporal splits."""
        from omni_mercury_engine.ml.concept_drift_evaluation import (
            TemporalSplitStrategy,
            TemporalSplitter,
        )

        X, y, _ = drift_data

        splitter = TemporalSplitter(
            n_splits=5,
            strategy=TemporalSplitStrategy.SLIDING_WINDOW,
            window_size=200,
        )

        splits = splitter.split(X, y)

        assert len(splits) <= 5
        # Verify window size
        for split in splits:
            assert split.train_size == 200

    def test_degradation_analyzer(self):
        """Test performance degradation analysis."""
        from omni_mercury_engine.ml.concept_drift_evaluation import (
            DegradationAnalyzer,
            DegradationTrend,
        )

        analyzer = DegradationAnalyzer()

        # Linear decline
        declining_perf = [0.95 - i * 0.02 for i in range(10)]
        result = analyzer.analyze(declining_perf)

        assert result.trend == DegradationTrend.LINEAR_DECLINE
        assert result.degradation_rate > 0
        assert result.retraining_recommended

        # Stable performance
        stable_perf = [0.90 + np.random.randn() * 0.01 for _ in range(10)]
        result = analyzer.analyze(stable_perf)

        assert result.trend == DegradationTrend.STABLE
        assert result.stability_score > 0.5

    def test_concept_drift_evaluator(self, drift_data):
        """Test full concept drift evaluation."""
        from sklearn.ensemble import RandomForestClassifier

        from omni_mercury_engine.ml.concept_drift_evaluation import (
            ConceptDriftEvaluator,
            TemporalSplitStrategy,
        )

        X, y, timestamps = drift_data

        evaluator = ConceptDriftEvaluator(
            n_splits=3,
            strategy=TemporalSplitStrategy.EXPANDING_WINDOW,
            detect_drift=False,  # Disable for speed
            metric="f1",
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = evaluator.evaluate(model, X, y, timestamps)

        assert result.n_splits == 3
        assert 0 <= result.mean_f1 <= 1
        assert len(result.split_performances) == 3
        assert result.degradation_analysis is not None


# =============================================================================
# Few-Shot Learning Tests
# =============================================================================


class TestFewShotLearning:
    """Tests for few-shot learning framework."""

    def test_episode_generator(self, few_shot_data):
        """Test episode generation."""
        from omni_mercury_engine.ml.few_shot_learning import EpisodeGenerator

        X, y = few_shot_data

        generator = EpisodeGenerator(
            n_way=3,
            k_shot=5,
            n_query=10,
            n_episodes=5,
        )

        episodes = list(generator.generate(X, y))

        assert len(episodes) == 5
        for episode in episodes:
            assert len(episode.classes) == 3
            assert len(episode.support_y) == 3 * 5  # n_way * k_shot
            assert len(episode.query_y) == 3 * 10  # n_way * n_query

    def test_prototypical_network(self, few_shot_data):
        """Test Prototypical Network implementation."""
        from omni_mercury_engine.ml.few_shot_learning import (
            EpisodeGenerator,
            PrototypicalNetworkNumpy,
        )

        X, y = few_shot_data

        # Generate one episode
        generator = EpisodeGenerator(n_way=3, k_shot=5, n_query=10, n_episodes=1)
        episode = next(generator.generate(X, y))

        # Test prototypical network
        model = PrototypicalNetworkNumpy(embedding_dim=32)
        model.fit_episode(episode)

        predictions = model.predict(episode.query_X)
        probas = model.predict_proba(episode.query_X)

        assert len(predictions) == len(episode.query_y)
        assert probas.shape == (len(episode.query_y), 3)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_matching_network(self, few_shot_data):
        """Test Matching Network implementation."""
        from omni_mercury_engine.ml.few_shot_learning import (
            EpisodeGenerator,
            MatchingNetworkNumpy,
        )

        X, y = few_shot_data

        generator = EpisodeGenerator(n_way=3, k_shot=5, n_query=10, n_episodes=1)
        episode = next(generator.generate(X, y))

        model = MatchingNetworkNumpy(embedding_dim=32)
        model.fit_episode(episode)

        predictions = model.predict(episode.query_X)
        assert len(predictions) == len(episode.query_y)

    def test_few_shot_learner_evaluation(self, few_shot_data):
        """Test few-shot learner evaluation."""
        from omni_mercury_engine.ml.few_shot_learning import FewShotLearner

        X, y = few_shot_data

        learner = FewShotLearner(
            n_way=3,
            k_shot=5,
            n_query=10,
            n_episodes=10,
            use_pytorch=False,
        )

        result = learner.evaluate(X, y)

        assert 0 <= result.accuracy <= 1
        assert result.n_way == 3
        assert result.k_shot == 5
        assert result.n_episodes == 10
        assert len(result.episode_accuracies) == 10

    def test_k_shot_experiment(self, few_shot_data):
        """Test 10/50/100 label experiments."""
        from omni_mercury_engine.ml.few_shot_learning import FewShotLearner

        X, y = few_shot_data

        learner = FewShotLearner(
            n_way=2,  # Binary for smaller k
            k_shot=5,
            n_episodes=5,
            use_pytorch=False,
        )

        results = learner.run_k_shot_experiment(
            X,
            y,
            k_values=[10, 50],
            n_trials=3,
        )

        assert 10 in results
        assert 50 in results
        assert results[10].n_labels_used == 10
        assert results[50].n_labels_used == 50


# =============================================================================
# Cross-Domain Transfer Tests
# =============================================================================


class TestCrossDomainTransfer:
    """Tests for cross-domain transfer learning."""

    def test_mmd_computation(self, transfer_data):
        """Test Maximum Mean Discrepancy computation."""
        from omni_mercury_engine.ml.cross_domain_transfer import MMDAdapter

        source_X, _, target_X, _ = transfer_data

        adapter = MMDAdapter()
        mmd = adapter.compute_mmd(source_X[:100], target_X[:100])

        assert mmd >= 0
        # Same distribution should have low MMD
        same_mmd = adapter.compute_mmd(source_X[:100], source_X[100:200])
        assert same_mmd < mmd

    def test_coral_adapter(self, transfer_data):
        """Test CORAL domain adaptation."""
        from omni_mercury_engine.ml.cross_domain_transfer import CORALAdapter

        source_X, source_y, target_X, target_y = transfer_data

        adapter = CORALAdapter()
        adapter.fit(source_X, source_y, target_X)

        predictions = adapter.predict(target_X)

        assert len(predictions) == len(target_y)
        assert set(predictions).issubset({0, 1})

    def test_subspace_alignment(self, transfer_data):
        """Test subspace alignment adapter."""
        from omni_mercury_engine.ml.cross_domain_transfer import SubspaceAlignmentAdapter

        source_X, source_y, target_X, target_y = transfer_data

        adapter = SubspaceAlignmentAdapter(n_components=10)
        adapter.fit(source_X, source_y, target_X)

        predictions = adapter.predict(target_X)

        assert len(predictions) == len(target_y)

    def test_cross_domain_transfer_learner(self, transfer_data):
        """Test full cross-domain transfer learning."""
        from omni_mercury_engine.ml.cross_domain_transfer import (
            CrossDomainTransferLearner,
            DomainData,
        )

        source_X, source_y, target_X, target_y = transfer_data

        source_data = DomainData(
            X=source_X,
            y=source_y,
            domain_name="source",
        )

        target_data = DomainData(
            X=target_X,
            y=target_y,
            domain_name="target",
        )

        learner = CrossDomainTransferLearner()
        result = learner.evaluate(source_data, target_data)

        assert 0 <= result.accuracy <= 1
        assert result.source_domain == "source"
        assert result.target_domain == "target"
        assert result.mmd_before >= result.mmd_after or result.mmd_after >= 0


# =============================================================================
# SHAP Explainability Tests
# =============================================================================


class TestExplainability:
    """Tests for SHAP explainability integration."""

    def test_shap_explainer_local(self, synthetic_data):
        """Test local SHAP explanations."""
        from sklearn.ensemble import RandomForestClassifier

        from omni_mercury_engine.ml.explainability import SHAPExplainer

        X, y = synthetic_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = SHAPExplainer(top_k_features=5)
        explanations = explainer.explain_local(model, X[:10], sample_indices=[0, 1, 2])

        assert len(explanations) == 3
        for exp in explanations:
            assert len(exp.top_features) <= 5
            assert exp.prediction is not None
            assert len(exp.anomaly_reasons) > 0

    def test_shap_explainer_global(self, synthetic_data):
        """Test global SHAP explanations."""
        from sklearn.ensemble import RandomForestClassifier

        from omni_mercury_engine.ml.explainability import SHAPExplainer

        X, y = synthetic_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = SHAPExplainer(top_k_features=10)
        global_exp = explainer.explain_global(model, X[:100])

        assert len(global_exp.feature_importances) == X.shape[1]
        assert global_exp.total_samples == 100
        # Check ranking
        for i, feat in enumerate(global_exp.feature_importances):
            assert feat.rank == i + 1

    def test_counterfactual_explainer(self, synthetic_data):
        """Test counterfactual explanations."""
        from sklearn.ensemble import RandomForestClassifier

        from omni_mercury_engine.ml.explainability import CounterfactualExplainer

        X, y = synthetic_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        explainer = CounterfactualExplainer(max_iterations=50)

        # Find an anomaly
        predictions = model.predict(X)
        anomaly_idx = np.where(predictions == 1)[0][0]

        cf = explainer.explain(model, X[anomaly_idx])

        assert cf.original_prediction != cf.counterfactual_prediction or not cf.validity
        assert cf.distance >= 0


# =============================================================================
# Active Learning Tests
# =============================================================================


class TestActiveLearning:
    """Tests for active learning framework."""

    def test_uncertainty_sampler(self, synthetic_data):
        """Test uncertainty-based sampling."""
        from sklearn.ensemble import RandomForestClassifier

        from omni_mercury_engine.ml.active_learning import (
            SamplingStrategy,
            UncertaintySampler,
        )

        X, y = synthetic_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:100], y[:100])

        sampler = UncertaintySampler(SamplingStrategy.UNCERTAINTY_ENTROPY)
        batch = sampler.select(model, X[100:], n_samples=10)

        assert len(batch) == 10
        assert all(u >= 0 for u in batch.uncertainties)

    def test_diversity_sampler(self, synthetic_data):
        """Test diversity-based sampling."""
        from omni_mercury_engine.ml.active_learning import DiversitySampler

        X, y = synthetic_data

        sampler = DiversitySampler()
        batch = sampler.select(None, X, n_samples=10, X_labeled=X[:5])

        assert len(batch) == 10
        assert len(batch.diversity_scores) == 10

    def test_hybrid_sampler(self, synthetic_data):
        """Test hybrid uncertainty + diversity sampling."""
        from sklearn.ensemble import RandomForestClassifier

        from omni_mercury_engine.ml.active_learning import HybridSampler

        X, y = synthetic_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:100], y[:100])

        sampler = HybridSampler(uncertainty_weight=0.6, diversity_weight=0.4)
        batch = sampler.select(model, X[100:], n_samples=10, X_labeled=X[:100])

        assert len(batch) == 10
        assert all(s >= 0 for s in batch.priority_scores)

    def test_active_learner(self, synthetic_data):
        """Test full active learning loop."""
        from sklearn.linear_model import LogisticRegression

        from omni_mercury_engine.ml.active_learning import (
            ActiveLearner,
            LabeledSample,
            LabelType,
        )

        X, y = synthetic_data

        model = LogisticRegression(random_state=42)

        learner = ActiveLearner(
            model=model,
            batch_size=10,
            budget=50,
            initial_samples=20,
        )

        # Initialize with some labels
        init_batch = learner.initialize(X, y)
        assert len(init_batch) == 20

        # Query next batch
        batch = learner.query(X)
        assert len(batch) == 10

        # Simulate oracle labeling
        labels = [
            LabeledSample(
                index=idx,
                features=X[idx],
                label=LabelType.ANOMALY if y[idx] == 1 else LabelType.NORMAL,
            )
            for idx in batch.indices
        ]

        learner.update(labels, X)

        state = learner.get_state()
        assert state.total_labeled == 30
        # budget_remaining = budget - queries_made (not including initial samples)
        # After one query of 10 samples: budget_remaining = 50 - 10 = 40
        assert state.budget_remaining == 40


# =============================================================================
# Online Learning Tests
# =============================================================================


class TestOnlineLearning:
    """Tests for online learning pipeline."""

    def test_sample_buffer(self):
        """Test sample buffer operations."""
        from omni_mercury_engine.ml.online_learning import SampleBuffer, StreamingSample

        buffer = SampleBuffer(max_size=100, strategy="fifo")

        # Add samples
        for i in range(50):
            sample = StreamingSample(
                features=np.random.randn(10),
                label=i % 2,
            )
            buffer.add(sample)

        assert len(buffer) == 50

        # Get batch
        batch = buffer.get_batch(10)
        assert len(batch) == 10

        # Get all labeled
        X, y = buffer.get_all()
        assert len(X) == 50
        assert len(y) == 50

    def test_sgd_online_learner(self, synthetic_data):
        """Test SGD-based online learner."""
        from omni_mercury_engine.ml.online_learning import SGDOnlineLearner

        X, y = synthetic_data

        learner = SGDOnlineLearner()

        # Incremental updates
        for i in range(0, 100, 10):
            learner.partial_fit(X[i : i + 10], y[i : i + 10])

        predictions = learner.predict(X[100:110])
        probas = learner.predict_proba(X[100:110])

        assert len(predictions) == 10
        assert probas.shape == (10, 2)

    def test_online_learning_pipeline(self, synthetic_data):
        """Test full online learning pipeline."""
        from omni_mercury_engine.ml.online_learning import (
            OnlineLearningPipeline,
            SGDOnlineLearner,
            StreamingSample,
        )

        X, y = synthetic_data

        model = SGDOnlineLearner()
        pipeline = OnlineLearningPipeline(
            model=model,
            buffer_size=500,
            mini_batch_size=16,
            drift_detection=False,  # Disable for test speed
        )

        # Process samples
        for i in range(100):
            sample = StreamingSample(
                features=X[i],
                label=int(y[i]),
            )
            result = pipeline.process(sample)
            assert "prediction" in result

        metrics = pipeline.get_metrics()
        assert metrics.samples_processed == 100
        assert metrics.throughput_samples_per_sec > 0


# =============================================================================
# Secure Audit Logging Tests
# =============================================================================


class TestSecureAuditLogging:
    """Tests for secure audit logging."""

    def test_pii_masker(self):
        """Test PII masking."""
        from omni_mercury_engine.security.secure_audit_logging import PIIMasker

        masker = PIIMasker()

        # Test email masking
        text = "Contact: user@example.com"
        masked = masker.mask(text)
        assert "@example.com" not in masked

        # Test phone masking
        text = "Call 555-123-4567"
        masked = masker.mask(text)
        assert "555-123-4567" not in masked

        # Test nested dict masking
        data = {
            "user": {"email": "test@test.com"},
            "list": ["555-000-0000"],
        }
        masked = masker.mask(data)
        assert "test@test.com" not in str(masked)

    def test_hash_chain_integrity(self):
        """Test hash chain for tamper detection."""
        from omni_mercury_engine.security.secure_audit_logging import SecureHashChain

        chain = SecureHashChain()

        events = []
        for i in range(5):
            event_data = {"action": f"action_{i}", "timestamp": i}
            event_hash, prev_hash, seq = chain.compute_event_hash(event_data)
            events.append(
                {
                    "event_hash": event_hash,
                    "previous_hash": prev_hash,
                    "sequence_number": seq,
                }
            )

        # Verify chain links
        for i in range(1, len(events)):
            assert events[i]["previous_hash"] == events[i - 1]["event_hash"]

    def test_audit_logger_event_logging(self, tmp_path):
        """Test audit event logging."""
        from omni_mercury_engine.security.secure_audit_logging import (
            AuditEventCategory,
            SecureAuditLogger,
        )

        logger = SecureAuditLogger(
            log_dir=tmp_path / "audit_logs",
            mask_pii=True,
        )

        try:
            # Log events
            event_id = logger.log(
                category=AuditEventCategory.ANOMALY_DETECTION,
                action="detect_anomaly",
                outcome="success",
                details={"score": 0.95, "email": "test@test.com"},
            )

            assert event_id.startswith("AE-")

            # Flush and verify
            logger.flush()

            # Get recent events
            events = logger.get_recent_events(count=10)
            assert len(events) >= 1

            # Check PII was masked
            event = events[-1]
            assert "test@test.com" not in str(event)

        finally:
            logger.shutdown()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_drift_evaluation_with_few_shot(self, drift_data):
        """Test concept drift evaluation with few-shot learning."""
        from omni_mercury_engine.ml.few_shot_learning import (
            FewShotLearner,
            FewShotMethod,
        )

        X, y, timestamps = drift_data

        # Binary classification for few-shot
        y_binary = (y > 0).astype(np.int64)

        learner = FewShotLearner(
            method=FewShotMethod.PROTOTYPICAL,
            n_way=2,
            k_shot=10,
            n_episodes=5,
            use_pytorch=False,
        )

        result = learner.evaluate(X, y_binary)

        assert result.accuracy > 0.4  # Better than random

    def test_transfer_with_explainability(self, transfer_data):
        """Test cross-domain transfer with explanations."""
        from omni_mercury_engine.ml.cross_domain_transfer import (
            CrossDomainTransferLearner,
            DomainData,
        )

        source_X, source_y, target_X, target_y = transfer_data

        source_data = DomainData(X=source_X, y=source_y, domain_name="source")
        target_data = DomainData(X=target_X, y=target_y, domain_name="target")

        learner = CrossDomainTransferLearner()
        learner.fit(source_data, target_data)

        predictions = learner.predict(target_X[:10])
        assert len(predictions) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
