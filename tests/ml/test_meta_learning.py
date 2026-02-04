"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for Meta-Learning Module (arxiv 2508.11957v1 - AI Agents Survey)

Verifies meta-learning components for few-shot anomaly detection:
- MetaLearningAdapter: Unified interface for meta-learning algorithms
- PrototypicalNetworks: Prototype-based classification
- MAML: Model-Agnostic Meta-Learning
- Reptile: Simplified meta-learning
- AnomalyMetaLearner: Mercury Agent integration
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMetaLearningAdapter:
    """Tests for MetaLearningAdapter unified interface."""

    def test_initialization_prototypical(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.PROTOTYPICAL,
            input_dim=64,
            hidden_dim=32,
        )
        assert adapter is not None
        stats = adapter.get_statistics()
        assert stats["algorithm"] == "prototypical"

    def test_initialization_maml(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.MAML,
            input_dim=64,
            hidden_dim=32,
        )
        assert adapter is not None
        stats = adapter.get_statistics()
        assert stats["algorithm"] == "maml"

    def test_initialization_reptile(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.REPTILE,
            input_dim=64,
            hidden_dim=32,
        )
        assert adapter is not None
        stats = adapter.get_statistics()
        assert stats["algorithm"] == "reptile"

    def test_few_shot_adaptation(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.PROTOTYPICAL,
            input_dim=10,
            hidden_dim=8,
        )

        # Create few-shot support set
        support_set = {
            "normal": np.random.randn(5, 10),  # 5 examples of normal
            "anomaly": np.random.randn(3, 10) + 2,  # 3 examples of anomaly
        }

        # Adapt to support set
        adapter.adapt(support_set)

        # Query with new samples
        query_samples = np.random.randn(10, 10)
        predictions = adapter.predict(query_samples)

        assert len(predictions) == 10
        assert all(p in ["normal", "anomaly"] for p in predictions)

    def test_meta_training(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.REPTILE,
            input_dim=10,
            hidden_dim=8,
        )

        # Create meta-training tasks
        tasks = []
        for _ in range(5):
            task = {
                "support": {
                    "class_0": np.random.randn(3, 10),
                    "class_1": np.random.randn(3, 10) + 1,
                },
                "query": {
                    "class_0": np.random.randn(2, 10),
                    "class_1": np.random.randn(2, 10) + 1,
                },
            }
            tasks.append(task)

        # Meta-train
        metrics = adapter.meta_train(tasks, epochs=2)

        assert "loss" in metrics or "accuracy" in metrics

    def test_get_embeddings(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.PROTOTYPICAL,
            input_dim=10,
            hidden_dim=8,
        )

        samples = np.random.randn(5, 10)
        embeddings = adapter.get_embeddings(samples)

        assert embeddings.shape == (5, 8)  # hidden_dim


class TestPrototypicalNetworks:
    """Tests for PrototypicalNetworks."""

    def test_initialization(self):
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks

        proto = PrototypicalNetworks(input_dim=20, embedding_dim=16)
        assert proto is not None

    def test_compute_prototypes(self):
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks

        proto = PrototypicalNetworks(input_dim=10, embedding_dim=8)

        support_set = {
            "class_a": np.random.randn(5, 10),
            "class_b": np.random.randn(5, 10),
        }

        prototypes = proto.compute_prototypes(support_set)

        assert "class_a" in prototypes
        assert "class_b" in prototypes
        assert prototypes["class_a"].shape == (8,)  # embedding_dim

    def test_classify(self):
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks

        proto = PrototypicalNetworks(input_dim=10, embedding_dim=8)

        # Set up support set
        support_set = {
            "normal": np.random.randn(5, 10),
            "anomaly": np.random.randn(5, 10) + 3,  # Shifted distribution
        }
        proto.fit(support_set)

        # Classify query
        query = np.random.randn(10)
        result = proto.classify(query)

        assert "predicted_class" in result
        assert "confidence" in result
        assert "distances" in result

    def test_batch_classify(self):
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks

        proto = PrototypicalNetworks(input_dim=10, embedding_dim=8)

        support_set = {
            "class_0": np.random.randn(3, 10),
            "class_1": np.random.randn(3, 10),
        }
        proto.fit(support_set)

        queries = np.random.randn(20, 10)
        results = proto.batch_classify(queries)

        assert len(results) == 20
        for r in results:
            assert "predicted_class" in r

    def test_distance_metrics(self):
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks

        # Test with different distance metrics
        for metric in ["euclidean", "cosine"]:
            proto = PrototypicalNetworks(
                input_dim=10, embedding_dim=8, distance_metric=metric
            )

            support_set = {
                "a": np.random.randn(3, 10),
                "b": np.random.randn(3, 10),
            }
            proto.fit(support_set)

            query = np.random.randn(10)
            result = proto.classify(query)

            assert result is not None


class TestMAML:
    """Tests for Model-Agnostic Meta-Learning."""

    def test_initialization(self):
        from omni_mercury_engine.ml.meta_learning import MAML

        maml = MAML(
            input_dim=20,
            hidden_dim=16,
            output_dim=2,
            inner_lr=0.01,
            outer_lr=0.001,
        )
        assert maml is not None

    def test_inner_loop_adaptation(self):
        from omni_mercury_engine.ml.meta_learning import MAML

        maml = MAML(input_dim=10, hidden_dim=8, output_dim=2)

        support_x = np.random.randn(5, 10)
        support_y = np.array([0, 0, 1, 1, 0])

        # Perform inner loop updates
        adapted_params = maml.inner_loop_adapt(support_x, support_y, num_steps=3)

        assert adapted_params is not None

    def test_meta_step(self):
        from omni_mercury_engine.ml.meta_learning import MAML

        maml = MAML(input_dim=10, hidden_dim=8, output_dim=2)

        # Create a batch of tasks
        tasks = []
        for _ in range(3):
            task = {
                "support_x": np.random.randn(5, 10),
                "support_y": np.random.randint(0, 2, 5),
                "query_x": np.random.randn(5, 10),
                "query_y": np.random.randint(0, 2, 5),
            }
            tasks.append(task)

        # Perform meta-step
        loss = maml.meta_step(tasks)

        assert isinstance(loss, (int, float))

    def test_predict_after_adaptation(self):
        from omni_mercury_engine.ml.meta_learning import MAML

        maml = MAML(input_dim=10, hidden_dim=8, output_dim=2)

        # Adapt to new task
        support_x = np.random.randn(5, 10)
        support_y = np.array([0, 0, 1, 1, 0])
        maml.adapt(support_x, support_y)

        # Predict on query
        query_x = np.random.randn(3, 10)
        predictions = maml.predict(query_x)

        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)

    def test_first_order_approximation(self):
        from omni_mercury_engine.ml.meta_learning import MAML

        # First-order MAML (FOMAML) for efficiency
        maml = MAML(
            input_dim=10,
            hidden_dim=8,
            output_dim=2,
            first_order=True,
        )

        support_x = np.random.randn(5, 10)
        support_y = np.random.randint(0, 2, 5)

        adapted = maml.inner_loop_adapt(support_x, support_y)
        assert adapted is not None


class TestReptile:
    """Tests for Reptile meta-learning algorithm."""

    def test_initialization(self):
        from omni_mercury_engine.ml.meta_learning import Reptile

        reptile = Reptile(
            input_dim=20,
            hidden_dim=16,
            output_dim=2,
            epsilon=0.1,
        )
        assert reptile is not None

    def test_task_adaptation(self):
        from omni_mercury_engine.ml.meta_learning import Reptile

        reptile = Reptile(input_dim=10, hidden_dim=8, output_dim=2)

        task_x = np.random.randn(10, 10)
        task_y = np.random.randint(0, 2, 10)

        # Run SGD on task
        task_params = reptile.task_training(task_x, task_y, num_steps=5)

        assert task_params is not None

    def test_meta_update(self):
        from omni_mercury_engine.ml.meta_learning import Reptile

        reptile = Reptile(input_dim=10, hidden_dim=8, output_dim=2, epsilon=0.1)

        # Train on multiple tasks
        for _ in range(3):
            task_x = np.random.randn(10, 10)
            task_y = np.random.randint(0, 2, 10)
            reptile.meta_update(task_x, task_y)

        # Should have updated meta-parameters
        stats = reptile.get_statistics()
        assert stats["meta_updates"] == 3

    def test_few_shot_evaluation(self):
        from omni_mercury_engine.ml.meta_learning import Reptile

        reptile = Reptile(input_dim=10, hidden_dim=8, output_dim=2)

        # Few-shot evaluation
        support_x = np.random.randn(5, 10)
        support_y = np.array([0, 0, 1, 1, 0])
        query_x = np.random.randn(10, 10)
        query_y = np.random.randint(0, 2, 10)

        accuracy = reptile.evaluate_few_shot(support_x, support_y, query_x, query_y)

        assert 0 <= accuracy <= 1


class TestAnomalyMetaLearner:
    """Tests for Mercury Agent integration."""

    def test_initialization(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner()
        assert learner is not None
        stats = learner.get_statistics()
        assert stats["anomaly_types_learned"] == 0

    def test_learn_new_anomaly_type(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner(feature_dim=20)

        # Provide examples of a new anomaly type
        examples = {
            "normal": np.random.randn(10, 20),
            "new_attack_type": np.random.randn(5, 20) + 2,
        }

        learner.learn_new_type("new_attack_type", examples)

        stats = learner.get_statistics()
        assert stats["anomaly_types_learned"] == 1

    def test_detect_with_meta_learning(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner(feature_dim=10)

        # Learn from few examples
        examples = {
            "normal": np.random.randn(5, 10),
            "ddos": np.random.randn(3, 10) + 3,
            "sql_injection": np.random.randn(3, 10) - 2,
        }
        learner.fit(examples)

        # Detect new sample
        sample = np.random.randn(10) + 3  # Similar to ddos
        result = learner.detect(sample)

        assert result is not None
        assert "predicted_type" in result
        assert "confidence" in result
        assert "is_anomaly" in result

    def test_batch_detection(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner(feature_dim=10)

        examples = {
            "normal": np.random.randn(10, 10),
            "anomaly": np.random.randn(5, 10) + 2,
        }
        learner.fit(examples)

        # Batch detection
        batch = np.random.randn(20, 10)
        results = learner.batch_detect(batch)

        assert len(results) == 20

    def test_online_adaptation(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner(feature_dim=10)

        # Initial fit
        examples = {
            "normal": np.random.randn(10, 10),
            "type_a": np.random.randn(3, 10) + 1,
        }
        learner.fit(examples)

        # Online adaptation with new examples
        new_examples = {
            "type_b": np.random.randn(3, 10) - 1,
        }
        learner.adapt_online(new_examples)

        stats = learner.get_statistics()
        assert stats["anomaly_types_learned"] >= 2

    def test_confidence_calibration(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner(feature_dim=10, calibrate_confidence=True)

        examples = {
            "normal": np.random.randn(20, 10),
            "anomaly": np.random.randn(10, 10) + 3,
        }
        learner.fit(examples)

        # Confidence should be calibrated
        sample = np.random.randn(10)
        result = learner.detect(sample)

        assert 0 <= result["confidence"] <= 1

    def test_feature_importance(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner(feature_dim=10)

        examples = {
            "normal": np.random.randn(10, 10),
            "anomaly": np.random.randn(5, 10),
        }
        # Make feature 0 discriminative
        examples["anomaly"][:, 0] += 5

        learner.fit(examples)

        importance = learner.get_feature_importance()

        assert len(importance) == 10
        # Feature 0 should be most important
        assert importance[0] >= np.mean(importance)


class TestMetaLearningEpisodes:
    """Tests for episodic meta-learning."""

    def test_episode_generation(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.PROTOTYPICAL,
            input_dim=10,
            hidden_dim=8,
        )

        # Generate episodes from dataset
        dataset = {
            "class_0": np.random.randn(50, 10),
            "class_1": np.random.randn(50, 10),
            "class_2": np.random.randn(50, 10),
        }

        episodes = adapter.generate_episodes(
            dataset,
            n_way=2,  # 2 classes per episode
            k_shot=5,  # 5 support examples per class
            n_query=10,  # 10 query examples per class
            n_episodes=5,
        )

        assert len(episodes) == 5
        for episode in episodes:
            assert "support" in episode
            assert "query" in episode
            assert len(episode["support"]) == 2  # n_way

    def test_n_way_k_shot_evaluation(self):
        from omni_mercury_engine.ml.meta_learning import (
            MetaLearningAdapter,
            MetaLearningAlgorithm,
        )

        adapter = MetaLearningAdapter(
            algorithm=MetaLearningAlgorithm.PROTOTYPICAL,
            input_dim=10,
            hidden_dim=8,
        )

        # Evaluate on N-way K-shot tasks
        dataset = {
            f"class_{i}": np.random.randn(30, 10) + i for i in range(5)
        }

        accuracy = adapter.evaluate_n_way_k_shot(
            dataset,
            n_way=3,
            k_shot=5,
            n_episodes=10,
        )

        assert 0 <= accuracy <= 1


class TestModuleImports:
    """Test that meta-learning module can be imported correctly."""

    def test_import_from_ml(self):
        from omni_mercury_engine.ml import (
            MAML,
            AnomalyMetaLearner,
            MetaLearningAdapter,
            PrototypicalNetworks,
            Reptile,
            create_meta_learner,
        )

        assert MetaLearningAdapter is not None
        assert PrototypicalNetworks is not None
        assert MAML is not None
        assert Reptile is not None
        assert AnomalyMetaLearner is not None
        assert create_meta_learner is not None

    def test_create_meta_learner_factory(self):
        from omni_mercury_engine.ml import create_meta_learner

        learner = create_meta_learner(feature_dim=20)
        assert learner is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_support_set(self):
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks

        proto = PrototypicalNetworks(input_dim=10, embedding_dim=8)

        # Empty support set should raise or handle gracefully
        with pytest.raises((ValueError, KeyError)):
            proto.fit({})

    def test_single_example_per_class(self):
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks

        proto = PrototypicalNetworks(input_dim=10, embedding_dim=8)

        # Single example per class (1-shot)
        support_set = {
            "class_0": np.random.randn(1, 10),
            "class_1": np.random.randn(1, 10),
        }
        proto.fit(support_set)

        query = np.random.randn(10)
        result = proto.classify(query)

        assert result is not None

    def test_high_dimensional_input(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        # Test with high-dimensional features
        learner = AnomalyMetaLearner(feature_dim=1000)

        examples = {
            "normal": np.random.randn(10, 1000),
            "anomaly": np.random.randn(5, 1000),
        }
        learner.fit(examples)

        sample = np.random.randn(1000)
        result = learner.detect(sample)

        assert result is not None

    def test_many_classes(self):
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner

        learner = AnomalyMetaLearner(feature_dim=20)

        # Many anomaly types
        examples = {f"type_{i}": np.random.randn(3, 20) + i for i in range(10)}
        examples["normal"] = np.random.randn(20, 20)

        learner.fit(examples)

        sample = np.random.randn(20) + 5
        result = learner.detect(sample)

        assert result is not None
        assert result["predicted_type"] in examples.keys()
