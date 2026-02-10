"""
Tests for Ensemble Learning module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.ml.ensemble import (
    DetectorWeightLearner,
    EnsembleConfig,
    EnsembleMethod,
    EnsembleOmniFusionModel,
    MetaLearner,
    VotingEnsemble,
    create_ensemble_model,
)


class TestEnsembleMethod:
    """Tests for EnsembleMethod enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert EnsembleMethod.STACKING.value == "stacking"
        assert EnsembleMethod.BOOSTING.value == "boosting"
        assert EnsembleMethod.BAGGING.value == "bagging"
        assert EnsembleMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert EnsembleMethod.VOTING.value == "voting"


class TestEnsembleConfig:
    """Tests for EnsembleConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EnsembleConfig()
        assert config.method == EnsembleMethod.STACKING
        assert config.num_base_models == 5
        assert config.meta_learner_hidden_dim == 64
        assert config.meta_learner_layers == 2
        assert config.boosting_rounds == 10
        assert config.learning_rate == 0.1
        assert config.dropout == 0.1
        assert config.use_detector_confidence is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = EnsembleConfig(
            method=EnsembleMethod.BOOSTING,
            num_base_models=10,
            meta_learner_hidden_dim=128,
            boosting_rounds=20,
        )
        assert config.method == EnsembleMethod.BOOSTING
        assert config.num_base_models == 10
        assert config.meta_learner_hidden_dim == 128
        assert config.boosting_rounds == 20


class TestMetaLearner:
    """Tests for MetaLearner class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        model = MetaLearner(input_dim=10)
        assert isinstance(model, nn.Module)

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        model = MetaLearner(
            input_dim=20,
            hidden_dim=128,
            output_dim=2,
            num_layers=3,
            dropout=0.2,
        )
        assert isinstance(model, nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        model = MetaLearner(input_dim=10, hidden_dim=32, output_dim=1)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 1)

    def test_forward_batch(self) -> None:
        """Test forward pass with different batch sizes."""
        model = MetaLearner(input_dim=10, hidden_dim=32, output_dim=1)
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 10)
            output = model(x)
            assert output.shape == (batch_size, 1)

    def test_forward_multi_output(self) -> None:
        """Test forward pass with multiple outputs."""
        model = MetaLearner(input_dim=10, hidden_dim=32, output_dim=5)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 5)


class TestDetectorWeightLearner:
    """Tests for DetectorWeightLearner class."""

    def test_init(self) -> None:
        """Test initialization."""
        model = DetectorWeightLearner(num_detectors=5, input_dim=64)
        assert isinstance(model, nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        model = DetectorWeightLearner(num_detectors=5, input_dim=64)
        context = torch.randn(4, 64)
        weights = model(context)
        assert weights.shape == (4, 5)

    def test_weights_sum_to_one(self) -> None:
        """Test that output weights sum to 1 (softmax)."""
        model = DetectorWeightLearner(num_detectors=5, input_dim=64)
        context = torch.randn(4, 64)
        weights = model(context)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_weights_positive(self) -> None:
        """Test that all weights are positive."""
        model = DetectorWeightLearner(num_detectors=5, input_dim=64)
        context = torch.randn(4, 64)
        weights = model(context)
        assert (weights >= 0).all()


class MockBaseModel(nn.Module):
    """Mock base model for testing ensemble."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(64, 1)

    def forward(
        self,
        detector_features: dict[str, torch.Tensor],
        detector_scores: dict[str, torch.Tensor] | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size = next(iter(detector_features.values())).shape[0]
        return {
            "anomaly_probs": torch.sigmoid(torch.randn(batch_size, 1)),
            "attention_weights": torch.softmax(torch.randn(batch_size, 5), dim=-1),
        }


class TestEnsembleOmniFusionModel:
    """Tests for EnsembleOmniFusionModel class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        base_model = MockBaseModel()
        ensemble = EnsembleOmniFusionModel(base_model)
        assert ensemble.config.method == EnsembleMethod.STACKING

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        base_model = MockBaseModel()
        config = EnsembleConfig(method=EnsembleMethod.BOOSTING)
        ensemble = EnsembleOmniFusionModel(base_model, config=config)
        assert ensemble.config.method == EnsembleMethod.BOOSTING

    def test_init_with_detector_names(self) -> None:
        """Test initialization with detector names."""
        base_model = MockBaseModel()
        detector_names = ["det1", "det2", "det3"]
        ensemble = EnsembleOmniFusionModel(base_model, detector_names=detector_names)
        assert ensemble.detector_names == detector_names

    def test_forward_stacking(self) -> None:
        """Test forward pass with stacking method."""
        base_model = MockBaseModel()
        config = EnsembleConfig(method=EnsembleMethod.STACKING)
        ensemble = EnsembleOmniFusionModel(base_model, config=config)

        detector_features = {"det1": torch.randn(4, 64)}
        detector_scores = {"det1": torch.randn(4, 1)}

        output = ensemble(detector_features, detector_scores)
        assert "anomaly_probs" in output
        assert "ensemble_method" in output
        assert output["ensemble_method"] == "stacking"

    def test_forward_weighted_average(self) -> None:
        """Test forward pass with weighted average method."""
        base_model = MockBaseModel()
        config = EnsembleConfig(method=EnsembleMethod.WEIGHTED_AVERAGE)
        ensemble = EnsembleOmniFusionModel(base_model, config=config)

        detector_features = {"det1": torch.randn(4, 64)}
        detector_scores = {"det1": torch.randn(4, 1)}

        output = ensemble(detector_features, detector_scores)
        assert "anomaly_probs" in output
        assert "ensemble_method" in output
        assert output["ensemble_method"] == "weighted_average"

    def test_forward_boosting(self) -> None:
        """Test forward pass with boosting method."""
        base_model = MockBaseModel()
        config = EnsembleConfig(method=EnsembleMethod.BOOSTING)
        ensemble = EnsembleOmniFusionModel(base_model, config=config)

        detector_features = {"det1": torch.randn(4, 64)}
        output = ensemble(detector_features)
        assert "ensemble_method" in output
        assert output["ensemble_method"] == "boosting"

    def test_forward_without_ensemble(self) -> None:
        """Test forward pass without ensemble."""
        base_model = MockBaseModel()
        ensemble = EnsembleOmniFusionModel(base_model)

        detector_features = {"det1": torch.randn(4, 64)}
        output = ensemble(detector_features, use_ensemble=False)
        assert "anomaly_probs" in output

    def test_update_boosting_weights(self) -> None:
        """Test boosting weight update."""
        base_model = MockBaseModel()
        config = EnsembleConfig(method=EnsembleMethod.BOOSTING)
        ensemble = EnsembleOmniFusionModel(base_model, config=config)

        predictions = torch.tensor([0.9, 0.8, 0.6, 0.4])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        ensemble.detector_weights.clone()
        ensemble.update_boosting_weights(predictions, targets, detector_idx=0)

        assert len(ensemble._training_errors) > 0

    def test_get_ensemble_stats(self) -> None:
        """Test getting ensemble statistics."""
        base_model = MockBaseModel()
        detector_names = ["det1", "det2"]
        ensemble = EnsembleOmniFusionModel(base_model, detector_names=detector_names)

        stats = ensemble.get_ensemble_stats()
        assert "method" in stats
        assert "num_detectors" in stats
        assert "detector_weights" in stats
        assert "meta_learner_params" in stats


class TestVotingEnsemble:
    """Tests for VotingEnsemble class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        ensemble = VotingEnsemble()
        assert ensemble.voting_type == "soft"
        assert ensemble.weights is None

    def test_init_hard_voting(self) -> None:
        """Test hard voting initialization."""
        ensemble = VotingEnsemble(voting_type="hard")
        assert ensemble.voting_type == "hard"

    def test_init_with_weights(self) -> None:
        """Test initialization with weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = VotingEnsemble(weights=weights)
        assert ensemble.weights == weights

    def test_predict_soft_voting(self) -> None:
        """Test soft voting prediction."""
        ensemble = VotingEnsemble(voting_type="soft")
        detector_outputs = {
            "det1": np.array([0.8, 0.2, 0.9]),
            "det2": np.array([0.7, 0.3, 0.8]),
        }
        result = ensemble.predict(detector_outputs)
        assert "predictions" in result
        assert "confidence" in result
        assert len(result["predictions"]) == 3

    def test_predict_hard_voting(self) -> None:
        """Test hard voting prediction."""
        ensemble = VotingEnsemble(voting_type="hard")
        detector_outputs = {
            "det1": np.array([0.8, 0.2, 0.9]),
            "det2": np.array([0.7, 0.3, 0.8]),
        }
        result = ensemble.predict(detector_outputs)
        assert "predictions" in result
        assert all(p in [0, 1] for p in result["predictions"])

    def test_predict_with_weights(self) -> None:
        """Test prediction with custom weights."""
        ensemble = VotingEnsemble(voting_type="soft", weights=[0.7, 0.3])
        detector_outputs = {
            "det1": np.array([0.9, 0.1]),
            "det2": np.array([0.1, 0.9]),
        }
        result = ensemble.predict(detector_outputs)
        assert "predictions" in result

    def test_predict_empty_outputs(self) -> None:
        """Test prediction with empty outputs."""
        ensemble = VotingEnsemble()
        result = ensemble.predict({})
        assert len(result["predictions"]) == 0

    def test_predict_custom_threshold(self) -> None:
        """Test prediction with custom threshold."""
        ensemble = VotingEnsemble(voting_type="soft")
        detector_outputs = {
            "det1": np.array([0.6, 0.4]),
        }
        result_low = ensemble.predict(detector_outputs, threshold=0.3)
        result_high = ensemble.predict(detector_outputs, threshold=0.7)
        assert result_low["predictions"][0] == 1
        assert result_high["predictions"][0] == 0


class TestCreateEnsembleModel:
    """Tests for create_ensemble_model factory function."""

    def test_create_stacking(self) -> None:
        """Test creating stacking ensemble."""
        base_model = MockBaseModel()
        ensemble = create_ensemble_model(base_model, method="stacking")
        assert ensemble.config.method == EnsembleMethod.STACKING

    def test_create_boosting(self) -> None:
        """Test creating boosting ensemble."""
        base_model = MockBaseModel()
        ensemble = create_ensemble_model(base_model, method="boosting")
        assert ensemble.config.method == EnsembleMethod.BOOSTING

    def test_create_weighted_average(self) -> None:
        """Test creating weighted average ensemble."""
        base_model = MockBaseModel()
        ensemble = create_ensemble_model(base_model, method="weighted_average")
        assert ensemble.config.method == EnsembleMethod.WEIGHTED_AVERAGE

    def test_create_with_detector_names(self) -> None:
        """Test creating ensemble with detector names."""
        base_model = MockBaseModel()
        detector_names = ["det1", "det2", "det3"]
        ensemble = create_ensemble_model(base_model, detector_names=detector_names)
        assert ensemble.detector_names == detector_names

    def test_create_with_kwargs(self) -> None:
        """Test creating ensemble with additional kwargs."""
        base_model = MockBaseModel()
        ensemble = create_ensemble_model(
            base_model,
            method="stacking",
            hidden_dim=128,
            num_layers=3,
            dropout=0.2,
        )
        assert ensemble.config.meta_learner_hidden_dim == 128
        assert ensemble.config.meta_learner_layers == 3
        assert ensemble.config.dropout == 0.2

    def test_create_unknown_method(self) -> None:
        """Test creating ensemble with unknown method defaults to stacking."""
        base_model = MockBaseModel()
        ensemble = create_ensemble_model(base_model, method="unknown")
        assert ensemble.config.method == EnsembleMethod.STACKING
