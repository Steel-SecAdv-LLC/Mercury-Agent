# Copyright (C) 2025 Steel Security Advisors LLC
"""Mercury Agent - Tests for Lightweight Neural Primitives.

Tests for pure NumPy implementations of neural network operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.ml.lightweight_primitives import (
    Activation,
    IsolationScorer,
    LightweightAutoencoder,
    LightweightMLP,
    MLPConfig,
    get_activation,
    leaky_relu,
    quick_anomaly_score,
    relu,
    sigmoid,
    softmax,
    tanh,
)


class TestActivationFunctions:
    """Tests for activation functions."""

    def test_relu(self) -> None:
        """Test ReLU activation."""
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        result = relu(x)

        np.testing.assert_array_equal(result, np.array([0, 0, 0, 1, 2]))

    def test_relu_preserves_positive(self) -> None:
        """Test ReLU preserves positive values."""
        x = np.random.randn(100).astype(np.float32)
        result = relu(x)

        # Positive values should be unchanged
        positive_mask = x > 0
        np.testing.assert_array_almost_equal(result[positive_mask], x[positive_mask])

    def test_leaky_relu(self) -> None:
        """Test Leaky ReLU activation."""
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        result = leaky_relu(x, alpha=0.1)

        expected = np.array([-0.2, -0.1, 0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sigmoid(self) -> None:
        """Test sigmoid activation."""
        x = np.array([-10, 0, 10], dtype=np.float32)
        result = sigmoid(x)

        # Sigmoid should be in (0, 1)
        assert np.all(result > 0)
        assert np.all(result < 1)

        # Sigmoid(0) = 0.5
        assert np.abs(result[1] - 0.5) < 1e-6

    def test_sigmoid_extreme_values(self) -> None:
        """Test sigmoid handles extreme values without overflow."""
        x = np.array([-1000, 1000], dtype=np.float32)
        result = sigmoid(x)

        # Should not be NaN or Inf
        assert np.isfinite(result).all()
        assert result[0] < 0.01
        assert result[1] > 0.99

    def test_tanh(self) -> None:
        """Test tanh activation."""
        x = np.array([-10, 0, 10], dtype=np.float32)
        result = tanh(x)

        # Tanh should be in [-1, 1] (saturates at boundaries for extreme values)
        assert np.all(result >= -1)
        assert np.all(result <= 1)

        # Tanh(0) = 0
        assert np.abs(result[1]) < 1e-6

    def test_softmax(self) -> None:
        """Test softmax activation."""
        x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32)
        result = softmax(x, axis=1)

        # Softmax should sum to 1 along axis
        np.testing.assert_array_almost_equal(result.sum(axis=1), np.array([1, 1]))

        # All values should be positive
        assert np.all(result > 0)

    def test_softmax_numerical_stability(self) -> None:
        """Test softmax handles large values without overflow."""
        x = np.array([[1000, 1001, 1002]], dtype=np.float32)
        result = softmax(x, axis=1)

        # Should not be NaN or Inf
        assert np.isfinite(result).all()
        np.testing.assert_almost_equal(result.sum(), 1.0)

    def test_get_activation(self) -> None:
        """Test activation function getter."""
        assert get_activation("relu") is relu
        assert get_activation("sigmoid") is sigmoid
        assert get_activation("tanh") is tanh
        assert get_activation(Activation.RELU) is relu


class TestLightweightMLP:
    """Tests for LightweightMLP."""

    @pytest.fixture
    def config(self) -> MLPConfig:
        """Create test configuration."""
        return MLPConfig(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=1,
            activation="relu",
            output_activation="sigmoid",
            seed=42,
        )

    @pytest.fixture
    def mlp(self, config: Any) -> LightweightMLP:
        """Create MLP instance."""
        return LightweightMLP(config)

    def test_init(self, mlp: Any, config: Any) -> None:
        """Test MLP initialization."""
        assert mlp.config.input_dim == 64
        assert mlp.config.output_dim == 1
        assert len(mlp.layers) == 3  # 2 hidden + 1 output

    def test_layer_dimensions(self, mlp: Any) -> None:
        """Test layer weight dimensions."""
        # First layer: input_dim -> hidden_dim[0]
        assert mlp.layers[0].weights.shape == (64, 128)
        assert mlp.layers[0].bias.shape == (128,)

        # Second layer: hidden_dim[0] -> hidden_dim[1]
        assert mlp.layers[1].weights.shape == (128, 64)
        assert mlp.layers[1].bias.shape == (64,)

        # Output layer: hidden_dim[-1] -> output_dim
        assert mlp.layers[2].weights.shape == (64, 1)
        assert mlp.layers[2].bias.shape == (1,)

    def test_forward_single_sample(self, mlp: Any) -> None:
        """Test forward pass with single sample."""
        x = np.random.randn(64).astype(np.float32)
        output = mlp.forward(x)

        assert output.shape == (1, 1)
        # Sigmoid output should be in (0, 1)
        assert 0 < output[0, 0] < 1

    def test_forward_batch(self, mlp: Any) -> None:
        """Test forward pass with batch."""
        x = np.random.randn(32, 64).astype(np.float32)
        output = mlp.forward(x)

        assert output.shape == (32, 1)
        assert np.all((output > 0) & (output < 1))

    def test_forward_padding(self, mlp: Any) -> None:
        """Test forward handles input padding."""
        x = np.random.randn(10, 32).astype(np.float32)  # Only 32 features
        output = mlp.forward(x)

        assert output.shape == (10, 1)

    def test_forward_truncation(self, mlp: Any) -> None:
        """Test forward handles input truncation."""
        x = np.random.randn(10, 128).astype(np.float32)  # 128 features
        output = mlp.forward(x)

        assert output.shape == (10, 1)

    def test_predict_alias(self, mlp: Any) -> None:
        """Test predict is alias for forward."""
        x = np.random.randn(64).astype(np.float32)
        forward_result = mlp.forward(x)
        predict_result = mlp.predict(x)

        np.testing.assert_array_equal(forward_result, predict_result)

    def test_export_load_weights(self, mlp: Any) -> None:
        """Test weight export and loading."""
        x = np.random.randn(10, 64).astype(np.float32)

        # Get output with original weights
        original_output = mlp.forward(x)

        # Export weights
        weights = mlp.export_weights()

        # Create new MLP and load weights
        new_mlp = LightweightMLP(mlp.config)
        new_mlp.load_weights(weights)

        # Output should match
        loaded_output = new_mlp.forward(x)
        np.testing.assert_array_almost_equal(original_output, loaded_output)

    def test_param_count(self, mlp: Any) -> None:
        """Test parameter counting."""
        # Expected: 64*128 + 128 + 128*64 + 64 + 64*1 + 1 = 8192 + 128 + 8192 + 64 + 64 + 1 = 16641
        expected = 64 * 128 + 128 + 128 * 64 + 64 + 64 * 1 + 1
        assert mlp.get_param_count() == expected

    def test_batch_normalization(self) -> None:
        """Test MLP with batch normalization."""
        config = MLPConfig(
            input_dim=32,
            hidden_dims=[64],
            output_dim=1,
            use_batch_norm=True,
            seed=42,
        )
        mlp = LightweightMLP(config)

        # Check BN params exist for hidden layer
        assert mlp.layers[0].bn_gamma is not None
        assert mlp.layers[0].bn_beta is not None
        assert mlp.layers[0].bn_mean is not None
        assert mlp.layers[0].bn_var is not None

        # Output layer should not have BN
        assert mlp.layers[1].bn_gamma is None

        # Forward should work
        x = np.random.randn(10, 32).astype(np.float32)
        output = mlp.forward(x)
        assert output.shape == (10, 1)

    def test_different_activations(self) -> None:
        """Test MLP with different activations."""
        for activation in ["relu", "tanh", "sigmoid", "leaky_relu"]:
            config = MLPConfig(
                input_dim=16,
                hidden_dims=[32],
                output_dim=1,
                activation=activation,
                seed=42,
            )
            mlp = LightweightMLP(config)

            x = np.random.randn(5, 16).astype(np.float32)
            output = mlp.forward(x)

            assert output.shape == (5, 1)
            assert np.isfinite(output).all()


class TestLightweightAutoencoder:
    """Tests for LightweightAutoencoder."""

    @pytest.fixture
    def autoencoder(self) -> LightweightAutoencoder:
        """Create autoencoder instance."""
        return LightweightAutoencoder(
            input_dim=64,
            latent_dim=16,
            hidden_dim=32,
            seed=42,
        )

    def test_init(self, autoencoder: Any) -> None:
        """Test autoencoder initialization."""
        assert autoencoder.input_dim == 64
        assert autoencoder.latent_dim == 16

    def test_encode(self, autoencoder: Any) -> None:
        """Test encoding to latent space."""
        x = np.random.randn(10, 64).astype(np.float32)
        z = autoencoder.encode(x)

        assert z.shape == (10, 16)

    def test_decode(self, autoencoder: Any) -> None:
        """Test decoding from latent space."""
        z = np.random.randn(10, 16).astype(np.float32)
        reconstruction = autoencoder.decode(z)

        assert reconstruction.shape == (10, 64)

    def test_reconstruct(self, autoencoder: Any) -> None:
        """Test full reconstruction."""
        x = np.random.randn(10, 64).astype(np.float32)
        reconstruction = autoencoder.reconstruct(x)

        assert reconstruction.shape == x.shape

    def test_anomaly_score(self, autoencoder: Any) -> None:
        """Test anomaly scoring."""
        x = np.random.randn(10, 64).astype(np.float32)
        scores = autoencoder.anomaly_score(x)

        assert scores.shape == (10,)
        # Scores should be in [0, 1)
        assert np.all(scores >= 0)
        assert np.all(scores < 1)

    def test_anomaly_score_single(self, autoencoder: Any) -> None:
        """Test anomaly scoring with single sample."""
        x = np.random.randn(64).astype(np.float32)
        scores = autoencoder.anomaly_score(x)

        assert scores.shape == (1,)


class TestIsolationScorer:
    """Tests for IsolationScorer."""

    @pytest.fixture
    def scorer(self) -> IsolationScorer:
        """Create scorer instance."""
        return IsolationScorer(n_projections=20, seed=42)

    @pytest.fixture
    def normal_data(self) -> np.ndarray:
        """Generate normal data."""
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (100, 10)).astype(np.float32)

    def test_fit(self, scorer: Any, normal_data: Any) -> None:
        """Test fitting scorer."""
        result = scorer.fit(normal_data)

        assert result is scorer  # Method chaining
        assert scorer._fitted
        assert scorer._projections is not None
        assert scorer._thresholds is not None

    def test_score_not_fitted(self, scorer: Any, normal_data: Any) -> None:
        """Test score raises error when not fitted."""
        with pytest.raises(ValueError, match="not fitted"):
            scorer.score(normal_data)

    def test_score(self, scorer: Any, normal_data: Any) -> None:
        """Test scoring."""
        scorer.fit(normal_data)
        scores = scorer.score(normal_data)

        assert scores.shape == (100,)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_anomaly_detection(self, scorer: Any, normal_data: Any) -> None:
        """Test that anomalies get higher scores."""
        scorer.fit(normal_data)

        # Create obvious anomalies
        rng = np.random.default_rng(123)
        anomalies = rng.normal(10, 1, (10, 10)).astype(np.float32)

        normal_scores = scorer.score(normal_data)
        anomaly_scores = scorer.score(anomalies)

        # Anomalies should have higher mean score
        assert anomaly_scores.mean() > normal_scores.mean()


class TestQuickAnomalyScore:
    """Tests for quick_anomaly_score convenience function."""

    @pytest.fixture
    def data(self) -> np.ndarray:
        """Generate test data."""
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (50, 16)).astype(np.float32)

    def test_isolation_method(self, data: Any) -> None:
        """Test isolation method."""
        scores = quick_anomaly_score(data, method="isolation", seed=42)

        assert scores.shape == (50,)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_reconstruction_method(self, data: Any) -> None:
        """Test reconstruction method."""
        scores = quick_anomaly_score(data, method="reconstruction", seed=42)

        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    def test_statistical_method(self, data: Any) -> None:
        """Test statistical method."""
        scores = quick_anomaly_score(data, method="statistical")

        assert scores.shape == (50,)

    def test_unknown_method(self, data: Any) -> None:
        """Test unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            quick_anomaly_score(data, method="unknown")


class TestWeightInitialization:
    """Tests for weight initialization methods."""

    def test_xavier_init(self) -> None:
        """Test Xavier initialization."""
        config = MLPConfig(
            input_dim=100,
            hidden_dims=[200],
            output_dim=10,
            weight_init="xavier",
            seed=42,
        )
        mlp = LightweightMLP(config)

        # Xavier: std = sqrt(2 / (fan_in + fan_out))
        expected_std = np.sqrt(2 / (100 + 200))
        actual_std = mlp.layers[0].weights.std()

        # Should be close to expected
        assert np.abs(actual_std - expected_std) < 0.1

    def test_he_init(self) -> None:
        """Test He initialization."""
        config = MLPConfig(
            input_dim=100,
            hidden_dims=[200],
            output_dim=10,
            weight_init="he",
            seed=42,
        )
        mlp = LightweightMLP(config)

        # He: std = sqrt(2 / fan_in)
        expected_std = np.sqrt(2 / 100)
        actual_std = mlp.layers[0].weights.std()

        # Should be close to expected
        assert np.abs(actual_std - expected_std) < 0.1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_hidden_dims(self) -> None:
        """Test MLP with no hidden layers."""
        config = MLPConfig(
            input_dim=10,
            hidden_dims=[],
            output_dim=1,
            seed=42,
        )
        mlp = LightweightMLP(config)

        assert len(mlp.layers) == 1

        x = np.random.randn(5, 10).astype(np.float32)
        output = mlp.forward(x)
        assert output.shape == (5, 1)

    def test_single_feature(self) -> None:
        """Test MLP with single input feature."""
        config = MLPConfig(
            input_dim=1,
            hidden_dims=[8],
            output_dim=1,
            seed=42,
        )
        mlp = LightweightMLP(config)

        x = np.random.randn(5, 1).astype(np.float32)
        output = mlp.forward(x)
        assert output.shape == (5, 1)

    def test_multiclass_output(self) -> None:
        """Test MLP with multiclass output."""
        config = MLPConfig(
            input_dim=32,
            hidden_dims=[64],
            output_dim=10,
            output_activation="softmax",
            seed=42,
        )
        mlp = LightweightMLP(config)

        x = np.random.randn(5, 32).astype(np.float32)
        output = mlp.forward(x)

        assert output.shape == (5, 10)
        # Softmax should sum to 1
        np.testing.assert_array_almost_equal(output.sum(axis=1), np.ones(5), decimal=5)
