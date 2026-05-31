"""
Mercury Agent

Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Lightweight Neural Network Primitives - Pure NumPy Implementation

This module provides dependency-light neural network operations using only NumPy.
Designed for edge deployments and environments where PyTorch is not available.

Based on fundamental concepts from:
- CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)
- Neural network fundamentals: linear transforms + nonlinearities

Features:
- Multi-layer perceptron with configurable architecture
- Common activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU)
- Batch normalization (inference mode)
- Xavier/He weight initialization
- Forward-only inference (training requires PyTorch/full ML stack)

Usage:
    from omni_mercury_engine.ml.lightweight_primitives import (
        LightweightMLP, relu, sigmoid, softmax
    )

    # Create a 3-layer network
    mlp = LightweightMLP(
        input_dim=64,
        hidden_dims=[128, 64],
        output_dim=1,
        activation='relu'
    )

    # Inference
    scores = mlp.forward(features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class Activation(Enum):
    """Supported activation functions."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"
    LINEAR = "linear"


def relu(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """
    Relu activation: f(x) = max(0, x)

    Advantages: Fast convergence, computationally simple
    Disadvantage: Can cause 'dying ReLU' with high learning rates

    Args:
        x: Input array

    Returns:
        Activated output
    """
    return np.maximum(0, x)


def leaky_relu(x: NDArray[np.floating[Any]], alpha: float = 0.01) -> NDArray[np.floating[Any]]:
    """
    Leaky ReLU: f(x) = x if x > 0, else alpha * x

    Addresses dying ReLU problem with small negative slope.

    Args:
        x: Input array
        alpha: Negative slope coefficient

    Returns:
        Activated output
    """
    return np.where(x > 0, x, alpha * x)


def sigmoid(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """
    Sigmoid activation: f(x) = 1 / (1 + exp(-x))

    Squashes values to [0, 1] range. Useful for binary classification.
    Note: Can suffer from vanishing gradients for extreme values.

    Args:
        x: Input array

    Returns:
        Activated output in [0, 1]
    """
    # Numerically-stable form: avoid the ``exp(+large)`` overflow
    # path entirely.  For ``x >= 0``: ``sigmoid(x) = 1 / (1 + exp(-x))``
    # where ``exp(-x) <= 1``.  For ``x < 0``: rewrite as
    # ``sigmoid(x) = exp(x) / (1 + exp(x))`` where ``exp(x) <= 1``.
    # This is exact in float64 and never overflows, replacing the
    # earlier clip-to-[-500, 500] hack which still tripped
    # ``exp(500)`` (1.4e217, finite) but blew up on ``exp(-(-500))``.
    pos_mask = x >= 0
    exp_neg_abs = np.exp(-np.abs(x))
    return np.asarray(
        np.where(pos_mask, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))
    )


def tanh(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """
    Tanh activation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Zero-centered, preferred over sigmoid for hidden layers.

    Args:
        x: Input array

    Returns:
        Activated output in [-1, 1]
    """
    return np.tanh(x)


def softmax(x: NDArray[np.floating[Any]], axis: int = -1) -> NDArray[np.floating[Any]]:
    """
    Softmax: converts logits to probability distribution.

    f(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        x: Input array (logits)
        axis: Axis along which to compute softmax

    Returns:
        Probability distribution summing to 1
    """
    # Numerical stability: subtract max
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def get_activation(name: str | Activation) -> Any:
    """
    Get activation function by name.

    Args:
        name: Activation name or enum

    Returns:
        Activation function
    """
    if isinstance(name, Activation):
        name = name.value

    activations = {
        "relu": relu,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "leaky_relu": leaky_relu,
        "softmax": softmax,
        "linear": lambda x: x,
    }

    return activations.get(name.lower(), relu)


@dataclass
class LayerParams:
    """Parameters for a single layer."""

    weights: NDArray[np.floating[Any]]
    bias: NDArray[np.floating[Any]]
    activation: str = "relu"

    # Optional batch norm parameters (inference mode)
    bn_gamma: NDArray[np.floating[Any]] | None = None
    bn_beta: NDArray[np.floating[Any]] | None = None
    bn_mean: NDArray[np.floating[Any]] | None = None
    bn_var: NDArray[np.floating[Any]] | None = None


@dataclass
class MLPConfig:
    """Configuration for lightweight MLP."""

    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 1
    activation: str = "relu"
    output_activation: str = "sigmoid"
    use_batch_norm: bool = False
    dropout_rate: float = 0.0  # Inference-time dropout (not recommended)
    weight_init: str = "xavier"  # xavier, he, normal
    seed: int = 42


class LightweightMLP:
    """
    Lightweight Multi-Layer Perceptron using pure NumPy.

    Implements forward-only inference for anomaly detection scoring.
    For training, use PyTorch-based models and export weights.

    Architecture:
        Input -> [Linear + BN + Activation + Dropout] x N -> Output

    Example:
        mlp = LightweightMLP(MLPConfig(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=1,
            activation='relu',
            output_activation='sigmoid'
        ))

        # Random initialization for inference
        scores = mlp.forward(features)
    """

    def __init__(self, config: MLPConfig | None = None, **kwargs: Any) -> None:
        """
        Initialize MLP.

        Args:
            config: MLPConfig object or None to use kwargs
            **kwargs: Configuration parameters if config is None
        """
        if config is None:
            config = MLPConfig(**kwargs)

        self.config = config
        self.layers: list[LayerParams] = []
        self.rng = np.random.default_rng(config.seed)

        self._initialize_layers()
        logger.info(
            f"LightweightMLP initialized: {config.input_dim} -> "
            f"{config.hidden_dims} -> {config.output_dim}"
        )

    def _initialize_layers(self) -> None:
        """Initialize layer parameters with proper weight initialization."""
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]

        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]

            # Weight initialization
            if self.config.weight_init == "xavier":
                # Xavier/Glorot: good for tanh/sigmoid
                std = np.sqrt(2.0 / (fan_in + fan_out))
            elif self.config.weight_init == "he":
                # He: good for ReLU
                std = np.sqrt(2.0 / fan_in)
            else:
                std = 0.01

            weights = self.rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)
            bias = np.zeros(fan_out, dtype=np.float32)

            # Batch norm parameters (if enabled)
            bn_params: dict[str, NDArray[np.floating[Any]] | None] = {
                "bn_gamma": None,
                "bn_beta": None,
                "bn_mean": None,
                "bn_var": None,
            }

            if self.config.use_batch_norm and i < len(dims) - 2:
                bn_params["bn_gamma"] = np.ones(fan_out, dtype=np.float32)
                bn_params["bn_beta"] = np.zeros(fan_out, dtype=np.float32)
                bn_params["bn_mean"] = np.zeros(fan_out, dtype=np.float32)
                bn_params["bn_var"] = np.ones(fan_out, dtype=np.float32)

            if i == len(dims) - 2:
                activation = self.config.output_activation
            else:
                activation = self.config.activation

            self.layers.append(
                LayerParams(
                    weights=weights,
                    bias=bias,
                    activation=activation,
                    **bn_params,
                )
            )

    def forward(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Forward pass through the network.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Output scores (batch_size, output_dim)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Ensure correct input dimension
        if x.shape[1] != self.config.input_dim:
            if x.shape[1] < self.config.input_dim:
                # Pad with zeros
                padding = np.zeros(
                    (x.shape[0], self.config.input_dim - x.shape[1]),
                    dtype=np.float32,
                )
                x = np.concatenate([x, padding], axis=1)
            else:
                # Truncate
                x = x[:, : self.config.input_dim]

        x = x.astype(np.float32)
        out = x

        for layer in self.layers:
            # Linear transformation: y = Wx + b
            out = np.dot(out, layer.weights) + layer.bias

            # Batch normalization (inference mode)
            if layer.bn_gamma is not None and layer.bn_var is not None:
                eps = 1e-5
                out = (out - layer.bn_mean) / np.sqrt(layer.bn_var + eps)
                out = layer.bn_gamma * out + layer.bn_beta

            # Activation
            activation_fn = get_activation(layer.activation)
            out = activation_fn(out)

        return out

    def predict(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Alias for forward pass."""
        return self.forward(x)

    def load_weights(self, weights_dict: dict[str, Any]) -> None:
        """
        Load weights from a dictionary (e.g., exported from PyTorch).

        Args:
            weights_dict: Dictionary with layer weights
                Format: {'layer_0_weights': ..., 'layer_0_bias': ..., ...}
        """
        for i, layer in enumerate(self.layers):
            w_key = f"layer_{i}_weights"
            b_key = f"layer_{i}_bias"

            if w_key in weights_dict:
                layer.weights = np.array(weights_dict[w_key], dtype=np.float32)
            if b_key in weights_dict:
                layer.bias = np.array(weights_dict[b_key], dtype=np.float32)

            # Batch norm
            if self.config.use_batch_norm:
                if f"layer_{i}_bn_gamma" in weights_dict:
                    layer.bn_gamma = np.array(weights_dict[f"layer_{i}_bn_gamma"], dtype=np.float32)
                if f"layer_{i}_bn_beta" in weights_dict:
                    layer.bn_beta = np.array(weights_dict[f"layer_{i}_bn_beta"], dtype=np.float32)
                if f"layer_{i}_bn_mean" in weights_dict:
                    layer.bn_mean = np.array(weights_dict[f"layer_{i}_bn_mean"], dtype=np.float32)
                if f"layer_{i}_bn_var" in weights_dict:
                    layer.bn_var = np.array(weights_dict[f"layer_{i}_bn_var"], dtype=np.float32)

        logger.info(f"Loaded weights for {len(self.layers)} layers")

    def export_weights(self) -> dict[str, Any]:
        """
        Export weights to dictionary format.

        Returns:
            Dictionary with all layer parameters
        """
        weights_dict: dict[str, Any] = {}

        for i, layer in enumerate(self.layers):
            weights_dict[f"layer_{i}_weights"] = layer.weights.tolist()
            weights_dict[f"layer_{i}_bias"] = layer.bias.tolist()

            if (
                layer.bn_gamma is not None
                and layer.bn_beta is not None
                and layer.bn_mean is not None
                and layer.bn_var is not None
            ):
                weights_dict[f"layer_{i}_bn_gamma"] = layer.bn_gamma.tolist()
                weights_dict[f"layer_{i}_bn_beta"] = layer.bn_beta.tolist()
                weights_dict[f"layer_{i}_bn_mean"] = layer.bn_mean.tolist()
                weights_dict[f"layer_{i}_bn_var"] = layer.bn_var.tolist()

        return weights_dict

    def get_param_count(self) -> int:
        """Get total number of parameters."""
        count = 0
        for layer in self.layers:
            count += layer.weights.size + layer.bias.size
            if layer.bn_gamma is not None:
                count += layer.bn_gamma.size * 4  # gamma, beta, mean, var
        return count


class LightweightAutoencoder:
    """
    Lightweight Autoencoder for unsupervised anomaly detection.

    Uses reconstruction error as anomaly score.

    Architecture:
        Encoder: Input -> Hidden -> Latent
        Decoder: Latent -> Hidden -> Output

    Anomaly Score = ||input - reconstruction||^2
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        """
        Initialize autoencoder.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function
            seed: Random seed
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input -> hidden -> latent
        self.encoder = LightweightMLP(
            MLPConfig(
                input_dim=input_dim,
                hidden_dims=[hidden_dim],
                output_dim=latent_dim,
                activation=activation,
                output_activation="linear",
                seed=seed,
            )
        )

        # Decoder: latent -> hidden -> output
        self.decoder = LightweightMLP(
            MLPConfig(
                input_dim=latent_dim,
                hidden_dims=[hidden_dim],
                output_dim=input_dim,
                activation=activation,
                output_activation="linear",
                seed=seed + 1,
            )
        )

    def encode(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Encode input to latent space."""
        return self.encoder.forward(x)

    def decode(self, z: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Decode latent representation."""
        return self.decoder.forward(z)

    def reconstruct(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Reconstruct input through autoencoder."""
        z = self.encode(x)
        return self.decode(z)

    def anomaly_score(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Compute anomaly scores based on reconstruction error.

        Args:
            x: Input features

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        reconstruction = self.reconstruct(x)
        mse = np.mean((x - reconstruction) ** 2, axis=1)

        # Normalize to [0, 1] using sigmoid-like transform
        scores = 1 - np.exp(-mse)

        return scores


class IsolationScorer:
    """
    Lightweight isolation-based anomaly scorer.

    Uses random projections and isolation concepts without full Isolation Forest. Good for quick
    anomaly screening.
    """

    def __init__(
        self,
        n_projections: int = 50,
        contamination: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Initialize isolation scorer.

        Args:
            n_projections: Number of random projections
            contamination: Expected proportion of anomalies
            seed: Random seed
        """
        self.n_projections = n_projections
        self.contamination = contamination
        self.rng = np.random.default_rng(seed)

        self._projections: NDArray[np.floating[Any]] | None = None
        self._thresholds: NDArray[np.floating[Any]] | None = None
        self._fitted = False

    def fit(self, X: NDArray[np.floating[Any]]) -> IsolationScorer:
        """
        Fit the scorer to normal data.

        Args:
            X: Training data (assumed mostly normal)

        Returns:
            Self for method chaining
        """
        n_features = X.shape[1]

        # Generate random projections
        self._projections = self.rng.standard_normal((self.n_projections, n_features)).astype(
            np.float32
        )

        norms = np.linalg.norm(self._projections, axis=1, keepdims=True)
        self._projections = self._projections / (norms + 1e-10)

        # Project training data and compute thresholds
        projected = X @ self._projections.T

        # Thresholds based on percentiles
        self._thresholds = np.percentile(projected, [5, 95], axis=0).astype(np.float32)

        self._fitted = True
        return self

    def score(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Compute anomaly scores.

        Args:
            X: Input features

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self._fitted or self._projections is None or self._thresholds is None:
            raise ValueError("Scorer not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        projected = X @ self._projections.T

        # Count how many projections fall outside thresholds
        below = projected < self._thresholds[0]
        above = projected > self._thresholds[1]
        outside = below | above

        # Score = proportion of projections outside normal range
        scores = np.mean(outside, axis=1).astype(np.float32)

        return scores


# Convenience function for quick anomaly scoring
def quick_anomaly_score(
    X: NDArray[np.floating[Any]],
    method: str = "isolation",
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Quick anomaly scoring with minimal setup.

    Args:
        X: Input features (n_samples, n_features)
        method: 'isolation', 'reconstruction', or 'statistical'
        **kwargs: Method-specific parameters

    Returns:
        Anomaly scores
    """
    if method == "isolation":
        scorer = IsolationScorer(**kwargs)
        scorer.fit(X)
        return scorer.score(X)

    elif method == "reconstruction":
        input_dim = X.shape[1]
        ae = LightweightAutoencoder(input_dim=input_dim, **kwargs)
        return ae.anomaly_score(X)

    elif method == "statistical":
        # Simple z-score based anomaly detection
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-10
        z_scores = np.abs((X - mean) / std)
        return np.max(z_scores, axis=1).astype(np.float32)

    else:
        raise ValueError(f"Unknown method: {method}")


__all__ = [
    "Activation",
    "IsolationScorer",
    "LayerParams",
    "LightweightAutoencoder",
    "LightweightMLP",
    "MLPConfig",
    "get_activation",
    "leaky_relu",
    "quick_anomaly_score",
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
]
