"""
Mercury Agent
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

"""
Federated Learning for Distributed Robustness

Implements federated learning framework for distributed anomaly detection
with privacy-preserving aggregation and Byzantine fault tolerance.

References:
- McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data" (2017)
- Bonawitz et al., "Towards Federated Learning at Scale" (2019)

MIT-compatible implementation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


@dataclass
class ClientModel:
    """Client model in federated learning."""

    client_id: str
    model_weights: np.ndarray
    num_samples: int
    loss: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GlobalModel:
    """Global aggregated model."""

    round_number: int
    weights: np.ndarray
    participating_clients: int
    aggregated_loss: float
    timestamp: datetime = field(default_factory=datetime.now)


class FederatedAnomalyDetection:
    """
    Federated learning framework for distributed anomaly detection.

    Features:
    - Privacy-preserving aggregation
    - Byzantine fault tolerance
    - Adaptive client selection
    - Differential privacy support
    """

    def __init__(
        self,
        model_dim: int = 50,
        aggregation_method: str = "fedavg",
        byzantine_tolerance: bool = True,
        differential_privacy: bool = False,
        epsilon: float = 1.0,
        rng: DeterministicRNG | None = None,
    ):
        """
        Initialize federated anomaly detection.

        Args:
            model_dim: Dimension of model parameters
            aggregation_method: Aggregation method ('fedavg', 'fedprox', 'median')
            byzantine_tolerance: Enable Byzantine fault tolerance
            differential_privacy: Enable differential privacy
            epsilon: Privacy budget for differential privacy
            rng: Optional DeterministicRNG for reproducibility
        """
        self.model_dim = model_dim
        self.aggregation_method = aggregation_method
        self.byzantine_tolerance = byzantine_tolerance
        self.differential_privacy = differential_privacy
        self.epsilon = epsilon
        self._rng = rng or get_global_rng()

        self.global_model = GlobalModel(
            round_number=0,
            weights=self._rng.randn(model_dim) * 0.01,
            participating_clients=0,
            aggregated_loss=0.0,
        )

        self.client_models: dict[str, ClientModel] = {}
        self.round_history: list[GlobalModel] = []

    def register_client(
        self, client_id: str, initial_weights: np.ndarray[Any, Any] | None = None
    ) -> None:
        """
        Register new client in federated system.

        Args:
            client_id: Unique client identifier
            initial_weights: Optional initial model weights
        """
        if initial_weights is None:
            initial_weights = self.global_model.weights.copy()

        self.client_models[client_id] = ClientModel(
            client_id=client_id, model_weights=initial_weights, num_samples=0, loss=0.0
        )

    def client_update(
        self,
        client_id: str,
        local_data: np.ndarray[Any, Any],
        learning_rate: float = 0.01,
        local_epochs: int = 1,
    ) -> ClientModel:
        """
        Perform local client update.

        Args:
            client_id: Client identifier
            local_data: Local training data
            learning_rate: Learning rate for local updates
            local_epochs: Number of local training epochs

        Returns:
            Updated client model
        """
        if client_id not in self.client_models:
            raise ValueError(f"Client {client_id} not registered")

        client = self.client_models[client_id]

        weights = client.model_weights.copy()

        for _epoch in range(local_epochs):
            gradient = self._compute_gradient(weights, local_data)
            weights = weights - learning_rate * gradient

        loss = self._compute_loss(weights, local_data)

        client.model_weights = weights
        client.num_samples = len(local_data)
        client.loss = loss
        client.timestamp = datetime.now()

        return client

    def _compute_gradient(
        self, weights: np.ndarray[Any, Any], data: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Compute gradient for anomaly detection objective.

        Args:
            weights: Current model weights
            data: Training data

        Returns:
            Gradient vector
        """
        if len(data) == 0:
            return np.zeros_like(weights)

        data_mean = np.mean(data, axis=0)
        if len(data_mean) != len(weights):
            data_mean = np.resize(data_mean, len(weights))

        reconstruction_error = weights - data_mean

        gradient = 2 * reconstruction_error / len(data)

        return gradient

    def _compute_loss(self, weights: np.ndarray[Any, Any], data: np.ndarray[Any, Any]) -> float:
        """
        Compute loss for anomaly detection objective.

        Args:
            weights: Model weights
            data: Data samples

        Returns:
            Loss value
        """
        if len(data) == 0:
            return 0.0

        data_mean = np.mean(data, axis=0)
        if len(data_mean) != len(weights):
            data_mean = np.resize(data_mean, len(weights))

        mse = np.mean((weights - data_mean) ** 2)

        return float(mse)

    def aggregate(self, selected_clients: list[str] | None = None) -> GlobalModel:
        """
        Aggregate client models into global model.

        Args:
            selected_clients: Optional list of client IDs to aggregate

        Returns:
            Updated global model
        """
        if selected_clients is None:
            selected_clients = list(self.client_models.keys())

        if not selected_clients:
            return self.global_model

        client_data = [
            (self.client_models[cid].model_weights, self.client_models[cid].num_samples)
            for cid in selected_clients
            if cid in self.client_models
        ]

        if not client_data:
            return self.global_model

        if self.aggregation_method == "fedavg":
            aggregated_weights = self._fedavg_aggregation(client_data)
        elif self.aggregation_method == "median":
            aggregated_weights = self._median_aggregation(client_data)
        elif self.aggregation_method == "fedprox":
            aggregated_weights = self._fedprox_aggregation(client_data)
        else:
            aggregated_weights = self._fedavg_aggregation(client_data)

        if self.byzantine_tolerance:
            aggregated_weights = self._apply_byzantine_filter(aggregated_weights, client_data)

        if self.differential_privacy:
            aggregated_weights = self._add_differential_privacy_noise(aggregated_weights)

        aggregated_loss = np.mean(
            [self.client_models[cid].loss for cid in selected_clients if cid in self.client_models]
        )

        self.global_model = GlobalModel(
            round_number=self.global_model.round_number + 1,
            weights=aggregated_weights,
            participating_clients=len(selected_clients),
            aggregated_loss=float(aggregated_loss),
        )

        self.round_history.append(self.global_model)

        for cid in selected_clients:
            if cid in self.client_models:
                self.client_models[cid].model_weights = aggregated_weights.copy()

        return self.global_model

    def _fedavg_aggregation(
        self, client_data: list[tuple[np.ndarray[Any, Any], int]]
    ) -> np.ndarray[Any, Any]:
        """
        FedAvg: Weighted average by number of samples.

        Args:
            client_data: List of (weights, num_samples) tuples

        Returns:
            Aggregated weights
        """
        total_samples = sum(num for _, num in client_data)

        if total_samples == 0:
            return self.global_model.weights.copy()

        aggregated = np.zeros(self.model_dim)

        for weights, num_samples in client_data:
            weight_contribution = (num_samples / total_samples) * weights
            aggregated += weight_contribution

        return aggregated

    def _median_aggregation(
        self, client_data: list[tuple[np.ndarray[Any, Any], int]]
    ) -> np.ndarray[Any, Any]:
        """
        Median aggregation for Byzantine tolerance.

        Args:
            client_data: List of (weights, num_samples) tuples

        Returns:
            Aggregated weights using coordinate-wise median
        """
        if not client_data:
            return self.global_model.weights.copy()

        all_weights = np.array([weights for weights, _ in client_data])

        median_weights = np.median(all_weights, axis=0)

        return median_weights

    def _fedprox_aggregation(
        self, client_data: list[tuple[np.ndarray[Any, Any], int]]
    ) -> np.ndarray[Any, Any]:
        """
        FedProx: Proximal term for handling heterogeneous data.

        Args:
            client_data: List of (weights, num_samples) tuples

        Returns:
            Aggregated weights with proximal regularization
        """
        mu = 0.01

        fedavg_weights = self._fedavg_aggregation(client_data)

        proximal_term = mu * (self.global_model.weights - fedavg_weights)

        aggregated = fedavg_weights + proximal_term

        return aggregated

    def _apply_byzantine_filter(
        self, aggregated: np.ndarray[Any, Any], client_data: list[tuple[np.ndarray[Any, Any], int]]
    ) -> np.ndarray[Any, Any]:
        """
        Apply Byzantine fault tolerance filter.

        Args:
            aggregated: Preliminary aggregated weights
            client_data: Client data for validation

        Returns:
            Filtered weights
        """
        if len(client_data) < 3:
            return aggregated

        distances = []
        for weights, _ in client_data:
            dist = np.linalg.norm(weights - aggregated)
            distances.append(dist)

        median_distance = np.median(distances)
        threshold = median_distance * 3.0

        filtered_data = [
            (weights, num)
            for (weights, num), dist in zip(client_data, distances, strict=False)
            if dist < threshold
        ]

        if len(filtered_data) < len(client_data) * 0.5:
            return aggregated

        return self._fedavg_aggregation(filtered_data)

    def _add_differential_privacy_noise(
        self, weights: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Add Gaussian noise for differential privacy.

        Args:
            weights: Model weights

        Returns:
            Noisy weights
        """
        sensitivity = 1.0

        sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / self.epsilon

        noise = self._rng.normal(0, sigma, size=weights.shape)

        return weights + noise

    def get_federation_stats(self) -> dict[str, Any]:
        """Get federated learning statistics."""
        return {
            "num_clients": len(self.client_models),
            "current_round": self.global_model.round_number,
            "global_loss": self.global_model.aggregated_loss,
            "aggregation_method": self.aggregation_method,
            "byzantine_tolerance_enabled": self.byzantine_tolerance,
            "differential_privacy_enabled": self.differential_privacy,
            "epsilon": self.epsilon if self.differential_privacy else None,
            "total_rounds": len(self.round_history),
            "participating_clients_last_round": self.global_model.participating_clients,
        }
