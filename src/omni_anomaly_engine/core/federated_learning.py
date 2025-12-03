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

"""Federated Learning for Privacy-Preserving Anomaly Detection.

Based on: Federated quantum-inspired anomaly detection using collaborative neural clients
(Frontiers in AI, 2025:
https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1648609/full)

Implements FedAvg aggregation algorithm for distributed model training without sharing raw data.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from omni_anomaly_engine.utils.rng import DeterministicRNG, get_global_rng


class FederatedAnomalyDetector:
    """Federated learning for privacy-preserving anomaly detection."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        rng: Optional[DeterministicRNG] = None,
    ):
        """Initialize federated detector.

        Args:
            config: Configuration including:
                - num_clients: Number of federated clients (default: 5)
                - learning_rate: Learning rate for local training (default: 0.001)
                - local_epochs: Epochs per client per round (default: 5)
                - aggregation_method: 'fedavg' or 'weighted_fedavg' (default: 'fedavg')
            rng: Optional DeterministicRNG for reproducibility
        """
        self.config = config or {}
        self.num_clients = self.config.get("num_clients", 5)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.local_epochs = self.config.get("local_epochs", 5)
        self.aggregation_method = self.config.get("aggregation_method", "fedavg")
        self.global_model: Optional[np.ndarray] = None
        self._rng = rng or get_global_rng()

    def federated_average(
        self, client_weights: List[np.ndarray], client_sizes: Optional[List[int]] = None
    ) -> np.ndarray:
        """FedAvg: Aggregate client model weights into global model.

        Based on FedAvg algorithm from the research paper.

        Args:
            client_weights: List of weight arrays from each client
            client_sizes: Optional list of dataset sizes per client for weighted averaging

        Returns:
            Aggregated global model weights
        """
        if self.aggregation_method == "weighted_fedavg" and client_sizes:
            total_samples = sum(client_sizes)
            weights_list = [size / total_samples for size in client_sizes]
            result: np.ndarray = np.average(client_weights, axis=0, weights=weights_list)
            return result
        else:
            result_mean: np.ndarray = np.mean(client_weights, axis=0)
            return result_mean

    def train_federated(
        self, client_data: List[np.ndarray], num_rounds: int = 10
    ) -> Dict[str, Any]:
        """Train using federated learning.

        Args:
            client_data: List of local datasets (one per client)
            num_rounds: Number of federated learning rounds

        Returns:
            Training results including global model and metrics
        """
        round_losses: List[float] = []
        results: Dict[str, Any] = {
            "round_losses": round_losses,
            "privacy_preserved": True,
            "num_clients": len(client_data),
            "num_rounds": num_rounds,
        }

        if self.global_model is None:
            self.global_model = self._rng.randn(10)

        for round_idx in range(num_rounds):
            client_weights: List[np.ndarray] = []
            client_sizes: List[int] = []

            for client_idx, data in enumerate(client_data):
                local_weights = self._local_training(data, self.global_model)
                client_weights.append(local_weights)
                client_sizes.append(len(data))

            self.global_model = self.federated_average(client_weights, client_sizes)

            round_loss = np.mean([(np.mean((w - self.global_model) ** 2)) for w in client_weights])
            round_losses.append(float(round_loss))

        results["final_model"] = self.global_model
        return results

    def _local_training(self, local_data: np.ndarray, global_weights: np.ndarray) -> np.ndarray:
        """Simulate local training on client data.

        Args:
            local_data: Client's local dataset
            global_weights: Current global model weights

        Returns:
            Updated local model weights
        """
        updated_weights = global_weights + self._rng.randn(*global_weights.shape) * 0.01
        return updated_weights
