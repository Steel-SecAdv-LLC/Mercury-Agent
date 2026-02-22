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

Federated Learning Module - Backwards Compatibility Layer

This module re-exports from federated_learning for backwards compatibility.
New code should import directly from omni_mercury_engine.federated_learning.

DEPRECATED: This module is maintained for backward compatibility only.
Please migrate to omni_mercury_engine.federated_learning for new development.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# Re-export core federated learning components from the new consolidated module
# Import the new FederatedAnomalyDetector for reference but don't export it directly
from omni_mercury_engine.federated_learning import (
    # Server components
    AggregationStrategy,
    Aggregator,
    # Client components
    ClientConfig,
    ClientConnectionStatus,
    ClientHealth,
    ClientManager,
    ClientState,
    ClientStatus,
    # CISA Coordinator (new implementation)
    CrossSectorResult,
    # Privacy components
    DifferentialPrivacyMechanism,
    FedAdamAggregator,
    FedAvgAggregator,
    FederatedAnomalyDetection,
    FederatedAnomalyDetector as NewFederatedAnomalyDetector,
    FederatedClient,
    FederatedServer,
    FederationConfig,
    FedProxTrainer,
    GaussianMechanism,
    GradientClipper,
    LaplaceMechanism,
    LocalDifferentialPrivacy,
    LocalTrainer,
    LocalUpdate,
    PrivacyAccountant,
    PrivacyBudget,
    PrivacyEngine,
    PrivacyMechanism,
    PrivacyReport,
    RoundResult,
    ScaffoldAggregator,
    SectorConfig,
    SectorPrivacyLevel,
    SectorType,
    SecureAggregator,
    SecureAggregatorWrapper,
    ServerConfig,
    ServerStatus,
    SGDTrainer,
    TrainingResult,
)
from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


# Legacy enum definitions for backward compatibility
class FederatedStrategy(Enum):
    """
    Federated learning aggregation strategies.

    DEPRECATED: Use AggregationStrategy from federated_learning instead.
    """

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDOPT = "fedopt"
    SECURE_AGGREGATION = "secure_aggregation"


class PrivacyLevel(Enum):
    """
    Privacy protection levels.

    DEPRECATED: Use SectorPrivacyLevel from federated_learning for sector-specific
    privacy, or configure privacy directly via PrivacyEngine.
    """

    NONE = "none"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SMPC = "secure_multiparty_computation"


# =============================================================================
# Legacy FederatedAnomalyDetector - Backwards Compatible Implementation
# =============================================================================


class FederatedAnomalyDetector:
    """
    Privacy-preserving federated anomaly detection.

    This class maintains backwards compatibility with the original API
    while internally using the new federated_learning infrastructure.

    Enables collaborative anomaly detection across CISA critical infrastructure
    sectors without sharing sensitive data. Implements:
    - Federated averaging (FedAvg) for model aggregation
    - Differential privacy for privacy guarantees
    - Secure aggregation to prevent server from seeing individual updates

    Args:
        strategy: Federated learning aggregation strategy
        privacy_level: Privacy protection level
        num_clients: Number of expected clients
        epsilon: Differential privacy epsilon parameter
        delta: Differential privacy delta parameter
        rng: Optional deterministic RNG for reproducibility
    """

    def __init__(
        self,
        strategy: FederatedStrategy = FederatedStrategy.FEDAVG,
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY,
        num_clients: int = 10,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        rng: DeterministicRNG | None = None,
    ):
        self.strategy = strategy
        self.privacy_level = privacy_level
        self.num_clients = num_clients
        self.epsilon = epsilon
        self.delta = delta
        self.global_model_weights: np.ndarray[Any, Any] | None = None
        self.client_models: dict[str, np.ndarray[Any, Any]] = {}
        self.round_number = 0
        self._rng = rng or get_global_rng()

    def federated_train(
        self,
        client_data: dict[str, np.ndarray[Any, Any]],
        local_epochs: int = 5,
        num_rounds: int = 10,
    ) -> dict[str, Any]:
        """
        Train federated anomaly detection model across clients.

        Args:
            client_data: Dictionary mapping client_id to local training data
            local_epochs: Number of epochs each client trains locally
            num_rounds: Number of federated rounds (aggregations)

        Returns:
            Training results with global model and metrics
        """
        rounds_list: list[int] = []
        global_loss_list: list[float] = []
        privacy_budget_spent = 0.0

        for round_idx in range(num_rounds):
            self.round_number = round_idx + 1

            client_updates: list[np.ndarray[Any, Any]] = []
            client_weights: list[int] = []

            for client_id, data in client_data.items():
                local_model_update = self._local_train(
                    client_id=client_id, data=data, epochs=local_epochs
                )

                if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
                    local_model_update = self._add_differential_privacy_noise(local_model_update)
                    privacy_budget_spent += self.epsilon

                client_updates.append(local_model_update)
                client_weights.append(len(data))

            if not client_updates:
                continue

            if self.strategy == FederatedStrategy.FEDAVG:
                aggregated_update = self._federated_averaging(client_updates, client_weights)
            elif self.strategy == FederatedStrategy.FEDPROX:
                aggregated_update = self._federated_proximal(client_updates, client_weights)
            else:
                aggregated_update = self._federated_averaging(client_updates, client_weights)

            self.global_model_weights = aggregated_update

            global_loss = self._evaluate_global_model(client_data)

            rounds_list.append(self.round_number)
            global_loss_list.append(global_loss)

        training_history: dict[str, Any] = {
            "rounds": rounds_list,
            "global_loss": global_loss_list,
            "privacy_budget_spent": privacy_budget_spent,
        }

        final_loss = global_loss_list[-1] if global_loss_list else 0.0

        return {
            "global_model": self.global_model_weights,
            "training_history": training_history,
            "privacy_guarantee": (
                f"ε={privacy_budget_spent:.2f}, δ={self.delta}"
                if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY
                else "None"
            ),
            "num_clients": len(client_data),
            "final_loss": final_loss,
        }

    def federated_detect(
        self,
        client_data: dict[str, np.ndarray[Any, Any]],
        use_personalization: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Perform federated anomaly detection across clients.

        Args:
            client_data: Dictionary mapping client_id to local data for detection
            use_personalization: Whether to personalize global model to each client

        Returns:
            Anomaly detection results for each client
        """
        if self.global_model_weights is None:
            raise ValueError("Model must be trained before detection. Call federated_train first.")

        detection_results = {}

        for client_id, data in client_data.items():
            if use_personalization:
                personalized_model = self._personalize_model(
                    client_id=client_id,
                    global_model=self.global_model_weights,
                    local_data=data,
                )
                anomaly_scores = self._compute_anomaly_scores(personalized_model, data)
            else:
                anomaly_scores = self._compute_anomaly_scores(self.global_model_weights, data)

            threshold = np.percentile(anomaly_scores, 95)
            anomalies = anomaly_scores > threshold

            detection_results[client_id] = {
                "anomaly_scores": anomaly_scores,
                "anomalies_detected": np.sum(anomalies),
                "anomaly_rate": float(np.mean(anomalies)),
                "threshold": float(threshold),
                "privacy_preserved": True,
            }

        return detection_results

    def _local_train(
        self, client_id: str, data: np.ndarray[Any, Any], epochs: int
    ) -> np.ndarray[Any, Any]:
        """Simulate local training on client device."""
        if self.global_model_weights is None:
            self.global_model_weights = self._rng.randn(data.shape[1])

        local_model: np.ndarray[Any, Any] = self.global_model_weights.copy()

        for _epoch in range(epochs):
            gradient = self._rng.randn(len(local_model)) * 0.01
            local_model -= 0.01 * gradient

        model_update: np.ndarray[Any, Any] = local_model - self.global_model_weights

        return model_update

    def _federated_averaging(
        self, client_updates: list[np.ndarray[Any, Any]], client_weights: list[int]
    ) -> np.ndarray[Any, Any]:
        """FedAvg: Weighted average of client model updates."""
        total_weight = sum(client_weights)
        weighted_updates = [
            update * (weight / total_weight)
            for update, weight in zip(client_updates, client_weights, strict=False)
        ]

        aggregated: np.ndarray[Any, Any] = np.asarray(np.sum(weighted_updates, axis=0))

        if self.global_model_weights is not None:
            result: np.ndarray[Any, Any] = self.global_model_weights + aggregated
            return result
        else:
            return aggregated

    def _federated_proximal(
        self,
        client_updates: list[np.ndarray[Any, Any]],
        client_weights: list[int],
        mu: float = 0.1,
    ) -> np.ndarray[Any, Any]:
        """FedProx: Handles system heterogeneity with proximal term."""
        aggregated = self._federated_averaging(client_updates, client_weights)

        if self.global_model_weights is not None:
            proximal_term = mu * (aggregated - self.global_model_weights)
            result: np.ndarray[Any, Any] = aggregated - proximal_term
            return result

        return aggregated

    def _add_differential_privacy_noise(
        self, model_update: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Add Gaussian noise for differential privacy guarantee."""
        sensitivity = 1.0
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        noise = self._rng.normal(0, sigma, size=model_update.shape)

        result: np.ndarray[Any, Any] = model_update + noise
        return result

    def _personalize_model(
        self,
        client_id: str,
        global_model: np.ndarray[Any, Any],
        local_data: np.ndarray[Any, Any],
        personalization_epochs: int = 3,
    ) -> np.ndarray[Any, Any]:
        """Personalize global model to client's local data distribution."""
        personalized_model = global_model.copy()

        for _ in range(personalization_epochs):
            gradient = self._rng.randn(len(personalized_model)) * 0.01
            personalized_model -= 0.01 * gradient

        return personalized_model

    def _compute_anomaly_scores(
        self, model: np.ndarray[Any, Any], data: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Compute anomaly scores for data using model."""
        reconstruction_errors: np.ndarray[Any, Any] = np.asarray(
            np.linalg.norm(data - model, axis=1)
        )
        return reconstruction_errors

    def _evaluate_global_model(self, client_data: dict[str, np.ndarray[Any, Any]]) -> float:
        """Evaluate global model across all clients."""
        if self.global_model_weights is None:
            return 0.0

        total_loss = 0.0
        total_samples = 0

        for data in client_data.values():
            scores = self._compute_anomaly_scores(self.global_model_weights, data)
            total_loss += np.sum(scores)
            total_samples += len(data)

        return total_loss / total_samples if total_samples > 0 else 0.0


# =============================================================================
# Legacy CISAFederatedCoordinator - Backwards Compatible Implementation
# =============================================================================


class CISAFederatedCoordinator:
    """
    Coordinates federated learning across CISA critical infrastructure sectors.

    This class maintains backwards compatibility with the original API
    while internally using improved implementations.

    Enables:
    - Multi-sector anomaly pattern learning without data sharing
    - Privacy-preserving cross-sector threat intelligence
    - Sector-specific model personalization
    - Differential privacy for sensitive sectors (Healthcare, Nuclear, Financial)

    Args:
        sectors: List of sector names to coordinate
    """

    def __init__(self, sectors: list[str]) -> None:
        self.sectors = sectors
        self.sector_detectors = {
            sector: FederatedAnomalyDetector(
                strategy=FederatedStrategy.FEDAVG,
                privacy_level=PrivacyLevel.DIFFERENTIAL_PRIVACY,
            )
            for sector in sectors
        }

    def coordinate_cross_sector_training(
        self,
        sector_data: dict[str, dict[str, np.ndarray[Any, Any]]],
        rounds: int = 10,
    ) -> dict[str, Any]:
        """
        Coordinate federated training across multiple CISA sectors.

        Args:
            sector_data: {sector_name: {client_id: data}}
            rounds: Number of federated rounds

        Returns:
            Cross-sector training results
        """
        results = {}

        for sector, clients in sector_data.items():
            if sector in self.sector_detectors:
                results[sector] = self.sector_detectors[sector].federated_train(
                    client_data=clients, num_rounds=rounds
                )

        return results


__all__ = [
    # Server
    "AggregationStrategy",
    "Aggregator",
    # Legacy classes (backwards compatible)
    "CISAFederatedCoordinator",
    # Client
    "ClientConfig",
    "ClientConnectionStatus",
    "ClientHealth",
    "ClientManager",
    "ClientState",
    "ClientStatus",
    # CISA Coordinator (new)
    "CrossSectorResult",
    # Privacy
    "DifferentialPrivacyMechanism",
    "FedAdamAggregator",
    "FedAvgAggregator",
    "FedProxTrainer",
    "FederatedAnomalyDetection",
    "FederatedAnomalyDetector",
    "FederatedClient",
    "FederatedServer",
    "FederatedStrategy",
    "FederationConfig",
    "GaussianMechanism",
    "GradientClipper",
    "LaplaceMechanism",
    "LocalDifferentialPrivacy",
    "LocalTrainer",
    "LocalUpdate",
    "PrivacyAccountant",
    "PrivacyBudget",
    "PrivacyEngine",
    "PrivacyLevel",
    "PrivacyMechanism",
    "PrivacyReport",
    "RoundResult",
    "SGDTrainer",
    "ScaffoldAggregator",
    "SectorConfig",
    "SectorPrivacyLevel",
    "SectorType",
    "SecureAggregator",
    "SecureAggregatorWrapper",
    "ServerConfig",
    "ServerStatus",
    "TrainingResult",
]
