"""
Mercury Agent
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
Federated Learning for Privacy-Preserving Anomaly Detection

DEPRECATED: This module is deprecated and maintained for backward compatibility only.
Please use omni_mercury_engine.federated_learning instead.

All functionality has been migrated to:
- omni_mercury_engine.federated_learning.server.FederatedAnomalyDetector
- omni_mercury_engine.federated_learning.cisa_coordinator.CISAFederatedCoordinator
- omni_mercury_engine.federated_learning.client.ClientConnectionStatus (was ClientStatus)
- omni_mercury_engine.federated_learning.client.ClientHealth
- omni_mercury_engine.federated_learning.client.FederationConfig

Migration guide:
    # Old import (deprecated)
    from omni_mercury_engine.federated.federated_detector import (
        FederatedAnomalyDetector,
        CISAFederatedCoordinator,
        ClientStatus,
        ClientHealth,
        FederationConfig,
        FederatedStrategy,
        PrivacyLevel,
    )

    # New import (recommended)
    from omni_mercury_engine.federated_learning import (
        FederatedAnomalyDetector,
        CISAFederatedCoordinator,
        ClientConnectionStatus,  # renamed from ClientStatus
        ClientHealth,
        FederationConfig,
        AggregationStrategy,  # replaces FederatedStrategy
        SectorPrivacyLevel,  # replaces PrivacyLevel
    )

Research sources:
- Flower Framework (https://flower.dev/)
- PySyft (https://github.com/OpenMined/PySyft)
- McMahan et al. "Communication-Efficient Learning" (2017)
"""

import warnings
from enum import Enum
from typing import Any

import numpy as np

# Re-export from new consolidated module for backward compatibility
from omni_mercury_engine.federated_learning import (
    CISAFederatedCoordinator,
    ClientConnectionStatus,
    ClientHealth,
    FederatedAnomalyDetector as _NewFederatedAnomalyDetector,
    FederationConfig,
)
from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

# Emit deprecation warning when module is imported
warnings.warn(
    "omni_mercury_engine.federated.federated_detector is deprecated. "
    "Please use omni_mercury_engine.federated_learning instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Legacy ClientStatus enum (different from new ClientStatus in federated_learning.client)
# The new module uses ClientConnectionStatus for this purpose
class ClientStatus(Enum):
    """
    Client connection status.

    DEPRECATED: Use ClientConnectionStatus from federated_learning.client instead.
    """

    CONNECTED = "connected"
    TIMEOUT = "timeout"
    PARTITIONED = "partitioned"
    BYZANTINE = "byzantine"
    DROPPED = "dropped"


class FederatedStrategy(Enum):
    """
    Federated learning aggregation strategies.

    DEPRECATED: Use AggregationStrategy from federated_learning.server instead.
    """

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDOPT = "fedopt"
    SECURE_AGGREGATION = "secure_aggregation"


class PrivacyLevel(Enum):
    """
    Privacy protection levels.

    DEPRECATED: Use SectorPrivacyLevel from federated_learning.cisa_coordinator instead.
    """

    NONE = "none"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SMPC = "secure_multiparty_computation"


class FederatedAnomalyDetector:
    """
    Privacy-preserving federated anomaly detection.

    DEPRECATED: This class is maintained for backward compatibility.
    Please use FederatedAnomalyDetector from federated_learning.server instead.

    This wrapper provides backward compatibility for code using the old interface.
    The new FederatedAnomalyDetector has a slightly different API.
    """

    def __init__(
        self,
        strategy: FederatedStrategy = FederatedStrategy.FEDAVG,
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY,
        num_clients: int = 10,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        rng: DeterministicRNG | None = None,
        federation_config: FederationConfig | None = None,
    ):
        """Initialize the deprecated FederatedAnomalyDetector."""
        warnings.warn(
            "FederatedAnomalyDetector from federated.federated_detector is deprecated. "
            "Use FederatedAnomalyDetector from federated_learning instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.strategy = strategy
        self.privacy_level = privacy_level
        self.num_clients = num_clients
        self.epsilon = epsilon
        self.delta = delta
        self.global_model_weights: np.ndarray[Any, Any] | None = None
        self.client_models: dict[str, np.ndarray[Any, Any]] = {}
        self.round_number = 0
        self._rng = rng or get_global_rng()

        # Timeout and partition handling configuration
        self.config = federation_config or FederationConfig()
        self.client_health: dict[str, ClientHealth] = {}
        self._partitioned_clients: set[str] = set()
        self._byzantine_clients: set[str] = set()

        # Map to new aggregation strategy
        strategy_map = {
            FederatedStrategy.FEDAVG: "fedavg",
            FederatedStrategy.FEDPROX: "fedprox",
            FederatedStrategy.FEDOPT: "fedadam",
            FederatedStrategy.SECURE_AGGREGATION: "secure_agg",
        }

        # Create internal new-style detector
        self._detector = _NewFederatedAnomalyDetector(
            model_dim=50,  # Default dimension
            n_rounds=10,
            use_privacy=privacy_level != PrivacyLevel.NONE,
            epsilon=epsilon,
            delta=delta,
            aggregation=strategy_map.get(strategy, "fedavg"),
        )

    def federated_train(
        self,
        client_data: dict[str, np.ndarray[Any, Any]],
        local_epochs: int = 5,
        num_rounds: int = 10,
    ) -> dict[str, Any]:
        """
        Train federated anomaly detection model across clients.

        DEPRECATED: Use the new FederatedAnomalyDetector.fit() method instead.
        """
        warnings.warn(
            "federated_train is deprecated. Use add_client and fit methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Initialize client health tracking
        for client_id in client_data:
            if client_id not in self.client_health:
                self.client_health[client_id] = ClientHealth(client_id=client_id)

        # Add clients to the new detector
        for client_id, data in client_data.items():
            self._detector.add_client(client_id, data)

        # Update rounds
        self._detector._n_rounds = num_rounds
        self._detector._local_epochs = local_epochs

        # Train
        result = self._detector.fit()

        # Store global model
        self.global_model_weights = result.final_weights
        self.round_number = result.n_rounds

        # Build legacy return format
        rounds_list = list(range(1, result.n_rounds + 1))
        global_loss_list = [r.avg_loss for r in result.round_results]

        privacy_budget_spent = 0.0
        if result.privacy_report:
            privacy_budget_spent = result.privacy_report.spent_epsilon

        return {
            "global_model": self.global_model_weights,
            "training_history": {
                "rounds": rounds_list,
                "global_loss": global_loss_list,
                "privacy_budget_spent": privacy_budget_spent,
                "timeout_count": 0,
                "byzantine_count": 0,
            },
            "privacy_guarantee": (
                f"epsilon={privacy_budget_spent:.2f}, delta={self.delta}"
                if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY
                else "None"
            ),
            "num_clients": len(client_data),
            "final_loss": global_loss_list[-1] if global_loss_list else 0.0,
            "partitioned_clients": list(self._partitioned_clients),
            "byzantine_clients": list(self._byzantine_clients),
            "timeout_count": 0,
        }

    def federated_detect(
        self, client_data: dict[str, np.ndarray[Any, Any]], use_personalization: bool = True
    ) -> dict[str, dict[str, Any]]:
        """
        Perform federated anomaly detection across clients.

        DEPRECATED: Use predict or decision_function methods instead.
        """
        warnings.warn(
            "federated_detect is deprecated. Use predict or decision_function instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if self.global_model_weights is None:
            raise ValueError("Model must be trained before detection. Call federated_train first.")

        detection_results = {}

        for client_id, data in client_data.items():
            anomaly_scores = self._detector.decision_function(data)
            threshold = np.percentile(anomaly_scores, 95)
            anomalies = anomaly_scores > threshold

            detection_results[client_id] = {
                "anomaly_scores": anomaly_scores,
                "anomalies_detected": int(np.sum(anomalies)),
                "anomaly_rate": float(np.mean(anomalies)),
                "threshold": float(threshold),
                "privacy_preserved": True,
            }

        return detection_results

    def get_client_health_report(self) -> dict[str, Any]:
        """Get health report for all tracked clients."""
        return {
            client_id: {
                "status": health.connection_status.value,
                "consecutive_timeouts": health.consecutive_timeouts,
                "total_rounds": health.total_rounds_participated,
                "suspicious_updates": health.suspicious_updates,
                "last_seen": health.last_seen,
            }
            for client_id, health in self.client_health.items()
        }

    def recover_partitioned_clients(self) -> list[str]:
        """Attempt to recover partitioned clients by resetting their status."""
        import time

        recovered = []
        current_time = time.time()

        for client_id in list(self._partitioned_clients):
            health = self.client_health.get(client_id)
            if health and (current_time - health.last_seen) > self.config.recovery_cooldown_seconds:
                health.connection_status = ClientConnectionStatus.CONNECTED
                health.consecutive_timeouts = 0
                self._partitioned_clients.discard(client_id)
                recovered.append(client_id)

        return recovered


__all__ = [
    "CISAFederatedCoordinator",
    "ClientHealth",
    "ClientStatus",
    "FederatedAnomalyDetector",
    "FederatedStrategy",
    "FederationConfig",
    "PrivacyLevel",
]
