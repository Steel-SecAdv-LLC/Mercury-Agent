"""
Federated Learning Client for Mercury Agent.

Implements local training and update computation for federated learning
with support for differential privacy and secure communication.

References:
- McMahan et al. (2017): Communication-Efficient Learning of Deep Networks
- Li et al. (2020): Federated Optimization in Heterogeneous Networks (FedProx)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.federated_learning.privacy import (
    LocalDifferentialPrivacy,
    PrivacyEngine,
    PrivacyReport,
)


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class ClientStatus(Enum):
    """Status of a federated client."""

    IDLE = auto()
    TRAINING = auto()
    UPLOADING = auto()
    WAITING = auto()
    OFFLINE = auto()


@dataclass
class ClientConfig:
    """Configuration for a federated learning client."""

    client_id: str
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    use_privacy: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    proximal_mu: float = 0.0
    min_samples: int = 10


@dataclass
class LocalUpdate:
    """Local model update from a client."""

    client_id: str
    model_update: np.ndarray
    n_samples: int
    n_epochs: int
    loss_history: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    privacy_report: PrivacyReport | None = None
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientState:
    """State of a federated client."""

    client_id: str
    status: ClientStatus
    current_round: int
    total_samples: int
    total_updates: int
    cumulative_loss: float
    last_update_time: float
    privacy_spent: tuple[float, float] = (0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalTrainer(ABC):
    """Base class for local model trainers."""

    @abstractmethod
    def train(
        self,
        model_weights: np.ndarray,
        data: tuple[np.ndarray, np.ndarray | None],
        config: ClientConfig,
    ) -> tuple[np.ndarray, list[float]]:
        """Train locally and return weight update."""
        pass


class SGDTrainer(LocalTrainer):
    """Stochastic Gradient Descent trainer."""

    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        Initialize SGD trainer.

        Args:
            loss_fn: Loss function(weights, X, y) -> loss
            grad_fn: Gradient function(weights, X, y) -> gradients
        """
        self._loss_fn = loss_fn or self._default_loss
        self._grad_fn = grad_fn or self._default_gradient

    def _default_loss(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Default MSE loss."""
        predictions = X @ weights
        return float(np.mean((predictions - y) ** 2))

    def _default_gradient(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Default gradient for MSE loss."""
        predictions = X @ weights
        return 2 * X.T @ (predictions - y) / len(y)

    def train(
        self,
        model_weights: np.ndarray,
        data: tuple[np.ndarray, np.ndarray | None],
        config: ClientConfig,
    ) -> tuple[np.ndarray, list[float]]:
        """Train using SGD."""
        X, y = data
        if y is None:
            y = np.zeros(len(X))

        weights = model_weights.copy()
        losses = []
        velocity = np.zeros_like(weights)

        n_samples = len(X)
        n_batches = max(1, n_samples // config.batch_size)

        for epoch in range(config.local_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * config.batch_size
                end = min(start + config.batch_size, n_samples)
                batch_indices = indices[start:end]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                gradient = self._grad_fn(weights, X_batch, y_batch)

                if config.weight_decay > 0:
                    gradient += config.weight_decay * weights

                if config.momentum > 0:
                    velocity = config.momentum * velocity + gradient
                    weights -= config.learning_rate * velocity
                else:
                    weights -= config.learning_rate * gradient

                batch_loss = self._loss_fn(weights, X_batch, y_batch)
                epoch_loss += batch_loss

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

        update = weights - model_weights
        return update, losses


class FedProxTrainer(LocalTrainer):
    """
    FedProx trainer with proximal regularization.

    Adds proximal term to handle heterogeneous data.
    """

    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Initialize FedProx trainer."""
        self._loss_fn = loss_fn or self._default_loss
        self._grad_fn = grad_fn or self._default_gradient

    def _default_loss(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Default MSE loss."""
        predictions = X @ weights
        return float(np.mean((predictions - y) ** 2))

    def _default_gradient(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Default gradient for MSE loss."""
        predictions = X @ weights
        return 2 * X.T @ (predictions - y) / len(y)

    def train(
        self,
        model_weights: np.ndarray,
        data: tuple[np.ndarray, np.ndarray | None],
        config: ClientConfig,
    ) -> tuple[np.ndarray, list[float]]:
        """Train using FedProx with proximal term."""
        X, y = data
        if y is None:
            y = np.zeros(len(X))

        weights = model_weights.copy()
        global_weights = model_weights.copy()
        losses = []

        n_samples = len(X)
        n_batches = max(1, n_samples // config.batch_size)

        for epoch in range(config.local_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * config.batch_size
                end = min(start + config.batch_size, n_samples)
                batch_indices = indices[start:end]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                gradient = self._grad_fn(weights, X_batch, y_batch)

                if config.proximal_mu > 0:
                    gradient += config.proximal_mu * (weights - global_weights)

                weights -= config.learning_rate * gradient

                batch_loss = self._loss_fn(weights, X_batch, y_batch)
                if config.proximal_mu > 0:
                    prox_term = (config.proximal_mu / 2) * np.sum((weights - global_weights) ** 2)
                    batch_loss += prox_term

                epoch_loss += batch_loss

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

        update = weights - model_weights
        return update, losses


class FederatedClient:
    """
    Federated Learning Client.

    Handles local training, privacy enforcement, and communication
    with the federated server.

    Example:
        client = FederatedClient(
            client_id="client_1",
            local_data=(X_train, y_train),
            config=ClientConfig(
                client_id="client_1",
                local_epochs=5,
                use_privacy=True,
                epsilon=1.0,
            ),
        )

        # Receive global model and train locally
        update = client.train_round(global_weights, round_num=1)

        # Send update to server
        server.receive_update(update)
    """

    def __init__(
        self,
        client_id: str,
        local_data: tuple[np.ndarray, np.ndarray | None],
        config: ClientConfig | None = None,
        trainer: LocalTrainer | None = None,
    ) -> None:
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            local_data: Local training data (X, y) or (X, None) for unsupervised
            config: Client configuration
            trainer: Local trainer instance
        """
        self._client_id = client_id
        self._local_data = local_data
        self._config = config or ClientConfig(client_id=client_id)
        self._trainer = trainer or SGDTrainer()

        self._status = ClientStatus.IDLE
        self._current_round = 0
        self._total_updates = 0
        self._cumulative_loss = 0.0
        self._last_update_time = 0.0

        self._privacy_engine: PrivacyEngine | None = None
        if self._config.use_privacy:
            self._privacy_engine = PrivacyEngine(
                epsilon=self._config.epsilon,
                delta=self._config.delta,
                max_grad_norm=self._config.max_grad_norm,
                noise_multiplier=self._config.noise_multiplier,
            )

        self._local_ldp: LocalDifferentialPrivacy | None = None

    @property
    def client_id(self) -> str:
        """Get client ID."""
        return self._client_id

    @property
    def status(self) -> ClientStatus:
        """Get client status."""
        return self._status

    @property
    def n_samples(self) -> int:
        """Get number of local samples."""
        return len(self._local_data[0])

    def train_round(
        self,
        global_weights: np.ndarray,
        round_num: int,
    ) -> LocalUpdate:
        """
        Execute one round of local training.

        Args:
            global_weights: Current global model weights
            round_num: Current federated round number

        Returns:
            LocalUpdate with model update and metadata
        """
        if self.n_samples < self._config.min_samples:
            logger.warning(
                f"Client {self._client_id} has insufficient samples "
                f"({self.n_samples} < {self._config.min_samples})"
            )
            return LocalUpdate(
                client_id=self._client_id,
                model_update=np.zeros_like(global_weights),
                n_samples=0,
                n_epochs=0,
                timestamp=time.time(),
            )

        self._status = ClientStatus.TRAINING
        self._current_round = round_num
        start_time = time.time()

        update, losses = self._trainer.train(
            global_weights,
            self._local_data,
            self._config,
        )

        privacy_report = None
        if self._privacy_engine is not None:
            update = self._privacy_engine.privatize_model_update(
                update,
                sensitivity=self._config.max_grad_norm,
            )
            privacy_report = self._privacy_engine.get_privacy_report()

        self._status = ClientStatus.UPLOADING
        self._total_updates += 1
        self._cumulative_loss += losses[-1] if losses else 0.0
        self._last_update_time = time.time()

        local_update = LocalUpdate(
            client_id=self._client_id,
            model_update=update,
            n_samples=self.n_samples,
            n_epochs=self._config.local_epochs,
            loss_history=losses,
            metrics={
                "final_loss": losses[-1] if losses else 0.0,
                "training_time": time.time() - start_time,
            },
            privacy_report=privacy_report,
            timestamp=self._last_update_time,
        )

        self._status = ClientStatus.WAITING
        return local_update

    def get_state(self) -> ClientState:
        """Get current client state."""
        privacy_spent = (0.0, 0.0)
        if self._privacy_engine is not None:
            privacy_spent = self._privacy_engine.get_privacy_spent()

        return ClientState(
            client_id=self._client_id,
            status=self._status,
            current_round=self._current_round,
            total_samples=self.n_samples,
            total_updates=self._total_updates,
            cumulative_loss=self._cumulative_loss,
            last_update_time=self._last_update_time,
            privacy_spent=privacy_spent,
        )

    def update_data(
        self,
        new_data: tuple[np.ndarray, np.ndarray | None],
        append: bool = True,
    ) -> None:
        """
        Update local training data.

        Args:
            new_data: New data (X, y)
            append: If True, append to existing data; else replace
        """
        if append:
            X_old, y_old = self._local_data
            X_new, y_new = new_data

            X_combined = np.vstack([X_old, X_new])
            if y_old is not None and y_new is not None:
                y_combined = np.concatenate([y_old, y_new])
            else:
                y_combined = None

            self._local_data = (X_combined, y_combined)
        else:
            self._local_data = new_data

    def set_offline(self) -> None:
        """Mark client as offline."""
        self._status = ClientStatus.OFFLINE

    def set_online(self) -> None:
        """Mark client as online."""
        self._status = ClientStatus.IDLE

    def get_privacy_report(self) -> PrivacyReport | None:
        """Get privacy report if privacy is enabled."""
        if self._privacy_engine is not None:
            return self._privacy_engine.get_privacy_report()
        return None

    def is_privacy_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        if self._privacy_engine is not None:
            return self._privacy_engine.is_budget_exhausted()
        return False


class ClientManager:
    """
    Manager for multiple federated clients.

    Handles client registration, selection, and coordination.
    """

    def __init__(
        self,
        min_clients: int = 2,
        selection_fraction: float = 1.0,
        selection_strategy: str = "random",
    ) -> None:
        """
        Initialize client manager.

        Args:
            min_clients: Minimum clients required per round
            selection_fraction: Fraction of clients to select
            selection_strategy: "random", "round_robin", or "weighted"
        """
        self._min_clients = min_clients
        self._selection_fraction = selection_fraction
        self._selection_strategy = selection_strategy

        self._clients: dict[str, FederatedClient] = {}
        self._round_robin_idx = 0

    def register(self, client: FederatedClient) -> None:
        """Register a client."""
        self._clients[client.client_id] = client
        logger.info(f"Registered client {client.client_id}")

    def unregister(self, client_id: str) -> bool:
        """Unregister a client."""
        if client_id in self._clients:
            del self._clients[client_id]
            return True
        return False

    def get_client(self, client_id: str) -> FederatedClient | None:
        """Get client by ID."""
        return self._clients.get(client_id)

    def get_all_clients(self) -> list[FederatedClient]:
        """Get all registered clients."""
        return list(self._clients.values())

    def get_available_clients(self) -> list[FederatedClient]:
        """Get clients that are available for training."""
        return [
            c for c in self._clients.values()
            if c.status != ClientStatus.OFFLINE and not c.is_privacy_exhausted()
        ]

    def select_clients(self, n_clients: int | None = None) -> list[FederatedClient]:
        """
        Select clients for the current round.

        Args:
            n_clients: Number of clients to select (defaults to fraction)

        Returns:
            List of selected clients
        """
        available = self.get_available_clients()

        if len(available) < self._min_clients:
            logger.warning(
                f"Only {len(available)} clients available, "
                f"need {self._min_clients}"
            )
            return available

        if n_clients is None:
            n_clients = max(
                self._min_clients,
                int(len(available) * self._selection_fraction),
            )
        n_clients = min(n_clients, len(available))

        if self._selection_strategy == "random":
            indices = np.random.choice(len(available), n_clients, replace=False)
            return [available[i] for i in indices]

        elif self._selection_strategy == "round_robin":
            selected = []
            for i in range(n_clients):
                idx = (self._round_robin_idx + i) % len(available)
                selected.append(available[idx])
            self._round_robin_idx = (self._round_robin_idx + n_clients) % len(available)
            return selected

        elif self._selection_strategy == "weighted":
            weights = np.array([c.n_samples for c in available], dtype=float)
            weights /= weights.sum()
            indices = np.random.choice(
                len(available),
                n_clients,
                replace=False,
                p=weights,
            )
            return [available[i] for i in indices]

        return available[:n_clients]

    def aggregate_stats(self) -> dict[str, Any]:
        """Aggregate statistics across all clients."""
        clients = self.get_all_clients()
        if not clients:
            return {}

        total_samples = sum(c.n_samples for c in clients)
        total_updates = sum(c.get_state().total_updates for c in clients)

        status_counts = {}
        for c in clients:
            status = c.status.name
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "n_clients": len(clients),
            "n_available": len(self.get_available_clients()),
            "total_samples": total_samples,
            "total_updates": total_updates,
            "status_distribution": status_counts,
            "avg_samples_per_client": total_samples / len(clients),
        }
