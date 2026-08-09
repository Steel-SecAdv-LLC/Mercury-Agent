# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Federated Learning Server for Mercury Agent.

Implements federated aggregation algorithms including FedAvg, FedProx,
and secure aggregation with differential privacy.

References:
- McMahan et al. (2017): Communication-Efficient Learning of Deep Networks
- Li et al. (2020): Federated Optimization in Heterogeneous Networks
- Bonawitz et al. (2017): Practical Secure Aggregation
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    sanitize_domain,
)
from omni_mercury_engine.federated_learning.client import (
    ClientManager,
    FederatedClient,
    LocalUpdate,
)
from omni_mercury_engine.federated_learning.gosnn_coupling import (
    GOSNNCouplingClient,
    GOSNNCouplingServer,
)
from omni_mercury_engine.federated_learning.privacy import (
    PrivacyEngine,
    PrivacyReport,
    SecureAggregator,
)
from omni_mercury_engine.security.sigma_immutable_gate import (
    SigmaImmutableGate,
    enforce_dual_ethical_gate,
    get_sigma_immutable_gate,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Federated aggregation strategies."""

    FEDAVG = auto()
    FEDPROX = auto()
    FEDADAM = auto()
    SCAFFOLD = auto()
    SECURE_AGG = auto()


class ServerStatus(Enum):
    """Status of the federated server."""

    IDLE = auto()
    COLLECTING = auto()
    AGGREGATING = auto()
    DISTRIBUTING = auto()


@dataclass
class ServerConfig:
    """Configuration for federated server."""

    n_rounds: int = 100
    min_clients: int = 2
    client_fraction: float = 1.0
    aggregation_strategy: str = "fedavg"
    use_privacy: bool = True
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    server_learning_rate: float = 1.0
    server_momentum: float = 0.0
    timeout: float = 300.0


@dataclass
class RoundResult:
    """Result of a federated round."""

    round_num: int
    global_weights: np.ndarray[Any, Any]
    n_clients: int
    total_samples: int
    avg_loss: float
    client_losses: dict[str, float]
    aggregation_time: float
    privacy_spent: tuple[float, float] | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of federated training."""

    final_weights: np.ndarray[Any, Any]
    n_rounds: int
    total_time: float
    round_results: list[RoundResult]
    final_metrics: dict[str, float]
    privacy_report: PrivacyReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Aggregator(ABC):
    """Base class for federated aggregators."""

    @abstractmethod
    def aggregate(
        self,
        global_weights: np.ndarray[Any, Any],
        updates: list[LocalUpdate],
    ) -> np.ndarray[Any, Any]:
        """Aggregate client updates into new global weights."""
        pass


class FedAvgAggregator(Aggregator):
    """Federated Averaging (FedAvg) aggregator.

    Computes weighted average of client updates based on sample counts.
    """

    def __init__(self, learning_rate: float = 1.0) -> None:
        """Initialize FedAvg aggregator."""
        self._learning_rate = learning_rate

    def aggregate(
        self,
        global_weights: np.ndarray[Any, Any],
        updates: list[LocalUpdate],
    ) -> np.ndarray[Any, Any]:
        """Aggregate using weighted averaging."""
        if not updates:
            return global_weights

        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            return global_weights

        aggregated_update = np.zeros_like(global_weights)
        for update in updates:
            weight = update.n_samples / total_samples
            aggregated_update += weight * update.model_update

        new_weights = global_weights + self._learning_rate * aggregated_update
        return new_weights


class FedAdamAggregator(Aggregator):
    """FedAdam aggregator with adaptive learning rates.

    Applies Adam optimizer on the server side for aggregation.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        tau: float = 1e-3,
    ) -> None:
        """Initialize FedAdam aggregator.

        Args:
            learning_rate: Server learning rate
            beta1: First moment decay
            beta2: Second moment decay
            epsilon: Numerical stability
            tau: Server aggregate-gradient (FedOpt delta) scaling
        """
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._tau = tau

        self._m: np.ndarray[Any, Any] | None = None
        self._v: np.ndarray[Any, Any] | None = None
        self._t = 0

    def aggregate(
        self,
        global_weights: np.ndarray[Any, Any],
        updates: list[LocalUpdate],
    ) -> np.ndarray[Any, Any]:
        """Aggregate using FedAdam."""
        if not updates:
            return global_weights

        if self._m is None:
            self._m = np.zeros_like(global_weights)
            self._v = np.zeros_like(global_weights)

        assert self._v is not None

        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            return global_weights

        delta = np.zeros_like(global_weights)
        for update in updates:
            weight = update.n_samples / total_samples
            delta += weight * update.model_update

        self._t += 1

        self._m = self._beta1 * self._m + (1 - self._beta1) * delta
        assert self._v is not None
        self._v = self._beta2 * self._v + (1 - self._beta2) * (delta**2)

        m_hat = self._m / (1 - self._beta1**self._t)
        v_hat = self._v / (1 - self._beta2**self._t)

        new_weights = global_weights + self._lr * m_hat / (np.sqrt(v_hat) + self._tau)
        return new_weights


class ScaffoldAggregator(Aggregator):
    """SCAFFOLD aggregator with variance reduction.

    Uses control variates to reduce client drift.
    """

    def __init__(self, learning_rate: float = 1.0) -> None:
        """Initialize SCAFFOLD aggregator."""
        self._learning_rate = learning_rate
        self._server_control: np.ndarray[Any, Any] | None = None
        self._client_controls: dict[str, np.ndarray[Any, Any]] = {}

    def aggregate(
        self,
        global_weights: np.ndarray[Any, Any],
        updates: list[LocalUpdate],
    ) -> np.ndarray[Any, Any]:
        """Aggregate using SCAFFOLD."""
        if not updates:
            return global_weights

        if self._server_control is None:
            self._server_control = np.zeros_like(global_weights)

        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            return global_weights

        aggregated_update = np.zeros_like(global_weights)
        delta_controls = []

        for update in updates:
            weight = update.n_samples / total_samples
            aggregated_update += weight * update.model_update

            if update.client_id not in self._client_controls:
                self._client_controls[update.client_id] = np.zeros_like(global_weights)

            old_control = self._client_controls[update.client_id]
            new_control = old_control - self._server_control + update.model_update
            delta_controls.append(new_control - old_control)
            self._client_controls[update.client_id] = new_control

        if delta_controls:
            avg_delta = np.mean(delta_controls, axis=0)
            self._server_control = (
                self._server_control
                + (len(updates) / max(1, len(self._client_controls))) * avg_delta
            )

        new_weights = global_weights + self._learning_rate * aggregated_update
        return new_weights


class SecureAggregatorWrapper(Aggregator):
    """Secure aggregation with differential privacy.

    Wraps FedAvg with DP noise for privacy-preserving aggregation.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        min_clients: int = 3,
        seed: int | None = None,
    ) -> None:
        """Initialize secure aggregator."""
        self._secure_agg = SecureAggregator(
            mechanism="gaussian",
            epsilon=epsilon,
            delta=delta,
            min_clients=min_clients,
            seed=seed,
        )
        self._max_grad_norm = max_grad_norm

    def aggregate(
        self,
        global_weights: np.ndarray[Any, Any],
        updates: list[LocalUpdate],
    ) -> np.ndarray[Any, Any]:
        """Aggregate with secure aggregation and DP."""
        if not updates:
            return global_weights

        update_arrays = [u.model_update for u in updates]
        weights = [float(u.n_samples) for u in updates]

        aggregated = self._secure_agg.aggregate(
            update_arrays,
            weights=weights,
            sensitivity=self._max_grad_norm,
        )

        return global_weights + aggregated

    def get_privacy_spent(self) -> tuple[float, float]:
        """Get privacy spent."""
        return self._secure_agg.get_privacy_spent()


class FederatedServer:
    """Federated Learning Server.

    Orchestrates federated training by managing clients, coordinating
    rounds, and aggregating updates.

    Example:
        server = FederatedServer(
            initial_weights=model.get_weights(),
            config=ServerConfig(
                n_rounds=100,
                min_clients=5,
                use_privacy=True,
            ),
        )

        # Register clients
        for client in clients:
            server.register_client(client)

        # Run federated training
        result = server.train()

        # Get final model
        final_weights = result.final_weights
    """

    def __init__(
        self,
        initial_weights: np.ndarray[Any, Any],
        config: ServerConfig | None = None,
        eval_fn: Callable[[np.ndarray[Any, Any]], dict[str, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize federated server.

        Args:
            initial_weights: Initial model weights
            config: Server configuration
            eval_fn: Optional evaluation function for global model
            seed: Optional seed for the embedded :class:`ClientManager`'s
                per-instance numpy ``Generator``. With a seed set, the
                ``random`` and ``weighted`` client-selection strategies
                become deterministic across runs; without it, selection
                order varies. Pass an explicit seed for reproducible
                federated experiments.
        """
        self._global_weights = initial_weights.copy()
        self._config = config or ServerConfig()
        self._eval_fn = eval_fn
        self._seed = seed

        self._client_manager = ClientManager(
            min_clients=self._config.min_clients,
            selection_fraction=self._config.client_fraction,
            seed=seed,
        )

        self._aggregator = self._create_aggregator()

        self._status = ServerStatus.IDLE
        self._current_round = 0
        self._round_results: list[RoundResult] = []

        self._privacy_engine: PrivacyEngine | None = None
        if self._config.use_privacy and self._config.aggregation_strategy != "secure_agg":
            self._privacy_engine = PrivacyEngine(
                epsilon=self._config.epsilon,
                delta=self._config.delta,
                max_grad_norm=self._config.max_grad_norm,
                seed=seed,
            )

        # σ_Immutable Wave C: round-level dual ethical gate + bidirectional
        # GOSNN coupling, both constructed eagerly so the first round cannot
        # race them into existence.  The benevolence floor is clamped to
        # ``MINIMUM_BENEVOLENCE_FLOOR`` to match the engine/orchestrator
        # boundary contract.
        self._ethics_domain = sanitize_domain(None)
        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR
        )
        self._sigma_immutable_gate: SigmaImmutableGate = get_sigma_immutable_gate()
        # Bidirectional GOSNN coupling server, seeded from the same initial
        # weights as the FL global model so the publish → ingest → aggregate
        # → receive round-trip stays shape-locked to the model.
        self._gosnn_server = GOSNNCouplingServer(initial_weights=self._global_weights)

    def _create_aggregator(self) -> Aggregator:
        """Create aggregator based on strategy."""
        strategy = self._config.aggregation_strategy.lower()

        if strategy == "fedadam":
            return FedAdamAggregator(learning_rate=self._config.server_learning_rate)
        elif strategy == "scaffold":
            return ScaffoldAggregator(learning_rate=self._config.server_learning_rate)
        elif strategy == "secure_agg":
            return SecureAggregatorWrapper(
                epsilon=self._config.epsilon,
                delta=self._config.delta,
                max_grad_norm=self._config.max_grad_norm,
                min_clients=self._config.min_clients,
                seed=self._seed,
            )
        else:
            return FedAvgAggregator(learning_rate=self._config.server_learning_rate)

    def register_client(self, client: FederatedClient) -> None:
        """Register a client with the server."""
        self._client_manager.register(client)

    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client."""
        return self._client_manager.unregister(client_id)

    @property
    def global_weights(self) -> np.ndarray[Any, Any]:
        """Get current global model weights."""
        return self._global_weights.copy()

    @property
    def n_clients(self) -> int:
        """Get number of registered clients."""
        return len(self._client_manager.get_all_clients())

    def train(
        self,
        n_rounds: int | None = None,
        early_stopping: bool = False,
        patience: int = 10,
    ) -> TrainingResult:
        """Run federated training.

        Args:
            n_rounds: Number of rounds (defaults to config)
            early_stopping: Whether to use early stopping
            patience: Rounds without improvement before stopping

        Returns:
            TrainingResult with final model and metrics
        """
        if n_rounds is None:
            n_rounds = self._config.n_rounds

        start_time = time.time()
        best_loss = float("inf")
        rounds_without_improvement = 0

        for round_num in range(n_rounds):
            self._current_round = round_num + 1
            logger.info(f"Starting round {self._current_round}/{n_rounds}")

            round_result = self._execute_round()
            self._round_results.append(round_result)

            if early_stopping:
                if round_result.avg_loss < best_loss:
                    best_loss = round_result.avg_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= patience:
                    logger.info(f"Early stopping at round {self._current_round}")
                    break

            logger.info(
                f"Round {self._current_round}: "
                f"loss={round_result.avg_loss:.6f}, "
                f"clients={round_result.n_clients}"
            )

        total_time = time.time() - start_time

        final_metrics = {}
        if self._eval_fn is not None:
            final_metrics = self._eval_fn(self._global_weights)

        privacy_report = None
        if self._privacy_engine is not None:
            privacy_report = self._privacy_engine.get_privacy_report()
        elif isinstance(self._aggregator, SecureAggregatorWrapper):
            eps, delta = self._aggregator.get_privacy_spent()
            privacy_report = PrivacyReport(
                mechanism="gaussian",
                epsilon=self._config.epsilon,
                delta=self._config.delta,
                spent_epsilon=eps,
                spent_delta=delta,
                n_queries=len(self._round_results),
                composition_method="secure_aggregation",
                estimated_remaining_queries=0,
                noise_multiplier=1.0,
            )

        return TrainingResult(
            final_weights=self._global_weights.copy(),
            n_rounds=len(self._round_results),
            total_time=total_time,
            round_results=self._round_results,
            final_metrics=final_metrics,
            privacy_report=privacy_report,
        )

    def _enforce_round_ethics(self, n_clients: int, total_samples: int) -> None:
        """Round-level benevolence + σ_Immutable dual hard ethical gate.

        The gated decision is the real one — this boundary carries no caller
        free text, so the subject is the operation plus the round's own
        provenance (round number, client count, sample count). It used to be a
        hand-written positive-keyword string asserting the round was benign,
        which told the gate nothing it could check. Severity / anomaly_prob
        stay benign. Both gates fail closed; a violation raises
        :class:`EthicalConstraintViolationError` and the round halts. Invoked
        at round granularity **outside** the per-client ``try/except`` in
        :meth:`_execute_round` so a violation cannot be swallowed as a single
        client's failure.
        """
        from omni_mercury_engine.cognitive.decision_gate import DecisionSubject

        round_provenance = {
            "round": self._current_round,
            "n_clients": n_clients,
            "total_samples": total_samples,
        }
        enforce_dual_ethical_gate(
            subject=DecisionSubject(
                surface="FederatedServer._execute_round",
                operation=(
                    "aggregate client updates into a new global model for one "
                    "federated training round"
                ),
                domain=self._ethics_domain,
                payload=round_provenance,
            ),
            sigma_gate=self._sigma_immutable_gate,
            advisory_scorer=self._benevolence_scorer,
            boundary="FederatedServer._execute_round",
            domain=self._ethics_domain,
            extra_details=round_provenance,
        )

    def _route_round_through_gosnn(self, updates: list[LocalUpdate]) -> np.ndarray[Any, Any]:
        """Route a round's client weights through the bidirectional GOSNN coupling.

        ``publish → ingest → aggregate → receive``:

        * Each client's **absolute** post-training weights (``global +
          model_update``) are published with a SHA3-256 digest and ingested
          under shape / digest / round verification.
        * Unit-learning-rate FedAvg aggregates through the coupling's own
          digested weighted mean — mathematically identical to
          ``global + Σ wᵢ·model_updateᵢ`` because the per-client FedAvg
          weights sum to one — so the weights flow through the *digested
          FedAvg path*.  Every other strategy (FedAdam / SCAFFOLD /
          secure-aggregation / non-unit-LR FedAvg) stays authoritative for
          the numerics and is installed + broadcast through the coupling.
        * Every contributing client then ``receive``s the new digest-checked
          global state, closing the previously one-way (server → client) loop.

        Runs **outside** the per-client ``try/except`` in
        :meth:`_execute_round`, so a :class:`GOSNNCouplingError` (digest /
        shape / round mismatch) fails the whole round closed.
        """
        original_dtype = self._global_weights.dtype
        round_num = self._gosnn_server.round_num
        coupling_clients: list[GOSNNCouplingClient] = []
        for update in updates:
            # Reconstruct each client's absolute post-training weights from
            # the FedAvg delta the client published.
            absolute = self._global_weights + update.model_update
            client = GOSNNCouplingClient(update.client_id, self._global_weights)
            published = client.publish(
                round_num=round_num,
                local_update=absolute,
                n_samples=update.n_samples,
            )
            # Fail closed on shape / digest / round mismatch.
            self._gosnn_server.ingest(published)
            coupling_clients.append(client)

        use_digested_fedavg = (
            isinstance(self._aggregator, FedAvgAggregator)
            and float(getattr(self._aggregator, "_learning_rate", 1.0)) == 1.0
        )
        if use_digested_fedavg:
            global_state = self._gosnn_server.aggregate()
        else:
            # Strategy-specific aggregation stays authoritative; the coupling
            # still installs + broadcasts the result so the bidirectional
            # round-trip closes and the integrity digest is published.
            new_weights = self._aggregator.aggregate(self._global_weights, updates)
            global_state = self._gosnn_server.install_global_state(
                new_weights, [u.client_id for u in updates]
            )

        # Bidirectional closure: every contributing client receives the new
        # digest-checked global state (fail closed on shape / digest mismatch).
        for client in coupling_clients:
            client.receive(global_state)

        return np.asarray(global_state.weights, dtype=original_dtype)

    def _execute_round(self) -> RoundResult:
        """Execute a single federated round."""
        self._status = ServerStatus.COLLECTING

        selected_clients = self._client_manager.select_clients()

        if len(selected_clients) < self._config.min_clients:
            logger.warning(
                f"Insufficient clients: {len(selected_clients)} < {self._config.min_clients}"
            )

        updates: list[LocalUpdate] = []
        client_losses: dict[str, float] = {}

        for client in selected_clients:
            try:
                update = client.train_round(
                    self._global_weights,
                    round_num=self._current_round,
                )
                updates.append(update)
                if update.loss_history:
                    client_losses[client.client_id] = update.loss_history[-1]
            except Exception as e:
                logger.error(f"Client {client.client_id} failed: {e}")

        if not updates:
            logger.warning("No updates received from clients")
            return RoundResult(
                round_num=self._current_round,
                global_weights=self._global_weights.copy(),
                n_clients=0,
                total_samples=0,
                avg_loss=float("inf"),
                client_losses={},
                aggregation_time=0.0,
            )

        # σ_Immutable Wave C round-level dual hard ethical gate.  Runs OUTSIDE
        # the per-client try/except above so a benevolence- or σ_Immutable-
        # violation fails the whole round closed instead of being swallowed
        # as one client's error.
        self._enforce_round_ethics(
            n_clients=len(updates),
            total_samples=sum(u.n_samples for u in updates),
        )

        self._status = ServerStatus.AGGREGATING
        agg_start = time.time()

        # Route the round's weights through the bidirectional GOSNN coupling
        # (publish → ingest → aggregate → receive) with SHA3-256 + shape +
        # round integrity, replacing the prior one-way aggregate-only path.
        self._global_weights = self._route_round_through_gosnn(updates)

        agg_time = time.time() - agg_start

        total_samples = sum(u.n_samples for u in updates)
        avg_loss = (
            np.mean([u.loss_history[-1] for u in updates if u.loss_history])
            if any(u.loss_history for u in updates)
            else 0.0
        )

        privacy_spent = None
        if self._privacy_engine is not None:
            privacy_spent = self._privacy_engine.get_privacy_spent()
        elif isinstance(self._aggregator, SecureAggregatorWrapper):
            privacy_spent = self._aggregator.get_privacy_spent()

        metrics = {}
        if self._eval_fn is not None:
            metrics = self._eval_fn(self._global_weights)

        self._status = ServerStatus.IDLE

        return RoundResult(
            round_num=self._current_round,
            global_weights=self._global_weights.copy(),
            n_clients=len(updates),
            total_samples=total_samples,
            avg_loss=float(avg_loss),
            client_losses=client_losses,
            aggregation_time=agg_time,
            privacy_spent=privacy_spent,
            metrics=metrics,
        )

    def get_round_results(self) -> list[RoundResult]:
        """Get results from all completed rounds."""
        return self._round_results.copy()

    def get_client_stats(self) -> dict[str, Any]:
        """Get aggregated client statistics."""
        return self._client_manager.aggregate_stats()


class FederatedAnomalyDetector:
    """Federated Anomaly Detection system.

    High-level interface for training anomaly detection models
    in a federated setting with privacy guarantees.

    Example:
        detector = FederatedAnomalyDetector(
            model_dim=100,
            n_rounds=50,
            use_privacy=True,
            epsilon=1.0,
        )

        # Add client data
        detector.add_client("client_1", X_train_1)
        detector.add_client("client_2", X_train_2)

        # Train federated model
        detector.fit()

        # Detect anomalies
        scores = detector.predict(X_test)
    """

    def __init__(
        self,
        model_dim: int,
        n_rounds: int = 50,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        use_privacy: bool = True,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        aggregation: str = "fedavg",
        seed: int | None = None,
    ) -> None:
        """Initialize federated anomaly detector.

        Args:
            model_dim: Dimension of model (feature dimension)
            n_rounds: Number of federated rounds
            local_epochs: Local training epochs per round
            learning_rate: Learning rate
            use_privacy: Whether to use differential privacy
            epsilon: Privacy budget
            delta: Privacy parameter delta
            aggregation: Aggregation strategy
            seed: Optional seed for the per-instance numpy ``Generator``
                that controls **all** stochastic surfaces inside the
                federated training loop:

                - the initial server-weights draw (``standard_normal``);
                - the per-client :class:`SGDTrainer` minibatch-shuffle
                  seed (one distinct sub-seed per ``add_client`` call);
                - the per-client :class:`PrivacyEngine` Gaussian /
                  Laplace noise seed (one distinct sub-seed per
                  ``add_client`` call when ``use_privacy=True``);
                - the embedded :class:`ClientManager`'s
                  ``random`` / ``weighted`` selection seed; and
                - the server-side :class:`PrivacyEngine` /
                  :class:`SecureAggregatorWrapper` noise seed.

                With an explicit seed, two runs with the same
                ``add_client`` order produce identical final weights
                regardless of ``use_privacy``. The legacy global
                ``np.random`` state is never used.
        """
        self._model_dim = model_dim
        self._n_rounds = n_rounds
        self._local_epochs = local_epochs
        self._learning_rate = learning_rate
        self._use_privacy = use_privacy
        self._epsilon = epsilon
        self._delta = delta
        self._aggregation = aggregation
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self._weights: np.ndarray[Any, Any] | None = None
        self._mean: np.ndarray[Any, Any] | None = None
        self._std: np.ndarray[Any, Any] | None = None
        self._threshold: float = 0.0

        self._clients: list[FederatedClient] = []
        self._server: FederatedServer | None = None
        self._result: TrainingResult | None = None

    def add_client(
        self,
        client_id: str,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any] | None = None,
    ) -> None:
        """Add a client with local data.

        Args:
            client_id: Unique client identifier
            X: Local training data
            y: Optional labels

        Notes:
            When the detector was constructed with an explicit ``seed``,
            each client receives an :class:`SGDTrainer` with a distinct
            sub-seed drawn deterministically from ``self._rng``, so
            minibatch-shuffle order is reproducible across runs and
            distinct between clients within a single run.
        """
        from omni_mercury_engine.federated_learning.client import (
            ClientConfig,
            SGDTrainer,
        )

        config = ClientConfig(
            client_id=client_id,
            local_epochs=self._local_epochs,
            learning_rate=self._learning_rate,
            use_privacy=self._use_privacy,
            epsilon=self._epsilon,
            delta=self._delta,
        )

        client_trainer_seed = int(self._rng.integers(0, 2**31 - 1))
        client_privacy_seed = int(self._rng.integers(0, 2**31 - 1))
        trainer = SGDTrainer(seed=client_trainer_seed)

        client = FederatedClient(
            client_id=client_id,
            local_data=(X, y),
            config=config,
            trainer=trainer,
            privacy_seed=client_privacy_seed,
        )
        self._clients.append(client)

    def fit(self) -> TrainingResult:
        """Train the federated anomaly detector.

        Returns:
            TrainingResult with training metrics
        """
        if not self._clients:
            raise ValueError("No clients registered")

        initial_weights = self._rng.standard_normal(self._model_dim) * 0.01
        server_seed = int(self._rng.integers(0, 2**31 - 1))

        config = ServerConfig(
            n_rounds=self._n_rounds,
            min_clients=max(1, len(self._clients) // 2),
            client_fraction=1.0,
            aggregation_strategy=self._aggregation,
            use_privacy=self._use_privacy,
            epsilon=self._epsilon,
            delta=self._delta,
        )

        self._server = FederatedServer(
            initial_weights=initial_weights,
            config=config,
            eval_fn=self._evaluate_global,
            seed=server_seed,
        )

        for client in self._clients:
            self._server.register_client(client)

        self._result = self._server.train()
        self._weights = self._result.final_weights

        self._compute_statistics()

        return self._result

    def _compute_statistics(self) -> None:
        """Compute global statistics for anomaly scoring."""
        all_data = []
        for client in self._clients:
            X, _ = client._local_data
            all_data.append(X)

        if all_data:
            combined = np.vstack(all_data)
            self._mean = np.mean(combined, axis=0)
            self._std = np.std(combined, axis=0) + 1e-10

            scores = self._compute_scores(combined)
            self._threshold = np.percentile(scores, 95)

    def _compute_scores(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute anomaly scores."""
        if self._mean is None:
            return np.zeros(len(X))

        z_scores = np.abs((X - self._mean) / self._std)
        return np.mean(z_scores, axis=1)

    def _evaluate_global(self, weights: np.ndarray[Any, Any]) -> dict[str, float]:
        """Evaluate global model."""
        return {"weight_norm": float(np.linalg.norm(weights))}

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict anomaly labels.

        Args:
            X: Data to score

        Returns:
            Binary anomaly labels (1 = anomaly)
        """
        scores = self.decision_function(X)
        return (scores > self._threshold).astype(int)

    def decision_function(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute anomaly scores.

        Args:
            X: Data to score

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        return self._compute_scores(X)

    def get_privacy_report(self) -> PrivacyReport | None:
        """Get privacy report from training."""
        if self._result is not None:
            return self._result.privacy_report
        return None
