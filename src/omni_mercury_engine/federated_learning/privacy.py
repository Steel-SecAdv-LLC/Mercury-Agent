"""
Differential Privacy Mechanisms for Federated Learning.

Implements various differential privacy techniques including Gaussian
and Laplace mechanisms, gradient clipping, and privacy accounting.

References:
- Dwork & Roth (2014): The Algorithmic Foundations of Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy
- McMahan et al. (2018): A General Approach to Adding Differential Privacy
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    """Types of differential privacy mechanisms."""

    GAUSSIAN = auto()
    LAPLACE = auto()
    EXPONENTIAL = auto()
    DISCRETE_GAUSSIAN = auto()


@dataclass
class PrivacyBudget:
    """Privacy budget tracker using (epsilon, delta) accounting."""

    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0

    @property
    def remaining_epsilon(self) -> float:
        """Remaining epsilon budget."""
        return max(0.0, self.epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        """Remaining delta budget."""
        return max(0.0, self.delta - self.spent_delta)

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.spent_epsilon >= self.epsilon or self.spent_delta >= self.delta

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if we can spend the given budget."""
        return (
            self.spent_epsilon + epsilon <= self.epsilon and
            self.spent_delta + delta <= self.delta
        )

    def spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Spend privacy budget. Returns False if insufficient."""
        if not self.can_spend(epsilon, delta):
            return False
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        return True


@dataclass
class PrivacyAccountant:
    """
    Privacy accountant for tracking cumulative privacy loss.

    Supports multiple composition theorems including basic, advanced,
    and Renyi differential privacy (RDP) composition.
    """

    total_epsilon: float
    total_delta: float
    composition: str = "rdp"
    noise_multiplier: float = 1.0
    sample_rate: float = 1.0

    _queries: list[tuple[float, float]] = field(default_factory=list)
    _rdp_orders: np.ndarray = field(
        default_factory=lambda: np.array([1.5, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64])
    )
    _rdp_eps: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Initialize RDP epsilon array."""
        self._rdp_eps = np.zeros(len(self._rdp_orders))

    def add_query(self, sensitivity: float, noise_scale: float) -> tuple[float, float]:
        """
        Record a query and return its privacy cost.

        Args:
            sensitivity: L2 sensitivity of the query
            noise_scale: Standard deviation of noise added

        Returns:
            (epsilon, delta) for this query
        """
        if self.composition == "rdp":
            epsilon, delta = self._rdp_query(sensitivity, noise_scale)
        elif self.composition == "advanced":
            epsilon, delta = self._advanced_composition_query(sensitivity, noise_scale)
        else:
            epsilon, delta = self._basic_composition_query(sensitivity, noise_scale)

        self._queries.append((epsilon, delta))
        return epsilon, delta

    def _rdp_query(self, sensitivity: float, noise_scale: float) -> tuple[float, float]:
        """Compute privacy cost using Renyi DP."""
        sigma = noise_scale / (sensitivity + 1e-10)

        for i, order in enumerate(self._rdp_orders):
            rdp_eps = self._compute_rdp_single(order, sigma, self.sample_rate)
            self._rdp_eps[i] += rdp_eps

        epsilon = self._rdp_to_dp(self._rdp_eps, self.total_delta)
        return epsilon, self.total_delta

    def _compute_rdp_single(self, order: float, sigma: float, q: float) -> float:
        """Compute RDP for a single Gaussian mechanism query."""
        if sigma <= 0:
            return float("inf")

        if q == 0:
            return 0.0

        if q == 1:
            return order / (2 * sigma ** 2)

        log_terms = []
        for k in range(int(order) + 1):
            log_coeff = (
                math.lgamma(order + 1) -
                math.lgamma(k + 1) -
                math.lgamma(order - k + 1)
            )
            log_term = (
                log_coeff +
                k * math.log(q) +
                (order - k) * math.log(1 - q) +
                k * (k - 1) / (2 * sigma ** 2)
            )
            log_terms.append(log_term)

        max_log = max(log_terms)
        sum_exp = sum(math.exp(lt - max_log) for lt in log_terms)

        rdp = (max_log + math.log(sum_exp)) / (order - 1)
        return max(0.0, rdp)

    def _rdp_to_dp(self, rdp_eps: np.ndarray, delta: float) -> float:
        """Convert RDP to (epsilon, delta)-DP."""
        if delta <= 0:
            return float("inf")

        epsilons = []
        for i, order in enumerate(self._rdp_orders):
            if order <= 1:
                continue
            eps = rdp_eps[i] - (math.log(delta) + math.log(order)) / (order - 1) + \
                  math.log((order - 1) / order)
            epsilons.append(eps)

        return min(epsilons) if epsilons else float("inf")

    def _advanced_composition_query(
        self,
        sensitivity: float,
        noise_scale: float,
    ) -> tuple[float, float]:
        """Advanced composition theorem."""
        sigma = noise_scale / (sensitivity + 1e-10)
        single_eps = sensitivity / (sigma * math.sqrt(2 * math.log(1.25 / self.total_delta)))

        k = len(self._queries) + 1
        composed_eps = math.sqrt(2 * k * math.log(1 / self.total_delta)) * single_eps + \
                       k * single_eps * (math.exp(single_eps) - 1)

        return composed_eps, self.total_delta

    def _basic_composition_query(
        self,
        sensitivity: float,
        noise_scale: float,
    ) -> tuple[float, float]:
        """Basic composition theorem."""
        sigma = noise_scale / (sensitivity + 1e-10)
        single_eps = sensitivity / (sigma * math.sqrt(2 * math.log(1.25 / self.total_delta)))

        total_eps = sum(e for e, _ in self._queries) + single_eps
        total_delta = sum(d for _, d in self._queries) + self.total_delta / len(self._queries)

        return total_eps, total_delta

    def get_current_epsilon(self) -> float:
        """Get current total epsilon."""
        if not self._queries:
            return 0.0
        return max(e for e, _ in self._queries)

    def remaining_queries(self, target_epsilon: float | None = None) -> int:
        """Estimate remaining queries before exceeding budget."""
        if target_epsilon is None:
            target_epsilon = self.total_epsilon

        current_eps = self.get_current_epsilon()
        if current_eps >= target_epsilon:
            return 0

        if not self._queries:
            return 1000

        avg_eps = current_eps / len(self._queries)
        if avg_eps <= 0:
            return 1000

        remaining = int((target_epsilon - current_eps) / avg_eps)
        return max(0, remaining)


class DifferentialPrivacyMechanism(ABC):
    """Base class for differential privacy mechanisms."""

    @abstractmethod
    def add_noise(
        self,
        value: np.ndarray,
        sensitivity: float,
        epsilon: float,
    ) -> np.ndarray:
        """Add noise to satisfy differential privacy."""
        pass

    @abstractmethod
    def compute_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Compute required noise scale for given privacy parameters."""
        pass


class GaussianMechanism(DifferentialPrivacyMechanism):
    """
    Gaussian mechanism for (epsilon, delta)-differential privacy.

    Adds Gaussian noise calibrated to the L2 sensitivity.
    """

    def __init__(self, delta: float = 1e-5) -> None:
        """Initialize Gaussian mechanism."""
        self._delta = delta
        self._rng = np.random.default_rng()

    def add_noise(
        self,
        value: np.ndarray,
        sensitivity: float,
        epsilon: float,
    ) -> np.ndarray:
        """Add Gaussian noise to the value."""
        sigma = self.compute_noise_scale(sensitivity, epsilon)
        noise = self._rng.normal(0, sigma, size=value.shape)
        return value + noise

    def compute_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Compute noise standard deviation for Gaussian mechanism."""
        if epsilon <= 0:
            return float("inf")

        c = math.sqrt(2 * math.log(1.25 / self._delta))
        return c * sensitivity / epsilon


class LaplaceMechanism(DifferentialPrivacyMechanism):
    """
    Laplace mechanism for epsilon-differential privacy.

    Adds Laplace noise calibrated to the L1 sensitivity.
    """

    def __init__(self) -> None:
        """Initialize Laplace mechanism."""
        self._rng = np.random.default_rng()

    def add_noise(
        self,
        value: np.ndarray,
        sensitivity: float,
        epsilon: float,
    ) -> np.ndarray:
        """Add Laplace noise to the value."""
        scale = self.compute_noise_scale(sensitivity, epsilon)
        noise = self._rng.laplace(0, scale, size=value.shape)
        return value + noise

    def compute_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Compute noise scale for Laplace mechanism."""
        if epsilon <= 0:
            return float("inf")
        return sensitivity / epsilon


class GradientClipper:
    """
    Gradient clipping for differential privacy in deep learning.

    Clips per-sample gradients to bound sensitivity before aggregation.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: str = "l2",
    ) -> None:
        """
        Initialize gradient clipper.

        Args:
            max_norm: Maximum gradient norm
            norm_type: "l2" or "l_inf"
        """
        self._max_norm = max_norm
        self._norm_type = norm_type

    def clip(self, gradients: np.ndarray) -> np.ndarray:
        """
        Clip gradients to bounded norm.

        Args:
            gradients: Gradient array (n_samples, n_params) or (n_params,)

        Returns:
            Clipped gradients
        """
        if gradients.ndim == 1:
            return self._clip_single(gradients)

        clipped = np.zeros_like(gradients)
        for i in range(len(gradients)):
            clipped[i] = self._clip_single(gradients[i])

        return clipped

    def _clip_single(self, gradient: np.ndarray) -> np.ndarray:
        """Clip a single gradient vector."""
        if self._norm_type == "l2":
            norm = np.linalg.norm(gradient)
            if norm > self._max_norm:
                return gradient * (self._max_norm / norm)
        elif self._norm_type == "l_inf":
            return np.clip(gradient, -self._max_norm, self._max_norm)

        return gradient

    @property
    def sensitivity(self) -> float:
        """Get the L2 sensitivity bound."""
        return self._max_norm


class SecureAggregator:
    """
    Secure aggregation for federated learning.

    Implements secure sum with differential privacy, ensuring
    the server only learns the aggregate, not individual updates.
    """

    def __init__(
        self,
        mechanism: str = "gaussian",
        epsilon: float = 1.0,
        delta: float = 1e-5,
        min_clients: int = 3,
    ) -> None:
        """
        Initialize secure aggregator.

        Args:
            mechanism: "gaussian" or "laplace"
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta (for Gaussian)
            min_clients: Minimum clients for aggregation
        """
        self._epsilon = epsilon
        self._delta = delta
        self._min_clients = min_clients

        if mechanism == "laplace":
            self._mechanism = LaplaceMechanism()
        else:
            self._mechanism = GaussianMechanism(delta)

        self._accountant = PrivacyAccountant(
            total_epsilon=epsilon * 100,
            total_delta=delta,
            composition="rdp",
        )

    def aggregate(
        self,
        updates: list[np.ndarray],
        weights: list[float] | None = None,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """
        Securely aggregate client updates with DP noise.

        Args:
            updates: List of client updates (gradient or model delta)
            weights: Optional weights for weighted average
            sensitivity: Sensitivity bound (e.g., from gradient clipping)

        Returns:
            Aggregated update with DP noise
        """
        if len(updates) < self._min_clients:
            raise ValueError(
                f"Need at least {self._min_clients} clients, got {len(updates)}"
            )

        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        aggregated = np.zeros_like(updates[0])
        for update, weight in zip(updates, weights):
            aggregated += weight * update

        per_client_epsilon = self._epsilon / math.sqrt(len(updates))

        noisy_aggregate = self._mechanism.add_noise(
            aggregated,
            sensitivity=sensitivity / len(updates),
            epsilon=per_client_epsilon,
        )

        self._accountant.add_query(sensitivity, per_client_epsilon)

        return noisy_aggregate

    def get_privacy_spent(self) -> tuple[float, float]:
        """Get total privacy spent."""
        return self._accountant.get_current_epsilon(), self._delta


class LocalDifferentialPrivacy:
    """
    Local Differential Privacy for client-side privatization.

    Each client adds noise locally before sending to the server,
    providing stronger privacy guarantees than central DP.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        mechanism: str = "gaussian",
        delta: float = 1e-5,
    ) -> None:
        """Initialize LDP mechanism."""
        self._epsilon = epsilon
        self._delta = delta

        if mechanism == "laplace":
            self._mechanism = LaplaceMechanism()
        else:
            self._mechanism = GaussianMechanism(delta)

    def privatize(
        self,
        value: np.ndarray,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """
        Apply local differential privacy to a value.

        Args:
            value: Value to privatize
            sensitivity: Sensitivity of the value

        Returns:
            Privatized value
        """
        return self._mechanism.add_noise(value, sensitivity, self._epsilon)

    def privatize_gradients(
        self,
        gradients: np.ndarray,
        max_norm: float = 1.0,
    ) -> np.ndarray:
        """
        Privatize gradients with clipping and noise.

        Args:
            gradients: Gradient array
            max_norm: Maximum gradient norm for clipping

        Returns:
            Privatized gradients
        """
        clipper = GradientClipper(max_norm)
        clipped = clipper.clip(gradients)
        return self.privatize(clipped, sensitivity=max_norm)


@dataclass
class PrivacyReport:
    """Report on privacy guarantees and budget usage."""

    mechanism: str
    epsilon: float
    delta: float
    spent_epsilon: float
    spent_delta: float
    n_queries: int
    composition_method: str
    estimated_remaining_queries: int
    noise_multiplier: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "n_queries": self.n_queries,
            "composition_method": self.composition_method,
            "estimated_remaining_queries": self.estimated_remaining_queries,
            "noise_multiplier": self.noise_multiplier,
            "details": self.details,
        }


class PrivacyEngine:
    """
    High-level privacy engine for federated learning.

    Manages privacy budget, gradient clipping, and noise injection.

    Example:
        engine = PrivacyEngine(
            epsilon=8.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.1,
        )

        # During training
        for batch in data_loader:
            gradients = compute_gradients(model, batch)
            private_gradients = engine.privatize_gradients(gradients)
            apply_gradients(model, private_gradients)

        # Check privacy
        report = engine.get_privacy_report()
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        mechanism: str = "gaussian",
        composition: str = "rdp",
    ) -> None:
        """
        Initialize privacy engine.

        Args:
            epsilon: Total privacy budget (epsilon)
            delta: Privacy parameter delta
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise scale multiplier
            mechanism: "gaussian" or "laplace"
            composition: Privacy composition method
        """
        self._epsilon = epsilon
        self._delta = delta
        self._max_grad_norm = max_grad_norm
        self._noise_multiplier = noise_multiplier
        self._mechanism_type = mechanism

        self._clipper = GradientClipper(max_grad_norm)

        if mechanism == "laplace":
            self._mechanism = LaplaceMechanism()
        else:
            self._mechanism = GaussianMechanism(delta)

        self._accountant = PrivacyAccountant(
            total_epsilon=epsilon,
            total_delta=delta,
            composition=composition,
            noise_multiplier=noise_multiplier,
        )

        self._n_queries = 0

    def privatize_gradients(
        self,
        gradients: np.ndarray,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Privatize gradients with clipping and noise.

        Args:
            gradients: Per-sample or aggregated gradients
            batch_size: Batch size for noise scaling

        Returns:
            Privatized gradients
        """
        clipped = self._clipper.clip(gradients)

        if gradients.ndim > 1:
            aggregated = np.mean(clipped, axis=0)
        else:
            aggregated = clipped

        noise_scale = self._noise_multiplier * self._max_grad_norm
        if batch_size is not None and batch_size > 0:
            noise_scale /= batch_size

        private_grads = self._mechanism.add_noise(
            aggregated,
            sensitivity=self._max_grad_norm,
            epsilon=self._epsilon / (self._n_queries + 1),
        )

        self._accountant.add_query(self._max_grad_norm, noise_scale)
        self._n_queries += 1

        return private_grads

    def privatize_model_update(
        self,
        update: np.ndarray,
        sensitivity: float | None = None,
    ) -> np.ndarray:
        """
        Privatize a model update (weights delta).

        Args:
            update: Model update vector
            sensitivity: Update sensitivity (defaults to max_grad_norm)

        Returns:
            Privatized update
        """
        if sensitivity is None:
            sensitivity = self._max_grad_norm

        private_update = self._mechanism.add_noise(
            update,
            sensitivity=sensitivity,
            epsilon=self._epsilon / (self._n_queries + 1),
        )

        self._accountant.add_query(sensitivity, self._noise_multiplier * sensitivity)
        self._n_queries += 1

        return private_update

    def get_privacy_spent(self) -> tuple[float, float]:
        """Get current privacy expenditure."""
        return self._accountant.get_current_epsilon(), self._delta

    def get_privacy_report(self) -> PrivacyReport:
        """Generate comprehensive privacy report."""
        spent_eps = self._accountant.get_current_epsilon()

        return PrivacyReport(
            mechanism=self._mechanism_type,
            epsilon=self._epsilon,
            delta=self._delta,
            spent_epsilon=spent_eps,
            spent_delta=self._delta,
            n_queries=self._n_queries,
            composition_method=self._accountant.composition,
            estimated_remaining_queries=self._accountant.remaining_queries(self._epsilon),
            noise_multiplier=self._noise_multiplier,
            details={
                "max_grad_norm": self._max_grad_norm,
                "total_queries": len(self._accountant._queries),
            },
        )

    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self._accountant.get_current_epsilon() >= self._epsilon
