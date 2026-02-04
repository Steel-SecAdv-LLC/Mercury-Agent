"""
Federated Learning Module for Mercury Agent.

Provides privacy-preserving distributed machine learning with support for
differential privacy, secure aggregation, and heterogeneous client data.

Key Components:
- FederatedServer: Orchestrates federated training rounds
- FederatedClient: Handles local training and privacy enforcement
- PrivacyEngine: Manages differential privacy budget and mechanisms
- FederatedAnomalyDetector: High-level interface for federated anomaly detection

References:
- McMahan et al. (2017): Communication-Efficient Learning of Deep Networks
- Li et al. (2020): Federated Optimization in Heterogeneous Networks
- Abadi et al. (2016): Deep Learning with Differential Privacy
- Bonawitz et al. (2017): Practical Secure Aggregation
"""

from omni_mercury_engine.federated_learning.client import (
    ClientConfig,
    ClientManager,
    ClientState,
    ClientStatus,
    FederatedClient,
    FedProxTrainer,
    LocalTrainer,
    LocalUpdate,
    SGDTrainer,
)
from omni_mercury_engine.federated_learning.privacy import (
    DifferentialPrivacyMechanism,
    GaussianMechanism,
    GradientClipper,
    LaplaceMechanism,
    LocalDifferentialPrivacy,
    PrivacyAccountant,
    PrivacyBudget,
    PrivacyEngine,
    PrivacyMechanism,
    PrivacyReport,
    SecureAggregator,
)
from omni_mercury_engine.federated_learning.server import (
    AggregationStrategy,
    Aggregator,
    FedAdamAggregator,
    FedAvgAggregator,
    FederatedAnomalyDetector,
    FederatedServer,
    RoundResult,
    ScaffoldAggregator,
    SecureAggregatorWrapper,
    ServerConfig,
    ServerStatus,
    TrainingResult,
)


__all__ = [
    # Server
    "AggregationStrategy",
    "Aggregator",
    "ClientConfig",
    "ClientManager",
    "ClientState",
    # Client
    "ClientStatus",
    "DifferentialPrivacyMechanism",
    "FedAdamAggregator",
    "FedAvgAggregator",
    "FedProxTrainer",
    "FederatedAnomalyDetector",
    "FederatedClient",
    "FederatedServer",
    "GaussianMechanism",
    "GradientClipper",
    "LaplaceMechanism",
    "LocalDifferentialPrivacy",
    "LocalTrainer",
    "LocalUpdate",
    "PrivacyAccountant",
    "PrivacyBudget",
    "PrivacyEngine",
    # Privacy
    "PrivacyMechanism",
    "PrivacyReport",
    "RoundResult",
    "SGDTrainer",
    "ScaffoldAggregator",
    "SecureAggregator",
    "SecureAggregatorWrapper",
    "ServerConfig",
    "ServerStatus",
    "TrainingResult",
]
