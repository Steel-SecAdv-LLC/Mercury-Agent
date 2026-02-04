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

from omni_mercury_engine.federated_learning.privacy import (
    PrivacyMechanism,
    PrivacyBudget,
    PrivacyAccountant,
    PrivacyReport,
    PrivacyEngine,
    DifferentialPrivacyMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    GradientClipper,
    SecureAggregator,
    LocalDifferentialPrivacy,
)
from omni_mercury_engine.federated_learning.client import (
    ClientStatus,
    ClientConfig,
    ClientState,
    LocalUpdate,
    LocalTrainer,
    SGDTrainer,
    FedProxTrainer,
    FederatedClient,
    ClientManager,
)
from omni_mercury_engine.federated_learning.server import (
    AggregationStrategy,
    ServerStatus,
    ServerConfig,
    RoundResult,
    TrainingResult,
    Aggregator,
    FedAvgAggregator,
    FedAdamAggregator,
    ScaffoldAggregator,
    SecureAggregatorWrapper,
    FederatedServer,
    FederatedAnomalyDetector,
)

__all__ = [
    # Privacy
    "PrivacyMechanism",
    "PrivacyBudget",
    "PrivacyAccountant",
    "PrivacyReport",
    "PrivacyEngine",
    "DifferentialPrivacyMechanism",
    "GaussianMechanism",
    "LaplaceMechanism",
    "GradientClipper",
    "SecureAggregator",
    "LocalDifferentialPrivacy",
    # Client
    "ClientStatus",
    "ClientConfig",
    "ClientState",
    "LocalUpdate",
    "LocalTrainer",
    "SGDTrainer",
    "FedProxTrainer",
    "FederatedClient",
    "ClientManager",
    # Server
    "AggregationStrategy",
    "ServerStatus",
    "ServerConfig",
    "RoundResult",
    "TrainingResult",
    "Aggregator",
    "FedAvgAggregator",
    "FedAdamAggregator",
    "ScaffoldAggregator",
    "SecureAggregatorWrapper",
    "FederatedServer",
    "FederatedAnomalyDetector",
]
