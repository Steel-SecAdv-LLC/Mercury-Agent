# Copyright (C) 2025 Steel Security Advisors LLC
"""Federated Learning Module for Mercury Agent.

Provides privacy-preserving distributed machine learning with support for
differential privacy, secure aggregation, and heterogeneous client data.

Key Components:
- FederatedServer: Orchestrates federated training rounds
- FederatedClient: Handles local training and privacy enforcement
- PrivacyEngine: Manages differential privacy budget and mechanisms
- FederatedAnomalyDetector: High-level interface for federated anomaly detection
- CISAFederatedCoordinator: Cross-sector coordination for CISA infrastructure

Enhanced with:
- Timeout handling for unresponsive clients
- Network partition detection and recovery
- Byzantine fault tolerance for malicious clients
- Graceful degradation under partial failures

References:
- McMahan et al. (2017): Communication-Efficient Learning of Deep Networks
- Li et al. (2020): Federated Optimization in Heterogeneous Networks
- Abadi et al. (2016): Deep Learning with Differential Privacy
- Bonawitz et al. (2017): Practical Secure Aggregation
"""

from omni_mercury_engine.federated_learning.cisa_coordinator import (
    CISAFederatedCoordinator,
    CrossSectorResult,
    SectorConfig,
    SectorPrivacyLevel,
    SectorType,
)
from omni_mercury_engine.federated_learning.client import (
    ClientConfig,
    ClientConnectionStatus,
    ClientHealth,
    ClientManager,
    ClientState,
    ClientStatus,
    FederatedClient,
    FederationConfig,
    FedProxTrainer,
    LocalTrainer,
    LocalUpdate,
    SGDTrainer,
)
from omni_mercury_engine.federated_learning.federated_robust import (
    FederatedAnomalyDetection,
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
    # CISA Coordinator
    "CISAFederatedCoordinator",
    "ClientConfig",
    # Fault Tolerance
    "ClientConnectionStatus",
    "ClientHealth",
    "ClientManager",
    "ClientState",
    # Client
    "ClientStatus",
    "CrossSectorResult",
    "DifferentialPrivacyMechanism",
    "FedAdamAggregator",
    "FedAvgAggregator",
    "FedProxTrainer",
    "FederatedAnomalyDetection",
    "FederatedAnomalyDetector",
    "FederatedClient",
    "FederatedServer",
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
    # Privacy
    "PrivacyMechanism",
    "PrivacyReport",
    "RoundResult",
    "SGDTrainer",
    "ScaffoldAggregator",
    # Sector Types
    "SectorConfig",
    "SectorPrivacyLevel",
    "SectorType",
    "SecureAggregator",
    "SecureAggregatorWrapper",
    "ServerConfig",
    "ServerStatus",
    "TrainingResult",
]
