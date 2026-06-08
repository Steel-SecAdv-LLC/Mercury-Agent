# Copyright (C) 2025 Steel Security Advisors LLC
"""Mercury-native federated anomaly detection.

Enables privacy-preserving, decentralized training by exchanging sufficient statistics between nodes
instead of raw data.
"""

from omni_mercury_engine.federation.aggregator import FederatedAggregator
from omni_mercury_engine.federation.node import FederatedNode
from omni_mercury_engine.federation.privacy import DifferentialPrivacy
from omni_mercury_engine.federation.statistics import FittedStatistics

__all__ = [
    "DifferentialPrivacy",
    "FederatedAggregator",
    "FederatedNode",
    "FittedStatistics",
]
