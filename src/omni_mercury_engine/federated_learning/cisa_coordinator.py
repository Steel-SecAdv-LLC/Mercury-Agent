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

CISA Federated Coordinator for Mercury Agent.

Coordinates federated learning across CISA critical infrastructure sectors,
enabling privacy-preserving cross-sector threat intelligence and anomaly detection.

Inspired by Flower/PySyft frameworks for distributed ML across CISA sectors.

References:
- Flower Framework (https://flower.dev/)
- PySyft (https://github.com/OpenMined/PySyft)
- McMahan et al. "Communication-Efficient Learning" (2017)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from omni_mercury_engine.federated_learning.server import (
    FederatedAnomalyDetector,
    TrainingResult,
)


logger = logging.getLogger(__name__)


class SectorPrivacyLevel(Enum):
    """Privacy protection levels for different CISA sectors."""

    NONE = auto()
    STANDARD = auto()
    ENHANCED = auto()
    MAXIMUM = auto()


class SectorType(Enum):
    """CISA Critical Infrastructure Sectors."""

    CHEMICAL = "chemical"
    COMMERCIAL_FACILITIES = "commercial_facilities"
    COMMUNICATIONS = "communications"
    CRITICAL_MANUFACTURING = "critical_manufacturing"
    DAMS = "dams"
    DEFENSE_INDUSTRIAL_BASE = "defense_industrial_base"
    EMERGENCY_SERVICES = "emergency_services"
    ENERGY = "energy"
    FINANCIAL_SERVICES = "financial_services"
    FOOD_AND_AGRICULTURE = "food_and_agriculture"
    GOVERNMENT_FACILITIES = "government_facilities"
    HEALTHCARE = "healthcare"
    INFORMATION_TECHNOLOGY = "information_technology"
    NUCLEAR = "nuclear"
    TRANSPORTATION = "transportation"
    WATER = "water"


@dataclass
class SectorConfig:
    """Configuration for a CISA sector in federated learning."""

    sector_name: str
    privacy_level: SectorPrivacyLevel = SectorPrivacyLevel.ENHANCED
    epsilon: float = 1.0
    delta: float = 1e-5
    min_clients: int = 2
    aggregation_strategy: str = "fedavg"
    local_epochs: int = 5
    learning_rate: float = 0.01

    # Sector-specific settings
    cross_sector_sharing: bool = True
    anonymize_sector_id: bool = False


@dataclass
class CrossSectorResult:
    """Results from cross-sector federated training."""

    sector_results: dict[str, TrainingResult]
    global_model_weights: np.ndarray | None
    total_clients: int
    total_samples: int
    participating_sectors: list[str]
    cross_sector_metrics: dict[str, float] = field(default_factory=dict)
    privacy_reports: dict[str, Any] = field(default_factory=dict)


class CISAFederatedCoordinator:
    """
    Coordinates federated learning across CISA critical infrastructure sectors.

    Enables:
    - Multi-sector anomaly pattern learning without data sharing
    - Privacy-preserving cross-sector threat intelligence
    - Sector-specific model personalization
    - Differential privacy for sensitive sectors (Healthcare, Nuclear, Financial)
    - Hierarchical aggregation (intra-sector then cross-sector)

    Example:
        coordinator = CISAFederatedCoordinator(
            sectors=["healthcare", "financial_services", "energy"]
        )

        # Configure sector-specific privacy
        coordinator.configure_sector(
            "healthcare",
            SectorConfig(
                sector_name="healthcare",
                privacy_level=SectorPrivacyLevel.MAXIMUM,
                epsilon=0.5,
            )
        )

        # Add clients to sectors
        coordinator.add_sector_client("healthcare", "hospital_1", X_train_1)
        coordinator.add_sector_client("healthcare", "hospital_2", X_train_2)
        coordinator.add_sector_client("financial_services", "bank_1", X_train_3)

        # Train across all sectors
        results = coordinator.coordinate_cross_sector_training(rounds=10)

        # Get sector-specific model for detection
        healthcare_model = coordinator.get_sector_model("healthcare")
    """

    # Sectors requiring enhanced privacy by default
    HIGH_PRIVACY_SECTORS = {
        SectorType.HEALTHCARE.value,
        SectorType.NUCLEAR.value,
        SectorType.FINANCIAL_SERVICES.value,
        SectorType.DEFENSE_INDUSTRIAL_BASE.value,
    }

    def __init__(
        self,
        sectors: list[str],
        model_dim: int = 50,
        default_config: SectorConfig | None = None,
    ) -> None:
        """
        Initialize CISA Federated Coordinator.

        Args:
            sectors: List of sector names to coordinate
            model_dim: Dimension of the model (feature dimension)
            default_config: Default configuration for sectors
        """
        self.sectors = sectors
        self.model_dim = model_dim
        self._default_config = default_config

        # Sector-specific configurations
        self._sector_configs: dict[str, SectorConfig] = {}

        # Sector detectors using FederatedAnomalyDetector from federated_learning
        self._sector_detectors: dict[str, FederatedAnomalyDetector] = {}

        # Track sector clients
        self._sector_clients: dict[str, list[tuple[str, np.ndarray, np.ndarray | None]]] = {
            sector: [] for sector in sectors
        }

        # Initialize sectors
        for sector in sectors:
            self._initialize_sector(sector)

        logger.info(f"Initialized CISAFederatedCoordinator with {len(sectors)} sectors")

    def _initialize_sector(self, sector: str) -> None:
        """Initialize a sector with appropriate privacy settings."""
        # Determine default privacy level based on sector type
        if sector in self.HIGH_PRIVACY_SECTORS:
            default_privacy = SectorPrivacyLevel.MAXIMUM
            default_epsilon = 0.5
        else:
            default_privacy = SectorPrivacyLevel.ENHANCED
            default_epsilon = 1.0

        config = self._sector_configs.get(sector) or SectorConfig(
            sector_name=sector,
            privacy_level=default_privacy,
            epsilon=default_epsilon,
        )

        self._sector_configs[sector] = config

        # Create FederatedAnomalyDetector for this sector
        self._sector_detectors[sector] = FederatedAnomalyDetector(
            model_dim=self.model_dim,
            n_rounds=50,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            use_privacy=config.privacy_level != SectorPrivacyLevel.NONE,
            epsilon=config.epsilon,
            delta=config.delta,
            aggregation=config.aggregation_strategy,
        )

    def configure_sector(self, sector: str, config: SectorConfig) -> None:
        """
        Configure a specific sector with custom settings.

        Args:
            sector: Sector name
            config: Sector configuration
        """
        if sector not in self.sectors:
            raise ValueError(f"Unknown sector: {sector}")

        self._sector_configs[sector] = config

        # Reinitialize the detector with new config
        self._sector_detectors[sector] = FederatedAnomalyDetector(
            model_dim=self.model_dim,
            n_rounds=50,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            use_privacy=config.privacy_level != SectorPrivacyLevel.NONE,
            epsilon=config.epsilon,
            delta=config.delta,
            aggregation=config.aggregation_strategy,
        )

        logger.info(f"Configured sector {sector} with privacy level {config.privacy_level.name}")

    def add_sector_client(
        self,
        sector: str,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> None:
        """
        Add a client to a specific sector.

        Args:
            sector: Sector name
            client_id: Unique client identifier within the sector
            X: Client's local training data
            y: Optional labels
        """
        if sector not in self.sectors:
            raise ValueError(f"Unknown sector: {sector}")

        full_client_id = f"{sector}_{client_id}"
        self._sector_clients[sector].append((full_client_id, X, y))

        # Add to the sector's detector
        self._sector_detectors[sector].add_client(full_client_id, X, y)

        logger.debug(f"Added client {client_id} to sector {sector}")

    def get_sector_client_count(self, sector: str) -> int:
        """Get the number of clients in a sector."""
        return len(self._sector_clients.get(sector, []))

    def coordinate_cross_sector_training(
        self,
        rounds: int = 10,
        cross_sector_aggregation: bool = True,
    ) -> CrossSectorResult:
        """
        Coordinate federated training across multiple CISA sectors.

        This implements a hierarchical aggregation approach:
        1. Each sector trains its federated model independently
        2. Optionally aggregates across sectors for shared threat intelligence

        Args:
            rounds: Number of federated rounds per sector
            cross_sector_aggregation: Whether to aggregate models across sectors

        Returns:
            CrossSectorResult with per-sector and global results
        """
        sector_results: dict[str, TrainingResult] = {}
        privacy_reports: dict[str, Any] = {}
        total_clients = 0
        total_samples = 0
        participating_sectors = []

        # Phase 1: Intra-sector training
        for sector in self.sectors:
            if not self._sector_clients[sector]:
                logger.warning(f"Sector {sector} has no clients, skipping")
                continue

            detector = self._sector_detectors[sector]
            # config is used for logging privacy level in future enhancements
            _ = self._sector_configs[sector]

            logger.info(
                f"Training sector {sector} with {len(self._sector_clients[sector])} clients"
            )

            try:
                # Update rounds for this training run
                detector._n_rounds = rounds
                result = detector.fit()
                sector_results[sector] = result
                participating_sectors.append(sector)

                # Track statistics
                total_clients += len(self._sector_clients[sector])
                total_samples += sum(len(data) for _, data, _ in self._sector_clients[sector])

                # Get privacy report if available
                privacy_report = detector.get_privacy_report()
                if privacy_report:
                    privacy_reports[sector] = privacy_report.to_dict()

                logger.info(
                    f"Sector {sector} training complete: "
                    f"{result.n_rounds} rounds, loss={result.round_results[-1].avg_loss:.6f}"
                    if result.round_results
                    else f"Sector {sector} training complete"
                )

            except Exception as e:
                logger.error(f"Error training sector {sector}: {e}")
                continue

        # Phase 2: Cross-sector aggregation (optional)
        global_weights = None
        cross_sector_metrics: dict[str, float] = {}

        if cross_sector_aggregation and len(participating_sectors) > 1:
            global_weights, cross_sector_metrics = self._aggregate_across_sectors(sector_results)
            logger.info(
                f"Cross-sector aggregation complete across {len(participating_sectors)} sectors"
            )

        return CrossSectorResult(
            sector_results=sector_results,
            global_model_weights=global_weights,
            total_clients=total_clients,
            total_samples=total_samples,
            participating_sectors=participating_sectors,
            cross_sector_metrics=cross_sector_metrics,
            privacy_reports=privacy_reports,
        )

    def _aggregate_across_sectors(
        self,
        sector_results: dict[str, TrainingResult],
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        """
        Aggregate models across sectors using weighted averaging.

        Only aggregates sectors that allow cross-sector sharing.

        Args:
            sector_results: Results from each sector's training

        Returns:
            Tuple of (aggregated_weights, metrics)
        """
        weights_list = []
        sample_counts = []

        for sector, result in sector_results.items():
            config = self._sector_configs[sector]

            # Skip sectors that don't allow cross-sector sharing
            if not config.cross_sector_sharing:
                continue

            if result.final_weights is not None:
                weights_list.append(result.final_weights)
                # Weight by total samples in sector
                sector_samples = sum(len(data) for _, data, _ in self._sector_clients[sector])
                sample_counts.append(sector_samples)

        if not weights_list:
            return None, {}

        # Weighted average across sectors
        total_samples = sum(sample_counts)
        aggregated = np.zeros_like(weights_list[0])

        for weights, count in zip(weights_list, sample_counts):
            aggregated += (count / total_samples) * weights

        metrics = {
            "n_sectors_aggregated": len(weights_list),
            "total_samples_aggregated": total_samples,
            "avg_weight_norm": float(np.mean([np.linalg.norm(w) for w in weights_list])),
        }

        return aggregated, metrics

    def get_sector_model(self, sector: str) -> np.ndarray | None:
        """
        Get the trained model weights for a specific sector.

        Args:
            sector: Sector name

        Returns:
            Model weights or None if not trained
        """
        if sector not in self._sector_detectors:
            raise ValueError(f"Unknown sector: {sector}")

        detector = self._sector_detectors[sector]
        return detector._weights

    def predict_sector(
        self,
        sector: str,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict anomalies using a sector's trained model.

        Args:
            sector: Sector name
            X: Data to analyze

        Returns:
            Binary anomaly labels (1 = anomaly)
        """
        if sector not in self._sector_detectors:
            raise ValueError(f"Unknown sector: {sector}")

        return self._sector_detectors[sector].predict(X)

    def decision_function_sector(
        self,
        sector: str,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Get anomaly scores using a sector's trained model.

        Args:
            sector: Sector name
            X: Data to analyze

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if sector not in self._sector_detectors:
            raise ValueError(f"Unknown sector: {sector}")

        return self._sector_detectors[sector].decision_function(X)

    def get_coordination_stats(self) -> dict[str, Any]:
        """Get statistics about the federated coordination."""
        stats: dict[str, Any] = {
            "total_sectors": len(self.sectors),
            "sectors": {},
        }

        for sector in self.sectors:
            config = self._sector_configs.get(sector)
            stats["sectors"][sector] = {
                "n_clients": len(self._sector_clients.get(sector, [])),
                "privacy_level": config.privacy_level.name if config else "UNKNOWN",
                "epsilon": config.epsilon if config else None,
                "cross_sector_sharing": config.cross_sector_sharing if config else False,
            }

        stats["total_clients"] = sum(len(clients) for clients in self._sector_clients.values())

        return stats
