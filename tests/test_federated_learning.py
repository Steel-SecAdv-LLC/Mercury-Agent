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
"""

from __future__ import annotations

"""
Tests for Federated Learning module (canonical API).
"""

import numpy as np

from omni_mercury_engine.federated_learning import (
    AggregationStrategy,
    CISAFederatedCoordinator,
    FederatedAnomalyDetector,
    SectorConfig,
    SectorPrivacyLevel,
)


class TestFederatedAnomalyDetector:
    """Tests for Federated Anomaly Detector."""

    def test_initialization(self):
        """Test detector initialization with default and custom parameters."""
        detector = FederatedAnomalyDetector(model_dim=10)
        assert detector._model_dim == 10
        assert detector._use_privacy is True
        assert detector._aggregation == "fedavg"

    def test_fit_and_predict(self):
        """Test full training and prediction cycle."""
        detector = FederatedAnomalyDetector(
            model_dim=5,
            n_rounds=3,
            local_epochs=2,
            use_privacy=False,
        )

        detector.add_client("client1", np.random.randn(100, 5))
        detector.add_client("client2", np.random.randn(120, 5))
        detector.add_client("client3", np.random.randn(110, 5))

        result = detector.fit()

        assert result.final_weights is not None
        assert result.n_rounds == 3
        assert len(result.round_results) == 3

        # Predict on new data
        predictions = detector.predict(np.random.randn(50, 5))
        assert predictions.shape == (50,)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_decision_function(self):
        """Test anomaly scoring."""
        detector = FederatedAnomalyDetector(
            model_dim=5, n_rounds=2, use_privacy=False
        )

        detector.add_client("client1", np.random.randn(100, 5))
        detector.add_client("client2", np.random.randn(100, 5))
        detector.fit()

        scores = detector.decision_function(np.random.randn(50, 5))
        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    def test_privacy_report(self):
        """Test that privacy reporting works when privacy is enabled."""
        detector = FederatedAnomalyDetector(
            model_dim=5,
            n_rounds=2,
            use_privacy=True,
            epsilon=1.0,
            delta=1e-5,
        )

        detector.add_client("client1", np.random.randn(100, 5))
        detector.add_client("client2", np.random.randn(100, 5))
        detector.fit()

        report = detector.get_privacy_report()
        assert report is not None
        assert report.epsilon > 0

    def test_no_clients_raises(self):
        """Test that fit raises when no clients are registered."""
        detector = FederatedAnomalyDetector(model_dim=5)
        try:
            detector.fit()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestCISAFederatedCoordinator:
    """Tests for CISA Federated Coordinator."""

    def test_initialization(self):
        """Test coordinator initialization."""
        sectors = ["healthcare", "energy", "financial_services"]
        coordinator = CISAFederatedCoordinator(sectors, model_dim=5)

        assert len(coordinator.sectors) == 3
        assert coordinator.get_sector_client_count("healthcare") == 0

    def test_add_sector_client(self):
        """Test adding clients to sectors."""
        coordinator = CISAFederatedCoordinator(["healthcare", "energy"], model_dim=5)

        coordinator.add_sector_client("healthcare", "hospital1", np.random.randn(100, 5))
        coordinator.add_sector_client("healthcare", "hospital2", np.random.randn(100, 5))
        coordinator.add_sector_client("energy", "utility1", np.random.randn(100, 5))

        assert coordinator.get_sector_client_count("healthcare") == 2
        assert coordinator.get_sector_client_count("energy") == 1

    def test_cross_sector_training(self):
        """Test cross-sector federated training."""
        sectors = ["healthcare", "energy"]
        coordinator = CISAFederatedCoordinator(sectors, model_dim=5)

        coordinator.add_sector_client("healthcare", "hospital1", np.random.randn(100, 5))
        coordinator.add_sector_client("healthcare", "hospital2", np.random.randn(100, 5))
        coordinator.add_sector_client("energy", "utility1", np.random.randn(100, 5))
        coordinator.add_sector_client("energy", "utility2", np.random.randn(100, 5))

        result = coordinator.coordinate_cross_sector_training(rounds=2)

        assert "healthcare" in result.sector_results
        assert "energy" in result.sector_results
        assert result.total_clients == 4
        assert len(result.participating_sectors) == 2
        assert result.global_model_weights is not None

    def test_sector_configuration(self):
        """Test configuring sector-specific privacy."""
        coordinator = CISAFederatedCoordinator(["healthcare"], model_dim=5)

        config = SectorConfig(
            sector_name="healthcare",
            privacy_level=SectorPrivacyLevel.MAXIMUM,
            epsilon=0.5,
        )
        coordinator.configure_sector("healthcare", config)

        stats = coordinator.get_coordination_stats()
        assert stats["sectors"]["healthcare"]["privacy_level"] == "MAXIMUM"
        assert stats["sectors"]["healthcare"]["epsilon"] == 0.5

    def test_unknown_sector_raises(self):
        """Test that adding to an unknown sector raises."""
        coordinator = CISAFederatedCoordinator(["healthcare"], model_dim=5)
        try:
            coordinator.add_sector_client("energy", "x", np.random.randn(10, 5))
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
