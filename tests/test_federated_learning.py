"""
Mercury Agent ♱
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
Tests for Federated Learning module.
"""

import numpy as np

from omni_mercury_engine.federated import (
    CISAFederatedCoordinator,
    FederatedAnomalyDetector,
    FederatedStrategy,
    PrivacyLevel,
)


class TestFederatedAnomalyDetector:
    """Tests for Federated Anomaly Detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = FederatedAnomalyDetector()
        assert detector.strategy == FederatedStrategy.FEDAVG
        assert detector.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY
        assert detector.num_clients == 10

    def test_federated_train(self):
        """Test federated training."""
        detector = FederatedAnomalyDetector(num_clients=3)

        client_data = {
            "client1": np.random.randn(100, 5),
            "client2": np.random.randn(120, 5),
            "client3": np.random.randn(110, 5),
        }

        result = detector.federated_train(client_data=client_data, local_epochs=2, num_rounds=3)

        assert "global_model" in result
        assert result["global_model"] is not None
        assert "training_history" in result
        assert len(result["training_history"]["rounds"]) == 3
        assert result["num_clients"] == 3

    def test_federated_detect(self):
        """Test federated anomaly detection."""
        detector = FederatedAnomalyDetector(num_clients=2)

        client_data_train = {"client1": np.random.randn(100, 5), "client2": np.random.randn(100, 5)}

        detector.federated_train(client_data_train, local_epochs=1, num_rounds=2)

        client_data_test = {"client1": np.random.randn(50, 5), "client2": np.random.randn(50, 5)}

        results = detector.federated_detect(client_data_test)

        assert len(results) == 2
        assert "client1" in results
        assert "anomaly_scores" in results["client1"]
        assert results["client1"]["privacy_preserved"]

    def test_differential_privacy(self):
        """Test differential privacy noise addition."""
        detector = FederatedAnomalyDetector(
            privacy_level=PrivacyLevel.DIFFERENTIAL_PRIVACY, epsilon=1.0
        )

        model_update = np.array([0.1, 0.2, 0.3])
        noisy_update = detector._add_differential_privacy_noise(model_update)

        assert noisy_update.shape == model_update.shape
        assert not np.allclose(noisy_update, model_update)

    def test_federated_averaging(self):
        """Test FedAvg aggregation."""
        detector = FederatedAnomalyDetector(strategy=FederatedStrategy.FEDAVG)
        detector.global_model_weights = np.array([1.0, 2.0, 3.0])

        updates = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]
        weights = [100, 200]

        aggregated = detector._federated_averaging(updates, weights)

        assert aggregated.shape == (3,)

    def test_personalization(self):
        """Test model personalization."""
        detector = FederatedAnomalyDetector()
        global_model = np.array([1.0, 2.0, 3.0])
        local_data = np.random.randn(50, 3)

        personalized = detector._personalize_model("client1", global_model, local_data)

        assert personalized.shape == global_model.shape


class TestCISAFederatedCoordinator:
    """Tests for CISA Federated Coordinator."""

    def test_initialization(self):
        """Test coordinator initialization."""
        sectors = ["healthcare", "energy", "financial"]
        coordinator = CISAFederatedCoordinator(sectors)

        assert len(coordinator.sector_detectors) == 3
        assert "healthcare" in coordinator.sector_detectors

    def test_cross_sector_training(self):
        """Test cross-sector federated training."""
        sectors = ["healthcare", "energy"]
        coordinator = CISAFederatedCoordinator(sectors)

        sector_data = {
            "healthcare": {
                "hospital1": np.random.randn(100, 5),
                "hospital2": np.random.randn(100, 5),
            },
            "energy": {"utility1": np.random.randn(100, 5), "utility2": np.random.randn(100, 5)},
        }

        results = coordinator.coordinate_cross_sector_training(sector_data, rounds=2)

        assert "healthcare" in results
        assert "energy" in results
        assert "global_model" in results["healthcare"]
