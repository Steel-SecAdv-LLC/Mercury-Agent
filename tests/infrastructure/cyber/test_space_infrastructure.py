"""
OMNI ♱ AVA (O♱A)
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
"""

"""
Tests for Space Infrastructure Monitor (EU Critical Entities unique sector).
"""

import pytest
import numpy as np
from omni_anomaly_engine.infrastructure.cyber.space_infrastructure import SpaceInfrastructureMonitor


class TestSpaceInfrastructureMonitor:
    """Test suite for SpaceInfrastructureMonitor."""

    def test_space_monitor_instantiation(self):
        """Test space infrastructure monitor can be instantiated."""
        monitor = SpaceInfrastructureMonitor()
        assert monitor is not None

    def test_asset_types_defined(self):
        """Test space asset types are defined."""
        monitor = SpaceInfrastructureMonitor()

        assert "satellites" in monitor.asset_types
        assert "ground_stations" in monitor.asset_types
        assert "launch_facilities" in monitor.asset_types

    def test_detect_satellite_anomaly(self):
        """Test satellite anomaly detection."""
        monitor = SpaceInfrastructureMonitor()
        data = np.random.randn(100, 10)

        result = monitor.detect(data, asset_type="satellite", asset_id="SAT-001")

        assert "anomaly_score" in result
        assert "asset_type" in result
        assert result["asset_type"] == "satellite"

    def test_detect_ground_station_anomaly(self):
        """Test ground station anomaly detection."""
        monitor = SpaceInfrastructureMonitor()
        data = np.random.randn(100, 10)

        result = monitor.detect(data, asset_type="ground_station", asset_id="GS-001")

        assert "anomaly_score" in result
        assert result["asset_type"] == "ground_station"
