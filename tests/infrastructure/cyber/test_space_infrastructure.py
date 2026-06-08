# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Space Infrastructure Monitor (EU Critical Entities unique sector)."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.infrastructure.cyber.space_infrastructure import SpaceInfrastructureMonitor


class TestSpaceInfrastructureMonitor:
    """Test suite for SpaceInfrastructureMonitor."""

    def test_space_monitor_instantiation(self) -> None:
        """Test space infrastructure monitor can be instantiated."""
        monitor = SpaceInfrastructureMonitor()
        assert monitor is not None

    def test_asset_types_defined(self) -> None:
        """Test space asset types are defined."""
        monitor = SpaceInfrastructureMonitor()

        assert "satellites" in monitor.asset_types
        assert "ground_stations" in monitor.asset_types
        assert "launch_facilities" in monitor.asset_types

    def test_detect_satellite_anomaly(self) -> None:
        """Test satellite anomaly detection."""
        monitor = SpaceInfrastructureMonitor()
        data = np.random.randn(100, 10)

        result = monitor.detect(data, asset_type="satellite", asset_id="SAT-001")

        assert "anomaly_score" in result
        assert "asset_type" in result
        assert result["asset_type"] == "satellite"

    def test_detect_ground_station_anomaly(self) -> None:
        """Test ground station anomaly detection."""
        monitor = SpaceInfrastructureMonitor()
        data = np.random.randn(100, 10)

        result = monitor.detect(data, asset_type="ground_station", asset_id="GS-001")

        assert "anomaly_score" in result
        assert result["asset_type"] == "ground_station"
