# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for NCF Monitor (55 CISA National Critical Functions)."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.infrastructure.resilience.ncf_monitor import NCFMonitor


class TestNCFMonitor:
    """Test suite for NCFMonitor."""

    def test_ncf_monitor_instantiation(self) -> None:
        """Test NCF monitor can be instantiated."""
        monitor = NCFMonitor()
        assert monitor is not None

    def test_ncf_categories(self) -> None:
        """Test NCF categories are defined."""
        monitor = NCFMonitor()

        assert "connect" in monitor.ncf_categories
        assert "distribute" in monitor.ncf_categories
        assert "manage" in monitor.ncf_categories
        assert "supply" in monitor.ncf_categories

        assert len(monitor.ncf_categories["connect"]) == 9
        assert len(monitor.ncf_categories["distribute"]) == 9
        assert len(monitor.ncf_categories["manage"]) == 24
        assert len(monitor.ncf_categories["supply"]) == 13

    def test_detect_connect_ncf(self) -> None:
        """Test anomaly detection for Connect NCFs."""
        monitor = NCFMonitor()
        data = np.random.randn(100, 10)

        result = monitor.detect(data, "operate_core_network")

        assert "anomaly_score" in result
        assert "ncf_id" in result
        assert result["ncf_id"] == "operate_core_network"
        assert "category" in result
        assert result["category"] == "connect"

    def test_detect_distribute_ncf(self) -> None:
        """Test anomaly detection for Distribute NCFs."""
        monitor = NCFMonitor()
        data = np.random.randn(100, 10)

        result = monitor.detect(data, "distribute_electricity")

        assert result["category"] == "distribute"
        assert "anomaly_score" in result

    def test_cascading_failure_analysis(self) -> None:
        """Test cascading failure analysis across NCFs."""
        monitor = NCFMonitor()

        initial_failures = ["generate_electricity", "distribute_electricity"]
        cascading = monitor.analyze_cascading_failures(initial_failures)

        assert "initial_failures" in cascading
        assert "cascading_impacts" in cascading
        assert len(cascading["initial_failures"]) == 2

    def test_invalid_ncf_id(self) -> None:
        """Test handling of invalid NCF ID."""
        monitor = NCFMonitor()
        data = np.random.randn(100, 10)

        result = monitor.detect(data, "invalid_ncf_id")

        assert result["ncf_id"] == "invalid_ncf_id"
        assert "anomaly_score" in result
