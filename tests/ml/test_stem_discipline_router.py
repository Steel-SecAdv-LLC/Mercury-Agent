# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for STEM Discipline Router for multi-engine fusion."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.ml.fusion_network import STEMDisciplineRouter


class TestSTEMDisciplineRouter:
    """Test suite for STEMDisciplineRouter."""

    def test_router_instantiation(self) -> None:
        """Test router can be instantiated."""
        router = STEMDisciplineRouter()
        assert router is not None

    def test_discipline_mappings_exist(self) -> None:
        """Test discipline mappings are defined."""
        router = STEMDisciplineRouter()

        assert len(router.discipline_weights) > 0
        assert "biology" in router.discipline_weights
        assert "physics" in router.discipline_weights
        assert "chemistry" in router.discipline_weights

    def test_route_biology_data(self) -> None:
        """Test routing biology data to appropriate engines."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "biology")

        assert "biometric" in weights
        assert weights["biometric"] > 0.5
        assert "neural" in weights

    def test_route_physics_data(self) -> None:
        """Test routing physics data to appropriate engines."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "physics")

        assert "quantum" in weights
        assert weights["quantum"] > 0.5
        assert "astrophysical" in weights

    def test_route_cybersecurity_data(self) -> None:
        """Test routing cybersecurity data to security engine."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "cybersecurity")

        assert "security" in weights
        assert weights["security"] >= 0.9

    def test_adaptive_weight_adjustment(self) -> None:
        """Test adaptive weight adjustment based on data type."""
        router = STEMDisciplineRouter()

        numerical_data = torch.randn(10, 5)
        timeseries_data = torch.randn(50, 5)

        weights_num = router.route(numerical_data, "physics")
        weights_ts = router.route(timeseries_data, "physics")

        assert isinstance(weights_num, dict)
        assert isinstance(weights_ts, dict)
        assert len(weights_num) > 0
        assert len(weights_ts) > 0

    def test_explain_routing(self) -> None:
        """Test routing explanation functionality."""
        router = STEMDisciplineRouter()

        explanation = router.explain_routing("neuroscience")

        assert "discipline" in explanation
        assert "top_engines" in explanation
        assert "status" in explanation
        assert explanation["status"] == "routed"

    def test_unknown_discipline_fallback(self) -> None:
        """Test fallback for unknown disciplines."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "unknown_discipline")

        assert len(weights) > 0
