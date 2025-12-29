"""
Mercury Agent ♱
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

from __future__ import annotations

"""
Tests for STEM Discipline Router for multi-engine fusion.
"""

import torch

from omni_mercury_engine.ml.fusion_network import STEMDisciplineRouter


class TestSTEMDisciplineRouter:
    """Test suite for STEMDisciplineRouter."""

    def test_router_instantiation(self):
        """Test router can be instantiated."""
        router = STEMDisciplineRouter()
        assert router is not None

    def test_discipline_mappings_exist(self):
        """Test discipline mappings are defined."""
        router = STEMDisciplineRouter()

        assert len(router.discipline_weights) > 0
        assert "biology" in router.discipline_weights
        assert "physics" in router.discipline_weights
        assert "chemistry" in router.discipline_weights

    def test_route_biology_data(self):
        """Test routing biology data to appropriate engines."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "biology")

        assert "biometric" in weights
        assert weights["biometric"] > 0.5
        assert "neural" in weights

    def test_route_physics_data(self):
        """Test routing physics data to appropriate engines."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "physics")

        assert "quantum" in weights
        assert weights["quantum"] > 0.5
        assert "astrophysical" in weights

    def test_route_cybersecurity_data(self):
        """Test routing cybersecurity data to security engine."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "cybersecurity")

        assert "security" in weights
        assert weights["security"] >= 0.9

    def test_adaptive_weight_adjustment(self):
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

    def test_explain_routing(self):
        """Test routing explanation functionality."""
        router = STEMDisciplineRouter()

        explanation = router.explain_routing("neuroscience")

        assert "discipline" in explanation
        assert "top_engines" in explanation
        assert "status" in explanation
        assert explanation["status"] == "routed"

    def test_unknown_discipline_fallback(self):
        """Test fallback for unknown disciplines."""
        router = STEMDisciplineRouter()
        data = torch.randn(10, 5)

        weights = router.route(data, "unknown_discipline")

        assert len(weights) > 0
