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
from __future__ import annotations

"""
Tests for InfrastructureCoordinator flexible module selection system.
"""

from omni_anomaly_engine.infrastructure import InfrastructureCoordinator


class TestInfrastructureCoordinator:
    """Test suite for InfrastructureCoordinator."""

    def test_coordinator_instantiation(self):
        """Test coordinator can be instantiated."""
        coord = InfrastructureCoordinator()
        assert coord is not None

    def test_list_all_modules(self):
        """Test listing all available modules."""
        coord = InfrastructureCoordinator()
        modules = coord.list_all_modules()

        assert len(modules) == 12
        assert "ncf_monitor" in modules
        assert "space_infrastructure" in modules
        assert "essential_workers" in modules

    def test_filter_by_priority(self):
        """Test filtering modules by priority."""
        coord = InfrastructureCoordinator()

        high_priority = coord.filter_modules(priorities=["high"])
        assert len(high_priority) == 8
        assert "ncf_monitor" in high_priority
        assert "space_infrastructure" in high_priority

        medium_priority = coord.filter_modules(priorities=["medium"])
        assert len(medium_priority) == 4

    def test_filter_by_category(self):
        """Test filtering modules by category."""
        coord = InfrastructureCoordinator()

        cyber = coord.filter_modules(categories=["cyber"])
        assert len(cyber) == 2
        assert "space_infrastructure" in cyber
        assert "cross_border_intel" in cyber

        humanitarian = coord.filter_modules(categories=["humanitarian"])
        assert len(humanitarian) == 2
        assert "essential_workers" in humanitarian

    def test_filter_by_module_names(self):
        """Test filtering by explicit module names."""
        coord = InfrastructureCoordinator()

        specific = coord.filter_modules(module_names=["ncf_monitor", "space_infrastructure"])
        assert len(specific) == 2
        assert "ncf_monitor" in specific
        assert "space_infrastructure" in specific

    def test_combined_filters(self):
        """Test combining multiple filter criteria."""
        coord = InfrastructureCoordinator()

        result = coord.filter_modules(priorities=["high"], categories=["cyber"])
        assert len(result) == 1
        assert "space_infrastructure" in result

    def test_get_single_module(self):
        """Test getting a single module instance."""
        coord = InfrastructureCoordinator()

        ncf = coord.get_module("ncf_monitor")
        assert ncf is not None
        assert hasattr(ncf, "detect")

    def test_instantiate_filtered_modules(self):
        """Test instantiating multiple filtered modules."""
        coord = InfrastructureCoordinator()

        high_priority = coord.instantiate_filtered_modules(priorities=["high"])
        assert len(high_priority) == 8

        for module in high_priority.values():
            assert hasattr(module, "detect")

    def test_flexible_selection_1_module(self):
        """Test running 1 module at a time."""
        coord = InfrastructureCoordinator()

        single = coord.instantiate_filtered_modules(module_names=["ncf_monitor"])
        assert len(single) == 1

    def test_flexible_selection_5_modules(self):
        """Test running 5 modules at a time."""
        coord = InfrastructureCoordinator()

        five = coord.instantiate_filtered_modules(priorities=["high"])
        assert len(five) >= 5

    def test_flexible_selection_all_modules(self):
        """Test running all 12 modules simultaneously."""
        coord = InfrastructureCoordinator()

        all_modules = coord.instantiate_filtered_modules()
        assert len(all_modules) == 12
