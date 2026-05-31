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

from omni_mercury_engine.security.pqc_guards import (
    PQCSimulationWarning,
    assert_no_simulation_in_production,
    check_pqc_production_readiness,
)


def test_check_pqc_production_readiness() -> None:
    result = check_pqc_production_readiness()
    assert isinstance(result, dict)
    assert "backend" in result
    assert "dilithium" in result
    assert "kyber" in result


def test_pqc_simulation_warning_class() -> None:
    assert issubclass(PQCSimulationWarning, UserWarning)


def test_assert_no_simulation_callable() -> None:
    assert callable(assert_no_simulation_in_production)
