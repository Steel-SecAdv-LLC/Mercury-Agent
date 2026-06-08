# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test pqc guards smoke."""

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
