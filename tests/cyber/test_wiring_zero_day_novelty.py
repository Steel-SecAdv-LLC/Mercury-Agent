# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired ``known_vulnerabilities`` novelty term.

The multiverse search is seeded deterministically, so the assessment is
reproducible; cataloguing a vulnerability on the most reachable region
penalises candidates near it and changes the converged outcome.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.security.cyber_fortress import MultiverseZeroDaySimulator

_STATE = np.linspace(-1.0, 1.0, 64)


def test_assessment_is_reproducible() -> None:
    a = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(_STATE)
    b = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(_STATE)
    assert a["zero_day_risk"] == b["zero_day_risk"]
    assert a["known_vulnerabilities_considered"] == 0


def test_known_vulnerability_changes_outcome() -> None:
    plain = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(_STATE)
    # Catalogue a signature on the most reachable region so novelty steers away.
    catalogued = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(
        _STATE, known_vulnerabilities=[_STATE.copy()]
    )
    assert catalogued["known_vulnerabilities_considered"] == 1
    top_plain = plain["potential_attack_vectors"][0]["state_summary"]
    top_catalogued = catalogued["potential_attack_vectors"][0]["state_summary"]
    assert top_plain != top_catalogued


def test_known_vulnerabilities_accepts_2d_array() -> None:
    sigs = np.vstack([_STATE.copy(), -_STATE.copy()])
    out = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(
        _STATE, known_vulnerabilities=sigs
    )
    assert out["known_vulnerabilities_considered"] == 2
    assert 0.0 <= out["zero_day_risk"] <= 1.0
