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


def test_measured_exposure_raises_simulated_risk() -> None:
    baseline = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(_STATE)
    exposed = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(
        _STATE, exposure=np.full(64, 1.5)
    )
    # Measured weakness discounts the reachability distance, so the same
    # system state simulates as more reachable -> strictly higher risk.
    assert exposed["zero_day_risk"] > baseline["zero_day_risk"]
    assert exposed["exposure_informed"] is True
    assert baseline["exposure_informed"] is False


def test_zero_exposure_is_identical_to_baseline() -> None:
    baseline = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(_STATE)
    zeros = MultiverseZeroDaySimulator(num_universes=20).simulate_zero_day(
        _STATE, exposure=np.zeros(64)
    )
    assert zeros["zero_day_risk"] == baseline["zero_day_risk"]
    assert zeros["exposure_informed"] is False  # nothing measured -> no discount


def test_fortress_scan_feeds_measured_tampering_into_simulation() -> None:
    from omni_mercury_engine.security.cyber_fortress import CyberFortress

    fortress = CyberFortress(enable_traffic_detection=False, enable_auto_refactor=False)
    clean_chain = [f"hash_{i}" for i in range(100)]
    tampered_chain = ["dup"] * 20 + [f"h{i}" for i in range(80)]  # replay duplicates

    clean = fortress.fortress_scan({"hash_chain": clean_chain, "system_state": _STATE})
    tampered = fortress.fortress_scan({"hash_chain": tampered_chain, "system_state": _STATE})

    assert not tampered.hash_integrity_verified
    # The integrity finding is real measured exposure: the identical system
    # state must simulate as higher zero-day risk after observed tampering.
    assert tampered.zero_day_risk > clean.zero_day_risk
