# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired ``patient_state`` conditioning of treatment
pathway optimization.

The risk burden is a hand-computable mean of normalised components, and a
frailer patient weights the safety term more heavily so the top pathway is at
least as safe as the robust-patient pathway.
"""

from __future__ import annotations

from omni_mercury_engine.medical.medical_cure_predictor import (
    TreatmentPathwayOptimizer,
    _patient_risk_burden,
)

_LOW = {"age": 20, "severity": 0.0, "comorbidities": [], "organ_function": 1.0}
_HIGH = {
    "age": 90,
    "severity": 0.95,
    "comorbidities": ["a", "b", "c", "d", "e"],
    "organ_function": 0.1,
}


def test_risk_burden_hand_values() -> None:
    assert _patient_risk_burden({}) == 0.5  # unrecognised -> neutral
    assert _patient_risk_burden({"severity": 1.0}) == 1.0
    # mean([age 0.2, severity 0.0, comorbidities 0.0, organ 0.0]) == 0.05
    assert abs(_patient_risk_burden(_LOW) - 0.05) < 1e-9
    # mean([age 0.9, severity 0.95, comorbidities 1.0, organ 0.9]) == 0.9375
    assert abs(_patient_risk_burden(_HIGH) - 0.9375) < 1e-9


def test_higher_risk_weights_safety_more() -> None:
    low = TreatmentPathwayOptimizer(num_universes=25).optimize_treatment(_LOW, "disease")
    high = TreatmentPathwayOptimizer(num_universes=25).optimize_treatment(_HIGH, "disease")
    # safety_weight == 0.5 + risk_burden
    assert abs(low["safety_weight"] - 0.55) < 1e-9
    assert abs(high["safety_weight"] - 1.4375) < 1e-9
    assert high["safety_weight"] > low["safety_weight"]


def test_frailer_patient_gets_safer_top_pathway() -> None:
    low = TreatmentPathwayOptimizer(num_universes=25).optimize_treatment(_LOW, "disease")
    high = TreatmentPathwayOptimizer(num_universes=25).optimize_treatment(_HIGH, "disease")
    # safety_score == -std(safety block); closer to 0 is safer.
    assert (
        high["treatment_pathways"][0]["safety_score"]
        >= low["treatment_pathways"][0]["safety_score"]
    )
