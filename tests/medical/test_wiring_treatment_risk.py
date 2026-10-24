# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioral tests for the wired ``patient_state`` conditioning of treatment pathway optimization.

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


_FRAMINGHAM_LOW = {
    "age": 35,
    "gender": "female",
    "total_cholesterol_mg_dl": 170.0,
    "hdl_cholesterol_mg_dl": 70.0,
    "systolic_bp_mmhg": 110.0,
    "smoker": False,
    "diabetes": False,
}
_FRAMINGHAM_HIGH = {
    "age": 72,
    "gender": "male",
    "total_cholesterol_mg_dl": 280.0,
    "hdl_cholesterol_mg_dl": 30.0,
    "systolic_bp_mmhg": 180.0,
    "smoker": True,
    "diabetes": True,
}


def test_framingham_burden_matches_the_real_instrument() -> None:
    from omni_mercury_engine.medical.cardiology.cardiology_predictor import (
        FraminghamRiskCalculator,
    )
    from omni_mercury_engine.medical.medical_cure_predictor import _framingham_risk_fraction

    calc = FraminghamRiskCalculator()
    for patient in (_FRAMINGHAM_LOW, _FRAMINGHAM_HIGH):
        instrument = float(calc.calculate_risk(patient)["10_year_cvd_risk_percent"]) / 100.0
        expected = min(instrument / 0.20, 1.0)  # 20% ten-year risk = high-risk boundary
        burden = _framingham_risk_fraction(patient)
        assert burden is not None
        assert abs(burden - expected) < 1e-12


def test_framingham_requires_all_core_covariates() -> None:
    from omni_mercury_engine.medical.medical_cure_predictor import _framingham_risk_fraction

    incomplete = dict(_FRAMINGHAM_HIGH)
    del incomplete["hdl_cholesterol_mg_dl"]
    assert _framingham_risk_fraction(incomplete) is None  # no default-masquerading
    unknown_gender = dict(_FRAMINGHAM_HIGH, gender="unspecified")
    assert _framingham_risk_fraction(unknown_gender) is None


def test_optimizer_uses_instrument_when_available_and_reports_it() -> None:
    low = TreatmentPathwayOptimizer(num_universes=25).optimize_treatment(_FRAMINGHAM_LOW, "cvd")
    high = TreatmentPathwayOptimizer(num_universes=25).optimize_treatment(_FRAMINGHAM_HIGH, "cvd")
    assert low["risk_model"] == high["risk_model"] == "framingham_10yr_cvd"
    assert high["patient_risk_burden"] > low["patient_risk_burden"]
    assert high["safety_weight"] > low["safety_weight"]
    # Heuristic records (no Framingham covariates) fall back and say so.
    heuristic = TreatmentPathwayOptimizer(num_universes=25).optimize_treatment(_LOW, "x")
    assert heuristic["risk_model"] == "heuristic"


def test_instrument_risk_at_or_above_high_risk_boundary_saturates_burden() -> None:
    from omni_mercury_engine.medical.cardiology.cardiology_predictor import (
        FraminghamRiskCalculator,
    )
    from omni_mercury_engine.medical.medical_cure_predictor import _framingham_risk_fraction

    instrument = (
        float(
            FraminghamRiskCalculator().calculate_risk(_FRAMINGHAM_HIGH)["10_year_cvd_risk_percent"]
        )
        / 100.0
    )
    assert instrument >= 0.20  # this patient is in the clinical high-risk category
    assert _framingham_risk_fraction(_FRAMINGHAM_HIGH) == 1.0  # burden saturates at the boundary
