"""Tests for clinical governance scalars (SOFA reuse + NEWS2), abstention-first."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # governance.contract -> GOSNN imports numpy.

from omni_mercury_engine.governance import clinical
from omni_mercury_engine.governance.contract import ScalarStatus


def _by_name(scalars: list, name: str):
    """Return the scalar with ``name`` from a list of governance scalars."""
    return next(s for s in scalars if s.name == name)


def test_sofa_subscores_match_published_thresholds() -> None:
    """Each SOFA organ sub-score matches the published point table for given inputs."""
    pytest.importorskip("omni_mercury_engine.medical.critical_care.sepsis_detector")
    data = {
        "pao2_fio2_ratio": 250,  # 200<=x<300 -> 2
        "platelets_k_ul": 30,  # 20<=x<50  -> 3
        "bilirubin_mg_dl": 8.0,  # 6<=x<12   -> 3
        "mean_arterial_pressure": 65,  # <70, no pressors -> 1
        "gcs_score": 13,  # >=13      -> 1
        "creatinine_mg_dl": 2.5,  # 2.0<=x<3.5 -> 2
    }
    scalars = clinical.sofa_scalars(data)
    assert _by_name(scalars, "omni_sofa_respiration").value == pytest.approx(2 / 4)
    assert _by_name(scalars, "omni_sofa_coagulation").value == pytest.approx(3 / 4)
    assert _by_name(scalars, "omni_sofa_liver").value == pytest.approx(3 / 4)
    assert _by_name(scalars, "omni_sofa_cardiovascular").value == pytest.approx(1 / 4)
    assert _by_name(scalars, "omni_sofa_cns").value == pytest.approx(1 / 4)
    assert _by_name(scalars, "omni_sofa_renal").value == pytest.approx(2 / 4)
    # total = 2+3+3+1+1+2 = 12 of 24
    assert _by_name(scalars, "omni_sofa_total").value == pytest.approx(12 / 24)


def test_sofa_abstains_on_missing_inputs() -> None:
    """A missing organ input abstains rather than reporting a healthy organ."""
    pytest.importorskip("omni_mercury_engine.medical.critical_care.sepsis_detector")
    scalars = clinical.sofa_scalars({"platelets_k_ul": 30})
    assert _by_name(scalars, "omni_sofa_coagulation").status is ScalarStatus.AVAILABLE
    assert _by_name(scalars, "omni_sofa_liver").status is ScalarStatus.UNAVAILABLE
    # The total is undefined unless all six organ inputs are present.
    assert _by_name(scalars, "omni_sofa_total").status is ScalarStatus.UNAVAILABLE


def test_news2_all_normal_is_zero() -> None:
    """NEWS2 aggregates to 0 when every parameter is in its normal band."""
    vitals = {
        "respiratory_rate_bpm": 14,
        "spo2_pct": 98,
        "on_supplemental_o2": False,
        "systolic_bp_mmhg": 120,
        "pulse_bpm": 70,
        "consciousness": "A",
        "temperature_c": 37.0,
    }
    scalar = clinical.news2_scalar(vitals)
    assert scalar.status is ScalarStatus.AVAILABLE
    assert scalar.value == pytest.approx(0.0)
    assert scalar.provenance["aggregate"] == 0


def test_news2_worked_example_aggregate() -> None:
    """NEWS2 reproduces a known high-acuity aggregate (18/20) from the table."""
    vitals = {
        "respiratory_rate_bpm": 26,  # >=25 -> 3
        "spo2_pct": 90,  # <=91 -> 3
        "on_supplemental_o2": True,  # O2   -> 2
        "systolic_bp_mmhg": 88,  # <=90 -> 3
        "pulse_bpm": 120,  # 111-130 -> 2
        "consciousness": "V",  # not alert -> 3
        "temperature_c": 39.5,  # >=39.1 -> 2
    }
    scalar = clinical.news2_scalar(vitals)
    assert scalar.provenance["aggregate"] == 18
    assert scalar.value == pytest.approx(18 / 20)


def test_news2_abstains_on_missing_parameter() -> None:
    """NEWS2 abstains when any of its seven parameters is absent."""
    vitals = {
        "respiratory_rate_bpm": 14,
        "spo2_pct": 98,
        "on_supplemental_o2": False,
        "systolic_bp_mmhg": 120,
        "pulse_bpm": 70,
        "consciousness": "A",
        # temperature_c missing
    }
    scalar = clinical.news2_scalar(vitals)
    assert scalar.status is ScalarStatus.UNAVAILABLE
    assert scalar.value is None
