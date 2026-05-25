"""Tests for clinical governance scalars (SOFA, NEWS2, MEWS, MELD-Na), three-state.

CI honesty: the torch-backed SOFA worked examples are skipped *loudly* (explicit reason)
when the ``[ml]`` extra is absent, and :func:`test_clinical_ml_suite_must_run_under_gate`
fails the build if they would silently skip in the lane that owns them
(``MERCURY_REQUIRES_ML=1``).  The numpy-only families (NEWS2/MEWS/MELD-Na) run everywhere.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("numpy")  # governance.contract -> GOSNN imports numpy.

from omni_mercury_engine._compat import HAS_TORCH
from omni_mercury_engine.governance import clinical
from omni_mercury_engine.governance.contract import ScalarState

# A LOUD, explicit skip -- never a silent green-wash of the clinical arithmetic.
requires_ml = pytest.mark.skipif(
    not HAS_TORCH,
    reason=(
        "SKIPPED LOUDLY: SOFACalculator needs the [ml] extra (torch); without it the SOFA "
        "scalar abstains UNAVAILABLE. Worked-example arithmetic is validated only in an [ml] "
        "lane (set MERCURY_REQUIRES_ML=1 to make this skip a hard failure)."
    ),
)


def _by_name(scalars, name):
    """Return the scalar with ``name`` from a list of governance scalars."""
    return next(s for s in scalars if s.name == name)


def test_clinical_ml_suite_must_run_under_gate() -> None:
    """If MERCURY_REQUIRES_ML=1, the torch-backed SOFA examples must NOT silently skip."""
    if os.environ.get("MERCURY_REQUIRES_ML") == "1":
        assert HAS_TORCH, (
            "MERCURY_REQUIRES_ML=1 but torch is absent: the clinical SOFA worked-example "
            "suite would skip and report green having validated nothing."
        )


@requires_ml
def test_sofa_subscores_match_published_thresholds() -> None:
    """Each SOFA organ sub-score matches the published point table (Vincent 1996)."""
    data: dict[str, object] = {
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


@requires_ml
def test_sofa_abstains_on_missing_inputs() -> None:
    """A missing organ input is UNAVAILABLE (kept), never a healthy organ."""
    scalars = clinical.sofa_scalars({"platelets_k_ul": 30})
    assert _by_name(scalars, "omni_sofa_coagulation").state is ScalarState.GROUNDED
    liver = _by_name(scalars, "omni_sofa_liver")
    assert liver.state is ScalarState.UNAVAILABLE
    assert liver.missing_inputs == ("bilirubin_mg_dl",)
    # The total is undefined unless all six organ inputs are present.
    assert _by_name(scalars, "omni_sofa_total").state is ScalarState.UNAVAILABLE


def test_sofa_abstains_unavailable_when_calculator_absent() -> None:
    """Without the [ml] detector stack SOFA abstains UNAVAILABLE (signal real, deferred)."""
    if HAS_TORCH:
        pytest.skip("torch present: the calculator-absent path is exercised only in a thin env")
    scalars = clinical.sofa_scalars({"pao2_fio2_ratio": 250})
    assert all(s.state is ScalarState.UNAVAILABLE for s in scalars)


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
    assert scalar.state is ScalarState.GROUNDED
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
    """NEWS2 is UNAVAILABLE when any of its seven parameters is absent (records which)."""
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
    assert scalar.state is ScalarState.UNAVAILABLE
    assert scalar.value is None
    assert scalar.missing_inputs == ("temperature_c",)


def test_mews_worked_example_against_table() -> None:
    """MEWS reproduces the Subbe 2001 table: SBP85=1, HR135=3, RR32=3, T39.0=2, V=1 -> 10/14."""
    vitals = {
        "systolic_bp_mmhg": 85,
        "pulse_bpm": 135,
        "respiratory_rate_bpm": 32,
        "temperature_c": 39.0,
        "avpu": "V",
    }
    scalar = clinical.mews_scalar(vitals)
    assert scalar.state is ScalarState.GROUNDED
    assert scalar.provenance["aggregate"] == 10
    assert scalar.value == pytest.approx(10 / 14)
    assert scalar.provenance["points"] == {
        "systolic_bp": 1,
        "pulse": 3,
        "respiration": 3,
        "temperature": 2,
        "avpu": 1,
    }


def test_mews_all_normal_is_zero() -> None:
    """MEWS aggregates to 0 when every parameter sits in its normal band."""
    vitals = {
        "systolic_bp_mmhg": 120,
        "pulse_bpm": 70,
        "respiratory_rate_bpm": 12,
        "temperature_c": 37.0,
        "avpu": "A",
    }
    scalar = clinical.mews_scalar(vitals)
    assert scalar.state is ScalarState.GROUNDED
    assert scalar.value == pytest.approx(0.0)


def test_mews_abstains_on_missing_or_bad_avpu() -> None:
    """MEWS is UNAVAILABLE when a vital is missing or AVPU is not A/V/P/U."""
    missing = clinical.mews_scalar(
        {
            "systolic_bp_mmhg": 120,
            "pulse_bpm": 70,
            "respiratory_rate_bpm": 12,
            "temperature_c": 37.0,
        }
    )
    assert missing.state is ScalarState.UNAVAILABLE
    assert missing.missing_inputs == ("avpu",)
    bad = clinical.mews_scalar(
        {
            "systolic_bp_mmhg": 120,
            "pulse_bpm": 70,
            "respiratory_rate_bpm": 12,
            "temperature_c": 37.0,
            "avpu": "Z",
        }
    )
    assert bad.state is ScalarState.UNAVAILABLE


def test_meld_na_worked_example_against_formula() -> None:
    """MELD-Na (OPTN): bili2.0, Cr1.5, INR1.2, Na130 -> MELD 15, MELD-Na 21 -> 21/40."""
    scalar = clinical.meld_na_scalar(
        {"bilirubin_mg_dl": 2.0, "creatinine_mg_dl": 1.5, "inr": 1.2, "sodium_meq_l": 130}
    )
    assert scalar.state is ScalarState.GROUNDED
    assert scalar.provenance == {"meld": 15, "meld_na": 21}
    assert scalar.value == pytest.approx(21 / 40)


def test_meld_na_abstains_until_inr_and_sodium_flow() -> None:
    """MELD-Na is UNAVAILABLE (kept) when its INR/sodium labs have not flowed yet."""
    scalar = clinical.meld_na_scalar({"bilirubin_mg_dl": 2.0, "creatinine_mg_dl": 1.5})
    assert scalar.state is ScalarState.UNAVAILABLE
    assert scalar.value is None
    assert scalar.missing_inputs == ("inr", "sodium_meq_l")


def test_not_for_clinical_use_boundary_is_documented() -> None:
    """The clinical module must carry the not-for-clinical-use boundary in its source."""
    import inspect

    assert "NOT FOR CLINICAL USE" in inspect.getsource(clinical).upper()
