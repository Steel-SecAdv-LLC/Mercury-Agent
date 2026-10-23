# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""InterstellarObjectDetector on the two recorded interstellar objects.

The fixtures are JPL Small-Body Database solutions, fetched once from
``https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=<1I|2I>&phys-par=1&full-prec=1``
(JPL solution of 2018-06-26 for 1I/'Oumuamua and 2024-06-24 for 2I/Borisov,
both by D. Farnocchia) and re-serialised with sorted keys; their SHA-256 is
pinned below so the recorded elements cannot drift unnoticed.

Outgassing is not in the SBDB record, so it is an input taken from the
discovery papers: none detected for 1I (Meech et al. 2017, Nature 552:378;
Micheli et al. 2018, Nature 559:223), a coma for 2I (Guzik et al. 2020, Nat
Astron 4:53).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.space.interstellar_objects import (
    InterstellarObjectDetector,
    InterstellarObjectResult,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "space"
SHA256 = {
    "1I": "ee3c4f0f6ae890480f0d5f6874ddc27db097962efb190a8359ef69511fa72099",
    "2I": "6e9759f606ed5955713aa5da3db2ef2384590114314afeed4b4876165b811f2b",
}


def _record(name: str) -> dict[str, Any]:
    path = FIXTURES / f"jpl_sbdb_{name}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256[name]
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _iso_data(name: str, *, outgassing: bool) -> dict[str, Any]:
    """The detector's input, built from the recorded SBDB solution."""
    record = _record(name)
    # The period and aphelion are null for a hyperbolic orbit: they are undefined.
    elements = {
        e["name"]: float(e["value"]) for e in record["orbit"]["elements"] if e["value"] is not None
    }
    model = {m["name"]: float(m["value"]) for m in record["orbit"]["model_pars"]}
    physical = {p["name"]: float(p["value"]) for p in record.get("phys_par", [])}
    morphology = {"rotation_period_h": physical["rot_per"]} if "rot_per" in physical else {}
    return {
        "designation": record["object"]["fullname"],
        "orbital_parameters": {
            "eccentricity": elements["e"],
            "perihelion_au": elements["q"],
            "inclination_deg": elements["i"],
            "semi_major_axis_au": elements["a"],
            "hyperbolic": elements["e"] > 1.0,
            "non_gravitational_acceleration": any(model[k] != 0.0 for k in ("A1", "A2", "A3")),
            "radial_non_grav_a1": model["A1"],
            "transverse_non_grav_a2": model["A2"],
            "normal_non_grav_a3": model["A3"],
        },
        "spectroscopy": {"outgassing_detected": outgassing},
        "morphology": morphology,
    }


OUMUAMUA = _iso_data("1I", outgassing=False)
BORISOV = _iso_data("2I", outgassing=True)


def _detect(data: dict[str, Any], seed: int = 0) -> InterstellarObjectResult:
    torch.manual_seed(seed)
    return InterstellarObjectDetector().detect_interstellar_anomaly(data)


def test_the_recorded_solutions_are_hyperbolic_interstellar_orbits() -> None:
    """A fixture sanity check: e > 1 is what makes these objects interstellar."""
    assert OUMUAMUA["orbital_parameters"]["eccentricity"] == pytest.approx(1.2011338, abs=1e-6)
    assert BORISOV["orbital_parameters"]["eccentricity"] == pytest.approx(3.3564758, abs=1e-6)


def test_the_shipped_reference_table_matches_the_recorded_elements() -> None:
    known = InterstellarObjectDetector().known_isos
    for key, data in (("1I/Oumuamua", OUMUAMUA), ("2I/Borisov", BORISOV)):
        orbital = data["orbital_parameters"]
        assert known[key]["eccentricity"] == pytest.approx(orbital["eccentricity"], rel=1e-2)
        assert known[key]["perihelion_au"] == pytest.approx(orbital["perihelion_au"], rel=1e-2)
        assert known[key]["inclination_deg"] == pytest.approx(orbital["inclination_deg"], rel=1e-2)


def test_oumuamuas_orbital_flags_follow_from_its_recorded_elements() -> None:
    result = _detect(OUMUAMUA)
    # i = 122.74 deg is past the 120 deg flag; e = 1.2011 is under the 1.5 flag.
    assert "extreme_orbital_inclination" in result.orbital_anomalies
    assert "extreme_hyperbolic_orbit" not in result.orbital_anomalies
    # A fitted non-gravitational term and no outgassing is the flag Micheli et al. made notable.
    assert "non_gravitational_acceleration_without_outgassing" in result.orbital_anomalies


def test_borisovs_orbital_flags_follow_from_its_recorded_elements() -> None:
    result = _detect(BORISOV)
    # e = 3.356 is past the 1.5 flag; i = 44.05 deg is not; its coma explains the non-grav term.
    assert "extreme_hyperbolic_orbit" in result.orbital_anomalies
    assert "extreme_orbital_inclination" not in result.orbital_anomalies
    assert "non_gravitational_acceleration_without_outgassing" not in result.orbital_anomalies


def test_the_non_gravitational_hypotheses_follow_the_orbital_flag() -> None:
    """The hypothesis step once read a key nothing set, so these never appeared."""
    unexplained = _detect(OUMUAMUA).alternative_hypotheses
    explained = _detect(BORISOV).alternative_hypotheses
    assert "Sublimation of supervolatiles (H2, N2)" in unexplained
    assert "Sublimation of supervolatiles (H2, N2)" not in explained


@pytest.mark.parametrize("data", [OUMUAMUA, BORISOV], ids=["1I", "2I"])
def test_an_untrained_network_assesses_nothing(data: dict[str, Any]) -> None:
    """Random weights once chose the anomaly type and the detected flag from the seed."""
    outcomes = {
        (r.anomaly_type, r.anomaly_detected, r.anomaly_score, r.confidence, r.ml_available)
        for r in (_detect(data, seed) for seed in range(6))
    }
    assert outcomes == {("unassessed", False, 0.0, 0.0, False)}


@pytest.mark.parametrize("data", [OUMUAMUA, BORISOV], ids=["1I", "2I"])
def test_the_rule_based_findings_do_not_depend_on_the_network(data: dict[str, Any]) -> None:
    first, second = _detect(data, 0), _detect(data, 1)
    assert first.orbital_anomalies == second.orbital_anomalies
    assert first.natural_explanation_assessment == second.natural_explanation_assessment
    assert first.alternative_hypotheses == second.alternative_hypotheses
    assert first.follow_up_observations_recommended == second.follow_up_observations_recommended


def test_loaded_weights_are_scored_and_reported() -> None:
    torch.manual_seed(0)
    detector = InterstellarObjectDetector()
    detector.model.load_trained_weights(detector.model.state_dict())
    assert detector.model.is_fitted is True

    result = detector.detect_interstellar_anomaly(OUMUAMUA)

    assert result.ml_available is True
    orbital, spectro, physical = (
        torch.tensor(f, dtype=torch.float32).unsqueeze(0)
        for f in detector._extract_features(OUMUAMUA)
    )
    with torch.no_grad():
        _, confidence = detector.model.eval()(orbital, spectro, physical)
    assert result.confidence == pytest.approx(float(confidence[0]))
    expected = float(confidence[0]) * detector.omni_interstellar_scalars["omni_orbital_precision"]
    assert result.anomaly_score == pytest.approx(expected)
