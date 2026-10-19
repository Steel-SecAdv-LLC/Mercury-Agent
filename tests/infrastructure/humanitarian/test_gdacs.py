# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``crisis_monitoring.gdacs``: the published GDACS earthquake and flood models.

Expected values are worked by hand from gdacs.org/Knowledge/models_eq.aspx and
models_fl.aspx; country terms are spot-checked against GDACS's 2017
coefficient table (2017_GDACS_EQ_algorithm_coefficients.xlsx).
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from omni_mercury_engine.infrastructure.humanitarian.crisis_monitoring import gdacs

_RADII = {20: 50_000, 50: 200_000, 75: 400_000, 100: 600_000}


def test_shakemap_score_global_model() -> None:
    # Scaled population 10 x 1,000 + 20,000 + 0.1 x 100,000 = 40,000.
    score = gdacs.earthquake_shakemap_score(100_000, 20_000, 1_000)
    assert score == pytest.approx(-0.59 + 0.53 * math.log10(40_000))
    assert score == pytest.approx(1.849092, abs=1e-6)


def test_nobody_above_mmi_vi_is_green_zero() -> None:
    assert gdacs.earthquake_shakemap_score(0, 0, 0) == 0.0
    assert gdacs.alert_level(0.0) == "green"


def test_country_terms_change_the_level() -> None:
    """Japan's shakemap vulnerability (-1.0) takes the same exposure from Orange to Green."""
    exposure = (100_000, 20_000, 1_000)
    japan = gdacs.earthquake_coefficients("JPN")
    assert gdacs.alert_level(gdacs.earthquake_shakemap_score(*exposure)) == "orange"
    score = gdacs.earthquake_shakemap_score(*exposure, japan)
    assert score == pytest.approx(0.849092, abs=1e-6)
    assert gdacs.alert_level(score) == "green"


def test_parameter_score_global_and_country_model() -> None:
    # Scaled population 500,000 + 300,000 + 100,000 + 20,000 = 920,000; depth 10 km.
    raw = -7.75 + 0.82 * 7.0 - 0.53 + 0.72 * math.log10(920_000)
    assert gdacs.earthquake_parameter_score(7.0, 10.0, _RADII) == pytest.approx(raw)
    assert raw == pytest.approx(1.753927, abs=1e-6)
    nepal = gdacs.earthquake_parameter_score(
        7.0, 10.0, _RADII, gdacs.earthquake_coefficients("NPL")
    )
    assert nepal == pytest.approx(4.73 * raw - 9.39)


@pytest.mark.parametrize(
    ("iso3", "expected"),
    [
        ("JPN", (0.0, 0.0, -1.0, 0.73, -0.22, 0.0)),
        ("CHL", (-0.57, 0.01, 0.0, 1.44, -1.48, 0.0)),
        ("PAK", (-2.53, 0.63, 0.0, 2.1, -0.9, 0.0)),
        ("HTI", (0.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
        ("NPL", (0.0, 0.0, 0.57, 4.73, -9.39, 0.0)),
    ],
)
def test_table_rows_match_the_2017_spreadsheet(iso3: str, expected: tuple[float, ...]) -> None:
    assert dataclasses.astuple(gdacs.COUNTRY_COEFFICIENTS[iso3]) == expected


def test_table_shape() -> None:
    table = gdacs.COUNTRY_COEFFICIENTS
    assert len(table) == 38
    assert all(len(code) == 3 and code.isupper() for code in table)
    assert gdacs.GLOBAL_COEFFICIENTS not in table.values()
    with pytest.raises(TypeError):
        table["USA"] = gdacs.GLOBAL_COEFFICIENTS  # type: ignore[index]


def test_untabled_country_uses_the_global_model() -> None:
    assert gdacs.earthquake_coefficients("usa") is gdacs.GLOBAL_COEFFICIENTS
    assert gdacs.earthquake_coefficients(" jpn ") is gdacs.COUNTRY_COEFFICIENTS["JPN"]


@pytest.mark.parametrize("code", ["JP", "JAPAN", "J1N", ""])
def test_bad_country_code_is_rejected(code: str) -> None:
    with pytest.raises(ValueError, match="alpha-3"):
        gdacs.earthquake_coefficients(code)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_bad_coefficient_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="c1_shakemap"):
        gdacs.EarthquakeCoefficients(c1_shakemap=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lcc", "factor"),
    [(0.9, 0.5), (9.3, 1.5), (5.1, 1.0), (6.0, 0.5 + 5.1 / 8.4), (0.0, 0.5), (10.0, 1.5)],
)
def test_coping_capacity_factor(lcc: float, factor: float) -> None:
    assert gdacs.coping_capacity_factor(lcc) == pytest.approx(factor)


@pytest.mark.parametrize("lcc", [-0.1, 10.1, float("nan")])
def test_lcc_outside_its_scale_is_rejected(lcc: float) -> None:
    with pytest.raises(ValueError, match="inform_lcc"):
        gdacs.coping_capacity_factor(lcc)


def test_coping_capacity_multiplies() -> None:
    assert gdacs.apply_coping_capacity(2.5, 0.5) == pytest.approx(1.25)
    assert gdacs.apply_coping_capacity(1.8, 1.5) == pytest.approx(2.7)


@pytest.mark.parametrize(("score", "factor"), [(1.0, 0.4), (1.0, 1.6), (float("nan"), 1.0)])
def test_coping_capacity_inputs_are_validated(score: float, factor: float) -> None:
    with pytest.raises(ValueError):
        gdacs.apply_coping_capacity(score, factor)


@pytest.mark.parametrize(
    ("score", "level"),
    [(2.0, "red"), (1.9999, "orange"), (1.0, "orange"), (0.9999, "green"), (-1.5, "green")],
)
def test_alert_level_boundaries(score: float, level: str) -> None:
    assert gdacs.alert_level(score) == level


@pytest.mark.parametrize(
    ("deaths", "displaced", "level"),
    [
        (1000, 0, "orange"),  # "more than 1,000 dead" is Red
        (1001, 0, "red"),
        (0, 800_000, "orange"),
        (0, 800_001, "red"),
        (100, 80_000, "green"),
        (101, 0, "orange"),
        (0, 80_001, "orange"),
    ],
)
def test_flood_levels(deaths: float, displaced: float, level: str) -> None:
    assert gdacs.flood_alert_level(deaths, displaced) == level


@pytest.mark.parametrize(
    ("depth", "radii", "match"),
    [
        (0.0, _RADII, "depth_km"),
        (10.0, {20: 1, 50: 2, 75: 3}, "missing 100"),
        (10.0, {20: 5, 50: 2, 75: 3, 100: 4}, "cumulative"),
        (10.0, {20: 0, 50: 0, 75: 0, 100: 0}, "no population"),
        (10.0, {20: -1, 50: 2, 75: 3, 100: 4}, "non-negative"),
    ],
)
def test_parameter_model_inputs_are_validated(
    depth: float, radii: dict[int, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        gdacs.earthquake_parameter_score(7.0, depth, radii)


@pytest.mark.parametrize("population", [-1.0, float("nan"), float("inf"), 10**400, True])
def test_populations_are_validated(population: float) -> None:
    with pytest.raises(ValueError, match="population"):
        gdacs.earthquake_shakemap_score(population, 0, 0)
    with pytest.raises(ValueError):
        gdacs.flood_alert_level(population, 0)
