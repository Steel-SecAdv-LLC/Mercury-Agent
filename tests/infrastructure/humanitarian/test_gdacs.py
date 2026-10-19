# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``crisis_monitoring.gdacs``: the published GDACS earthquake, flood and cyclone models.

Expected values are worked by hand from gdacs.org/Knowledge/models_eq.aspx,
models_fl.aspx and the GDACS MHEWS user manual's cyclone table; country terms
are spot-checked against GDACS's 2017 coefficient table; and the module is
checked against GDACS's own scores and levels for real events
(tests/fixtures/gdacs).
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
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


# --- Real GDACS events (tests/fixtures/gdacs, see PROVENANCE.json) -----------

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "gdacs"


def _alert_parameters(path: Path) -> dict[str, str]:
    scalars: dict[str, str] = {}
    for group in json.loads(path.read_text())["datums"]:
        for datum in group["datum"]:
            for scalar in (datum.get("scalars") or {}).get("scalar", []):
                scalars[scalar["name"]] = scalar["value"]
    return scalars


def _shakemap_events() -> list[tuple[str, str, dict[str, str]]]:
    events = []
    for path in sorted(_FIXTURES.glob("eq_shakemap_alert_*.json")):
        event_id = path.stem.rsplit("_", 1)[1]
        details = json.loads((_FIXTURES / f"eq_event_{event_id}.json").read_text())
        events.append((event_id, details["properties"]["iso3"], _alert_parameters(path)))
    return events


# Cross-border events: GDACS averaged the countries' coefficients (AFG-PAK
# reproduces its -1.855/0.845, PHL-IDN its -0.65/0.53) and summed their coping
# factors, a rule its pages do not publish.
_CROSS_BORDER = {"1498339", "1526507", "1544720"}


def test_cross_border_events_are_the_ones_with_summed_factors() -> None:
    """Exactly these events carry a coping factor above GDACS's published 1.5 maximum."""
    summed = {e for e, _, alert in _shakemap_events() if float(alert["LccNormalised"]) > 1.5}
    assert summed == _CROSS_BORDER


def test_country_terms_match_gdacs_live_coefficients() -> None:
    """GDACS's reported intercept and slope are the 2017 table's for each country."""
    checked = 0
    for event_id, iso3, alert in _shakemap_events():
        if event_id in _CROSS_BORDER or float(alert["ScaledPopulation"]) == 0.0:
            continue
        terms = gdacs.earthquake_coefficients(iso3)
        assert float(alert["CoeffIntersept"]) == pytest.approx(-0.59 + terms.c1_shakemap), event_id
        assert float(alert["CoeffSlope"]) == pytest.approx(0.53 + terms.c2_shakemap), event_id
        checked += 1
    assert checked == 21


def test_shakemap_scores_reproduce_gdacs() -> None:
    """GDACS's own scaled population and coping factor give GDACS's own score."""
    reproduced = 0
    for event_id, iso3, alert in _shakemap_events():
        if event_id in _CROSS_BORDER:
            continue
        log_scaled = float(alert["ScaledPopulation"])
        score = gdacs.shakemap_score(
            0.0 if log_scaled == 0.0 else 10.0**log_scaled, gdacs.earthquake_coefficients(iso3)
        )
        level = alert["AlertLevel"].lower()
        if log_scaled == 0.0:  # nobody above MMI VI: GDACS records 0 or 0.01, Green
            assert score == 0.0 and level == "green", event_id
            continue
        final = gdacs.apply_coping_capacity(score, float(alert["LccNormalised"]))
        assert final == pytest.approx(float(alert["AlertScore"]), rel=1e-9), event_id
        assert gdacs.alert_level(final) == level, event_id
        reproduced += 1
    assert reproduced == 21


_COUNTRY_ISO3 = {
    "Bahamas": "BHS",
    "Bermuda": "BMU",
    "Canada": "CAN",
    "China": "CHN",
    "Cuba": "CUB",
    "Greenland": "GRL",
    "Guam": "GUM",
    "Haiti": "HTI",
    "India": "IND",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Northern Mariana Islands": "MNP",
    "Philippines": "PHL",
    "Russian Federation": "RUS",
    "Sri Lanka": "LKA",
    "United States": "USA",
}


def _advisory_time(text: str) -> datetime:
    return datetime.strptime(text, "%d %b %Y %H:%M")


def test_cyclone_levels_reproduce_gdacs_track_points() -> None:
    """Every GDACS track point of five storms.

    One is a Red more than three days ahead that GDACS lowered to Orange
    (MELISSA-25 over Jamaica, 28 Oct 2025 06:00); the 72 h rule must give the same.
    """
    points = 0
    lowered = []
    for path in sorted(_FIXTURES.glob("tc_timeline_*.json")):
        items = json.loads(path.read_text())["items"]
        now = _advisory_time(next(i["advisory_datetime"] for i in items if i["current"] == "true"))
        for item in items:
            forecast = item["actual"] != "True"
            lead = (_advisory_time(item["advisory_datetime"]) - now).total_seconds() / 3600
            country = _COUNTRY_ISO3.get(item["country"])
            point = gdacs.CyclonePoint(
                max_wind_kt=round(float(item["wind_speed"]) / 0.514444),  # m/s from whole knots
                population_34kt=float(item["pop39"]),
                population_64kt=float(item["pop74"]),
                vulnerability=(
                    "low" if country is None else gdacs.tropical_cyclone_vulnerability(country)
                ),
                lead_time_hours=max(lead, 0.0) if forecast else 0.0,
            )
            level = gdacs.tropical_cyclone_point_level(point)
            assert level == item["alertcolor"].lower(), (path.name, item["advisory_datetime"])
            points += 1
            observed = dataclasses.replace(point, lead_time_hours=0.0)
            if gdacs.tropical_cyclone_point_level(observed) == "red" and level == "orange":
                assert item["alertdecreased"] == "true"
                lowered.append((item["country"], item["advisory_datetime"]))
    assert points == 152
    assert lowered == [("Jamaica", "28 Oct 2025 06:00")]


# --- Tropical cyclones --------------------------------------------------------


@pytest.mark.parametrize(
    ("wind", "category"),
    [
        (0, "tropical_depression"),
        (33, "tropical_depression"),
        (34, "tropical_storm"),
        (63, "tropical_storm"),
        (64, "category_1"),
        (82, "category_1"),
        (83, "category_2"),
        (95, "category_2"),
        (96, "category_3"),
        (112, "category_3"),
        (113, "category_4"),
        (136, "category_4"),
        (137, "category_5"),
    ],
)
def test_saffir_simpson_classes_in_knots(wind: float, category: str) -> None:
    assert gdacs.tropical_cyclone_category(wind) == category


def test_ladder_matches_the_hurricane_detector() -> None:
    """One Saffir-Simpson definition across the codebase."""
    pytest.importorskip("torch")
    from omni_mercury_engine.detectors.geological.hurricane_detector import HurricaneDetector

    detector = HurricaneDetector()
    for wind in np.arange(0.0, 200.5, 0.5):
        assert gdacs.tropical_cyclone_category(wind) == detector._classify_category(wind), wind


_TD, _TS, _C1, _C2, _C3, _C4, _C5 = 30, 50, 70, 90, 100, 120, 150


@pytest.mark.parametrize(
    ("wind", "vulnerability", "p34", "p64", "country", "level"),
    [
        (_TD, "high", 2e7, 2e6, None, "green"),
        (_TS, "high", 10_000_000, 0, None, "green"),
        (_TS, "high", 10_000_001, 0, None, "orange"),
        (_TS, "medium", 2e7, 0, None, "green"),
        (_C1, "low", 2e7, 5e6, None, "green"),
        (_C1, "medium", 0, 100_000, None, "green"),
        (_C1, "medium", 0, 100_001, None, "orange"),
        (_C1, "medium", 0, 50_001, 500_000, "orange"),  # over 10% of the country
        (_C1, "high", 0, 1_000_001, None, "red"),
        (_C1, "high", 0, 1_000_000, None, "orange"),
        (_C1, "high", 10_000_001, 0, None, "orange"),
        (_C1, "high", 0, 1_000, None, "green"),
        (_C2, "medium", 0, 50_001, None, "orange"),
        (_C2, "medium", 0, 50_000, None, "green"),
        (_C2, "high", 0, 1_000_001, None, "red"),
        (_C2, "high", 0, 60_000, None, "orange"),
        (_C3, "low", 0, 500_001, None, "orange"),
        (_C3, "low", 0, 500_000, None, "green"),
        (_C3, "low", 0, 400_000, 100_000, "green"),  # no 10% rule for Low at Cat 3
        (_C3, "medium", 0, 100_001, None, "red"),
        (_C3, "medium", 0, 100_000, None, "orange"),
        (_C3, "high", 0, 20_000, None, "green"),
        (_C3, "high", 0, 15_001, 150_000, "red"),
        (_C4, "low", 0, 1_000_001, None, "red"),
        (_C4, "low", 0, 1_000_000, None, "orange"),
        (_C4, "low", 0, 100_000, None, "green"),
        (_C4, "low", 0, 60_000, 500_000, "orange"),
        (_C5, "medium", 0, 100_001, None, "red"),
        (_C5, "high", 0, 20_001, None, "orange"),
        (_C5, "high", 0, 20_000, None, "green"),
    ],
)
def test_mhews_table_cells(
    wind: float, vulnerability: str, p34: float, p64: float, country: float | None, level: str
) -> None:
    point = gdacs.CyclonePoint(wind, p34, p64, vulnerability, country)
    assert gdacs.tropical_cyclone_point_level(point) == level


def test_red_more_than_three_days_out_is_orange() -> None:
    red: dict[str, Any] = {
        "max_wind_kt": _C4,
        "population_34kt": 0,
        "population_64kt": 2e5,
        "vulnerability": "medium",
    }
    assert (
        gdacs.tropical_cyclone_point_level(gdacs.CyclonePoint(**red, lead_time_hours=72)) == "red"
    )
    assert gdacs.tropical_cyclone_point_level(gdacs.CyclonePoint(**red, lead_time_hours=72.5)) == (
        "orange"
    )
    orange: dict[str, Any] = {**red, "population_64kt": 3e4}
    assert gdacs.tropical_cyclone_point_level(
        gdacs.CyclonePoint(**orange, lead_time_hours=100)
    ) == ("orange")


def test_storm_level_is_its_highest_point() -> None:
    track = [
        gdacs.CyclonePoint(_TS, 1e6, 0, "medium"),
        gdacs.CyclonePoint(_C3, 2e6, 3e4, "medium"),
        gdacs.CyclonePoint(_C1, 1e6, 1e4, "medium"),
    ]
    assert gdacs.tropical_cyclone_level(track) == "orange"
    assert gdacs.tropical_cyclone_score("orange") == 1.5
    with pytest.raises(ValueError, match="at least one"):
        gdacs.tropical_cyclone_level([])


@pytest.mark.parametrize(("level", "score"), [("green", 0.5), ("orange", 1.5), ("red", 2.5)])
def test_cyclone_scores(level: str, score: float) -> None:
    assert gdacs.tropical_cyclone_score(level) == score


@pytest.mark.parametrize(
    ("surge", "level"),
    [(0.0, None), (0.5, "green"), (1.0, "orange"), (2.99, "orange"), (3.0, "red")],
)
def test_storm_surge_levels(surge: float, level: str | None) -> None:
    assert gdacs.storm_surge_level(surge) == level


def test_vulnerability_classes() -> None:
    table = gdacs.TC_VULNERABILITY
    assert sum(v == "medium" for v in table.values()) == 35
    assert sum(v == "high" for v in table.values()) == 19
    assert gdacs.tropical_cyclone_vulnerability("jam") == "medium"
    assert gdacs.tropical_cyclone_vulnerability("HTI") == "high"
    assert gdacs.tropical_cyclone_vulnerability("CUB") == "low"
    assert gdacs.tropical_cyclone_vulnerability("TZA") == "medium"  # GDACS live feed
    assert gdacs.tropical_cyclone_vulnerability("YEM") == "high"  # GDACS live feed
    with pytest.raises(ValueError, match="alpha-3"):
        gdacs.tropical_cyclone_vulnerability("Jamaica")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_wind_kt": -1.0}, "max_wind_kt"),
        ({"max_wind_kt": math.nan}, "max_wind_kt"),
        ({"population_64kt": -5.0}, "population_64kt"),
        ({"vulnerability": "severe"}, "vulnerability"),
        ({"country_population": math.inf}, "country_population"),
        ({"lead_time_hours": math.nan}, "lead_time_hours"),
    ],
)
def test_cyclone_point_is_validated(kwargs: dict[str, Any], match: str) -> None:
    point = {"max_wind_kt": 70.0, "population_34kt": 0.0, "population_64kt": 0.0} | kwargs
    with pytest.raises(ValueError, match=match):
        gdacs.CyclonePoint(**point)


def test_cyclone_point_stores_its_numbers_as_floats() -> None:
    """Numeric strings and Decimals pass validation and are compared as floats afterwards."""
    point = gdacs.CyclonePoint(
        70,
        Decimal("0"),  # type: ignore[arg-type]
        "60000",  # type: ignore[arg-type]
        "medium",
        country_population="500000",  # type: ignore[arg-type]
        lead_time_hours="100",  # type: ignore[arg-type]
    )
    assert all(
        type(value) is float
        for value in (
            point.max_wind_kt,
            point.population_34kt,
            point.population_64kt,
            point.country_population,
            point.lead_time_hours,
        )
    )
    assert gdacs.tropical_cyclone_point_level(point) == "orange"  # 60,000 > 10% of 500,000
