# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1", np.bool_(True)])
def test_bad_coefficient_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="c1_shakemap"):
        gdacs.EarthquakeCoefficients(c1_shakemap=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [np.int64(1), np.float32(1.0), np.float64(1.0), 1])
def test_numpy_numbers_are_accepted_and_stored_as_floats(value: object) -> None:
    """numpy integers do not subclass int; the table must still take them."""
    coefficients = gdacs.EarthquakeCoefficients(c1_shakemap=value)  # type: ignore[arg-type]
    assert type(coefficients.c1_shakemap) is float
    assert coefficients.c1_shakemap == 1.0


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


def test_zero_exposure_factor_zero_is_accepted() -> None:
    """GDACS reports factor 0 with score 0 on zero-exposure alerts (event 1568948)."""
    assert gdacs.apply_coping_capacity(0.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="zero-exposure"):
        gdacs.apply_coping_capacity(1.2, 0.0)


def test_summed_cross_border_factor_is_refused_with_the_reason() -> None:
    """A factor above 1.5 is a cross-border sum GDACS does not multiply by."""
    with pytest.raises(ValueError, match="cross-border"):
        gdacs.apply_coping_capacity(3.62, 2.35714285714286)  # event 1498339's values


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


# Cross-border events: GDACS reports the exposed countries' coefficients
# averaged (AFG-PAK reproduces its -1.855/0.845, PHL-IDN its -0.65/0.53, and
# the three-country CHL event its -0.78/0.5333) and their coping factors
# summed, a rule its pages do not publish. The sums decompose exactly into
# single-country factors recorded in this same sample (see
# test_cross_border_factors_are_sums_of_single_country_factors).
_CROSS_BORDER = {"1365556", "1437973", "1492605", "1498339", "1526507", "1544720"}

# GDACS keys the terms by the shaken territory, not the event's iso3:
# 1338219 (Chihshang, Taiwan) files under CHN but is scored with a Taiwan
# entry absent from the published 2017 table — the global intercept and slope
# with shakemap vulnerability exactly -0.56 (backed out of this event's
# score to 1.5e-15) — and China's coping factor; 1202436 (Kashgar) files
# under KGZ but is scored entirely with China's terms and factor.
_TERRITORY_KEYED = {
    "1338219": gdacs.EarthquakeCoefficients(vulnerability_shakemap=-0.56),
    "1202436": gdacs.COUNTRY_COEFFICIENTS["CHN"],
}


def test_cross_border_events_are_the_ones_with_summed_factors() -> None:
    """Exactly these events carry a coping factor above GDACS's published 1.5 maximum."""
    summed = {e for e, _, alert in _shakemap_events() if float(alert["LccNormalised"]) > 1.5}
    assert summed == _CROSS_BORDER


def test_country_terms_match_gdacs_live_coefficients() -> None:
    """GDACS's reported intercept and slope are the 2017 table's for each country."""
    checked = 0
    for event_id, iso3, alert in _shakemap_events():
        if event_id in _CROSS_BORDER or event_id in _TERRITORY_KEYED:
            continue
        if float(alert["ScaledPopulation"]) == 0.0:
            continue
        terms = gdacs.earthquake_coefficients(iso3)
        assert float(alert["CoeffIntersept"]) == pytest.approx(-0.59 + terms.c1_shakemap), event_id
        assert float(alert["CoeffSlope"]) == pytest.approx(0.53 + terms.c2_shakemap), event_id
        checked += 1
    assert checked == 25


def test_shakemap_scores_reproduce_gdacs() -> None:
    """GDACS's own scaled population and coping factor give GDACS's own score."""
    reproduced = 0
    for event_id, iso3, alert in _shakemap_events():
        if event_id in _CROSS_BORDER:
            continue
        coefficients = _TERRITORY_KEYED.get(event_id) or gdacs.earthquake_coefficients(iso3)
        log_scaled = float(alert["ScaledPopulation"])
        score = gdacs.shakemap_score(0.0 if log_scaled == 0.0 else 10.0**log_scaled, coefficients)
        level = alert["AlertLevel"].lower()
        if log_scaled == 0.0:  # nobody above MMI VI: GDACS records 0 or 0.01, Green
            assert score == 0.0 and level == "green", event_id
            continue
        final = gdacs.apply_coping_capacity(score, float(alert["LccNormalised"]))
        assert final == pytest.approx(float(alert["AlertScore"]), rel=1e-9), event_id
        assert gdacs.alert_level(final) == level, event_id
        reproduced += 1
    assert reproduced == 27


def _factor(event_id: str) -> float:
    """The recorded LccNormalised of one fixture event."""
    for eid, _iso3, alert in _shakemap_events():
        if eid == event_id:
            return float(alert["LccNormalised"])
    raise AssertionError(f"no fixture for event {event_id}")


def test_cross_border_factors_are_sums_of_single_country_factors() -> None:
    """Each cross-border factor equals the sum of the countries' own recorded factors.

    Every addend is a single-country event's LccNormalised from this same
    fixture sample, so the sum rule is verified from GDACS's own outputs
    alone (to the full precision GDACS reports).
    """
    sums = {
        "1498339": ("1498907", "1487880"),  # AFG + PAK, both 2025
        "1526507": ("1511252", "1500178"),  # BGD + IND
        "1544720": ("1504796", "1496100"),  # PHL + IDN
        "1365556": ("1478853", "1376865"),  # ECU + PER
    }
    for cross, parts in sums.items():
        total = sum(_factor(part) for part in parts)
        assert total == pytest.approx(_factor(cross), abs=1e-9), cross


def test_cross_border_coefficients_are_country_averages() -> None:
    """Reported cross-border intercept/slope average the countries' terms.

    Two-country events average two tabled countries' terms; 1437973 (the
    2024 Atacama M7.4 near the Chile-Bolivia-Argentina triple border)
    averages Chile's terms with TWO global-model countries, pinning the rule
    for three countries.
    """
    by_id = {eid: alert for eid, _iso3, alert in _shakemap_events()}

    def live(terms: gdacs.EarthquakeCoefficients) -> tuple[float, float]:
        return (-0.59 + terms.c1_shakemap, 0.53 + terms.c2_shakemap)

    cases = {
        "1498339": [
            live(gdacs.COUNTRY_COEFFICIENTS["AFG"]),
            live(gdacs.COUNTRY_COEFFICIENTS["PAK"]),
        ],
        "1544720": [
            live(gdacs.COUNTRY_COEFFICIENTS["PHL"]),
            live(gdacs.COUNTRY_COEFFICIENTS["IDN"]),
        ],
        "1437973": [
            live(gdacs.COUNTRY_COEFFICIENTS["CHL"]),
            live(gdacs.GLOBAL_COEFFICIENTS),
            live(gdacs.GLOBAL_COEFFICIENTS),
        ],
    }
    for event_id, terms in cases.items():
        alert = by_id[event_id]
        n = len(terms)
        assert float(alert["CoeffIntersept"]) == pytest.approx(
            sum(t[0] for t in terms) / n
        ), event_id
        assert float(alert["CoeffSlope"]) == pytest.approx(sum(t[1] for t in terms) / n), event_id


def test_live_flood_levels_do_not_follow_the_published_model() -> None:
    """The recorded 2024-2026 flood sample pins the documented 25/56 agreement.

    GDACS's models_fl page gives deaths/displaced thresholds; its live
    levels demonstrably come from something else (Orange events with
    single-digit death tolls). flood_alert_level implements the published
    model and must keep saying so rather than claiming live parity.
    """
    recorded = json.loads((_FIXTURES / "fl_levels_vs_sendai_2024_2026.json").read_text())
    events = recorded["events"]
    assert len(events) == 56
    agree = sum(
        gdacs.flood_alert_level(e["death"], e["displaced"]) == e["level"].lower() for e in events
    )
    assert agree == 25
    assert "does not reproduce GDACS's live level" in (gdacs.flood_alert_level.__doc__ or "")


def test_no_final_shakemap_events_are_scored_on_the_preliminary_shakemap() -> None:
    """GDACS scores a no-final-shakemap event with the shakemap model anyway.

    Event 1499045 (Orange, Afghanistan 2025-09-05) lists no final shakemap;
    its score comes from the ``shake_preliminary`` impact, and it is the
    SHAKEMAP formula on the preliminary map — Afghanistan's terms including
    the 1.0 shakemap vulnerability, times Afghanistan's coping factor — not
    the published "EQ parameters" model (whose formula on the event's own
    published SUMPOP radii gives 1.65 before coping, irreconcilable with the
    reported score). No event in a 2018-2026 sweep of GDACS's live alerts
    carried a score the parameters model reproduces, so that model is
    implemented from gdacs.org/Knowledge/models_eq.aspx but unexercised by
    the live feed.
    """
    alert = _alert_parameters(_FIXTURES / "eq_shakemap_prelim_1499045.json")
    afg = gdacs.earthquake_coefficients("AFG")
    score = gdacs.shakemap_score(10.0 ** float(alert["ScaledPopulation"]), afg)
    final = gdacs.apply_coping_capacity(score, float(alert["LccNormalised"]))
    assert final == pytest.approx(float(alert["AlertScore"]), rel=1e-9)
    assert gdacs.alert_level(final) == alert["AlertLevel"].lower()
    details = json.loads((_FIXTURES / "eq_event_1499045.json").read_text())["properties"]
    resources: dict[str, str] = {}
    for impact in details["impacts"]:
        resources.update(impact.get("resource") or {})
    assert "shakemap" not in resources and "shake_preliminary" in resources
    assert float(details["episodealertscore"]) == pytest.approx(final)


def test_territory_keyed_events_use_the_shaken_territorys_terms() -> None:
    """GDACS keys coefficients by the shaken territory, not the event's iso3.

    1338219 (Chihshang, Taiwan; iso3 CHN) carries the GLOBAL intercept and
    slope, not China's calibrated ones; 1202436 (Kashgar; iso3 KGZ) carries
    China's. Both carry China's coping factor and reproduce GDACS's score
    with those terms (checked in test_shakemap_scores_reproduce_gdacs).
    """
    by_id = {eid: (iso3, alert) for eid, iso3, alert in _shakemap_events()}
    iso3, alert = by_id["1338219"]
    assert iso3 == "CHN"
    assert float(alert["CoeffIntersept"]) == pytest.approx(-0.59)  # global, not CHN's -0.37
    iso3, alert = by_id["1202436"]
    assert iso3 == "KGZ"
    chn = gdacs.COUNTRY_COEFFICIENTS["CHN"]
    assert float(alert["CoeffIntersept"]) == pytest.approx(-0.59 + chn.c1_shakemap)
    assert float(alert["CoeffSlope"]) == pytest.approx(0.53 + chn.c2_shakemap)


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


def test_small_territories_need_the_ten_percent_rule() -> None:
    """The manual's "(100k or 10% poptot)" reading, confirmed on live points.

    Sweeping all 84 orange/red storms of 2023-2026 (2,302 track points), the
    table reproduces GDACS's colour everywhere except small-territory points
    whose 64 kt population exceeds 10% of the territory's population — there
    GDACS upgrades although the absolute thresholds are not met. With the
    census populations supplied, those points grade to GDACS's colour; a
    caller omitting ``country_population`` under-grades them. Two recorded
    exceptions stay open: a Vanuatu pair fits only a dated (~250k)
    population grid, and one Tanzania tropical-storm point needs either a
    High class for Tanzania or the 10M rule at Medium.
    """
    recorded = json.loads((_FIXTURES / "tc_small_territory_points.json").read_text())
    populations = {k: v["population"] for k, v in recorded["populations"].items()}
    checked = 0
    for point in recorded["points"]:
        iso3 = point["iso3"]
        if iso3 not in ("GUM", "MNP", "TCA", "VUT") or not point["gdacs"]:
            continue
        population = populations[iso3]
        if point["p64"] <= 0.1 * population:
            continue  # the two Vanuatu marginals: below 10% of the census count
        with_population = gdacs.CyclonePoint(
            point["wind_kt"],
            point["p34"],
            point["p64"],
            point["vul"],
            country_population=population,
            lead_time_hours=point["lead_h"],
        )
        assert gdacs.tropical_cyclone_point_level(with_population) == point["gdacs"], point
        without = dataclasses.replace(with_population, country_population=None)
        assert gdacs.tropical_cyclone_point_level(without) == point["ours"], point
        checked += 1
    assert checked >= 18
