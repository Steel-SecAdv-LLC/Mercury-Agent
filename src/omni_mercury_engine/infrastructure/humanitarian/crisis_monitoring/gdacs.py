# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""GDACS alert levels for earthquakes, floods and tropical cyclones.

Implements the published Global Disaster Alert and Coordination System
(EC JRC / UN OCHA) models, which aim to classify events "according to the
likelihood that the affected societies can no longer cope at national level and
will require humanitarian intervention":

- Earthquakes, https://www.gdacs.org/Knowledge/models_eq.aspx
  - Shakemap model: if no populated place is shaken above MMI VI the level is
    Green with score 0; otherwise Scaled Population = 10·P(IX) + P(VIII) +
    0.1·P(VII) and Score = (-0.59 + C1) + (0.53 + C2)·log10(Scaled Population)
    + country shakemap vulnerability.
  - "EQ parameters" model (no shakemap): Scaled Population = 10·P20 +
    2·(P50-P20) + 0.5·(P75-P50) + 0.1·(P100-P75), with Px the population within
    x km of the epicentre; Raw = -7.75 + 0.82·Mw - 0.53·log10(depth km) +
    0.72·log10(Scaled Population); Score = C1·Raw + C2 + country vulnerability.
    Implemented from the published page, but UNEXERCISED by GDACS's live
    feed: a 2018-2026 sweep found no alert this formula reproduces — events
    without a final shakemap are scored with the shakemap model on a
    PRELIMINARY shakemap (event 1499045 reproduces to full precision), even
    though the SUMPOP radii this model needs are published per event.
  - Final score = model score x the coping-capacity factor, INFORM Lack of
    Coping Capacity normalised to 0.5 (Switzerland) - 1.5 (South Sudan).
  - Level: Red >= 2, Orange >= 1, Green below 1.
- Floods, https://www.gdacs.org/Knowledge/models_fl.aspx: Red for more than
  1,000 dead or 800,000 displaced, Orange for more than 100 dead or 80,000
  displaced, Green otherwise. This is the PUBLISHED model only: GDACS's live
  flood levels demonstrably do not follow it (25 of 56 orange/red events of
  2024-2026 match it against their own Sendai casualty tallies, with Orange
  granted at single-digit death tolls — the live levels come from GDACS's
  flood-impact forecasting, which it does not publish; see
  tests/fixtures/gdacs/fl_levels_vs_sendai_2024_2026.json).
- Tropical cyclones, GDACS MHEWS user manual (JRC141661, 2025; version of
  23/09/2026), section 4.7: the wind level of each track point from its
  Saffir-Simpson class, the people in its 34 kt and 64 kt wind buffers and the
  exposed country's vulnerability (Table 1, :func:`tropical_cyclone_point_level`);
  the storm takes its highest point; a Red more than three days ahead is reduced
  to Orange; scores 0.5, 1.5 and 2.5. The storm-surge level (Red >= 3 m, Orange
  >= 1 m) is reported beside it and is not in the score; the manual is its only
  source — GDACS's public API exposes surge solely as map resources
  (surge_maxheight_* images on TC events), with no numeric height or surge
  colour to check the thresholds against.

Earthquake country terms come from GDACS's published table
(gdacs.org/Documents/2017_GDACS_EQ_algorithm_coefficients.xlsx, © European
Union, CC BY 4.0): the 38 countries with their own calibration are in
:data:`COUNTRY_COEFFICIENTS`; every other country uses the global model, as
GDACS does. The table's 2017 INFORM LCC column is not included: GDACS's live
factors use newer LCC values with the same normalisation, so callers supply a
current LCC.

Checked against GDACS's own outputs (tests/fixtures/gdacs):

- Earthquakes: given GDACS's ``ScaledPopulation`` and coping factor,
  :func:`shakemap_score` reproduces GDACS's shakemap ``AlertScore`` for every
  single-country event in the sample (27 of 27). On cross-border events GDACS
  reports the exposed countries' coefficients averaged and their coping
  factors summed — checked on six recorded events (a 2018-2026 sweep of its
  live alerts found no counterexample), including a three-country event
  (1437973, Atacama 2024: Chile's terms averaged with two global-model
  neighbours'), and four of the six summed factors decompose exactly into
  single-country factors recorded in this same sample. But GDACS's final
  cross-border ``AlertScore`` is NOT the averaged-coefficient score times the
  summed factor (1544720: 1.86 reported vs 5.24 multiplied) and the rule it
  does use is unpublished, so this module scores single-country exposure
  only. GDACS keys the terms by the shaken territory, not the event's
  ``iso3``: a Taiwan event files under CHN with a live Taiwan entry absent
  from the 2017 table (global intercept/slope, shakemap vulnerability
  -0.56), and an event can file under a border country (1202436, Kashgar,
  filed KGZ) while being scored entirely with its neighbour's terms and
  factor. Its live ``ScaledPopulation`` is also not the band formula above
  applied to its own MMI band totals (-0.27 to +0.17 apart in log10 over 21
  scored single-country events), so scores from
  :func:`earthquake_shakemap_score` approximate GDACS's: they give GDACS's
  level for 19 of the 21.
- Tropical cyclones: the look-up table reproduces GDACS's level at every track
  point of the sample's five storms, and at 2,278 of the 2,302 points of all
  84 orange/red storms of 2023-2026 with no country population supplied. The
  other 24 points confirm the manual's "(100k or 10% poptot)" reading on live
  data: they lie in small territories (Guam, the Northern Marianas, Turks and
  Caicos, Vanuatu) whose 64 kt population tops 10% of the territory's
  population, and grade to GDACS's colour once ``country_population`` is
  supplied — so callers must supply it for small countries (see
  tests/fixtures/gdacs/tc_small_territory_points.json; one Tanzania
  tropical-storm point and a dated-population Vanuatu pair stay recorded as
  open).

GDACS calls its scores empirical and "intended to give a fast and rough warning".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "COUNTRY_COEFFICIENTS",
    "GLOBAL_COEFFICIENTS",
    "TC_VULNERABILITY",
    "CyclonePoint",
    "EarthquakeCoefficients",
    "alert_level",
    "apply_coping_capacity",
    "coping_capacity_factor",
    "earthquake_coefficients",
    "earthquake_parameter_score",
    "earthquake_shakemap_score",
    "flood_alert_level",
    "shakemap_score",
    "storm_surge_level",
    "tropical_cyclone_category",
    "tropical_cyclone_level",
    "tropical_cyclone_point_level",
    "tropical_cyclone_score",
    "tropical_cyclone_vulnerability",
]

# The 2017 table normalises INFORM LCC linearly from its 0.9 minimum to its
# 9.3 maximum onto 0.5-1.5 (exact for all 191 rows).
_LCC_MIN, _LCC_MAX = 0.9, 9.3


def _finite(name: str, value: float) -> float:
    """``value`` as a finite float; booleans, non-numbers and overflow raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value}")
    return number


@dataclass(frozen=True)
class EarthquakeCoefficients:
    """Country terms of the GDACS earthquake models; the defaults are the global model.

    Attributes:
        c1_shakemap: Intercept correction of the shakemap model.
        c2_shakemap: Slope correction of the shakemap model.
        vulnerability_shakemap: Country shakemap vulnerability.
        c1_classic: Slope on the raw "EQ parameters" score.
        c2_classic: Intercept of the "EQ parameters" score.
        vulnerability_classic: Country classic vulnerability.
    """

    c1_shakemap: float = 0.0
    c2_shakemap: float = 0.0
    vulnerability_shakemap: float = 0.0
    c1_classic: float = 1.0
    c2_classic: float = 0.0
    vulnerability_classic: float = 0.0

    def __post_init__(self) -> None:
        """Reject a term that is not a finite number and store it as a float.

        numpy numbers do not all subclass ``int``/``float`` (np.int64 and
        np.float32 do not), so they are accepted explicitly; strings and
        other non-numbers stay rejected.
        """
        for name, value in vars(self).items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            object.__setattr__(self, name, _finite(name, value))  # frozen: set once, here


GLOBAL_COEFFICIENTS = EarthquakeCoefficients()

# GDACS 2017 table, countries whose terms differ from the global model, by ISO3.
COUNTRY_COEFFICIENTS: Mapping[str, EarthquakeCoefficients] = MappingProxyType(
    {
        "AFG": EarthquakeCoefficients(0.0, 0.0, 1.0, 1.5, 0.07, 0.0),  # Afghanistan
        "BTN": EarthquakeCoefficients(0.0, 0.0, -0.06, 1.0, 0.0, -0.07),  # Bhutan
        "CHL": EarthquakeCoefficients(-0.57, 0.01, 0.0, 1.44, -1.48, 0.0),  # Chile
        "CHN": EarthquakeCoefficients(0.22, 0.23, 0.0, 1.39, -0.24, 0.0),  # China
        "COD": EarthquakeCoefficients(0.0, 0.0, 0.32, 1.83, -2.02, 0.0),  # Congo DR
        "COL": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, -0.15),  # Colombia
        "CRI": EarthquakeCoefficients(0.0, 0.0, -1.0, 1.0, 0.0, -0.58),  # Costa Rica
        "DZA": EarthquakeCoefficients(0.0, 0.0, 0.0, 0.32, 0.24, 0.0),  # Algeria
        "ECU": EarthquakeCoefficients(0.0, 0.0, 0.37, 2.46, -2.57, 0.0),  # Ecuador
        "ESP": EarthquakeCoefficients(0.0, 0.0, -0.41, 1.0, 0.0, 0.13),  # Spain
        "GRC": EarthquakeCoefficients(0.0, 0.0, -0.25, 1.0, 0.0, -0.61),  # Greece
        "GTM": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, -0.19),  # Guatemala
        "HND": EarthquakeCoefficients(0.0, 0.0, -0.17, 1.0, 0.0, -0.43),  # Honduras
        "HTI": EarthquakeCoefficients(0.0, 0.0, 1.0, 1.0, 0.0, 1.0),  # Haiti
        "IDN": EarthquakeCoefficients(-1.14, 0.26, 0.0, 1.54, -0.64, 0.0),  # Indonesia
        "IND": EarthquakeCoefficients(0.0, 0.0, -0.64, 0.74, -0.08, 0.0),  # India
        "IRN": EarthquakeCoefficients(0.7, -0.13, 0.0, 1.29, 0.26, 0.0),  # Iran
        "ITA": EarthquakeCoefficients(0.0, 0.0, 0.85, 1.0, 0.0, 0.29),  # Italy
        "JPN": EarthquakeCoefficients(0.0, 0.0, -1.0, 0.73, -0.22, 0.0),  # Japan
        "KGZ": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, -0.25),  # Kyrgyzstan
        "MAR": EarthquakeCoefficients(0.0, 0.0, 0.68, 1.0, 0.0, 0.83),  # Morocco
        "MEX": EarthquakeCoefficients(0.0, 0.0, -1.0, 1.0, 0.0, -1.0),  # Mexico
        "MMR": EarthquakeCoefficients(0.0, 0.0, 0.23, 0.89, -0.28, 0.0),  # Myanmar
        "MWI": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, -0.49),  # Malawi
        "MYS": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, 0.32),  # Malaysia
        "NIC": EarthquakeCoefficients(0.0, 0.0, -0.53, 1.0, 0.0, -1.0),  # Nicaragua
        "NPL": EarthquakeCoefficients(0.0, 0.0, 0.57, 4.73, -9.39, 0.0),  # Nepal
        "NZL": EarthquakeCoefficients(0.0, 0.0, -0.71, 1.0, 0.0, -0.33),  # New Zealand
        "PAK": EarthquakeCoefficients(-2.53, 0.63, 0.0, 2.1, -0.9, 0.0),  # Pakistan
        "PER": EarthquakeCoefficients(0.0, 0.0, 0.43, 1.0, 0.0, 0.27),  # Peru
        "PHL": EarthquakeCoefficients(1.02, -0.26, 0.0, 1.0, 0.0, -0.18),  # Philippines
        "RUS": EarthquakeCoefficients(0.0, 0.0, 0.0, 0.74, -0.13, 0.0),  # Russian Federation
        "SLB": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, 0.44),  # Solomon Islands
        "SRB": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, -0.97),  # Serbia
        "TJK": EarthquakeCoefficients(0.0, 0.0, 0.27, 1.0, 0.0, 0.04),  # Tajikistan
        "TUR": EarthquakeCoefficients(0.0, 0.0, 0.32, 1.37, -0.11, 0.0),  # Turkey
        "TZA": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, 0.32),  # Tanzania
        "ZAF": EarthquakeCoefficients(0.0, 0.0, 0.0, 1.0, 0.0, -0.12),  # South Africa
    }
)


def earthquake_coefficients(iso3: str) -> EarthquakeCoefficients:
    """GDACS terms for a country by ISO 3166-1 alpha-3 code; the global model if not tabled."""
    code = str(iso3).strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"expected an ISO 3166-1 alpha-3 code, got {iso3!r}")
    return COUNTRY_COEFFICIENTS.get(code, GLOBAL_COEFFICIENTS)


def _population(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative population, got {value}")
    return number


def earthquake_shakemap_score(
    population_mmi_vii: float,
    population_mmi_viii: float,
    population_mmi_ix: float,
    coefficients: EarthquakeCoefficients = GLOBAL_COEFFICIENTS,
) -> float:
    """GDACS shakemap score before coping capacity.

    Args:
        population_mmi_vii: People exposed to MMI VII.
        population_mmi_viii: People exposed to MMI VIII.
        population_mmi_ix: People exposed to MMI IX, as GDACS bands it.
        coefficients: Country terms (:func:`earthquake_coefficients`).

    Returns:
        The score; 0 when nobody is exposed above MMI VI.
    """
    p7 = _population("population_mmi_vii", population_mmi_vii)
    p8 = _population("population_mmi_viii", population_mmi_viii)
    p9 = _population("population_mmi_ix", population_mmi_ix)
    return shakemap_score(10.0 * p9 + p8 + 0.1 * p7, coefficients)


def shakemap_score(
    scaled_population: float, coefficients: EarthquakeCoefficients = GLOBAL_COEFFICIENTS
) -> float:
    """GDACS shakemap score from a scaled population, before coping capacity.

    GDACS reports ``ScaledPopulation`` with each shakemap alert as its log10,
    or as 0 when nobody is shaken at MMI VII or above (``PopulationOverVII``
    0). Passing 10**that value, or 0 in that case, reproduces GDACS's score.

    Args:
        scaled_population: Intensity-weighted exposed population.
        coefficients: Country terms (:func:`earthquake_coefficients`).

    Returns:
        The score; 0 when the scaled population is 0.
    """
    scaled = _population("scaled_population", scaled_population)
    if scaled <= 0.0:
        return 0.0
    c = coefficients
    return (
        (-0.59 + c.c1_shakemap)
        + (0.53 + c.c2_shakemap) * math.log10(scaled)
        + c.vulnerability_shakemap
    )


def earthquake_parameter_score(
    magnitude: float,
    depth_km: float,
    population_within_km: Mapping[int, float],
    coefficients: EarthquakeCoefficients = GLOBAL_COEFFICIENTS,
) -> float:
    """GDACS "EQ parameters" score before coping capacity.

    Args:
        magnitude: Moment magnitude Mw.
        depth_km: Hypocentre depth in km, as GDACS reports it.
        population_within_km: Cumulative population within 20, 50, 75 and
            100 km of the epicentre, keyed by those radii.
        coefficients: Country terms (:func:`earthquake_coefficients`).

    Returns:
        The score.

    Raises:
        ValueError: If a radius is missing, the populations are not cumulative,
            or the depth or scaled population is not positive.
    """
    magnitude, depth_km = _finite("magnitude", magnitude), _finite("depth_km", depth_km)
    if not depth_km > 0.0:
        raise ValueError(f"depth_km must be > 0, got {depth_km}")
    try:
        p20, p50, p75, p100 = (
            _population(f"population within {r} km", population_within_km[r])
            for r in (20, 50, 75, 100)
        )
    except KeyError as missing:
        raise ValueError(
            f"population_within_km needs the 20, 50, 75 and 100 km radii; missing {missing}"
        ) from None
    if not p20 <= p50 <= p75 <= p100:
        raise ValueError("population_within_km must be cumulative (non-decreasing with radius)")
    scaled = 10.0 * p20 + 2.0 * (p50 - p20) + 0.5 * (p75 - p50) + 0.1 * (p100 - p75)
    if scaled <= 0.0:
        raise ValueError("no population within 100 km: the parameter model is undefined")
    raw = -7.75 + 0.82 * magnitude - 0.53 * math.log10(depth_km) + 0.72 * math.log10(scaled)
    c = coefficients
    return c.c1_classic * raw + c.c2_classic + c.vulnerability_classic


def coping_capacity_factor(inform_lcc: float) -> float:
    """GDACS factor from an INFORM Lack of Coping Capacity value (0-10).

    The 2017 table's linear map, 0.5 at LCC 0.9 to 1.5 at LCC 9.3, clipped to
    that range.
    """
    inform_lcc = _finite("inform_lcc", inform_lcc)
    if not 0.0 <= inform_lcc <= 10.0:
        raise ValueError(f"inform_lcc must be in [0, 10], got {inform_lcc}")
    factor = 0.5 + (inform_lcc - _LCC_MIN) / (_LCC_MAX - _LCC_MIN)
    return min(max(factor, 0.5), 1.5)


def apply_coping_capacity(score: float, factor: float) -> float:
    """Final GDACS score, ``score x factor``.

    GDACS also sets a final score to 1 when a model score above 2 falls below 1
    through coping capacity; with factors in the published 0.5-1.5 range that
    cannot happen, so no floor is applied.

    Two values outside the published range appear in GDACS's own alerts:

    - 0, reported (with score 0) when nobody is shaken above MMI VI. Accepted
      here only alongside a zero score, where the product is still 0.
    - Sums of several countries' factors on cross-border events (observed up
      to 2.36). Refused: GDACS's recorded alerts prove its final score is NOT
      the model score times that sum (event 1544720 reports 1.857, while its
      averaged-coefficient model score times its summed factor is 5.24), and
      GDACS does not publish the combination it uses, so multiplying here
      would fabricate a score far above GDACS's own.
    """
    score, factor = _finite("score", score), _finite("factor", factor)
    if factor == 0.0:
        if score != 0.0:
            raise ValueError(
                "a coping-capacity factor of 0 is only reported by GDACS for "
                f"zero-exposure alerts (score 0), got score {score}"
            )
        return 0.0
    if not 0.5 <= factor <= 1.5:
        raise ValueError(
            f"the coping-capacity factor must be in [0.5, 1.5], got {factor}. "
            "GDACS reports factors above 1.5 on cross-border events as the SUM "
            "of the countries' factors; its recorded alerts show the final "
            "score is not the model score times that sum, and the rule it "
            "does use is unpublished, so this module refuses to guess."
        )
    return score * factor


def alert_level(score: float) -> str:
    """GDACS level of a final earthquake score: ``red``, ``orange`` or ``green``."""
    score = _finite("score", score)
    if score >= 2.0:
        return "red"
    if score >= 1.0:
        return "orange"
    return "green"


def flood_alert_level(deaths: float, displaced: float) -> str:
    """Flood level per GDACS's PUBLISHED model (models_fl.aspx) from casualty counts.

    GDACS's live flood alert levels do not follow this rule (25 of 56
    recorded 2024-2026 orange/red events match it against their own Sendai
    tallies; the live levels come from unpublished flood-impact
    forecasting), so this classifies reported impacts on the published
    scale — it does not reproduce GDACS's live level for an event.
    """
    dead = _population("deaths", deaths)
    moved = _population("displaced", displaced)
    if dead > 1000 or moved > 800_000:
        return "red"
    if dead > 100 or moved > 80_000:
        return "orange"
    return "green"


# --- Tropical cyclones -------------------------------------------------------

# Saffir-Simpson Hurricane Wind Scale on 1-minute sustained wind in knots, as
# GDACS (models_tc.aspx) and NHC band it. Keep in step with
# detectors.geological.hurricane_detector, which a test checks knot by knot.
_SSHS_KT: tuple[tuple[float, str], ...] = (
    (137.0, "category_5"),
    (113.0, "category_4"),
    (96.0, "category_3"),
    (83.0, "category_2"),
    (64.0, "category_1"),
    (34.0, "tropical_storm"),
)

_TC_VULNERABILITY_CLASSES = ("low", "medium", "high")

# GDACS's published tropical-cyclone vulnerability classes (Giovanna report,
# EUR 25629, 2012, section 2.1.3; every other country is Low), with the two
# classes GDACS's live feed reported differently for 2023-2026 events
# (Tanzania Medium, Yemen High); the other 49 countries seen in that feed match.
TC_VULNERABILITY: Mapping[str, str] = MappingProxyType(
    {
        **dict.fromkeys(
            (
                "ABW", "AIA", "ASM", "BLZ", "CHN", "COK", "CPV", "CRI", "DOM", "FJI", "GUM",
                "HND", "JAM", "LAO", "MEX", "MNP", "MUS", "MYS", "NCL", "NIC", "PAK", "PHL",
                "PLW", "PRI", "REU", "SWZ", "TCA", "THA", "TON", "TTO", "TZA", "VCT", "VGB",
                "VUT", "WSM",
            ),
            "medium",
        ),
        **dict.fromkeys(
            (
                "BGD", "COM", "GTM", "HTI", "IND", "KHM", "MDG", "MHL", "MMR", "MOZ", "MSR",
                "NIU", "PNG", "PRK", "PYF", "SLB", "SLV", "VNM", "YEM",
            ),
            "high",
        ),
    }
)  # fmt: skip

_TC_SCORE = {"green": 0.5, "orange": 1.5, "red": 2.5}
_LEVEL_ORDER = ("green", "orange", "red")
# GDACS reduces a Red more than three days ahead of the forecast to Orange.
_TC_RED_HORIZON_HOURS = 72.0


@dataclass(frozen=True)
class CyclonePoint:
    """One track point of a tropical cyclone, as GDACS assesses it.

    Attributes:
        max_wind_kt: 1-minute maximum sustained wind, knots.
        population_34kt: People inside the 34 kt (39 mph) wind buffer.
        population_64kt: People inside the 64 kt (74 mph) wind buffer.
        vulnerability: The exposed country's class, ``low``, ``medium`` or
            ``high`` (:func:`tropical_cyclone_vulnerability`).
        country_population: The country's total population, for the "10% of
            the population" thresholds; ``None`` applies only the absolute ones.
        lead_time_hours: Hours ahead of the latest advisory; 0 for observed points.
    """

    max_wind_kt: float
    population_34kt: float
    population_64kt: float
    vulnerability: str = "low"
    country_population: float | None = None
    lead_time_hours: float = 0.0

    def __post_init__(self) -> None:
        """Validate the point and store its numbers as floats."""
        wind = _finite("max_wind_kt", self.max_wind_kt)
        if wind < 0.0:
            raise ValueError(f"max_wind_kt must be >= 0, got {wind}")
        numbers = {
            "max_wind_kt": wind,
            "population_34kt": _population("population_34kt", self.population_34kt),
            "population_64kt": _population("population_64kt", self.population_64kt),
            "lead_time_hours": _finite("lead_time_hours", self.lead_time_hours),
        }
        if self.vulnerability not in _TC_VULNERABILITY_CLASSES:
            raise ValueError(
                f"vulnerability must be one of {_TC_VULNERABILITY_CLASSES}, got {self.vulnerability!r}"
            )
        if self.country_population is not None:
            numbers["country_population"] = _population(
                "country_population", self.country_population
            )
        for name, value in numbers.items():
            object.__setattr__(self, name, value)  # frozen: set once, here


def tropical_cyclone_category(max_wind_kt: float) -> str:
    """Saffir-Simpson class of a 1-minute sustained wind in knots.

    Returns:
        ``tropical_depression`` (<= 33 kt), ``tropical_storm`` (34-63 kt), or
        ``category_1`` to ``category_5`` (64, 83, 96, 113 and 137 kt and up).
    """
    wind = _finite("max_wind_kt", max_wind_kt)
    if wind < 0.0:
        raise ValueError(f"max_wind_kt must be >= 0, got {wind}")
    for floor, category in _SSHS_KT:
        if wind >= floor:
            return category
    return "tropical_depression"


def tropical_cyclone_vulnerability(iso3: str) -> str:
    """GDACS tropical-cyclone vulnerability class of a country (:data:`TC_VULNERABILITY`)."""
    code = str(iso3).strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"expected an ISO 3166-1 alpha-3 code, got {iso3!r}")
    return TC_VULNERABILITY.get(code, "low")


def tropical_cyclone_point_level(point: CyclonePoint) -> str:
    """GDACS wind level of one track point, by the MHEWS look-up table.

    GDACS MHEWS user manual (JRC141661, 2025; version of 23/09/2026), section
    4.7, Table 1, where "(100k or 10% poptot)" reads as more than 100,000 people
    or more than 10% of the country's population in the 64 kt buffer:

    - Tropical depression: Green.
    - Tropical storm: Orange for High vulnerability with > 10M in the 34 kt
      buffer; else Green.
    - Category 1 and 2 (threshold 100k for Category 1, 50k for Category 2):
      Low Green; Medium Orange above the threshold; High Red above 1M, Orange
      above the threshold or with > 10M in the 34 kt buffer.
    - Category 3: Low Orange above 500k; Medium/High Red above 100k (or 10%),
      Orange above 20k.
    - Category 4 and 5: Low Red above 1M, Orange above 100k (or 10%);
      Medium/High Red above 100k (or 10%), Orange above 20k.

    Anything else is Green, and a Red more than 72 hours ahead becomes Orange.
    """
    category = tropical_cyclone_category(point.max_wind_kt)
    p34, p64 = float(point.population_34kt), float(point.population_64kt)
    vulnerable = point.vulnerability in ("medium", "high")

    def exceeds(threshold: float) -> bool:
        share = point.country_population
        return p64 > threshold or (share is not None and share > 0 and p64 > 0.1 * share)

    level = "green"
    if category == "tropical_storm":
        if point.vulnerability == "high" and p34 > 10e6:
            level = "orange"
    elif category in ("category_1", "category_2"):
        threshold = 100e3 if category == "category_1" else 50e3
        if point.vulnerability == "medium" and exceeds(threshold):
            level = "orange"
        elif point.vulnerability == "high":
            if p64 > 1e6:
                level = "red"
            elif exceeds(threshold) or p34 > 10e6:
                level = "orange"
    elif category == "category_3":
        if not vulnerable:
            level = "orange" if p64 > 500e3 else "green"
        elif exceeds(100e3):
            level = "red"
        elif p64 > 20e3:
            level = "orange"
    elif category in ("category_4", "category_5"):
        if not vulnerable:
            if p64 > 1e6:
                level = "red"
            elif exceeds(100e3):
                level = "orange"
        elif exceeds(100e3):
            level = "red"
        elif p64 > 20e3:
            level = "orange"
    if level == "red" and point.lead_time_hours > _TC_RED_HORIZON_HOURS:
        level = "orange"
    return level


def tropical_cyclone_level(points: Iterable[CyclonePoint]) -> str:
    """GDACS level of a storm: the highest level over its track points."""
    levels = [tropical_cyclone_point_level(point) for point in points]
    if not levels:
        raise ValueError("a tropical cyclone level needs at least one track point")
    return max(levels, key=_LEVEL_ORDER.index)


def tropical_cyclone_score(level: str) -> float:
    """GDACS tropical-cyclone score of a wind level: 0.5, 1.5 or 2.5.

    GDACS's API reports these as 1, 2 and 3.
    """
    if level not in _TC_SCORE:
        raise ValueError(f"level must be green, orange or red, got {level!r}")
    return _TC_SCORE[level]


def storm_surge_level(max_surge_m: float) -> str | None:
    """GDACS storm-surge level: Red >= 3 m, Orange >= 1 m, Green above 0, else ``None``.

    GDACS reports it beside the wind level; it does not enter the score.
    """
    surge = _finite("max_surge_m", max_surge_m)
    if surge < 0.0:
        raise ValueError(f"max_surge_m must be >= 0, got {surge}")
    if surge >= 3.0:
        return "red"
    if surge >= 1.0:
        return "orange"
    return "green" if surge > 0.0 else None
