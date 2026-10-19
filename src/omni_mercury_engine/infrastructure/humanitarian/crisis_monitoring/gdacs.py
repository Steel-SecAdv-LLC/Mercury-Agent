# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""GDACS alert scores for earthquakes and alert levels for floods.

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
  - Final score = model score x the coping-capacity factor, INFORM Lack of
    Coping Capacity normalised to 0.5 (Switzerland) - 1.5 (South Sudan).
  - Level: Red >= 2, Orange >= 1, Green below 1.
- Floods, https://www.gdacs.org/Knowledge/models_fl.aspx: Red for more than
  1,000 dead or 800,000 displaced, Orange for more than 100 dead or 80,000
  displaced, Green otherwise.

Country terms come from GDACS's published table
(gdacs.org/Documents/2017_GDACS_EQ_algorithm_coefficients.xlsx, © European
Union, CC BY 4.0): the 38 countries with their own calibration are in
:data:`COUNTRY_COEFFICIENTS`; every other country uses the global model, as
GDACS does. The table's INFORM LCC column is from 2017 and is not included;
callers supply a current LCC. The formulas are implemented as published; they
have not been checked against GDACS's own event scores. GDACS calls its scores
empirical and "intended to give a fast and rough warning".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "COUNTRY_COEFFICIENTS",
    "GLOBAL_COEFFICIENTS",
    "EarthquakeCoefficients",
    "alert_level",
    "apply_coping_capacity",
    "coping_capacity_factor",
    "earthquake_coefficients",
    "earthquake_parameter_score",
    "earthquake_shakemap_score",
    "flood_alert_level",
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
        """Reject a term that is not a finite number."""
        for name, value in vars(self).items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            _finite(name, value)


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
    scaled = 10.0 * p9 + p8 + 0.1 * p7
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
    """
    score, factor = _finite("score", score), _finite("factor", factor)
    if not 0.5 <= factor <= 1.5:
        raise ValueError(f"the coping-capacity factor must be in [0.5, 1.5], got {factor}")
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
    """GDACS flood level from reported deaths and displaced people."""
    dead = _population("deaths", deaths)
    moved = _population("displaced", displaced)
    if dead > 1000 or moved > 800_000:
        return "red"
    if dead > 100 or moved > 80_000:
        return "orange"
    return "green"
