# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PM2.5 Air Quality Index, as EPA defines it.

The index and its category follow 40 CFR Part 58, Appendix G ("Uniform Air
Quality Index (AQI) and Daily Reporting"), Table 2 and section 3.4, as revised
at 89 FR 16403 (Mar. 6, 2024): the 24-hour PM2.5 concentration is truncated
to 0.1 ug/m3, the index is interpolated linearly between the breakpoints that
contain it (equation 1) and rounded to the nearest integer, and above the
concentration of AQI 500 (325.4 ug/m3) the hazardous segment's equation is
extended (Table 2, footnote 4).

Checked against EPA's own AQI column in its AirData daily PM2.5 file (see
``tests/detectors/test_air_quality_recorded.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction
from typing import Any

from omni_mercury_engine.detectors._hazard_input import finite_scalar
from omni_mercury_engine.detectors.provenance import DecisionSource

__all__ = [
    "AQI_CATEGORIES",
    "PM25_BREAKPOINTS",
    "AirQualityAssessment",
    "AirQualityDetector",
    "pm25_aqi",
]

#: Table 2, PM2.5 24-hour column: (C_lo, C_hi, I_lo, I_hi, category), in
#: ug/m3.  The hazardous row is open-ended ("225.5+"); 325.4 ug/m3 is the
#: concentration footnote 4 assigns to AQI 500.
PM25_BREAKPOINTS: tuple[tuple[str, str, int, int, str], ...] = (
    ("0.0", "9.0", 0, 50, "good"),
    ("9.1", "35.4", 51, 100, "moderate"),
    ("35.5", "55.4", 101, 150, "unhealthy_for_sensitive_groups"),
    ("55.5", "125.4", 151, 200, "unhealthy"),
    ("125.5", "225.4", 201, 300, "very_unhealthy"),
    ("225.5", "325.4", 301, 500, "hazardous"),
)
#: The six categories, lowest first.
AQI_CATEGORIES: tuple[str, ...] = tuple(row[4] for row in PM25_BREAKPOINTS)
#: First AQI of "Unhealthy for Sensitive Groups", the first category with a
#: health message beyond "unusually sensitive people" (Table 2).
SENSITIVE_GROUPS_AQI = 101


def _truncated(concentration: float) -> Fraction:
    """``concentration`` truncated to 0.1 (section 3.4.a.i), exactly.

    Decimal of the float's shortest repr, so 2.3 truncates to 2.3 (a binary
    ``floor(2.3 * 10) / 10`` gives 2.2).
    """
    return Fraction(Decimal(repr(concentration)).quantize(Decimal("0.1"), rounding=ROUND_FLOOR))


def pm25_aqi(concentration_ug_m3: float) -> tuple[int, str]:
    """EPA AQI and category for one 24-hour PM2.5 concentration.

    Args:
        concentration_ug_m3: 24-hour mean PM2.5 in ug/m3 (local conditions).
            A negative mean (an instrument offset near zero) is indexed as 0,
            as EPA's AirData daily files report it; Table 2 starts at 0.0.

    Returns:
        ``(aqi, category)``; AQI may exceed 500 above 325.4 ug/m3.

    Raises:
        InvalidObservationError: If the concentration is not a finite number.
    """
    value = finite_scalar(concentration_ug_m3, "pm25_24h_ug_m3", unit="ug/m3")
    c = max(_truncated(value), Fraction(0))
    # The first row whose upper breakpoint holds c; the open-ended hazardous
    # row (the last) takes everything above, extended past AQI 500.
    row = next(
        (r for r in PM25_BREAKPOINTS[:-1] if c <= Fraction(r[1])),
        PM25_BREAKPOINTS[-1],
    )
    lo_text, hi_text, i_lo, i_hi, category = row
    lo, hi = Fraction(lo_text), Fraction(hi_text)
    index = Fraction(i_hi - i_lo) / (hi - lo) * (c - lo) + i_lo
    return math.floor(index + Fraction(1, 2)), category


@dataclass(frozen=True)
class AirQualityAssessment:
    """One graded PM2.5 day."""

    pm25_24h_ug_m3: float
    aqi: int
    category: str
    #: AQI >= 101: "Unhealthy for Sensitive Groups" or worse.
    unhealthy_for_sensitive_groups: bool


class AirQualityDetector:
    """EPA PM2.5 Air Quality Index from a 24-hour mean concentration."""

    decision_source = DecisionSource(
        kind="published",
        description=(
            "EPA AQI for 24-hour PM2.5: truncate to 0.1 ug/m3, interpolate "
            "between the Table 2 breakpoints (0.0-9.0 good, 9.1-35.4 moderate, "
            "35.5-55.4 unhealthy for sensitive groups, 55.5-125.4 unhealthy, "
            "125.5-225.4 very unhealthy, 225.5+ hazardous), round to the "
            "nearest integer; the flag is AQI >= 101. A negative mean is "
            "indexed as 0, as EPA AirData reports it (not CFR text)."
        ),
        citation=(
            "40 CFR Part 58, Appendix G, Table 2 and section 3.4, as revised "
            "at 89 FR 16403 (Mar. 6, 2024)"
        ),
    )

    def assess(self, pm25_24h_ug_m3: Any) -> AirQualityAssessment:
        """Grade one 24-hour PM2.5 mean.

        Raises:
            InvalidObservationError: If the concentration is not a finite number.
        """
        aqi, category = pm25_aqi(pm25_24h_ug_m3)
        return AirQualityAssessment(
            pm25_24h_ug_m3=float(pm25_24h_ug_m3),
            aqi=aqi,
            category=category,
            unhealthy_for_sensitive_groups=aqi >= SENSITIVE_GROUPS_AQI,
        )
