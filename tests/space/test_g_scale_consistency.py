# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every G-scale site gives the SWPC level for Kp in thirds.

SWPC files each third under its integer: G1 "Kp = 5" (5-, 5o, 5+), G4
"Kp = 8, 9-", G5 "Kp = 9o". Before, five sites used integer ladders, so 5-
read no storm (and the DONKI parser read "5-" as 5.0, "5o" as nothing), 9-
read G5, and the Schumann labeller rounded 9- to G5. An SWPC outage also read
as a quiet feed (an empty list).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

from omni_mercury_engine.data_sources.base import AlertLevel
from omni_mercury_engine.data_sources.space_weather import NASADONKISource, NOAASWPCSource
from omni_mercury_engine.evaluation import hazard_metrics as hm
from omni_mercury_engine.ml.hazard_training import schumann_harmonics, solar_storm
from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

#: Kp in thirds -> SWPC G level.
SWPC_LEVELS = {"4+": 0, "5-": 1, "5o": 1, "5+": 1, "6-": 2, "7-": 3, "8-": 4, "9-": 4, "9o": 5}


@pytest.mark.parametrize(("text", "level"), SWPC_LEVELS.items())
def test_every_site_agrees(text: str, level: int) -> None:
    kp = hm.parse_kp(text)
    assert hm.noaa_g_scale(kp) == level
    assert NASADONKISource()._parse_kp_index(text) == AlertLevel.from_noaa_g_scale(level)
    assert NOAASWPCSource()._parse_kp_alert_level(kp) == AlertLevel.from_noaa_g_scale(level)
    detector = SolarStormDetector.__new__(SolarStormDetector)
    assert detector._classify_geostorm(kp) == hm.G_SCALE_NAMES[level]
    assert solar_storm._g_bucket(kp) == hm.G_SCALE_NAMES[level]
    assert schumann_harmonics._noaa_g_scale(kp) == (f"G{level}" if level else "sub-G1")


def test_regressed_kp_is_held_to_the_scale() -> None:
    detector = SolarStormDetector.__new__(SolarStormDetector)
    assert detector._classify_geostorm(9.3) == "extreme"
    assert detector._classify_geostorm(-0.2) == "none"
    with pytest.raises(ValueError):
        detector._classify_geostorm(float("nan"))


def test_swpc_fetch_failure_is_not_a_quiet_feed() -> None:
    import asyncio

    from omni_mercury_engine.data_sources.base import DataSourceError

    source = NOAASWPCSource()

    async def _down(endpoint: str, params: dict[str, Any] | None = None) -> None:
        raise DataSourceError("HTTP 503", source_id="noaa_swpc")

    source._http_get = _down  # type: ignore[method-assign,assignment]
    with pytest.raises(DataSourceError):
        asyncio.run(source._fetch_impl())
