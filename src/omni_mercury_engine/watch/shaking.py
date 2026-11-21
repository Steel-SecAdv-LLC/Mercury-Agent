# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Who an earthquake concerns: where USGS says it was felt.

A subscriber was told of an earthquake when its epicentre lay within their
radius, 200 km unless they changed it, whatever its size. That is wrong both
ways. The magnitude 6.6 earthquake off Tual, Indonesia on 2026-11-16
(``us7000tymu``) was reported felt in Darwin, 708 km away, and in Nhulunbuy
at 886 km; and a magnitude 4.5 at 190 km is felt by no one there.

USGS publishes, for each significant earthquake, two statements of where it
was felt, and this reads both:

* **ShakeMap**: its estimate of shaking intensity (Modified Mercalli) on a
  grid around the earthquake, from the recordings and its ground-motion
  models. A place inside the grid has an estimated intensity.
* **Did You Feel It?**: what people reported, gathered into 10 km cells,
  each with an intensity and a number of reports. These reach as far as
  anyone felt it and said so.

A place is concerned when USGS estimates shaking there that people notice
(intensity III, "weak", and above) or when people in its cell reported
feeling it. Where USGS has said nothing about a place (outside the mapped
grid, with no report there) the subscriber's own radius still decides, as
it does for every earthquake until USGS has published either product.
Mercury estimates nothing here: both statements are USGS's.

The grid is CoverageJSON, values in rows from south to north. That order was
checked on three earthquakes against USGS's own contour lines for the same
maps: for ``us7000tuku`` the mean difference at the contour points is 0.07
intensity units in that order and 0.22 in the other.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FELT_MMI",
    "Felt",
    "felt_at",
    "intensity_at",
    "intensity_words",
    "read_reach",
    "reported_at",
]

logger = logging.getLogger(__name__)

_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
_PRODUCT_HOST = "https://earthquake.usgs.gov/"
_EVENT_ID = re.compile(r"[a-z0-9]{4,24}")
_GRID = "download/coverage_mmi_low_res.covjson"
_CELLS = "dyfi_geo_10km.geojson"

#: The least estimated intensity at which a place is told. On the Modified
#: Mercalli scale as USGS abbreviates it, III is "felt quite noticeably by
#: persons indoors"; below it few people notice. 2.5 rounds to III.
FELT_MMI = 2.5
#: The least reported intensity of a "Did You Feel It?" cell that counts as felt.
FELT_CDI = 2.0

#: USGS's ShakeMap legend: intensity to perceived shaking.
_SHAKING = (
    (9.5, "X+", "extreme"),
    (8.5, "IX", "violent"),
    (7.5, "VIII", "severe"),
    (6.5, "VII", "very strong"),
    (5.5, "VI", "strong"),
    (4.5, "V", "moderate"),
    (3.5, "IV", "light"),
    (2.5, "III", "weak"),
    (1.5, "II", "weak"),
)


def intensity_words(mmi: float) -> str:
    """An intensity as USGS's legend says it: ``IV (light shaking)``."""
    for floor, numeral, word in _SHAKING:
        if mmi >= floor:
            return f"{numeral} ({word} shaking)"
    return "I (not felt)"


@dataclass(frozen=True)
class Felt:
    """What USGS says of an earthquake at one place.

    Attributes:
        estimated: ShakeMap's estimated intensity there, when the place is on its grid.
        reported: The intensity people in its 10 km cell reported, when any did.
        reports: How many reported.
    """

    estimated: float | None = None
    reported: float | None = None
    reports: int = 0

    @property
    def felt(self) -> bool:
        """Whether USGS says it was felt there."""
        return (self.estimated is not None and self.estimated >= FELT_MMI) or (
            self.reported is not None and self.reported >= FELT_CDI and self.reports > 0
        )

    def said(self, place: str) -> str:
        """One sentence for a message: what USGS says of ``place``."""
        if self.reported is not None and self.reports > 0 and self.reported >= FELT_CDI:
            people = "1 person" if self.reports == 1 else f"{self.reports:,} people"
            line = (
                f'{people} near "{place}" told USGS they felt it '
                f"(intensity {intensity_words(self.reported)})."
            )
            if self.estimated is not None:
                line += f" USGS estimates intensity {intensity_words(self.estimated)} there."
            return line
        if self.estimated is not None:
            return f'USGS estimates intensity {intensity_words(self.estimated)} at "{place}".'
        return ""


def _usable(url: Any) -> str:
    """A product URL, only when it is USGS's own."""
    text = str(url or "")
    return text if text.startswith(_PRODUCT_HOST) else ""


def read_reach(event_id: str) -> dict[str, Any]:
    """Read where USGS says an earthquake was felt.

    Returns:
        ``{"event": id, "grid": {...} | None, "cells": [...]}``. ``grid`` is
        ShakeMap's intensity grid (``x0``, ``x1``, ``nx``, ``y0``, ``y1``,
        ``ny`` and ``mmi``, rows south to north); ``cells`` are the "Did You
        Feel It?" cells with reports (``ring``, ``cdi``, ``nresp``). Either
        is absent until USGS publishes it.

    Raises:
        ValueError: If ``event_id`` is not a USGS event identifier.
    """
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    if not _EVENT_ID.fullmatch(event_id):
        raise ValueError(f"not a USGS event id: {event_id!r}")
    detail = SafeHTTPClient.get_json(
        _EVENT_URL, params={"format": "geojson", "eventid": event_id}, timeout=30.0
    )
    products = (detail.get("properties") or {}).get("products") or {}
    reach: dict[str, Any] = {"event": event_id, "grid": None, "cells": []}

    def content(product: str, name: str) -> str:
        listed = products.get(product) or []
        first = listed[0] if listed and isinstance(listed[0], dict) else {}
        return _usable(((first.get("contents") or {}).get(name) or {}).get("url"))

    grid_url = content("shakemap", _GRID)
    if grid_url:
        reach["grid"] = _grid(SafeHTTPClient.get_json(grid_url, timeout=30.0))
    cells_url = content("dyfi", _CELLS)
    if cells_url:
        reach["cells"] = _cells(SafeHTTPClient.get_json(cells_url, timeout=30.0))
    return reach


def _grid(coverage: dict[str, Any]) -> dict[str, Any] | None:
    """The intensity grid in a ShakeMap CoverageJSON, or ``None`` when it is not one."""
    try:
        axes = coverage["domain"]["axes"]
        values = coverage["ranges"]["MMI"]
        if list(values["axisNames"]) != ["y", "x"]:
            return None
        ny, nx = (int(n) for n in values["shape"])
        mmi = [None if v is None else round(float(v), 1) for v in values["values"]]
        grid = {
            "x0": float(axes["x"]["start"]), "x1": float(axes["x"]["stop"]), "nx": nx,
            "y0": float(axes["y"]["start"]), "y1": float(axes["y"]["stop"]), "ny": ny,
            "mmi": mmi,
        }  # fmt: skip
    except (KeyError, TypeError, ValueError):
        return None
    if nx < 2 or ny < 2 or len(mmi) != nx * ny or int(axes["x"]["num"]) != nx:
        return None
    return grid


def _cells(collection: dict[str, Any]) -> list[dict[str, Any]]:
    """The "Did You Feel It?" cells that have reports."""
    cells = []
    for feature in collection.get("features") or []:
        try:
            properties, geometry = feature["properties"], feature["geometry"]
            if geometry["type"] != "Polygon" or int(properties["nresp"]) < 1:
                continue
            cells.append(
                {
                    "ring": [[float(x), float(y)] for x, y in geometry["coordinates"][0]],
                    "cdi": float(properties["cdi"]),
                    "nresp": int(properties["nresp"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return cells


def intensity_at(reach: dict[str, Any] | None, latitude: float, longitude: float) -> float | None:
    """The intensity ShakeMap estimates at a point, or ``None`` when it is off the grid."""
    grid = (reach or {}).get("grid")
    if not grid:
        return None
    for lon in (longitude, longitude + 360.0, longitude - 360.0):  # a map across the date line
        if grid["x0"] <= lon <= grid["x1"] and grid["y0"] <= latitude <= grid["y1"]:
            column = round((lon - grid["x0"]) / (grid["x1"] - grid["x0"]) * (grid["nx"] - 1))
            row = round((latitude - grid["y0"]) / (grid["y1"] - grid["y0"]) * (grid["ny"] - 1))
            value = grid["mmi"][row * grid["nx"] + column]  # rows run south to north
            return None if value is None else float(value)
    return None


def _inside(ring: list[list[float]], x: float, y: float) -> bool:
    """Whether ``(x, y)`` is inside the polygon ``ring`` (ray casting)."""
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def reported_at(
    reach: dict[str, Any] | None, latitude: float, longitude: float
) -> tuple[float, int] | None:
    """The reported intensity and number of reports in the place's cell, when there are any."""
    for cell in (reach or {}).get("cells") or []:
        if _inside(cell["ring"], longitude, latitude):
            return float(cell["cdi"]), int(cell["nresp"])
    return None


def felt_at(reach: dict[str, Any] | None, latitude: float, longitude: float) -> Felt | None:
    """What USGS says of an earthquake at a place, or ``None`` when it has said nothing there."""
    estimated = intensity_at(reach, latitude, longitude)
    reported = reported_at(reach, latitude, longitude)
    if estimated is None and reported is None:
        return None
    return Felt(estimated, reported[0] if reported else None, reported[1] if reported else 0)
