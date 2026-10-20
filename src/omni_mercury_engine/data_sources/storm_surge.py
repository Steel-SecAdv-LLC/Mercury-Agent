# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""NOAA NHC Peak Storm Surge forecasts as a numeric storm-surge data source.

The National Hurricane Center's Peak Storm Surge product states, per coastal
segment and per inland water body, the peak height the water could reach above
normally dry ground ("Tampa Bay...10-15 ft"). NHC publishes it for the United
States Gulf and Atlantic coasts, Puerto Rico and the U.S. Virgin Islands with
each tropical-cyclone public advisory, refreshed every 3-6 hours while surge
is expected within about 48 hours; outside an active advisory the service
serves no features, which this module reports as an empty result, never as a
zero surge measurement.

Live feed: the public ArcGIS service
``https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/
NHC_PeakStormSurge/MapServer`` (no key required), layer 1 (coastal line
segments) and layer 2 (inundation-area polygons). Archived advisories for past
storms are plain KML under ``https://www.nhc.noaa.gov/gis/peakSurge/`` (for
example ``AL142024_PeakStormSurge_017adv.kml``, Hurricane Milton), parsed by
:func:`parse_peak_surge_kml`.

The parsed heights are in feet as NHC states them; metres use the exact
international foot (0.3048 m), and :meth:`PeakSurgeForecast.gdacs_level`
grades the upper bound on the GDACS storm-surge scale (Red >= 3 m,
Orange >= 1 m) from
:func:`omni_mercury_engine.infrastructure.humanitarian.crisis_monitoring.gdacs.storm_surge_level`.
GDACS's own per-event surge heights remain unpublished (its API serves surge
only as map images); this module supplies the NHC-measured values instead,
with NHC's coverage, not GDACS's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from defusedxml.ElementTree import fromstring as _safe_fromstring

from omni_mercury_engine.data_sources.base import (
    AlertLevel,
    DataPoint,
    DataSourceBase,
    DataSourceError,
    DataSourceType,
)

__all__ = [
    "FEET_TO_METERS",
    "NHCPeakStormSurgeSource",
    "PeakSurgeForecast",
    "parse_peak_surge_kml",
    "parse_peak_surge_name",
]

#: Exact international foot.
FEET_TO_METERS = 0.3048

_KML_NS = "{http://www.opengis.net/kml/2.2}"

# "<area>...<lo>-<hi> ft" as NHC names every forecast feature. Bare "4-7 ft"
# names (legend and label placemarks) carry no area and are not forecasts.
_NAME_RE = re.compile(r"^(?P<area>.+?)\.\.\.(?P<lo>\d+(?:\.\d+)?)-(?P<hi>\d+(?:\.\d+)?)\s*ft\s*$")

# KML folder name / MapServer layer -> the kind of geography the value covers.
_KML_FOLDER_KINDS = {"Polygons": "inundation_area", "Lines": "coastal_segment"}
_LAYER_KINDS = {2: "inundation_area", 1: "coastal_segment"}

_GDACS_ALERTS = {
    "red": AlertLevel.SEVERE,
    "orange": AlertLevel.STRONG,
    "green": AlertLevel.MINOR,
    None: AlertLevel.NONE,
}


@dataclass(frozen=True)
class PeakSurgeForecast:
    """One NHC peak-surge statement: an area and its forecast height range.

    Attributes:
        area: The coastal segment or water body as NHC names it.
        kind: ``"inundation_area"`` (polygon) or ``"coastal_segment"`` (line).
        min_surge_ft: Lower bound of the stated range, feet above ground.
        max_surge_ft: Upper bound of the stated range, feet above ground.
    """

    area: str
    kind: str
    min_surge_ft: float
    max_surge_ft: float

    @property
    def min_surge_m(self) -> float:
        """Lower bound in metres (exact international foot)."""
        return self.min_surge_ft * FEET_TO_METERS

    @property
    def max_surge_m(self) -> float:
        """Upper bound in metres (exact international foot)."""
        return self.max_surge_ft * FEET_TO_METERS

    def gdacs_level(self) -> str | None:
        """GDACS storm-surge level of the range's upper bound."""
        from omni_mercury_engine.infrastructure.humanitarian.crisis_monitoring.gdacs import (
            storm_surge_level,
        )

        return storm_surge_level(self.max_surge_m)


def parse_peak_surge_name(name: str, kind: str) -> PeakSurgeForecast | None:
    """Parse one feature name; ``None`` for label names that carry no area.

    Raises:
        ValueError: If the name states a range whose bounds are inverted, or
            ``kind`` is not a known geography kind.
    """
    if kind not in _KML_FOLDER_KINDS.values():
        raise ValueError(f"unknown peak-surge kind {kind!r}")
    match = _NAME_RE.match(name.strip())
    if match is None:
        return None
    low, high = float(match["lo"]), float(match["hi"])
    if low > high:
        raise ValueError(f"inverted surge range in {name!r}")
    return PeakSurgeForecast(
        area=match["area"].strip(), kind=kind, min_surge_ft=low, max_surge_ft=high
    )


def parse_peak_surge_kml(document: bytes | str) -> list[PeakSurgeForecast]:
    """Peak-surge forecasts from a Peak Storm Surge KML (live or archived).

    Reads the product's ``Polygons`` and ``Lines`` folders; label, legend and
    breakpoint folders are cartography and carry no forecast. A placemark in a
    data folder whose name does not state an area and a range is refused.

    Raises:
        ValueError: On XML that is not this product, or an unreadable name.
    """
    if isinstance(document, bytes):
        document = document.decode("utf-8", errors="strict")
    root = _safe_fromstring(document)
    doc = root.find(f"{_KML_NS}Document")
    if doc is None:
        raise ValueError("not a Peak Storm Surge KML: no Document element")
    forecasts: list[PeakSurgeForecast] = []
    for folder in doc.iter(f"{_KML_NS}Folder"):
        folder_name = folder.findtext(f"{_KML_NS}name", "")
        kind = _KML_FOLDER_KINDS.get(folder_name)
        if kind is None:
            continue
        for placemark in folder.findall(f"{_KML_NS}Placemark"):
            name = placemark.findtext(f"{_KML_NS}name", "")
            forecast = parse_peak_surge_name(name, kind)
            if forecast is None:
                raise ValueError(f"unreadable peak-surge name {name!r} in folder {folder_name!r}")
            forecasts.append(forecast)
    if not forecasts and not any(
        folder.findtext(f"{_KML_NS}name", "") in _KML_FOLDER_KINDS
        for folder in doc.iter(f"{_KML_NS}Folder")
    ):
        raise ValueError("not a Peak Storm Surge KML: no Polygons or Lines folder")
    return forecasts


class NHCPeakStormSurgeSource(DataSourceBase):
    """NOAA NHC Peak Storm Surge forecasts from the public ArcGIS service.

    One :class:`DataPoint` per stated area. An empty result means NHC has no
    active peak-surge forecast (no U.S. landfalling storm within the product
    window), which is the feed's real state, not a zero surge height.

    Example:
        >>> source = NHCPeakStormSurgeSource()
        >>> result = await source.fetch()  # [] outside active advisories
    """

    DEFAULT_BASE_URL = (
        "https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/"
        "NHC_PeakStormSurge/MapServer"
    )

    @property
    def source_id(self) -> str:
        """Stable identifier of this source."""
        return "nhc_peak_storm_surge"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """The source types this source produces."""
        return [DataSourceType.STORM_SURGE]

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        fetched_at = datetime.now(UTC)
        points: list[DataPoint] = []
        for layer, kind in sorted(_LAYER_KINDS.items(), reverse=True):
            response = await self._http_get(
                f"/{layer}/query",
                params={
                    "where": "1=1",
                    "outFields": "name,folderpath,idp_filedate",
                    "returnGeometry": "false",
                    "f": "json",
                },
            )
            points.extend(self._points_from_query(response.json(), kind, fetched_at))
        return points

    def _points_from_query(
        self, payload: dict[str, Any], kind: str, fetched_at: datetime
    ) -> list[DataPoint]:
        """Build the data points of one layer's ArcGIS query payload.

        Raises:
            DataSourceError: If the service answered with an error body.
        """
        if "error" in payload:
            raise DataSourceError(
                f"{self.source_id}: service error {payload['error']!r}",
                source_id=self.source_id,
            )
        if "features" not in payload:
            raise DataSourceError(
                f"{self.source_id}: no features key in query payload",
                source_id=self.source_id,
            )
        points: list[DataPoint] = []
        for feature in payload["features"]:
            attributes = feature.get("attributes") or {}
            name = attributes.get("name") or ""
            forecast = parse_peak_surge_name(name, kind)
            if forecast is None:
                # Label/legend text has no area; the data layers should not
                # carry it, so refuse rather than silently drop a forecast.
                raise DataSourceError(
                    f"{self.source_id}: unreadable feature name {name!r}",
                    source_id=self.source_id,
                )
            filedate_ms = attributes.get("idp_filedate")
            timestamp = (
                datetime.fromtimestamp(filedate_ms / 1000.0, tz=UTC)
                if isinstance(filedate_ms, (int, float))
                else fetched_at
            )
            level = forecast.gdacs_level()
            points.append(
                DataPoint(
                    source_id=self.source_id,
                    source_type=DataSourceType.STORM_SURGE,
                    event_id=f"peak-surge-{kind}-{forecast.area}-{timestamp.isoformat()}",
                    timestamp=timestamp,
                    data={
                        "area": forecast.area,
                        "kind": forecast.kind,
                        "min_surge_ft": forecast.min_surge_ft,
                        "max_surge_ft": forecast.max_surge_ft,
                        "min_surge_m": forecast.min_surge_m,
                        "max_surge_m": forecast.max_surge_m,
                        "gdacs_level": level,
                    },
                    alert_level=_GDACS_ALERTS[level],
                    metadata={"service": self.base_url, "product": "Peak Storm Surge"},
                )
            )
        return points
