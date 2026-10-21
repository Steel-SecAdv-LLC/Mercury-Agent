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

Observed history comes from SURGEDAT (Louisiana State University, Southern
Regional Climate Center; Needham & Keim 2012, Int. J. Climatol. 32(14),
2108-2123, doi:10.1002/joc.2425): the global peak-surge database and the
Gulf of Mexico database, read by :func:`parse_surgedat_csv`. Both are static
archives (last modified 2014-12-05 and 2013-01-22) in classic Mac OS form:
carriage-return line ends and Mac Roman text (the global file's only
non-ASCII bytes, 0xD0 and 0xD3, frame eleven storm names: an en dash and a
closing quote in Mac Roman, "Ð"/"Ó" in Latin-1). The database states most heights in both feet and metres; a
pair whose rounding intervals do not overlap is a conflict the database
holds against itself, kept as stated and never resolved by choosing a side.
"""

from __future__ import annotations

import csv
import io
import math
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
    "SURGEDAT_GLOBAL_URL",
    "SURGEDAT_GULF_URL",
    "NHCPeakStormSurgeSource",
    "PeakSurgeForecast",
    "SurgeHeight",
    "SurgeObservation",
    "parse_peak_surge_kml",
    "parse_peak_surge_name",
    "parse_surgedat_csv",
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


#: SURGEDAT global peak-surge database (745 observations, 1737-2014).
SURGEDAT_GLOBAL_URL = "https://surge.climate.lsu.edu/files/globalpeaksurgedb.csv"

#: SURGEDAT U.S. Gulf of Mexico peak-surge database (240 observations, 1880-2011).
SURGEDAT_GULF_URL = "https://surge.climate.lsu.edu/files/gompeaksurgedb.csv"

# Column names of the two published layouts: database -> (name, year, dates,
# location, state, lat, lon, confidence, (ft, m) per height kind).
_SURGEDAT_LAYOUTS: dict[str, dict[str, Any]] = {
    "global": {
        "header": (
            "Year",
            "Storm Name",
            "Storm Dates",
            "Time",
            "Reg",
            "Sub Reg",
            "Country",
            "State",
            "Location",
            "Lat",
            "Lon",
            "Surge_m",
            "Surge_ft",
            "Storm_Tide_m",
            "Storm_Tide_ft",
            "Storm_Tide_Waves_m",
            "Storm_Tide_Waves_ft",
            "Datum",
            "Type of Obs",
            "Tropical",
            "Confidence",
            "Surge ID",
            "Storm ID",
        ),
        "name": "Storm Name",
        "dates": "Storm Dates",
        "heights": {
            "surge": ("Surge_ft", "Surge_m"),
            "storm_tide": ("Storm_Tide_ft", "Storm_Tide_m"),
            "storm_tide_waves": ("Storm_Tide_Waves_ft", "Storm_Tide_Waves_m"),
        },
    },
    "gulf": {
        "header": (
            "Name",
            "Year",
            "Dates",
            "Surge_FT",
            "Surge_M",
            "Storm_Tide_FT",
            "Storm_Tide_M",
            "Location",
            "State",
            "Lat",
            "Lon",
            "Confidence",
        ),
        "name": "Name",
        "dates": "Dates",
        "heights": {
            "surge": ("Surge_FT", "Surge_M"),
            "storm_tide": ("Storm_Tide_FT", "Storm_Tide_M"),
        },
    },
}

# A stated height: a number or an "a-b" range, non-negative decimals.
_HEIGHT_RE = re.compile(r"^(?P<lo>\d+(?:\.(?P<lo_dp>\d+))?)(?:-(?P<hi>\d+(?:\.(?P<hi_dp>\d+))?))?$")


def _stated(text: str, unit_m: float) -> tuple[float, float, float, float] | None:
    """A stated cell as (low, high, low half-width, high half-width), metres."""
    text = text.strip()
    if not text:
        return None
    match = _HEIGHT_RE.match(text)
    if match is None:
        raise ValueError(f"unreadable SURGEDAT height {text!r}")
    low = float(match["lo"]) * unit_m
    high = float(match["hi"]) * unit_m if match["hi"] else low
    if low > high:
        raise ValueError(f"inverted SURGEDAT height range {text!r}")
    low_hw = 0.5 * 10.0 ** -len(match["lo_dp"] or "") * unit_m
    high_hw = 0.5 * 10.0 ** -len(match["hi_dp"] or "") * unit_m if match["hi"] else low_hw
    return low, high, low_hw, high_hw


@dataclass(frozen=True)
class SurgeHeight:
    """One SURGEDAT height as the database states it, in feet, metres or both.

    Attributes:
        stated_ft: The feet cell verbatim (``""`` when blank).
        stated_m: The metres cell verbatim (``""`` when blank).
        low_m: The lowest stated reading, metres (feet x 0.3048).
        high_m: The highest stated reading, metres.
        metres: The height as one value: the more precise reading of a
            consistent pair (the metres cell on a tie), the single stated
            reading otherwise; ``None`` for a stated range or a conflict.
        conflict: Both cells are stated and their rounding intervals do not
            overlap (``1.3`` ft is 0.38-0.41 m, not ``0.7`` m).
    """

    stated_ft: str
    stated_m: str
    low_m: float
    high_m: float
    metres: float | None
    conflict: bool

    def gdacs_level(self) -> str | None:
        """GDACS storm-surge level, when every stated reading agrees on it.

        ``None`` when the readings straddle a band edge (a conflict across
        1 m or 3 m, or a range that crosses one): the database does not
        settle the level, so none is assigned.
        """
        from omni_mercury_engine.infrastructure.humanitarian.crisis_monitoring.gdacs import (
            storm_surge_level,
        )

        low, high = storm_surge_level(self.low_m), storm_surge_level(self.high_m)
        return low if low == high else None


def _surge_height(ft_text: str, m_text: str) -> SurgeHeight | None:
    """Combine a feet cell and a metres cell; ``None`` when both are blank."""
    feet = _stated(ft_text, FEET_TO_METERS)
    metres = _stated(m_text, 1.0)
    readings = [r for r in (metres, feet) if r is not None]
    if not readings:
        return None
    conflict = False
    if feet is not None and metres is not None:
        conflict = (
            feet[0] - feet[2] > metres[1] + metres[3] + 1e-9
            or metres[0] - metres[2] > feet[1] + feet[3] + 1e-9
        )
    point = [r for r in readings if r[0] == r[1]]
    value: float | None = None
    if not conflict and len(point) == len(readings):
        value = min(point, key=lambda r: r[2])[0]  # metres first, so a tie keeps it
    return SurgeHeight(
        stated_ft=ft_text.strip(),
        stated_m=m_text.strip(),
        low_m=min(r[0] for r in readings),
        high_m=max(r[1] for r in readings),
        metres=value,
        conflict=conflict,
    )


@dataclass(frozen=True)
class SurgeObservation:
    """One SURGEDAT peak-water observation.

    Attributes:
        database: ``"global"`` or ``"gulf"``.
        year: Year of the event.
        storm: Storm name as stated (``""`` when the database names none).
        dates: Storm dates as stated (free text).
        location: Observation site as stated.
        country: ISO 3166 alpha-2 code as stated; ``"US"`` for the Gulf
            database, which covers the U.S. Gulf Coast only.
        state: State or province code as stated.
        latitude: Decimal degrees, or ``None`` when the site is unlocated.
        longitude: Decimal degrees, or ``None`` with ``latitude``.
        surge: Storm surge (water above the predicted tide).
        storm_tide: Storm tide (surge plus astronomical tide).
        storm_tide_waves: Storm tide with wave run-up (global database only).
        tropical: Tropical-cyclone surge; ``None`` where not stated (Gulf).
        confidence: SURGEDAT confidence class, or ``None`` when unstated.
    """

    database: str
    year: int
    storm: str
    dates: str
    location: str
    country: str
    state: str
    latitude: float | None
    longitude: float | None
    surge: SurgeHeight | None
    storm_tide: SurgeHeight | None
    storm_tide_waves: SurgeHeight | None
    tropical: bool | None
    confidence: int | None

    def gdacs_level(self) -> str | None:
        """GDACS level of the storm surge (the quantity GDACS grades).

        Storm tide and wave run-up include the tide and the waves, which
        GDACS's surge thresholds do not, so they are not graded.
        """
        return self.surge.gdacs_level() if self.surge is not None else None


def _coordinate(text: str, limit: float, name: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    value = float(text)
    if not math.isfinite(value) or abs(value) > limit:
        raise ValueError(f"SURGEDAT {name} {text!r} is out of range")
    return value


def parse_surgedat_csv(document: bytes, encoding: str = "mac_roman") -> list[SurgeObservation]:
    """Every observation of a SURGEDAT CSV (global or Gulf layout).

    Args:
        document: The file's bytes as served.
        encoding: Text encoding; the published files are Mac Roman (the
            Gulf file is plain ASCII, which Mac Roman decodes identically).

    Raises:
        ValueError: On a header that is neither published layout, or a
            cell this function cannot read (a height, coordinate, year,
            confidence or tropical flag).
    """
    text = document.decode(encoding, errors="strict")
    # Classic Mac OS line ends; quoted fields never contain a line end here.
    reader = csv.reader(io.StringIO(text.replace("\r\n", "\n").replace("\r", "\n")))
    rows = [row for row in reader if row]
    if not rows:
        raise ValueError("empty SURGEDAT document")
    header = tuple(cell.strip() for cell in rows[0])
    database = next((k for k, v in _SURGEDAT_LAYOUTS.items() if v["header"] == header), None)
    if database is None:
        raise ValueError(f"not a SURGEDAT layout: header {header!r}")
    layout = _SURGEDAT_LAYOUTS[database]
    observations: list[SurgeObservation] = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f"SURGEDAT row {number} has {len(row)} cells, not {len(header)}")
        cell = dict(zip(header, (value.strip() for value in row)))
        latitude = _coordinate(cell["Lat"], 90.0, "latitude")
        longitude = _coordinate(cell["Lon"], 180.0, "longitude")
        if (latitude is None) != (longitude is None):
            raise ValueError(f"SURGEDAT row {number} states only one coordinate")
        heights = {
            kind: _surge_height(cell[ft], cell[m]) for kind, (ft, m) in layout["heights"].items()
        }
        tropical: bool | None = None
        if database == "global":
            flags = {"Y": True, "N": False}
            if cell["Tropical"] not in flags:
                raise ValueError(f"SURGEDAT row {number}: tropical flag {cell['Tropical']!r}")
            tropical = flags[cell["Tropical"]]
        observations.append(
            SurgeObservation(
                database=database,
                year=int(cell["Year"]),
                storm=cell[layout["name"]],
                dates=cell[layout["dates"]],
                location=cell["Location"],
                country=cell["Country"] if database == "global" else "US",
                state=cell["State"],
                latitude=latitude,
                longitude=longitude,
                surge=heights["surge"],
                storm_tide=heights["storm_tide"],
                storm_tide_waves=heights.get("storm_tide_waves"),
                tropical=tropical,
                confidence=int(cell["Confidence"]) if cell["Confidence"] else None,
            )
        )
    return observations


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
