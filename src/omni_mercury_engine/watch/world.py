# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feeds from outside the United States: national authorities' alerts, and GDACS.

The watch's warnings, river gauges and volcano levels are United States
sources. Outside it a subscriber was told of earthquakes, tsunami buoys and
geomagnetic storms, and of nothing their own weather service had warned of.

* **National authorities' alerts** (``world_warnings``). The World
  Meteorological Organization's Severe Weather Information Centre carries
  the Common Alerting Protocol alerts of the national services that publish
  them (fifty countries on 2026-11-17: most of Europe's MeteoAlarm members,
  China, Russia, Australia, Canada, Argentina, Indonesia and others; not
  Japan or India). Its map service answers "which alerts cover this point",
  the question the National Weather Service answers for a place in the
  United States, so a subscriber's place is matched the same way: by the
  issuing authority's own polygon. Each alert is then read from its CAP
  document, which is the authority's own text: who issued it, how it rates
  it, when it expires and what it says to do.
* **GDACS** (``gdacs``). The Global Disaster Alert and Coordination System
  (United Nations and European Commission) rates tropical cyclones, floods,
  volcanoes and wildfires Green, Orange or Red by their expected
  humanitarian impact. Orange and Red events that are current are findings,
  placed at the point GDACS gives.

What was read to build these, on 2026-11-17: the centre's alert list and map
service; point queries at Erfurt (the German weather service's wind
warning), Zurich (MeteoSwiss, "Strong gale") and Cocoa, Florida, where the
service returned the same Flood Warning and Flood Watch the National Weather
Service's own API returned; a MeteoSwiss CAP document; and GDACS's event
search.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from omni_mercury_engine.watch.feeds import Feed, Judgment, _clock, _count, _shorten, _utc
from omni_mercury_engine.watch.model import Finding, Outcome, Severity

if TYPE_CHECKING:
    from omni_mercury_engine.watch.store import WatchStore

logger = logging.getLogger(__name__)

__all__ = ["GdacsFeed", "WorldWarningsFeed", "parse_cap"]

_SWIC = "https://severeweather.wmo.int"
#: The centre's map service (GeoServer WFS) over every alert's polygons.
_SWIC_WFS = f"{_SWIC}/f/wfs"
#: Where it serves each alert's CAP document.
_SWIC_CAP = f"{_SWIC}/v2/cap-alerts/"
#: A CAP document's path as the map service gives it:
#: ``ch-meteoswiss-de/2026/11/17/05/51/55-b6da….xml``. It comes from a remote
#: service and goes into a URL, so it is used only when it has this shape.
_CAP_PATH = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*/(?:\d{2,4}/){5}[A-Za-z0-9._-]+\.xml")
#: The National Weather Service's alerts are relayed from the Service itself.
_FROM_NWS = "us-noaa-nws"
_GDACS_SEARCH = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
#: What some authorities write as an area's name when it has none but its polygon.
_NO_NAME = "polygonal event area"
#: How many of an alert's area names are said.
_AREAS_SAID = 8
#: Place lookups made at once. Each is a request to the centre's map service.
_LOOKUPS_AT_ONCE = 4


def _local(tag: str) -> str:
    """An XML tag without its namespace (CAP 1.1 and 1.2 differ only in that)."""
    return tag.rsplit("}", 1)[-1]


def parse_cap(document: str, prefer: str = "en") -> dict[str, Any]:
    """Read one Common Alerting Protocol alert.

    A CAP alert may say the same thing in several languages, one ``info``
    block each. The block in ``prefer`` is taken when there is one, else the
    first: an alert is never dropped for want of a translation.

    Returns:
        ``identifier``, ``status``, ``msg_type``, ``sent`` and, from the
        chosen block, ``language``, ``event``, ``headline``, ``severity``,
        ``urgency``, ``certainty``, ``expires``, ``sender`` (its
        ``senderName``, else the alert's ``sender``), ``instruction``,
        ``web`` and ``area`` (its ``areaDesc`` values joined).

    Raises:
        ValueError: If the document is not a CAP alert with an ``info`` block.
    """
    from defusedxml import ElementTree

    try:
        root = ElementTree.fromstring(document)
    except Exception as exc:
        raise ValueError(f"not XML: {exc}") from exc
    if _local(root.tag) != "alert":
        raise ValueError(f"not a CAP alert: root element is {_local(root.tag)!r}")

    def text(parent: Any, name: str) -> str:
        for child in parent:
            if _local(child.tag) == name:
                return " ".join((child.text or "").split())
        return ""

    blocks = [child for child in root if _local(child.tag) == "info"]
    if not blocks:
        raise ValueError("a CAP alert with no info block")
    info = next(
        (b for b in blocks if text(b, "language").lower().split("-")[0] == prefer), blocks[0]
    )
    areas = [text(child, "areaDesc") for child in info if _local(child.tag) == "area"]
    named = list(dict.fromkeys(area for area in areas if area and area.lower() != _NO_NAME))
    if len(named) > _AREAS_SAID:
        # One German wind warning names 1,205 municipalities. The polygon says
        # where it applies; the text names the first few and counts the rest.
        named = [*named[:_AREAS_SAID], f"and {len(named) - _AREAS_SAID:,} more areas"]
    return {
        "identifier": text(root, "identifier"),
        "status": text(root, "status"),
        "msg_type": text(root, "msgType"),
        "sent": text(root, "sent"),
        "language": text(info, "language"),
        "event": text(info, "event"),
        "headline": text(info, "headline"),
        "severity": text(info, "severity"),
        "urgency": text(info, "urgency"),
        "certainty": text(info, "certainty"),
        "expires": text(info, "expires"),
        "sender": text(info, "senderName") or text(root, "sender"),
        "instruction": text(info, "instruction"),
        "web": text(info, "web"),
        "area": "; ".join(named),
    }


class WorldWarningsFeed(Feed):
    """Alerts in effect at subscribed places, from the national authority that issued them."""

    name = "world_warnings"
    title = "Warnings in effect outside the United States (national authorities, via WMO)"
    source = "WMO Severe Weather Information Centre"
    cadence = timedelta(minutes=10)
    stale_after = timedelta(days=36500)  # a quiet day has no recent alert; the list is current

    #: CAP severity to the watch's. As for the National Weather Service's alerts.
    SEVERITY: ClassVar[dict[str, Severity]] = {
        "Extreme": Severity.SEVERE,
        "Severe": Severity.SIGNIFICANT,
    }
    #: Distinct places looked up per read. Each is one request to the centre.
    MAX_PLACES: ClassVar[int] = 250
    _COLUMNS: ClassVar[list[str]] = [
        "path", "identifier", "event", "headline", "severity", "urgency", "certainty",
        "sender", "sent", "expires", "instruction", "area", "web", "status", "msg_type",
        "place_ids",
    ]  # fmt: skip

    def __init__(self) -> None:
        """Start with no store bound and no alert read."""
        self._store: WatchStore | None = None
        #: CAP documents already read, by path. A published alert does not change.
        self._alerts: dict[str, dict[str, Any]] = {}

    def bind(self, store: WatchStore) -> None:
        """Take the store, for the subscribed places."""
        self._store = store

    def _covering(self, latitude: float, longitude: float) -> list[str]:
        """Paths of the CAP documents whose polygon contains the point."""
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        payload = SafeHTTPClient.get_json(
            _SWIC_WFS,
            params={
                "service": "WFS",
                "version": "1.0.0",  # longitude first, as GeoJSON writes a point
                "request": "GetFeature",
                "typeName": "local_postgis:postgis_geojsons",
                "outputFormat": "application/json",
                "propertyName": "capurl,row_type",
                "cql_filter": (
                    f"INTERSECTS(wkb_geometry, POINT({longitude:.2f} {latitude:.2f})) "
                    "AND row_type <> 'BOUNDARY'"
                ),
            },
            timeout=30.0,
        )
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            raise ValueError("the WMO map service's answer has no 'features' list")
        paths = {
            str((feature.get("properties") or {}).get("capurl") or "")
            for feature in features
            if isinstance(feature, dict)
        }
        return sorted(path for path in paths if _CAP_PATH.fullmatch(path))

    def _document(self, path: str) -> str:
        """The CAP document at ``path``, as the centre serves it."""
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        return SafeHTTPClient.get_text(_SWIC_CAP + path, timeout=30.0)

    def read(self) -> pd.DataFrame:
        """Ask which alerts cover each subscribed place, and read each new one."""
        points = self._store.places() if self._store is not None else {}
        beyond = max(0, len(points) - self.MAX_PLACES)
        rows: dict[str, dict[str, Any]] = {}
        asked = unanswered = unread = 0
        chosen = list(points.items())[: self.MAX_PLACES]

        def lookup(point: tuple[float, float]) -> list[str] | Exception:
            try:
                return self._covering(*point)
            except Exception as exc:  # one place failing costs the others nothing
                return exc

        # A lookup takes a second or two; one after another, 250 places would
        # take most of the cadence. A few at a time keeps the read short
        # without leaning on the centre.
        with ThreadPoolExecutor(max_workers=_LOOKUPS_AT_ONCE) as pool:
            answers = list(pool.map(lookup, [point for point, _ in chosen]))
        for (_point, ids), paths in zip(chosen, answers, strict=True):
            asked += 1
            if isinstance(paths, Exception):
                unanswered += 1
                logger.warning("watch: no alert lookup for a subscribed place: %s", paths)
                continue
            for path in paths:
                if path.startswith(_FROM_NWS):
                    continue
                if path not in self._alerts:
                    try:
                        self._alerts[path] = parse_cap(self._document(path))
                    except Exception as exc:
                        unread += 1
                        logger.warning("watch: could not read the alert %s: %s", path, exc)
                        continue
                row = rows.setdefault(path, {**self._alerts[path], "path": path, "place_ids": []})
                for place in ids:
                    if place not in row["place_ids"]:
                        row["place_ids"].append(place)
        if asked and unanswered == asked:
            raise ConnectionError(
                f"the alert lookup failed for all {unanswered} subscribed places checked"
            )
        # What no place is covered by any more need not be kept.
        self._alerts = {path: alert for path, alert in self._alerts.items() if path in rows}
        frame = pd.DataFrame(list(rows.values()), columns=self._COLUMNS)
        frame.attrs.update(
            places=asked - unanswered, unanswered=unanswered, unread=unread, beyond=beyond
        )
        return frame

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        """Each alert that is actual, not cancelled and not expired is a finding."""
        places = int(frame.attrs.get("places", 0))
        unanswered = int(frame.attrs.get("unanswered", 0))
        unread = int(frame.attrs.get("unread", 0))
        beyond = int(frame.attrs.get("beyond", 0))
        findings = []
        for row in frame.itertuples(index=False):
            expires = pd.to_datetime(row.expires, utc=True, errors="coerce")
            if pd.notna(expires) and _utc(expires) <= now:
                continue
            if row.msg_type == "Cancel" or (row.status and row.status != "Actual"):
                continue  # withdrawn, or an exercise or test message
            sent = pd.to_datetime(row.sent, utc=True, errors="coerce")
            rated = ", ".join(
                part for part in (row.severity, row.urgency, row.certainty) if part
            ).lower()
            issuer = str(row.sender) or "the issuing authority"
            event = str(row.event or "Alert")
            event = event[:1].upper() + event[1:]
            findings.append(
                Finding(
                    feed=self.name,
                    kind="cap_alert",
                    subject=str(row.identifier or row.path),
                    severity=self.SEVERITY.get(str(row.severity), Severity.NOTABLE),
                    headline=(
                        f"{event}: {_shorten(str(row.area), 90)}"
                        if row.area
                        else f"{event} ({issuer})"
                    ),
                    detail=(
                        (f"{row.headline}. " if row.headline else f"{row.event} from {issuer}. ")
                        + (
                            f"In effect until {_clock(_utc(expires))}. "
                            if pd.notna(expires)
                            else ""
                        )
                        + f"{issuer} rates it {rated or 'without a rating'}. This is its own "
                        "alert, relayed as issued through the WMO Severe Weather Information "
                        "Centre."
                    ),
                    observed_at=_utc(sent) if pd.notna(sent) else now,
                    source=issuer,
                    basis="an alert its national authority has in effect at a subscribed place",
                    ongoing=True,
                    reference=str(row.web or row.path),
                    action=str(row.instruction),
                    action_source=issuer if row.instruction else "",
                    area=str(row.area),
                    place_ids=tuple(int(place) for place in row.place_ids),
                )
            )
        if not places and not unanswered:
            summary = "no place is subscribed; alerts are looked up for subscribed places"
        else:
            summary = (
                f"{_count(len(findings), 'alert')} in effect at "
                f"{_count(places, 'subscribed place')} checked"
            )
        if unanswered:
            summary += f"; {_count(unanswered, 'place')} could not be checked this time"
        if unread:
            summary += f"; {_count(unread, 'alert')} could not be read this time"
        if beyond:
            summary += (
                f"; {_count(beyond, 'place')} NOT checked: more are subscribed than the "
                f"{self.MAX_PLACES} one read looks up"
            )
        return Judgment(
            findings, summary, None, len(frame), len(frame), partial=bool(unanswered or unread)
        )

    review_after = timedelta(minutes=30)
    review_until = timedelta(days=2)

    def review(self, finding: dict[str, Any], now: datetime) -> Outcome | None:
        """Read the alert again from the centre: was it issued as relayed."""
        record = "the alert's CAP document at the WMO Severe Weather Information Centre"
        path = next(
            (
                p
                for p, alert in self._alerts.items()
                if alert.get("identifier") == finding.get("subject")
            ),
            "",
        )
        if not path:
            return None  # no longer in effect anywhere subscribed; nothing to read it from
        try:
            again = parse_cap(self._document(path))
        except Exception:
            return None
        if again["msg_type"] == "Cancel":
            return Outcome("withdrawn", "Its authority has cancelled it.", record, final=True)
        return Outcome(
            "confirmed",
            f"{again['sender'] or 'Its authority'} has it on record as issued"
            + (f", rated {again['severity'].lower()}." if again["severity"] else "."),
            record,
        )


class GdacsFeed(Feed):
    """Disasters GDACS rates Orange or Red: cyclones, floods, volcanoes, wildfires."""

    name = "gdacs"
    title = "Disasters worldwide (GDACS)"
    source = (
        "Global Disaster Alert and Coordination System (United Nations and European Commission)"
    )
    cadence = timedelta(minutes=30)
    stale_after = timedelta(days=36500)  # a quiet week has no Orange or Red event

    #: GDACS's alert level to the watch's severity.
    SEVERITY: ClassVar[dict[str, Severity]] = {
        "Orange": Severity.SIGNIFICANT,
        "Red": Severity.SEVERE,
    }
    #: What is read. Earthquakes are the USGS feed's (the same event would be
    #: said twice); a drought has no point a radius can be drawn from.
    KINDS: ClassVar[dict[str, str]] = {
        "TC": "tropical cyclone",
        "FL": "flood",
        "VO": "volcanic eruption",
        "WF": "wildfire",
    }
    _COLUMNS: ClassVar[list[str]] = [
        "eventtype", "eventid", "name", "level", "current", "country", "began", "until",
        "modified", "measure", "latitude", "longitude", "report",
    ]  # fmt: skip

    def read(self) -> pd.DataFrame:
        """Read GDACS's Orange and Red events of the past two weeks."""
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        today = datetime.now(UTC)
        payload = SafeHTTPClient.get_json(
            _GDACS_SEARCH,
            params={
                "fromDate": (today - timedelta(days=14)).strftime("%Y-%m-%d"),
                "toDate": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
                "alertlevel": "Orange;Red",
                "eventlist": ";".join(self.KINDS),
            },
            timeout=30.0,
        )
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            raise ValueError("GDACS's answer is not the documented shape: no 'features' list")
        rows = []
        for feature in features:
            properties = feature.get("properties") if isinstance(feature, dict) else None
            point = ((feature.get("geometry") or {}).get("coordinates") or []) if properties else []
            if not isinstance(properties, dict) or len(point) < 2:
                continue
            measure = properties.get("severitydata") or {}
            links = properties.get("url") or {}
            rows.append(
                {
                    "eventtype": str(properties.get("eventtype") or ""),
                    "eventid": str(properties.get("eventid") or ""),
                    "name": str(properties.get("name") or ""),
                    "level": str(properties.get("alertlevel") or ""),
                    "current": str(properties.get("iscurrent") or "").lower() == "true",
                    "country": str(properties.get("country") or ""),
                    "began": properties.get("fromdate"),
                    "until": properties.get("todate"),
                    "modified": properties.get("datemodified"),
                    "measure": (
                        str(measure.get("severitytext") or "") if isinstance(measure, dict) else ""
                    ),
                    "longitude": float(point[0]),
                    "latitude": float(point[1]),
                    "report": str(links.get("report") or "") if isinstance(links, dict) else "",
                }
            )
        return pd.DataFrame(rows, columns=self._COLUMNS)

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        """A current event of a kind this feed reads, rated Orange or Red, is a finding."""
        if frame.empty:
            return Judgment([], "no Orange or Red event in the past two weeks", None, 0, 0)
        findings = []
        for row in frame.itertuples(index=False):
            if not row.current or row.level not in self.SEVERITY or row.eventtype not in self.KINDS:
                continue
            modified = pd.to_datetime(row.modified, utc=True, errors="coerce")
            where = f" in {row.country}" if row.country else ""
            findings.append(
                Finding(
                    feed=self.name,
                    kind="gdacs_event",
                    subject=f"{row.eventtype}{row.eventid}",
                    severity=self.SEVERITY[row.level],
                    headline=f"{row.name or self.KINDS[row.eventtype].capitalize() + where}: "
                    f"GDACS {row.level} alert",
                    detail=(
                        f"GDACS rates this {self.KINDS[row.eventtype]}{where} {row.level}"
                        + (f": {row.measure.strip()}" if row.measure.strip() else "")
                        + ". Its levels estimate the likely need for international "
                        "humanitarian help (Green, Orange, Red); they are not a public "
                        "warning. The position is the point GDACS gives for the event."
                    ),
                    observed_at=_utc(modified) if pd.notna(modified) else now,
                    source=self.source,
                    basis=f"GDACS alert level {row.level} on a current event",
                    latitude=float(row.latitude),
                    longitude=float(row.longitude),
                    ongoing=True,
                    reference=str(row.report),
                )
            )
        current = frame[frame["current"] & frame["eventtype"].isin(list(self.KINDS))]
        levels = current["level"].value_counts()
        summary = (
            f"{_count(len(current), 'current event')} rated Orange or Red "
            f"({int(levels.get('Red', 0))} red, {int(levels.get('Orange', 0))} orange)"
            if len(current)
            else "no current event is rated Orange or Red"
        )
        latest = pd.to_datetime(frame["modified"], utc=True, errors="coerce").max()
        return Judgment(
            findings, summary, _utc(latest) if pd.notna(latest) else None, len(frame), len(current)
        )
