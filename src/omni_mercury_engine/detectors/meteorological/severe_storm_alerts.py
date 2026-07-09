# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared NWS CAP alert plumbing for the severe-storm detector cluster.

The hail, winter-storm, derecho, and dust-storm detectors each cross-check
their physics-based assessment against official National Weather Service
alerts.  Alerts arrive in one of three shapes:

* Raw ``api.weather.gov`` GeoJSON -- either a ``FeatureCollection`` dict or a
  list of feature dicts, each carrying CAP fields under ``"properties"``
  (including Impact-Based Warning threat tags under ``"parameters"``, e.g.
  ``maxHailSize`` / ``hailThreat`` / ``maxWindGust``).
* Pre-flattened CAP property dicts (``{"event": ..., "severity": ...}``).
* :class:`~omni_mercury_engine.data_sources.base.DataPoint` objects produced
  by :class:`~omni_mercury_engine.data_sources.earth_science.NWSWeatherAlertsSource`,
  whose ``data`` dict carries ``event`` / ``severity`` / ``description`` but
  not the raw CAP ``parameters`` block.  For those, threat magnitudes are
  recovered from the IBW text tags embedded in ``description`` (e.g.
  ``"MAX HAIL SIZE...1.00 IN"``).

This module normalizes all three shapes into plain CAP property dicts and
provides the event-type filter and threat-tag parsers used by every detector
in the cluster.  Unrecognized payload shapes raise ``TypeError`` -- the
cross-check must never silently score an empty list because a caller passed
the wrong object.

CAP field reference: NWS API documentation (https://www.weather.gov/documentation/services-web-api)
and the NWS Impact-Based Warning tag specification (maxHailSize in inches,
maxWindGust in mph).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "filter_alerts_by_event",
    "normalize_alert_records",
    "parse_max_hail_size_in",
    "parse_max_wind_gust_mph",
    "parse_threat_tag",
]

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?|\.\d+)")
_HAIL_DESC_RE = re.compile(r"MAX\s+HAIL\s+SIZE\s*\.{2,}\s*(\d*\.?\d+)\s*IN", re.IGNORECASE)
_GUST_DESC_RE = re.compile(r"MAX\s+WIND\s+GUST\s*\.{2,}\s*(\d*\.?\d+)\s*MPH", re.IGNORECASE)


def normalize_alert_records(alerts: Any) -> list[dict[str, Any]]:
    """Normalize NWS alert payloads into a list of CAP property dicts.

    Args:
        alerts: One of (a) a GeoJSON ``FeatureCollection`` dict with a
            ``"features"`` list, (b) an iterable of GeoJSON feature dicts
            (``{"properties": {...}}``), (c) an iterable of flat CAP property
            dicts (``{"event": ...}``), or (d) an iterable of DataPoint-like
            objects exposing a ``data`` dict attribute.

    Returns:
        List of flat CAP property dicts, one per alert.

    Raises:
        TypeError: If the payload (or any element) has none of the
            recognized shapes.  Fail-loud: a mis-shaped payload must not be
            silently treated as "no active alerts".
    """
    if isinstance(alerts, dict):
        if "features" not in alerts:
            raise TypeError(
                "Alert dict payload has no 'features' key; expected an "
                "api.weather.gov GeoJSON FeatureCollection."
            )
        alerts = alerts["features"]

    if isinstance(alerts, (str, bytes)) or not isinstance(alerts, Iterable):
        raise TypeError(
            f"Unsupported alert payload type {type(alerts).__name__}; expected "
            "a FeatureCollection dict or an iterable of alert records."
        )

    records: list[dict[str, Any]] = []
    for item in alerts:
        if isinstance(item, dict):
            if "properties" in item and isinstance(item["properties"], dict):
                records.append(item["properties"])
            elif "event" in item:
                records.append(item)
            else:
                raise TypeError(
                    "Alert record dict has neither 'properties' nor 'event'; "
                    f"keys={sorted(item.keys())[:8]}"
                )
        elif hasattr(item, "data") and isinstance(item.data, dict):
            records.append(item.data)
        else:
            raise TypeError(
                f"Unsupported alert record type {type(item).__name__}; expected "
                "a GeoJSON feature dict, CAP property dict, or DataPoint."
            )
    return records


def filter_alerts_by_event(
    records: list[dict[str, Any]],
    event_types: Iterable[str],
) -> list[dict[str, Any]]:
    """Return records whose CAP ``event`` matches one of ``event_types``.

    Matching is case-insensitive and exact (CAP event names are a controlled
    vocabulary, e.g. ``"Severe Thunderstorm Warning"``).

    Args:
        records: Flat CAP property dicts from :func:`normalize_alert_records`.
        event_types: CAP event names to keep.

    Returns:
        Matching records, in input order.
    """
    wanted = {e.strip().lower() for e in event_types}
    return [r for r in records if str(r.get("event", "")).strip().lower() in wanted]


def _first_parameter_value(record: dict[str, Any], key: str) -> str | None:
    """Return the first CAP ``parameters[key]`` entry as a string, if present."""
    params = record.get("parameters")
    if isinstance(params, dict):
        values = params.get(key)
        if isinstance(values, (list, tuple)) and values:
            return str(values[0])
        if isinstance(values, (str, int, float)):
            return str(values)
    return None


def parse_max_hail_size_in(record: dict[str, Any]) -> float | None:
    """Extract the IBW maximum hail size (inches) from a CAP alert record.

    Reads the ``maxHailSize`` CAP parameter when present (values are strings
    such as ``"1.00"`` or ``"Up to .75"``), falling back to the
    ``"MAX HAIL SIZE...X.XX IN"`` tag in the alert ``description`` text (the
    only channel available on flattened DataPoint records).

    Args:
        record: Flat CAP property dict.

    Returns:
        Hail size in inches, or ``None`` when the alert carries no hail tag.
    """
    raw = _first_parameter_value(record, "maxHailSize")
    if raw is not None:
        match = _NUMBER_RE.search(raw)
        if match:
            return float(match.group(1))
    description = record.get("description")
    if isinstance(description, str):
        match = _HAIL_DESC_RE.search(description)
        if match:
            return float(match.group(1))
    return None


def parse_max_wind_gust_mph(record: dict[str, Any]) -> float | None:
    """Extract the IBW maximum wind gust (mph) from a CAP alert record.

    Reads the ``maxWindGust`` CAP parameter (strings such as ``"60 MPH"``),
    falling back to the ``"MAX WIND GUST...NN MPH"`` description tag.

    Args:
        record: Flat CAP property dict.

    Returns:
        Gust in mph, or ``None`` when the alert carries no wind tag.
    """
    raw = _first_parameter_value(record, "maxWindGust")
    if raw is not None:
        match = _NUMBER_RE.search(raw)
        if match:
            return float(match.group(1))
    description = record.get("description")
    if isinstance(description, str):
        match = _GUST_DESC_RE.search(description)
        if match:
            return float(match.group(1))
    return None


def parse_threat_tag(record: dict[str, Any], key: str) -> str | None:
    """Return a categorical IBW threat tag (e.g. ``hailThreat``) if present.

    Args:
        record: Flat CAP property dict.
        key: CAP parameter name (``"hailThreat"``, ``"windThreat"``,
            ``"thunderstormDamageThreat"``, ...).

    Returns:
        The tag string (e.g. ``"RADAR INDICATED"``, ``"OBSERVED"``) or ``None``.
    """
    return _first_parameter_value(record, key)
