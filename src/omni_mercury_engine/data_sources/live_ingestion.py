# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Uniform live-ingestion seam between hazard detectors and data-source clients.

Every hazard detector that supports optional live ingestion follows ONE pattern,
implemented on top of this module:

1. **Dependency injection** — the detector constructor accepts an optional
   ``data_sources`` client instance (default ``None`` = fully offline, no
   network is ever touched). The detector never constructs network clients
   implicitly unless its documented convenience knob (e.g.
   ``MeteorDetector(use_nasa_data=True)``) says so.
2. **fetch_live_data()** — a thin wrapper over :func:`fetch_live_datapoints`
   that returns a :class:`LiveFetch`. It fails loud: a failed fetch raises
   :class:`LiveDataError` instead of silently returning empty data.
3. **Simulated-source gate** — a client that labels its output with
   ``DataPoint.metadata["simulated"] = True`` is refused with
   :class:`SimulatedDataError` unless the caller passes ``allow_simulated=True``
   explicitly. Simulated data can therefore never masquerade as a real feed.
4. **detect_live()/predict_*_live()** — a convenience that maps the fetched
   :class:`~omni_mercury_engine.data_sources.base.DataPoint` objects onto the
   detector's *existing* input contract (never inventing raw measurements the
   source did not provide) and returns the detector's native result dataclass
   with three provenance fields populated: ``source_id``,
   ``data_provenance`` (``"live"`` or ``"simulated"``) and ``live_context``
   (source-derived quantities that do not fit the native result fields).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omni_mercury_engine.data_sources.base import (
        DataPoint,
        DataSourceBase,
        DataSourceType,
    )

PROVENANCE_LIVE = "live"
PROVENANCE_SIMULATED = "simulated"


class LiveDataError(RuntimeError):
    """A live data fetch failed or no client is wired for the requested feed.

    Raised instead of silently degrading: callers that opted into live
    ingestion must see fetch failures, not fabricated or empty stand-ins.
    """


class SimulatedDataError(LiveDataError):
    """A simulated source was consumed without an explicit opt-in.

    Sources that fabricate their payload label every emitted
    :class:`~omni_mercury_engine.data_sources.base.DataPoint` with
    ``metadata["simulated"] = True``. Consuming such a source requires the
    caller to pass ``allow_simulated=True`` explicitly so simulation can never
    be mistaken for a real feed.
    """


@dataclass
class LiveFetch:
    """Result of a provenance-checked live fetch.

    Attributes:
        source_id: ``source_id`` of the client that produced the data.
        data_points: The fetched (optionally type-filtered) data points.
        data_provenance: ``"live"`` for real feeds, ``"simulated"`` when any
            data point is labelled ``metadata["simulated"] = True`` (only
            reachable with an explicit ``allow_simulated=True`` opt-in).
        fetched_at: UTC wall-clock time of the fetch.
        cached: Whether the client served the fetch from its own cache.
        metadata: Free-form fetch metadata (e.g. filter parameters).
    """

    source_id: str
    data_points: list[DataPoint]
    data_provenance: str
    fetched_at: datetime
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def fetch_live_datapoints(
    client: DataSourceBase,
    *,
    allow_simulated: bool = False,
    source_types: list[DataSourceType] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> LiveFetch:
    """Fetch data points from a client with fail-loud + provenance semantics.

    Args:
        client: Any :class:`~omni_mercury_engine.data_sources.base.DataSourceBase`.
        allow_simulated: Explicit opt-in required to consume a source whose
            data points carry ``metadata["simulated"] = True``.
        source_types: Optional filter; only data points whose ``source_type``
            is in this list are returned.
        start_time: Optional start of the fetch window (client-specific).
        end_time: Optional end of the fetch window (client-specific).
        use_cache: Whether the client may serve from its own cache (the cache
            TTL is the client's own ``CacheConfig``).
        **kwargs: Passed through to the client's ``_fetch_impl``.

    Returns:
        A :class:`LiveFetch` with provenance resolved.

    Raises:
        LiveDataError: The fetch failed (network error, rate limit, breaker
            open, HTTP error). Never returns fabricated stand-in data.
        SimulatedDataError: The source emitted simulated data points and
            ``allow_simulated`` is False.
    """
    result = client.fetch_sync(
        start_time=start_time, end_time=end_time, use_cache=use_cache, **kwargs
    )
    if not result.success:
        raise LiveDataError(f"{client.source_id}: live fetch failed: {result.error}")

    points = result.data_points
    if source_types is not None:
        points = [dp for dp in points if dp.source_type in source_types]

    simulated = any(bool(dp.metadata.get("simulated", False)) for dp in points)
    if simulated and not allow_simulated:
        raise SimulatedDataError(
            f"{client.source_id}: source emitted SIMULATED data points; refusing to "
            f"present them as live. Pass allow_simulated=True to opt in explicitly."
        )

    return LiveFetch(
        source_id=client.source_id,
        data_points=points,
        data_provenance=PROVENANCE_SIMULATED if simulated else PROVENANCE_LIVE,
        fetched_at=datetime.now(UTC),
        cached=result.cached,
        metadata={"source_types": [t.value for t in source_types] if source_types else None},
    )


def require_live_client(client: DataSourceBase | None, detector: str, feed: str) -> DataSourceBase:
    """Return ``client`` or raise a loud, actionable error when none is wired.

    Args:
        client: The injected data-source client (may be None).
        detector: Detector class name for the error message.
        feed: Human-readable feed description for the error message.

    Returns:
        The non-None client.

    Raises:
        LiveDataError: When no client was injected at construction time.
    """
    if client is None:
        raise LiveDataError(
            f"{detector}: no {feed} client injected; construct the detector with a "
            f"data_sources client instance to enable the optional live path."
        )
    return client


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points in kilometres.

    Args:
        lat1: Latitude of the first point (degrees).
        lon1: Longitude of the first point (degrees).
        lat2: Latitude of the second point (degrees).
        lon2: Longitude of the second point (degrees).

    Returns:
        Distance in kilometres (mean Earth radius 6371.0 km).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * 6371.0 * math.asin(math.sqrt(a))
