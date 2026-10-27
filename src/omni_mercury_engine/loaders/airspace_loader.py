# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain loader for live airspace state vectors from the OpenSky Network.

Connects to OpenSky's anonymous REST API for crowd-sourced, real ADS-B /
Mode S receptions: every tracked aircraft's position, altitude, velocity,
heading, vertical rate and transponder squawk, worldwide or within a
bounding box. This is Mercury's live-GIS airspace surface: kinematic
anomaly detection over what is actually flying right now.

Honest semantics, stated plainly:

* **Emergency squawks are real declared events.** Codes 7500 (unlawful
  interference), 7600 (radio failure) and 7700 (general emergency) are set
  by the crew; the loader surfaces them as the ``squawk_emergency`` column.
  They are kept **out** of the engineered feature matrix, so across
  snapshots they can serve as sparse genuine event markers that are
  independent of the kinematics Mercury scores.
* **Kinematic outliers are "aircraft behaving unusually", nothing more.**
  A track that sits outside the live envelope (altitude-velocity
  relationship, vertical rate, baro/geo altitude disagreement) is a
  candidate for attention - sensor faults, data errors, genuine
  emergencies, and unusual flight profiles all land here. The loader
  manufactures no stronger claim, which is why ``LABEL_SOURCE`` is
  ``"statistical"``.

Anonymous access is rate-limited by OpenSky (hundreds of calls per day);
callers that poll should hold a cadence of minutes, not seconds, and
production deployments should authenticate. API reference:
https://openskynetwork.github.io/opensky-api/rest.html
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader
from omni_mercury_engine.security.input_validation import TrustedEndpoints

logger = logging.getLogger(__name__)

#: OpenSky state-vector field order (REST API, 17 fields).
_STATE_FIELDS: tuple[str, ...] = (
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
)

#: Transponder codes crews set for real declared events.
EMERGENCY_SQUAWKS: frozenset[str] = frozenset({"7500", "7600", "7700"})


def parse_states(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse an OpenSky ``states/all`` payload into a DataFrame.

    Args:
        payload: Decoded JSON with ``time`` and ``states`` keys; each state
            is the API's 17-element list.

    Returns:
        One row per aircraft with :data:`_STATE_FIELDS` columns plus
        ``snapshot_time`` and ``squawk_emergency``. Rows without position,
        velocity or barometric altitude are dropped (no imputation).

    Raises:
        ValueError: The payload lacks the documented keys.
    """
    if "time" not in payload or "states" not in payload:
        raise ValueError("airspace: OpenSky payload lacks 'time'/'states'; schema drifted")
    states = payload["states"] or []
    rows: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, (list, tuple)) or len(state) < len(_STATE_FIELDS):
            continue
        record = dict(zip(_STATE_FIELDS, state))
        if (
            record["longitude"] is None
            or record["latitude"] is None
            or record["velocity"] is None
            or record["baro_altitude"] is None
        ):
            continue
        record["snapshot_time"] = int(payload["time"])
        squawk = str(record["squawk"]).strip() if record["squawk"] is not None else ""
        record["squawk_emergency"] = squawk in EMERGENCY_SQUAWKS
        rows.append(record)
    return pd.DataFrame(rows)


class AirspaceLoader(BaseDomainLoader):
    """Loader for live OpenSky airspace state vectors.

    An optional bounding box (``lamin``/``lomin``/``lamax``/``lomax``)
    scopes the query; without one the snapshot is global (thousands of
    aircraft, a heavier response, and a costlier anonymous-quota hit).
    """

    DOMAIN: str = "airspace"
    SOURCE_URL: str = TrustedEndpoints.OPENSKY_STATES
    # Kinematic-envelope "anomalies" are statistical by construction; the
    # genuine event markers (emergency squawks) are deliberately excluded
    # from the scored features and carried as a separate column.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False

    FEATURE_COLUMNS: list[str] = [
        "baro_altitude_m",
        "geo_altitude_m",
        "alt_disagreement_m",
        "velocity_ms",
        "vertical_rate_ms",
        "true_track_deg",
        "latitude_deg",
        "longitude_deg",
        "on_ground",
    ]

    def __init__(
        self,
        *,
        lamin: float | None = None,
        lomin: float | None = None,
        lamax: float | None = None,
        lomax: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the loader, optionally scoped to a bounding box.

        Args:
            lamin: Minimum latitude of the bounding box (degrees).
            lomin: Minimum longitude of the bounding box (degrees).
            lamax: Maximum latitude of the bounding box (degrees).
            lomax: Maximum longitude of the bounding box (degrees).
            **kwargs: Forwarded to :class:`BaseDomainLoader`.

        Raises:
            ValueError: A partial bounding box (some but not all corners).
        """
        super().__init__(**kwargs)
        corners = (lamin, lomin, lamax, lomax)
        if any(c is not None for c in corners) and not all(c is not None for c in corners):
            raise ValueError("airspace: a bounding box needs all of lamin/lomin/lamax/lomax")
        self.bbox: tuple[float, float, float, float] | None = (
            (float(lamin), float(lomin), float(lamax), float(lomax))  # type: ignore[arg-type]
            if lamin is not None
            else None
        )

    def _states_url(self) -> str:
        if self.bbox is None:
            return TrustedEndpoints.OPENSKY_STATES
        lamin, lomin, lamax, lomax = self.bbox
        return (
            f"{TrustedEndpoints.OPENSKY_STATES}"
            f"?lamin={lamin:g}&lomin={lomin:g}&lamax={lamax:g}&lomax={lomax:g}"
        )

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the current airspace snapshot.

        Returns:
            DataFrame of live aircraft states (see :func:`parse_states`).

        Raises:
            ConnectionError: OpenSky is unreachable after retries.
            ValueError: The response is not the documented payload.
        """
        cache_key = "airspace_realtime" if self.bbox is None else f"airspace_realtime_{self.bbox}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached real-time airspace data.")
            return pd.DataFrame(cached)
        raw = self._fetch_url(self._states_url())
        payload = json.loads(raw.decode("utf-8"))
        df = parse_states(payload)
        if not df.empty:
            self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d live aircraft states from OpenSky (%d emergency squawk).",
            len(df),
            int(df["squawk_emergency"].sum()) if len(df) else 0,
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Historical state vectors are not served anonymously.

        OpenSky's historical database requires an authenticated account
        with granted access; fabricating past airspace would be synthetic
        data wearing a real event's name, so this loader refuses instead.

        Raises:
            NotImplementedError: Always; use an authenticated OpenSky
                account (or its Impala/Trino interface) for history.
        """
        raise NotImplementedError(
            "airspace: OpenSky serves historical state vectors only to "
            "authenticated accounts with granted access; this loader does "
            "not fabricate past airspace. Requested event: " + repr(event_id)
        )

    def list_events(self) -> list[dict[str, Any]]:
        """No anonymous historical catalog exists; see :meth:`fetch_historical`."""
        return []

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """No anonymous historical catalog exists; see :meth:`fetch_historical`.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "airspace: no anonymous historical events; see fetch_historical. "
            "Requested event: " + repr(event_id)
        )

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Kinematic feature matrix for the live snapshot.

        The transponder squawk (the genuine event channel) is deliberately
        excluded so statistical scores stay independent of it.

        Args:
            raw_data: Output of :meth:`fetch_realtime`.

        Returns:
            ``(n_aircraft, len(FEATURE_COLUMNS))`` float64 matrix; rows
            with a missing geo altitude fall back to the barometric value
            (disagreement 0 rather than an invented offset), and rows with
            a missing vertical rate or track are dropped.
        """
        if raw_data.empty:
            return np.empty((0, len(self.FEATURE_COLUMNS)), dtype=np.float64)
        frame = raw_data.copy()
        baro = pd.to_numeric(frame["baro_altitude"], errors="coerce")
        geo = pd.to_numeric(frame["geo_altitude"], errors="coerce").fillna(baro)
        frame["baro_altitude_m"] = baro
        frame["geo_altitude_m"] = geo
        frame["alt_disagreement_m"] = (geo - baro).abs()
        frame["velocity_ms"] = pd.to_numeric(frame["velocity"], errors="coerce")
        frame["vertical_rate_ms"] = pd.to_numeric(frame["vertical_rate"], errors="coerce")
        frame["true_track_deg"] = pd.to_numeric(frame["true_track"], errors="coerce")
        frame["latitude_deg"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude_deg"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame["on_ground"] = frame["on_ground"].astype(bool).astype(np.float64)
        matrix = frame[self.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        return np.asarray(matrix[np.isfinite(matrix).all(axis=1)], dtype=np.float64)
