# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for meteor / near-Earth-object data.

Two real, live sources:

* **NASA/JPL CNEOS Fireball archive** (``ssd-api.jpl.nasa.gov/fireball.api``)
  — sensor-derived atmospheric-impact events since 1988 with total radiated
  energy (in units of 1e10 J) and computed total impact energy (kt TNT).
* **NASA NeoWs close-approach feed** (``api.nasa.gov/neo/rest/v1/feed``) —
  near-Earth-object close approaches with miss distance, relative velocity,
  estimated diameter, absolute magnitude H, and JPL hazard designations.

Ground-truth events cover fireball-archive year windows around major
bolides (Chelyabinsk 2013, Bering Sea 2018) plus documented NEO
close-approach weeks (2019 OK, 2023 BU).

Labels are threshold-derived, hence ``LABEL_SOURCE = "statistical"`` (see
``loaders.label_provenance``):

* Fireball events: anomalous when computed impact energy >= 1 kt. The
  small-impactor flux N(>E) ~ 3.7 E^-0.9 per year (E in kt; Brown et al.
  2002, Nature 420, 294) makes >= 1 kt a genuinely rare class (~4/yr
  globally). The energy is also an engineered feature, so this labelling
  is circular by construction and declared as such.
* NEO events: anomalous when JPL designates the object a Potentially
  Hazardous Asteroid — the PHA definition is itself a threshold rule
  (Earth MOID <= 0.05 au AND H <= 22.0; CNEOS NEO Basics,
  https://cneos.jpl.nasa.gov/about/neo_groups.html), and H is an
  engineered feature.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
_FIREBALL_URL = "https://ssd-api.jpl.nasa.gov/fireball.api"
_NEOWS_FEED_URL = "https://api.nasa.gov/neo/rest/v1/feed"

#: Impact-energy threshold (kt TNT) labelling a fireball anomalous.
#: ~4 events/yr exceed it globally (Brown et al. 2002 flux law).
FIREBALL_ANOMALY_THRESHOLD_KT: float = 1.0

# ---------------------------------------------------------------------------
# Ground-truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "chelyabinsk_2013": {
        "type": "fireball",
        "name": "Chelyabinsk superbolide year",
        "date": "2013-02-15",
        "description": (
            "CNEOS fireball archive 2012-09..2013-08 including the 2013-02-15 "
            "Chelyabinsk superbolide (441 kt impact energy)."
        ),
        "start": "2012-09-01",
        "end": "2013-08-31",
    },
    "bering_sea_2018": {
        "type": "fireball",
        "name": "Bering Sea bolide year",
        "date": "2018-12-18",
        "description": (
            "CNEOS fireball archive 2018-06..2019-05 including the 2018-12-18 "
            "Bering Sea bolide (49 kt archive impact energy)."
        ),
        "start": "2018-06-01",
        "end": "2019-05-31",
    },
    "fireballs_2020": {
        "type": "fireball",
        "name": "2020 fireball background year",
        "date": "2020-12-22",
        "description": (
            "CNEOS fireball archive calendar 2020: background flux with two "
            ">= 1 kt events (2020-12-22 12 kt, 2020-10-23 1.1 kt)."
        ),
        "start": "2020-01-01",
        "end": "2020-12-31",
    },
    "neo_2019_ok_week": {
        "type": "neo",
        "name": "2019 OK close-approach week",
        "date": "2019-07-25",
        "description": (
            "NeoWs close approaches 2019-07-20..27, the week asteroid 2019 OK "
            "passed at ~0.19 lunar distances with ~1 day of warning."
        ),
        "start": "2019-07-20",
        "end": "2019-07-27",
    },
    "neo_2023_bu_week": {
        "type": "neo",
        "name": "2023 BU close-approach week",
        "date": "2023-01-26",
        "description": (
            "NeoWs close approaches 2023-01-24..31, the week asteroid 2023 BU "
            "passed within ~10,000 km of Earth's surface."
        ),
        "start": "2023-01-24",
        "end": "2023-01-31",
    },
}

_FIREBALL_COLUMNS = [
    "time",
    "radiated_energy_e10_j",
    "impact_energy_kt",
    "latitude",
    "longitude",
    "altitude_km",
    "velocity_km_s",
]
_NEO_COLUMNS = [
    "time",
    "miss_distance_km",
    "miss_distance_ld",
    "relative_velocity_km_s",
    "diameter_min_km",
    "diameter_max_km",
    "absolute_magnitude_h",
    "is_pha",
    "is_sentry",
]


class MeteorLoader(BaseDomainLoader):
    """Loader for fireball-archive and NEO close-approach data.

    Event rows are heterogeneous across the two sources; each cataloged
    event is internally consistent (all-fireball or all-NEO), and
    :meth:`engineer_features` dispatches on the schema present.
    """

    DOMAIN: str = "meteor"
    SOURCE_URL: str = _FIREBALL_URL
    # Fireball labels threshold the scored impact-energy feature; NEO
    # labels are JPL's PHA designation, itself a MOID/H threshold rule on
    # scored features. Declared statistical (see loaders.label_provenance).
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False
    API_KEY_ENV_VAR: str = "NASA_API_KEY"

    FIREBALL_FEATURE_COLUMNS: list[str] = [
        "log10_impact_energy_kt",
        "log10_radiated_energy",
        "velocity_km_s",
        "altitude_km",
        "abs_latitude",
        "days_since_prev",
        "log_energy_dev",
    ]
    NEO_FEATURE_COLUMNS: list[str] = [
        "log10_miss_distance_km",
        "relative_velocity_km_s",
        "log10_diameter_avg_km",
        "absolute_magnitude_h",
        "is_sentry",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the 30 most recent fireball-archive events.

        Returns:
            DataFrame with the fireball schema (time, energies, location,
            velocity), newest events included.

        Raises:
            ConnectionError: If the JPL fireball API is unreachable.
        """
        payload = self._fetch_json(_FIREBALL_URL, params={"limit": "30"})
        df = self._fireball_to_dataframe(payload)
        logger.info("Fetched %d recent fireball records from JPL CNEOS.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch archive data for a cataloged event window.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            Chronologically sorted DataFrame — fireball schema for
            ``type="fireball"`` events, NEO schema for ``type="neo"``.

        Raises:
            ValueError: If *event_id* is unknown or the source returns an
                empty catalog for the window.
            ConnectionError: If the upstream API is unreachable.
        """
        event = self._event(event_id)
        cache_key = f"meteor_historical_{event_id}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached data for '%s'.", event_id)
            return pd.DataFrame(cached)

        if event["type"] == "fireball":
            payload = self._fetch_json(
                _FIREBALL_URL,
                params={"date-min": event["start"], "date-max": event["end"]},
            )
            df = self._fireball_to_dataframe(payload)
        else:
            params = {
                "start_date": event["start"],
                "end_date": event["end"],
                "api_key": self._api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY"),
            }
            payload = self._fetch_json(_NEOWS_FEED_URL, params=params)
            df = self._neows_to_dataframe(payload)

        if df.empty:
            raise ValueError(
                f"Upstream returned no records for cataloged event '{event_id}' "
                f"({event['start']}..{event['end']}); refusing to fabricate data."
            )
        df = df.sort_values("time").reset_index(drop=True)
        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d records for event '%s'.", len(df), event_id)
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground-truth meteor/NEO events.

        Returns:
            List of dicts with *event_id*, *name*, *date*, *description*.
        """
        return [
            {
                "event_id": event_id,
                "name": meta["name"],
                "date": meta["date"],
                "description": meta["description"],
            }
            for event_id, meta in _EVENT_CATALOG.items()
        ]

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Binary labels per archive row.

        Fireball events: 1 when computed impact energy >=
        :data:`FIREBALL_ANOMALY_THRESHOLD_KT` (kt TNT); rows without a
        computed impact energy are labelled 0 (small events — CNEOS
        computes impact energy for every archive row in practice).
        NEO events: 1 when JPL designates the object a PHA.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            1-D binary array aligned with :meth:`fetch_historical` rows.

        Raises:
            ValueError: If *event_id* is unknown.
        """
        event = self._event(event_id)
        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        if event["type"] == "fireball":
            energy = df["impact_energy_kt"].to_numpy(dtype=np.float64)
            labels = (np.nan_to_num(energy, nan=0.0) >= FIREBALL_ANOMALY_THRESHOLD_KT).astype(
                np.int64
            )
        else:
            labels = df["is_pha"].to_numpy(dtype=np.int64)
        logger.info(
            "Ground truth for '%s': %d anomalies / %d total.",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Engineer features, dispatching on the schema present.

        Fireball rows (7 features): log10 impact energy (kt), log10
        radiated energy (1e10 J), entry velocity (km/s), burst altitude
        (km), |latitude|, days since the previous event, and log-energy
        deviation from the trailing 10-event median.

        NEO rows (5 features): log10 miss distance (km), relative velocity
        (km/s), log10 mean estimated diameter (km), absolute magnitude H,
        and the Sentry-monitored flag.

        Args:
            raw_data: DataFrame from :meth:`fetch_historical` or
                :meth:`fetch_realtime`.

        Returns:
            2-D array of shape ``(n_samples, 7)`` or ``(n_samples, 5)``.

        Raises:
            ValueError: If the DataFrame matches neither schema.
        """
        if raw_data.empty:
            return np.empty((0, len(self.FIREBALL_FEATURE_COLUMNS)), dtype=np.float64)

        if "impact_energy_kt" in raw_data.columns:
            features = self._fireball_features(raw_data)
        elif "miss_distance_km" in raw_data.columns:
            features = self._neo_features(raw_data)
        else:
            raise ValueError(
                "DataFrame matches neither the fireball nor the NEO schema "
                f"(columns: {list(raw_data.columns)})."
            )

        features = np.where(np.isinf(features), np.nan, features)
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                median_val = np.nanmedian(col)
                col[mask] = median_val if np.isfinite(median_val) else 0.0
        return features

    def _fireball_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        df = raw_data.sort_values("time").reset_index(drop=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_impact = np.log10(df["impact_energy_kt"].to_numpy(dtype=np.float64))
            log_radiated = np.log10(df["radiated_energy_e10_j"].to_numpy(dtype=np.float64))
        velocity = df["velocity_km_s"].to_numpy(dtype=np.float64)
        altitude = df["altitude_km"].to_numpy(dtype=np.float64)
        abs_lat = np.abs(df["latitude"].to_numpy(dtype=np.float64))

        time_s = df["time"].to_numpy(dtype=np.float64)
        days_since_prev = np.zeros(len(df), dtype=np.float64)
        if len(df) > 1:
            days_since_prev[1:] = np.diff(time_s) / 86400.0

        log_energy_dev = (
            pd.Series(log_impact) - pd.Series(log_impact).rolling(10, min_periods=1).median()
        ).to_numpy()

        return np.column_stack(
            [
                log_impact,
                log_radiated,
                velocity,
                altitude,
                abs_lat,
                days_since_prev,
                log_energy_dev,
            ]
        )

    def _neo_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        df = raw_data.sort_values("time").reset_index(drop=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_miss = np.log10(df["miss_distance_km"].to_numpy(dtype=np.float64))
            diameter_avg = 0.5 * (
                df["diameter_min_km"].to_numpy(dtype=np.float64)
                + df["diameter_max_km"].to_numpy(dtype=np.float64)
            )
            log_diameter = np.log10(diameter_avg)
        return np.column_stack(
            [
                log_miss,
                df["relative_velocity_km_s"].to_numpy(dtype=np.float64),
                log_diameter,
                df["absolute_magnitude_h"].to_numpy(dtype=np.float64),
                df["is_sentry"].to_numpy(dtype=np.float64),
            ]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _event(self, event_id: str) -> dict[str, Any]:
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )
        return _EVENT_CATALOG[event_id]

    @staticmethod
    def _fireball_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
        """Convert a JPL fireball API response to a DataFrame."""
        fields = payload.get("fields", [])
        rows = payload.get("data", []) or []
        if not fields or not rows:
            return pd.DataFrame(columns=_FIREBALL_COLUMNS)
        index = {name: i for i, name in enumerate(fields)}

        def _get(row: list[Any], name: str) -> Any:
            pos = index.get(name)
            return row[pos] if pos is not None and pos < len(row) else None

        records: list[dict[str, Any]] = []
        for row in rows:
            date_str = _get(row, "date")
            if not date_str:
                continue
            timestamp = (
                datetime.strptime(str(date_str), "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=UTC)
                .timestamp()
            )
            lat = _to_float(_get(row, "lat"))
            if lat is not None and _get(row, "lat-dir") == "S":
                lat = -lat
            lon = _to_float(_get(row, "lon"))
            if lon is not None and _get(row, "lon-dir") == "W":
                lon = -lon
            records.append(
                {
                    "time": timestamp,
                    "radiated_energy_e10_j": _to_float(_get(row, "energy")),
                    "impact_energy_kt": _to_float(_get(row, "impact-e")),
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_km": _to_float(_get(row, "alt")),
                    "velocity_km_s": _to_float(_get(row, "vel")),
                }
            )
        return pd.DataFrame(records, columns=_FIREBALL_COLUMNS)

    @staticmethod
    def _neows_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
        """Convert a NeoWs feed response to a per-close-approach DataFrame."""
        days = payload.get("near_earth_objects", {}) or {}
        records: list[dict[str, Any]] = []
        for neos in days.values():
            for neo in neos:
                diameter = neo.get("estimated_diameter", {}).get("kilometers", {})
                for approach in neo.get("close_approach_data", []):
                    epoch_ms = approach.get("epoch_date_close_approach")
                    if epoch_ms is None:
                        continue
                    miss = approach.get("miss_distance", {})
                    velocity = approach.get("relative_velocity", {})
                    records.append(
                        {
                            "time": float(epoch_ms) / 1000.0,
                            "miss_distance_km": _to_float(miss.get("kilometers")),
                            "miss_distance_ld": _to_float(miss.get("lunar")),
                            "relative_velocity_km_s": _to_float(
                                velocity.get("kilometers_per_second")
                            ),
                            "diameter_min_km": _to_float(diameter.get("estimated_diameter_min")),
                            "diameter_max_km": _to_float(diameter.get("estimated_diameter_max")),
                            "absolute_magnitude_h": _to_float(neo.get("absolute_magnitude_h")),
                            "is_pha": int(
                                bool(neo.get("is_potentially_hazardous_asteroid", False))
                            ),
                            "is_sentry": int(bool(neo.get("is_sentry_object", False))),
                        }
                    )
        return pd.DataFrame(records, columns=_NEO_COLUMNS)


def _to_float(value: Any) -> float | None:
    """Parse an optional numeric API field, mapping missing to None."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["FIREBALL_ANOMALY_THRESHOLD_KT", "MeteorLoader"]
