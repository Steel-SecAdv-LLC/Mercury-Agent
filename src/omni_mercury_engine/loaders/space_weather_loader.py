# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for space-weather (geomagnetic storm) data.

Two real, live sources:

* **USGS Geomagnetism web service** (``geomag.usgs.gov/ws/data/``) —
  per-observatory 1-minute adjusted variation magnetometer data (X, Y, Z,
  F in nT). This is the feature substrate: rapid horizontal-field
  variation (dB/dt) is the physically meaningful storm observable at
  ground level (Viljanen et al. 2001, Ann. Geophys. 19).
* **NASA DONKI GST** (``api.nasa.gov/DONKI/GST``) — geomagnetic-storm
  records with the observed NOAA planetary Kp series (``allKpIndex``),
  used for labels.

Ground-truth events are five documented major storms with archived
magnetometer coverage (the USGS web service serves adjusted minute data
from ~2017 onward; earlier storms return empty series and are therefore
not cataloged here). Labels mark the 3-hour planetary Kp windows at or
above Kp 5 (NOAA G1) reported by DONKI GST — a threshold on a
geomagnetic index, hence ``LABEL_SOURCE = "statistical"`` (see
``loaders.label_provenance``): Kp is derived from the same class of
ground-magnetometer disturbance that the engineered features score, so
these labels are excluded from the governed-fusion headline as circular.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
_GEOMAG_URL = "https://geomag.usgs.gov/ws/data/"
_DONKI_GST_URL = "https://api.nasa.gov/DONKI/GST"

#: Kp threshold (NOAA G1) defining a storm-labelled 3-hour window.
KP_STORM_THRESHOLD: float = 5.0

# ---------------------------------------------------------------------------
# Ground-truth event catalog: documented major geomagnetic storms with
# archived USGS minute data AND DONKI GST Kp records (all verified live).
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "september_2017": {
        "name": "September 2017 G4 storms",
        "date": "2017-09-07",
        "description": (
            "G4 (severe) geomagnetic storms of 2017-09-07/08 driven by the "
            "AR12673 X9.3/X8.2 flare-CME sequence; observed Kp 8."
        ),
        "start": "2017-09-06T00:00:00Z",
        "end": "2017-09-10T00:00:00Z",
    },
    "march_2023": {
        "name": "March 2023 G4 storm",
        "date": "2023-03-23",
        "description": "G4 storm of 2023-03-23/24 (observed Kp 8) from a stealth CME.",
        "start": "2023-03-22T00:00:00Z",
        "end": "2023-03-26T00:00:00Z",
    },
    "april_2023": {
        "name": "April 2023 G4 storm",
        "date": "2023-04-23",
        "description": "G4 storm of 2023-04-23/24 (observed Kp 8) with strong GIC activity.",
        "start": "2023-04-22T00:00:00Z",
        "end": "2023-04-26T00:00:00Z",
    },
    "gannon_may_2024": {
        "name": "May 2024 Gannon G5 storm",
        "date": "2024-05-10",
        "description": (
            "The 2024-05-10/11 G5 (extreme) Gannon storm from the AR13664 CME "
            "sequence; observed Kp 9, strongest storm since 2003."
        ),
        "start": "2024-05-09T00:00:00Z",
        "end": "2024-05-13T00:00:00Z",
    },
    "october_2024": {
        "name": "October 2024 G4 storm",
        "date": "2024-10-10",
        "description": "G4+ storm of 2024-10-10/11 (observed Kp 8.67).",
        "start": "2024-10-09T00:00:00Z",
        "end": "2024-10-13T00:00:00Z",
    },
}


class SpaceWeatherLoader(BaseDomainLoader):
    """Loader for geomagnetic-storm data (USGS magnetometer + DONKI Kp).

    Features are engineered from real 1-minute magnetometer data at the
    Boulder (BOU) reference observatory; labels come from DONKI GST
    observed planetary Kp windows at/above the G1 threshold.
    """

    DOMAIN: str = "space_weather"
    SOURCE_URL: str = _GEOMAG_URL
    # Labels = 3-hour windows with NOAA planetary Kp >= 5 (G1) from DONKI
    # GST. Kp is a thresholded disturbance index derived from ground
    # magnetometers — the same physics the dB/dt features score — so the
    # provenance is declared statistical (see loaders.label_provenance).
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False
    API_KEY_ENV_VAR: str = "NASA_API_KEY"

    #: Reference observatory for the feature substrate.
    OBSERVATORY: str = "BOU"

    FEATURE_COLUMNS: list[str] = [
        "dbdt_x",
        "dbdt_y",
        "dbdt_h",
        "dbdt_h_10min_mean",
        "h_range_60min",
        "h_dev_from_daily_median",
    ]

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent 2 hours of BOU minute magnetometer data.

        Returns:
            DataFrame with columns: time (epoch seconds), x, y, z, f (nT).

        Raises:
            ConnectionError: If the USGS web service is unreachable.
        """
        end = datetime.now(UTC).replace(second=0, microsecond=0)
        start = end - timedelta(hours=2)
        payload = self._fetch_json(_GEOMAG_URL, params=self._geomag_params(start, end))
        df = self._geomag_to_dataframe(payload)
        logger.info("Fetched %d real-time magnetometer records from USGS.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch minute magnetometer data spanning a cataloged storm.

        Args:
            event_id: Key into the ground-truth catalog
                (e.g. ``"gannon_may_2024"``).

        Returns:
            DataFrame with columns: time (epoch seconds), x, y, z, f (nT),
            sorted chronologically.

        Raises:
            ValueError: If *event_id* is unknown or the service returns an
                empty series for a cataloged storm.
            ConnectionError: If the USGS web service is unreachable.
        """
        event = self._event(event_id)
        cache_key = f"space_weather_geomag_{event_id}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.debug("Returning cached magnetometer data for '%s'.", event_id)
            return pd.DataFrame(cached)

        start = _parse_utc(event["start"])
        end = _parse_utc(event["end"])
        payload = self._fetch_json(_GEOMAG_URL, params=self._geomag_params(start, end))
        df = self._geomag_to_dataframe(payload)
        if df.empty or not np.isfinite(df[["x", "y"]].to_numpy()).any():
            raise ValueError(
                f"USGS geomag returned no usable {self.OBSERVATORY} minute data "
                f"for cataloged storm '{event_id}' ({event['start']}..{event['end']}); "
                "refusing to fabricate a series."
            )
        df = df.sort_values("time").reset_index(drop=True)
        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info("Fetched %d magnetometer records for event '%s'.", len(df), event_id)
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground-truth storm events.

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
        """Label each minute sample by DONKI GST observed-Kp storm windows.

        A sample is anomalous (1) when it falls inside a 3-hour planetary
        Kp window at or above :data:`KP_STORM_THRESHOLD` (Kp 5 = NOAA G1)
        reported by a DONKI GST record. DONKI tags each ``allKpIndex``
        entry with the *end* of its 3-hour synoptic window, so the window
        spans ``[observedTime - 3 h, observedTime)``.

        Args:
            event_id: Key into the ground-truth catalog.

        Returns:
            1-D binary array aligned with :meth:`fetch_historical` rows.

        Raises:
            ValueError: If *event_id* is unknown, or DONKI reports no
                storm-level Kp window for a cataloged storm (upstream
                failure — these events are documented G4/G5 storms).
        """
        event = self._event(event_id)
        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        windows = self._storm_windows(event_id, event)
        if not windows:
            raise ValueError(
                f"DONKI GST returned no Kp >= {KP_STORM_THRESHOLD} windows for "
                f"cataloged storm '{event_id}'; refusing to emit all-quiet labels "
                "for a documented storm."
            )

        times = df["time"].to_numpy(dtype=np.float64)
        labels = np.zeros(len(times), dtype=np.int64)
        for window_start, window_end in windows:
            labels |= (times >= window_start) & (times < window_end)
        logger.info(
            "Ground truth for '%s': %d anomalies / %d total (%d Kp>=%.0f windows).",
            event_id,
            int(labels.sum()),
            len(labels),
            len(windows),
            KP_STORM_THRESHOLD,
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Engineer storm-relevant observables from magnetometer minutes.

        Features per minute sample:

        1. **dbdt_x** — X-component time derivative, nT/min.
        2. **dbdt_y** — Y-component time derivative, nT/min.
        3. **dbdt_h** — horizontal-vector derivative magnitude, nT/min
           (the GIC-relevant observable; Viljanen et al. 2001).
        4. **dbdt_h_10min_mean** — trailing 10-minute mean of (3)
           (sustained-disturbance measure).
        5. **h_range_60min** — trailing 60-minute max-min range of the
           horizontal intensity H = sqrt(X^2 + Y^2) (the K-index-style
           range observable).
        6. **h_dev_from_daily_median** — H minus its trailing 24-hour
           rolling median (storm-time depression/elevation proxy).

        Args:
            raw_data: DataFrame from :meth:`fetch_historical` or
                :meth:`fetch_realtime`.

        Returns:
            2-D array of shape ``(n_samples, 6)``.
        """
        if raw_data.empty:
            return np.empty((0, len(self.FEATURE_COLUMNS)), dtype=np.float64)

        df = raw_data.sort_values("time").reset_index(drop=True)
        time_s = df["time"].to_numpy(dtype=np.float64)
        x = df["x"].to_numpy(dtype=np.float64)
        y = df["y"].to_numpy(dtype=np.float64)

        dt_min = np.empty_like(time_s)
        dt_min[0] = np.nan
        dt_min[1:] = np.diff(time_s) / 60.0
        with np.errstate(invalid="ignore", divide="ignore"):
            dbdt_x = np.concatenate([[np.nan], np.diff(x)]) / dt_min
            dbdt_y = np.concatenate([[np.nan], np.diff(y)]) / dt_min
        dbdt_h = np.hypot(dbdt_x, dbdt_y)

        h = np.hypot(x, y)
        h_series = pd.Series(h)
        dbdt_h_series = pd.Series(dbdt_h)

        dbdt_h_10min = dbdt_h_series.rolling(10, min_periods=1).mean().to_numpy()
        h_range_60 = (
            h_series.rolling(60, min_periods=2).max() - h_series.rolling(60, min_periods=2).min()
        ).to_numpy()
        h_dev = (h_series - h_series.rolling(1440, min_periods=30).median()).to_numpy()

        features = np.column_stack([dbdt_x, dbdt_y, dbdt_h, dbdt_h_10min, h_range_60, h_dev])

        # Clean non-finite values with column medians (house convention).
        features = np.where(np.isinf(features), np.nan, features)
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                median_val = np.nanmedian(col)
                col[mask] = median_val if np.isfinite(median_val) else 0.0
        return features

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _event(self, event_id: str) -> dict[str, Any]:
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. Available: {list(_EVENT_CATALOG.keys())}"
            )
        return _EVENT_CATALOG[event_id]

    def _geomag_params(self, start: datetime, end: datetime) -> dict[str, str]:
        return {
            "id": self.OBSERVATORY,
            "elements": "X,Y,Z,F",
            "sampling_period": "60",
            "format": "json",
            "type": "adjusted",
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    @staticmethod
    def _geomag_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
        """Convert a USGS geomag web-service response to a DataFrame."""
        times = payload.get("times", [])
        values = payload.get("values", [])
        if not times or not values:
            return pd.DataFrame(columns=["time", "x", "y", "z", "f"])

        series: dict[str, list[Any]] = {
            str(elem.get("id", "")).lower(): elem.get("values", []) for elem in values
        }
        epoch = [datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for t in times]

        def _col(name: str) -> list[float]:
            vals = series.get(name, [None] * len(times))
            return [float(v) if v is not None else np.nan for v in vals]

        return pd.DataFrame(
            {
                "time": epoch,
                "x": _col("x"),
                "y": _col("y"),
                "z": _col("z"),
                "f": _col("f"),
            }
        )

    def _storm_windows(self, event_id: str, event: dict[str, Any]) -> list[tuple[float, float]]:
        """Kp >= threshold 3-hour windows (epoch seconds) from DONKI GST."""
        cache_key = f"space_weather_gst_{event_id}"
        gst_records = self._read_cache(cache_key)
        if gst_records is None:
            start = _parse_utc(event["start"]) - timedelta(days=1)
            end = _parse_utc(event["end"]) + timedelta(days=1)
            params = {
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": end.strftime("%Y-%m-%d"),
                "api_key": self._api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY"),
            }
            gst_records = self._fetch_json(_DONKI_GST_URL, params=params)
            self._write_cache(cache_key, gst_records)

        # Route both nesting levels through the shared shape-flip absorber:
        # DONKI has served array-of-objects to date, but a positional flip at
        # either level would otherwise raise AttributeError — the same outage
        # class as the SWPC ``KeyError: 1`` incident.
        windows: list[tuple[float, float]] = []
        for record in self._iter_feed_rows(
            gst_records or [], ("gstID", "startTime", "allKpIndex", "link")
        ):
            for entry in self._iter_feed_rows(
                record.get("allKpIndex") or [], ("observedTime", "kpIndex", "source")
            ):
                kp = entry.get("kpIndex")
                observed = entry.get("observedTime")
                if kp is None or observed is None:
                    continue
                if float(kp) < KP_STORM_THRESHOLD:
                    continue
                end_s = _parse_utc(str(observed)).timestamp()
                windows.append((end_s - 3.0 * 3600.0, end_s))
        return windows


def _parse_utc(value: str) -> datetime:
    """Parse an ISO timestamp (with optional Z suffix) to aware UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = ["KP_STORM_THRESHOLD", "SpaceWeatherLoader"]
