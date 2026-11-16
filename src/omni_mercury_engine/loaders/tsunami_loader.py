# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain loader for tsunami data from NOAA NDBC DART buoys.

Fetches real-time and historical bottom pressure recorder (BPR)
data from the Deep-ocean Assessment and Reporting of Tsunamis
(DART) network operated by NOAA's National Data Buoy Center.

Features engineered from raw BPR readings include tidal deviation,
rate of sea-level change, and rolling variability measures.  Ground
truth labels are derived from documented tsunami arrival windows
at specific DART stations.
"""

from __future__ import annotations

import gzip
import io
import logging
import zlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.exceptions import OfflineModeError, check_synthetic_allowed
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DART station registry
# ---------------------------------------------------------------------------

#: Key DART buoy station IDs used for tsunami monitoring.
DART_STATIONS: list[str] = [
    "46402",
    "46407",
    "46410",
    "46413",
    "51407",
    "32401",
    "32412",
    "21413",
    "21418",
    "52402",
]

#: NDBC's yearly archive of every DART station:
#: ``<station>t<year>.txt.gz``, the same ``YY MM DD hh mm ss T HEIGHT`` layout
#: as the real-time files. The archive holds the event-mode (1-minute and
#: 15-second) samples a buoy sends while a tsunami passes, so each catalogued
#: event below is a recorded series, not one that needs to be made up.
_HISTORICAL_URL = "https://www.ndbc.noaa.gov/data/historical/dart/"

#: ``HEIGHT`` value NDBC writes for a sample that was not received.
_MISSING_HEIGHT = 9999.0

#: Columns of a parsed DART file.
_DART_COLUMNS = ["timestamp", "bpr", "sample_type", "station_id"]

#: ``bpr_series_source`` values: where a historical frame's series came from.
SERIES_SOURCE_ARCHIVE = "ndbc_dart_archive"
SERIES_SOURCE_SYNTHETIC = "synthetic"

#: Fewest recorded samples an event span must hold to count as retrieved.
_MIN_RECORDED_SAMPLES = 30

#: Share of an event span's grid steps that must hold a recorded sample. The
#: three catalogued events hold 99% or more (measured against NDBC 2026-11-16).
_MIN_GRID_COVERAGE = 0.9

#: The grid a historical event is returned on: the network's standard
#: reporting cadence, which is also what ``fetch_realtime`` delivers.
_GRID = "15min"

#: Recorded baseline kept either side of the event window. The feature
#: windows are counted in samples (the detrending mean spans 60 of them, 15
#: hours at this cadence), so the baseline has to be longer than that.
_BASELINE = pd.Timedelta(days=2)

#: A water-column height outside this range is not a DART reading (the
#: deepest station sits under about 6 km of water).
_HEIGHT_RANGE_M = (1.0, 9000.0)

# ---------------------------------------------------------------------------
# Ground-truth event catalog
# ---------------------------------------------------------------------------

_EVENT_CATALOG: list[dict[str, Any]] = [
    {
        "event_id": "tohoku_2011",
        "name": "2011 Tohoku Earthquake Tsunami",
        "date": "2011-03-11",
        "description": (
            "Magnitude 9.1 earthquake off the Pacific coast of Tohoku, Japan. "
            "Generated a devastating tsunami with waves exceeding 40 m in some "
            "coastal areas.  DART station 21418 recorded major wave signatures."
        ),
        "station_id": "21418",
        "window_start": "2011-03-11T05:46:00Z",
        "window_end": "2011-03-11T12:00:00Z",
    },
    {
        "event_id": "chile_2010",
        "name": "2010 Chile Earthquake Tsunami",
        "date": "2010-02-27",
        "description": (
            "Magnitude 8.8 earthquake off the coast of central Chile. "
            "Generated a Pacific-wide tsunami.  DART station 32412 recorded "
            "significant pressure deviations."
        ),
        "station_id": "32412",
        "window_start": "2010-02-27T06:34:00Z",
        "window_end": "2010-02-27T14:00:00Z",
    },
    {
        "event_id": "tonga_2022",
        "name": "2022 Tonga Eruption Tsunami",
        "date": "2022-01-15",
        "description": (
            "Submarine eruption of Hunga Tonga-Hunga Ha'apai volcano generated "
            "a trans-oceanic tsunami and atmospheric pressure wave.  DART stations "
            "51407 and 52402 captured the event."
        ),
        "station_id": "51407",
        "secondary_station_id": "52402",
        "window_start": "2022-01-15T04:15:00Z",
        "window_end": "2022-01-15T16:00:00Z",
    },
]

# Lookup for fast access by event_id.
_EVENTS_BY_ID: dict[str, dict[str, Any]] = {e["event_id"]: e for e in _EVENT_CATALOG}

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------

#: Window size (number of time steps) for the rolling standard deviation.
_ROLLING_STD_WINDOW: int = 20

#: Window size for the simple moving average used for tidal detrending.
_SMA_DETREND_WINDOW: int = 60


class TsunamiLoader(BaseDomainLoader):
    """Domain loader for NOAA NDBC DART tsunami buoy data.

    Connects to the NDBC real-time data service and retrieves bottom
    pressure recorder (BPR) readings from the DART network.  Historical
    ground-truth events are read from NDBC's yearly DART archive: the
    series each buoy recorded, labelled by the documented event window.

    Attributes:
        DOMAIN: ``"tsunami"``
        SOURCE_URL: NDBC real-time data endpoint.
        REQUIRES_API_KEY: ``False`` -- NDBC data is freely available.
    """

    DOMAIN: str = "tsunami"
    SOURCE_URL: str = "https://www.ndbc.noaa.gov/data/realtime2/"
    # The BPR series is the recorded one (NDBC yearly DART archive). The
    # labels are a catalogued window per event, from the origin time to a
    # fixed end, not a per-sample annotation of when the wave was at the
    # buoy, so they stay outside the supervised headline: ``statistical``
    # is the registry's class for labels that are not independent per-row
    # ground truth.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "bpr",
        "tidal_deviation",
        "abs_deviation",
        "rate_of_change",
        "rolling_std",
        "short_energy",
    ]

    # Cache historical event data for 24 hours (events are static).
    CACHE_TTL: int = 86400

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent real-time DART data for all key stations.

        Iterates over :pydata:`DART_STATIONS`, downloads the current
        ``.dart`` file from NDBC and concatenates the results into a
        single :class:`~pandas.DataFrame`.

        Returns:
            DataFrame with columns ``station_id``, ``timestamp``, and
            the raw measurement columns provided by NDBC.

        Raises:
            ConnectionError: If no station data could be retrieved after
                exhausting retries.
        """
        frames: list[pd.DataFrame] = []
        errors: list[str] = []

        for station_id in DART_STATIONS:
            try:
                df = self._fetch_station(station_id)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                msg = f"Station {station_id}: {exc}"
                logger.warning("tsunami: failed to fetch %s", msg)
                errors.append(msg)

        if not frames:
            raise ConnectionError(
                f"tsunami: could not retrieve data from any DART station. " f"Errors: {errors}"
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "tsunami: fetched real-time data — %d rows from %d stations",
            len(combined),
            len(frames),
        )
        return combined

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch the recorded DART series for a historical tsunami event.

        Reads the event station's year from NDBC's DART archive and returns
        two days either side of the event window on a 15-minute grid, the
        network's standard cadence. Each step is the mean of the samples
        recorded in it; a step with none is interpolated in time and
        ``measured`` is False for it.

        A buoy that detects a wave switches itself to event mode and reports
        every minute. A finer grid would therefore be dense inside the
        window and interpolated outside it, and features computed on it
        would separate the two by sampling density: the buoy's own detection,
        not the water. Measured on the three events, the bare ``measured``
        flag of a 1-minute grid scored AUROC 0.60 to 0.67 against the labels.
        At 15 minutes every step has a recorded sample in either mode.

        Args:
            event_id: One of ``"tohoku_2011"``, ``"chile_2010"``, or
                ``"tonga_2022"``.

        Returns:
            DataFrame with columns ``timestamp``, ``bpr`` (water-column
            height, metres), ``station_id``, ``measured`` and
            ``bpr_series_source``.

        Raises:
            ValueError: If *event_id* is not recognised.
            OfflineModeError: If ``MERCURY_OFFLINE`` is set and the event is
                not cached.
            DataSourceUnavailableError: If the archive cannot be read and the
                synthetic opt-in (``MERCURY_ALLOW_SYNTHETIC=1``) is not set.
                With the opt-in a synthetic stand-in is returned instead,
                carrying the synthetic mark (:meth:`is_synthetic`) and
                ``bpr_series_source == "synthetic"``; it is never cached.
        """
        if event_id not in _EVENTS_BY_ID:
            raise ValueError(f"Unknown event_id {event_id!r}. " f"Available: {list(_EVENTS_BY_ID)}")

        # v2 key: frames cached under the old key were synthesised and carry
        # no mark, so they must not be served as the recorded series.
        cache_key = f"tsunami_historical_v3_{event_id}"
        cached = self._read_frame_cache(cache_key)
        if cached is not None:
            return cached

        event = _EVENTS_BY_ID[event_id]
        try:
            df = self._fetch_event_archive(event)
        except OfflineModeError:
            raise
        except (ConnectionError, OSError, ValueError) as exc:
            # This used to be the only path: every historical frame was
            # synthesised, unmarked, whether or not NDBC was reachable.
            check_synthetic_allowed(
                "TsunamiLoader",
                f"NDBC DART archive ({_HISTORICAL_URL}) unavailable for event "
                f"'{event_id}': {type(exc).__name__}: {exc}",
            )
            df = self._synthesize_event(event)
            df["measured"] = False
            df["bpr_series_source"] = SERIES_SOURCE_SYNTHETIC
            return self.mark_synthetic(df)

        df["bpr_series_source"] = SERIES_SOURCE_ARCHIVE
        self._write_frame_cache(cache_key, df)
        logger.info(
            "tsunami: event %s from the NDBC archive, station %s: %d minutes, %d recorded",
            event_id,
            event["station_id"],
            len(df),
            int(df["measured"].sum()),
        )
        return df

    @staticmethod
    def _event_span(event: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The observation span of an event: its window plus baseline either side."""
        window_start = pd.Timestamp(event["window_start"])
        window_end = pd.Timestamp(event["window_end"])
        return window_start - _BASELINE, window_end + _BASELINE

    def _fetch_event_archive(self, event: dict[str, Any]) -> pd.DataFrame:
        """Read an event's recorded span from the NDBC DART archive.

        Args:
            event: Event dict from :pydata:`_EVENT_CATALOG`.

        Returns:
            15-minute frame with ``timestamp``, ``bpr``, ``station_id`` and
            ``measured``.

        Raises:
            ConnectionError: If the archive file cannot be downloaded.
            ValueError: If the file is not a readable gzip archive, or holds
                too few recorded samples in the event span to stand for the
                event.
        """
        station = str(event["station_id"])
        span_start, span_end = self._event_span(event)
        frames = []
        for year in range(span_start.year, span_end.year + 1):
            raw = self._fetch_url(f"{_HISTORICAL_URL}{station}t{year}.txt.gz")
            try:
                text = gzip.decompress(raw).decode("utf-8", errors="replace")
            except (EOFError, zlib.error) as exc:
                # A truncated or corrupt transfer. Neither exception is an
                # OSError, so they used to leave ``fetch_historical`` bare.
                raise ValueError(
                    f"DART archive for station {station}, {year} is not a complete "
                    f"gzip file ({type(exc).__name__}: {exc})"
                ) from exc
            frames.append(self._parse_dart_text(text, station))
        recorded = pd.concat(frames, ignore_index=True)
        in_span = recorded[
            (recorded["timestamp"] >= span_start) & (recorded["timestamp"] <= span_end)
        ]
        if len(in_span) < _MIN_RECORDED_SAMPLES:
            raise ValueError(
                f"station {station} has {len(in_span)} recorded samples between "
                f"{span_start.isoformat()} and {span_end.isoformat()}; "
                f"at least {_MIN_RECORDED_SAMPLES} are needed"
            )

        per_step = in_span.set_index("timestamp")["bpr"].resample(_GRID).mean()
        grid = pd.date_range(span_start.ceil(_GRID), span_end.floor(_GRID), freq=_GRID)
        on_grid = per_step.reindex(grid)
        measured_share = float(on_grid.notna().mean()) if len(grid) else 0.0
        if measured_share < _MIN_GRID_COVERAGE:
            # Interpolating most of a span would present a drawn line as the
            # buoy's record.
            raise ValueError(
                f"station {station} recorded {measured_share:.0%} of the "
                f"{len(grid)} {_GRID} steps between {span_start.isoformat()} and "
                f"{span_end.isoformat()}; at least {_MIN_GRID_COVERAGE:.0%} are needed"
            )
        return pd.DataFrame(
            {
                "timestamp": grid,
                "bpr": on_grid.interpolate(method="time", limit_direction="both").to_numpy(),
                "station_id": station,
                "measured": on_grid.notna().to_numpy(),
            }
        )

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground-truth tsunami events.

        Returns:
            List of dicts, each containing at least ``event_id``,
            ``name``, ``date``, and ``description``.
        """
        return [
            {
                "event_id": e["event_id"],
                "name": e["name"],
                "date": e["date"],
                "description": e["description"],
            }
            for e in _EVENT_CATALOG
        ]

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Return binary anomaly labels for a historical event.

        Time steps falling within the documented tsunami arrival window
        are labeled ``1`` (anomaly); all other time steps are ``0``
        (normal tide).

        Args:
            event_id: Identifier for the event.

        Returns:
            1-D numpy array of shape ``(n_samples,)`` with binary labels.

        Raises:
            ValueError: If *event_id* is not recognised.
        """
        if event_id not in _EVENTS_BY_ID:
            raise ValueError(f"Unknown event_id {event_id!r}. " f"Available: {list(_EVENTS_BY_ID)}")

        event = _EVENTS_BY_ID[event_id]
        df = self.fetch_historical(event_id)

        window_start = datetime.fromisoformat(event["window_start"].replace("Z", "+00:00"))
        window_end = datetime.fromisoformat(event["window_end"].replace("Z", "+00:00"))

        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        labels = np.where(
            (timestamps >= window_start) & (timestamps <= window_end),
            1,
            0,
        ).astype(np.intp)

        logger.info(
            "tsunami: ground truth for %s — %d anomaly / %d total",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw DART data into a feature matrix.

        Engineered features (per time step):

        1. **BPR reading** -- raw bottom pressure recorder value.
        2. **Tidal deviation** -- residual after removing a simple
           moving average (proxy for predicted tide).
        3. **Absolute deviation** -- magnitude of tidal deviation,
           emphasises departure regardless of sign.
        4. **Rate of change** -- first-order difference of BPR between
           consecutive time steps.
        5. **Rolling standard deviation** -- local variability measure
           over a 20-sample sliding window.
        6. **Short-window energy** -- rolling standard deviation over
           a 5-sample window, capturing rapid oscillations typical of
           tsunami arrivals.

        Args:
            raw_data: DataFrame returned by :meth:`fetch_realtime` or
                :meth:`fetch_historical`.  Must contain a ``bpr``
                column with numeric pressure values.

        Returns:
            2-D numpy array of shape ``(n_samples, 6)``.
        """
        if "bpr" not in raw_data.columns:
            # Fall back to base class behaviour for non-standard data.
            return super().engineer_features(raw_data)

        if "station_id" in raw_data.columns and raw_data["station_id"].nunique() > 1:
            # A real-time frame holds every station, one after another. Each
            # buoy sits at its own depth (3,265 m to 5,886 m in one live
            # fetch), so differencing or averaging across the seam between
            # two stations turned a change of buoy into a kilometre-high
            # "wave". Features are computed per station and returned in the
            # frame's own row order.
            features = np.zeros((len(raw_data), len(self.FEATURE_COLUMNS)), dtype=np.float64)
            positions = np.arange(len(raw_data))
            for _, rows in raw_data.assign(_position=positions).groupby("station_id", sort=False):
                features[rows["_position"].to_numpy()] = self.engineer_features(
                    rows.drop(columns=["station_id", "_position"])
                )
            return features

        bpr = raw_data["bpr"].astype(np.float64).values.copy()

        # Replace non-finite values with forward-fill then backward-fill.
        bpr = _fill_non_finite(bpr)

        # Feature 1: raw BPR
        feat_bpr = bpr.copy()

        # Feature 2: deviation from simple moving average (tidal detrend)
        sma = _rolling_mean(bpr, _SMA_DETREND_WINDOW)
        feat_deviation = bpr - sma

        # Feature 3: absolute deviation (magnitude of departure)
        feat_abs_dev = np.abs(feat_deviation)

        # Feature 4: rate of sea-level change (first difference)
        feat_rate = np.diff(bpr, prepend=bpr[0])

        # Feature 5: rolling standard deviation (20-sample window)
        feat_rolling_std = _rolling_std(bpr, _ROLLING_STD_WINDOW)

        # Feature 6: short-window energy (5-sample rolling std)
        feat_short_energy = _rolling_std(bpr, 5)

        features = np.column_stack(
            [
                feat_bpr,
                feat_deviation,
                feat_abs_dev,
                feat_rate,
                feat_rolling_std,
                feat_short_energy,
            ]
        )

        # Final cleanup: replace any remaining inf/nan with 0.
        features = np.where(np.isfinite(features), features, 0.0)
        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_station(self, station_id: str) -> pd.DataFrame:
        """Download and parse DART data for a single station.

        Args:
            station_id: NDBC station identifier (e.g. ``"21418"``).

        Returns:
            DataFrame with parsed columns including ``timestamp``,
            ``bpr``, and ``station_id``.
        """
        url = f"{self.SOURCE_URL}{station_id}.dart"
        raw_bytes = self._fetch_url(url)
        df = self._parse_dart_text(raw_bytes.decode("utf-8", errors="replace"), station_id)
        if df.empty:
            logger.warning("tsunami: empty data for station %s", station_id)
        return df

    @staticmethod
    def _parse_dart_text(text: str, station_id: str) -> pd.DataFrame:
        """Parse an NDBC DART file (real-time or archive) into recorded samples.

        The layout is ``YY MM DD hh mm ss T HEIGHT``: a timestamp to the
        second, the measurement type (1 = 15-minute, 2 = 1-minute, 3 =
        15-second) and the water-column height in metres. The previous
        parser assumed there was no seconds column, which shifted every
        field after the minute by one: ``bpr`` held the type code (1, 2 or
        3) on every live row, the height was left in an unnamed column, and
        15-second samples shared one timestamp.

        Args:
            text: Decoded file contents; ``#`` lines are headers.
            station_id: Station identifier for tagging.

        Returns:
            DataFrame with ``timestamp`` (UTC), ``bpr`` (height, metres),
            ``sample_type`` and ``station_id``, oldest first, without the
            samples NDBC marks as not received and without rows that cannot
            be read. Empty when the file has no data rows.

        Raises:
            ValueError: If the rows do not have the eight DART columns.
        """
        try:
            frame = pd.read_csv(
                io.StringIO(text),
                comment="#",
                sep=r"\s+",
                header=None,
                dtype=str,
                on_bad_lines="skip",
            )
        except pd.errors.EmptyDataError:
            # A file with headers and no rows: a buoy with nothing to report.
            return pd.DataFrame(columns=_DART_COLUMNS)
        return TsunamiLoader._dart_samples(frame, station_id)

    @staticmethod
    def _dart_samples(frame: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """Turn a headerless DART table into recorded samples.

        Args:
            frame: One row per line of the file, eight columns (or seven,
                the layout without seconds that the previous parser assumed).
            station_id: Station identifier for tagging.

        Returns:
            The frame :meth:`_parse_dart_text` documents.

        Raises:
            ValueError: If the table has neither seven nor eight columns.
        """
        if frame.empty:
            return pd.DataFrame(columns=_DART_COLUMNS)
        if frame.shape[1] == 7:
            frame = frame.copy()
            frame.insert(5, "second", "0")
        if frame.shape[1] != 8:
            raise ValueError(
                f"DART file for station {station_id} has {frame.shape[1]} columns; "
                "expected 8 (YY MM DD hh mm ss T HEIGHT)"
            )
        frame = frame.set_axis(
            ["year", "month", "day", "hour", "minute", "second", "sample_type", "bpr"], axis=1
        )
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        timestamps = pd.to_datetime(
            numeric[["year", "month", "day", "hour", "minute", "second"]],
            utc=True,
            errors="coerce",
        )
        parsed = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bpr": numeric["bpr"],
                "sample_type": numeric["sample_type"],
                "station_id": station_id,
            }
        )
        # One unreadable cell costs its own row, never the file: a row with a
        # bad date, a non-numeric type or height, NDBC's 9999.0 placeholder,
        # or a height no buoy sits under is dropped.
        low, high = _HEIGHT_RANGE_M
        readable = (
            parsed["timestamp"].notna()
            & parsed["sample_type"].notna()
            & parsed["bpr"].between(low, high)
        )
        parsed = parsed[readable].astype({"sample_type": "int64"})
        return parsed.sort_values("timestamp", kind="stable").reset_index(drop=True)

    @staticmethod
    def _parse_dart_columns(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """Name the columns of an already-read DART table (kept for callers).

        Args:
            df: Headerless frame read from an NDBC DART file, with eight
                columns or the seven of the layout without seconds.
            station_id: Station identifier for tagging.

        Returns:
            The frame :meth:`_parse_dart_text` returns for the same rows.
        """
        return TsunamiLoader._dart_samples(df, station_id)

    @staticmethod
    def _synthesize_event(event: dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic BPR data mimicking a tsunami event.

        The synthetic time series contains a calm tidal background with
        superimposed tsunami-like oscillations during the event window.
        Statistical parameters are tuned per event to approximate
        published observations.

        Args:
            event: Event dict from :pydata:`_EVENT_CATALOG`.

        Returns:
            DataFrame with ``timestamp``, ``bpr``, and ``station_id``.
        """
        # Use a deterministic seed (Python's hash() is randomized per
        # process).  hashlib.sha256 produces the same bytes every time.
        import hashlib

        seed_bytes = hashlib.sha256(event["event_id"].encode()).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "little") % (2**31)
        rng = np.random.default_rng(seed_int)

        window_start = datetime.fromisoformat(event["window_start"].replace("Z", "+00:00"))
        window_end = datetime.fromisoformat(event["window_end"].replace("Z", "+00:00"))

        # Scale observation window so the anomaly ratio stays below ~35%.
        # Long-duration events (>8 h) need proportionally more padding to
        # provide sufficient normal baseline for unsupervised detectors.
        event_duration_s = window_end.timestamp() - window_start.timestamp()
        padding_mult = 1.0 if event_duration_s > 8 * 3600 else 0.75
        pre_seconds = max(6 * 3600, int(event_duration_s * padding_mult))
        post_seconds = max(6 * 3600, int(event_duration_s * padding_mult))
        total_start = window_start.timestamp() - pre_seconds
        total_end = window_end.timestamp() + post_seconds
        step_seconds = 60  # 1-minute resolution

        time_array = np.arange(total_start, total_end, step_seconds)
        n_samples = len(time_array)

        # Background tide: slow sinusoidal (~12.42 hour period for M2 tidal constituent).
        tidal_period = 12.42 * 3600  # seconds
        baseline_pressure = 5000.0  # nominal deep-ocean BPR in dbar
        tidal_amplitude = 0.5  # typical deep-ocean tidal amplitude (dbar)

        tide = baseline_pressure + tidal_amplitude * np.sin(2 * np.pi * time_array / tidal_period)

        # Noise floor.
        noise = rng.normal(0, 0.005, size=n_samples)

        # Tsunami signal: applied only within the event window.
        tsunami_signal = np.zeros(n_samples)
        ws_epoch = window_start.timestamp()
        we_epoch = window_end.timestamp()
        in_window = (time_array >= ws_epoch) & (time_array <= we_epoch)

        if event["event_id"] == "tohoku_2011":
            amplitude = 0.15  # large signal (~15 cm water equiv.)
            period = 20 * 60  # ~20-minute dominant period
            decay_rate = 2.0
        elif event["event_id"] == "chile_2010":
            # Pacific-wide tsunami with sustained oscillations over ~7 h.
            amplitude = 0.08
            period = 25 * 60
            decay_rate = 1.5
        else:  # tonga_2022
            # Hunga Tonga eruption: VEI-5+ submarine volcanic event.
            # Pressure wave sustained oscillations over ~12 h across
            # the Pacific, with slower decay than seismic tsunamis.
            amplitude = 0.08
            period = 15 * 60
            decay_rate = 1.5

        # Decaying oscillation within the window.
        window_time = time_array[in_window] - ws_epoch
        window_duration = we_epoch - ws_epoch
        decay = np.exp(-decay_rate * window_time / window_duration)
        tsunami_signal[in_window] = amplitude * decay * np.sin(2 * np.pi * window_time / period)

        bpr = tide + tsunami_signal + noise

        timestamps = pd.to_datetime(time_array, unit="s", utc=True)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bpr": bpr,
                "station_id": event["station_id"],
            }
        )
        return df


# ---------------------------------------------------------------------------
# Pure-numpy helper functions (no sklearn dependency)
# ---------------------------------------------------------------------------


def _rolling_mean(arr: np.ndarray[Any, Any], window: int) -> np.ndarray[Any, Any]:
    """Compute a rolling mean over *arr* with the given *window* size.

    Edge values are filled with the nearest valid mean to avoid NaNs.

    Args:
        arr: 1-D numpy array.
        window: Rolling window size (number of samples).

    Returns:
        1-D numpy array of the same length as *arr*.
    """
    if window < 1:
        return arr.copy()
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0.0)
    result = np.empty_like(arr)
    half = window // 2

    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + window - half)
        result[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return result


def _rolling_std(arr: np.ndarray[Any, Any], window: int) -> np.ndarray[Any, Any]:
    """Compute a rolling standard deviation over *arr*.

    Uses the two-pass algorithm (mean then variance) for numerical
    stability.  Edge samples use a shrunk window.

    Args:
        arr: 1-D numpy array.
        window: Rolling window size (number of samples).

    Returns:
        1-D numpy array of the same length as *arr*.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + window - half)
        segment = arr[lo:hi]
        result[i] = np.std(segment) if len(segment) > 1 else 0.0

    return result


def _fill_non_finite(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Replace non-finite values with forward-fill then backward-fill.

    If the entire array is non-finite, fills with ``0.0``.

    Args:
        arr: 1-D numpy array (modified in place and returned).

    Returns:
        The same array with non-finite values replaced.
    """
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr

    # Forward fill.
    last_valid = np.nan
    for i in range(len(arr)):
        if mask[i]:
            if np.isfinite(last_valid):
                arr[i] = last_valid
        else:
            last_valid = arr[i]

    # Backward fill any remaining leading NaNs.
    first_valid = np.nan
    for i in range(len(arr) - 1, -1, -1):
        if np.isfinite(arr[i]):
            first_valid = arr[i]
            break

    if np.isfinite(first_valid):
        for i in range(len(arr)):
            if not np.isfinite(arr[i]):
                arr[i] = first_valid
            else:
                break
    else:
        # Entire array was non-finite.
        arr[:] = 0.0

    return arr
