# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Surges in FEMA's disaster-declaration stream, labelled by NOAA.

:class:`~omni_mercury_engine.loaders.fema_loader.FEMALoader` labels each
declaration by its own programme flags ("major disaster with individual and
public assistance"). That label is an administrative category, not an anomaly:
a third of the rows carry it, and no column that is independent of it tracks
it in a consistent direction (measured on its five events 2026-11-16:
designated areas rank it at AUROC 0.71 on one event and 0.23 on another).

This loader poses the question a monitor of that stream actually asks: *is
declaration activity surging?* Each event is one calendar year of tropical
cyclone declarations, one row per day. The labels come from outside FEMA's
record: NOAA NCEI's Billion-Dollar Weather and Climate Disasters catalogue. A
day is labelled when it falls between the first day of a billion-dollar
tropical cyclone and thirty days after its last, thirty days being the period
a governor has to request a major disaster declaration (44 CFR 206.36(a)).

Data sources:
    OpenFEMA ``DisasterDeclarationsSummaries`` (public, keyless) and the NOAA
    NCEI billion-dollar disaster event list (public, keyless, 1980 to 2024).
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError, OfflineModeError
from omni_mercury_engine.loaders.fema_loader import _DECLARATIONS_URL, FEMALoader

logger = logging.getLogger(__name__)

__all__ = ["NOAA_WINDOW_COLUMN", "FEMAActivityLoader"]

#: NOAA NCEI Billion-Dollar Weather and Climate Disasters, event list.
_NOAA_EVENTS_URL = "https://www.ncei.noaa.gov/access/billions/events-US-1980-2024.csv"
#: The catalogue's name for hurricanes, tropical storms and typhoons.
_NOAA_TROPICAL_CYCLONE = "Tropical Cyclone"
#: Last year the catalogue covers.
_NOAA_LAST_YEAR = 2024

#: FEMA incident types that are tropical cyclones.
_FEMA_TROPICAL_FILTER = (
    "(incidentType eq 'Hurricane' or incidentType eq 'Tropical Storm' "
    "or incidentType eq 'Typhoon')"
)

#: Days a governor has to request a major disaster declaration after the
#: incident (44 CFR 206.36(a)); declarations for a storm arrive in this period.
REQUEST_WINDOW_DAYS = 30

#: Column holding the NOAA-derived label in a historical frame.
NOAA_WINDOW_COLUMN = "noaa_billion_dollar_window"

_PAGE_SIZE = 1000
_MAX_RECORDS = 20000

#: Years offered as events: the catalogue's last eight.
_EVENT_YEARS = tuple(range(2017, _NOAA_LAST_YEAR + 1))


def _event_id(year: int) -> str:
    return f"tropical_declarations_{year}"


class FEMAActivityLoader(FEMALoader):
    """Daily tropical-cyclone declaration activity, labelled by NOAA's catalogue.

    One event per calendar year from 2017 to 2024. A historical frame has one
    row per day with the count of county designations declared that day, the
    number of states and of disasters they belong to, and the NOAA label.
    """

    DOMAIN: str = "fema_activity"
    SOURCE_URL: str = _DECLARATIONS_URL
    # Labels are NOAA NCEI's billion-dollar disaster catalogue: an expert
    # assessment made outside FEMA's record, from no column this loader scores.
    LABEL_SOURCE: str = "expert_annotated"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        "designations",
        "states",
        "disasters",
        "designations_7d",
        "states_7d_max",
    ]
    LABEL_DERIVED_FEATURES: tuple[str, ...] = ()
    LABEL_FEATURE_DISJOINT: bool = True

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def list_events(self) -> list[dict[str, Any]]:
        """One event per calendar year the NOAA catalogue covers, 2017 onward."""
        return [
            {
                "event_id": _event_id(year),
                "name": f"Tropical cyclone declarations, {year}",
                "date": f"{year}-01-01",
                "description": (
                    f"Daily FEMA county designations for hurricanes, tropical storms and "
                    f"typhoons in {year}, labelled by NOAA NCEI's billion-dollar tropical "
                    f"cyclones of that year plus the {REQUEST_WINDOW_DAYS}-day request window."
                ),
            }
            for year in _EVENT_YEARS
        ]

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """One year of daily declaration activity with its NOAA labels.

        Args:
            event_id: ``tropical_declarations_<year>``.

        Returns:
            DataFrame with ``timestamp`` (UTC day), ``designations``,
            ``states``, ``disasters`` and :data:`NOAA_WINDOW_COLUMN`.

        Raises:
            ValueError: If *event_id* is not offered.
            DataSourceUnavailableError: If OpenFEMA or the NOAA catalogue
                cannot be read. There is no synthetic stand-in.
        """
        year = self._year_of(event_id)
        cache_key = f"fema_activity_v1_{event_id}"
        cached = self._read_frame_cache(cache_key)
        if cached is not None:
            return cached
        try:
            declarations = self._fetch_tropical_declarations(year)
            storms = self.fetch_billion_dollar_events(_NOAA_TROPICAL_CYCLONE)
        except OfflineModeError:
            raise
        except (ConnectionError, OSError, ValueError) as exc:
            raise DataSourceUnavailableError(
                loader_name=type(self).__name__,
                reason=(
                    f"OpenFEMA or the NOAA billion-dollar catalogue is unavailable for "
                    f"'{event_id}': {type(exc).__name__}: {exc}"
                ),
            ) from exc
        frame = self._daily_activity(declarations, year)
        frame[NOAA_WINDOW_COLUMN] = self._window_labels(frame["timestamp"], storms)
        self._write_frame_cache(cache_key, frame)
        return frame

    def fetch_realtime(self, days_before_year: int = 0) -> pd.DataFrame:
        """Daily tropical-cyclone declaration activity for the current year so far.

        Args:
            days_before_year: While the year is younger than this many days,
                the last that-many days of the year before are included. A
                reader that looks back a few days (the watch) otherwise sees
                nothing of late December from 1 January: a declaration of
                30 December vanished from its record on New Year's Day.

        Returns:
            DataFrame with ``timestamp``, ``designations``, ``states`` and
            ``disasters``; no labels (the catalogue is not live).
        """
        today = pd.Timestamp.now(tz="UTC").floor("D")
        declarations = self._fetch_tropical_declarations(today.year)
        frame = self._daily_activity(declarations, today.year)
        frame = frame[frame["timestamp"] <= today]
        if days_before_year > 0 and today.dayofyear <= days_before_year:
            before = self._daily_activity(
                self._fetch_tropical_declarations(today.year - 1), today.year - 1
            )
            frame = pd.concat([before.tail(days_before_year), frame])
        return frame.reset_index(drop=True)

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Days inside a billion-dollar tropical cyclone's window, as NOAA lists them.

        Args:
            event_id: ``tropical_declarations_<year>``.

        Returns:
            1-D binary array, one entry per day of the year.
        """
        frame = self.fetch_historical(event_id)
        return np.asarray(frame[NOAA_WINDOW_COLUMN], dtype=np.int64)

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Feature matrix of a daily activity frame.

        Columns, in :attr:`FEATURE_COLUMNS` order: the day's designations,
        states and disasters, the designations of the trailing seven days,
        and the most states declared on one day in the trailing seven.

        Args:
            raw_data: Frame from :meth:`fetch_historical` or
                :meth:`fetch_realtime`.

        Returns:
            2-D array of shape ``(n_days, 5)``.
        """
        if raw_data.empty:
            return np.empty((0, len(self.FEATURE_COLUMNS)), dtype=np.float64)
        frame = raw_data.sort_values("timestamp").reset_index(drop=True)
        designations = frame["designations"].astype(float)
        states = frame["states"].astype(float)
        return np.column_stack(
            [
                designations.to_numpy(),
                states.to_numpy(),
                frame["disasters"].astype(float).to_numpy(),
                designations.rolling(7, min_periods=1).sum().to_numpy(),
                states.rolling(7, min_periods=1).max().to_numpy(),
            ]
        )

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def fetch_billion_dollar_events(self, disaster: str | None = None) -> pd.DataFrame:
        """NOAA NCEI's billion-dollar disaster events, optionally of one kind.

        Args:
            disaster: The catalogue's disaster kind (``"Tropical Cyclone"``,
                ``"Flooding"``, ``"Severe Storm"``, ...); ``None`` for all.

        Returns:
            One row per event: ``name``, ``disaster``, ``began`` and
            ``ended`` (UTC days), ``cost_millions`` (CPI-adjusted) and
            ``deaths``.

        Raises:
            ConnectionError: If the catalogue cannot be downloaded.
            ValueError: If the file is not the catalogue's event list.
        """
        cache_key = "fema_activity_noaa_events_v1"
        events = self._read_frame_cache(cache_key)
        if events is None:
            events = self._parse_billion_dollar_events(
                self._fetch_url(_NOAA_EVENTS_URL).decode("utf-8", errors="replace")
            )
            self._write_frame_cache(cache_key, events)
        if disaster is not None:
            events = events[events["disaster"] == disaster].reset_index(drop=True)
        return events

    @staticmethod
    def _parse_billion_dollar_events(text: str) -> pd.DataFrame:
        """Parse the catalogue's CSV: two title lines, a header, one event per row."""
        lines = text.splitlines()
        header = next(
            (index for index, line in enumerate(lines) if line.startswith("Name,Disaster,")), None
        )
        if header is None:
            raise ValueError("not the NOAA billion-dollar event list: no 'Name,Disaster' header")
        rows: list[dict[str, Any]] = []
        for record in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
            began = pd.to_datetime(record.get("Begin Date"), format="%Y%m%d", errors="coerce")
            ended = pd.to_datetime(record.get("End Date"), format="%Y%m%d", errors="coerce")
            if pd.isna(began) or pd.isna(ended) or ended < began:
                continue
            rows.append(
                {
                    "name": str(record.get("Name") or ""),
                    "disaster": str(record.get("Disaster") or ""),
                    "began": began.tz_localize("UTC"),
                    "ended": ended.tz_localize("UTC"),
                    "cost_millions": pd.to_numeric(
                        record.get("CPI-Adjusted Cost"), errors="coerce"
                    ),
                    "deaths": pd.to_numeric(record.get("Deaths"), errors="coerce"),
                }
            )
        if not rows:
            raise ValueError("the NOAA billion-dollar event list holds no dated events")
        return pd.DataFrame(rows)

    def _fetch_tropical_declarations(self, year: int) -> pd.DataFrame:
        """Every tropical-cyclone county designation declared in one calendar year."""
        declared_in_year = (
            f"declarationDate ge '{year}-01-01T00:00:00.000z' "
            f"and declarationDate lt '{year + 1}-01-01T00:00:00.000z'"
        )
        records: list[dict[str, Any]] = []
        skip = 0
        while skip < _MAX_RECORDS:
            page = self._fetch_fema_json(
                _DECLARATIONS_URL,
                {
                    "$filter": f"{_FEMA_TROPICAL_FILTER} and {declared_in_year}",
                    "$skip": str(skip),
                    "$top": str(_PAGE_SIZE),
                    "$orderby": "declarationDate asc",
                },
            )
            batch = page.get("DisasterDeclarationsSummaries", [])
            if not isinstance(batch, list):
                raise ValueError("OpenFEMA response is not a list of declarations")
            records.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            skip += _PAGE_SIZE
        return self._records_to_dataframe(records)

    # ------------------------------------------------------------------
    # Shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _year_of(event_id: str) -> int:
        for year in _EVENT_YEARS:
            if event_id == _event_id(year):
                return year
        raise ValueError(
            f"Unknown event_id '{event_id}'. Available: {[_event_id(y) for y in _EVENT_YEARS]}"
        )

    @staticmethod
    def _daily_activity(declarations: pd.DataFrame, year: int) -> pd.DataFrame:
        """Count designations, states and disasters per day of ``year``.

        A day with no declaration is a row of zeros: quiet days are most of
        the record and are what a surge is measured against.
        """
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="UTC")
        if declarations.empty:
            counts = pd.DataFrame(0, index=days, columns=["designations", "states", "disasters"])
        else:
            declared = pd.to_datetime(
                declarations["declarationDate"], utc=True, errors="coerce"
            ).dt.floor("D")
            grouped = declarations.assign(_day=declared).dropna(subset=["_day"]).groupby("_day")
            counts = pd.DataFrame(
                {
                    "designations": grouped.size(),
                    "states": grouped["state"].nunique(),
                    "disasters": grouped["disasterNumber"].nunique(),
                }
            ).reindex(days, fill_value=0)
        counts = counts.astype(np.int64)
        counts.insert(0, "timestamp", days)
        return counts.reset_index(drop=True)

    @staticmethod
    def _window_labels(timestamps: pd.Series[Any], storms: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Label the days from a storm's first day to thirty days after its last."""
        days = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        labels = np.zeros(len(days), dtype=np.int64)
        window = pd.Timedelta(days=REQUEST_WINDOW_DAYS)
        for began, ended in zip(storms["began"], storms["ended"], strict=True):
            labels[(days >= began) & (days <= ended + window)] = 1
        return labels
