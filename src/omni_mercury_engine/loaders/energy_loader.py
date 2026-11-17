# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain loader for EMP/energy grid data from NOAA Space Weather and EIA.

Connects to the NOAA Space Weather Prediction Center (SWPC) for real-time and historical
geomagnetic/solar data, and optionally to the U.S. Energy Information Administration (EIA) API v2
for electricity grid data.

Ground truth events cover major geomagnetic storms and grid disruptions where severe Kp indices (>=
7) or grid supply/demand anomalies serve as anomaly labels for Mercury detection benchmarks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import zipfile
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import ParseError

import numpy as np
import pandas as pd
from defusedxml import ElementTree as SafeElementTree

from omni_mercury_engine.datasets.exceptions import OfflineModeError, check_synthetic_allowed
from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale
from omni_mercury_engine.loaders.base import (
    BaseDomainLoader,
    FetchHTTPError,
    SyntheticDataNotDeleakable,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOAA SWPC API endpoints
# ---------------------------------------------------------------------------
_KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
_XRAY_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
#: NOAA NCEI's yearly GOES XRS flare reports (1975 to 2017), one text file per year.
_FLARE_REPORT_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/"
    "x-rays/goes/xrs/goes-xrs-report_{year}.txt"
)
# SWPC retired the whole ``/products/solar-wind/`` tree (every ``plasma-*`` and
# ``mag-*`` file under it now answers 404); real-time solar wind plasma moved to
# the RTSW feed, which serves 1-minute records for the trailing 24 hours.
_SOLAR_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
# RTSW only reaches back 24 hours; ACE SWEPAM hourly supplies the rest of the
# 7-day Kp window that ``plasma-7-day`` used to cover on its own.
_ACE_SWEPAM_URL = "https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json"

# ---------------------------------------------------------------------------
# Recorded history
# ---------------------------------------------------------------------------
#: NASA/GSFC OMNI2: one file per year since 1963 of hourly near-Earth solar
#: wind and geomagnetic indices, merged from every spacecraft that measured
#: them. It holds each catalogued storm, so no historical event has to be
#: reconstructed.
_OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"

#: OMNI2 whitespace-token positions (0-based) and fill values, from the
#: format description beside the files (``omni2.text``): word 1 year, 2 day
#: of year, 3 hour, 24 proton density (N/cm^3), 25 bulk speed (km/s), 39
#: Kp * 10. The same positions ``ml/hazard_training/solar_storm.parse_omni2``
#: reads; ``tests/loaders/test_energy_recorded_series.py`` holds the two to
#: each other.
_OMNI2_COLUMNS = {"density": (23, 999.9), "speed": (24, 9999.0), "kp10": (38, 99.0)}

#: EIA-930, the Hourly Electric Grid Monitor: what every U.S. balancing
#: authority reported each hour (demand, day-ahead demand forecast, net
#: generation, interchange). The six-month balance files need no API key.
_EIA930_BALANCE_URL = (
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_{half}.csv"
)

#: Share of a grid event's hours EIA-930 must hold for the frame to be the
#: event's record. The files start a few hours into their first UTC day.
_EIA930_MIN_COVERAGE = 0.95

#: A grid step is labelled when served demand fell this far below the
#: day-ahead forecast. Measured on ERCOT, January to March 2021 (2,153
#: hours): the median shortfall is 0.1%, the 95th percentile 14.6%, and the
#: only hours above 20% are 2021-02-15 13:00 to 2021-02-18 14:00 UTC, the
#: load-shed period of Winter Storm Uri.
GRID_SHORTFALL_THRESHOLD = 0.20

#: DOE Form OE-417, "Electric Emergency Incident and Disturbance Report":
#: what utilities and grid operators must file with the Department of Energy
#: when service is lost or the bulk system is disturbed. Served by Oak Ridge
#: National Laboratory's Open Energy Hub. DOE's own site
#: (``oe.netl.doe.gov``) was unreachable when this was written, and the hub's
#: copy is the 2023 annual summary; the reader reports the years it finds, so
#: an event outside them is refused, not labelled empty.
_OE417_URL = (
    "https://openenergyhub.ornl.gov/api/explore/v2.1/catalog/datasets/"
    "oe-417-annual-summaries/records"
)
_OE417_PAGE = 100

#: EIA's Electric Power Monthly republished DOE's OE-417 reports as Appendix B
#: (Table B.1, the year to date; Table B.2, the whole previous year), with the
#: hour each disturbance began and was restored. The February issue of year
#: ``Y + 2`` holds year ``Y`` complete in Table B.2; EIA carried the appendix
#: through the February 2024 issue, so 2023 is the last complete year.
_EPM_ISSUE_URL = "https://www.eia.gov/electricity/monthly/archive/february{year}.zip"
#: Largest table accepted from an issue's archive (the real ones are ~35 kB).
_EPM_TABLE_MAX_BYTES = 5_000_000
_EPM_EVENT_TIME = "%m/%d/%Y %I:%M %p"
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

#: Name of the label column a DOE-labelled grid frame carries.
DOE_DISTURBANCE_COLUMN = "doe_disturbance"

#: The EIA-930 series a grid-event frame holds.
_GRID_SERIES_COLUMNS: tuple[str, ...] = (
    "grid_demand",
    "grid_demand_forecast",
    "grid_net_generation",
    "grid_interchange",
)

#: Columns of :meth:`EnergyLoader.engineer_grid_features`, in order.
GRID_FEATURE_COLUMNS: tuple[str, ...] = (
    "grid_demand",
    "grid_net_generation",
    "grid_interchange",
    "grid_forecast_error",
    "grid_demand_change",
    "grid_generation_change",
)

#: Grid features that are functions of demand or its forecast. A grid event
#: labelled by the shortfall of demand against forecast is not scored on them.
GRID_SHORTFALL_DERIVED_FEATURES: frozenset[str] = frozenset(
    {"grid_demand", "grid_forecast_error", "grid_demand_change"}
)

#: ``series_source`` values of a historical frame.
SERIES_SOURCE_OMNI2 = "nasa_omni2"
SERIES_SOURCE_EIA930 = "eia_930+nasa_omni2"
SERIES_SOURCE_SYNTHETIC = "synthetic"

#: The loader's cadence: Kp is a 3-hour index, and the rolling features count
#: steps of it (8 steps = 24 h).
_STEP = "3h"

# ---------------------------------------------------------------------------
# Solar flare classification mapping (X-ray peak flux class -> numeric)
# ---------------------------------------------------------------------------
_FLARE_CLASS_MAP: dict[str, float] = {
    "A": 1.0,
    "B": 2.0,
    "C": 3.0,
    "M": 4.0,
    "X": 5.0,
}

# ---------------------------------------------------------------------------
# Geomagnetic storm thresholds: NOAA G levels via hazard_metrics.noaa_g_scale
# (storm = G1, Kp 5- and up; severe = G3, Kp 7- and up). Literal integer Kp
# cutoffs (5, 7) dropped every "minus"-third storm hour.
# ---------------------------------------------------------------------------
_STORM_G_LEVEL = 1
_SEVERE_G_LEVEL = 3


def _g_levels(kp_values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """NOAA G level per Kp value (values outside [0, 9] are clipped first).

    A missing Kp has no G level and is 0 (no storm): a step with no reading
    cannot be labelled a storm, and ``noaa_g_scale`` refuses a NaN outright.
    """
    values = np.asarray(kp_values, dtype=np.float64)
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 9.0)
    levels = np.fromiter(
        (noaa_g_scale(float(k)) for k in clipped), dtype=np.int64, count=clipped.size
    )
    levels[np.isnan(values)] = 0
    return levels


# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "quebec_1989": {
        "name": "1989 Quebec Blackout",
        "date": "1989-03-13",
        "description": (
            "Geomagnetic storm caused by a coronal mass ejection collapsed "
            "the Hydro-Quebec power grid for 9 hours. Peak Kp=9. Extended "
            "to 60-day window for adequate sample size."
        ),
        "start": "1989-02-01",
        "end": "1989-04-30",
        "peak_kp": 9,
        "synthetic_hours": 2160,  # 90 days at 3h = 720 steps
        "storm_day": 40,  # day within window when storm peaks
    },
    "halloween_2003": {
        "name": "2003 Halloween Solar Storms",
        "date": "2003-10-28",
        "description": (
            "Series of solar flares and coronal mass ejections during "
            "October-November 2003. Extended to multi-month window."
        ),
        "start": "2003-09-01",
        "end": "2003-12-31",
        "peak_kp": 9,
        "synthetic_hours": 2928,  # 122 days
        "storm_day": 58,  # Oct 28 within window
    },
    "texas_2021": {
        "name": "2021 Texas Grid Crisis",
        "date": "2021-02-15",
        "description": (
            "Winter Storm Uri caused widespread power outages across Texas "
            "due to extreme cold and grid supply/demand imbalance. "
            "Grid demand/supply data anomaly (not geomagnetic)."
        ),
        "start": "2021-01-01",
        "end": "2021-03-31",
        "peak_kp": 2,
        "synthetic_hours": 2160,
        "grid_event": True,
        # EIA-930 code of the grid operator: ERCOT.
        "balancing_authority": "ERCO",
        # Labelled by the disturbances reported to DOE on Form OE-417 for the
        # Texas reliability region, in ERCOT's local time.
        "oe417_region": "TRE",
        "oe417_timezone": "America/Chicago",
    },
    "texas_2023": {
        "name": "2023 Texas Grid Disturbances (DOE OE-417)",
        "date": "2023-01-31",
        "description": (
            "ERCOT's recorded hours for 2023 (EIA-930), labelled by the "
            "disturbances utilities reported to the Department of Energy on "
            "Form OE-417 for the Texas reliability region: nine incidents "
            "with a stated loss of load or of customers, among them the "
            "January-February ice storm (360,000 customers) and the March "
            "windstorm (470,000). The labels come from DOE, not from any "
            "column the detector scores."
        ),
        "start": "2023-01-01",
        "end": "2023-12-31",
        "peak_kp": 0,
        "synthetic_hours": 8760,
        "grid_event": True,
        "balancing_authority": "ERCO",
        # NERC region code OE-417 files Texas incidents under.
        "oe417_region": "TRE",
        "oe417_timezone": "America/Chicago",
    },
    "bastille_day_2000": {
        "name": "2000 Bastille Day Solar Event",
        "date": "2000-07-14",
        "description": (
            "X5.7-class solar flare on Bastille Day followed by a severe "
            "geomagnetic storm. Peak Kp=9. Extended to multi-month window."
        ),
        "start": "2000-06-01",
        "end": "2000-08-31",
        "peak_kp": 9,
        "synthetic_hours": 2208,  # 92 days
        "storm_day": 44,
    },
}


def _safe_xml(document: bytes) -> Any:
    """Parse XML with the hardened parser (no entity expansion, no external DTD)."""
    return SafeElementTree.fromstring(document)


def _xlsx_column(reference: str) -> int:
    """Zero-based column of a cell reference: ``"A7"`` is 0, ``"K7"`` is 10."""
    column = 0
    for character in reference:
        if not character.isalpha():
            break
        column = column * 26 + (ord(character.upper()) - ord("A") + 1)
    return max(column - 1, 0)


class EnergyLoader(BaseDomainLoader):
    """Loader for EMP/energy grid data from NOAA SWPC and EIA.

    Uses NOAA Space Weather Prediction Center endpoints for real-time
    geomagnetic and solar data:

    * **Kp index** -- planetary geomagnetic disturbance index (0-9 scale),
      updated every 3 hours.
    * **Solar flares** -- GOES X-ray flare detections with classification.
    * **Solar wind** -- proton density, bulk speed, and temperature from
      the DSCOVR/ACE spacecraft at L1.

    Historical events are recorded series: NASA's OMNI2 archive (hourly Kp
    and solar wind since 1963) for the geomagnetic storms, and EIA-930, the
    U.S. Energy Information Administration's hourly record of each grid
    operator, for grid events. Neither needs an API key.

    Feature engineering produces space-weather observables suitable for
    anomaly detection: Kp index, Kp rate of change, solar wind speed
    and density, X-ray flux classification, and geomagnetic storm
    indicators.
    """

    DOMAIN: str = "energy"
    SOURCE_URL: str = "https://services.swpc.noaa.gov/json/"
    # Labels = (kp >= 7) and feature[0] is the same ``kp`` index, so they are
    # circular against the full matrix (``LABEL_DERIVED_FEATURES`` removes the
    # Kp columns from the scored one). The series themselves are recorded:
    # OMNI2 and EIA-930.
    LABEL_SOURCE: str = "statistical"
    REQUIRES_API_KEY: bool = False
    API_KEY_ENV_VAR: str = ""
    FEATURE_COLUMNS: list[str] = [
        "kp",
        "kp_rate_of_change",
        "kp_sum_24h",
        "delta_kp_6h",
        "max_kp_24h",
        "solar_wind_speed",
        "solar_wind_density",
        "xray_class",
    ]

    #: De-leak: labels are ``kp >= 7`` and every one of these is a function of ``kp``.
    #: Still computed and returned by ``engineer_features``; excluded only
    #: from the scored matrix, so nothing is lost and the labels stop
    #: describing the thing being scored. This holds for fetched frames only:
    #: a synthetic historical frame derives its solar-wind and flare channels
    #: from the same Kp profile, so ``engineer_scored_features`` refuses it
    #: (``SyntheticDataNotDeleakable``) rather than report a disjoint matrix.
    LABEL_DERIVED_FEATURES: tuple[str, ...] = (
        "kp",
        "kp_rate_of_change",
        "kp_sum_24h",
        "delta_kp_6h",
        "max_kp_24h",
    )
    LABEL_FEATURE_DISJOINT: bool = True

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the energy loader.

        Args:
            **kwargs: Passed through to :class:`BaseDomainLoader`.
        """
        super().__init__(**kwargs)
        self._eia_api_key: str = os.environ.get("EIA_API_KEY", "")
        # Kept as an attribute for callers that read it. Nothing here needs
        # the key: grid events are read from the keyless EIA-930 files.
        if self._eia_api_key:
            logger.debug("EIA_API_KEY is set; grid events use the keyless EIA-930 files.")

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent Kp index and solar wind data from SWPC.

        Combines the NOAA SWPC Kp index feed with the 7-day solar wind
        plasma feed into a unified time-series DataFrame.

        Returns:
            DataFrame with columns: timestamp, kp, solar_wind_speed,
            solar_wind_density, solar_wind_temperature.

        Raises:
            ConnectionError: If the SWPC feeds are unreachable after retries.
        """
        # The frame cache: the ``timestamp`` column is datetimes, which the
        # plain JSON cache cannot write, so this frame was never cached and
        # both SWPC feeds were fetched again on every call.
        cache_key = "energy_realtime"
        cached = self._read_frame_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time energy/space-weather data.")
            return cached

        # Fetch Kp index data
        kp_df = self._fetch_kp_index()

        # Fetch solar wind data
        sw_df = self._fetch_solar_wind()

        # Merge on nearest timestamp
        df = self._merge_kp_and_solar_wind(kp_df, sw_df)

        self._write_frame_cache(cache_key, df)
        logger.info("Fetched %d real-time energy/space-weather records.", len(df))
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch the recorded series for a historical energy/space-weather event.

        A geomagnetic storm is read from NASA's OMNI2 archive: Kp and the
        solar-wind speed and density near Earth, in 3-hour steps. A grid
        event is read from EIA-930 (the grid operator's hourly demand,
        day-ahead demand forecast, net generation and interchange) joined to
        OMNI2 for the same hours, so its space-weather columns are what was
        measured then, not constants.

        ``solar_wind_speed`` and ``solar_wind_density`` are NaN where no
        spacecraft was measuring (more than half of 1989), and ``xray_class``
        is NaN throughout: OMNI2 carries no X-ray flux.

        Args:
            event_id: Key into the ground truth catalog (e.g.
                ``"quebec_1989"``).

        Returns:
            DataFrame with ``timestamp``, ``kp``, ``solar_wind_speed``,
            ``solar_wind_density``, ``xray_class`` and ``series_source``;
            a grid event adds ``grid_demand``, ``grid_demand_forecast``,
            ``grid_net_generation`` and ``grid_interchange`` (MW).

        Raises:
            ValueError: If *event_id* is not in the catalog.
            OfflineModeError: If ``MERCURY_OFFLINE`` is set and the event is
                not cached.
            DataSourceUnavailableError: If the archive cannot be read and the
                synthetic opt-in (``MERCURY_ALLOW_SYNTHETIC=1``) is not set.
                With the opt-in a synthetic Kp profile is returned instead,
                carrying the synthetic mark; it is never cached.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        # v2 key: frames under the old key were synthetic Kp profiles (or, for
        # the grid event, EIA demand beside invented columns).
        # v3 key: a v2 frame could be a truncated download or lack the hours
        # EIA files under the neighbouring half-year.
        # v4 key: earlier frames have no recorded X-ray class.
        # v5 key: DOE disturbance labels at the hour, and for Texas 2021.
        cache_key = f"energy_historical_v5_{event_id}"
        cached_frame = self._read_frame_cache(cache_key)

        if cached_frame is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return cached_frame

        event = _EVENT_CATALOG[event_id]
        start = pd.Timestamp(event["start"], tz="UTC")
        end = pd.Timestamp(event["end"], tz="UTC") + pd.Timedelta(days=1)
        try:
            df = self._fetch_omni2(start, end)
            if event.get("grid_event"):
                grid = self._fetch_eia930(str(event["balancing_authority"]), start, end)
                df = grid.merge(df, on="timestamp", how="left")
                df["series_source"] = SERIES_SOURCE_EIA930
                if event.get("oe417_region"):
                    reports = self.fetch_disturbance_reports(
                        str(event["oe417_region"]),
                        years=range(start.year, (end - pd.Timedelta(seconds=1)).year + 1),
                        local_timezone=str(event.get("oe417_timezone", "UTC")),
                    )
                    df[DOE_DISTURBANCE_COLUMN] = self._disturbance_labels(
                        df["timestamp"], reports, start, end
                    )
            else:
                df["series_source"] = SERIES_SOURCE_OMNI2
        except OfflineModeError:
            raise
        except (ConnectionError, OSError, ValueError) as exc:
            # Every historical event used to take this path, unconditionally
            # for the storms: a synthetic Kp profile whose solar-wind and
            # flare channels are functions of the same profile.
            check_synthetic_allowed(
                "EnergyLoader",
                f"recorded series for event '{event_id}' unavailable "
                f"({type(exc).__name__}: {exc})",
            )
            synthetic = self._generate_synthetic_kp_series(event)
            synthetic["series_source"] = SERIES_SOURCE_SYNTHETIC
            return self.mark_synthetic(synthetic)

        df = df.sort_values("timestamp").reset_index(drop=True)
        self._write_frame_cache(cache_key, df)
        logger.info(
            "Fetched %d recorded 3-hour steps for event '%s'.",
            len(df),
            event_id,
        )
        return df

    def fetch_disturbance_reports(
        self,
        nerc_region: str | None = None,
        *,
        years: Iterable[int] = (2023,),
        local_timezone: str = "UTC",
    ) -> pd.DataFrame:
        """DOE OE-417 electric disturbance reports, optionally for one NERC region.

        Read from EIA's Electric Power Monthly, Appendix B, which carries the
        hour each disturbance began and was restored. For 2023, when that
        archive cannot be read, Oak Ridge National Laboratory's copy of the
        annual summary is used instead; it carries dates only.

        Args:
            nerc_region: NERC region code as OE-417 writes it (``"TRE"``,
                ``"WECC"``, ``"SERC"``, ...); ``None`` for every region. A
                report filed under several regions (``"SERC/RF"``) matches
                each of them.
            years: Calendar years wanted.
            local_timezone: IANA zone the form's local times are read in.
                OE-417 records the reporting entity's local time and no zone.

        Returns:
            One row per report: ``began`` and ``restored`` (UTC; ``restored``
            is NaT when the form gives none), ``precision`` (``"minute"`` or
            ``"day"``), ``nerc_region``, ``event_type``, ``area_affected``,
            ``demand_loss_mw`` and ``customers_affected`` (NaN where the form
            leaves them blank or says "Unknown"), and ``report_year``.

        Raises:
            ConnectionError: If no source for a year can be reached.
            ValueError: If no source holds a complete table for a year.
        """
        frames = []
        for year in sorted(set(years)):
            cache_key = f"energy_oe417_v2_{year}_{local_timezone}"
            reports = self._read_frame_cache(cache_key)
            if reports is None:
                try:
                    reports = self._fetch_epm_disturbances(year, local_timezone)
                except (ConnectionError, OSError, ValueError) as exc:
                    if year != 2023:
                        raise
                    logger.warning(
                        "EIA Electric Power Monthly OE-417 table for 2023 unavailable (%s); "
                        "using the ORNL annual summary, which has dates only.",
                        exc,
                    )
                    reports = self._fetch_ornl_disturbances(local_timezone)
                self._write_frame_cache(cache_key, reports)
            frames.append(reports)
        combined = pd.concat(frames, ignore_index=True)
        if nerc_region is not None:
            filed_under = combined["nerc_region"].str.split(r"[/,;\s]+", regex=True)
            matches = filed_under.apply(lambda regions: nerc_region in regions)
            combined = combined[matches].reset_index(drop=True)
        return combined

    def _fetch_epm_disturbances(self, year: int, local_timezone: str) -> pd.DataFrame:
        """One year of OE-417 reports from EIA's Electric Power Monthly.

        Args:
            year: Calendar year wanted.
            local_timezone: IANA zone the table's local times are read in.

        Returns:
            Frame in the layout of :meth:`fetch_disturbance_reports`.

        Raises:
            ConnectionError: If an issue cannot be downloaded for a reason
                other than not existing.
            ValueError: If no issue holds a complete table for the year.
        """
        # Table B.2 of the issue two Februaries on is the year complete; Table
        # B.1 of the next February is the year to date, complete only when it
        # reaches December.
        for issue_year, table in ((year + 2, "Table_B_2.xlsx"), (year + 1, "Table_B_1.xlsx")):
            try:
                archive = self._fetch_url(_EPM_ISSUE_URL.format(year=issue_year))
            except FetchHTTPError as exc:
                if exc.status_code == 404:
                    continue
                raise
            rows = self._read_epm_table(archive, table)
            records = [row for row in rows[2:] if row and row[0].strip() == str(year)]
            months = {row[1].strip() for row in records if len(row) > 1}
            if not records or "12" not in months:
                continue
            return self._epm_reports(records, year, local_timezone)
        raise ValueError(
            f"EIA's Electric Power Monthly holds no complete OE-417 table for {year} "
            "(Appendix B ends with the February 2024 issue)"
        )

    @staticmethod
    def _read_epm_table(archive: bytes, table: str) -> list[list[str]]:
        """Rows of one spreadsheet inside an Electric Power Monthly archive.

        The workbook is read as what it is, a zip of XML, with the hardened
        parser: no spreadsheet library, no formula evaluation, no external
        entities. A table missing from the issue is an empty list.

        Args:
            archive: The issue's zip file.
            table: File name of the table inside it.

        Returns:
            One list of cell texts per row, empty cells as ``""``.

        Raises:
            ValueError: If the archive or the workbook cannot be read, or the
                table is implausibly large.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as issue:
                names = {name.rsplit("/", 1)[-1].lower(): name for name in issue.namelist()}
                if table.lower() not in names:
                    return []
                member = issue.getinfo(names[table.lower()])
                if member.file_size > _EPM_TABLE_MAX_BYTES:
                    raise ValueError(f"{table} is {member.file_size} bytes; refusing to read it")
                workbook_bytes = issue.read(member)
            with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as workbook:
                parts = {info.filename: info for info in workbook.infolist()}
                sheet = parts.get("xl/worksheets/sheet1.xml")
                if sheet is None or sheet.file_size > 10 * _EPM_TABLE_MAX_BYTES:
                    raise ValueError(f"{table} has no readable first sheet")
                shared: list[str] = []
                strings = parts.get("xl/sharedStrings.xml")
                if strings is not None:
                    if strings.file_size > 10 * _EPM_TABLE_MAX_BYTES:
                        raise ValueError(f"{table} has an implausibly large string table")
                    for item in _safe_xml(workbook.read(strings)).iter(f"{_XLSX_NS}si"):
                        shared.append(
                            "".join(text.text or "" for text in item.iter(f"{_XLSX_NS}t"))
                        )
                rows: list[list[str]] = []
                for row in _safe_xml(workbook.read(sheet)).iter(f"{_XLSX_NS}row"):
                    cells: list[str] = []
                    for cell in row.iter(f"{_XLSX_NS}c"):
                        column = _xlsx_column(cell.get("r", ""))
                        value = cell.find(f"{_XLSX_NS}v")
                        text = "" if value is None or value.text is None else value.text
                        if cell.get("t") == "s" and text.isdigit() and int(text) < len(shared):
                            text = shared[int(text)]
                        elif cell.get("t") == "inlineStr":
                            text = "".join(part.text or "" for part in cell.iter(f"{_XLSX_NS}t"))
                        while len(cells) < column:
                            cells.append("")
                        cells.append(text)
                    rows.append(cells)
                return rows
        except (zipfile.BadZipFile, KeyError, ParseError) as exc:
            raise ValueError(f"cannot read {table}: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _epm_reports(records: list[list[str]], year: int, local_timezone: str) -> pd.DataFrame:
        """Turn the rows of an Appendix B table into the report frame."""

        def cell(row: list[str], index: int) -> str:
            return row[index].strip() if index < len(row) else ""

        def moment(text: str) -> pd.Timestamp:
            stamp = pd.to_datetime(" ".join(text.split()), format=_EPM_EVENT_TIME, errors="coerce")
            if pd.isna(stamp):
                return pd.NaT
            # A local time that a clock change skipped or repeated is moved to
            # the nearest instant that exists, never dropped.
            return stamp.tz_localize(
                local_timezone, ambiguous=True, nonexistent="shift_forward"
            ).tz_convert("UTC")

        frame = pd.DataFrame(
            {
                "began": [moment(cell(row, 2)) for row in records],
                "restored": [moment(cell(row, 3)) for row in records],
                "precision": "minute",
                "nerc_region": [cell(row, 6) for row in records],
                "event_type": [cell(row, 8) for row in records],
                "area_affected": [cell(row, 7) for row in records],
                "demand_loss_mw": pd.to_numeric(
                    [cell(row, 9).replace(",", "") for row in records], errors="coerce"
                ),
                "customers_affected": pd.to_numeric(
                    [cell(row, 10).replace(",", "") for row in records], errors="coerce"
                ),
                "report_year": year,
            }
        )
        frame["began"] = pd.to_datetime(frame["began"], utc=True)
        frame["restored"] = pd.to_datetime(frame["restored"], utc=True)
        frame = frame.dropna(subset=["began"]).reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"the OE-417 table for {year} holds no dated reports")
        return frame

    def _fetch_ornl_disturbances(self, local_timezone: str) -> pd.DataFrame:
        """The 2023 OE-417 annual summary from ORNL's Open Energy Hub (dates only)."""
        records: list[dict[str, Any]] = []
        total = None
        while total is None or len(records) < total:
            page = self._fetch_json(
                _OE417_URL,
                params={
                    "limit": str(_OE417_PAGE),
                    "offset": str(len(records)),
                    "order_by": "date_event_began",
                },
            )
            results = page.get("results") if isinstance(page, dict) else None
            if not isinstance(results, list) or "total_count" not in page:
                raise ValueError("OE-417 response is not a record list")
            total = int(page["total_count"])
            if not results:
                break
            records.extend(results)

        def day(values: list[Any]) -> pd.Series[Any]:
            naive = pd.to_datetime(values, errors="coerce").tz_localize(None)
            return (
                pd.Series(naive)
                .dt.tz_localize(local_timezone, ambiguous=True, nonexistent="shift_forward")
                .dt.tz_convert("UTC")
            )

        reports = pd.DataFrame(
            {
                "began": day([r.get("date_event_began") for r in records]),
                "restored": day([r.get("date_of_restoration") for r in records]),
                "precision": "day",
                "nerc_region": [str(r.get("nerc_region") or "") for r in records],
                "event_type": [str(r.get("event_type") or "") for r in records],
                "area_affected": [str(r.get("area_affected") or "") for r in records],
                "demand_loss_mw": pd.to_numeric(
                    [r.get("demand_loss_mw") for r in records], errors="coerce"
                ),
                "customers_affected": pd.to_numeric(
                    [r.get("number_of_customers_affected") for r in records], errors="coerce"
                ),
                "report_year": 2023,
            }
        )
        reports = reports.dropna(subset=["began"]).reset_index(drop=True)
        if reports.empty:
            raise ValueError("OE-417 dataset holds no dated reports")
        return reports

    @staticmethod
    def _disturbance_labels(
        timestamps: pd.Series,
        reports: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> np.ndarray[Any, Any]:
        """Label the steps that fall inside a reported disturbance with an impact.

        A report counts when the form states a loss: of load
        (``demand_loss_mw`` > 0) or of customers (``customers_affected`` >
        0). Reports of vandalism or suspicious activity with neither are
        security filings, not interruptions of service. A step is labelled
        when it overlaps the disturbance: from the minute it began to the
        minute it was restored, or, for a source that gives dates only, from
        its first local day through its restoration day. With no restoration
        given, the step in which it began.

        Args:
            timestamps: Step timestamps (UTC).
            reports: Frame from :meth:`fetch_disturbance_reports`.
            start: First instant of the event window.
            end: End of the event window, exclusive.

        Returns:
            1-D binary array, one entry per step.

        Raises:
            ValueError: If the reports do not cover the window's years: an
                uncovered period would otherwise read as "no disturbances".
        """
        covered = set(reports["report_year"]) if "report_year" in reports else set()
        wanted = set(range(start.year, (end - pd.Timedelta(seconds=1)).year + 1))
        if not wanted <= covered:
            raise ValueError(
                f"OE-417 reports cover {sorted(covered)} and the event needs "
                f"{sorted(wanted)}; refusing to label an uncovered period as quiet"
            )
        impact = reports[
            (reports["demand_loss_mw"].fillna(0) > 0)
            | (reports["customers_affected"].fillna(0) > 0)
        ]
        labels = np.zeros(len(timestamps), dtype=np.int64)
        opens = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        closes = opens + pd.Timedelta(_STEP)
        for began, restored, precision in zip(
            impact["began"], impact["restored"], impact["precision"], strict=True
        ):
            last = restored if pd.notna(restored) and restored >= began else began
            if precision == "day":
                # A date stands for the whole local day.
                last = last + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            labels[(closes > began) & (opens <= last)] = 1
        return labels

    def _fetch_omni2(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Recorded Kp and solar wind for ``[start, end)`` from OMNI2, in 3-hour steps.

        Args:
            start: First instant wanted (UTC).
            end: End of the span, exclusive (UTC).

        Returns:
            DataFrame with ``timestamp``, ``kp``, ``solar_wind_speed``,
            ``solar_wind_density`` and ``xray_class`` (the strongest recorded
            flare of each step; NaN for a year outside the flare archive).

        Raises:
            ConnectionError: If a yearly file cannot be downloaded.
            ValueError: If a file does not have the OMNI2 layout, or holds
                no Kp for the span.
        """
        frames = []
        for year in range(start.year, (end - pd.Timedelta(seconds=1)).year + 1):
            text = self._fetch_url(_OMNI2_URL.format(year=year)).decode("ascii", errors="replace")
            frames.append(self._parse_omni2(text))
        hourly = pd.concat(frames, ignore_index=True)
        hourly = hourly[(hourly["timestamp"] >= start) & (hourly["timestamp"] < end)]
        if hourly["kp"].notna().sum() == 0:
            raise ValueError(f"OMNI2 holds no Kp between {start.date()} and {end.date()}")
        steps = (
            hourly.set_index("timestamp")[["kp", "solar_wind_speed", "solar_wind_density"]]
            .resample(_STEP)
            .mean()
            .reset_index()
        )
        steps["xray_class"] = self._recorded_flare_classes(steps["timestamp"], start, end)
        return steps

    def _recorded_flare_classes(
        self, timestamps: pd.Series[Any], start: pd.Timestamp, end: pd.Timestamp
    ) -> np.ndarray[Any, Any]:
        """Strongest recorded GOES X-ray flare in each step, as a class number.

        Read from NOAA NCEI's yearly GOES XRS flare reports. A step in which
        no flare peaked is 0. A year the archive does not hold (it ends in
        2017) is NaN: unknown, not quiet.

        Args:
            timestamps: Step starts (UTC), one per row.
            start: First instant of the span (UTC).
            end: End of the span, exclusive (UTC).

        Returns:
            1-D float array aligned with ``timestamps``.

        Raises:
            ConnectionError: If a yearly report cannot be downloaded for a
                reason other than not existing.
        """
        stamps = pd.to_datetime(timestamps, utc=True)
        classes = np.full(len(stamps), np.nan, dtype=np.float64)
        step = pd.Timedelta(_STEP)
        for year in range(start.year, (end - pd.Timedelta(seconds=1)).year + 1):
            try:
                raw = self._fetch_url(_FLARE_REPORT_URL.format(year=year))
            except FetchHTTPError as exc:
                if exc.status_code == 404:
                    logger.info("No GOES XRS flare report for %d; xray_class stays unknown.", year)
                    continue
                raise
            flares = self._parse_flare_report(raw.decode("ascii", errors="replace"))
            in_year = np.asarray(stamps.dt.year == year)
            classes[in_year] = 0.0
            for position in np.flatnonzero(in_year):
                opened = stamps.iloc[position]
                peaked = (flares["timestamp"] >= opened) & (flares["timestamp"] < opened + step)
                if peaked.any():
                    classes[position] = float(flares.loc[peaked, "xray_class"].max())
        return classes

    @staticmethod
    def _parse_flare_report(text: str) -> pd.DataFrame:
        """Parse one yearly GOES XRS flare report.

        Fixed columns: the date ``YYMMDD`` at 5-11, start, end and peak times
        (``HHMM``) at 13-17, 18-22 and 23-27, the class letter at 59. A row
        that does not fit (a short line, an unreadable date, no class) is
        skipped.

        Args:
            text: Contents of ``goes-xrs-report_YYYY.txt``.

        Returns:
            DataFrame with ``timestamp`` (UTC, the peak time, or the start
            time when no peak is given) and ``xray_class`` (A=1 ... X=5).
        """
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if len(line) < 60:
                continue
            value = _FLARE_CLASS_MAP.get(line[59])
            date, begun, peak = line[5:11], line[13:17], line[23:27]
            clock = peak if peak.isdigit() else begun
            if value is None or not date.isdigit() or not clock.isdigit():
                continue
            year = int(date[:2])
            stamp = pd.to_datetime(
                f"{1900 + year if year >= 70 else 2000 + year}{date[2:]}{clock}",
                format="%Y%m%d%H%M",
                utc=True,
                errors="coerce",
            )
            if pd.isna(stamp):
                continue
            rows.append({"timestamp": stamp, "xray_class": value})
        return pd.DataFrame(rows, columns=["timestamp", "xray_class"])

    @staticmethod
    def _parse_omni2(text: str) -> pd.DataFrame:
        """Parse one OMNI2 yearly file into hourly Kp and solar wind.

        Args:
            text: Contents of ``omni2_YYYY.dat``.

        Returns:
            DataFrame with ``timestamp`` (UTC), ``kp``, ``solar_wind_speed``
            (km/s) and ``solar_wind_density`` (N/cm^3); the archive's fill
            values are NaN.

        Raises:
            ValueError: If a row is too short to hold the Kp word.
        """
        table = np.loadtxt(io.StringIO(text), dtype=np.float64, ndmin=2)
        last = max(position for position, _ in _OMNI2_COLUMNS.values())
        if table.shape[1] <= last:
            raise ValueError(
                f"OMNI2 row has {table.shape[1]} words, expected more than {last}; "
                "the format may have changed"
            )

        def column(name: str) -> np.ndarray[Any, Any]:
            position, fill = _OMNI2_COLUMNS[name]
            values = table[:, position].copy()
            values[np.isclose(values, fill, rtol=0.0, atol=abs(fill) * 1e-6)] = np.nan
            return values

        timestamps = (
            pd.to_datetime(table[:, 0].astype(int).astype(str), format="%Y", utc=True)
            + pd.to_timedelta(table[:, 1] - 1, unit="D")
            + pd.to_timedelta(table[:, 2], unit="h")
        )
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "kp": column("kp10") / 10.0,
                "solar_wind_speed": column("speed"),
                "solar_wind_density": column("density"),
            }
        )

    def _fetch_eia930(
        self, balancing_authority: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """A grid operator's recorded hours from EIA-930, in 3-hour steps.

        Args:
            balancing_authority: EIA-930 code, e.g. ``"ERCO"``.
            start: First instant wanted (UTC).
            end: End of the span, exclusive (UTC).

        Returns:
            DataFrame with ``timestamp``, ``grid_demand``,
            ``grid_demand_forecast``, ``grid_net_generation`` and
            ``grid_interchange`` (MW; the mean of the hours in each step).

        Raises:
            ConnectionError: If a balance file cannot be downloaded.
            ValueError: If the operator has no rows in the span.
        """
        # The files are cut at local midnight, not UTC: the January-June file
        # for ERCOT runs from 06:00 UTC on 1 January to 04:00 UTC on 1 July.
        # A day either side of the window is therefore included when choosing
        # files, or a window that starts on a boundary loses its first hours.
        day = pd.Timedelta(days=1)
        months = pd.date_range(
            (start - day).normalize().replace(day=1), end + day, freq="MS", tz="UTC"
        )
        halves = sorted({(m.year, "Jan_Jun" if m.month <= 6 else "Jul_Dec") for m in months})
        records: list[dict[str, Any]] = []
        for year, half in halves:
            try:
                raw = self._fetch_url(_EIA930_BALANCE_URL.format(year=year, half=half))
            except ConnectionError:
                # A neighbouring half that EIA has not published yet (or no
                # longer serves) is tolerated; completeness is checked below.
                if (year, half) in self._required_halves(start, end):
                    raise
                continue
            records.extend(self._parse_eia930(raw, balancing_authority))
        hourly = pd.DataFrame.from_records(records)
        if hourly.empty:
            raise ValueError(f"EIA-930 has no rows for balancing authority {balancing_authority}")
        hourly = hourly.drop_duplicates(subset="timestamp")
        hourly = hourly[(hourly["timestamp"] >= start) & (hourly["timestamp"] < end)]
        expected = int((end - start) / pd.Timedelta(hours=1))
        if len(hourly) < _EIA930_MIN_COVERAGE * expected:
            # A truncated download or a missing file would otherwise be cached
            # as the event's record with most of its hours absent.
            raise ValueError(
                f"EIA-930 holds {len(hourly)} of {expected} hours for {balancing_authority} "
                f"between {start.date()} and {end.date()}; at least "
                f"{_EIA930_MIN_COVERAGE:.0%} are needed"
            )
        return hourly.set_index("timestamp").resample(_STEP).mean().reset_index()

    @staticmethod
    def _required_halves(start: pd.Timestamp, end: pd.Timestamp) -> set[tuple[int, str]]:
        """The half-year files that hold the window's own months."""
        months = pd.date_range(
            start.normalize().replace(day=1), end - pd.Timedelta(seconds=1), freq="MS", tz="UTC"
        )
        return {(m.year, "Jan_Jun" if m.month <= 6 else "Jul_Dec") for m in months}

    @staticmethod
    def _parse_eia930(raw: bytes, balancing_authority: str) -> list[dict[str, Any]]:
        """Read one operator's hours out of an EIA-930 balance file.

        The file holds every balancing authority (about 43 MB per half
        year), so it is streamed and only the wanted operator's rows are
        kept. A blank cell is a value the operator did not report: NaN,
        not zero.

        Args:
            raw: The CSV file's bytes.
            balancing_authority: EIA-930 code to keep.

        Returns:
            One dict per reported hour, stamped at the hour's start (UTC).

        Raises:
            ValueError: If the header lacks an expected column.
        """
        fields = {
            "grid_demand_forecast": "Demand Forecast (MW)",
            "grid_demand": "Demand (MW)",
            "grid_net_generation": "Net Generation (MW)",
            "grid_interchange": "Total Interchange (MW)",
        }
        reader = csv.reader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline=""))
        header = next(reader, [])
        try:
            authority_at = header.index("Balancing Authority")
            time_at = header.index("UTC Time at End of Hour")
            value_at = {name: header.index(column) for name, column in fields.items()}
        except ValueError as exc:
            raise ValueError(f"EIA-930 balance file is missing a column: {exc}") from exc

        def number(cell: str) -> float:
            cell = cell.replace(",", "").strip()
            try:
                return float(cell) if cell else float("nan")
            except ValueError:
                # "NA" and the like: not reported, as a blank cell is.
                return float("nan")

        needed = max(time_at, authority_at, *value_at.values())

        records = []
        for row in reader:
            # A row cut short (a truncated transfer) is skipped; the caller's
            # coverage check then decides whether enough of the span arrived.
            if len(row) <= needed or row[authority_at] != balancing_authority:
                continue
            end_of_hour = pd.to_datetime(
                row[time_at], format="%m/%d/%Y %I:%M:%S %p", utc=True, errors="coerce"
            )
            if pd.isna(end_of_hour):
                continue
            record: dict[str, Any] = {"timestamp": end_of_hour - pd.Timedelta(hours=1)}
            record.update({name: number(row[at]) for name, at in value_at.items()})
            records.append(record)
        return records

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth energy/space-weather events.

        Returns:
            List of dicts each containing *event_id*, *name*, *date*,
            and *description* keys.
        """
        events: list[dict[str, Any]] = []
        for event_id, meta in _EVENT_CATALOG.items():
            events.append(
                {
                    "event_id": event_id,
                    "name": meta["name"],
                    "date": meta["date"],
                    "description": meta["description"],
                }
            )
        return events

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Generate binary anomaly labels for a historical event.

        Labeling strategy (NOAA G scale via ``noaa_g_scale``):

        * **G3 and stronger** (Kp 7- and up): labeled anomaly (``1``).
        * **Below G1** (quiet/unsettled/active): labeled normal (``0``).
        * **G1-G2** (minor/moderate storm): labeled normal (``0``) for
          conservative labeling to reduce false-positive contamination.

        For the ``texas_2021`` grid event, anomaly labels are based on
        grid demand exceeding supply capacity thresholds rather than Kp.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_samples,)``.

        Raises:
            ValueError: If *event_id* is not recognized or no data is
                available.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        event = _EVENT_CATALOG[event_id]

        if event.get("oe417_region"):
            # DOE's own record of the disturbances: independent of every
            # column the detector scores. A frame without the column (a
            # synthetic stand-in) has no such labels to give.
            if DOE_DISTURBANCE_COLUMN not in df.columns:
                raise ValueError(
                    f"event '{event_id}' is labelled by DOE OE-417 reports and this "
                    "frame carries none (it is a synthetic stand-in)"
                )
            labels = np.asarray(df[DOE_DISTURBANCE_COLUMN], dtype=np.int64)
        elif event.get("grid_event") and "grid_demand" in df.columns:
            # For grid events, anomaly = served demand well below the forecast
            labels = self._label_grid_anomalies(df)
        else:
            # For geomagnetic events, anomaly = NOAA G3+ (severe; Kp 7- and up)
            kp_values = df["kp"].values.astype(np.float64)
            labels = (_g_levels(kp_values) >= _SEVERE_G_LEVEL).astype(np.int64)

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
        """Transform raw energy/space-weather data into a feature matrix.

        Engineered features (per time step):

        1. **kp** -- Kp index (0-9 scale).
        2. **kp_rate_of_change** -- difference in Kp from previous time
           step (0 for the first row).
        3. **kp_sum_24h** -- sum of Kp over trailing 8 steps (24h),
           a proxy for the Ap index.
        4. **delta_kp_6h** -- Kp change over 2 steps (6h).
        5. **max_kp_24h** -- maximum Kp over trailing 8-step (24h)
           rolling window.
        6. **solar_wind_speed** -- solar wind bulk speed in km/s.
        7. **solar_wind_density** -- solar wind proton density (p/cm^3).
        8. **xray_class** -- numeric solar flare classification
           (A=1, B=2, C=3, M=4, X=5; 0 if absent).

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 8)``.
        """
        if raw_data.empty:
            return np.empty((0, 8), dtype=np.float64)

        df = raw_data.copy()

        # Ensure chronological order
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        n = len(df)

        # ---- Kp index ----
        kp = self._safe_column(df, "kp")

        # ---- Kp rate of change (3h step) ----
        kp_roc = np.zeros(n, dtype=np.float64)
        if n > 1:
            kp_roc[1:] = np.diff(kp)

        # ---- Kp sum over 24h (8 steps) — proxy for Ap index ----
        kp_sum_24h = self._compute_rolling_sum(kp, window=8)

        # ---- Delta Kp over 6h (2 steps) ----
        delta_kp_6h = np.zeros(n, dtype=np.float64)
        if n > 2:
            delta_kp_6h[2:] = kp[2:] - kp[:-2]

        # ---- Max Kp over 24h (8 steps) ----
        max_kp_24h = self._compute_rolling_max(kp, window=8)

        # ---- Solar wind speed (km/s) ----
        sw_speed = self._safe_column(df, "solar_wind_speed")

        # ---- Solar wind density (p/cm^3) ----
        sw_density = self._safe_column(df, "solar_wind_density")

        # ---- X-ray flux class (numeric) ----
        xray_class = self._safe_column(df, "xray_class")

        # Stack into feature matrix
        features = np.column_stack(
            [
                kp,
                kp_roc,
                kp_sum_24h,
                delta_kp_6h,
                max_kp_24h,
                sw_speed,
                sw_density,
                xray_class,
            ]
        )

        # A channel with no reading at all (X-ray class for a recorded
        # historical event) carries no information and becomes 0; a gap in a
        # channel takes the column median.
        return self._fill_missing(features)

    @staticmethod
    def _fill_missing(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Replace non-finite cells: the column median, or 0 for an empty column."""
        features = np.where(np.isinf(features), np.nan, features)
        for col_idx in range(features.shape[1]):
            col = features[:, col_idx]
            mask = np.isnan(col)
            if mask.all():
                col[:] = 0.0
            elif mask.any():
                col[mask] = np.nanmedian(col)
        return features

    @staticmethod
    def is_grid_frame(raw_data: pd.DataFrame) -> bool:
        """Whether a frame holds a balancing authority's recorded EIA-930 series."""
        return all(column in raw_data.columns for column in _GRID_SERIES_COLUMNS)

    def engineer_grid_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Feature matrix of a grid event, from the grid's own recorded series.

        :meth:`engineer_features` describes space weather. A grid event (a
        winter storm, a year of DOE-reported disturbances) is not a space
        weather event, and scoring it on solar wind measures nothing: the
        detector was never shown the grid.

        Columns, in :data:`GRID_FEATURE_COLUMNS` order:

        1. **grid_demand** -- served demand (MW).
        2. **grid_net_generation** -- net generation (MW).
        3. **grid_interchange** -- total interchange (MW, exports positive).
        4. **grid_forecast_error** -- ``(forecast - demand) / forecast``.
        5. **grid_demand_change** -- step-to-step change in demand (MW).
        6. **grid_generation_change** -- step-to-step change in generation (MW).

        Args:
            raw_data: A grid-event frame from :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 6)``.

        Raises:
            ValueError: If the frame does not hold the EIA-930 columns.
        """
        if not self.is_grid_frame(raw_data):
            raise ValueError("not a grid-event frame: it lacks " + ", ".join(_GRID_SERIES_COLUMNS))
        df = raw_data
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        demand = self._safe_column(df, "grid_demand")
        forecast = self._safe_column(df, "grid_demand_forecast")
        generation = self._safe_column(df, "grid_net_generation")
        interchange = self._safe_column(df, "grid_interchange")
        with np.errstate(divide="ignore", invalid="ignore"):
            forecast_error = (forecast - demand) / forecast

        def _change(series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            change = np.zeros(len(series), dtype=np.float64)
            if len(series) > 1:
                change[1:] = np.diff(series)
            return change

        features = np.column_stack(
            [demand, generation, interchange, forecast_error, _change(demand), _change(generation)]
        )
        return self._fill_missing(features)

    def engineer_scored_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """The matrix to score: label-disjoint, and of the event's own kind.

        A storm frame is the space-weather matrix without its Kp columns (the
        labels are ``kp >= 7``). A grid frame is the grid matrix: all of it
        when the labels are DOE's disturbance reports, which no column
        produces, and without the demand-derived columns when the labels are
        the shortfall of demand against forecast.

        Raises:
            SyntheticDataNotDeleakable: For a synthetic stand-in.
        """
        if not self.is_grid_frame(raw_data):
            return super().engineer_scored_features(raw_data)
        if self.is_synthetic(raw_data):
            raise SyntheticDataNotDeleakable(
                f"{type(self).__name__}: the frame is synthetic, so no label-disjoint scored "
                "matrix exists"
            )
        features = self.engineer_grid_features(raw_data)
        if DOE_DISTURBANCE_COLUMN in raw_data.columns:
            return features
        keep = [
            index
            for index, name in enumerate(GRID_FEATURE_COLUMNS)
            if name not in GRID_SHORTFALL_DERIVED_FEATURES
        ]
        return np.asarray(features[:, keep])

    # ------------------------------------------------------------------
    # Private helpers -- data fetching
    # ------------------------------------------------------------------

    # ``_iter_feed_rows`` moved to ``BaseDomainLoader`` so every JSON-table
    # consumer shares the same shape-flip absorber; this loader inherits it.

    @staticmethod
    def _as_float(value: Any) -> float:
        """Coerce a feed value to float, mapping missing/unparsable to NaN."""
        if value is None:
            return float(np.nan)
        try:
            return float(value)
        except (ValueError, TypeError):
            return float(np.nan)

    def _fetch_kp_index(self) -> pd.DataFrame:
        """Fetch the NOAA SWPC planetary Kp index feed.

        Accepts both feed shapes SWPC has served for this endpoint (see
        :meth:`_iter_feed_rows`); it currently returns objects keyed
        ``time_tag`` / ``Kp`` / ``a_running`` / ``station_count``.

        Returns:
            DataFrame with columns: timestamp, kp.
        """
        raw = self._fetch_json(_KP_INDEX_URL)
        rows = self._iter_feed_rows(raw, ("time_tag", "Kp", "a_running", "station_count"))

        if not rows:
            logger.warning("Kp index feed returned empty or malformed data.")
            return pd.DataFrame(columns=["timestamp", "kp"])

        records = [
            {"timestamp": str(row["time_tag"]), "kp": self._as_float(row["Kp"])}
            for row in rows
            if row.get("time_tag") is not None
        ]

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        return df

    def _solar_wind_records(
        self,
        url: str,
        columns: tuple[str, str, str, str],
        *,
        default_source: str = "",
        carries_provenance: bool = False,
    ) -> list[dict[str, Any]]:
        """Read one solar wind feed into canonical solar-wind records.

        Args:
            url: Feed to read.
            columns: Source field names in the order
                ``(time_tag, density, speed, temperature)``.
            default_source: Provenance label recorded when the feed does not
                carry a per-row ``source`` field (the ACE SWEPAM history).
            carries_provenance: When true, also select the feed's per-row
                ``source`` and ``active`` fields (the RTSW constellation
                feed carries both; as of 2026 the sources are SOLAR1/SWFO-L1
                — the active spacecraft — plus ACE and IMAP).

        Returns:
            Records keyed ``timestamp`` / ``solar_wind_density`` /
            ``solar_wind_speed`` / ``solar_wind_temperature`` plus the
            provenance keys ``_source`` / ``_active`` (dedup inputs, dropped
            before the frame leaves ``_fetch_solar_wind``). Instrument fill
            values (negative density/speed/temperature, which the feeds emit
            for dropouts) become NaN rather than being read as measurements.
            Returns ``[]`` when the feed is unreachable or unusable.
        """
        try:
            raw = self._fetch_json(url)
        except (OSError, json.JSONDecodeError) as exc:
            # Transport failure or a garbled body: the other feed still covers
            # part of the window, so degrade rather than abort. Configuration
            # faults (``UnsafeURLError`` and other ``ValueError``s from the
            # allowlist gate) deliberately propagate -- swallowing those would
            # turn an off-allowlist host into a silent coverage gap.
            logger.warning("Solar wind feed %s unavailable: %s", url, exc)
            return []

        ts_key, dens_key, speed_key, temp_key = columns
        selected = columns + (("source", "active") if carries_provenance else ())
        records: list[dict[str, Any]] = []
        for row in self._iter_feed_rows(raw, selected):
            if row.get(ts_key) is None:
                continue
            values = {
                "solar_wind_density": self._as_float(row[dens_key]),
                "solar_wind_speed": self._as_float(row[speed_key]),
                "solar_wind_temperature": self._as_float(row[temp_key]),
            }
            # ACE reports dropouts as large negatives (-9999.9); a negative
            # density, speed or temperature is unphysical from either feed.
            values = {k: (v if v > 0 else float(np.nan)) for k, v in values.items()}
            records.append(
                {
                    "timestamp": str(row[ts_key]),
                    **values,
                    "_source": str(row.get("source") or default_source),
                    "_active": bool(row.get("active")),
                }
            )
        return records

    def _fetch_solar_wind(self) -> pd.DataFrame:
        """Fetch NOAA SWPC solar wind plasma data across the Kp window.

        SWPC retired ``plasma-7-day``, and its RTSW replacement only serves the
        trailing 24 hours -- against a 7-day Kp feed that leaves ~86% of rows
        without plasma values. Two live feeds are therefore combined:

        * **RTSW 1-minute** (:data:`_SOLAR_WIND_URL`) -- freshest, high
          cadence, and multi-spacecraft: each minute can carry rows from the
          active spacecraft (SOLAR1/SWFO-L1 as of 2026) plus non-active
          ACE and IMAP rows whose calibrations differ materially (measured
          median deltas: 7.2 /cc density, 42% temperature).
        * **ACE SWEPAM 1-hour** (:data:`_ACE_SWEPAM_URL`) -- ~31 days of
          hourly history, which covers the whole Kp window.

        One row survives per minute, chosen by explicit preference -- most
        valid measurements first, then the active spacecraft, then feed
        order (RTSW before ACE history) -- rather than by feed order alone.
        Feed order was measured keeping the non-active spacecraft in 35% of
        duplicate minutes and once kept a fill row (NaN) while discarding a
        valid measurement at the same minute. Either feed alone still yields
        a usable frame, so one endpoint being retired degrades coverage
        instead of emptying the result.

        Returns:
            DataFrame with columns: timestamp, solar_wind_density,
            solar_wind_speed, solar_wind_temperature, sorted by timestamp.
        """
        records = self._solar_wind_records(
            _SOLAR_WIND_URL,
            ("time_tag", "proton_density", "proton_speed", "proton_temperature"),
            carries_provenance=True,
        )
        preferred = len(records)
        records += self._solar_wind_records(
            _ACE_SWEPAM_URL,
            ("time_tag", "dens", "speed", "temperature"),
            default_source="ace_swepam_1h",
        )

        if not records:
            logger.warning("Solar wind feeds returned empty or malformed data.")
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "solar_wind_density",
                    "solar_wind_speed",
                    "solar_wind_temperature",
                ]
            )

        value_columns = [
            "solar_wind_density",
            "solar_wind_speed",
            "solar_wind_temperature",
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        # RTSW is a 1-minute product, but non-active spacecraft (IMAP) tag
        # rows with per-spacecraft sampling jitter in the seconds field, so
        # exact-string dedup silently interleaved a third calibration between
        # grid minutes. Floor to the product's minute grid so one preference
        # rule picks one row per minute.
        df["timestamp"] = df["timestamp"].dt.floor("min")
        # Preference order within a minute: most valid measurements first
        # (a fill row must never shadow a valid one), then the active
        # spacecraft's calibration. The stable mergesort makes feed order
        # (RTSW appended before ACE history) the final tiebreak, preserving
        # the RTSW-over-ACE-hourly preference at overlapping stamps.
        df["_n_valid"] = df[value_columns].notna().sum(axis=1)
        df = df.sort_values(
            ["timestamp", "_n_valid", "_active"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.drop(columns=["_n_valid", "_active", "_source"]).reset_index(drop=True)
        logger.info(
            "Solar wind: %d RTSW + %d ACE records -> %d unique minute timestamps.",
            preferred,
            len(records) - preferred,
            len(df),
        )
        return df

    def _fetch_xray_flares(self) -> pd.DataFrame:
        """Fetch the latest GOES X-ray flare events from SWPC.

        Returns:
            DataFrame with columns: timestamp, xray_class (numeric).
        """
        raw: list[dict[str, Any]] = self._fetch_json(_XRAY_FLARES_URL)

        if not raw:
            logger.warning("X-ray flares feed returned empty data.")
            return pd.DataFrame(columns=["timestamp", "xray_class"])

        records: list[dict[str, Any]] = []
        for entry in raw:
            time_tag = entry.get("begin_time") or entry.get("time_tag", "")
            class_type = entry.get("current_class", "") or entry.get("max_class", "")
            numeric_class = self._parse_flare_class(class_type)
            records.append(
                {
                    "timestamp": str(time_tag),
                    "xray_class": numeric_class,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"])
        return df

    def _fetch_eia_grid_data(self, event: dict[str, Any]) -> pd.DataFrame:
        """Recorded grid data for a grid event (kept for callers of the old name).

        This method used to query the keyed EIA API v2 daily endpoint without
        a region filter and to fill the space-weather columns and a
        ``grid_supply`` of 95% of demand with constants. It now returns what
        :meth:`fetch_historical` does for the event: the operator's EIA-930
        hours joined to OMNI2. No API key is involved.

        Args:
            event: Event metadata dict from the catalog.

        Returns:
            The event's recorded frame, as :meth:`fetch_historical` builds it.
        """
        start = pd.Timestamp(event["start"], tz="UTC")
        end = pd.Timestamp(event["end"], tz="UTC") + pd.Timedelta(days=1)
        grid = self._fetch_eia930(str(event["balancing_authority"]), start, end)
        return grid.merge(self._fetch_omni2(start, end), on="timestamp", how="left")

    # ------------------------------------------------------------------
    # Private helpers -- synthetic data generation
    # ------------------------------------------------------------------

    def _generate_synthetic_kp_series(self, event: dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic Kp time-series for a historical event.

        Creates a plausible multi-week Kp profile with a background of
        quiet/unsettled conditions punctuated by the documented major
        storm. Solar wind speed and density are correlated with the
        simultaneous Kp (a CME arrival raises both at once); X-ray flare
        activity instead LEADS the storm by the Sun-Earth CME transit time
        (1-3 days for storm-capable CMEs), because flares happen at eruption
        while Kp responds only at arrival -- a synthetic series where
        X-ray class tracked instantaneous Kp taught the anti-physical
        shortcut that flares and storms are simultaneous.

        For extended time windows, minor storms (Kp 4-5) are sprinkled
        throughout the quiet period to provide realistic baseline variance.

        Args:
            event: Event metadata dict from the catalog.

        Returns:
            DataFrame with columns: timestamp, kp, solar_wind_speed,
            solar_wind_density, xray_class.
        """
        n_hours: int = event.get("synthetic_hours", 2160)
        peak_kp: int = event.get("peak_kp", 9)
        storm_day: int = event.get("storm_day", 40)
        start_date = pd.Timestamp(event["start"], tz="UTC")

        # Generate 3-hour resolution timestamps (standard Kp cadence)
        n_steps = n_hours // 3
        n_steps = max(n_steps, 1)
        timestamps = pd.date_range(start=start_date, periods=n_steps, freq="3h", tz="UTC")

        # Build Kp profile: extended quiet + major storm + recovery
        kp_values = self._build_storm_profile(n_steps, peak_kp, storm_day=storm_day)

        # Derive correlated solar wind speed: empirical Kp-speed relation.
        # Process-stable seed: Python's built-in hash() is salted per process,
        # so derive the seed from hashlib.sha256 instead -- identical every run
        # without relying on PYTHONHASHSEED.  Mirrors tsunami_loader.
        seed_bytes = hashlib.sha256(event["date"].encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes[:4], "little") % (2**31))
        base_speed = 350.0 + kp_values * 50.0
        sw_speed = base_speed + rng.normal(0, 20, size=n_steps)
        sw_speed = np.clip(sw_speed, 250.0, 1200.0)

        # Solar wind density: inversely related to speed during CMEs
        base_density = 10.0 - kp_values * 0.5
        sw_density = base_density + rng.normal(0, 2, size=n_steps)
        sw_density = np.clip(sw_density, 0.5, 50.0)

        # X-ray class: flares LEAD each geomagnetic disturbance by the CME
        # transit time (storm-capable CMEs reach Earth in 1-3 days), so the
        # flare window of a storm step sits 8-24 three-hour steps EARLIER.
        # The flare magnitude still scales with how strong the storm will
        # be (bigger CMEs from bigger flares), banded on the NOAA G scale.
        g_levels = _g_levels(kp_values)
        xray = np.ones(n_steps, dtype=np.float64)  # A-class background
        xray[rng.random(n_steps) < 0.05] = 3.0  # sporadic C-class, no storm follows
        # One eruption per storm EPISODE (contiguous G1+ run): the CME that
        # causes it erupts 8-24 three-hour steps (1-3 days) before onset.
        storm_mask = g_levels >= _STORM_G_LEVEL
        onsets = np.flatnonzero(storm_mask & ~np.concatenate(([False], storm_mask[:-1])))
        for onset in onsets:
            run_end = onset
            while run_end + 1 < n_steps and storm_mask[run_end + 1]:
                run_end += 1
            transit_steps = int(rng.integers(8, 25))
            eruption = max(int(onset) - transit_steps, 0)
            severe = bool(np.any(g_levels[onset : run_end + 1] >= _SEVERE_G_LEVEL))
            xray[eruption] = 5.0 if severe else 4.0

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "kp": kp_values,
                "solar_wind_speed": sw_speed,
                "solar_wind_density": sw_density,
                "xray_class": xray,
            }
        )
        # Every channel above is a function of ``kp_values``, the series the
        # labels threshold; the mark makes engineer_scored_features refuse a
        # "de-leaked" matrix from it (see SyntheticDataNotDeleakable).
        return self.mark_synthetic(df)

    @staticmethod
    def _build_storm_profile(
        n_steps: int, peak_kp: int, storm_day: int = 40
    ) -> np.ndarray[Any, Any]:
        """Build a synthetic Kp storm profile.

        For extended time windows (>100 steps), the profile includes:

        1. **Background quiet** -- Kp ~ 1-2 with Gaussian noise and
           occasional minor disturbances (Kp 3-5).
        2. **Major storm** -- a 2-day ramp-up, 1-day peak at peak_kp,
           and 3-day exponential recovery, positioned at storm_day.
        3. **Post-storm quiet** -- return to background levels.

        This produces a realistic Kp distribution: ~90% quiet (<3),
        ~7% unsettled/minor storm (3-6), ~3% severe (>=7).

        Args:
            n_steps: Total number of time steps.
            peak_kp: Maximum Kp value at storm peak.
            storm_day: Day within the window to place the storm center.

        Returns:
            1-D numpy array of Kp values.
        """
        rng = np.random.default_rng(seed=(peak_kp * 1000 + n_steps) & 0xFFFFFFFF)

        # Base: quiet conditions with slight variation
        kp = 1.5 + rng.normal(0, 0.4, size=n_steps)

        # Sprinkle minor storms every ~7-10 days (56-80 steps)
        steps_per_day = 8  # 24h / 3h
        n_minor = max(1, n_steps // (steps_per_day * 8))
        minor_positions = rng.integers(0, n_steps, size=n_minor)
        for pos in minor_positions:
            # Minor storm: 8-16 steps (1-2 days)
            storm_len = rng.integers(8, 17)
            minor_peak = rng.uniform(3.5, 5.5)
            for j in range(min(storm_len, n_steps - pos)):
                # Bell curve shape
                t = j / storm_len
                kp[pos + j] = max(
                    kp[pos + j],
                    minor_peak * np.sin(np.pi * t),
                )

        # Place major storm at storm_day
        storm_center = storm_day * steps_per_day
        if storm_center >= n_steps:
            storm_center = n_steps // 2

        # Ramp-up: 2 days (16 steps)
        ramp_len = 16
        ramp_start = max(0, storm_center - ramp_len - 8)
        for j in range(min(ramp_len, storm_center - ramp_start)):
            t = np.float64(j / ramp_len)
            kp[ramp_start + j] = 2.0 + (float(peak_kp) - 2.0) * t

        # Peak: 8 steps (24h)
        peak_start = max(0, storm_center - 8)
        peak_end = min(n_steps, storm_center + 8)
        kp[peak_start:peak_end] = float(peak_kp) + rng.uniform(
            -0.5, 0.0, size=peak_end - peak_start
        )

        # Recovery: 3 days (24 steps) exponential decay
        recovery_len = 24
        recovery_end = min(n_steps, peak_end + recovery_len)
        if recovery_end > peak_end:
            decay = np.exp(-np.linspace(0, 4.0, recovery_end - peak_end))
            kp[peak_end:recovery_end] = 1.5 + (float(peak_kp) - 1.5) * decay

        # Clamp to valid Kp range
        kp = np.clip(kp, 0.0, 9.0)
        return np.round(kp * 3) / 3  # Kp is reported in thirds

    # ------------------------------------------------------------------
    # Private helpers -- data merging
    # ------------------------------------------------------------------

    def _merge_kp_and_solar_wind(self, kp_df: pd.DataFrame, sw_df: pd.DataFrame) -> pd.DataFrame:
        """Merge Kp index and solar wind DataFrames on nearest timestamp.

        Uses a merge_asof to align the higher-cadence solar wind data
        with the 3-hour Kp index timestamps.

        Args:
            kp_df: DataFrame with timestamp and kp columns.
            sw_df: DataFrame with timestamp and solar wind columns.

        Returns:
            Merged DataFrame with all columns.
        """
        if kp_df.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "kp",
                    "solar_wind_speed",
                    "solar_wind_density",
                    "solar_wind_temperature",
                ]
            )

        if sw_df.empty:
            kp_df["solar_wind_speed"] = np.nan
            kp_df["solar_wind_density"] = np.nan
            kp_df["solar_wind_temperature"] = np.nan
            return kp_df

        # Ensure both are sorted by timestamp for merge_asof
        kp_sorted = kp_df.sort_values("timestamp").reset_index(drop=True)
        sw_sorted = sw_df.sort_values("timestamp").reset_index(drop=True)

        merged = pd.merge_asof(
            kp_sorted,
            sw_sorted,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("3h"),
        )
        return merged

    # ------------------------------------------------------------------
    # Private helpers -- classification and labeling
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_flare_class(class_str: str) -> float:
        """Parse a solar flare classification string to a numeric value.

        Extracts the letter class (A, B, C, M, X) from strings like
        ``"M2.5"`` or ``"X1.0"`` and returns the corresponding numeric
        value from ``_FLARE_CLASS_MAP``.

        Args:
            class_str: Flare classification string (e.g., ``"M2.5"``).

        Returns:
            Numeric class value (1.0-5.0), or 0.0 if unparseable.
        """
        if not class_str:
            return 0.0
        letter = class_str[0].upper()
        return _FLARE_CLASS_MAP.get(letter, 0.0)

    @staticmethod
    def _label_grid_anomalies(df: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Label the steps of a grid event where the grid could not serve its load.

        The measure is the shortfall of served demand against the operator's
        own day-ahead forecast, ``(forecast - demand) / forecast``, above
        :data:`GRID_SHORTFALL_THRESHOLD`. Both columns are EIA-930
        measurements. The previous rule compared demand with a
        ``grid_supply`` column this loader had set to 95% of demand, so it
        labelled every step.

        A frame without a forecast (one built before EIA-930 was read, or a
        synthetic stand-in) falls back to that ``demand > supply`` comparison
        when it has a supply column, and to no labels otherwise.

        Args:
            df: Grid-event frame.

        Returns:
            1-D binary numpy array; a step with a missing value is 0.
        """
        demand = pd.to_numeric(df["grid_demand"], errors="coerce").to_numpy(dtype=np.float64)
        if "grid_demand_forecast" in df.columns:
            forecast = pd.to_numeric(df["grid_demand_forecast"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                shortfall = (forecast - demand) / forecast
            return np.asarray(np.nan_to_num(shortfall, nan=0.0) > GRID_SHORTFALL_THRESHOLD).astype(
                np.int64
            )
        if "grid_supply" in df.columns:
            supply = pd.to_numeric(df["grid_supply"], errors="coerce").to_numpy(dtype=np.float64)
            return np.asarray(demand > supply).astype(np.int64)
        return np.zeros(len(df), dtype=np.int64)

    # ------------------------------------------------------------------
    # Private helpers -- utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_column(df: pd.DataFrame, column: str) -> np.ndarray[Any, Any]:
        """Extract a column as a float64 array, returning zeros if absent.

        Args:
            df: Source DataFrame.
            column: Column name to extract.

        Returns:
            1-D float64 numpy array of column values with missing readings
            as NaN, or zeros if the column does not exist.
        """
        if column in df.columns:
            # A missing reading stays NaN here and takes the column's median
            # at the end of ``engineer_features``. Filled with 0.0 at this
            # point, an hour with no spacecraft coverage became a solar wind
            # of 0 km/s, a value that is never observed.
            return np.asarray(pd.to_numeric(df[column], errors="coerce").values, dtype=np.float64)
        return np.zeros(len(df), dtype=np.float64)

    @staticmethod
    def _compute_rolling_sum(values: np.ndarray[Any, Any], window: int = 8) -> np.ndarray[Any, Any]:
        """Compute the rolling sum over a trailing window.

        Args:
            values: 1-D input array.
            window: Number of steps in the trailing window.

        Returns:
            1-D array of rolling sum values.
        """
        n = len(values)
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            # A missing Kp is skipped, not propagated: one gap would otherwise
            # blank this feature for the next ``window`` steps (a day).
            result[i] = np.nansum(values[start : i + 1])
        return result

    @staticmethod
    def _compute_rolling_max(values: np.ndarray[Any, Any], window: int = 8) -> np.ndarray[Any, Any]:
        """Compute the rolling maximum over a trailing window.

        Args:
            values: 1-D input array.
            window: Number of steps in the trailing window.

        Returns:
            1-D array of rolling maximum values.
        """
        n = len(values)
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            # As above: the largest reading in the window, ignoring gaps; NaN
            # only when the whole window is missing.
            result[i] = np.nan if np.isnan(chunk).all() else np.nanmax(chunk)
        return result
