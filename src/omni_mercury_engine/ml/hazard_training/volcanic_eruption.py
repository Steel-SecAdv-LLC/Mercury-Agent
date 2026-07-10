# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the EruptionForecastModel on real AVO seismic data + real GVP eruption labels.

SCOPE (stated wherever this checkpoint is described): the shipped model is
trained for a NAMED set of Alaska/AVO-monitored volcanoes -- Shishaldin,
Semisopochnoi, Pavlof, Great Sitkin, Veniaminof, Cleveland, Okmok and
Redoubt -- from their local AV-network seismic stations. It is NOT a
universal eruption forecaster: applying it to any other volcano is out of
its training distribution and unsupported.

Data sources (hook ``volcanic_eruption``,
``VolcanicEruptionDetector.load_neural_weights``):

* **Smithsonian GVP Holocene eruption catalog** (WFS CSV export from
  ``webservices.volcano.si.edu``) -- real eruption start/end dates and VEI.
  Only day-precision starts (day uncertainty <= 2 days) become label onsets;
  month-precision rows are used solely to EXCLUDE their eruptive spans from
  the negatives.
* **EarthScope FDSN web services** (``service.earthscope.org``) -- station
  inventory (channel sensitivities), the availability service (which days a
  station actually recorded), and dataselect miniSEED waveforms for the AV
  network stations within ~12 km of each named volcano.
* **USGS HANS notice archive** (``volcanoes.usgs.gov``) -- Volcanic Activity
  Notices used as an onset cross-check for the two test-era onsets whose GVP
  day carries 1-2 day uncertainty (Shishaldin 2023-07, Semisopochnoi
  2021-02). HANS search is a POST endpoint; ``http_get_with_retry`` is
  GET-only, so those requests go through ``SafeHTTPClient.post_json``
  directly (same allowlist + SSRF gates, documented here).

Task: for each (volcano, day) sample, predict from that day's seismic record
whether a GVP-cataloged eruption of that volcano begins within the next
K=14 days. The 128-dim ``fused_features`` vector built here is the canonical
``volcano-seismic-v1`` spec for the detector's learned path (the detector
only runs the neural forecast when ``fused_features`` is supplied). The
32-dim hourly sequence trains the ``SeismicSwarmDetector`` LSTM head on the
same label (documented: its "swarm probability" is P(eruption within 14 d)
from one day of hourly seismic statistics, not a picked-swarm label).

Temporal split (never random -- volcanic unrest autocorrelates over months):
train 2004-2016, validation 2017-2019, test 2020-2024. Every split contains
eruption onsets from the named volcano set (see ``SPLIT`` and the manifest).

Feature honesty rules: amplitude baselines are CAUSAL (median of previously
sampled quiet days at the same station-channel, never any future day);
standardization statistics come from TRAIN-year samples only; missing data
is carried as explicit presence flags, never imputed silently.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import torch

from omni_mercury_engine.datasets.base import http_get_with_retry
from omni_mercury_engine.ml.hazard_training.common import (
    EvaluationOutcome,
    PipelineContext,
    TemporalSplit,
    binary_auc,
    brier_score,
    cached_fetch,
    candidate_paths,
    save_candidate,
    save_evaluation,
    seed_everything,
    sha256_file,
    ship_checkpoint,
)

logger = logging.getLogger(__name__)

HOOK_NAME = "volcanic_eruption"
CHECKPOINT_NAME = "volcanic_avo_seismic"

FEATURE_SPEC_VERSION = "volcano-seismic-v1"
FEATURE_DIM = 128
HOURLY_DIM = 32
HOURS_PER_DAY = 24

#: Forecast horizon: label = eruption onset within this many days AHEAD.
K_DAYS = 14
#: Negatives must be at least this far from any eruptive interval.
QUIET_BUFFER_DAYS = 60
#: Hard cap on the CUMULATIVE waveform cache (== total EarthScope transfer,
#: since day files are fetched once and never deleted). Applies across
#: reruns, not per invocation -- restarting fetch cannot double the budget.
WAVEFORM_BYTE_BUDGET = 6_200_000_000
#: Sampled quiet (negative) days per volcano-year, per split. Test years get
#: the densest sampling (false-alarm-rate precision on held-out data).
NEG_PER_YEAR = {"train": 3, "val": 4, "test": 6}

SPLIT = TemporalSplit(
    train_years=tuple(range(2004, 2017)),
    val_years=(2017, 2018, 2019),
    test_years=(2020, 2021, 2022, 2023, 2024),
)

GVP_ERUPTIONS_URL = (
    "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=2.0.0"
    "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
    "&outputFormat=csv"
)
FDSN_BASE = "https://service.earthscope.org/fdsnws"
STATION_URL = f"{FDSN_BASE}/station/1/query?net=AV&format=text&level=channel&cha=BHZ,EHZ,SHZ"
HANS_VOLCANO_URL = "https://volcanoes.usgs.gov/hans-public/api/volcano/getUSVolcanoes"
HANS_SEARCH_URL = "https://volcanoes.usgs.gov/hans-public/api/search/search"


@dataclass(frozen=True)
class VolcanoSpec:
    """One named volcano in the training set.

    Attributes:
        name: Human name (GVP spelling).
        gvp_number: Smithsonian GVP volcano number (label join key).
        hans_cd: USGS HANS volcano code (VAN cross-check).
        stations: Ordered (station, channel) preference list; the first
            entry with waveform availability on a given day is fetched.
            Lower-rate broadband channels are preferred where concurrently
            available to respect the transfer budget.
    """

    name: str
    gvp_number: int
    hans_cd: str
    stations: tuple[tuple[str, str], ...]


#: The named AVO volcano set this checkpoint is trained for (order fixes the
#: one-hot encoding). Bogoslof was dropped: its only local station (BOGO)
#: was installed 2018-08, after its 2016-17 eruption -- no real precursor
#: waveforms exist. Augustine was dropped to keep the one-hot within 8 dims;
#: its single day-precise onset (2005-12-09) lies in the train era only.
VOLCANOES: tuple[VolcanoSpec, ...] = (
    VolcanoSpec(
        "Shishaldin",
        311360,
        "ak252",
        (("SSBA", "BHZ"), ("SSLS", "BHZ"), ("SSLN", "BHZ"), ("SSLS", "EHZ"), ("SSLN", "EHZ")),
    ),
    VolcanoSpec(
        "Semisopochnoi",
        311060,
        "ak248",
        (("CERB", "BHZ"), ("CERB", "SHZ"), ("CESW", "BHZ"), ("CESW", "SHZ"), ("CEPE", "SHZ")),
    ),
    VolcanoSpec(
        "Pavlof",
        312030,
        "ak210",
        (
            ("PS4A", "BHZ"),
            ("PN7A", "BHZ"),
            ("PVV", "BHZ"),
            ("PVV", "SHZ"),
            ("PVV", "EHZ"),
            ("PS4A", "EHZ"),
            ("PN7A", "EHZ"),
        ),
    ),
    VolcanoSpec(
        "Great Sitkin",
        311120,
        "ak111",
        (("GSTD", "BHZ"), ("GSSP", "BHZ"), ("GSTD", "EHZ"), ("GSSP", "EHZ"), ("GSMY", "EHZ")),
    ),
    VolcanoSpec(
        "Veniaminof",
        312070,
        "ak301",
        (("VNCG", "BHZ"), ("VNSS", "EHZ"), ("VNWF", "BHZ"), ("VNWF", "EHZ")),
    ),
    VolcanoSpec(
        "Cleveland",
        311240,
        "ak52",
        (("CLES", "BHZ"), ("CLNE", "BHZ"), ("CLCL", "BHZ")),
    ),
    VolcanoSpec(
        "Okmok",
        311290,
        "ak206",
        (("OKSO", "BHZ"), ("OKCE", "BHZ"), ("OKNC", "BHZ"), ("OKCF", "EHZ"), ("OKWR", "EHZ")),
    ),
    VolcanoSpec(
        "Redoubt",
        313030,
        "ak231",
        (("RDWB", "BHZ"), ("RED", "BHZ"), ("RDSO", "BHZ"), ("REF", "EHZ"), ("RSO", "EHZ")),
    ),
)

#: SSAM frequency bands (Hz) -- standard volcano-observatory quads.
SSAM_BANDS: tuple[tuple[float, float], ...] = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0))

_EPS = 1e-20

# ---------------------------------------------------------------------------
# Feature specification (every dimension documented; the rest zero-reserved).
# ---------------------------------------------------------------------------

_DAILY_FEATURE_NAMES: tuple[str, ...] = (
    "rsam_mean_24h_log10",  # 0: log10 mean |velocity| (m/s) over the day
    "rsam_last6h_log10",  # 1: same over the final 6 h of the day
    "rsam_last1h_log10",  # 2: same over the final 1 h of the day
    "rsam_median_minute_log10",  # 3: log10 median of per-minute RSAM
    "rsam_p95_minute_log10",  # 4: log10 95th percentile of per-minute RSAM
    "rsam_peak_over_median_log10",  # 5: log10(max minute / median minute)
    "rsam_6h_over_24h_log10",  # 6: log10(last-6h RSAM / 24h RSAM)
    "rsam_1h_over_24h_log10",  # 7: log10(last-1h RSAM / 24h RSAM)
    "ssam_power_0p5_2hz_log10",  # 8: log10 mean velocity PSD, 0.5-2 Hz
    "ssam_power_2_5hz_log10",  # 9: 2-5 Hz
    "ssam_power_5_10hz_log10",  # 10: 5-10 Hz
    "ssam_power_10_20hz_log10",  # 11: 10-20 Hz (0 + flag when sr < 50)
    "ssam_ratio_lf_hf_log10",  # 12: log10(band(0.5-2)/band(5-10))
    "ssam_ratio_mid_hf_log10",  # 13: log10(band(2-5)/band(5-10))
    "ssam_ratio_lf_vhf_log10",  # 14: log10(band(0.5-2)/band(10-20))
    "sta_lta_triggers_log1p",  # 15: log1p(# STA/LTA triggers in the day)
    "tremor_frac_over_2x_day_median",  # 16: frac minutes > 2x day median
    "hourly_rsam_std_log10",  # 17: log10 std of hourly RSAM
    "hourly_rsam_max_over_median_log10",  # 18: log10(max hour / median hour)
    "rsam_over_quiet_baseline_log10",  # 19: log10(24h RSAM / causal baseline)
    "baseline_available_flag",  # 20: 1 when the causal baseline exists
    "rsam_delta_1d_log10",  # 21: log10(today 24h RSAM / previous sampled day)
    "delta_available_flag",  # 22: 1 when a prior day <= 7 d back was sampled
    "tremor_frac_over_3x_baseline",  # 23: frac minutes > 3x causal baseline
    "data_fraction_of_day",  # 24: fraction of the day with samples
    "gap_count_log1p",  # 25: log1p(# recording gaps)
    "station_is_broadband_flag",  # 26: 1 for BHZ, 0 for EHZ/SHZ
    "sampling_rate_over_100",  # 27: sample rate / 100
)
_VOLCANO_ONEHOT_START = len(_DAILY_FEATURE_NAMES)  # 28

FEATURE_NAMES: tuple[str, ...] = (
    _DAILY_FEATURE_NAMES
    + tuple(f"volcano_onehot_{v.name.lower().replace(' ', '_')}" for v in VOLCANOES)
    + tuple(
        f"reserved_zero_{i}" for i in range(_VOLCANO_ONEHOT_START + len(VOLCANOES), FEATURE_DIM)
    )
)

HOURLY_FEATURE_NAMES: tuple[str, ...] = (
    "hour_rsam_log10",  # 0
    "hour_rsam_over_day_median_log10",  # 1
    "hour_band_0p5_2hz_log10",  # 2
    "hour_band_2_5hz_log10",  # 3
    "hour_band_5_10hz_log10",  # 4
    "hour_band_10_20hz_log10",  # 5
    "hour_lf_hf_ratio_log10",  # 6
    "hour_sta_lta_triggers_log1p",  # 7
    "hour_tremor_frac_over_2x_day_median",  # 8
    "hour_peak_over_median_log10",  # 9
    "hour_data_fraction",  # 10
    "hour_rsam_over_quiet_baseline_log10",  # 11
    "hour_baseline_available_flag",  # 12
) + tuple(f"reserved_zero_{i}" for i in range(13, HOURLY_DIM))


# ---------------------------------------------------------------------------
# Small date helpers (dataset days are ISO "YYYY-MM-DD" strings throughout).
# ---------------------------------------------------------------------------


def _d(iso: str) -> _dt.date:
    """Parse an ISO date string."""
    return _dt.date.fromisoformat(iso)


def _daterange(first: _dt.date, last: _dt.date) -> list[_dt.date]:
    """Inclusive list of days from ``first`` to ``last``."""
    return [first + _dt.timedelta(days=i) for i in range((last - first).days + 1)]


# ---------------------------------------------------------------------------
# Labels: GVP eruption catalog -> onsets + eruptive exclusion intervals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EruptionOnset:
    """A day-precision eruption onset used as a label anchor."""

    volcano: str
    date: _dt.date
    vei: int | None
    day_uncertainty: int
    eruption_number: int


@dataclass
class VolcanoLabels:
    """Per-volcano label material derived from the GVP catalog.

    Attributes:
        onsets: Day-precision onsets inside the split years.
        eruptive: (start, end) closed intervals of ALL cataloged eruptions
            (any precision) used to exclude days from the negatives; ends
            unknown are conservatively extended by 365 days.
        excluded_onsets: Human-readable records of catalog rows that could
            not become onsets (month precision / high uncertainty).
    """

    onsets: list[EruptionOnset] = field(default_factory=list)
    eruptive: list[tuple[_dt.date, _dt.date]] = field(default_factory=list)
    excluded_onsets: list[str] = field(default_factory=list)


def _month_last_day(year: int, month: int) -> int:
    """Last day number of a month."""
    if month == 12:
        return 31
    return (_dt.date(year, month + 1, 1) - _dt.timedelta(days=1)).day


def parse_gvp_labels(csv_path: Path, today: _dt.date) -> dict[str, VolcanoLabels]:
    """Parse the GVP Holocene eruption CSV into per-volcano label material.

    Args:
        csv_path: Cached GVP WFS CSV export.
        today: Upper clamp for open-ended (continuing) eruptions.

    Returns:
        Mapping of volcano name -> :class:`VolcanoLabels`. Only rows whose
        start day is known with <= 2 days uncertainty become onsets; every
        row still contributes its eruptive interval to the exclusions.

    Raises:
        RuntimeError: If any named volcano has zero catalog rows (a parse or
            catalog regression must not silently produce an empty label set).
    """
    import pandas as pd

    frame = pd.read_csv(csv_path, low_memory=False)
    by_number = {v.gvp_number: v.name for v in VOLCANOES}
    out: dict[str, VolcanoLabels] = {v.name: VolcanoLabels() for v in VOLCANOES}
    sub = frame[frame["Volcano_Number"].isin(by_number)]
    if sub.empty:
        raise RuntimeError(f"GVP catalog contains no rows for volcano numbers {sorted(by_number)}")

    for _, row in sub.iterrows():
        name = by_number[int(row["Volcano_Number"])]
        labels = out[name]
        sy = row["StartDateYear"]
        if not np.isfinite(sy) or int(sy) < 1990:
            continue
        sy = int(sy)
        sm = int(row["StartDateMonth"]) if np.isfinite(row["StartDateMonth"]) else None
        sd = int(row["StartDateDay"]) if np.isfinite(row["StartDateDay"]) else None
        if sd is not None and sd == 0:
            sd = None
        unc = row["StartDateDayUncertainty"]
        unc_days = int(unc) if np.isfinite(unc) else 0

        # Eruptive interval (conservative when imprecise).
        start = _dt.date(sy, sm or 1, sd or 1)
        ey = row["EndDateYear"]
        if np.isfinite(ey):
            em = int(row["EndDateMonth"]) if np.isfinite(row["EndDateMonth"]) else 12
            if em == 0:
                em = 12
            ed_raw = row["EndDateDay"]
            ed = int(ed_raw) if np.isfinite(ed_raw) and int(ed_raw) > 0 else None
            end = _dt.date(int(ey), em, ed or _month_last_day(int(ey), em))
        else:
            end = start + _dt.timedelta(days=365)
        labels.eruptive.append((start, min(end, today)))

        vei = int(row["ExplosivityIndexMax"]) if np.isfinite(row["ExplosivityIndexMax"]) else None
        if sm is None or sd is None:
            labels.excluded_onsets.append(
                f"{name} eruption #{int(row['Eruption_Number'])} start {sy}-{sm or '??'}"
                f"-?? excluded: month/day-imprecise start"
            )
            continue
        if unc_days > 2:
            labels.excluded_onsets.append(
                f"{name} eruption #{int(row['Eruption_Number'])} start {start.isoformat()} "
                f"excluded: day uncertainty {unc_days} d > 2 d"
            )
            continue
        if sy in SPLIT.all_years:
            labels.onsets.append(
                EruptionOnset(
                    volcano=name,
                    date=start,
                    vei=vei,
                    day_uncertainty=unc_days,
                    eruption_number=int(row["Eruption_Number"]),
                )
            )
    for labels in out.values():
        labels.onsets.sort(key=lambda o: o.date)
        labels.eruptive.sort()
    return out


def label_day(
    day: _dt.date,
    onsets: list[EruptionOnset],
    eruptive: list[tuple[_dt.date, _dt.date]],
) -> tuple[str, int | None, int | None]:
    """Classify one (volcano, day) against the label windows.

    Args:
        day: Candidate sample day.
        onsets: Day-precision onsets for the volcano.
        eruptive: Eruptive (start, end) exclusion intervals.

    Returns:
        Tuple ``(cls, days_to_onset, vei)`` where ``cls`` is one of:

        * ``"positive"`` -- an onset begins within 1..K_DAYS days after
          ``day`` and ``day`` itself is not inside an eruption;
        * ``"eruptive"`` -- ``day`` lies inside a cataloged eruption
          (start..end inclusive): excluded from BOTH classes;
        * ``"buffer"`` -- within QUIET_BUFFER_DAYS of an eruptive interval
          (not positive): excluded from the negatives;
        * ``"negative"`` -- a quiet day.
    """
    for start, end in eruptive:
        if start <= day <= end:
            return "eruptive", None, None
    for onset in onsets:
        lead = (onset.date - day).days
        if 1 <= lead <= K_DAYS:
            return "positive", lead, onset.vei
    for start, end in eruptive:
        gap = (start - day).days if day < start else (day - end).days
        if 0 <= gap < QUIET_BUFFER_DAYS:
            return "buffer", None, None
    return "negative", None, None


# ---------------------------------------------------------------------------
# EarthScope inventory + availability + waveform fetch (budgeted).
# ---------------------------------------------------------------------------


def _parse_channel_sensitivities(
    inventory_path: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Parse FDSN station text inventory into per-(station, channel) epochs.

    Returns:
        Mapping ``(sta, cha) -> [{"start", "end", "scale", "sample_rate"}]``
        with ISO date strings ("9999-12-31" for open epochs) and the overall
        sensitivity in counts per m/s.

    Raises:
        RuntimeError: On a row whose sensitivity is missing or zero -- raw
            counts cannot be honestly compared across instruments without it.
    """
    epochs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in inventory_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("|")
        sta, cha = f[1], f[3]
        scale = float(f[11]) if f[11] else 0.0
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(
                f"inventory row {sta}.{cha} has no usable sensitivity ({f[11]!r}); "
                "refusing to mix un-scalable raw counts into the features"
            )
        epochs.setdefault((sta, cha), []).append(
            {
                "start": f[15][:10],
                "end": f[16][:10] if f[16].strip() else "9999-12-31",
                "scale": scale,
                "sample_rate": float(f[14]),
            }
        )
    return epochs


def _fetch_availability(ctx: PipelineContext, sta: str, cha: str) -> list[tuple[str, str]]:
    """Fetch (and cache) the recorded-data spans for one station-channel.

    A 404 from the FDSN availability service means "no data matched", which
    is a legitimate, recordable absence -- cached as an empty file so reruns
    do not re-ask.

    Returns:
        List of (earliest, latest) ISO timestamp strings.
    """
    import requests

    dest = ctx.data_dir / "volcanic" / "availability" / f"AV_{sta}_{cha}.txt"
    if not dest.exists():
        first = _dt.date(min(SPLIT.all_years), 1, 1)
        last = _dt.date(max(SPLIT.all_years), 12, 31)
        url = (
            f"{FDSN_BASE}/availability/1/query?net=AV&sta={sta}&cha={cha}"
            f"&starttime={first}T00:00:00&endtime={last}T23:59:59&format=text&mergegaps=3600"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            body = http_get_with_retry(url, timeout=120)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                body = b""
            else:
                raise
        dest.write_bytes(body)
    spans: list[tuple[str, str]] = []
    for line in dest.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        spans.append((f[6], f[7]))
    return spans


def _coverage_by_day(spans: list[tuple[str, str]]) -> dict[_dt.date, float]:
    """Seconds of recorded data per UTC day, computed once per channel."""
    coverage: dict[_dt.date, float] = {}
    for a, b in spans:
        t0 = _dt.datetime.fromisoformat(a.replace("Z", "+00:00"))
        t1 = _dt.datetime.fromisoformat(b.replace("Z", "+00:00"))
        day = t0.date()
        while day <= t1.date():
            d0 = _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.UTC)
            d1 = d0 + _dt.timedelta(days=1)
            lo, hi = max(t0, d0), min(t1, d1)
            if hi > lo:
                coverage[day] = coverage.get(day, 0.0) + (hi - lo).total_seconds()
            day += _dt.timedelta(days=1)
    return coverage


def _waveform_path(ctx: PipelineContext, sta: str, cha: str, day: _dt.date) -> Path:
    """Cache path of one station-day miniSEED file."""
    return ctx.data_dir / "volcanic" / "waveforms" / f"AV_{sta}_{cha}_{day.isoformat()}.mseed"


def _dataselect_url(sta: str, cha: str, day: _dt.date) -> str:
    """FDSN dataselect URL for one full UTC station-day."""
    nxt = day + _dt.timedelta(days=1)
    return (
        f"{FDSN_BASE}/dataselect/1/query?net=AV&sta={sta}&cha={cha}"
        f"&starttime={day.isoformat()}T00:00:00&endtime={nxt.isoformat()}T00:00:00"
    )


def _fetch_waveform_day(ctx: PipelineContext, sta: str, cha: str, day: _dt.date) -> Path | None:
    """Fetch one station-day of miniSEED through the allowlisted gate.

    Returns:
        The cached path, or None when the datacenter holds no data for the
        day (recorded as a ``.absent`` marker so reruns skip the request).
    """
    import requests

    dest = _waveform_path(ctx, sta, cha, day)
    absent = dest.with_suffix(".absent")
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if absent.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = http_get_with_retry(_dataselect_url(sta, cha, day), timeout=300)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            absent.write_bytes(b"")
            return None
        raise
    if not body:
        absent.write_bytes(b"")
        return None
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# HANS Volcanic Activity Notice cross-check (POST endpoint; documented).
# ---------------------------------------------------------------------------


def _hans_van_escalations(
    volc_cd: str, earliest_needed: _dt.date, max_pages: int = 40
) -> list[dict[str, str]]:
    """Pull VAN notices for one volcano and extract alert-level statements.

    Uses ``SafeHTTPClient.post_json`` directly because the HANS search API is
    POST-only while ``http_get_with_retry`` is a GET transport; the host
    (``volcanoes.usgs.gov``) is on the same trusted allowlist and all
    SafeHTTPClient SSRF gates run unchanged.

    Args:
        volc_cd: HANS volcano code (e.g. ``ak252``).
        earliest_needed: Stop paging once notices older than this date have
            been reached (the archive is served newest-first, 20 per page).
        max_pages: Hard page cap.

    Returns:
        List of ``{"sent_utc", "alert_level", "color_code"}`` records.
    """
    import time

    import requests

    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    level_re = re.compile(r"Current Volcano Alert Level:\s*([A-Z]+)", re.IGNORECASE)
    color_re = re.compile(r"Current Aviation Color Code:\s*([A-Z]+)", re.IGNORECASE)
    records: list[dict[str, str]] = []
    for page in range(max_pages):
        payload: dict[str, Any] | None = None
        last_exc: Exception | None = None
        for attempt in range(4):  # POST transport has no built-in retry
            try:
                payload = SafeHTTPClient.post_json(
                    HANS_SEARCH_URL,
                    json_body={
                        "obsAbbr": "avo",
                        "volcCd": volc_cd,
                        "noticeTypeCd": "VAN",
                        "pageIndex": page,
                    },
                    timeout=60,
                )
                break
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in (408, 425, 429, 500, 502, 503, 504):
                    raise
                last_exc = exc
                time.sleep(2.0 ** (attempt + 1))
        if payload is None:
            raise RuntimeError(
                f"HANS VAN search for {volc_cd} page {page} failed after retries"
            ) from last_exc
        notices = payload.get("noticeData") or []
        oldest: _dt.date | None = None
        for notice in notices:
            html = str(notice.get("noticeHtml", ""))
            level = level_re.search(html)
            color = color_re.search(html)
            sent = str(notice.get("sentUtc", ""))
            records.append(
                {
                    "sent_utc": sent,
                    "alert_level": level.group(1).upper() if level else "",
                    "color_code": color.group(1).upper() if color else "",
                }
            )
            if len(sent) >= 10:
                sent_date = _d(sent[:10])
                oldest = sent_date if oldest is None else min(oldest, sent_date)
        if len(notices) < 20:
            break
        if oldest is not None and oldest < earliest_needed:
            break
    return records


def _crosscheck_onsets_with_hans(
    ctx: PipelineContext,
    labels: dict[str, VolcanoLabels],
) -> list[dict[str, Any]]:
    """Cross-check uncertain (1-2 day) onsets against HANS VAN escalations.

    For each onset with nonzero day uncertainty, finds the nearest VAN whose
    alert level is WATCH or WARNING and records the time delta. GVP
    day-precision rows stay authoritative; an uncertain onset whose nearest
    escalation is more than 3 days away is NOT corroborated and is demoted
    to the excluded list (its eruptive interval still blocks negatives) --
    reported loudly, never trained on. Example caught in the first real run:
    Semisopochnoi 2021-02-02 (+/-2 d), whose activity was recognized
    retrospectively and whose VAN escalation came 6 days later. Check
    results are cached on disk (the HANS search endpoint is POST-only and
    not byte-cached) and re-applied deterministically on reruns.

    Returns:
        The per-onset check records (also merged into the manifest).
    """
    cache = ctx.data_dir / "volcanic" / "hans_crosschecks.json"
    checks: list[dict[str, Any]]
    if cache.exists():
        checks = json.loads(cache.read_text())
    else:
        by_name = {v.name: v for v in VOLCANOES}
        checks = []
        for name, material in labels.items():
            uncertain = [o for o in material.onsets if o.day_uncertainty > 0]
            if not uncertain:
                continue
            earliest_needed = min(o.date for o in uncertain) - _dt.timedelta(days=30)
            vans = _hans_van_escalations(by_name[name].hans_cd, earliest_needed)
            escalations = [
                v for v in vans if v["alert_level"] in ("WATCH", "WARNING") and v["sent_utc"]
            ]
            for onset in uncertain:
                deltas = [abs((_d(v["sent_utc"][:10]) - onset.date).days) for v in escalations]
                nearest = min(deltas) if deltas else None
                checks.append(
                    {
                        "volcano": name,
                        "onset": onset.date.isoformat(),
                        "day_uncertainty": onset.day_uncertainty,
                        "nearest_van_escalation_days": nearest,
                        "n_van_notices_scanned": len(vans),
                        "corroborated": bool(nearest is not None and nearest <= 3),
                    }
                )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(checks, indent=1))

    # Apply exclusions (cached or fresh) so reruns are deterministic.
    for check in checks:
        if check["corroborated"]:
            continue
        material = labels[check["volcano"]]
        onset_date = _d(check["onset"])
        material.onsets = [o for o in material.onsets if o.date != onset_date]
        material.excluded_onsets.append(
            f"{check['volcano']} onset {check['onset']} excluded: GVP day uncertainty "
            f"{check['day_uncertainty']} d and nearest HANS WATCH/WARNING VAN "
            f"{check['nearest_van_escalation_days']} d away (> 3 d) -- uncorroborated"
        )
        logger.warning("label excluded by HANS cross-check: %s", material.excluded_onsets[-1])
    return checks


# ---------------------------------------------------------------------------
# Waveform -> per-day feature extraction (obspy lazy-imported).
# ---------------------------------------------------------------------------


def _sta_lta_trigger_count(
    velocity: np.ndarray, sr: float, on: float = 3.5, off: float = 1.5
) -> int:
    """Count STA/LTA trigger onsets in one contiguous velocity segment.

    Classic 1 s / 60 s ratio via cumulative sums; a trigger begins when the
    ratio crosses ``on`` and re-arms after it falls below ``off``.
    """
    n_sta = max(int(sr * 1.0), 1)
    n_lta = max(int(sr * 60.0), n_sta * 2)
    if velocity.size < n_lta + n_sta:
        return 0
    a = np.abs(velocity)
    csum = np.concatenate(([0.0], np.cumsum(a, dtype=np.float64)))
    sta = (csum[n_sta:] - csum[:-n_sta]) / n_sta
    lta = (csum[n_lta:] - csum[:-n_lta]) / n_lta
    # Align: STA at sample i uses [i-n_sta, i); LTA must END there too so no
    # event energy pollutes its own baseline.
    sta = sta[n_lta - n_sta :]
    lta = lta[: sta.size]
    ratio = sta / np.maximum(lta, _EPS)
    count = 0
    armed = True
    for r in ratio[:: max(int(sr * 0.5), 1)]:  # 0.5 s decimation for speed
        if armed and r >= on:
            count += 1
            armed = False
        elif not armed and r <= off:
            armed = True
    return count


def _band_powers(velocity: np.ndarray, sr: float) -> np.ndarray:
    """Mean velocity PSD in the four SSAM bands via segmented periodograms.

    Returns:
        Array of 4 band powers ((m/s)^2/Hz); NaN for bands above Nyquist or
        when the segment is too short.
    """
    out = np.full(len(SSAM_BANDS), np.nan)
    seg = 4096
    if velocity.size < seg:
        return out
    n_segs = velocity.size // seg
    data = velocity[: n_segs * seg].reshape(n_segs, seg)
    data = data - data.mean(axis=1, keepdims=True)
    window = np.hanning(seg)
    spec = np.fft.rfft(data * window, axis=1)
    # One-sided PSD with Hann window power correction.
    psd = (np.abs(spec) ** 2) * (2.0 / (sr * (window**2).sum()))
    psd = psd.mean(axis=0)
    freqs = np.fft.rfftfreq(seg, d=1.0 / sr)
    nyquist = sr / 2.0
    for i, (lo, hi) in enumerate(SSAM_BANDS):
        if lo >= nyquist:
            continue
        mask = (freqs >= lo) & (freqs < min(hi, nyquist))
        if mask.any():
            out[i] = float(psd[mask].mean())
    return out


def _log10(x: float) -> float:
    """log10 with an epsilon floor."""
    return float(np.log10(max(float(x), _EPS)))


def compute_day_record(mseed_bytes: bytes, scale: float, expected_sr: float) -> dict[str, Any]:
    """Compute the per-day raw feature record from one station-day miniSEED.

    This is the ONLY function that touches waveform bytes; everything it
    returns is derived solely from the given day (no cross-day state), so
    the assembled features cannot look ahead by construction.

    Args:
        mseed_bytes: Raw miniSEED bytes for one UTC station-day.
        scale: Instrument overall sensitivity (counts per m/s). Counts are
            divided by this flat-response approximation (documented: no full
            deconvolution -- all downstream use is ratio/log based).
        expected_sr: Sample rate from the inventory (sanity check).

    Returns:
        Dict with per-minute RSAM (1440, NaN where absent), hourly features
        (24 x HOURLY_DIM building blocks), day-level statistics, and
        presence metadata. See ``FEATURE_NAMES``/``HOURLY_FEATURE_NAMES``.

    Raises:
        RuntimeError: If the payload cannot be parsed as miniSEED or NO
            trace matches the inventory sample rate within 1%. Individual
            traces at a different rate (observed in the wild: 25 Hz segments
            inside a nominally 50 Hz CERB.SHZ epoch) are dropped, and the
            minutes they covered stay honestly absent (NaN -> presence
            flags); they are never index-mapped at the wrong rate.
    """
    import obspy  # type: ignore[import-untyped, unused-ignore]  # lazy heavy import

    try:
        stream = obspy.read(io.BytesIO(mseed_bytes))
    except Exception as exc:  # obspy raises TypeError on unknown formats
        raise RuntimeError(f"unreadable miniSEED payload: {exc}") from exc

    traces = [
        tr
        for tr in stream
        if abs(float(tr.stats.sampling_rate) - expected_sr) <= 0.01 * expected_sr
    ]
    if not traces:
        rates = sorted({float(tr.stats.sampling_rate) for tr in stream})
        raise RuntimeError(
            f"sample rate {rates} disagrees with inventory {expected_sr}; refusing to mis-scale"
        )
    day0 = min(tr.stats.starttime for tr in traces)
    day_start = obspy.UTCDateTime(day0.year, day0.month, day0.day)

    minute_sum = np.zeros(1440, dtype=np.float64)
    minute_count = np.zeros(1440, dtype=np.int64)
    hour_trigs = np.zeros(24, dtype=np.int64)
    hour_band_acc: list[list[np.ndarray]] = [[] for _ in range(24)]
    sr = float(traces[0].stats.sampling_rate)
    gap_count = max(len(traces) - 1, 0)

    for tr in traces:
        v = np.asarray(tr.data, dtype=np.float64) / scale
        offset = float(tr.stats.starttime - day_start)
        idx_min = (offset + np.arange(v.size) / sr) / 60.0
        minute_idx = np.clip(idx_min.astype(np.int64), 0, 1439)
        np.add.at(minute_sum, minute_idx, np.abs(v))
        np.add.at(minute_count, minute_idx, 1)
        # Hour-sliced triggers and band powers.
        start_hour = int(offset // 3600)
        for hour in range(max(start_hour, 0), 24):
            h0, h1 = hour * 3600.0, (hour + 1) * 3600.0
            i0 = max(int((h0 - offset) * sr), 0)
            i1 = min(int((h1 - offset) * sr), v.size)
            if i1 - i0 < sr * 120:  # need >= 2 minutes to say anything
                continue
            seg = v[i0:i1]
            hour_trigs[hour] += _sta_lta_trigger_count(seg, sr)
            hour_band_acc[hour].append(_band_powers(seg, sr))

    minute_rsam = np.where(minute_count > 0, minute_sum / np.maximum(minute_count, 1), np.nan)
    observed = minute_rsam[np.isfinite(minute_rsam)]
    if observed.size < 60:
        raise RuntimeError("fewer than 60 minutes of data in the day; treat as absent upstream")

    hour_bands = np.full((24, len(SSAM_BANDS)), np.nan)
    for hour in range(24):
        if hour_band_acc[hour]:
            hour_bands[hour] = np.nanmean(np.stack(hour_band_acc[hour]), axis=0)

    hour_rsam = np.full(24, np.nan)
    hour_frac = np.zeros(24)
    day_median = float(np.median(observed))
    hour_tremor = np.zeros(24)
    hour_peak_med = np.zeros(24)
    for hour in range(24):
        chunk = minute_rsam[hour * 60 : (hour + 1) * 60]
        good = chunk[np.isfinite(chunk)]
        hour_frac[hour] = good.size / 60.0
        if good.size:
            hour_rsam[hour] = float(good.mean())
            hour_tremor[hour] = float(np.mean(good > 2.0 * day_median))
            hour_peak_med[hour] = _log10(good.max() / max(np.median(good), _EPS))

    day_bands = np.nanmean(hour_bands, axis=0)
    return {
        "minute_rsam": minute_rsam,
        "hour_rsam": hour_rsam,
        "hour_bands": hour_bands,
        "hour_trigs": hour_trigs,
        "hour_frac": hour_frac,
        "hour_tremor": hour_tremor,
        "hour_peak_med": hour_peak_med,
        "day_bands": day_bands,
        "day_median_minute": day_median,
        "rsam_24h": float(observed.mean()),
        "rsam_p95": float(np.percentile(observed, 95)),
        "rsam_peak": float(observed.max()),
        "trigger_count": int(hour_trigs.sum()),
        "gap_count": int(gap_count),
        "data_fraction": float(observed.size / 1440.0),
        "sampling_rate": sr,
    }


def _day_record_cache_path(ctx: PipelineContext, sta: str, cha: str, day: _dt.date) -> Path:
    """Cache path for one computed day record."""
    return (
        ctx.data_dir
        / "volcanic"
        / "day_records"
        / f"AV_{sta}_{cha}_{day.isoformat()}_{FEATURE_SPEC_VERSION}.npz"
    )


def _load_or_compute_day_record(
    ctx: PipelineContext, sta: str, cha: str, day: _dt.date, scale: float, sr: float
) -> dict[str, Any] | None:
    """Load a cached day record, computing it from the cached waveform once."""
    cache = _day_record_cache_path(ctx, sta, cha, day)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    wf = _waveform_path(ctx, sta, cha, day)
    if not wf.exists() or wf.stat().st_size == 0:
        return None
    try:
        rec = compute_day_record(wf.read_bytes(), scale=scale, expected_sr=sr)
    except RuntimeError as exc:
        logger.warning("day record %s.%s %s unusable: %s", sta, cha, day, exc)
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(cache, **arrays)  # type: ignore[arg-type, unused-ignore]
    return rec


# ---------------------------------------------------------------------------
# Sample planning (which station-days to fetch) -- deterministic + budgeted.
# ---------------------------------------------------------------------------


@dataclass
class PlannedDay:
    """One planned (volcano, day) sample with its chosen station-channel."""

    volcano: str
    day: _dt.date
    sta: str
    cha: str
    scale: float
    sample_rate: float
    cls: str  # "positive" | "negative"
    days_to_onset: int | None
    vei: int | None


def _split_of_year(year: int) -> str | None:
    """Name of the split a year belongs to, or None."""
    if year in SPLIT.train_years:
        return "train"
    if year in SPLIT.val_years:
        return "val"
    if year in SPLIT.test_years:
        return "test"
    return None


def _epoch_for(
    epochs: dict[tuple[str, str], list[dict[str, Any]]], sta: str, cha: str, day: _dt.date
) -> dict[str, Any] | None:
    """Inventory epoch covering a day for a station-channel, if any."""
    for ep in epochs.get((sta, cha), []):
        if _d(ep["start"]) <= day <= _d(ep["end"]):
            return ep
    return None


def plan_samples(
    ctx: PipelineContext,
    labels: dict[str, VolcanoLabels],
    epochs: dict[tuple[str, str], list[dict[str, Any]]],
    availability: dict[tuple[str, str], list[tuple[str, str]]],
) -> tuple[list[PlannedDay], dict[str, Any]]:
    """Choose the (volcano, day, station) samples to fetch, within budget.

    Positives are every day in the K-day window before each onset. Negatives
    are quiet days (>= 60 d from any eruptive interval) sampled evenly at
    ``NEG_PER_YEAR`` per volcano-year. For each day the first station in the
    volcano's preference list with >= 6 h of recorded data is chosen; days
    with no covered station are dropped and reported (never imputed).

    Returns:
        Tuple of (planned samples, plan report incl. dropped onsets/days).
    """
    plan: list[PlannedDay] = []
    planned_keys: set[tuple[str, str]] = set()
    report: dict[str, Any] = {"dropped_positive_days": [], "onsets": [], "neg_days_planned": 0}
    coverage = {key: _coverage_by_day(spans) for key, spans in availability.items()}

    def _pick_station(
        volcano: VolcanoSpec, day: _dt.date
    ) -> tuple[str, str, dict[str, Any]] | None:
        for sta, cha in volcano.stations:
            ep = _epoch_for(epochs, sta, cha, day)
            if ep is None:
                continue
            if coverage.get((sta, cha), {}).get(day, 0.0) >= 6 * 3600:
                return sta, cha, ep
        return None

    for volcano in VOLCANOES:
        material = labels[volcano.name]
        for onset in material.onsets:
            split = _split_of_year(onset.date.year)
            if split is None:
                continue
            window = [onset.date - _dt.timedelta(days=k) for k in range(1, K_DAYS + 1)]
            planned_here = 0
            for day in window:
                if day.year not in SPLIT.all_years:
                    continue  # window tail precedes the split era
                if (volcano.name, day.isoformat()) in planned_keys:
                    continue  # already planned by an overlapping onset window
                cls, lead, vei = label_day(day, material.onsets, material.eruptive)
                if cls != "positive":
                    continue  # e.g. the window overlaps a previous eruption
                picked = _pick_station(volcano, day)
                if picked is None:
                    report["dropped_positive_days"].append(
                        f"{volcano.name} {day.isoformat()} (onset {onset.date.isoformat()}): "
                        "no station with >=6h data"
                    )
                    continue
                sta, cha, ep = picked
                plan.append(
                    PlannedDay(
                        volcano.name,
                        day,
                        sta,
                        cha,
                        ep["scale"],
                        ep["sample_rate"],
                        "positive",
                        lead,
                        vei,
                    )
                )
                planned_keys.add((volcano.name, day.isoformat()))
                planned_here += 1
            report["onsets"].append(
                {
                    "volcano": volcano.name,
                    "onset": onset.date.isoformat(),
                    "split": split,
                    "vei": onset.vei,
                    "positive_days_planned": planned_here,
                }
            )

        # Negatives: quiet days sampled evenly per year.
        for year in SPLIT.all_years:
            split = _split_of_year(year)
            if split is None:
                continue
            quiet: list[tuple[_dt.date, str, str, dict[str, Any]]] = []
            for day in _daterange(_dt.date(year, 1, 1), _dt.date(year, 12, 31)):
                cls, _, _ = label_day(day, material.onsets, material.eruptive)
                if cls != "negative":
                    continue
                picked = _pick_station(volcano, day)
                if picked is not None:
                    quiet.append((day, *picked))
            n_want = NEG_PER_YEAR[split]
            if not quiet:
                continue
            step = max(len(quiet) // n_want, 1)
            for day, sta, cha, ep in quiet[::step][:n_want]:
                plan.append(
                    PlannedDay(
                        volcano.name,
                        day,
                        sta,
                        cha,
                        ep["scale"],
                        ep["sample_rate"],
                        "negative",
                        None,
                        None,
                    )
                )
                report["neg_days_planned"] += 1

    plan.sort(key=lambda p: (p.volcano, p.day))
    return plan, report


# ---------------------------------------------------------------------------
# Stage: fetch
# ---------------------------------------------------------------------------


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download labels, inventory, availability, HANS cross-checks, waveforms.

    Everything is disk-cached; reruns only fill gaps. The waveform transfer
    is hard-capped at ``WAVEFORM_BYTE_BUDGET`` NEW bytes per invocation:
    positives are fetched first (a budget that cannot cover the label
    windows fails loud), then negatives round-robin until the budget ends.

    Returns:
        Manifest dict (also written to ``<data_dir>/volcanic/manifest.json``).

    Raises:
        RuntimeError: If the byte budget cannot cover the positive windows,
            or a split ends up with no eruption onsets at all.
    """
    vol_dir = ctx.data_dir / "volcanic"
    vol_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today()

    gvp_path = cached_fetch(GVP_ERUPTIONS_URL, vol_dir / "gvp_holocene_eruptions.csv", timeout=300)
    inventory_path = cached_fetch(STATION_URL, vol_dir / "av_channel_inventory.txt", timeout=120)
    hans_volcanoes_path = cached_fetch(HANS_VOLCANO_URL, vol_dir / "hans_us_volcanoes.json")

    sources: list[dict[str, Any]] = [
        {
            "url": GVP_ERUPTIONS_URL,
            "sha256": sha256_file(gvp_path),
            "description": "Smithsonian GVP Holocene eruption catalog (labels: onsets, VEI)",
        },
        {
            "url": STATION_URL,
            "sha256": sha256_file(inventory_path),
            "description": "EarthScope FDSN AV-network channel inventory (sensitivities)",
        },
        {
            "url": HANS_VOLCANO_URL,
            "sha256": sha256_file(hans_volcanoes_path),
            "description": "USGS HANS volcano list (volcano-code join for VAN cross-check)",
        },
    ]

    labels = parse_gvp_labels(gvp_path, today)
    hans_checks = _crosscheck_onsets_with_hans(ctx, labels)

    epochs = _parse_channel_sensitivities(inventory_path)
    availability: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for volcano in VOLCANOES:
        for sta, cha in volcano.stations:
            availability[(sta, cha)] = _fetch_availability(ctx, sta, cha)

    plan, plan_report = plan_samples(ctx, labels, epochs, availability)

    onset_splits = {o["split"] for o in plan_report["onsets"] if o["positive_days_planned"] > 0}
    for split in ("train", "val", "test"):
        if split not in onset_splits:
            raise RuntimeError(
                f"split '{split}' contains no eruption onsets with usable station coverage; "
                "the temporal split must be re-designed, not silently trained around"
            )

    wf_dir = ctx.data_dir / "volcanic" / "waveforms"
    cache_bytes = sum(f.stat().st_size for f in wf_dir.glob("*.mseed")) if wf_dir.exists() else 0
    fetched_bytes = 0
    fetched_days = 0
    absent_days = 0
    budget_hit = False
    positives = [p for p in plan if p.cls == "positive"]
    # Negatives are fetched round-robin across (volcano, year) groups so a
    # budget stop degrades every volcano's quiet-day coverage evenly instead
    # of starving the alphabetically last volcanoes of their causal RSAM
    # baselines.
    neg_groups: dict[tuple[str, int], list[PlannedDay]] = {}
    for p in plan:
        if p.cls == "negative":
            neg_groups.setdefault((p.volcano, p.day.year), []).append(p)
    negatives: list[PlannedDay] = []
    round_i = 0
    while any(neg_groups.values()):
        for key in sorted(neg_groups):
            group = neg_groups[key]
            if round_i < len(group):
                negatives.append(group[round_i])
        round_i += 1
        if round_i > max(len(g) for g in neg_groups.values()):
            break
    for p in positives + negatives:  # positives first: they must all fit
        if budget_hit:
            break
        pre_existing = _waveform_path(ctx, p.sta, p.cha, p.day).exists()
        path = _fetch_waveform_day(ctx, p.sta, p.cha, p.day)
        if path is None:
            absent_days += 1
            continue
        fetched_days += 1
        if not pre_existing:
            fetched_bytes += path.stat().st_size
            cache_bytes += path.stat().st_size
        if cache_bytes > WAVEFORM_BYTE_BUDGET:
            if p.cls == "positive":
                raise RuntimeError(
                    f"waveform budget {WAVEFORM_BYTE_BUDGET} B exhausted while fetching "
                    "POSITIVE windows -- the plan cannot be trained honestly; raise the "
                    "budget or shrink the volcano set explicitly"
                )
            logger.warning(
                "cumulative waveform budget reached at %.2f GB cache; remaining "
                "negatives are skipped",
                cache_bytes / 1e9,
            )
            budget_hit = True

    plan_json = [
        {
            "volcano": p.volcano,
            "day": p.day.isoformat(),
            "sta": p.sta,
            "cha": p.cha,
            "scale": p.scale,
            "sample_rate": p.sample_rate,
            "cls": p.cls,
            "days_to_onset": p.days_to_onset,
            "vei": p.vei,
        }
        for p in plan
    ]
    (vol_dir / "sample_plan.json").write_text(json.dumps(plan_json, indent=1))

    # Aggregate waveform provenance: one entry per station-channel with a
    # digest over the per-day file digests (the full per-day table would be
    # ~10^3 rows; the detailed list is reproducible from the plan + cache).
    per_chan: dict[tuple[str, str], list[str]] = {}
    for p in plan:
        wf = _waveform_path(ctx, p.sta, p.cha, p.day)
        if wf.exists() and wf.stat().st_size > 0:
            per_chan.setdefault((p.sta, p.cha), []).append(sha256_file(wf))
    for (sta, cha), digests in sorted(per_chan.items()):
        agg = hashlib.sha256("".join(sorted(digests)).encode()).hexdigest()
        sources.append(
            {
                "url": f"{FDSN_BASE}/dataselect/1/query?net=AV&sta={sta}&cha={cha}&<per-day>",
                "sha256": agg,
                "description": (
                    # nosec B608 - provenance text for a manifest: "dataselect"
                    # is the FDSN endpoint name (bandit's SQL heuristic matches
                    # the substring "select"); nothing here builds or executes
                    # a query, and sta/cha come from the fixed volcano table.
                    f"EarthScope FDSN dataselect miniSEED, AV.{sta}.{cha}, "  # nosec B608
                    f"{len(digests)} station-days (sha256 over sorted per-day sha256s; "
                    "per-day URLs reconstructable from sample_plan.json)"
                ),
            }
        )

    manifest = {
        "hook": HOOK_NAME,
        "feature_spec": FEATURE_SPEC_VERSION,
        "volcanoes": [v.name for v in VOLCANOES],
        "sources": sources,
        "hans_onset_crosschecks": hans_checks,
        "excluded_onsets": {n: m.excluded_onsets for n, m in labels.items()},
        "plan_report": plan_report,
        "fetched_station_days": fetched_days,
        "absent_station_days": absent_days,
        "new_bytes_this_run": fetched_bytes,
        "waveform_cache_bytes_total": cache_bytes,
    }
    (vol_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "fetch complete: %d station-days cached (%d absent upstream), %.2f GB new",
        fetched_days,
        absent_days,
        fetched_bytes / 1e9,
    )
    return manifest


# ---------------------------------------------------------------------------
# Stage: build_dataset
# ---------------------------------------------------------------------------


@dataclass
class VolcanicDataset:
    """Assembled (volcano, day) samples ready for training/evaluation.

    Attributes:
        features: (N, 128) raw (un-standardized) ``volcano-seismic-v1``.
        hourly: (N, 24, 32) raw hourly sequences for the swarm LSTM.
        labels: (N,) 1 = eruption onset within K_DAYS days.
        days_to_onset: (N,) lead days for positives, 0 for negatives.
        vei: (N,) VEI of the upcoming eruption, -1 where absent/negative.
        years: (N,) calendar year of the sample day (split key).
        volcano_idx: (N,) index into ``VOLCANOES``.
        day_iso: Sample days (ISO strings).
        minute_rsam: (N, 1440) per-minute velocity RSAM (physics-path input).
        feature_mean / feature_std: TRAIN-year standardization statistics.
        hourly_mean / hourly_std: TRAIN-year hourly-dim statistics.
        meta: Per-sample dicts (station, channel, class, onset).
    """

    features: np.ndarray
    hourly: np.ndarray
    labels: np.ndarray
    days_to_onset: np.ndarray
    vei: np.ndarray
    years: np.ndarray
    volcano_idx: np.ndarray
    day_iso: list[str]
    minute_rsam: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    hourly_mean: np.ndarray
    hourly_std: np.ndarray
    meta: list[dict[str, Any]]


def assemble_feature_vector(
    rec: dict[str, Any],
    *,
    volcano_index: int,
    is_broadband: bool,
    baseline: float | None,
    prev_rsam: float | None,
) -> np.ndarray:
    """Pack one day record + causal context into the 128-dim feature vector.

    Args:
        rec: Output of :func:`compute_day_record` for the sample day.
        volcano_index: Index into ``VOLCANOES`` (one-hot position).
        is_broadband: Channel is BHZ.
        baseline: Causal quiet-day RSAM baseline (median of previously
            sampled quiet days at the same station-channel), or None when
            fewer than 3 prior quiet days exist -- flagged, never imputed.
        prev_rsam: 24h RSAM of the most recent sampled day <= 7 days back at
            the same station-channel, or None -- flagged, never imputed.

    Returns:
        (128,) float32 vector per ``FEATURE_NAMES`` (see module docstring).
    """
    minute = np.asarray(rec["minute_rsam"], dtype=np.float64)
    observed = minute[np.isfinite(minute)]
    hour_rsam = np.asarray(rec["hour_rsam"], dtype=np.float64)
    good_hours = hour_rsam[np.isfinite(hour_rsam)]
    day_bands = np.asarray(rec["day_bands"], dtype=np.float64)
    rsam_24h = float(rec["rsam_24h"])

    last6 = minute[-360:]
    last6 = last6[np.isfinite(last6)]
    last1 = minute[-60:]
    last1 = last1[np.isfinite(last1)]
    rsam_6h = float(last6.mean()) if last6.size else rsam_24h
    rsam_1h = float(last1.mean()) if last1.size else rsam_24h

    day_median = float(rec["day_median_minute"])
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    vec[0] = _log10(rsam_24h)
    vec[1] = _log10(rsam_6h)
    vec[2] = _log10(rsam_1h)
    vec[3] = _log10(day_median)
    vec[4] = _log10(float(rec["rsam_p95"]))
    vec[5] = _log10(float(rec["rsam_peak"]) / max(day_median, _EPS))
    vec[6] = _log10(rsam_6h / max(rsam_24h, _EPS))
    vec[7] = _log10(rsam_1h / max(rsam_24h, _EPS))
    for i in range(4):
        vec[8 + i] = _log10(day_bands[i]) if np.isfinite(day_bands[i]) else 0.0
    lf, mid, hf, vhf = day_bands
    vec[12] = _log10(lf / max(hf, _EPS)) if np.isfinite(lf) and np.isfinite(hf) else 0.0
    vec[13] = _log10(mid / max(hf, _EPS)) if np.isfinite(mid) and np.isfinite(hf) else 0.0
    vec[14] = _log10(lf / max(vhf, _EPS)) if np.isfinite(lf) and np.isfinite(vhf) else 0.0
    vec[15] = float(np.log1p(float(rec["trigger_count"])))
    vec[16] = float(np.mean(observed > 2.0 * day_median)) if observed.size else 0.0
    vec[17] = _log10(float(good_hours.std())) if good_hours.size > 1 else 0.0
    vec[18] = (
        _log10(float(good_hours.max()) / max(float(np.median(good_hours)), _EPS))
        if good_hours.size
        else 0.0
    )
    if baseline is not None:
        vec[19] = _log10(rsam_24h / max(baseline, _EPS))
        vec[20] = 1.0
        vec[23] = float(np.mean(observed > 3.0 * baseline)) if observed.size else 0.0
    if prev_rsam is not None:
        vec[21] = _log10(rsam_24h / max(prev_rsam, _EPS))
        vec[22] = 1.0
    vec[24] = float(rec["data_fraction"])
    vec[25] = float(np.log1p(float(rec["gap_count"])))
    vec[26] = 1.0 if is_broadband else 0.0
    vec[27] = float(rec["sampling_rate"]) / 100.0
    vec[_VOLCANO_ONEHOT_START + volcano_index] = 1.0
    return vec


def assemble_hourly_matrix(rec: dict[str, Any], baseline: float | None) -> np.ndarray:
    """Pack one day record into the (24, 32) swarm-LSTM input sequence."""
    hour_rsam = np.asarray(rec["hour_rsam"], dtype=np.float64)
    hour_bands = np.asarray(rec["hour_bands"], dtype=np.float64)
    hour_trigs = np.asarray(rec["hour_trigs"], dtype=np.float64)
    hour_frac = np.asarray(rec["hour_frac"], dtype=np.float64)
    hour_tremor = np.asarray(rec["hour_tremor"], dtype=np.float64)
    hour_peak = np.asarray(rec["hour_peak_med"], dtype=np.float64)
    good = hour_rsam[np.isfinite(hour_rsam)]
    day_median_hour = float(np.median(good)) if good.size else _EPS

    seq = np.zeros((HOURS_PER_DAY, HOURLY_DIM), dtype=np.float32)
    for h in range(HOURS_PER_DAY):
        if np.isfinite(hour_rsam[h]):
            seq[h, 0] = _log10(hour_rsam[h])
            seq[h, 1] = _log10(hour_rsam[h] / max(day_median_hour, _EPS))
            if baseline is not None:
                seq[h, 11] = _log10(hour_rsam[h] / max(baseline, _EPS))
                seq[h, 12] = 1.0
        for b in range(4):
            if np.isfinite(hour_bands[h, b]):
                seq[h, 2 + b] = _log10(hour_bands[h, b])
        if np.isfinite(hour_bands[h, 0]) and np.isfinite(hour_bands[h, 2]):
            seq[h, 6] = _log10(hour_bands[h, 0] / max(hour_bands[h, 2], _EPS))
        seq[h, 7] = float(np.log1p(hour_trigs[h]))
        seq[h, 8] = float(hour_tremor[h])
        seq[h, 9] = float(hour_peak[h])
        seq[h, 10] = float(hour_frac[h])
    return seq


def build_dataset(ctx: PipelineContext) -> VolcanicDataset:
    """Assemble the (volcano, day) dataset from the fetched caches.

    Causality: samples are processed in chronological order per
    station-channel; the quiet-RSAM baseline for a day is the median 24h
    RSAM of up to the 90 most recent PREVIOUSLY sampled quiet days (>= 3
    required, else flagged absent), and the day-over-day delta uses the most
    recent sampled day <= 7 days back. No statistic ever sees a later day.
    Standardization statistics come from TRAIN-year samples only.

    Raises:
        FileNotFoundError: When the fetch stage has not run.
        RuntimeError: When a split has no positive or no negative samples.
    """
    vol_dir = ctx.data_dir / "volcanic"
    plan_path = vol_dir / "sample_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"missing {plan_path}; run the --fetch stage first")
    plan_rows = json.loads(plan_path.read_text())
    if ctx.limit_samples is not None:
        plan_rows = plan_rows[: ctx.limit_samples]

    volcano_index = {v.name: i for i, v in enumerate(VOLCANOES)}
    by_chan: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in plan_rows:
        by_chan.setdefault((row["sta"], row["cha"]), []).append(row)

    feats: list[np.ndarray] = []
    hourly: list[np.ndarray] = []
    labels: list[float] = []
    leads: list[int] = []
    veis: list[int] = []
    years: list[int] = []
    vol_idx: list[int] = []
    days: list[str] = []
    minute_all: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for (sta, cha), rows in sorted(by_chan.items()):
        rows.sort(key=lambda r: r["day"])
        quiet_history: list[tuple[_dt.date, float]] = []
        last_day: tuple[_dt.date, float] | None = None
        for row in rows:
            day = _d(row["day"])
            rec = _load_or_compute_day_record(
                ctx, sta, cha, day, scale=float(row["scale"]), sr=float(row["sample_rate"])
            )
            if rec is None:
                continue
            prior = [r for d0, r in quiet_history if d0 < day][-90:]
            baseline = float(np.median(prior)) if len(prior) >= 3 else None
            prev_rsam = None
            if last_day is not None and 0 < (day - last_day[0]).days <= 7:
                prev_rsam = last_day[1]
            vec = assemble_feature_vector(
                rec,
                volcano_index=volcano_index[row["volcano"]],
                is_broadband=(cha == "BHZ"),
                baseline=baseline,
                prev_rsam=prev_rsam,
            )
            seq = assemble_hourly_matrix(rec, baseline)
            rsam_24h = float(rec["rsam_24h"])
            feats.append(vec)
            hourly.append(seq)
            labels.append(1.0 if row["cls"] == "positive" else 0.0)
            leads.append(int(row["days_to_onset"] or 0))
            veis.append(int(row["vei"]) if row["vei"] is not None else -1)
            years.append(day.year)
            vol_idx.append(volcano_index[row["volcano"]])
            days.append(row["day"])
            minute_all.append(np.asarray(rec["minute_rsam"], dtype=np.float32))
            meta.append(
                {
                    "volcano": row["volcano"],
                    "day": row["day"],
                    "sta": sta,
                    "cha": cha,
                    "cls": row["cls"],
                    "baseline": baseline,
                }
            )
            last_day = (day, rsam_24h)
            if row["cls"] == "negative":
                quiet_history.append((day, rsam_24h))
                quiet_history = quiet_history[-200:]

    if not feats:
        raise RuntimeError("no samples could be assembled from the caches; fetch incomplete?")

    features = np.stack(feats)
    years_arr = np.asarray(years, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.float32)
    train_mask, val_mask, test_mask = SPLIT.masks(years_arr)
    for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not mask.any() or labels_arr[mask].min() == labels_arr[mask].max():
            raise RuntimeError(
                f"split '{split_name}' lacks both classes "
                f"(n={int(mask.sum())}); refusing to train/evaluate on it"
            )

    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    hourly_arr = np.stack(hourly)
    hmean = hourly_arr[train_mask].reshape(-1, HOURLY_DIM).mean(axis=0)
    hstd = hourly_arr[train_mask].reshape(-1, HOURLY_DIM).std(axis=0)
    hstd[hstd < 1e-6] = 1.0

    return VolcanicDataset(
        features=features,
        hourly=hourly_arr,
        labels=labels_arr,
        days_to_onset=np.asarray(leads, dtype=np.float32),
        vei=np.asarray(veis, dtype=np.int64),
        years=years_arr,
        volcano_idx=np.asarray(vol_idx, dtype=np.int64),
        day_iso=days,
        minute_rsam=np.stack(minute_all),
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        hourly_mean=hmean.astype(np.float32),
        hourly_std=hstd.astype(np.float32),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------


def _standardize(ds: VolcanicDataset) -> tuple[np.ndarray, np.ndarray]:
    """Standardized copies of daily features and hourly sequences."""
    x = (ds.features - ds.feature_mean) / ds.feature_std
    h = (ds.hourly - ds.hourly_mean) / ds.hourly_std
    return x.astype(np.float32), h.astype(np.float32)


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the EruptionForecastModel (+ swarm LSTM) with early stopping.

    Losses: positive-weighted BCE on the eruption head (primary); on
    positive samples only, cross-entropy on the VEI head (target = VEI of
    the upcoming eruption; VEI-less onsets masked out) and MSE on the time
    head (target = min(days_to_onset / 7, 1) so the detector's fixed
    ``time_norm * 168 h`` decode reads honestly up to 7 days, saturating for
    8-14 day leads -- documented diagnostic, not a gate metric). The swarm
    LSTM is trained on the same eruption-within-14d label from the hourly
    sequences (its "swarm probability" is exactly that, per module
    docstring). Early stopping on validation AUC of the eruption head,
    patience 8. Adam 1e-3, batch 64.

    Returns:
        Training record (also embedded in the candidate checkpoint).
    """
    from omni_mercury_engine.detectors.geological.volcanic import (
        EruptionForecastModel,
        SeismicSwarmDetector,
    )

    torch.set_num_threads(2)  # shared box
    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    x, h = _standardize(ds)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)

    xt = torch.from_numpy(x[train_mask])
    ht = torch.from_numpy(h[train_mask])
    yt = torch.from_numpy(ds.labels[train_mask])
    lead_t = torch.from_numpy(ds.days_to_onset[train_mask])
    vei_t = torch.from_numpy(ds.vei[train_mask])
    xv = torch.from_numpy(x[val_mask])
    yv = ds.labels[val_mask]

    n_pos = float(yt.sum().item())
    n_neg = float((1 - yt).sum().item())
    pos_weight = n_neg / max(n_pos, 1.0)
    logger.info(
        "training on %d rows (%d pos / %d neg, pos_weight %.2f), validating on %d rows",
        xt.shape[0],
        int(n_pos),
        int(n_neg),
        pos_weight,
        xv.shape[0],
    )

    model = EruptionForecastModel(input_dim=FEATURE_DIM)
    swarm = SeismicSwarmDetector(input_dim=HOURLY_DIM)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(swarm.parameters()), lr=1e-3, weight_decay=1e-5
    )

    def _weighted_bce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = p.clamp(1e-6, 1 - 1e-6)
        return -(pos_weight * y * p.log() + (1 - y) * (1 - p).log()).mean()

    time_target = (lead_t * 24.0 / 168.0).clamp(0.0, 1.0)
    batch_size = 64
    best_val_auc = -np.inf
    best_states: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
    patience, bad_epochs, epochs_run = 8, 0, 0

    for epoch in range(ctx.max_epochs):
        epochs_run = epoch + 1
        model.train()
        swarm.train()
        perm = torch.from_numpy(rng.permutation(xt.shape[0]))
        epoch_loss = 0.0
        for start in range(0, xt.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            if idx.shape[0] < 2:
                continue  # BatchNorm needs > 1 sample
            xb, hb, yb = xt[idx], ht[idx], yt[idx]
            ep_prob, vei_logits, time_norm = model(xb)
            loss = _weighted_bce(ep_prob.squeeze(-1), yb)
            pos = yb > 0.5
            if pos.any():
                vei_target = vei_t[idx][pos]
                vei_ok = vei_target >= 0
                if vei_ok.any():
                    loss = loss + 0.3 * torch.nn.functional.cross_entropy(
                        vei_logits[pos][vei_ok], vei_target[vei_ok].clamp(0, 7)
                    )
                loss = loss + 0.3 * torch.nn.functional.mse_loss(
                    time_norm.squeeze(-1)[pos], time_target[idx][pos]
                )
            swarm_prob, _ = swarm(hb)
            loss = loss + _weighted_bce(swarm_prob.squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            optimizer.step()
            epoch_loss += float(loss.item()) * idx.shape[0]

        model.eval()
        swarm.eval()
        with torch.no_grad():
            val_prob = model(xv)[0].squeeze(-1).numpy()
        val_auc = binary_auc(yv, val_prob)
        logger.info(
            "epoch %d: train loss %.4f, val AUC %.4f",
            epoch + 1,
            epoch_loss / xt.shape[0],
            val_auc,
        )
        if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-5:
            best_val_auc = float(val_auc)
            best_states = (
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                {k: v.detach().clone() for k, v in swarm.state_dict().items()},
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, patience)
                break

    if best_states is None:
        raise RuntimeError("training never produced a finite validation AUC; refusing to save")
    model.load_state_dict(best_states[0])
    swarm.load_state_dict(best_states[1])

    # Validation-selected alert threshold (reported + shipped for operators;
    # the detector's hardcoded eruption_imminent threshold stays 0.7).
    model.eval()
    with torch.no_grad():
        val_prob = model(xv)[0].squeeze(-1).numpy()
    tau, tau_stats = _select_threshold(yv, val_prob)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_auc": best_val_auc,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "train_positive_fraction": n_pos / max(n_pos + n_neg, 1.0),
        "pos_weight": pos_weight,
        "val_alert_threshold": tau,
        "val_alert_threshold_stats": tau_stats,
    }
    payload: dict[str, Any] = {
        "eruption_model": model.state_dict(),
        "seismic_detector": swarm.state_dict(),
        "feature_spec": FEATURE_SPEC_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "hourly_feature_names": list(HOURLY_FEATURE_NAMES),
        "volcanoes": [v.name for v in VOLCANOES],
        "label": f"eruption onset within {K_DAYS}d",
        "feature_mean": ds.feature_mean.tolist(),
        "feature_std": ds.feature_std.tolist(),
        "hourly_feature_mean": ds.hourly_mean.tolist(),
        "hourly_feature_std": ds.hourly_std.tolist(),
        "val_alert_threshold": tau,
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Pick the score threshold maximizing CSI on the given (val) samples."""
    labels = np.asarray(labels, dtype=bool)
    best_tau, best_csi, best = 0.5, -1.0, {"recall": 0.0, "far": 0.0, "csi": 0.0}
    for tau in np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 257))):
        detect = scores >= tau
        tp = float(np.sum(detect & labels))
        fn = float(np.sum(~detect & labels))
        fp = float(np.sum(detect & ~labels))
        csi = tp / max(tp + fn + fp, 1.0)
        if csi > best_csi:
            best_csi = csi
            best_tau = float(tau)
            best = {
                "recall": tp / max(tp + fn, 1.0),
                "far": fp / max(float(np.sum(~labels)), 1.0),
                "csi": csi,
            }
    return best_tau, best


# ---------------------------------------------------------------------------
# Stage: evaluate
# ---------------------------------------------------------------------------


def _rsam_baseline_scores(ds: VolcanicDataset) -> np.ndarray:
    """Documented RSAM-ratio baseline score for every sample.

    Score = log10(day 24h RSAM / causal quiet baseline), i.e. feature dim 19
    (0 where the baseline is unavailable -- flagged in dim 20). This is the
    classic observatory RSAM-threshold rule; its alert threshold is fitted
    on TRAIN years only in :func:`evaluate`.
    """
    return ds.features[:, 19].astype(np.float64)


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both paths see the IDENTICAL held-out (volcano, day) cases, each fed the
    honest input it consumes, derived from the same real seismic record:

    * learned -- ``{"fused_features": standardized volcano-seismic-v1
      vector, "seismic_sequence": standardized hourly (24, 32) sequence}``
      through ``VolcanicEruptionDetector.predict_eruption`` after
      ``load_neural_weights(candidate)``;
    * physics -- ``{"seismic_sequence": the day's per-minute velocity RSAM
      series}`` through the same method on an un-weighted detector: the
      deterministic robust-z swarm statistics + HMM belief drive the
      confidence. No degassing/InSAR/thermal exists in this dataset, so
      those precursor fields are honestly absent for BOTH paths (the physics
      noisy-OR forecast stage only runs at >= 2 indicators, which seismic
      alone cannot reach -- its confidence is the swarm/HMM path; recorded
      in extras).

    The HMM is reset before every case so sample order cannot leak state.

    Because seismic-only physics rarely raises the multi-parameter alarm, a
    documented RSAM-ratio threshold baseline (fitted on TRAIN years) is the
    stronger classical comparison: the merit gate additionally requires the
    learned AUC to be >= that baseline's AUC (see ``constraints``).

    Returns:
        The evaluation outcome (primary metric: eruption AUC, higher better).
    """
    from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

    torch.set_num_threads(2)  # shared box
    ds = build_dataset(ctx)
    x, h = _standardize(ds)
    _, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test rows found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")
    payload = torch.load(cand_path, map_location="cpu", weights_only=True)
    model_head_tau = float(payload["val_alert_threshold"])  # reported only

    physics_det = VolcanicEruptionDetector()
    learned_det = VolcanicEruptionDetector()
    learned_det.load_neural_weights(str(cand_path))

    y = ds.labels[test_idx]
    rsam_scores_all = _rsam_baseline_scores(ds)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)
    val_idx = np.flatnonzero(val_mask)
    tau_rsam, rsam_fit = _select_threshold(ds.labels[train_mask], rsam_scores_all[train_mask])
    rsam_scores = rsam_scores_all[test_idx]

    def _run_cases(indices: np.ndarray) -> dict[str, Any]:
        conf: dict[str, list[float]] = {"learned": [], "physics": []}
        imminent: dict[str, list[bool]] = {"learned": [], "physics": []}
        swarm_flags: list[bool] = []
        for i in indices:
            minute = ds.minute_rsam[i]
            minute_clean = minute[np.isfinite(minute)].astype(np.float64)
            case_learned = {
                "fused_features": x[i].tolist(),
                "seismic_sequence": h[i],
            }
            case_physics = {"seismic_sequence": minute_clean}
            for name, det, case in (
                ("learned", learned_det, case_learned),
                ("physics", physics_det, case_physics),
            ):
                if det.state_hmm is not None:
                    det.state_hmm.reset()  # per-case independence, no order leak
                result = det.predict_eruption(case)
                if not np.isfinite(result.confidence):
                    raise RuntimeError(f"{name} path returned non-finite confidence for case {i}")
                conf[name].append(float(result.confidence))
                imminent[name].append(bool(result.eruption_imminent))
                if name == "physics":
                    swarm_flags.append(bool(result.seismic_swarm_detected))
        return {"conf": conf, "imminent": imminent, "swarm": swarm_flags}

    # Alert threshold for the learned path is selected on the VALIDATION
    # years' API confidences (never test), so the deployed decision rule is
    # fixed before the held-out cases are scored.
    val_runs = _run_cases(val_idx)
    tau_learned, tau_val_stats = _select_threshold(
        ds.labels[val_idx], np.asarray(val_runs["conf"]["learned"])
    )
    runs = _run_cases(test_idx)
    conf = runs["conf"]
    imminent = runs["imminent"]
    swarm_flags = runs["swarm"]

    def _op_metrics(detect: np.ndarray) -> tuple[float, float, float]:
        is_pos = y == 1.0
        tp = float(np.sum(detect & is_pos))
        fn = float(np.sum(~detect & is_pos))
        fp = float(np.sum(detect & ~is_pos))
        recall = tp / max(tp + fn, 1.0)
        far = fp / max(float(np.sum(~is_pos)), 1.0)
        csi = tp / max(tp + fn + fp, 1.0)
        return recall, far, csi

    learned_scores = np.asarray(conf["learned"])
    physics_scores = np.asarray(conf["physics"])
    learned_alert = learned_scores >= tau_learned
    physics_alert = np.asarray(swarm_flags, dtype=bool)  # deployed seismic alarm
    rsam_alert = rsam_scores >= tau_rsam

    l_rec, l_far, l_csi = _op_metrics(learned_alert)
    p_rec, p_far, p_csi = _op_metrics(physics_alert)
    r_rec, r_far, r_csi = _op_metrics(rsam_alert)
    l_rec_dep, l_far_dep, _ = _op_metrics(np.asarray(imminent["learned"], dtype=bool))
    p_rec_dep, p_far_dep, _ = _op_metrics(np.asarray(imminent["physics"], dtype=bool))
    rsam_auc = binary_auc(y, rsam_scores)

    learned_metrics = {
        "auc": binary_auc(y, learned_scores),
        "brier": brier_score(y, np.clip(learned_scores, 0.0, 1.0)),
        "recall_op": l_rec,
        "far_op": l_far,
        "csi_op": l_csi,
        "recall_imminent_070": l_rec_dep,
        "far_imminent_070": l_far_dep,
        "auc_vs_rsam_baseline": binary_auc(y, learned_scores),
    }
    physics_metrics = {
        "auc": binary_auc(y, physics_scores),
        "brier": brier_score(y, np.clip(physics_scores, 0.0, 1.0)),
        "recall_op": p_rec,
        "far_op": p_far,
        "csi_op": p_csi,
        "recall_imminent_070": p_rec_dep,
        "far_imminent_070": p_far_dep,
        "auc_vs_rsam_baseline": rsam_auc,
    }

    hit_table = _per_onset_hit_table(ds, test_idx, learned_alert, physics_alert, rsam_alert)
    lead_stats = _lead_time_stats(ds, test_idx, learned_alert)

    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="auc",
        higher_is_better=True,
        learned=learned_metrics,
        physics=physics_metrics,
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "scope": (
                "trained/evaluated ONLY for the named AVO volcanoes: "
                + ", ".join(v.name for v in VOLCANOES)
            ),
            "label": f"eruption onset within {K_DAYS} d (GVP day-precision starts)",
            "comparison": (
                "identical held-out (volcano, day) cases through "
                "VolcanicEruptionDetector.predict_eruption; learned = candidate checkpoint "
                "on fused_features + hourly seismic_sequence; physics = un-weighted "
                "detector on the same day's per-minute RSAM series (robust-z swarm + HMM). "
                "Gas/InSAR/thermal are honestly absent for both paths. The physics "
                "noisy-OR forecast stage needs >= 2 indicators and therefore abstains on "
                "seismic-only input, so the documented RSAM-ratio threshold baseline "
                "(feature dim 19, threshold fitted on TRAIN years, see rsam_baseline) is "
                "included as the stronger classical comparison and gated via the "
                "auc_vs_rsam_baseline constraint."
            ),
            "operating_points": {
                "learned_tau_val_csi": tau_learned,
                "learned_tau_val_stats": tau_val_stats,
                "learned_model_head_tau_from_train_stage": model_head_tau,
                "physics_alarm": "seismic_swarm_detected flag (robust-z exceedance)",
                "rsam_tau_train_csi": tau_rsam,
                "rsam_train_fit": rsam_fit,
                "detector_deployed_imminent_threshold": 0.7,
            },
            "rsam_baseline": {
                "auc": rsam_auc,
                "recall_op": r_rec,
                "far_op": r_far,
                "csi_op": r_csi,
            },
            "per_onset_hits": hit_table,
            "lead_time_days": lead_stats,
            "test_positive_fraction": float(y.mean()),
            "hmm_reset_per_case": True,
        },
        constraints=[
            {
                "metric": "auc_vs_rsam_baseline",
                "higher_is_better": True,
                "description": (
                    "learned AUC must not fall below the classical RSAM-ratio "
                    "threshold baseline's AUC (physics-side value IS that baseline)"
                ),
            },
            {
                "metric": "recall_op",
                "higher_is_better": True,
                "description": (
                    "recall at the deployed alert point must not regress below the "
                    "physics swarm alarm's recall on identical cases"
                ),
            },
        ],
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned AUC %.4f vs physics %.4f (RSAM baseline %.4f) on %d held-out "
        "samples (%s)",
        outcome.learned["auc"],
        outcome.physics["auc"],
        rsam_auc,
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "NOT SHIPPED",
    )
    return outcome


def _per_onset_hit_table(
    ds: VolcanicDataset,
    test_idx: np.ndarray,
    learned_alert: np.ndarray,
    physics_alert: np.ndarray,
    rsam_alert: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-onset hit table: did each path alert on any pre-onset day?"""
    onset_days: dict[tuple[str, str], dict[str, Any]] = {}
    for pos, i in enumerate(test_idx):
        m = ds.meta[i]
        if m["cls"] != "positive":
            continue
        lead = int(ds.days_to_onset[i])
        onset = (_d(m["day"]) + _dt.timedelta(days=lead)).isoformat()
        key = (m["volcano"], onset)
        entry = onset_days.setdefault(
            key,
            {
                "volcano": m["volcano"],
                "onset": onset,
                "days_with_data": 0,
                "learned_hit": False,
                "physics_hit": False,
                "rsam_hit": False,
                "learned_earliest_lead_d": None,
            },
        )
        entry["days_with_data"] += 1
        if bool(physics_alert[pos]):
            entry["physics_hit"] = True
        if bool(rsam_alert[pos]):
            entry["rsam_hit"] = True
        if bool(learned_alert[pos]):
            entry["learned_hit"] = True
            prev = entry["learned_earliest_lead_d"]
            entry["learned_earliest_lead_d"] = max(prev or 0, lead)
    return [onset_days[k] for k in sorted(onset_days)]


def _lead_time_stats(
    ds: VolcanicDataset, test_idx: np.ndarray, learned_alert: np.ndarray
) -> dict[str, float | None]:
    """Earliest-alert lead-time statistics over caught test onsets."""
    leads: dict[tuple[str, str], int] = {}
    for pos, i in enumerate(test_idx):
        m = ds.meta[i]
        if m["cls"] != "positive" or not bool(learned_alert[pos]):
            continue
        lead = int(ds.days_to_onset[i])
        onset = (_d(m["day"]) + _dt.timedelta(days=lead)).isoformat()
        key = (m["volcano"], onset)
        leads[key] = max(leads.get(key, 0), lead)
    if not leads:
        return {"n_caught_onsets": 0, "mean": None, "median": None, "min": None, "max": None}
    values = np.asarray(sorted(leads.values()), dtype=np.float64)
    return {
        "n_caught_onsets": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------------
# Stage: ship
# ---------------------------------------------------------------------------


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = ctx.data_dir / "volcanic" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing fetch manifest {manifest_path}; run --fetch first")
    manifest = json.loads(manifest_path.read_text())
    return ship_checkpoint(
        hook=HOOK_NAME,
        checkpoint_name=CHECKPOINT_NAME,
        data_dir=ctx.data_dir,
        outcome=outcome,
        data_sources=manifest["sources"],
        seed=ctx.seed,
        out_dir=ctx.ship_dir,
    )
