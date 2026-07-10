# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Heatwave Detector - percentile climatology, Excess Heat Factor, heat index.

Implements the literature-standard heatwave definitions on real daily
temperature series:

- **Calendar-day percentile definition** (Perkins & Alexander 2013, "On the
  measurement of heat waves", *J. Climate* 26, 4500-4517): a heatwave is a
  run of >= 3 consecutive days on which the daily maximum temperature
  exceeds the calendar-day 90th percentile (CTX90pct), where the percentile
  for each calendar day is computed from a baseline climatology using a
  15-day window centred on that day.

- **Excess Heat Factor** (Nairn & Fawcett 2015, "The excess heat factor: a
  metric for heatwave intensity and its use in classifying heatwave
  severity", *Int. J. Environ. Res. Public Health* 12, 227-253)::

      EHI_sig(i)  = mean(DMT_i, DMT_{i+1}, DMT_{i+2}) - T95
      EHI_accl(i) = mean(DMT_i, DMT_{i+1}, DMT_{i+2})
                    - mean(DMT_{i-30} .. DMT_{i-1})
      EHF(i)      = EHI_sig(i) * max(1, EHI_accl(i))

  where DMT is the daily mean temperature (Tmax + Tmin) / 2 and T95 the
  95th percentile of DMT over the baseline.  Severity tiers follow the same
  paper: low-intensity for 0 < EHF < EHF85, severe for EHF >= EHF85, and
  extreme for EHF >= 3 * EHF85, where EHF85 is the 85th percentile of the
  positive baseline EHF values.

- **Heat index** (Rothfusz 1990, NWS Southern Region Technical Attachment
  SR 90-23, a regression fit to Steadman 1979's apparent temperature): the
  full NWS operational algorithm including the low-humidity and
  high-humidity adjustments, with the documented validity range guarded.

- **NWS alert categories**: when only forecasts/alerts are available, the
  detector maps active NWS heat products (fetched via
  ``NWSWeatherAlertsSource``) onto a HeatRisk-style 0-4 category.  This is
  an alert-derived category in the spirit of the NWS HeatRisk index (a
  separate gridded product not exposed through the alerts API), and is
  labelled as such - the detector never fabricates a HeatRisk grid value.

The detector is a pure statistics core: it works untrained, uses no neural
networks, and fails loudly on inadequate baselines instead of inventing
climatology.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Minimum samples required inside each calendar-day percentile window.
MIN_SAMPLES_PER_CALENDAR_DAY: int = 30

#: Minimum number of distinct baseline years (3 years x 15-day window = 45
#: samples per calendar day, above the 30-sample floor).
MIN_BASELINE_YEARS: int = 3

#: Plausible daily temperature bounds in degC (world extremes are -89.2 degC
#: and +56.7 degC); values outside this range indicate a unit error.
_TEMP_MIN_C: float = -90.0
_TEMP_MAX_C: float = 65.0

#: Days in the acclimatisation window of the EHF (Nairn & Fawcett 2015).
_EHF_ACCLIMATISATION_DAYS: int = 30

#: Cumulative day-of-year offsets for a fixed 365-day calendar.
_CUMULATIVE_DAYS: tuple[int, ...] = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


class HeatwaveSeverity(Enum):
    """EHF-based heatwave severity tiers (Nairn & Fawcett 2015)."""

    NONE = "none"
    LOW_INTENSITY = "low_intensity"
    SEVERE = "severe"
    EXTREME = "extreme"


class HeatRiskCategory(Enum):
    """HeatRisk-style categories derived from active NWS heat products.

    Mirrors the 0-4 banding of the NWS HeatRisk index; derived from alert
    products, NOT from the HeatRisk grid itself (which has no public API
    surface in the alerts feed).
    """

    MINIMAL = 0
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    EXTREME = 4


@dataclass
class HeatwaveEvent:
    """A single detected heatwave (>= 3 consecutive CTX90pct exceedances)."""

    start_index: int
    end_index: int
    start_date: str
    end_date: str
    duration_days: int
    max_exceedance_c: float
    mean_tmax_c: float


@dataclass
class HeatwaveAssessmentResult:
    """Result of a heatwave assessment against a fitted baseline.

    Attributes:
        heatwave_active: True when the final day of the series belongs to a
            detected heatwave run.
        events: All detected heatwave events.
        n_heatwave_days: Total days belonging to any event.
        exceedance_flags: Per-day boolean CTX90pct exceedance flags.
        ehf_series: EHF values aligned to three-day-period start days
            (NaN where undefined: within 30 days of a segment start or
            when Tmin was not supplied).
        max_ehf: Maximum finite EHF over the series, or None.
        severity: Worst EHF severity tier reached.
    """

    heatwave_active: bool
    events: list[HeatwaveEvent] = field(default_factory=list)
    n_heatwave_days: int = 0
    exceedance_flags: np.ndarray | None = None
    ehf_series: np.ndarray | None = None
    max_ehf: float | None = None
    severity: str = HeatwaveSeverity.NONE.value
    warning_actions: list[str] = field(default_factory=list)


def _to_date_array(dates: Any) -> np.ndarray:
    """Coerce input dates to a numpy datetime64[D] array.

    Args:
        dates: Sequence of date-likes (datetime64, date, ISO strings).

    Returns:
        datetime64[D] array.

    Raises:
        ValueError: If coercion fails or the array is empty.
    """
    arr = np.asarray(dates)
    try:
        out = arr.astype("datetime64[D]")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"could not interpret dates as calendar dates: {exc}") from exc
    if out.size == 0:
        raise ValueError("dates array is empty")
    return out


def _validate_temps(values: np.ndarray, name: str, n_expected: int) -> np.ndarray:
    """Validate a daily temperature array (finite, plausible degC range)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size != n_expected:
        raise ValueError(f"{name} must be 1-D with length {n_expected}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values; gap-fill or drop upstream")
    if np.any(arr < _TEMP_MIN_C) or np.any(arr > _TEMP_MAX_C):
        raise ValueError(
            f"{name} has values outside the plausible range "
            f"[{_TEMP_MIN_C}, {_TEMP_MAX_C}] degC - check units (degC expected)"
        )
    return arr


def _day_of_year_365(dates: np.ndarray) -> np.ndarray:
    """Map dates to a fixed 365-day calendar index (0-364; Feb 29 -> Feb 28)."""
    out = np.empty(dates.size, dtype=np.int64)
    for i, d in enumerate(dates):
        pydate = d.astype(_dt.date)
        month, day = pydate.month, pydate.day
        if month == 2 and day == 29:
            day = 28
        out[i] = _CUMULATIVE_DAYS[month - 1] + day - 1
    return out


def heat_index_f(temp_f: float, rh_pct: float) -> float:
    """Compute the NWS heat index (apparent temperature) in degF.

    Implements the full NWS operational algorithm: the Steadman-based
    simple formula is used first; when its average with the air temperature
    is >= 80 degF the Rothfusz (1990, SR 90-23) regression applies, with
    the NWS low-humidity (RH < 13%, 80-112 degF) and high-humidity
    (RH > 85%, 80-87 degF) adjustments.

    Args:
        temp_f: Air temperature in degF, within [-80, 150].
        rh_pct: Relative humidity in percent, within [0, 100].

    Returns:
        Heat index in degF.

    Raises:
        ValueError: If inputs are non-finite or outside the guarded
            validity ranges above.
    """
    if not (math.isfinite(temp_f) and math.isfinite(rh_pct)):
        raise ValueError("temp_f and rh_pct must be finite")
    if not -80.0 <= temp_f <= 150.0:
        raise ValueError(f"temp_f {temp_f} outside plausible range [-80, 150] degF")
    if not 0.0 <= rh_pct <= 100.0:
        raise ValueError(f"rh_pct {rh_pct} outside [0, 100]")

    simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh_pct * 0.094)
    if (simple + temp_f) / 2.0 < 80.0:
        return float(simple)

    t, r = temp_f, rh_pct
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * r
        - 0.22475541 * t * r
        - 6.83783e-3 * t * t
        - 5.481717e-2 * r * r
        + 1.22874e-3 * t * t * r
        + 8.5282e-4 * t * r * r
        - 1.99e-6 * t * t * r * r
    )
    if r < 13.0 and 80.0 <= t <= 112.0:
        hi -= ((13.0 - r) / 4.0) * math.sqrt((17.0 - abs(t - 95.0)) / 17.0)
    elif r > 85.0 and 80.0 <= t <= 87.0:
        hi += ((r - 85.0) / 10.0) * ((87.0 - t) / 5.0)
    return float(hi)


def heat_alert_category(alerts: list[Any]) -> HeatRiskCategory:
    """Map active NWS heat products to a HeatRisk-style category.

    Accepts the ``DataPoint`` objects produced by
    :class:`~omni_mercury_engine.data_sources.earth_science.NWSWeatherAlertsSource`
    (event name under ``point.data["event"]``) or plain dicts with an
    ``"event"`` key.

    Mapping (most severe product wins):

    - Extreme/Excessive Heat Warning -> ``EXTREME`` (4)
    - Extreme/Excessive Heat Watch   -> ``MAJOR`` (3)
    - Heat Advisory                  -> ``MODERATE`` (2)
    - any other product mentioning "heat" -> ``MINOR`` (1)
    - no heat product                -> ``MINIMAL`` (0)

    Args:
        alerts: Alert records (DataPoints or dicts).

    Returns:
        The derived :class:`HeatRiskCategory`.
    """
    best = HeatRiskCategory.MINIMAL
    for alert in alerts:
        payload = getattr(alert, "data", alert)
        event = str(payload.get("event", "")) if isinstance(payload, dict) else ""
        lowered = event.lower()
        if "heat" not in lowered:
            continue
        if "warning" in lowered and ("extreme" in lowered or "excessive" in lowered):
            category = HeatRiskCategory.EXTREME
        elif "watch" in lowered and ("extreme" in lowered or "excessive" in lowered):
            category = HeatRiskCategory.MAJOR
        elif "advisory" in lowered:
            category = HeatRiskCategory.MODERATE
        else:
            category = HeatRiskCategory.MINOR
        if category.value > best.value:
            best = category
    return best


class HeatwaveDetector:
    """Percentile-climatology heatwave detector with EHF severity.

    Workflow: call :meth:`fit_baseline` with a multi-year daily baseline
    series, then :meth:`detect_heatwaves` on the analysis series.  The
    baseline supplies the calendar-day CTX90pct thresholds, the DMT 95th
    percentile (T95) and the EHF 85th percentile (EHF85) used for severity
    tiers.
    """

    def __init__(
        self,
        percentile: float = 90.0,
        window_days: int = 15,
        min_duration_days: int = 3,
    ) -> None:
        """Initialize the detector.

        Args:
            percentile: Calendar-day exceedance percentile (90 or 95 in
                Perkins & Alexander 2013).
            window_days: Odd-width window centred on each calendar day used
                to pool baseline samples (15 in Perkins & Alexander 2013).
            min_duration_days: Minimum run length to declare a heatwave.

        Raises:
            ValueError: On out-of-range configuration.
        """
        if not 50.0 <= percentile < 100.0:
            raise ValueError(f"percentile must be in [50, 100), got {percentile}")
        if window_days < 1 or window_days % 2 == 0:
            raise ValueError(f"window_days must be odd and >= 1, got {window_days}")
        if min_duration_days < 1:
            raise ValueError(f"min_duration_days must be >= 1, got {min_duration_days}")
        self.percentile = percentile
        self.window_days = window_days
        self.min_duration_days = min_duration_days
        self.logger = logging.getLogger(__name__)

        self._ctx_thresholds: np.ndarray | None = None
        self._t95_dmt: float | None = None
        self._ehf85: float | None = None
        self._baseline_years: int = 0

    @property
    def is_fitted(self) -> bool:
        """Whether a baseline climatology has been fitted."""
        return self._ctx_thresholds is not None

    def fit_baseline(
        self,
        dates: Any,
        tmax_c: np.ndarray,
        tmin_c: np.ndarray | None = None,
    ) -> None:
        """Fit the calendar-day percentile climatology from a baseline series.

        Args:
            dates: Baseline dates (date-likes, one per sample).
            tmax_c: Daily maximum temperatures (degC).
            tmin_c: Optional daily minimum temperatures (degC); enables the
                EHF branch (T95 and EHF85 references).

        Raises:
            ValueError: If the baseline spans fewer than
                :data:`MIN_BASELINE_YEARS` distinct years, or any calendar
                day's pooled window holds fewer than
                :data:`MIN_SAMPLES_PER_CALENDAR_DAY` samples.
        """
        date_arr = _to_date_array(dates)
        tmax = _validate_temps(tmax_c, "tmax_c", date_arr.size)

        years = {d.astype(_dt.date).year for d in date_arr}
        if len(years) < MIN_BASELINE_YEARS:
            raise ValueError(
                f"baseline spans {len(years)} distinct year(s); >= "
                f"{MIN_BASELINE_YEARS} required for a calendar-day "
                f"{self.percentile:.0f}th-percentile climatology"
            )

        doy = _day_of_year_365(date_arr)
        half = self.window_days // 2
        thresholds = np.empty(365, dtype=np.float64)
        for target in range(365):
            dist = np.abs(doy - target)
            dist = np.minimum(dist, 365 - dist)  # circular calendar distance
            window_mask = dist <= half
            n_in_window = int(np.sum(window_mask))
            if n_in_window < MIN_SAMPLES_PER_CALENDAR_DAY:
                raise ValueError(
                    f"calendar day index {target} has only {n_in_window} "
                    f"baseline samples in its {self.window_days}-day window; "
                    f">= {MIN_SAMPLES_PER_CALENDAR_DAY} required - supply a "
                    "longer or more complete baseline"
                )
            thresholds[target] = np.percentile(tmax[window_mask], self.percentile)

        self._ctx_thresholds = thresholds
        self._baseline_years = len(years)
        self._t95_dmt = None
        self._ehf85 = None

        if tmin_c is not None:
            tmin = _validate_temps(tmin_c, "tmin_c", date_arr.size)
            dmt = (tmax + tmin) / 2.0
            self._t95_dmt = float(np.percentile(dmt, 95.0))
            baseline_ehf = self._compute_ehf(date_arr, dmt, self._t95_dmt)
            positive = baseline_ehf[np.isfinite(baseline_ehf) & (baseline_ehf > 0.0)]
            if positive.size >= 5:
                self._ehf85 = float(np.percentile(positive, 85.0))
            else:
                self.logger.warning(
                    "Baseline produced only %d positive EHF values; EHF "
                    "severity tiers disabled (need >= 5).",
                    positive.size,
                )

        self.logger.info(
            "Heatwave baseline fitted: %d years, CTX%dpct range %.1f-%.1f degC",
            self._baseline_years,
            int(self.percentile),
            float(np.min(thresholds)),
            float(np.max(thresholds)),
        )

    @staticmethod
    def _contiguous_segments(dates: np.ndarray) -> list[tuple[int, int]]:
        """Split a date array into [start, end) index ranges of daily-contiguous runs."""
        if dates.size == 0:
            return []
        gaps = np.where(np.diff(dates).astype("timedelta64[D]").astype(int) != 1)[0]
        starts = [0, *(int(g) + 1 for g in gaps)]
        ends = [*(int(g) + 1 for g in gaps), int(dates.size)]
        return list(zip(starts, ends))

    def _compute_ehf(
        self,
        dates: np.ndarray,
        dmt: np.ndarray,
        t95: float,
    ) -> np.ndarray:
        """Compute the EHF series (NaN where undefined).

        EHF(i) is assigned to the start day i of the three-day period
        (i, i+1, i+2) per Nairn & Fawcett (2015).  Values are computed only
        inside daily-contiguous segments with at least 30 days of history
        for the acclimatisation term.
        """
        ehf = np.full(dmt.size, np.nan)
        accl = _EHF_ACCLIMATISATION_DAYS
        for seg_start, seg_end in self._contiguous_segments(dates):
            for i in range(seg_start + accl, seg_end - 2):
                tdp = float(np.mean(dmt[i : i + 3]))
                ehi_sig = tdp - t95
                ehi_accl = tdp - float(np.mean(dmt[i - accl : i]))
                ehf[i] = ehi_sig * max(1.0, ehi_accl)
        return ehf

    def detect_heatwaves(
        self,
        dates: Any,
        tmax_c: np.ndarray,
        tmin_c: np.ndarray | None = None,
    ) -> HeatwaveAssessmentResult:
        """Detect heatwaves in an analysis series against the fitted baseline.

        Args:
            dates: Analysis dates (date-likes).
            tmax_c: Daily maximum temperatures (degC).
            tmin_c: Optional daily minimum temperatures (degC); enables EHF.

        Returns:
            A :class:`HeatwaveAssessmentResult`.

        Raises:
            RuntimeError: If :meth:`fit_baseline` has not been called.
            ValueError: On malformed input series.
        """
        if self._ctx_thresholds is None:
            raise RuntimeError(
                "detect_heatwaves called before fit_baseline; a percentile "
                "climatology is required - the detector will not invent one"
            )
        date_arr = _to_date_array(dates)
        tmax = _validate_temps(tmax_c, "tmax_c", date_arr.size)

        doy = _day_of_year_365(date_arr)
        exceed = tmax > self._ctx_thresholds[doy]

        events: list[HeatwaveEvent] = []
        run_start: int | None = None
        for i in range(exceed.size + 1):
            if i < exceed.size and exceed[i]:
                if run_start is None:
                    run_start = i
                continue
            if run_start is not None:
                run_len = i - run_start
                if run_len >= self.min_duration_days:
                    seg = slice(run_start, i)
                    events.append(
                        HeatwaveEvent(
                            start_index=run_start,
                            end_index=i - 1,
                            start_date=str(date_arr[run_start]),
                            end_date=str(date_arr[i - 1]),
                            duration_days=run_len,
                            max_exceedance_c=float(
                                np.max(tmax[seg] - self._ctx_thresholds[doy[seg]])
                            ),
                            mean_tmax_c=float(np.mean(tmax[seg])),
                        )
                    )
                run_start = None

        ehf_series: np.ndarray | None = None
        max_ehf: float | None = None
        severity = HeatwaveSeverity.NONE
        if tmin_c is not None and self._t95_dmt is not None:
            tmin = _validate_temps(tmin_c, "tmin_c", date_arr.size)
            dmt = (tmax + tmin) / 2.0
            ehf_series = self._compute_ehf(date_arr, dmt, self._t95_dmt)
            finite = ehf_series[np.isfinite(ehf_series)]
            if finite.size > 0:
                max_ehf = float(np.max(finite))
                severity = self._severity_tier(max_ehf)

        n_hw_days = int(sum(e.duration_days for e in events))
        active = bool(events) and events[-1].end_index == exceed.size - 1
        result = HeatwaveAssessmentResult(
            heatwave_active=active,
            events=events,
            n_heatwave_days=n_hw_days,
            exceedance_flags=exceed,
            ehf_series=ehf_series,
            max_ehf=max_ehf,
            severity=severity.value,
            warning_actions=self._generate_warnings(events, severity),
        )
        self.logger.info(
            "Heatwave detection: %d event(s), %d heatwave day(s), severity=%s",
            len(events),
            n_hw_days,
            severity.value,
        )
        return result

    def _severity_tier(self, ehf_value: float) -> HeatwaveSeverity:
        """Map a (maximum) EHF value onto the Nairn & Fawcett severity tiers."""
        if ehf_value <= 0.0:
            return HeatwaveSeverity.NONE
        if self._ehf85 is None or self._ehf85 <= 0.0:
            return HeatwaveSeverity.LOW_INTENSITY
        if ehf_value >= 3.0 * self._ehf85:
            return HeatwaveSeverity.EXTREME
        if ehf_value >= self._ehf85:
            return HeatwaveSeverity.SEVERE
        return HeatwaveSeverity.LOW_INTENSITY

    @staticmethod
    def _generate_warnings(events: list[HeatwaveEvent], severity: HeatwaveSeverity) -> list[str]:
        """Generate advisory strings."""
        warnings: list[str] = []
        if severity is HeatwaveSeverity.EXTREME:
            warnings.append("EXTREME heatwave (EHF >= 3x EHF85): dangerous to all")
            warnings.append("Activate emergency heat plans; check on vulnerable people")
        elif severity is HeatwaveSeverity.SEVERE:
            warnings.append("SEVERE heatwave (EHF >= EHF85): dangerous to vulnerable groups")
            warnings.append("Limit outdoor activity; ensure hydration and cooling")
        elif events:
            warnings.append("Heatwave conditions detected: stay hydrated, avoid midday sun")
        return warnings

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract a fixed 20-dim feature vector for ML fusion.

        Emits robust summary statistics of the flattened series plus simple
        run-length statistics of upper-decile exceedances within the series
        itself (a self-referential percentile, used only as a generic
        feature - the calibrated CTX90pct climatology requires
        :meth:`fit_baseline` and is not fabricated here).

        Args:
            data: Input array or tensor.

        Returns:
            Feature tensor of shape (20,).
        """
        if isinstance(data, torch.Tensor):
            arr: np.ndarray = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data, dtype=np.float64)
        flat = arr.astype(np.float64).flatten()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return torch.zeros(20, dtype=torch.float32)

        p90 = float(np.percentile(flat, 90.0))
        exceed = flat > p90
        # Longest run of exceedances.
        longest = current = 0
        for flag in exceed:
            current = current + 1 if flag else 0
            longest = max(longest, current)

        features: list[float] = [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.median(flat)),
            float(np.percentile(flat, 75) - np.percentile(flat, 25)),
            p90,
            float(np.percentile(flat, 95.0)),
            float(np.mean(exceed)),
            float(longest),
        ]
        while len(features) < 20:
            features.append(0.0)
        return torch.tensor(features[:20], dtype=torch.float32)
