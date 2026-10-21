# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Geomagnetically induced current (GIC) grid-risk detection from magnetometer data.

Consumes ground-magnetometer minute data (e.g.
:class:`omni_mercury_engine.data_sources.geomagnetic.USGSGeomagnetismSource`,
which serves real per-observatory 1-minute X/Y/Z/F values from the USGS
Geomagnetism web service, ``/ws/data``) and produces:

* **dB/dt metrics** — per-component and horizontal-magnitude time
  derivatives in nT/min on 1-minute data, peak value + timestamp, and a
  sustained metric (maximum 10-minute rolling mean). Rapid horizontal-field
  variation is the primary driver of GIC in transmission networks (Viljanen
  et al. 2001, Ann. Geophys. 19; Pulkkinen et al. 2013, Space Weather 11).
  dB/dt grows with sampling rate (on the same two hours of 2024 Gannon-storm
  data, 17-19Z on 10 May, FRD read 765 nT/min from 1-second differences but
  55 nT/min from 1-minute means; every 60th sample alone reads 57-84 nT/min,
  depending on which second is kept), so sub-minute data are averaged to
  1-minute means first, the cadence of the published thresholds; coarser
  data are refused because they under-read.

* **Geoelectric-field proxy** — the standard plane-wave (magnetotelluric)
  relation over a uniform half-space (Cagniard 1953, Geophysics 18;
  Viljanen & Pirjola 1994, Surv. Geophys. 15; Viljanen et al. 2004, Ann.
  Geophys. 22): in the frequency domain the surface impedance is
  ``Z(omega) = sqrt(i omega mu0 / sigma)`` and::

      E_x(omega) =  Z(omega) * B_y(omega) / mu0
      E_y(omega) = -Z(omega) * B_x(omega) / mu0

  The ground conductivity is parameterised with named reference models
  (default ``resistive_shield``: sigma = 1e-3 S/m, i.e. 1000 ohm-m,
  representative of resistive Precambrian shield terrain where GIC risk is
  highest — cf. Boteler 1994, IEEE Trans. Power Delivery 9; Boteler &
  Pirjola 1998, Geophys. J. Int. 132).

* **Risk tiers on dB/dt** (1-minute data), each anchored to a cited level:
  moderate from 90 nT/min (1.5 nT/s, the highest of the community dB/dt
  validation thresholds, Pulkkinen et al. 2013); high from 300 nT/min (the
  lower end of the 300-600 nT/min that accompanied US grid impacts in March
  1989, Oak Ridge Meta-R-321); severe from 480 nT/min (the regional level at
  the 1989 Hydro-Québec collapse, Bolduc 2002; Meta-R-321). Meta-R-321 also
  notes that ~5000 nT/min is possible in an extreme storm. NERC TPL-007 is a
  geoelectric-field standard (benchmark 8 V/km at 60° geomagnetic latitude,
  scaled for latitude and ground), not a dB/dt one; the plane-wave proxy here
  uses a uniform half-space, not a TPL-007 ground model.

Physics-only module: no neural network. Fail-loud on empty/misaligned
series, naive timestamps, and non-monotonic time; data gaps (None/NaN) are
excluded pairwise from dB/dt and reported in ``notes`` — the geoelectric
proxy requires a gap-free series and is reported as ``None`` with an
explicit note otherwise (FFT over gaps would fabricate spectral content).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import fftconvolve

from omni_mercury_engine.detectors._hazard_input import InvalidObservationError

if TYPE_CHECKING:
    from omni_mercury_engine.data_sources.base import DataPoint

logger = logging.getLogger(__name__)

#: Vacuum permeability, H/m.
MU0: float = 4.0e-7 * math.pi

#: Named uniform-half-space ground-conductivity reference models (S/m).
#: ``resistive_shield``: 1000 ohm-m Precambrian-shield-like ground (high
#: GIC hazard; Boteler 1994). ``conductive_sediment``: 10 ohm-m sedimentary
#: ground (low hazard).
CONDUCTIVITY_MODELS_S_PER_M: dict[str, float] = {
    "resistive_shield": 1.0e-3,
    "conductive_sediment": 1.0e-1,
}

#: dB/dt alerting tiers on 1-minute data, nT/min (see module docstring).
DBDT_RISK_TIERS_NT_PER_MIN: tuple[tuple[float, str], ...] = (
    (480.0, "severe"),
    (300.0, "high"),
    (90.0, "moderate"),
)

#: Cadence (s) at which dB/dt is computed and the tiers are defined.
DBDT_CADENCE_S: float = 60.0

#: Relative tolerance on :data:`DBDT_CADENCE_S` within which a series counts
#: as 1-minute data (finer is averaged, coarser refused).
DBDT_CADENCE_TOLERANCE: float = 0.05

#: Rolling window for the sustained metric: 10 one-minute samples.
SUSTAINED_WINDOW_SAMPLES: int = 10

#: Largest plausible magnitude of a field component, nT (the main field is
#: below ~70,000 nT everywhere); larger values are not nT.
MAX_FIELD_NT: float = 1.0e5


def classify_dbdt_risk(peak_dbdt_nt_per_min: float) -> str:
    """Classify peak horizontal 1-minute dB/dt against the cited tiers.

    Tiers: low < 90, moderate 90-300, high 300-480, severe >= 480 nT/min
    (Pulkkinen et al. 2013; Oak Ridge Meta-R-321; Bolduc 2002).

    Args:
        peak_dbdt_nt_per_min: Peak horizontal dB/dt, nT/min.

    Returns:
        One of ``low`` / ``moderate`` / ``high`` / ``severe``.

    Raises:
        ValueError: On negative or non-finite input.
    """
    if not math.isfinite(peak_dbdt_nt_per_min) or peak_dbdt_nt_per_min < 0.0:
        raise ValueError(f"Peak dB/dt must be finite and >= 0; got {peak_dbdt_nt_per_min!r}.")
    for threshold, name in DBDT_RISK_TIERS_NT_PER_MIN:
        if peak_dbdt_nt_per_min >= threshold:
            return name
    return "low"


@dataclass
class GICAssessment:
    """GIC risk assessment for one observatory / magnetometer series.

    Attributes:
        observatory: Observatory / station identifier ("" if unknown).
        risk_level: dB/dt tier: low / moderate / high / severe.
        peak_dbdt_nt_per_min: Peak horizontal dB/dt, nT/min.
        peak_dbdt_time: Timestamp of the peak (end of the minute step).
        peak_dbdt_x_nt_per_min: Peak |dBx/dt|, nT/min (None if X absent).
        peak_dbdt_y_nt_per_min: Peak |dBy/dt|, nT/min (None if Y absent).
        sustained_dbdt_nt_per_min: Max 10-minute rolling mean of the
            1-minute horizontal dB/dt (None if the series is too short/gappy).
        geoelectric_peak_v_per_km: Peak plane-wave |E| proxy, V/km (None
            when the series has gaps — never interpolated silently).
        conductivity_model: Named ground model used for the proxy.
        sigma_s_per_m: Conductivity value used, S/m.
        n_samples: Samples assessed (after averaging sub-minute data to 1 min).
        n_gaps: Count of non-finite input samples (X and Y each counted),
            before any averaging. Sub-minute samples are skipped within their
            minute's mean; minute bins left with no finite sample are counted
            separately in ``notes``.
        single_component: True when only one horizontal component was
            available (H-only assessment).
        notes: Explicit data-quality / scope notes.
    """

    observatory: str
    risk_level: str
    peak_dbdt_nt_per_min: float
    peak_dbdt_time: datetime
    peak_dbdt_x_nt_per_min: float | None
    peak_dbdt_y_nt_per_min: float | None
    sustained_dbdt_nt_per_min: float | None
    geoelectric_peak_v_per_km: float | None
    conductivity_model: str
    sigma_s_per_m: float
    n_samples: int
    n_gaps: int
    single_component: bool = False
    notes: list[str] = field(default_factory=list)


def _check_field_units(bx: np.ndarray[Any, Any], by: np.ndarray[Any, Any], single: bool) -> None:
    """Refuse field values that cannot be nanotesla.

    Raises:
        InvalidObservationError: When every finite value is below 0.01 in
            magnitude (tesla: 2e-5 T read "low" with a 3.7e-7 nT/min peak) or
            any exceeds :data:`MAX_FIELD_NT`.
    """
    values = np.concatenate([bx, np.zeros(0) if single else by])
    finite = values[np.isfinite(values)]
    if finite.size and (
        float(np.max(np.abs(finite))) < 1e-2 or float(np.max(np.abs(finite))) > MAX_FIELD_NT
    ):
        raise InvalidObservationError(
            f"magnetic field values span up to {float(np.max(np.abs(finite))):.3g}; "
            "they must be in nT (|B| between 0.01 and 1e5 nT)"
        )


def _to_minute_cadence(
    times: list[datetime],
    bx: np.ndarray[Any, Any],
    by: np.ndarray[Any, Any],
    notes: list[str],
) -> tuple[list[datetime], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Bring a series to the 1-minute cadence the dB/dt tiers are defined on.

    Sub-minute data are averaged into 1-minute bins (gap samples skipped; a
    bin with no finite sample is a gap); 1-minute data pass through.

    Raises:
        InvalidObservationError: On a cadence coarser than 1 minute, which
            under-reads dB/dt.
    """
    if len(times) < 2:
        return times, bx, by
    for ts in times:
        if ts.tzinfo is None:
            raise ValueError(f"Timestamp {ts!r} is naive; must be timezone-aware.")
    offsets = np.array([(t - times[0]).total_seconds() for t in times], dtype=np.float64)
    steps = np.diff(offsets)
    if np.any(steps <= 0.0):
        raise ValueError("Timestamps must be strictly ascending.")
    cadence = float(np.median(steps))
    if cadence > DBDT_CADENCE_S * (1.0 + DBDT_CADENCE_TOLERANCE):
        raise InvalidObservationError(
            f"magnetometer cadence is {cadence:g} s; dB/dt and its tiers are defined on "
            f"{DBDT_CADENCE_S:g} s data and a coarser series under-reads them"
        )
    if cadence >= DBDT_CADENCE_S * (1.0 - DBDT_CADENCE_TOLERANCE):
        return times, bx, by
    first = times[0]
    start = first - timedelta(seconds=first.second, microseconds=first.microsecond)
    bins = ((offsets + (first - start).total_seconds()) // DBDT_CADENCE_S).astype(int)
    n_bins = int(bins[-1]) + 1

    def _mean(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        ok = np.isfinite(values)
        total = np.bincount(bins[ok], weights=values[ok], minlength=n_bins)
        count = np.bincount(bins[ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    notes.append(
        f"{cadence:g} s data averaged to 1-minute means before dB/dt "
        "(the cadence the tiers are defined on)."
    )
    minute_times = [start + timedelta(seconds=DBDT_CADENCE_S * k) for k in range(n_bins)]
    mean_bx, mean_by = _mean(bx), _mean(by)

    # A structurally partial edge bin (the series starts after its first
    # wall-clock minute begins, or ends before its last one does) is NOT a
    # 1-minute mean: its centre sits off the minute's middle, so the first
    # or last dB/dt step spans less than a minute of signal and misreads a
    # steady ramp by the missing fraction (a series starting at :50 turns a
    # ramp's r nT/min into 0.58 r across the first step). The tiers are
    # defined on full 1-minute means, so partial edge bins are dropped.
    slack = DBDT_CADENCE_S * DBDT_CADENCE_TOLERANCE
    lead_missing = (first - start).total_seconds()
    covered_end = lead_missing + float(offsets[-1]) + cadence
    tail_missing = n_bins * DBDT_CADENCE_S - covered_end
    trimmed = []
    if lead_missing > slack:
        trimmed.append("first")
        minute_times, mean_bx, mean_by = minute_times[1:], mean_bx[1:], mean_by[1:]
    if tail_missing > slack and len(minute_times) > 1:
        trimmed.append("last")
        minute_times, mean_bx, mean_by = minute_times[:-1], mean_bx[:-1], mean_by[:-1]
    if trimmed:
        notes.append(
            f"partial edge minute(s) dropped ({' and '.join(trimmed)}): a "
            "partial bin is not a 1-minute mean and would misread dB/dt."
        )
    return minute_times, mean_bx, mean_by


class GICDetector:
    """dB/dt + plane-wave geoelectric GIC risk detector for grid operators.

    Physics-based (finite differences + Cagniard plane-wave impedance);
    no neural network. Consumes minute magnetometer series directly or
    via :meth:`assess_from_datapoints` from
    :class:`~omni_mercury_engine.data_sources.geomagnetic.USGSGeomagnetismSource`
    DataPoints.

    Example:
        >>> detector = GICDetector()
        >>> result = detector.assess(times, bx_nt, by_nt, observatory="BOU")
        >>> result.risk_level, result.peak_dbdt_nt_per_min
    """

    def __init__(
        self,
        conductivity_model: str = "resistive_shield",
        sigma_s_per_m: float | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            conductivity_model: Named half-space model from
                :data:`CONDUCTIVITY_MODELS_S_PER_M`.
            sigma_s_per_m: Explicit conductivity override, S/m. When given,
                ``conductivity_model`` is recorded as ``"custom"``.

        Raises:
            ValueError: On unknown model name or non-positive conductivity.
        """
        if sigma_s_per_m is not None:
            if not math.isfinite(sigma_s_per_m) or sigma_s_per_m <= 0.0:
                raise ValueError(
                    f"Conductivity must be positive and finite; got {sigma_s_per_m!r}."
                )
            self.conductivity_model = "custom"
            self.sigma_s_per_m = float(sigma_s_per_m)
        else:
            if conductivity_model not in CONDUCTIVITY_MODELS_S_PER_M:
                raise ValueError(
                    f"Unknown conductivity model {conductivity_model!r}; "
                    f"known models: {sorted(CONDUCTIVITY_MODELS_S_PER_M)}."
                )
            self.conductivity_model = conductivity_model
            self.sigma_s_per_m = CONDUCTIVITY_MODELS_S_PER_M[conductivity_model]
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_dbdt_nt_per_min(
        times: list[datetime],
        b_nt: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Finite-difference time derivative of a field component.

        Args:
            times: Sample timestamps (timezone-aware, strictly ascending).
            b_nt: Field component values, nT (NaN for gaps).

        Returns:
            Array of length ``len(times) - 1`` with dB/dt in nT/min for
            each step; NaN where either endpoint is a gap.

        Raises:
            ValueError: On fewer than 2 samples, length mismatch, naive
                timestamps, or non-increasing time.
        """
        arr = np.asarray(b_nt, dtype=np.float64).ravel()
        if len(times) < 2:
            raise ValueError(f"dB/dt needs >= 2 samples; got {len(times)}.")
        if len(times) != arr.size:
            raise ValueError(f"times ({len(times)}) and B ({arr.size}) lengths differ.")
        for ts in times:
            if ts.tzinfo is None:
                raise ValueError(f"Timestamp {ts!r} is naive; must be timezone-aware.")
        dt_s = np.array([(t2 - t1).total_seconds() for t1, t2 in pairwise(times)], dtype=np.float64)
        if np.any(dt_s <= 0.0):
            raise ValueError("Timestamps must be strictly ascending.")
        return np.diff(arr) / (dt_s / 60.0)

    def geoelectric_plane_wave_v_per_km(
        self,
        bx_nt: np.ndarray[Any, Any],
        by_nt: np.ndarray[Any, Any],
        dt_s: float,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Surface geoelectric field via the plane-wave half-space relation.

        Frequency-domain filter (Cagniard 1953; Viljanen & Pirjola 1994)::

            Z(omega) = sqrt(i omega mu0 / sigma)
            E_x =  Z * B_y / mu0,   E_y = -Z * B_x / mu0

        Applied in the time domain, where the half-space relation is the
        causal half-order integral of dB/dt (Pirjola 1982; Viljanen et al.
        2004)::

            E_x(t) =  (pi mu0 sigma)^(-1/2) * integral (t - u)^(-1/2) dB_y/du du

        For field increments ``dB_k`` over steps of ``dt`` (B linear within a
        step) this is exactly
        ``E_N = 2 / sqrt(pi mu0 sigma dt) * sum_k dB_k (sqrt(N-k+1) - sqrt(N-k))``,
        a finite linear convolution: E at each time depends only on the field
        up to that time. The previous circular FFT wrapped the end of the
        record onto its start (a 500 nT step in the last 5 minutes produced
        1.54 V/km at sample 0) and fitted its detrend on future samples;
        zero-padding the FFT would still alias the kernel's 1/sqrt(t) tail.

        Args:
            bx_nt: Northward field component, nT (gap-free).
            by_nt: Eastward field component, nT (gap-free).
            dt_s: Uniform sampling interval, seconds.

        Returns:
            Tuple ``(e_x, e_y)`` in V/km, same length as the input.

        Raises:
            ValueError: On gaps (non-finite samples), mismatched lengths,
                fewer than 8 samples, or non-positive dt.
        """
        bx = np.asarray(bx_nt, dtype=np.float64).ravel()
        by = np.asarray(by_nt, dtype=np.float64).ravel()
        if bx.size != by.size:
            raise ValueError(f"bx ({bx.size}) and by ({by.size}) lengths differ.")
        if bx.size < 8:
            raise ValueError(f"Geoelectric proxy needs >= 8 samples; got {bx.size}.")
        if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(by))):
            raise ValueError(
                "Geoelectric proxy requires a gap-free series; interpolating "
                "over gaps would fabricate spectral content."
            )
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive and finite; got {dt_s!r}.")

        n = bx.size
        d_bx = np.diff(bx, prepend=bx[0]) * 1e-9  # nT -> T increments
        d_by = np.diff(by, prepend=by[0]) * 1e-9
        lag = np.arange(n, dtype=np.float64)
        kernel = np.sqrt(lag + 1.0) - np.sqrt(lag)
        scale = 2.0 / math.sqrt(math.pi * MU0 * self.sigma_s_per_m * dt_s)
        e_x = scale * fftconvolve(d_by, kernel)[:n]  # V/m
        e_y = -scale * fftconvolve(d_bx, kernel)[:n]
        return e_x * 1000.0, e_y * 1000.0  # V/km

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        times: list[datetime],
        bx_nt: np.ndarray[Any, Any],
        by_nt: np.ndarray[Any, Any] | None = None,
        observatory: str = "",
    ) -> GICAssessment:
        """Assess GIC grid risk from a magnetometer minute series.

        Args:
            times: Sample timestamps (timezone-aware, strictly ascending).
            bx_nt: Northward (X) — or horizontal H when only one component
                exists — field values in nT; NaN/None allowed as gaps.
            by_nt: Eastward (Y) field values in nT, or None for a
                single-component (H-only) assessment.
            observatory: Station identifier for reporting.

        Returns:
            Full GIC risk assessment.

        Raises:
            ValueError: On the input contracts of
                :meth:`compute_dbdt_nt_per_min`, or if every dB/dt step
                falls in a data gap.
        """
        bx = np.asarray(
            [np.nan if v is None else v for v in np.asarray(bx_nt, dtype=object).ravel()],
            dtype=np.float64,
        )
        single_component = by_nt is None
        if single_component:
            by = np.zeros_like(bx)
        else:
            by = np.asarray(
                [np.nan if v is None else v for v in np.asarray(by_nt, dtype=object).ravel()],
                dtype=np.float64,
            )
            if by.size != bx.size:
                raise ValueError(f"bx ({bx.size}) and by ({by.size}) lengths differ.")
        if len(times) != bx.size:
            raise ValueError(f"times ({len(times)}) and B ({bx.size}) lengths differ.")
        _check_field_units(bx, by, single_component)

        def _non_finite(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> int:
            return int(
                np.sum(~np.isfinite(x)) + (0 if single_component else np.sum(~np.isfinite(y)))
            )

        # Counted on the input, before any averaging to 1-minute means.
        n_gaps = _non_finite(bx, by)
        notes: list[str] = []
        minute_times, bx, by = _to_minute_cadence(times, bx, by, notes)
        averaged = minute_times is not times  # 1-minute data come back as is
        times = minute_times

        dbdt_x = self.compute_dbdt_nt_per_min(times, bx)
        dbdt_y = self.compute_dbdt_nt_per_min(times, by)
        dbdt_h = np.sqrt(dbdt_x**2 + dbdt_y**2)

        finite = np.isfinite(dbdt_h)
        if not finite.any():
            raise ValueError(
                f"{observatory or 'series'}: every dB/dt step falls in a data "
                "gap; refusing to report a risk level from no usable data."
            )
        if not averaged:
            if n_gaps:
                notes.append(f"{n_gaps} non-finite input samples excluded pairwise from dB/dt.")
        else:
            if n_gaps:
                notes.append(f"{n_gaps} non-finite input samples skipped in the 1-minute means.")
            empty_bins = _non_finite(bx, by)
            if empty_bins:
                notes.append(
                    f"{empty_bins} minute bins (X and Y each counted) with no finite "
                    "sample excluded pairwise from dB/dt."
                )
        if single_component:
            notes.append(
                "Single-component (H-only) assessment: dB/dt and the geoelectric proxy "
                "use B_y = 0, so dB/dt, |E| and the risk level are lower bounds "
                "(d|H|/dt can be far below the vector dB/dt)."
            )

        peak_idx = int(np.nanargmax(np.where(finite, dbdt_h, -np.inf)))
        peak_dbdt = float(dbdt_h[peak_idx])
        peak_time = times[peak_idx + 1]  # derivative step ends at i+1

        sustained = self._sustained_max(dbdt_h)
        if sustained is None:
            notes.append(
                f"Series too short or gappy for the {SUSTAINED_WINDOW_SAMPLES}-sample "
                "sustained-dB/dt window."
            )

        geoelectric_peak: float | None = None
        dt_all = np.array(
            [(t2 - t1).total_seconds() for t1, t2 in pairwise(times)], dtype=np.float64
        )
        uniform = bool(np.all(np.abs(dt_all - dt_all[0]) < 1e-6))
        gap_free = bool(np.all(np.isfinite(bx)) and np.all(np.isfinite(by)))
        if gap_free and uniform and bx.size >= 8:
            e_x, e_y = self.geoelectric_plane_wave_v_per_km(bx, by, float(dt_all[0]))
            geoelectric_peak = float(np.max(np.hypot(e_x, e_y)))
        else:
            reasons = []
            if not gap_free:
                reasons.append("data gaps")
            if not uniform:
                reasons.append("non-uniform sampling")
            if bx.size < 8:
                reasons.append("fewer than 8 samples")
            notes.append("Geoelectric plane-wave proxy not computed (" + ", ".join(reasons) + ").")

        assessment = GICAssessment(
            observatory=observatory,
            risk_level=classify_dbdt_risk(peak_dbdt),
            peak_dbdt_nt_per_min=peak_dbdt,
            peak_dbdt_time=peak_time,
            peak_dbdt_x_nt_per_min=(
                float(np.nanmax(np.abs(dbdt_x))) if np.isfinite(dbdt_x).any() else None
            ),
            peak_dbdt_y_nt_per_min=(
                None
                if single_component
                else (float(np.nanmax(np.abs(dbdt_y))) if np.isfinite(dbdt_y).any() else None)
            ),
            sustained_dbdt_nt_per_min=sustained,
            geoelectric_peak_v_per_km=geoelectric_peak,
            conductivity_model=self.conductivity_model,
            sigma_s_per_m=self.sigma_s_per_m,
            n_samples=len(times),
            n_gaps=n_gaps,
            single_component=single_component,
            notes=notes,
        )
        self.logger.info(
            "GIC %s: %s (peak dB/dt %.1f nT/min at %s; E-proxy %s V/km, %s)",
            observatory or "<series>",
            assessment.risk_level,
            peak_dbdt,
            peak_time.isoformat(),
            f"{geoelectric_peak:.3f}" if geoelectric_peak is not None else "n/a",
            self.conductivity_model,
        )
        return assessment

    @staticmethod
    def _sustained_max(dbdt_h: np.ndarray[Any, Any]) -> float | None:
        """Max rolling mean over gap-free 10-sample windows (None if none)."""
        window = SUSTAINED_WINDOW_SAMPLES
        if dbdt_h.size < window:
            return None
        best: float | None = None
        for start in range(dbdt_h.size - window + 1):
            chunk = dbdt_h[start : start + window]
            if np.all(np.isfinite(chunk)):
                mean = float(np.mean(chunk))
                if best is None or mean > best:
                    best = mean
        return best

    # ------------------------------------------------------------------
    # DataPoint ingestion (USGSGeomagnetismSource)
    # ------------------------------------------------------------------

    def assess_from_datapoints(self, points: list[DataPoint]) -> dict[str, GICAssessment]:
        """Assess GIC risk per observatory from magnetometer DataPoints.

        Expects DataPoints shaped like
        :class:`~omni_mercury_engine.data_sources.geomagnetic.USGSGeomagnetismSource`
        output: ``data = {"observatory": code, "elements": {"X": nT, ...}}``.
        Prefers X/Y components; falls back to an H-only assessment when
        only H is present.

        Args:
            points: Magnetometer DataPoints (any observatory mix).

        Returns:
            Mapping of observatory code to assessment.

        Raises:
            ValueError: If no usable magnetometer points are supplied, or an
                observatory offers neither X/Y nor H elements.
        """
        by_observatory: dict[str, list[DataPoint]] = {}
        for point in points:
            elements = point.data.get("elements")
            if not isinstance(elements, dict):
                continue
            code = str(point.data.get("observatory", "")) or "unknown"
            by_observatory.setdefault(code, []).append(point)

        if not by_observatory:
            raise ValueError(
                "No magnetometer DataPoints with an 'elements' payload were "
                "supplied; cannot assess GIC risk."
            )

        results: dict[str, GICAssessment] = {}
        for code, obs_points in by_observatory.items():
            obs_points.sort(key=lambda p: p.timestamp)
            times = [p.timestamp for p in obs_points]
            elements_seen: set[str] = set()
            for p in obs_points:
                elements_seen.update(p.data["elements"].keys())

            def _series(element: str, pts: list[DataPoint]) -> np.ndarray[Any, Any]:
                return np.array(
                    [p.data["elements"].get(element, np.nan) for p in pts],
                    dtype=np.float64,
                )

            if {"X", "Y"} <= elements_seen:
                results[code] = self.assess(
                    times, _series("X", obs_points), _series("Y", obs_points), observatory=code
                )
            elif "H" in elements_seen:
                results[code] = self.assess(times, _series("H", obs_points), None, observatory=code)
            else:
                raise ValueError(
                    f"Observatory {code}: neither X/Y nor H elements present "
                    f"(saw {sorted(elements_seen)}); request X,Y from the "
                    "magnetometer source for GIC assessment."
                )
        return results


__all__ = [
    "CONDUCTIVITY_MODELS_S_PER_M",
    "DBDT_RISK_TIERS_NT_PER_MIN",
    "MU0",
    "SUSTAINED_WINDOW_SAMPLES",
    "GICAssessment",
    "GICDetector",
    "classify_dbdt_risk",
]
