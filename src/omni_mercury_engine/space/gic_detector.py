# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Geomagnetically induced current (GIC) grid-risk detection from magnetometer data.

Consumes ground-magnetometer minute data (e.g.
:class:`omni_mercury_engine.data_sources.geomagnetic.USGSGeomagnetismSource`,
which serves real per-observatory 1-minute X/Y/Z/F values from the USGS
Geomagnetism edge web service) and produces:

* **dB/dt metrics** — per-component and horizontal-magnitude time
  derivatives in nT/min, peak value + timestamp, and a sustained metric
  (maximum 10-minute rolling mean). Rapid horizontal-field variation is the
  primary driver of GIC in transmission networks (Viljanen et al. 2001,
  Ann. Geophys. 19; Pulkkinen et al. 2013, Space Weather 11).

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

* **Risk tiers on dB/dt** — operational alerting tiers at 100 / 300 /
  500 nT/min (low < 100, moderate 100-300, high 300-500, severe >= 500)
  following the range of published dB/dt hazard scales (Molinski 2002,
  J. Atmos. Sol.-Terr. Phys. 64; Marshall et al. 2012, Space Weather 10;
  Kappenman 2006, Space Weather 4). Context: the March 1989 Hydro-Québec
  collapse was associated with ~480 nT/min at ground level (Bolduc 2002,
  J. Atmos. Sol.-Terr. Phys. 64; Boteler 2019, Space Weather 17), while
  Carrington-class scenarios are benchmarked near ~5000 nT/min (Kappenman
  2006) — the reference context for NERC TPL-007 benchmark GMD planning.

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
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

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

#: Operational dB/dt alerting tiers, nT/min (see module docstring).
DBDT_RISK_TIERS_NT_PER_MIN: tuple[tuple[float, str], ...] = (
    (500.0, "severe"),
    (300.0, "high"),
    (100.0, "moderate"),
)

#: Rolling window (samples at minute cadence) for the sustained metric.
SUSTAINED_WINDOW_SAMPLES: int = 10


def classify_dbdt_risk(peak_dbdt_nt_per_min: float) -> str:
    """Classify peak horizontal dB/dt against the operational tiers.

    Tiers: low < 100, moderate 100-300, high 300-500, severe >= 500 nT/min
    (Molinski 2002; Marshall et al. 2012; Kappenman 2006).

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
            horizontal dB/dt (None if the series is too short/gappy).
        geoelectric_peak_v_per_km: Peak plane-wave |E| proxy, V/km (None
            when the series has gaps — never interpolated silently).
        conductivity_model: Named ground model used for the proxy.
        sigma_s_per_m: Conductivity value used, S/m.
        n_samples: Total input samples.
        n_gaps: Count of non-finite input samples excluded from dB/dt.
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

        The input series are linearly detrended before the FFT (removes the
        static field and secular drift; the zero-frequency bin carries no
        induced field).

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

        def _detrend(series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            x = np.arange(series.size, dtype=np.float64)
            slope, intercept = np.polyfit(x, series, 1)
            return series - (slope * x + intercept)

        bx_t = _detrend(bx) * 1e-9  # nT -> T
        by_t = _detrend(by) * 1e-9

        n = bx_t.size
        omega = 2.0 * math.pi * np.fft.rfftfreq(n, dt_s)
        impedance = np.sqrt(1j * omega * MU0 / self.sigma_s_per_m)  # ohm
        impedance[0] = 0.0

        e_x = np.fft.irfft(impedance * np.fft.rfft(by_t), n=n) / MU0  # V/m
        e_y = -np.fft.irfft(impedance * np.fft.rfft(bx_t), n=n) / MU0
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

        dbdt_x = self.compute_dbdt_nt_per_min(times, bx)
        dbdt_y = self.compute_dbdt_nt_per_min(times, by)
        dbdt_h = np.sqrt(dbdt_x**2 + dbdt_y**2)

        finite = np.isfinite(dbdt_h)
        if not finite.any():
            raise ValueError(
                f"{observatory or 'series'}: every dB/dt step falls in a data "
                "gap; refusing to report a risk level from no usable data."
            )
        n_gaps = int(
            np.sum(~np.isfinite(bx)) + (0 if single_component else np.sum(~np.isfinite(by)))
        )
        notes: list[str] = []
        if n_gaps:
            notes.append(f"{n_gaps} non-finite input samples excluded pairwise from dB/dt.")
        if single_component:
            notes.append(
                "Single-component (H-only) assessment: dB/dt and the "
                "geoelectric proxy use B_y = 0; |E| is a lower bound."
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
