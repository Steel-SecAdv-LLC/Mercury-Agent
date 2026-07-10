# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Atmospheric River Detector - IVT computation and the Ralph et al. AR scale.

Implements the literature-standard atmospheric-river diagnostics:

- **Integrated water Vapor Transport (IVT)** from supplied specific-humidity
  and wind profiles across pressure levels::

      IVT_u = (1/g) * integral( q * u  dp )
      IVT_v = (1/g) * integral( q * v  dp )
      IVT   = sqrt(IVT_u^2 + IVT_v^2)        [kg m^-1 s^-1]

  integrated with the trapezoidal rule over the supplied pressure levels
  (standard practice, e.g. Ralph et al. 2019 and the AR detection
  literature).  Level ordering, plausible units (hPa pressure, kg/kg
  humidity, m/s winds) and shape consistency are validated; the detector
  fails loudly rather than guessing units.  No vertical profile is ever
  fabricated - where only single-level data exists, a precomputed IVT
  series is accepted instead.

- **AR scale** (Ralph, Rutz, Cordeira et al. 2019, "A scale to characterize
  the strength and impacts of atmospheric rivers", *Bull. Amer. Meteor.
  Soc.* 100, 269-289): AR conditions exist while IVT >= 250 kg m^-1 s^-1.
  The preliminary rank is assigned from the maximum IVT during the episode
  (250-500 -> 1, 500-750 -> 2, 750-1000 -> 3, 1000-1250 -> 4, >= 1250 -> 5)
  and then adjusted by the episode duration: if AR conditions persist for
  more than 48 h the rank is promoted by one (capped at AR5); if they last
  less than 24 h the rank is demoted by one (an AR1 episode shorter than
  24 h drops off the scale entirely, rank 0).

This is a pure physics core: it works untrained and uses no neural
networks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Standard gravitational acceleration (m s^-2), CODATA.
GRAVITY_M_S2: float = 9.80665

#: IVT threshold for AR conditions (kg m^-1 s^-1), Ralph et al. (2019).
AR_IVT_THRESHOLD: float = 250.0

#: Preliminary-rank IVT bin edges (kg m^-1 s^-1), Ralph et al. (2019).
_RANK_EDGES: tuple[float, ...] = (250.0, 500.0, 750.0, 1000.0, 1250.0)

#: Duration rules (hours), Ralph et al. (2019): > 48 h promotes one rank,
#: < 24 h demotes one rank.
_PROMOTION_HOURS: float = 48.0
_DEMOTION_HOURS: float = 24.0

#: Plausibility bounds used for unit validation.
_PRESSURE_MIN_HPA: float = 50.0
_PRESSURE_MAX_HPA: float = 1100.0
_Q_MAX_KG_KG: float = 0.05
_WIND_MAX_M_S: float = 150.0


@dataclass
class IVTResult:
    """IVT computed from one vertical profile (or a time series of profiles).

    Attributes:
        ivt: IVT magnitude(s), kg m^-1 s^-1.
        ivt_u: Zonal component(s) of the vapor-transport integral.
        ivt_v: Meridional component(s).
    """

    ivt: np.ndarray
    ivt_u: np.ndarray
    ivt_v: np.ndarray


@dataclass
class AREpisode:
    """One contiguous period of AR conditions (IVT >= 250 kg m^-1 s^-1)."""

    start_index: int
    end_index: int
    duration_hours: float
    max_ivt: float
    mean_ivt: float
    preliminary_rank: int
    final_rank: int
    label: str  # "AR1".."AR5", or "below_scale" when demoted to 0


@dataclass
class ARAssessmentResult:
    """Result of an AR-scale assessment over an IVT time series.

    Attributes:
        ar_detected: True when at least one episode retains rank >= 1.
        episodes: All episodes of AR conditions found in the series.
        strongest: The episode with the highest final rank (ties broken by
            max IVT), or None.
        max_ivt: Series-wide maximum IVT.
    """

    ar_detected: bool
    episodes: list[AREpisode] = field(default_factory=list)
    strongest: AREpisode | None = None
    max_ivt: float = 0.0
    warning_actions: list[str] = field(default_factory=list)


def _validate_profile_inputs(
    q_kg_kg: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    pressure_hpa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate profile arrays: shapes, finiteness, units, level ordering.

    Returns:
        The validated arrays as float64, with 1-D profiles promoted to
        shape (1, n_levels).

    Raises:
        ValueError: On any inconsistency - the caller gets an actionable
            message naming the offending quantity and the expected units.
    """
    q = np.asarray(q_kg_kg, dtype=np.float64)
    u = np.asarray(u_m_s, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    p = np.asarray(pressure_hpa, dtype=np.float64)

    if p.ndim != 1:
        raise ValueError(f"pressure_hpa must be 1-D (levels), got shape {p.shape}")
    if p.size < 3:
        raise ValueError(
            f"need >= 3 pressure levels to resolve a moisture-flux profile, got {p.size}"
        )
    if q.ndim == 1:
        q, u, v = q[None, :], u[None, :], v[None, :]
    if not (q.shape == u.shape == v.shape):
        raise ValueError(f"q/u/v shapes differ: {q.shape} / {u.shape} / {v.shape}")
    if q.shape[1] != p.size:
        raise ValueError(f"profiles have {q.shape[1]} levels but pressure_hpa has {p.size}")
    for name, arr in (("q_kg_kg", q), ("u_m_s", u), ("v_m_s", v), ("pressure_hpa", p)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    diffs = np.diff(p)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            "pressure_hpa must be strictly monotonic (either surface-to-top "
            "or top-to-surface); mixed ordering indicates corrupted levels"
        )
    if np.any(p < _PRESSURE_MIN_HPA) or np.any(p > _PRESSURE_MAX_HPA):
        raise ValueError(
            f"pressure_hpa outside [{_PRESSURE_MIN_HPA}, {_PRESSURE_MAX_HPA}] hPa "
            "- if you passed pascals, convert to hPa first"
        )
    if np.any(q < 0.0) or np.any(q > _Q_MAX_KG_KG):
        raise ValueError(
            f"q_kg_kg outside [0, {_Q_MAX_KG_KG}] kg/kg - if you passed g/kg, "
            "divide by 1000 first"
        )
    if np.any(np.abs(u) > _WIND_MAX_M_S) or np.any(np.abs(v) > _WIND_MAX_M_S):
        raise ValueError(f"wind components exceed {_WIND_MAX_M_S} m/s - check units (m/s)")
    return q, u, v, p


def compute_ivt(
    q_kg_kg: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    pressure_hpa: np.ndarray,
) -> IVTResult:
    """Compute IVT from specific-humidity and wind profiles.

    IVT_u = (1/g) |integral(q u dp)| per component, trapezoidal rule over
    the supplied levels; magnitude = hypot of components.  Components keep
    the sign of the vertically integrated flux (positive = eastward /
    northward transport).

    Args:
        q_kg_kg: Specific humidity (kg/kg), shape (n_levels,) for a single
            profile or (n_times, n_levels) for a series of profiles.
        u_m_s: Zonal wind (m/s), same shape as ``q_kg_kg``.
        v_m_s: Meridional wind (m/s), same shape.
        pressure_hpa: Pressure levels (hPa), strictly monotonic, >= 3
            levels.

    Returns:
        An :class:`IVTResult`; arrays have shape (n_times,) (length 1 for
        a single profile).

    Raises:
        ValueError: On malformed shapes, non-finite values, implausible
            units, or non-monotonic level ordering.
    """
    q, u, v, p = _validate_profile_inputs(q_kg_kg, u_m_s, v_m_s, pressure_hpa)

    # Integrate over ascending pressure (top -> surface) so dp > 0; the
    # physical integral (1/g) int_top^surface q V dp keeps flux signs.
    order = np.argsort(p)
    p_pa = p[order] * 100.0
    q_o, u_o, v_o = q[:, order], u[:, order], v[:, order]

    # numpy 2.x renamed trapz -> trapezoid; the 1.x branch is evaluated
    # lazily only when trapezoid is absent, so the attribute reference is
    # runtime-safe on both majors (mypy checks against 2.x stubs).
    trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined, unused-ignore]
    ivt_u = trapz(q_o * u_o, x=p_pa, axis=1) / GRAVITY_M_S2
    ivt_v = trapz(q_o * v_o, x=p_pa, axis=1) / GRAVITY_M_S2
    ivt = np.hypot(ivt_u, ivt_v)
    return IVTResult(ivt=ivt, ivt_u=ivt_u, ivt_v=ivt_v)


def _preliminary_rank(max_ivt: float) -> int:
    """Preliminary AR rank from maximum IVT (Ralph et al. 2019)."""
    rank = 0
    for edge in _RANK_EDGES:
        if max_ivt >= edge:
            rank += 1
    return rank


class AtmosphericRiverDetector:
    """Atmospheric-river detector: IVT physics core + Ralph et al. AR scale.

    Accepts either full vertical profiles (specific humidity + winds over
    pressure levels, from which IVT is computed) or a precomputed IVT time
    series when only single-level data exists upstream.  Profiles are never
    fabricated.
    """

    def __init__(self, ivt_threshold: float = AR_IVT_THRESHOLD) -> None:
        """Initialize the detector.

        Args:
            ivt_threshold: IVT threshold defining AR conditions
                (250 kg m^-1 s^-1 in Ralph et al. 2019).

        Raises:
            ValueError: If the threshold is not positive.
        """
        if ivt_threshold <= 0.0:
            raise ValueError(f"ivt_threshold must be > 0, got {ivt_threshold}")
        self.ivt_threshold = ivt_threshold
        self.logger = logging.getLogger(__name__)

    def classify_ar_scale(
        self,
        ivt_series: np.ndarray,
        dt_hours: float | None = None,
        timestamps_hours: np.ndarray | None = None,
    ) -> ARAssessmentResult:
        """Apply the Ralph et al. (2019) AR scale to an IVT time series.

        Episodes are maximal runs with IVT >= threshold.  Episode duration
        is the inclusive span ``(n_steps) * dt`` (uniform spacing) or the
        difference of bounding timestamps plus one step (explicit
        timestamps).  Duration-dependent promotion/demotion follows the
        paper: > 48 h promotes one rank (cap AR5), < 24 h demotes one rank
        (AR1 episodes fall below the scale).

        Args:
            ivt_series: 1-D IVT magnitudes (kg m^-1 s^-1).
            dt_hours: Uniform time step in hours (mutually exclusive with
                ``timestamps_hours``).
            timestamps_hours: Explicit sample times in hours, strictly
                increasing, same length as ``ivt_series``.

        Returns:
            An :class:`ARAssessmentResult`.

        Raises:
            ValueError: On malformed series, negative IVT, missing/both
                time specifications, or non-increasing timestamps.
        """
        ivt = np.asarray(ivt_series, dtype=np.float64)
        if ivt.ndim != 1 or ivt.size == 0:
            raise ValueError(f"ivt_series must be non-empty 1-D, got shape {ivt.shape}")
        if not np.all(np.isfinite(ivt)):
            raise ValueError("ivt_series contains non-finite values")
        if np.any(ivt < 0.0):
            raise ValueError("ivt_series contains negative values; IVT is a magnitude")
        if (dt_hours is None) == (timestamps_hours is None):
            raise ValueError("supply exactly one of dt_hours or timestamps_hours")
        if dt_hours is not None and dt_hours <= 0.0:
            raise ValueError(f"dt_hours must be > 0, got {dt_hours}")
        times: np.ndarray | None = None
        if timestamps_hours is not None:
            times = np.asarray(timestamps_hours, dtype=np.float64)
            if times.shape != ivt.shape:
                raise ValueError("timestamps_hours must match ivt_series length")
            if np.any(np.diff(times) <= 0.0):
                raise ValueError("timestamps_hours must be strictly increasing")

        above = ivt >= self.ivt_threshold
        episodes: list[AREpisode] = []
        run_start: int | None = None
        for i in range(above.size + 1):
            if i < above.size and above[i]:
                if run_start is None:
                    run_start = i
                continue
            if run_start is not None:
                end = i - 1
                if times is not None:
                    # Median step approximates the sampling interval so the
                    # inclusive span covers the final sample.
                    step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
                    duration = float(times[end] - times[run_start]) + step
                else:
                    assert dt_hours is not None
                    duration = (end - run_start + 1) * dt_hours
                seg = ivt[run_start : end + 1]
                max_ivt = float(np.max(seg))
                prelim = _preliminary_rank(max_ivt)
                final = prelim
                if duration > _PROMOTION_HOURS:
                    final = min(prelim + 1, 5)
                elif duration < _DEMOTION_HOURS:
                    final = max(prelim - 1, 0)
                episodes.append(
                    AREpisode(
                        start_index=run_start,
                        end_index=end,
                        duration_hours=duration,
                        max_ivt=max_ivt,
                        mean_ivt=float(np.mean(seg)),
                        preliminary_rank=prelim,
                        final_rank=final,
                        label=f"AR{final}" if final >= 1 else "below_scale",
                    )
                )
                run_start = None

        ranked = [e for e in episodes if e.final_rank >= 1]
        strongest = max(ranked, key=lambda e: (e.final_rank, e.max_ivt)) if ranked else None
        result = ARAssessmentResult(
            ar_detected=bool(ranked),
            episodes=episodes,
            strongest=strongest,
            max_ivt=float(np.max(ivt)),
            warning_actions=self._generate_warnings(strongest),
        )
        self.logger.info(
            "AR scale: %d episode(s), strongest=%s",
            len(episodes),
            strongest.label if strongest else "none",
        )
        return result

    def analyze_profiles(
        self,
        q_kg_kg: np.ndarray,
        u_m_s: np.ndarray,
        v_m_s: np.ndarray,
        pressure_hpa: np.ndarray,
        dt_hours: float,
    ) -> tuple[IVTResult, ARAssessmentResult]:
        """Compute IVT from profiles, then classify on the AR scale.

        Args:
            q_kg_kg: Specific humidity (n_times, n_levels), kg/kg.
            u_m_s: Zonal wind, same shape, m/s.
            v_m_s: Meridional wind, same shape, m/s.
            pressure_hpa: Pressure levels (hPa).
            dt_hours: Time step between profiles in hours.

        Returns:
            Tuple of (:class:`IVTResult`, :class:`ARAssessmentResult`).

        Raises:
            ValueError: Propagated from validation.
        """
        ivt_result = compute_ivt(q_kg_kg, u_m_s, v_m_s, pressure_hpa)
        assessment = self.classify_ar_scale(ivt_result.ivt, dt_hours=dt_hours)
        return ivt_result, assessment

    @staticmethod
    def _generate_warnings(strongest: AREpisode | None) -> list[str]:
        """Generate advisory strings from the strongest episode."""
        if strongest is None:
            return []
        warnings: list[str] = []
        if strongest.final_rank >= 4:
            warnings.append(
                f"{strongest.label}: primarily hazardous - extreme rainfall " "and flood potential"
            )
        elif strongest.final_rank == 3:
            warnings.append(f"{strongest.label}: balance of beneficial and hazardous impacts")
        else:
            warnings.append(f"{strongest.label}: primarily beneficial precipitation")
        return warnings

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract a fixed 20-dim feature vector for ML fusion.

        Treats the input as a (precomputed, non-negative) IVT-like series
        when possible and emits threshold occupancy / episode statistics;
        otherwise emits robust summary statistics only.

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

        features: list[float] = [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.median(flat)),
        ]
        if np.all(flat >= 0.0):
            above = flat >= self.ivt_threshold
            longest = current = 0
            for flag in above:
                current = current + 1 if flag else 0
                longest = max(longest, current)
            features.extend(
                [
                    float(np.mean(above)),
                    float(longest),
                    float(_preliminary_rank(float(np.max(flat)))),
                ]
            )
        while len(features) < 20:
            features.append(0.0)
        return torch.tensor(features[:20], dtype=torch.float32)
