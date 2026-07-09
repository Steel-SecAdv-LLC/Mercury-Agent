# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rockfall Detector — triggers-and-precursors physics for rock-slope failure.

Deterministic, literature-anchored analysis of the four observable channels
used in operational rockfall monitoring. No neural network, no fabricated
signal: every channel that lacks input data is honestly reported as
"not assessed" and contributes nothing to the hazard level.

Channels implemented:

- **Freeze-thaw cycle counting** from a rock-face temperature series:
  effective frost weathering requires both oscillation of the rock surface
  through 0 °C and available moisture (Matsuoka & Murton, 2008), so a cycle
  is only counted when a downward 0 °C crossing occurs with moisture present
  (recent precipitation or an explicit wetness flag). Time spent inside the
  frost-cracking window (−8 to −3 °C; Anderson, 1998; Hales & Roering, 2007)
  is reported separately as a sustained ice-segregation indicator.
  Freeze-thaw episodes multiply observed rockfall frequency by a factor of
  up to ~7 at instrumented cliffs (D'Amato et al., 2016).
- **Precipitation-intensity trigger**: rockfall frequency increases sharply
  with rainfall; at the Mont Saint-Eynard limestone cliff the frequency is
  multiplied by up to ~26 when the mean episode intensity exceeds 5 mm/h
  (D'Amato et al., 2016). The published 5 mm/h mean-episode-intensity
  threshold and the published frequency multipliers are used directly.
- **Inverse-velocity failure forecasting** on a crack-aperture /
  extensometer displacement series (Fukuzono, 1985): for accelerating
  tertiary creep with ``dv/dt = A v^alpha`` and the commonly observed
  ``alpha = 2``, the inverse velocity ``1/v`` decreases *linearly* with time
  and extrapolates to zero at the failure time. The implementation smooths
  the velocity (short moving average, per the operational guidance of
  Carlà et al., 2017), fits ``1/v`` against time over the accelerating tail
  and gates the forecast on fit quality (R² >= 0.80, documented; negative
  slope; forecast in the future). A ±1σ slope/intercept propagation gives
  the failure-time window.
- **Microseismic event-rate ramp**: precursory acceleration of microseismic
  event rates is documented before cliff collapses (Amitrano et al., 2005).
  The recent-window event rate is compared against the baseline rate with a
  Poisson tail test; a ramp requires both a rate ratio >= 3 and
  ``P(N >= n_recent | baseline)`` < 0.01.

References:
    - Fukuzono, T. (1985). A new method for predicting the failure time of
      a slope. Proc. IVth Int. Conf. and Field Workshop on Landslides,
      Tokyo, 145-150.
    - Carlà, T., Intrieri, E., Di Traglia, F., Nolesini, T., Gigli, G.,
      Casagli, N. (2017). Guidelines on the use of inverse velocity method
      as a tool for setting alarm thresholds and forecasting landslides and
      structure collapses. Landslides 14(2), 517-534.
    - Matsuoka, N., Murton, J. (2008). Frost weathering: recent advances
      and future directions. Permafrost and Periglacial Processes 19(2),
      195-210.
    - Anderson, R.S. (1998). Near-surface thermal profiles in alpine
      bedrock: implications for the frost weathering of rock. Arctic and
      Alpine Research 30(4), 362-372. (Frost-cracking window ~ −3 to −8 °C;
      also Hales & Roering, 2007, JGR 112, F02033.)
    - D'Amato, J., Hantz, D., Guerin, A., Jaboyedoff, M., Baillet, L.,
      Mariscal, A. (2016). Influence of meteorological factors on rockfall
      occurrence in a middle mountain limestone cliff. NHESS 16, 719-735.
    - Amitrano, D., Grasso, J.R., Senfaute, G. (2005). Seismic precursory
      patterns before a cliff collapse and critical point phenomena.
      Geophysical Research Letters 32, L08314.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

#: Frost-cracking window bounds in °C (Anderson 1998; Hales & Roering 2007).
FROST_CRACKING_WINDOW_C: tuple[float, float] = (-8.0, -3.0)

#: Mean-episode rainfall intensity threshold, mm/h (D'Amato et al. 2016).
RAIN_INTENSITY_THRESHOLD_MM_H = 5.0

#: Published rockfall frequency multiplier during freeze-thaw episodes
#: (D'Amato et al. 2016: "as high as 7").
FREEZE_THAW_FREQUENCY_MULTIPLIER = 7.0

#: Published rockfall frequency multiplier for mean episode intensity above
#: 5 mm/h (D'Amato et al. 2016: "26").
HIGH_RAIN_FREQUENCY_MULTIPLIER = 26.0

#: Inverse-velocity fit-quality gate (documented; Carlà et al. 2017 stress
#: that noisy 1/v fits must not be extrapolated).
INVERSE_VELOCITY_R2_GATE = 0.80


class RockfallHazardLevel(Enum):
    """Rockfall hazard classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FreezeThawResult:
    """Freeze-thaw weathering assessment.

    Attributes:
        effective_cycles: 0 °C downward crossings with moisture present.
        total_crossings: All 0 °C downward crossings, wet or dry.
        cycles_per_day: Effective cycle rate.
        frost_cracking_fraction: Fraction of samples inside the
            frost-cracking window (−8 to −3 °C).
        active: True when at least one effective cycle occurred.
    """

    effective_cycles: int
    total_crossings: int
    cycles_per_day: float
    frost_cracking_fraction: float
    active: bool


@dataclass
class InverseVelocityResult:
    """Fukuzono (1985) inverse-velocity forecast record.

    Attributes:
        accelerating: True when the accelerating-creep selection found a
            usable tail (increasing velocity).
        forecast_valid: True when the linear 1/v fit passed every gate.
        failure_time: Predicted failure time (same units/origin as the
            input time axis) — only set when ``forecast_valid``.
        failure_window: (earliest, latest) failure time from ±1σ
            propagation of the fit parameters — only when valid.
        r_squared: R² of the 1/v linear fit (NaN if never fitted).
        slope: Fitted slope of 1/v vs t (negative when approaching failure).
        n_points_fit: Number of velocity points in the fit.
        rejection_reason: Why the forecast was rejected (empty when valid).
    """

    accelerating: bool
    forecast_valid: bool
    failure_time: float | None = None
    failure_window: tuple[float, float] | None = None
    r_squared: float = float("nan")
    slope: float = float("nan")
    n_points_fit: int = 0
    rejection_reason: str = ""


@dataclass
class RockfallPredictionResult:
    """Full rockfall hazard assessment.

    Attributes:
        hazard_level: One of :class:`RockfallHazardLevel` values.
        confidence: Deterministic evidence score in [0, 1] over the
            channels that actually ran.
        freeze_thaw: Freeze-thaw assessment (None = not assessed).
        rain_trigger_active: Mean episode intensity above the published
            threshold (None = not assessed).
        rain_mean_intensity_mm_h: Mean intensity of the ongoing episode.
        estimated_frequency_multiplier: Published relative-frequency
            multiplier implied by the active meteorological triggers
            (max of the applicable published factors, not their product).
        inverse_velocity: Inverse-velocity forecast (None = not assessed).
        microseismic_ramp: Event-rate ramp detected (None = not assessed).
        microseismic_rate_ratio: recent/baseline rate ratio.
        channels_assessed: Names of channels that received data.
        evidence: Ordered evidence trail with citations.
        warnings: Operational warning strings.
    """

    hazard_level: str
    confidence: float
    freeze_thaw: FreezeThawResult | None = None
    rain_trigger_active: bool | None = None
    rain_mean_intensity_mm_h: float | None = None
    estimated_frequency_multiplier: float = 1.0
    inverse_velocity: InverseVelocityResult | None = None
    microseismic_ramp: bool | None = None
    microseismic_rate_ratio: float | None = None
    channels_assessed: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RockfallDetector:
    """Rockfall trigger and precursor detector.

    Args:
        rain_intensity_threshold_mm_h: Mean-episode intensity threshold.
            Default 5.0 (D'Amato et al. 2016).
        iv_r2_gate: Inverse-velocity R² gate. Default 0.80.
        iv_min_points: Minimum velocity points for an inverse-velocity fit.
            Default 5.
        iv_smooth_window: Moving-average window (samples) applied to the
            velocity before fitting (Carlà et al. 2017 short-term filter).
            Default 3.
        ramp_rate_ratio: Required recent/baseline microseismic rate ratio.
            Default 3.0.
        ramp_p_value: Poisson tail significance for the ramp. Default 0.01.
        moisture_window_samples: How many samples after precipitation the
            rock face is considered wet for freeze-thaw counting. Default 24
            (i.e. 24 h at hourly sampling).
    """

    def __init__(
        self,
        rain_intensity_threshold_mm_h: float = RAIN_INTENSITY_THRESHOLD_MM_H,
        iv_r2_gate: float = INVERSE_VELOCITY_R2_GATE,
        iv_min_points: int = 5,
        iv_smooth_window: int = 3,
        ramp_rate_ratio: float = 3.0,
        ramp_p_value: float = 0.01,
        moisture_window_samples: int = 24,
    ) -> None:
        """Initialize the instance."""
        self.rain_intensity_threshold_mm_h = rain_intensity_threshold_mm_h
        self.iv_r2_gate = iv_r2_gate
        self.iv_min_points = iv_min_points
        self.iv_smooth_window = iv_smooth_window
        self.ramp_rate_ratio = ramp_rate_ratio
        self.ramp_p_value = ramp_p_value
        self.moisture_window_samples = moisture_window_samples
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Freeze-thaw
    # ------------------------------------------------------------------
    def analyze_freeze_thaw(
        self,
        rock_temperature_c: np.ndarray[Any, Any],
        sample_interval_hours: float = 1.0,
        precipitation_mm: np.ndarray[Any, Any] | None = None,
        surface_wet: np.ndarray[Any, Any] | None = None,
    ) -> FreezeThawResult:
        """Count moisture-effective freeze-thaw cycles in a temperature series.

        A cycle is a downward crossing through 0 °C; it is *effective* only
        when moisture is present at the crossing (Matsuoka & Murton, 2008):
        either ``surface_wet`` is True there, or precipitation fell within
        the preceding ``moisture_window_samples`` samples.

        Args:
            rock_temperature_c: Rock-surface temperature series, °C.
            sample_interval_hours: Sampling interval in hours.
            precipitation_mm: Optional per-sample precipitation, mm.
            surface_wet: Optional per-sample boolean wetness flags.

        Returns:
            FreezeThawResult.

        Raises:
            ValueError: On a series shorter than 2 samples, non-finite
                temperatures, shape mismatches, or when neither moisture
                input is provided (moisture is a physical prerequisite —
                refusing to guess it is the honest failure mode).
        """
        temp = np.asarray(rock_temperature_c, dtype=float)
        if temp.ndim != 1 or temp.size < 2:
            raise ValueError(f"temperature series must be 1-D with >= 2 samples, got {temp.shape}")
        if not np.all(np.isfinite(temp)):
            raise ValueError("temperature series contains non-finite values")
        if sample_interval_hours <= 0:
            raise ValueError(f"sample_interval_hours must be > 0, got {sample_interval_hours}")
        if precipitation_mm is None and surface_wet is None:
            raise ValueError(
                "Freeze-thaw efficacy requires moisture availability (Matsuoka & "
                "Murton 2008); provide precipitation_mm or surface_wet rather than "
                "having the detector assume wetness."
            )

        n = temp.size
        wet = np.zeros(n, dtype=bool)
        if surface_wet is not None:
            w = np.asarray(surface_wet, dtype=bool)
            if w.shape != temp.shape:
                raise ValueError(f"surface_wet shape {w.shape} != temperature shape {temp.shape}")
            wet |= w
        if precipitation_mm is not None:
            p = np.asarray(precipitation_mm, dtype=float)
            if p.shape != temp.shape:
                raise ValueError(
                    f"precipitation_mm shape {p.shape} != temperature shape {temp.shape}"
                )
            if np.any(p < 0) or not np.all(np.isfinite(p)):
                raise ValueError("precipitation_mm must be finite and non-negative")
            rained = p > 0.0
            for i in np.flatnonzero(rained):
                wet[i : i + self.moisture_window_samples + 1] = True

        down_crossings = np.flatnonzero((temp[:-1] > 0.0) & (temp[1:] <= 0.0)) + 1
        effective = int(np.sum(wet[down_crossings]))
        total = int(down_crossings.size)

        lo, hi = FROST_CRACKING_WINDOW_C
        frost_fraction = float(np.mean((temp >= lo) & (temp <= hi)))
        days = n * sample_interval_hours / 24.0

        return FreezeThawResult(
            effective_cycles=effective,
            total_crossings=total,
            cycles_per_day=float(effective / days) if days > 0 else 0.0,
            frost_cracking_fraction=frost_fraction,
            active=effective > 0,
        )

    # ------------------------------------------------------------------
    # Rainfall trigger
    # ------------------------------------------------------------------
    def analyze_rain_trigger(
        self, rain_mm: np.ndarray[Any, Any], sample_interval_hours: float = 1.0
    ) -> dict[str, Any]:
        """Assess the rainfall trigger from an episode precipitation series.

        Uses the mean intensity since the beginning of the *ongoing* rainfall
        episode (trailing run of samples back from the end until a dry
        sample), matching the D'Amato et al. (2016) episode definition.

        Args:
            rain_mm: Per-sample precipitation, mm.
            sample_interval_hours: Sampling interval in hours.

        Returns:
            Dict with mean episode intensity (mm/h), the active flag, and
            the published frequency multiplier when active.

        Raises:
            ValueError: On empty/negative/non-finite input.
        """
        p = np.asarray(rain_mm, dtype=float)
        if p.ndim != 1 or p.size == 0:
            raise ValueError(f"rain series must be 1-D and non-empty, got shape {p.shape}")
        if np.any(p < 0) or not np.all(np.isfinite(p)):
            raise ValueError("rain series must be finite and non-negative")
        if sample_interval_hours <= 0:
            raise ValueError(f"sample_interval_hours must be > 0, got {sample_interval_hours}")

        # Ongoing episode = trailing contiguous wet run.
        wet = p > 0.0
        if not wet[-1]:
            return {
                "episode_ongoing": False,
                "mean_intensity_mm_h": 0.0,
                "trigger_active": False,
                "frequency_multiplier": 1.0,
            }
        start = int(np.flatnonzero(~wet).max() + 1) if np.any(~wet) else 0
        episode = p[start:]
        mean_intensity = float(episode.sum() / (episode.size * sample_interval_hours))
        active = mean_intensity > self.rain_intensity_threshold_mm_h
        return {
            "episode_ongoing": True,
            "mean_intensity_mm_h": mean_intensity,
            "trigger_active": bool(active),
            "frequency_multiplier": HIGH_RAIN_FREQUENCY_MULTIPLIER if active else 1.0,
        }

    # ------------------------------------------------------------------
    # Inverse velocity (Fukuzono 1985)
    # ------------------------------------------------------------------
    def analyze_inverse_velocity(
        self,
        displacement_mm: np.ndarray[Any, Any],
        time: np.ndarray[Any, Any],
    ) -> InverseVelocityResult:
        """Fukuzono (1985) inverse-velocity failure-time forecast.

        Velocity is computed by central differences on the displacement
        series, smoothed with a short moving average (Carlà et al., 2017),
        restricted to the accelerating tail (trailing run of increasing,
        positive velocity), then ``1/v`` is regressed linearly on time.
        The forecast is only emitted when every gate passes (R² >= gate,
        negative slope, enough points, failure time in the future).

        Args:
            displacement_mm: Monotonically measured crack-aperture /
                extensometer displacement series, mm.
            time: Time axis, same length, strictly increasing (any unit;
                the failure time is returned in the same unit).

        Returns:
            InverseVelocityResult — with ``forecast_valid=False`` and a
            populated ``rejection_reason`` when a gate fails.

        Raises:
            ValueError: On shape mismatch, non-finite input, non-increasing
                time, or a series too short to differentiate.
        """
        d = np.asarray(displacement_mm, dtype=float)
        t = np.asarray(time, dtype=float)
        if d.shape != t.shape or d.ndim != 1:
            raise ValueError(f"displacement {d.shape} and time {t.shape} must be equal 1-D shapes")
        if d.size < self.iv_min_points + 2:
            raise ValueError(
                f"need >= {self.iv_min_points + 2} samples for an inverse-velocity fit, "
                f"got {d.size}"
            )
        if not (np.all(np.isfinite(d)) and np.all(np.isfinite(t))):
            raise ValueError("displacement/time contain non-finite values")
        if np.any(np.diff(t) <= 0):
            raise ValueError("time must be strictly increasing")

        # Central-difference velocity at interior points.
        v = (d[2:] - d[:-2]) / (t[2:] - t[:-2])
        tv = t[1:-1]
        # Short moving-average smoothing (Carlà et al. 2017 short-term filter).
        w = max(1, int(self.iv_smooth_window))
        if w > 1 and v.size >= w:
            kernel = np.ones(w) / w
            v = np.convolve(v, kernel, mode="valid")
            tv = tv[w - 1 :]  # right-aligned window

        positive = v > 0.0
        if not positive[-1]:
            return InverseVelocityResult(
                accelerating=False,
                forecast_valid=False,
                rejection_reason="latest velocity is not positive",
            )
        start = int(np.flatnonzero(~positive).max() + 1) if np.any(~positive) else 0
        v_tail = v[start:]
        t_tail = tv[start:]
        # Accelerating tail: trailing run where velocity increases.
        increasing = np.diff(v_tail) > 0.0
        if increasing.size and not np.all(increasing):
            last_break = int(np.flatnonzero(~increasing).max() + 1)
            v_tail = v_tail[last_break:]
            t_tail = t_tail[last_break:]
        if v_tail.size < self.iv_min_points:
            return InverseVelocityResult(
                accelerating=False,
                forecast_valid=False,
                n_points_fit=int(v_tail.size),
                rejection_reason=(
                    f"accelerating tail has {v_tail.size} points " f"(< {self.iv_min_points})"
                ),
            )

        inv_v = 1.0 / v_tail
        fit = stats.linregress(t_tail, inv_v)
        r2 = float(fit.rvalue**2)
        slope = float(fit.slope)

        if slope >= 0.0:
            return InverseVelocityResult(
                accelerating=True,
                forecast_valid=False,
                r_squared=r2,
                slope=slope,
                n_points_fit=int(v_tail.size),
                rejection_reason="1/v is not decreasing (slope >= 0)",
            )
        if r2 < self.iv_r2_gate:
            return InverseVelocityResult(
                accelerating=True,
                forecast_valid=False,
                r_squared=r2,
                slope=slope,
                n_points_fit=int(v_tail.size),
                rejection_reason=f"R^2 {r2:.3f} below gate {self.iv_r2_gate}",
            )

        tf = -float(fit.intercept) / slope
        if tf <= float(t_tail[-1]):
            return InverseVelocityResult(
                accelerating=True,
                forecast_valid=False,
                r_squared=r2,
                slope=slope,
                n_points_fit=int(v_tail.size),
                rejection_reason="extrapolated failure time is not in the future",
            )

        # ±1σ first-order window from the fit standard errors.
        slope_lo, slope_hi = slope - fit.stderr, slope + fit.stderr
        icept_lo = float(fit.intercept) - float(fit.intercept_stderr)
        icept_hi = float(fit.intercept) + float(fit.intercept_stderr)
        candidates = [-b / a for a in (slope_lo, slope_hi) for b in (icept_lo, icept_hi) if a < 0.0]
        window = (min(candidates), max(candidates)) if candidates else (tf, tf)

        return InverseVelocityResult(
            accelerating=True,
            forecast_valid=True,
            failure_time=tf,
            failure_window=(float(window[0]), float(window[1])),
            r_squared=r2,
            slope=slope,
            n_points_fit=int(v_tail.size),
        )

    # ------------------------------------------------------------------
    # Microseismic ramp
    # ------------------------------------------------------------------
    def analyze_microseismic_rate(
        self,
        event_times: np.ndarray[Any, Any],
        observation_window: tuple[float, float],
        recent_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """Detect a precursory event-rate ramp (Amitrano et al., 2005).

        Splits the observation window into a baseline part and a recent
        part (last ``recent_fraction``), compares rates, and tests the
        recent count against the baseline-rate Poisson expectation.

        Args:
            event_times: Microseismic event timestamps (same unit as the
                window bounds).
            observation_window: (start, end) of the monitored interval.
            recent_fraction: Fraction of the window treated as "recent".

        Returns:
            Dict with rates, ratio, Poisson p-value and the ramp flag.

        Raises:
            ValueError: On an empty/invalid window, events outside the
                window, or fewer than 3 baseline events (no meaningful
                baseline rate).
        """
        times = np.sort(np.asarray(event_times, dtype=float))
        start, end = float(observation_window[0]), float(observation_window[1])
        if not end > start:
            raise ValueError(f"observation_window must satisfy end > start, got {start}..{end}")
        if not 0.0 < recent_fraction < 1.0:
            raise ValueError(f"recent_fraction must be in (0, 1), got {recent_fraction}")
        if times.size and (times[0] < start or times[-1] > end):
            raise ValueError("event_times fall outside the observation window")

        split = end - (end - start) * recent_fraction
        n_base = int(np.sum(times < split))
        n_recent = int(np.sum(times >= split))
        if n_base < 3:
            raise ValueError(
                f"baseline window has {n_base} events (< 3); cannot establish a "
                "baseline rate — extend the observation window."
            )
        base_rate = n_base / (split - start)
        recent_duration = end - split
        recent_rate = n_recent / recent_duration
        ratio = recent_rate / base_rate
        expected = base_rate * recent_duration
        # P(N >= n_recent) under the baseline Poisson rate.
        p_value = float(stats.poisson.sf(n_recent - 1, expected))
        ramp = ratio >= self.ramp_rate_ratio and p_value < self.ramp_p_value
        return {
            "baseline_rate": float(base_rate),
            "recent_rate": float(recent_rate),
            "rate_ratio": float(ratio),
            "p_value": p_value,
            "ramp_detected": bool(ramp),
        }

    # ------------------------------------------------------------------
    # Full assessment
    # ------------------------------------------------------------------
    def predict_rockfall(self, data: dict[str, Any]) -> RockfallPredictionResult:
        """Full rockfall hazard assessment over the supplied channels.

        Args:
            data: Channel inputs; every channel is optional but at least one
                must be present:

                - ``rock_temperature_c`` (+ ``sample_interval_hours``,
                  ``precipitation_mm`` / ``surface_wet``): freeze-thaw.
                - ``rain_mm`` (+ ``sample_interval_hours``): rain trigger.
                - ``displacement_mm`` + ``displacement_time``: inverse
                  velocity.
                - ``microseismic_event_times`` + ``observation_window``:
                  event-rate ramp.

        Returns:
            RockfallPredictionResult with hazard level, forecast window
            (when the inverse-velocity gates pass) and the evidence trail.

        Raises:
            ValueError: When no channel input is present, or any channel
                validation fails.
        """
        channels: list[str] = []
        evidence: list[dict[str, Any]] = []

        freeze_thaw: FreezeThawResult | None = None
        if "rock_temperature_c" in data:
            freeze_thaw = self.analyze_freeze_thaw(
                np.asarray(data["rock_temperature_c"], dtype=float),
                sample_interval_hours=float(data.get("sample_interval_hours", 1.0)),
                precipitation_mm=(
                    np.asarray(data["precipitation_mm"], dtype=float)
                    if "precipitation_mm" in data
                    else None
                ),
                surface_wet=(np.asarray(data["surface_wet"]) if "surface_wet" in data else None),
            )
            channels.append("freeze_thaw")
            evidence.append(
                {
                    "criterion": "freeze_thaw_cycles",
                    "effective_cycles": freeze_thaw.effective_cycles,
                    "frost_cracking_fraction": freeze_thaw.frost_cracking_fraction,
                    "citation": "Matsuoka & Murton 2008; Anderson 1998",
                }
            )

        rain: dict[str, Any] | None = None
        if "rain_mm" in data:
            rain = self.analyze_rain_trigger(
                np.asarray(data["rain_mm"], dtype=float),
                sample_interval_hours=float(data.get("sample_interval_hours", 1.0)),
            )
            channels.append("rain_trigger")
            evidence.append(
                {
                    "criterion": "rain_intensity",
                    "mean_intensity_mm_h": rain["mean_intensity_mm_h"],
                    "trigger_active": rain["trigger_active"],
                    "citation": "D'Amato et al. 2016 (5 mm/h episode threshold)",
                }
            )

        iv: InverseVelocityResult | None = None
        if "displacement_mm" in data or "displacement_time" in data:
            if not ("displacement_mm" in data and "displacement_time" in data):
                raise ValueError(
                    "inverse-velocity channel needs both displacement_mm and " "displacement_time"
                )
            iv = self.analyze_inverse_velocity(
                np.asarray(data["displacement_mm"], dtype=float),
                np.asarray(data["displacement_time"], dtype=float),
            )
            channels.append("inverse_velocity")
            evidence.append(
                {
                    "criterion": "inverse_velocity",
                    "forecast_valid": iv.forecast_valid,
                    "r_squared": iv.r_squared,
                    "failure_time": iv.failure_time,
                    "citation": "Fukuzono 1985; Carla et al. 2017 (R^2 gate)",
                }
            )

        ms: dict[str, Any] | None = None
        if "microseismic_event_times" in data:
            if "observation_window" not in data:
                raise ValueError("microseismic channel needs observation_window=(start, end)")
            ms = self.analyze_microseismic_rate(
                np.asarray(data["microseismic_event_times"], dtype=float),
                tuple(data["observation_window"]),
            )
            channels.append("microseismic")
            evidence.append(
                {
                    "criterion": "microseismic_rate_ramp",
                    "rate_ratio": ms["rate_ratio"],
                    "p_value": ms["p_value"],
                    "ramp_detected": ms["ramp_detected"],
                    "citation": "Amitrano et al. 2005",
                }
            )

        if not channels:
            raise ValueError(
                "predict_rockfall received no channel input; supply at least one of "
                "rock_temperature_c, rain_mm, displacement_mm(+time), "
                "microseismic_event_times(+observation_window)"
            )

        multiplier = 1.0
        if freeze_thaw is not None and freeze_thaw.active:
            multiplier = max(multiplier, FREEZE_THAW_FREQUENCY_MULTIPLIER)
        if rain is not None and rain["trigger_active"]:
            multiplier = max(multiplier, float(rain["frequency_multiplier"]))

        hazard = self._hazard_level(freeze_thaw, rain, iv, ms)
        confidence = self._evidence_confidence(freeze_thaw, rain, iv, ms)

        return RockfallPredictionResult(
            hazard_level=hazard.value,
            confidence=confidence,
            freeze_thaw=freeze_thaw,
            rain_trigger_active=None if rain is None else bool(rain["trigger_active"]),
            rain_mean_intensity_mm_h=(None if rain is None else float(rain["mean_intensity_mm_h"])),
            estimated_frequency_multiplier=float(multiplier),
            inverse_velocity=iv,
            microseismic_ramp=None if ms is None else bool(ms["ramp_detected"]),
            microseismic_rate_ratio=None if ms is None else float(ms["rate_ratio"]),
            channels_assessed=channels,
            evidence=evidence,
            warnings=self._warnings_for(hazard, iv),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _hazard_level(
        freeze_thaw: FreezeThawResult | None,
        rain: dict[str, Any] | None,
        iv: InverseVelocityResult | None,
        ms: dict[str, Any] | None,
    ) -> RockfallHazardLevel:
        """Documented hazard mapping (first match wins).

        CRITICAL: valid inverse-velocity failure forecast; or a
            microseismic ramp together with any active meteorological
            trigger.
        HIGH: microseismic ramp alone; or accelerating displacement that
            failed only the fit gates; or both meteorological triggers
            active together.
        MODERATE: exactly one active meteorological trigger.
        LOW: otherwise.

        Args:
            freeze_thaw: Freeze-thaw result or None.
            rain: Rain-trigger result or None.
            iv: Inverse-velocity result or None.
            ms: Microseismic result or None.

        Returns:
            RockfallHazardLevel.
        """
        ft_active = freeze_thaw is not None and freeze_thaw.active
        rain_active = rain is not None and bool(rain["trigger_active"])
        ramp = ms is not None and bool(ms["ramp_detected"])
        iv_valid = iv is not None and iv.forecast_valid
        iv_accel = iv is not None and iv.accelerating and not iv.forecast_valid

        if iv_valid or (ramp and (ft_active or rain_active)):
            return RockfallHazardLevel.CRITICAL
        if ramp or iv_accel or (ft_active and rain_active):
            return RockfallHazardLevel.HIGH
        if ft_active or rain_active:
            return RockfallHazardLevel.MODERATE
        return RockfallHazardLevel.LOW

    @staticmethod
    def _evidence_confidence(
        freeze_thaw: FreezeThawResult | None,
        rain: dict[str, Any] | None,
        iv: InverseVelocityResult | None,
        ms: dict[str, Any] | None,
    ) -> float:
        """Deterministic score: mean activation over the assessed channels.

        Args:
            freeze_thaw: Freeze-thaw result or None.
            rain: Rain result or None.
            iv: Inverse-velocity result or None.
            ms: Microseismic result or None.

        Returns:
            Confidence in [0, 1]; unassessed channels are excluded rather
            than imputed.
        """
        lines: list[float] = []
        if freeze_thaw is not None:
            lines.append(1.0 if freeze_thaw.active else 0.0)
        if rain is not None:
            lines.append(1.0 if rain["trigger_active"] else 0.0)
        if iv is not None:
            lines.append(1.0 if iv.forecast_valid else 0.5 if iv.accelerating else 0.0)
        if ms is not None:
            lines.append(1.0 if ms["ramp_detected"] else 0.0)
        return float(round(sum(lines) / len(lines), 6)) if lines else 0.0

    @staticmethod
    def _warnings_for(hazard: RockfallHazardLevel, iv: InverseVelocityResult | None) -> list[str]:
        """Operational warning strings."""
        warnings: list[str] = []
        if hazard is RockfallHazardLevel.CRITICAL:
            warnings.append("ROCKFALL CRITICAL: close exposed corridors immediately")
            if iv is not None and iv.forecast_valid and iv.failure_window is not None:
                warnings.append(
                    "Inverse-velocity failure window: "
                    f"{iv.failure_window[0]:.2f} to {iv.failure_window[1]:.2f} "
                    "(input time units)"
                )
        elif hazard is RockfallHazardLevel.HIGH:
            warnings.append("ROCKFALL HIGH: intensify monitoring, restrict access")
        elif hazard is RockfallHazardLevel.MODERATE:
            warnings.append("Rockfall advisory: meteorological trigger active")
        return warnings
