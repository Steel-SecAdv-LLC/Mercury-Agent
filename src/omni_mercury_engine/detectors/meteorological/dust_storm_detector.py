# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dust-storm detector (physics core, works untrained).

Implements the literature-standard sand-and-dust-storm formulations:

* **Visibility-based event classification** per the WMO sand-and-dust-storm
  (SDS) present-weather definitions (WMO Manual on Codes, WMO-No. 306, and
  the WMO SDS-WAS Science and Implementation Plan): a *dust storm /
  sandstorm* is dust raised by strong winds reducing visibility below
  1000 m; a *severe dust storm* reduces visibility below 200 m; *blowing
  dust* covers visibility 1-10 km.  Classification requires a concurrent
  wind observation at or above a documented dust-raising threshold
  (default 10 m/s at 10 m, the typical mid-latitude dust-raising wind in
  the SDS literature) so that low fog/haze visibility alone is never
  mislabelled as a dust event.
* **Friction-velocity emission-potential model**: friction velocity u*
  from the neutral logarithmic wind profile
  ``u* = kappa * U(z) / ln(z / z0)`` (kappa = 0.4), compared against the
  threshold friction velocity.  The dry-particle threshold follows
  Shao & Lu (2000): "A simple expression for wind erosion threshold
  friction velocity", *J. Geophys. Res.*, 105(D17), 22437-22443
  (``u*t = sqrt(A_N * (sigma_p * g * d + gamma / (rho_a * d)))`` with
  A_N = 0.0123, gamma = 3.0e-4 kg s^-2), which places the minimum
  threshold near d = 75-100 um at u*t ~= 0.2 m/s.  The soil-moisture
  correction follows Fecan, Marticorena & Bergametti (1999): "Parametrization
  of the increase of the aeolian erosion threshold wind friction velocity
  due to soil moisture for arid and semi-arid areas", *Ann. Geophysicae*,
  17, 149-157: below the residual moisture ``w' = 0.0014*clay^2 + 0.17*clay``
  (gravimetric %, clay in %) the threshold is unchanged; above it,
  ``u*t_wet / u*t_dry = sqrt(1 + 1.21 * (w - w')**0.68)``.
* **Haboob gust-front signature** in a pressure/temperature/wind time
  series: haboobs are dust storms raised by convective-outflow density
  currents; the documented gust-front passage signature (Idso, Ingram &
  Pritchard, 1972: "An American Haboob", *Bull. Amer. Meteor. Soc.*, 53,
  930-935, describing the Phoenix haboob climatology) combines an abrupt
  pressure jump, a sharp temperature drop, a wind-direction shift, and a
  wind surge within tens of minutes.  The default window (60 min) and
  magnitude cuts (pressure rise >= 1.0 hPa, temperature drop >= 3.0 deg C,
  wind shift >= 30 deg, wind surge >= 5 m/s) are this module's documented
  operationalization of the signature magnitudes reported there.
* **NWS dust alert wiring** (Dust Storm Warning, Blowing Dust Advisory,
  Dust Advisory) via the shared CAP helpers.

No neural network; missing or non-finite inputs raise instead of being
silently defaulted.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.detectors.meteorological.severe_storm_alerts import (
    filter_alerts_by_event,
    normalize_alert_records,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DustEventClass",
    "DustStormDetector",
    "EmissionPotential",
    "HaboobSignature",
]

# --- WMO SDS visibility classes ------------------------------------------------
_SEVERE_DUST_STORM_VIS_M = 200.0
_DUST_STORM_VIS_M = 1000.0
_BLOWING_DUST_VIS_M = 10_000.0

#: Default dust-raising 10 m wind threshold (m/s) for the visibility classes.
_DEFAULT_WIND_THRESHOLD_MS = 10.0

# --- Friction-velocity model -----------------------------------------------------
_VON_KARMAN = 0.4
_SHAO_LU_AN = 0.0123
_SHAO_LU_GAMMA = 3.0e-4  # kg s^-2
_GRAVITY = 9.80665  # m s^-2

# --- Haboob gust-front signature defaults -----------------------------------------
_HABOOB_WINDOW_S = 3600.0
_HABOOB_PRESSURE_JUMP_HPA = 1.0
_HABOOB_TEMP_DROP_C = 3.0
_HABOOB_WIND_SHIFT_DEG = 30.0
_HABOOB_WIND_SURGE_MS = 5.0

_DUST_ALERT_EVENTS = ("Dust Storm Warning", "Blowing Dust Advisory", "Dust Advisory")


class DustEventClass:
    """Canonical WMO SDS visibility-class labels."""

    SEVERE_DUST_STORM = "severe_dust_storm"
    DUST_STORM = "dust_storm"
    BLOWING_DUST = "blowing_dust"
    NONE = "none"


@dataclass(frozen=True)
class EmissionPotential:
    """Friction-velocity emission-potential result.

    Attributes:
        friction_velocity_ms: u* from the neutral log profile.
        threshold_dry_ms: Dry threshold friction velocity u*t.
        threshold_wet_ms: Moisture-corrected threshold (Fecan et al. 1999).
        moisture_ratio: u*t_wet / u*t_dry applied.
        residual_moisture_pct: Fecan w' for the supplied clay fraction.
        emission_favorable: True when u* >= u*t_wet.
        excess_ratio: u* / u*t_wet (dimensionless emission-potential score).
    """

    friction_velocity_ms: float
    threshold_dry_ms: float
    threshold_wet_ms: float
    moisture_ratio: float
    residual_moisture_pct: float
    emission_favorable: bool
    excess_ratio: float


@dataclass
class HaboobSignature:
    """Haboob gust-front detection result.

    Attributes:
        detected: True when all four signature components co-occur within
            the window.
        onset_index: Sample index at which the qualifying window starts,
            or None.
        pressure_jump_hpa: Pressure rise across the best window.
        temp_drop_c: Temperature drop across the best window.
        wind_shift_deg: Wind-direction change across the best window.
        wind_surge_ms: Wind-speed increase across the best window.
        notes: Derivation notes.
    """

    detected: bool
    onset_index: int | None
    pressure_jump_hpa: float
    temp_drop_c: float
    wind_shift_deg: float
    wind_surge_ms: float
    notes: list[str] = field(default_factory=list)


def _require_finite(name: str, value: Any) -> float:
    """Coerce to float; raise ``ValueError`` on missing/non-finite input."""
    if value is None:
        raise ValueError(f"Required input '{name}' is missing (None).")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Input '{name}'={value!r} is not numeric.") from exc
    if not math.isfinite(out):
        raise ValueError(f"Input '{name}'={out} is not finite.")
    return out


def _series(name: str, values: Any, n: int | None = None) -> np.ndarray[Any, Any]:
    """Validate a 1-D finite series; optionally enforce length ``n``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Series '{name}' must be a non-empty 1-D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Series '{name}' contains non-finite values.")
    if n is not None and arr.size != n:
        raise ValueError(f"Series '{name}' length {arr.size} != expected {n}.")
    return arr


class DustStormDetector:
    """Dust-storm detector built on WMO / Fecan / Shao-Lu formulations.

    Deterministic physics core -- no training, no neural network.  See the
    module docstring for citations.

    Example:
        >>> detector = DustStormDetector()
        >>> detector.classify_visibility(visibility_m=500.0, wind_speed_ms=14.0)
        'dust_storm'
    """

    #: Fixed fusion feature dimension (see :meth:`extract_features`).
    FEATURE_DIM = 16

    def __init__(self, wind_threshold_ms: float = _DEFAULT_WIND_THRESHOLD_MS) -> None:
        """Initialize the detector.

        Args:
            wind_threshold_ms: Dust-raising 10 m wind threshold used by the
                visibility classifier (default 10 m/s; typical mid-latitude
                dust-raising wind in the SDS literature).
        """
        if wind_threshold_ms <= 0:
            raise ValueError("wind_threshold_ms must be positive.")
        self.wind_threshold_ms = float(wind_threshold_ms)

    # ------------------------------------------------------------------
    # WMO visibility classification
    # ------------------------------------------------------------------

    def classify_visibility(self, visibility_m: float, wind_speed_ms: float) -> str:
        """Classify a single observation into WMO SDS visibility classes.

        Requires a concurrent dust-raising wind: with wind below the
        threshold, reduced visibility alone is not attributed to dust
        (fog / haze produce identical visibility signatures), and the class
        is ``none``.

        Args:
            visibility_m: Horizontal visibility (m), >= 0.
            wind_speed_ms: 10 m wind speed (m/s), >= 0.

        Returns:
            One of :class:`DustEventClass` (``severe_dust_storm`` < 200 m,
            ``dust_storm`` < 1000 m, ``blowing_dust`` < 10 km, else
            ``none``; all per the WMO SDS definitions).

        Raises:
            ValueError: On missing/non-finite/negative inputs.
        """
        vis = _require_finite("visibility_m", visibility_m)
        wind = _require_finite("wind_speed_ms", wind_speed_ms)
        if vis < 0 or wind < 0:
            raise ValueError("visibility_m and wind_speed_ms must be >= 0.")

        if wind < self.wind_threshold_ms:
            return DustEventClass.NONE
        if vis < _SEVERE_DUST_STORM_VIS_M:
            return DustEventClass.SEVERE_DUST_STORM
        if vis < _DUST_STORM_VIS_M:
            return DustEventClass.DUST_STORM
        if vis < _BLOWING_DUST_VIS_M:
            return DustEventClass.BLOWING_DUST
        return DustEventClass.NONE

    def classify_series(
        self,
        visibility_m: Any,
        wind_speed_ms: Any,
    ) -> list[str]:
        """Classify an aligned observation series (see :meth:`classify_visibility`).

        Args:
            visibility_m: 1-D visibility series (m).
            wind_speed_ms: 1-D wind series (m/s), same length.

        Returns:
            Per-sample class labels.

        Raises:
            ValueError: On misaligned or invalid series.
        """
        vis = _series("visibility_m", visibility_m)
        wind = _series("wind_speed_ms", wind_speed_ms, n=vis.size)
        return [self.classify_visibility(v, w) for v, w in zip(vis, wind)]

    # ------------------------------------------------------------------
    # Friction-velocity emission potential
    # ------------------------------------------------------------------

    @staticmethod
    def threshold_friction_velocity_dry(
        particle_diameter_m: float,
        particle_density_kg_m3: float = 2650.0,
        air_density_kg_m3: float = 1.23,
    ) -> float:
        """Dry threshold friction velocity per Shao & Lu (2000).

        ``u*t = sqrt(A_N * ((rho_p / rho_a) * g * d + gamma / (rho_a * d)))``
        with A_N = 0.0123 and gamma = 3.0e-4 kg s^-2 (Shao & Lu 2000,
        Eq. 24).  The curve has its documented minimum (~0.2 m/s) near
        d = 75-100 um for quartz sand in air.

        Args:
            particle_diameter_m: Particle diameter (m), > 0.
            particle_density_kg_m3: Particle density (default 2650, quartz).
            air_density_kg_m3: Air density (default 1.23 kg/m^3).

        Returns:
            u*t in m/s.

        Raises:
            ValueError: On non-positive/non-finite inputs.
        """
        d = _require_finite("particle_diameter_m", particle_diameter_m)
        rho_p = _require_finite("particle_density_kg_m3", particle_density_kg_m3)
        rho_a = _require_finite("air_density_kg_m3", air_density_kg_m3)
        if d <= 0 or rho_p <= 0 or rho_a <= 0:
            raise ValueError("Diameter and densities must be > 0.")
        sigma_p = rho_p / rho_a
        return math.sqrt(_SHAO_LU_AN * (sigma_p * _GRAVITY * d + _SHAO_LU_GAMMA / (rho_a * d)))

    @staticmethod
    def fecan_moisture_ratio(soil_moisture_pct: float, clay_pct: float) -> tuple[float, float]:
        """Fecan et al. (1999) soil-moisture threshold correction.

        ``w' = 0.0014 * clay^2 + 0.17 * clay`` (gravimetric %, clay in %);
        for ``w <= w'`` the ratio is 1; above,
        ``u*t_wet / u*t_dry = sqrt(1 + 1.21 * (w - w')**0.68)``.

        Args:
            soil_moisture_pct: Gravimetric soil moisture w (%), >= 0.
            clay_pct: Soil clay content (%), 0-100.

        Returns:
            Tuple ``(ratio, w_prime_pct)``.

        Raises:
            ValueError: On invalid inputs.
        """
        w = _require_finite("soil_moisture_pct", soil_moisture_pct)
        clay = _require_finite("clay_pct", clay_pct)
        if w < 0:
            raise ValueError("soil_moisture_pct must be >= 0.")
        if not 0.0 <= clay <= 100.0:
            raise ValueError("clay_pct must be within [0, 100].")
        w_prime = 0.0014 * clay**2 + 0.17 * clay
        if w <= w_prime:
            return 1.0, w_prime
        return math.sqrt(1.0 + 1.21 * (w - w_prime) ** 0.68), w_prime

    def emission_potential(
        self,
        wind_speed_ms: Any = None,
        measurement_height_m: Any = None,
        roughness_length_m: Any = None,
        soil_moisture_pct: Any = None,
        clay_pct: Any = None,
        particle_diameter_m: float = 100e-6,
        particle_density_kg_m3: float = 2650.0,
        air_density_kg_m3: float = 1.23,
    ) -> EmissionPotential:
        """Dust-emission potential from wind, roughness, and soil moisture.

        u* is derived from the neutral logarithmic wind profile
        (``u* = kappa * U(z) / ln(z / z0)``, neutral stability assumed and
        documented); the threshold combines Shao & Lu (2000) dry particles
        with the Fecan et al. (1999) moisture correction.  Emission is
        favorable when ``u* >= u*t_wet``.

        Args:
            wind_speed_ms: Wind speed U(z) (m/s), required, >= 0.
            measurement_height_m: Anemometer height z (m), required, > z0.
            roughness_length_m: Aerodynamic roughness z0 (m), required, > 0.
            soil_moisture_pct: Gravimetric soil moisture (%), required.
            clay_pct: Soil clay content (%), required.
            particle_diameter_m: Erodible-particle diameter (default 100 um,
                near the Shao-Lu threshold minimum).
            particle_density_kg_m3: Particle density (default quartz).
            air_density_kg_m3: Air density (default 1.23 kg/m^3).

        Returns:
            :class:`EmissionPotential`.

        Raises:
            ValueError: When any required input is missing, non-finite, or
                unphysical (fail-loud: no default atmosphere or soil state
                is fabricated).
        """
        u = _require_finite("wind_speed_ms", wind_speed_ms)
        z = _require_finite("measurement_height_m", measurement_height_m)
        z0 = _require_finite("roughness_length_m", roughness_length_m)
        if u < 0:
            raise ValueError("wind_speed_ms must be >= 0.")
        if z0 <= 0:
            raise ValueError("roughness_length_m must be > 0.")
        if z <= z0:
            raise ValueError(
                f"measurement_height_m={z} must exceed roughness_length_m={z0} "
                "for the logarithmic profile to be defined."
            )

        ustar = _VON_KARMAN * u / math.log(z / z0)
        ut_dry = self.threshold_friction_velocity_dry(
            particle_diameter_m, particle_density_kg_m3, air_density_kg_m3
        )
        ratio, w_prime = self.fecan_moisture_ratio(soil_moisture_pct, clay_pct)
        ut_wet = ut_dry * ratio

        return EmissionPotential(
            friction_velocity_ms=float(ustar),
            threshold_dry_ms=float(ut_dry),
            threshold_wet_ms=float(ut_wet),
            moisture_ratio=float(ratio),
            residual_moisture_pct=float(w_prime),
            emission_favorable=bool(ustar >= ut_wet),
            excess_ratio=float(ustar / ut_wet) if ut_wet > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Haboob gust-front signature
    # ------------------------------------------------------------------

    def detect_haboob_signature(
        self,
        times_s: Any,
        pressure_hpa: Any,
        temperature_c: Any,
        wind_speed_ms: Any,
        wind_direction_deg: Any,
        window_s: float = _HABOOB_WINDOW_S,
    ) -> HaboobSignature:
        """Detect a haboob gust-front passage in an observation series.

        For every sample ``i``, examines the window ``[t_i, t_i + window_s]``
        and requires all four components (thresholds per the module
        docstring; Idso et al. 1972 signature structure):

        * pressure rise >= 1.0 hPa (max in window minus value at onset),
        * temperature drop >= 3.0 deg C (value at onset minus min in window),
        * wind-direction shift >= 30 deg (circular difference between onset
          direction and direction at the in-window wind maximum),
        * wind surge >= 5.0 m/s (max in window minus value at onset).

        Args:
            times_s: Strictly increasing epoch times (s).
            pressure_hpa: Station pressure series (hPa).
            temperature_c: Air-temperature series (deg C).
            wind_speed_ms: Wind-speed series (m/s).
            wind_direction_deg: Wind-direction series (deg, meteorological).
            window_s: Search window (s), default 3600.

        Returns:
            :class:`HaboobSignature` for the best (first fully qualifying,
            else strongest-pressure-jump) window.

        Raises:
            ValueError: On misaligned, non-finite, or non-monotonic series.
        """
        t = _series("times_s", times_s)
        p = _series("pressure_hpa", pressure_hpa, n=t.size)
        temp = _series("temperature_c", temperature_c, n=t.size)
        wspd = _series("wind_speed_ms", wind_speed_ms, n=t.size)
        wdir = _series("wind_direction_deg", wind_direction_deg, n=t.size)
        if t.size < 3:
            raise ValueError("Need >= 3 samples to resolve a gust-front passage.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times_s must be strictly increasing.")
        if float(window_s) <= 0:
            raise ValueError("window_s must be positive.")

        best: tuple[float, int, float, float, float, float] | None = None
        detected_at: int | None = None

        for i in range(t.size - 1):
            j_end = int(np.searchsorted(t, t[i] + float(window_s), side="right"))
            if j_end - i < 2:
                continue
            sl = slice(i, j_end)

            p_jump = float(np.max(p[sl]) - p[i])
            temp_drop = float(temp[i] - np.min(temp[sl]))
            k_gust = int(np.argmax(wspd[sl])) + i
            surge = float(wspd[k_gust] - wspd[i])
            shift = abs((float(wdir[k_gust]) - float(wdir[i]) + 180.0) % 360.0 - 180.0)

            qualifies = (
                p_jump >= _HABOOB_PRESSURE_JUMP_HPA
                and temp_drop >= _HABOOB_TEMP_DROP_C
                and shift >= _HABOOB_WIND_SHIFT_DEG
                and surge >= _HABOOB_WIND_SURGE_MS
            )
            if qualifies and detected_at is None:
                detected_at = i
                best = (p_jump, i, p_jump, temp_drop, shift, surge)
                break
            if best is None or p_jump > best[0]:
                best = (p_jump, i, p_jump, temp_drop, shift, surge)

        if best is None:
            raise ValueError(
                f"window_s={window_s} is shorter than the observation spacing; "
                "no window contains two samples, so no gust front can be resolved."
            )
        _, onset, p_jump, temp_drop, shift, surge = best
        detected = detected_at is not None
        notes = [
            f"window={float(window_s) / 60.0:.0f} min; thresholds: dP>="
            f"{_HABOOB_PRESSURE_JUMP_HPA} hPa, dT<=-{_HABOOB_TEMP_DROP_C} C, "
            f"shift>={_HABOOB_WIND_SHIFT_DEG} deg, surge>={_HABOOB_WIND_SURGE_MS} m/s",
            f"best window at index {onset}: dP={p_jump:.2f} hPa, dT=-{temp_drop:.2f} C, "
            f"shift={shift:.0f} deg, surge={surge:.1f} m/s",
        ]
        return HaboobSignature(
            detected=detected,
            onset_index=onset if detected else None,
            pressure_jump_hpa=p_jump,
            temp_drop_c=temp_drop,
            wind_shift_deg=shift,
            wind_surge_ms=surge,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # NWS wiring
    # ------------------------------------------------------------------

    def cross_check_nws_alerts(self, alerts: Any) -> dict[str, Any]:
        """Cross-check against active NWS dust products.

        Args:
            alerts: Alert payload in any shape accepted by
                :func:`normalize_alert_records`.

        Returns:
            Dict with ``n_dust_alerts``, ``events`` (sorted distinct CAP
            event names), and ``dust_storm_warned`` (Dust Storm Warning
            present).

        Raises:
            TypeError: If the payload shape is unrecognized.
        """
        records = normalize_alert_records(alerts)
        dust = filter_alerts_by_event(records, _DUST_ALERT_EVENTS)
        events = sorted({str(r.get("event")) for r in dust})
        return {
            "n_dust_alerts": len(dust),
            "events": events,
            "dust_storm_warned": "Dust Storm Warning" in events,
        }

    # ------------------------------------------------------------------
    # Fusion interface
    # ------------------------------------------------------------------

    def extract_features(self, data: Any) -> torch.Tensor:
        """Extract a fixed-width feature vector for the fusion registry.

        Dict input runs the real physics paths present in the dict
        (visibility/wind classification; emission-potential inputs).
        Array input yields documented robust summary statistics only.

        Args:
            data: Input dict or numeric array.

        Returns:
            ``torch.Tensor`` of shape ``(FEATURE_DIM,)``.

        Raises:
            ValueError: When a dict is supplied without any recognized
                physics input group.
        """
        features: list[float] = []
        if isinstance(data, dict):
            used = False
            if "visibility_m" in data and "wind_speed_ms" in data:
                cls = self.classify_visibility(data["visibility_m"], data["wind_speed_ms"])
                order = [
                    DustEventClass.NONE,
                    DustEventClass.BLOWING_DUST,
                    DustEventClass.DUST_STORM,
                    DustEventClass.SEVERE_DUST_STORM,
                ]
                features.append(float(order.index(cls)))
                features.append(float(data["visibility_m"]) / 10_000.0)
                features.append(float(data["wind_speed_ms"]) / 30.0)
                used = True
            if all(
                k in data
                for k in (
                    "wind_speed_ms",
                    "measurement_height_m",
                    "roughness_length_m",
                    "soil_moisture_pct",
                    "clay_pct",
                )
            ):
                ep = self.emission_potential(
                    wind_speed_ms=data["wind_speed_ms"],
                    measurement_height_m=data["measurement_height_m"],
                    roughness_length_m=data["roughness_length_m"],
                    soil_moisture_pct=data["soil_moisture_pct"],
                    clay_pct=data["clay_pct"],
                )
                features.extend(
                    [
                        ep.friction_velocity_ms,
                        ep.threshold_wet_ms,
                        ep.excess_ratio,
                        1.0 if ep.emission_favorable else 0.0,
                    ]
                )
                used = True
            if not used:
                raise ValueError(
                    "Dict input carries no recognized dust-storm physics inputs "
                    "(need visibility/wind and/or emission-potential inputs)."
                )
        else:
            arr = np.asarray(
                data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data,
                dtype=np.float64,
            ).ravel()
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                raise ValueError("extract_features requires a non-empty, finite numeric array.")
            q25, q75 = np.percentile(arr, [25, 75])
            features = [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.median(arr)),
                float(q75 - q25),
            ]

        features = features[: self.FEATURE_DIM]
        features.extend([0.0] * (self.FEATURE_DIM - len(features)))
        return torch.tensor(features, dtype=torch.float32)
