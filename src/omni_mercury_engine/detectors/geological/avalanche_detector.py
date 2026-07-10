# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Snow Avalanche Detector — snow-stability physics for dry-slab hazard.

Split out of :mod:`omni_mercury_engine.detectors.geological.landslide`, which
keeps its soil-mechanics scope: snowpack failure is governed by weak-layer
shear strength, skier/overburden stress and snow metamorphism, none of which
are soil quantities.  Everything here is deterministic, literature-anchored
snow physics; there is no neural network and nothing is inferred from
unavailable data.

Physics implemented:

- **Skier stability index SK38** (Föhn, 1987; refined by Jamieson &
  Johnston, 1998): ``SK38 = tau_strength / (tau_xz + delta_tau_xz)`` with all
  stresses normalised to a 38° slope.  The slab shear stress is
  ``tau_xz = rho_bar * g * h * sin(psi) * cos(psi)`` and the skier-induced
  shear stress follows Föhn's inclined-half-space line-load solution
  ``delta_tau_xz = 2 R cos(a) sin^2(a) sin(a + psi) / (pi h_eff cos(psi))``
  maximised numerically over the load angle ``a`` (the analytic maximiser is
  ~54.3° at psi = 38°, giving ~0.152 kPa per metre of slab for the standard
  R = 500 N/m line load — an 85 kg skier on 1.7 m skis).  Ski penetration
  reduces the effective slab depth (Jamieson & Johnston, 1998; the
  ``0.8 * 43.3 / rho`` adaptation follows the operational SNOWPACK
  implementation).  SK38 < 1 indicates likely skier triggering; 1-1.5 is
  transitional; > 1.5 is generally stable (Jamieson & Johnston, 1998).
- **Weak-layer shear strength**: a measured shear-frame strength (Pa) is
  preferred.  For *persistent* weak-layer grain types (facets, depth hoar)
  only, the density power law of Jamieson & Johnston (2001),
  ``sigma = 18.5 kPa * (rho / 917)^2.11``, is available as a documented
  fallback (same parameterisation used by SNOWPACK).  For other grain types
  a measured strength is required — the detector fails loudly rather than
  applying an out-of-scope regression.
- **Critical new-snow loading** (Schweizer, Jamieson & Schneebeli, 2003,
  Rev. Geophys. 41, 1016): ~30-50 cm of new snow in 24 h is the classical
  critical range for natural dry-snow avalanches; 10-20 cm can suffice for
  skier triggering under unfavourable conditions.  Wind drift multiplies the
  effective lee-slope loading — deposition during drifting commonly reaches
  several times the snowfall rate — so a documented, deliberately
  conservative 2x multiplier is applied when the 10 m wind exceeds the
  ~5 m/s transport threshold for fresh loose snow (Li & Pomeroy, 1997,
  report thresholds from ~4-11 m/s with mean ~7.7 m/s; 5 m/s is the fresh
  dry-snow lower bound).
- **Temperature-gradient metamorphism**: a sustained temperature gradient
  greater than ~10 K/m drives kinetic-growth (faceting) metamorphism and
  persistent-weak-layer formation (Akitaya, 1974; Colbeck, 1983; reviewed in
  Schweizer et al., 2003).
- **Rain-on-snow destabilisation**: rain falling on a snowpack causes rapid,
  short-lived strength loss and avalanching within hours (Conway & Raymond,
  1993, J. Glaciol. 39, 635-642).

The output danger level is mapped onto the 5-level EAWS European Avalanche
Danger Scale (1-Low, 2-Moderate, 3-Considerable, 4-High, 5-Very High; EAWS,
2023).  The EAWS scale is defined qualitatively; the mapping table used here
is an explicit, documented engineering rule set anchored to the SK38 classes
and loading flags above — it does not claim to be the official EAWS matrix.

References:
    - Föhn, P.M.B. (1987). The stability index and various triggering
      mechanisms. IAHS Publ. 162, 195-214.
    - Jamieson, J.B., Johnston, C.D. (1998). Refinements to the stability
      index for skier-triggered dry-slab avalanches. Annals of Glaciology
      26, 296-302.
    - Jamieson, J.B., Johnston, C.D. (2001). Evaluation of the shear frame
      test for weak snowpack layers. Annals of Glaciology 32, 59-69.
    - Schweizer, J., Jamieson, J.B., Schneebeli, M. (2003). Snow avalanche
      formation. Reviews of Geophysics 41(4), 1016.
    - Li, L., Pomeroy, J.W. (1997). Estimates of threshold wind speeds for
      snow transport using meteorological data. J. Appl. Meteor. 36,
      205-213.
    - Conway, H., Raymond, C.F. (1993). Snow stability during rain.
      Journal of Glaciology 39(133), 635-642.
    - EAWS (2023). European Avalanche Danger Scale.
      https://www.avalanches.org/standards/avalanche-danger-scale/
    - Monti, F., Gaume, J., van Herwijnen, A., Schweizer, J. (2016). Snow
      instability evaluation: calculating the skier-induced stress in a
      multi-layered snowpack. NHESS 16, 775-788. (Eq. 1 is the Föhn
      line-load solution implemented here.)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)

#: Gravitational acceleration (m/s^2).
G = 9.81

#: Density of ice (kg/m^3), reference for the strength power law.
RHO_ICE = 917.0

#: Standard skier line load (N/m): 85 kg skier on 1.7 m skis (Föhn 1987).
SKIER_LINE_LOAD_N_M = 500.0

#: Persistent weak-layer grain types for which the Jamieson & Johnston (2001)
#: density power law applies (SNOWPACK default branch: FC, DH, FCxr).
PERSISTENT_GRAIN_TYPES = frozenset({"facets", "depth_hoar", "rounding_facets"})

#: Classical critical new-snow depth for natural dry-snow avalanches, cm/24h
#: (Schweizer et al. 2003).
CRITICAL_NEW_SNOW_24H_CM = 30.0

#: New snow that can suffice for skier triggering under unfavourable
#: conditions, cm/24h (Schweizer et al. 2003).
SKIER_CRITICAL_NEW_SNOW_24H_CM = 10.0

#: 10 m wind speed above which fresh loose dry snow drifts (Li & Pomeroy 1997
#: lower bound for fresh snow).
WIND_DRIFT_THRESHOLD_MS = 5.0

#: Conservative lee-slope loading multiplier during active drift (documented
#: lower bound of the multi-fold deposition amplification, Schweizer et al.
#: 2003).
WIND_SLAB_MULTIPLIER = 2.0

#: Kinetic-growth (faceting) temperature-gradient threshold, K/m
#: (Akitaya 1974; Colbeck 1983; Schweizer et al. 2003).
TG_FACETING_THRESHOLD_K_M = 10.0


class AvalancheDangerLevel(Enum):
    """EAWS 5-level avalanche danger scale (EAWS, 2023)."""

    LOW = 1
    MODERATE = 2
    CONSIDERABLE = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class SnowLayer:
    """One snowpack layer, ordered from the surface downward.

    Attributes:
        thickness_m: Slope-normal layer thickness in metres.
        density_kg_m3: Layer density in kg/m^3.
        temperature_c: Layer mid-point temperature in °C.
        grain_type: Optional grain type tag. Recognised persistent types:
            ``facets``, ``depth_hoar``, ``rounding_facets``; anything else is
            treated as non-persistent.
    """

    thickness_m: float
    density_kg_m3: float
    temperature_c: float
    grain_type: str | None = None

    def __post_init__(self) -> None:
        """Validate physical ranges (fail loud on nonsense input)."""
        if not np.isfinite(self.thickness_m) or self.thickness_m <= 0:
            raise ValueError(f"layer thickness must be positive, got {self.thickness_m}")
        if not np.isfinite(self.density_kg_m3) or not 10.0 <= self.density_kg_m3 <= RHO_ICE:
            raise ValueError(
                f"layer density must be in [10, {RHO_ICE}] kg/m^3, got {self.density_kg_m3}"
            )
        if not np.isfinite(self.temperature_c) or not -60.0 <= self.temperature_c <= 0.5:
            raise ValueError(f"snow temperature must be in [-60, 0.5] °C, got {self.temperature_c}")


@dataclass
class SK38Result:
    """Skier stability index computation record.

    Attributes:
        sk38: The stability index value.
        stability_class: ``poor`` (< 1), ``fair`` (1-1.5) or ``good``
            (> 1.5) per Jamieson & Johnston (1998).
        tau_xz_pa: Slab shear stress on the weak layer at 38°, Pa.
        delta_tau_skier_pa: Skier-induced shear stress at 38°, Pa.
        weak_layer_strength_pa: Shear strength used, Pa.
        strength_source: ``measured`` or ``jamieson_johnston_2001``.
        slab_depth_m: Slope-normal slab depth above the weak layer.
        penetration_m: Ski penetration depth applied.
        alpha_max_deg: Load angle that maximises the skier stress.
    """

    sk38: float
    stability_class: str
    tau_xz_pa: float
    delta_tau_skier_pa: float
    weak_layer_strength_pa: float
    strength_source: str
    slab_depth_m: float
    penetration_m: float
    alpha_max_deg: float


@dataclass
class AvalanchePredictionResult:
    """Avalanche hazard assessment.

    Attributes:
        danger_level: EAWS danger level 1-5.
        danger_level_name: EAWS level name (low ... very_high).
        avalanche_likely: True at danger level >= 3 (considerable), the
            level at which most fatal accidents occur (EAWS statistics).
        confidence: Deterministic evidence score in [0, 1].
        sk38: SK38 computation record (None when no snow profile given).
        new_snow_loading_flag: Effective HN24 >= critical threshold.
        effective_new_snow_24h_cm: Wind-adjusted 24 h new-snow loading.
        wind_slab_flag: Wind drift multiplier was active.
        faceting_risk_flag: Sustained TG above the kinetic-growth threshold.
        max_temperature_gradient_k_m: Largest inter-layer |dT/dz|.
        rain_on_snow_flag: Rain fell on the snowpack.
        evidence: Ordered evidence trail (criterion, value, citation).
        warnings: Operational warning strings.
    """

    danger_level: int
    danger_level_name: str
    avalanche_likely: bool
    confidence: float
    sk38: SK38Result | None = None
    new_snow_loading_flag: bool = False
    effective_new_snow_24h_cm: float = 0.0
    wind_slab_flag: bool = False
    faceting_risk_flag: bool = False
    max_temperature_gradient_k_m: float = 0.0
    rain_on_snow_flag: bool = False
    evidence: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class NewSnowLoadingAssessment(TypedDict):
    """Return contract of :meth:`AvalancheDetector.assess_new_snow_loading`."""

    effective_new_snow_24h_cm: float
    wind_slab_active: bool
    multiplier: float
    critical: bool
    skier_critical: bool


class TemperatureGradientAssessment(TypedDict):
    """Return contract of :meth:`AvalancheDetector.assess_temperature_gradient`."""

    max_gradient_k_m: float
    exceeds_threshold: bool
    faceting_risk: bool


class AvalancheDetector:
    """Dry-slab avalanche hazard detector built on real snow-stability physics.

    Works fully untrained: every output derives from the documented
    formulations in the module docstring.

    Args:
        skier_line_load_n_m: Skier line load R (N/m). Default 500 (Föhn 1987).
        critical_new_snow_cm: Critical HN24 for natural release (cm).
            Default 30 (Schweizer et al. 2003).
        wind_drift_threshold_ms: 10 m wind speed enabling drift (m/s).
            Default 5 (Li & Pomeroy 1997, fresh snow lower bound).
        wind_slab_multiplier: Effective-loading multiplier during drift.
            Default 2.0 (conservative lower bound, Schweizer et al. 2003).
        tg_threshold_k_m: Faceting TG threshold (K/m). Default 10.
        tg_persistence_days: Days the gradient must persist for the faceting
            flag when a duration is supplied. Default 3.
    """

    def __init__(
        self,
        skier_line_load_n_m: float = SKIER_LINE_LOAD_N_M,
        critical_new_snow_cm: float = CRITICAL_NEW_SNOW_24H_CM,
        wind_drift_threshold_ms: float = WIND_DRIFT_THRESHOLD_MS,
        wind_slab_multiplier: float = WIND_SLAB_MULTIPLIER,
        tg_threshold_k_m: float = TG_FACETING_THRESHOLD_K_M,
        tg_persistence_days: float = 3.0,
    ) -> None:
        """Initialize the instance."""
        self.skier_line_load_n_m = skier_line_load_n_m
        self.critical_new_snow_cm = critical_new_snow_cm
        self.wind_drift_threshold_ms = wind_drift_threshold_ms
        self.wind_slab_multiplier = wind_slab_multiplier
        self.tg_threshold_k_m = tg_threshold_k_m
        self.tg_persistence_days = tg_persistence_days
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # SK38 core
    # ------------------------------------------------------------------
    def compute_sk38(
        self,
        layers: list[SnowLayer],
        weak_layer_index: int,
        slope_angle_deg: float = 38.0,
        measured_strength_pa: float | None = None,
        ski_penetration_m: float | None = None,
    ) -> SK38Result:
        """Compute the skier stability index normalised to a 38° slope.

        Args:
            layers: Snowpack layers ordered from the surface downward.
            weak_layer_index: Index of the weak layer in ``layers``; the slab
                is every layer above it. Must have at least one slab layer.
            slope_angle_deg: Actual slope angle (only used for validation
                context; stresses are evaluated at the 38° reference per
                Jamieson & Johnston 1998).
            measured_strength_pa: Shear-frame strength of the weak layer, Pa.
                Preferred whenever available.
            ski_penetration_m: Measured ski penetration. When None it is
                estimated as ``0.8 * 43.3 / rho_slab`` (Jamieson & Johnston
                1998, as adapted in SNOWPACK), capped at 90% of slab depth.

        Returns:
            SK38Result.

        Raises:
            ValueError: On an empty slab, an out-of-range weak-layer index,
                an unphysical slope angle, or a non-persistent weak layer
                without a measured strength.
        """
        if not layers:
            raise ValueError("layers must not be empty")
        if not 1 <= weak_layer_index < len(layers):
            raise ValueError(
                f"weak_layer_index must leave at least one slab layer above it "
                f"(got {weak_layer_index} with {len(layers)} layers)"
            )
        if not 0.0 < slope_angle_deg < 90.0:
            raise ValueError(f"slope_angle_deg must be in (0, 90), got {slope_angle_deg}")

        slab = layers[:weak_layer_index]
        weak = layers[weak_layer_index]
        h = float(sum(sl.thickness_m for sl in slab))
        rho_bar = float(sum(sl.thickness_m * sl.density_kg_m3 for sl in slab) / h)

        psi = np.deg2rad(38.0)  # J&J 1998: index is defined at the 38° reference
        tau_xz = rho_bar * G * h * np.sin(psi) * np.cos(psi)

        if ski_penetration_m is None:
            penetration = 0.8 * 43.3 / rho_bar  # J&J 1998 / SNOWPACK adaptation
        else:
            if ski_penetration_m < 0:
                raise ValueError(f"ski_penetration_m must be >= 0, got {ski_penetration_m}")
            penetration = ski_penetration_m
        penetration = min(penetration, 0.9 * h)
        h_eff = h - penetration

        delta_tau, alpha_max_deg = self._foehn_skier_stress(h_eff, psi)

        if measured_strength_pa is not None:
            if not np.isfinite(measured_strength_pa) or measured_strength_pa <= 0:
                raise ValueError(
                    f"measured_strength_pa must be positive, got {measured_strength_pa}"
                )
            strength = float(measured_strength_pa)
            strength_source = "measured"
        else:
            if weak.grain_type not in PERSISTENT_GRAIN_TYPES:
                raise ValueError(
                    "Weak-layer shear strength unavailable: the Jamieson & Johnston "
                    "(2001) density power law is validated for persistent grain types "
                    f"{sorted(PERSISTENT_GRAIN_TYPES)} only; weak layer grain_type="
                    f"{weak.grain_type!r}. Provide measured_strength_pa instead."
                )
            # Jamieson & Johnston (2001): sigma = 18.5 kPa * (rho/rho_ice)^2.11
            strength = 18.5e3 * (weak.density_kg_m3 / RHO_ICE) ** 2.11
            strength_source = "jamieson_johnston_2001"

        sk38 = float(strength / (tau_xz + delta_tau))
        if sk38 < 1.0:
            stability_class = "poor"
        elif sk38 <= 1.5:
            stability_class = "fair"
        else:
            stability_class = "good"

        return SK38Result(
            sk38=sk38,
            stability_class=stability_class,
            tau_xz_pa=float(tau_xz),
            delta_tau_skier_pa=float(delta_tau),
            weak_layer_strength_pa=strength,
            strength_source=strength_source,
            slab_depth_m=h,
            penetration_m=float(penetration),
            alpha_max_deg=alpha_max_deg,
        )

    def _foehn_skier_stress(self, h_eff_m: float, psi_rad: float) -> tuple[float, float]:
        """Föhn (1987) line-load skier shear stress, maximised over angle.

        ``delta_tau = 2 R cos(a) sin^2(a) sin(a + psi) / (pi h cos(psi))``
        (Monti et al., 2016, Eq. 1), maximised numerically over the load
        angle a in (0°, 90°).  At psi = 38° the maximiser is ~54.3° and the
        stress is ~0.152 kPa per metre of slab for R = 500 N/m (matches the
        operational SNOWPACK reference value).

        Args:
            h_eff_m: Effective slab depth (after ski penetration), m.
            psi_rad: Slope angle in radians (38° reference).

        Returns:
            Tuple of (maximum skier shear stress in Pa, maximising angle in
            degrees).

        Raises:
            ValueError: If the effective depth is not positive.
        """
        if h_eff_m <= 0:
            raise ValueError(f"effective slab depth must be positive, got {h_eff_m}")
        alphas = np.deg2rad(np.linspace(0.5, 89.5, 1781))  # 0.05° resolution
        r = self.skier_line_load_n_m
        stresses = (
            2.0
            * r
            * np.cos(alphas)
            * np.sin(alphas) ** 2
            * np.sin(alphas + psi_rad)
            / (np.pi * h_eff_m * np.cos(psi_rad))
        )
        i = int(np.argmax(stresses))
        return float(stresses[i]), float(np.rad2deg(alphas[i]))

    # ------------------------------------------------------------------
    # Weather-driven factors
    # ------------------------------------------------------------------
    def assess_new_snow_loading(
        self, new_snow_24h_cm: float, wind_speed_10m_ms: float = 0.0
    ) -> NewSnowLoadingAssessment:
        """Assess new-snow loading with the wind-slab multiplier.

        Args:
            new_snow_24h_cm: Measured 24 h new-snow depth (HN24), cm.
            wind_speed_10m_ms: 10 m wind speed, m/s.

        Returns:
            Dict with effective loading, flags and the applied multiplier.

        Raises:
            ValueError: On negative or non-finite inputs.
        """
        if not np.isfinite(new_snow_24h_cm) or new_snow_24h_cm < 0:
            raise ValueError(f"new_snow_24h_cm must be >= 0, got {new_snow_24h_cm}")
        if not np.isfinite(wind_speed_10m_ms) or wind_speed_10m_ms < 0:
            raise ValueError(f"wind_speed_10m_ms must be >= 0, got {wind_speed_10m_ms}")

        wind_active = wind_speed_10m_ms >= self.wind_drift_threshold_ms
        multiplier = self.wind_slab_multiplier if wind_active else 1.0
        effective = new_snow_24h_cm * multiplier
        return {
            "effective_new_snow_24h_cm": float(effective),
            "wind_slab_active": bool(wind_active),
            "multiplier": float(multiplier),
            "critical": bool(effective >= self.critical_new_snow_cm),
            "skier_critical": bool(effective >= SKIER_CRITICAL_NEW_SNOW_24H_CM),
        }

    def assess_temperature_gradient(
        self, layers: list[SnowLayer], gradient_duration_days: float | None = None
    ) -> TemperatureGradientAssessment:
        """Assess kinetic-growth metamorphism risk from the layer temperatures.

        The gradient between adjacent layer mid-points is
        ``|dT| / dz`` with dz the distance between mid-points.

        Args:
            layers: Snowpack layers from the surface downward (>= 2).
            gradient_duration_days: How long the present gradient regime has
                persisted, if known. When provided, the faceting flag also
                requires ``>= tg_persistence_days``.

        Returns:
            Dict with the max gradient, exceedance flag and faceting risk.

        Raises:
            ValueError: With fewer than two layers.
        """
        if len(layers) < 2:
            raise ValueError("temperature-gradient assessment needs >= 2 layers")
        gradients = []
        for upper, lower in itertools.pairwise(layers):
            dz = (upper.thickness_m + lower.thickness_m) / 2.0
            gradients.append(abs(upper.temperature_c - lower.temperature_c) / dz)
        max_tg = float(max(gradients))
        exceeds = max_tg > self.tg_threshold_k_m
        if gradient_duration_days is None:
            persistent = exceeds  # duration unknown: flag on exceedance alone
        else:
            persistent = exceeds and gradient_duration_days >= self.tg_persistence_days
        return {
            "max_gradient_k_m": max_tg,
            "exceeds_threshold": bool(exceeds),
            "faceting_risk": bool(persistent),
        }

    # ------------------------------------------------------------------
    # Full assessment
    # ------------------------------------------------------------------
    def predict_avalanche(
        self,
        layers: list[SnowLayer] | None = None,
        weak_layer_index: int | None = None,
        slope_angle_deg: float = 38.0,
        measured_strength_pa: float | None = None,
        ski_penetration_m: float | None = None,
        new_snow_24h_cm: float = 0.0,
        wind_speed_10m_ms: float = 0.0,
        rain_mm_24h: float = 0.0,
        air_temperature_c: float | None = None,
        gradient_duration_days: float | None = None,
    ) -> AvalanchePredictionResult:
        """Full avalanche hazard assessment mapped to the EAWS scale.

        Args:
            layers: Snowpack layers, surface downward (optional; without a
                profile only the weather factors are assessed).
            weak_layer_index: Weak-layer index for SK38 (required with
                ``layers``).
            slope_angle_deg: Slope angle in degrees.
            measured_strength_pa: Measured weak-layer shear strength, Pa.
            ski_penetration_m: Measured ski penetration, m.
            new_snow_24h_cm: 24 h new-snow depth, cm.
            wind_speed_10m_ms: 10 m wind speed, m/s.
            rain_mm_24h: 24 h liquid rain on the snowpack, mm.
            air_temperature_c: Air temperature, °C (used to distinguish rain
                from snow when rain is reported: rain requires > 0 °C).
            gradient_duration_days: Persistence of the current TG regime.

        Returns:
            AvalanchePredictionResult with the EAWS danger level and the full
            evidence trail.

        Raises:
            ValueError: If ``layers`` is given without ``weak_layer_index``,
                or any component validation fails.
        """
        evidence: list[dict[str, object]] = []
        sk38_result: SK38Result | None = None

        if layers is not None:
            if weak_layer_index is None:
                raise ValueError("weak_layer_index is required when layers are provided")
            sk38_result = self.compute_sk38(
                layers,
                weak_layer_index,
                slope_angle_deg=slope_angle_deg,
                measured_strength_pa=measured_strength_pa,
                ski_penetration_m=ski_penetration_m,
            )
            evidence.append(
                {
                    "criterion": "sk38",
                    "value": sk38_result.sk38,
                    "stability_class": sk38_result.stability_class,
                    "citation": "Foehn 1987; Jamieson & Johnston 1998",
                }
            )

        loading = self.assess_new_snow_loading(new_snow_24h_cm, wind_speed_10m_ms)
        evidence.append(
            {
                "criterion": "new_snow_loading",
                "value": loading["effective_new_snow_24h_cm"],
                "critical": loading["critical"],
                "wind_slab_active": loading["wind_slab_active"],
                "citation": "Schweizer et al. 2003; Li & Pomeroy 1997",
            }
        )

        faceting: TemperatureGradientAssessment = {
            "max_gradient_k_m": 0.0,
            "exceeds_threshold": False,
            "faceting_risk": False,
        }
        if layers is not None and len(layers) >= 2:
            faceting = self.assess_temperature_gradient(layers, gradient_duration_days)
            evidence.append(
                {
                    "criterion": "temperature_gradient",
                    "value": faceting["max_gradient_k_m"],
                    "faceting_risk": faceting["faceting_risk"],
                    "citation": "Akitaya 1974; Colbeck 1983; Schweizer et al. 2003",
                }
            )

        if rain_mm_24h < 0 or not np.isfinite(rain_mm_24h):
            raise ValueError(f"rain_mm_24h must be >= 0, got {rain_mm_24h}")
        rain_on_snow = rain_mm_24h > 0.0 and (air_temperature_c is None or air_temperature_c > 0.0)
        if rain_mm_24h > 0.0:
            evidence.append(
                {
                    "criterion": "rain_on_snow",
                    "value": rain_mm_24h,
                    "active": rain_on_snow,
                    "citation": "Conway & Raymond 1993",
                }
            )

        danger = self._map_to_eaws(
            sk38_result, loading, bool(faceting["faceting_risk"]), rain_on_snow
        )
        confidence = self._evidence_confidence(sk38_result, loading, faceting, rain_on_snow)

        return AvalanchePredictionResult(
            danger_level=danger.value,
            danger_level_name=danger.name.lower(),
            avalanche_likely=danger.value >= AvalancheDangerLevel.CONSIDERABLE.value,
            confidence=confidence,
            sk38=sk38_result,
            new_snow_loading_flag=loading["critical"],
            effective_new_snow_24h_cm=loading["effective_new_snow_24h_cm"],
            wind_slab_flag=loading["wind_slab_active"],
            faceting_risk_flag=faceting["faceting_risk"],
            max_temperature_gradient_k_m=faceting["max_gradient_k_m"],
            rain_on_snow_flag=rain_on_snow,
            evidence=evidence,
            warnings=self._warnings_for(danger),
        )

    # ------------------------------------------------------------------
    # Mapping + scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _map_to_eaws(
        sk38: SK38Result | None,
        loading: NewSnowLoadingAssessment,
        faceting_risk: bool,
        rain_on_snow: bool,
    ) -> AvalancheDangerLevel:
        """Documented engineering mapping onto the EAWS 5-level scale.

        Decision table (first match wins):
            5 VERY_HIGH:    SK38 poor AND (critical loading OR rain-on-snow)
            4 HIGH:         SK38 poor AND any loading/facet/rain flag;
                            or critical loading AND rain-on-snow
            3 CONSIDERABLE: SK38 poor alone; or SK38 fair with any flag;
                            or critical loading; or rain-on-snow
            2 MODERATE:     SK38 fair alone; or faceting risk; or
                            skier-critical loading
            1 LOW:          otherwise

        Args:
            sk38: SK38 record or None.
            loading: Output of :meth:`assess_new_snow_loading`.
            faceting_risk: Persistent-TG faceting flag.
            rain_on_snow: Rain-on-snow flag.

        Returns:
            AvalancheDangerLevel.
        """
        poor = sk38 is not None and sk38.stability_class == "poor"
        fair = sk38 is not None and sk38.stability_class == "fair"
        critical = bool(loading["critical"])
        skier_critical = bool(loading["skier_critical"])
        any_flag = critical or skier_critical or faceting_risk or rain_on_snow

        if poor and (critical or rain_on_snow):
            return AvalancheDangerLevel.VERY_HIGH
        if (poor and any_flag) or (critical and rain_on_snow):
            return AvalancheDangerLevel.HIGH
        if poor or (fair and any_flag) or critical or rain_on_snow:
            return AvalancheDangerLevel.CONSIDERABLE
        if fair or faceting_risk or skier_critical:
            return AvalancheDangerLevel.MODERATE
        return AvalancheDangerLevel.LOW

    @staticmethod
    def _evidence_confidence(
        sk38: SK38Result | None,
        loading: NewSnowLoadingAssessment,
        faceting: TemperatureGradientAssessment,
        rain_on_snow: bool,
    ) -> float:
        """Deterministic evidence score: fraction of independent lines fired.

        Args:
            sk38: SK38 record or None.
            loading: New-snow loading assessment.
            faceting: TG assessment.
            rain_on_snow: Rain flag.

        Returns:
            Confidence in [0, 1]; assessments that could not run (no
            profile) contribute zero weight rather than fabricated signal.
        """
        lines: list[float] = []
        if sk38 is not None:
            lines.append({"poor": 1.0, "fair": 0.5, "good": 0.0}[sk38.stability_class])
        lines.append(1.0 if loading["critical"] else 0.5 if loading["skier_critical"] else 0.0)
        lines.append(1.0 if faceting.get("faceting_risk") else 0.0)
        lines.append(1.0 if rain_on_snow else 0.0)
        return float(round(sum(lines) / len(lines), 6))

    @staticmethod
    def _warnings_for(danger: AvalancheDangerLevel) -> list[str]:
        """Operational warning strings per danger level."""
        if danger is AvalancheDangerLevel.VERY_HIGH:
            return [
                "AVALANCHE DANGER 5 (VERY HIGH): avoid all avalanche terrain",
                "Large spontaneous avalanches expected; consider evacuations of exposed areas",
            ]
        if danger is AvalancheDangerLevel.HIGH:
            return [
                "AVALANCHE DANGER 4 (HIGH): travel in avalanche terrain not recommended",
                "Natural avalanches likely; remote triggering possible",
            ]
        if danger is AvalancheDangerLevel.CONSIDERABLE:
            return [
                "AVALANCHE DANGER 3 (CONSIDERABLE): careful route selection required",
                "Human triggering probable on steep slopes",
            ]
        if danger is AvalancheDangerLevel.MODERATE:
            return ["AVALANCHE DANGER 2 (MODERATE): evaluate snowpack on steep slopes"]
        return ["AVALANCHE DANGER 1 (LOW): generally safe conditions"]
