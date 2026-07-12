# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wildfire → Debris-Flow Cascade — the USGS post-fire debris-flow model family.

Implements the published-coefficient models used operationally by the USGS
for emergency post-fire debris-flow hazard assessment. These are transparent
"trained" models: every coefficient below is a published regression value
with provenance, embedded verbatim — nothing is fitted, tuned or invented
here.

Models implemented:

- **Staley et al. (2017) M1 likelihood model** (logistic regression trained
  on 1,550 records from western-US burned areas):

  ``X = β + C1·(T·R) + C2·(F·R) + C3·(S·R)``,  ``p = 1 / (1 + e^-X)``

  where ``T`` is the proportion of upslope area with moderate-to-high burn
  severity AND gradient >= 23°, ``F`` is mean dNBR / 1000, ``S`` is the
  soil KF-factor, and ``R`` is the peak rainfall **accumulation in mm**
  over the model duration (15/30/60 min). Coefficients per duration are
  from USGS OFR 2016-1106, Table 1 (= Staley et al. 2017, Geomorphology
  278, Table 2) — see ``M1_COEFFICIENTS``. The rainfall
  intensity-duration threshold at likelihood p follows by inversion:
  ``R_p = (ln(p/(1-p)) - β) / (C1·T + C2·F + C3·S)``.
- **Cannon et al. (2008) intensity-duration thresholds** for recently
  burned areas: southwestern Colorado ``I = 9.5·D^-0.7`` and southern
  California ``I = 12.5·D^-0.4`` (I in mm/h, D in hours).
- **Gartner et al. (2014) emergency-assessment volume model** (92 post-fire
  debris-flow volumes, southern California Transverse Ranges):

  ``ln V = 4.22 + 0.39·sqrt(i15) + 0.36·ln(Bmh) + 0.13·sqrt(R)``

  with V in m³, ``i15`` the peak 15-min intensity (mm/h), ``Bmh`` the
  watershed area burned at moderate/high severity (km²), and ``R`` the
  watershed relief (m); residual standard error 1.04 in ln-space. Volume
  is *omitted transparently* when ``Bmh``/relief inputs are unavailable — the
  cascade never guesses them.

Staged composition (each stage requires real evidence):

1. ``BURN_EVIDENCE`` — quantitative burn-severity inputs (T, dNBR, KF).
   A :class:`~omni_mercury_engine.detectors.geological.wildfire.
   WildfirePredictionResult` (or equivalent dict) may be attached as
   context, but the M1 stage runs on the quantitative severity inputs only:
   the WildfireDetector's CNN confidence comes from an untrained network
   and must not feed a published regression.
2. ``RAINFALL`` — a rain gauge series is reduced to peak 15/30/60-min
   accumulations by rolling sums (backwards-differencing style of Kean et
   al. 2011 as used by Staley et al. 2017).
3. ``ASSESSMENT`` — M1 likelihood at the peak accumulation, threshold
   intensities, Cannon I-D exceedance for the storm, and the Gartner
   volume + USGS volume class when its inputs exist.

References:
    - Staley, D.M., Negri, J.A., Kean, J.W., Laber, J.L., Tillery, A.C.,
      Youberg, A.M. (2016). Updated logistic regression equations for the
      calculation of post-fire debris-flow likelihood in the western United
      States. USGS Open-File Report 2016-1106. (Table 1: M1 coefficients.)
    - Staley, D.M., Negri, J.A., Kean, J.W., Laber, J.L., Tillery, A.C.,
      Youberg, A.M. (2017). Prediction of spatially explicit rainfall
      intensity-duration thresholds for post-fire debris-flow generation
      in the western United States. Geomorphology 278, 149-162.
    - Cannon, S.H., Gartner, J.E., Wilson, R.C., Bowers, J.C., Laber, J.L.
      (2008). Storm rainfall conditions for floods and debris flows from
      recently burned areas in southwestern Colorado and southern
      California. Geomorphology 96, 250-269.
    - Gartner, J.E., Cannon, S.H., Santi, P.M. (2014). Empirical models for
      predicting volumes of sediment deposited by debris flows and
      sediment-laden floods in the transverse ranges of southern
      California. Engineering Geology 176, 45-56.
    - Kean, J.W., Staley, D.M., Cannon, S.H. (2011). In situ measurements
      of post-fire debris flows in southern California. JGR 116, F04019.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Staley et al. (2017) M1 logistic-regression coefficients by rainfall
#: duration (minutes). Provenance: USGS Open-File Report 2016-1106, Table 1
#: ("Model M1 ... intercept (β) and coefficient (C1, C2, C3) values ... for
#: rainfall durations of 15, 30, and 60 minutes"); identical values appear
#: in Staley et al. (2017), Geomorphology 278, Table 2. The rainfall term R
#: is the accumulation in millimetres over the duration (OFR 2016-1106,
#: Methods: "differing parameter values ... for rainfall accumulations (in
#: millimeters) measured over the analyzed duration").
M1_COEFFICIENTS: dict[int, dict[str, float]] = {
    15: {"beta": -3.63, "c1": 0.41, "c2": 0.67, "c3": 0.70},
    30: {"beta": -3.61, "c1": 0.26, "c2": 0.39, "c3": 0.50},
    60: {"beta": -3.21, "c1": 0.17, "c2": 0.20, "c3": 0.22},
}

#: Cannon et al. (2008) intensity-duration thresholds for recently burned
#: areas: I = a * D^b with I in mm/h and D in hours. Provenance: Cannon,
#: Gartner, Wilson, Bowers & Laber (2008), Geomorphology 96, 250-269
#: (southwestern Colorado: I = 9.5 D^-0.7; southern California:
#: I = 12.5 D^-0.4).
CANNON_ID_THRESHOLDS: dict[str, tuple[float, float]] = {
    "colorado": (9.5, -0.7),
    "southern_california": (12.5, -0.4),
}

#: Gartner et al. (2014) volume model coefficients:
#: ln V = 4.22 + 0.39 sqrt(i15) + 0.36 ln(Bmh) + 0.13 sqrt(relief).
#: Provenance: Gartner, Cannon & Santi (2014), Engineering Geology 176,
#: 45-56 (emergency-assessment model; ln-space standard error 1.04).
GARTNER_2014 = {"intercept": 4.22, "i15": 0.39, "bmh": 0.36, "relief": 0.13, "se_ln": 1.04}

#: USGS emergency-assessment volume classes (m³) used to bin the Gartner
#: model output in the published hazard maps.
VOLUME_CLASS_BOUNDS_M3: tuple[float, float, float] = (1e3, 1e4, 1e5)

#: Documented likelihood bins (quartiles of p). Engineering choice for the
#: combined output — deliberately plain quartiles, not a claimed USGS class
#: scheme.
LIKELIHOOD_BINS: tuple[float, float, float] = (0.25, 0.5, 0.75)

#: Maximum USDA soil-erodibility KF-factor (dimensionless as used in M1).
MAX_KF_FACTOR = 0.64


def staley_m1_likelihood(
    t_proportion: float,
    dnbr_mean: float,
    kf_factor: float,
    rain_accum_mm: float,
    duration_min: int = 15,
) -> float:
    """Staley et al. (2017) M1 debris-flow likelihood.

    Args:
        t_proportion: Proportion (0-1) of upslope area with moderate-to-high
            burn severity and gradient >= 23°.
        dnbr_mean: Mean dNBR of the upslope area (raw dNBR units; the model
            term F = dnbr_mean / 1000).
        kf_factor: Mean soil KF-factor of the upslope area.
        rain_accum_mm: Rainfall accumulation (mm) over ``duration_min``.
        duration_min: Model duration: 15, 30 or 60 minutes.

    Returns:
        Likelihood p in (0, 1).

    Raises:
        ValueError: On out-of-range inputs or an unsupported duration.
    """
    coef = _m1_coefficients(duration_min)
    _validate_m1_terrain(t_proportion, dnbr_mean, kf_factor)
    if not np.isfinite(rain_accum_mm) or rain_accum_mm < 0.0:
        raise ValueError(f"rain_accum_mm must be >= 0, got {rain_accum_mm}")

    f_term = dnbr_mean / 1000.0
    x = coef["beta"] + rain_accum_mm * (
        coef["c1"] * t_proportion + coef["c2"] * f_term + coef["c3"] * kf_factor
    )
    return float(1.0 / (1.0 + math.exp(-x)))


def staley_m1_threshold_intensity(
    t_proportion: float,
    dnbr_mean: float,
    kf_factor: float,
    likelihood: float = 0.5,
    duration_min: int = 15,
) -> float:
    """Rainfall intensity (mm/h) at which M1 reaches the given likelihood.

    Inverts the M1 link function (Staley et al. 2017, Eq. for R_p):
    ``R_p = (ln(p/(1-p)) - β) / (C1·T + C2·F + C3·S)`` (accumulation, mm),
    converted to intensity by ``* 60 / duration``.

    Args:
        t_proportion: See :func:`staley_m1_likelihood`.
        dnbr_mean: See :func:`staley_m1_likelihood`.
        kf_factor: See :func:`staley_m1_likelihood`.
        likelihood: Target likelihood p in (0, 1). Default 0.5, the
            threshold definition used in the USGS assessments.
        duration_min: Model duration: 15, 30 or 60 minutes.

    Returns:
        Threshold intensity in mm/h.

    Raises:
        ValueError: On out-of-range inputs, or when the terrain terms are
            all zero (unburned basin — the model does not apply; Staley et
            al. 2017 restrict it to recently burned areas).
    """
    coef = _m1_coefficients(duration_min)
    _validate_m1_terrain(t_proportion, dnbr_mean, kf_factor)
    if not 0.0 < likelihood < 1.0:
        raise ValueError(f"likelihood must be in (0, 1), got {likelihood}")

    denom = coef["c1"] * t_proportion + coef["c2"] * (dnbr_mean / 1000.0) + coef["c3"] * kf_factor
    if denom <= 0.0:
        raise ValueError(
            "M1 terrain terms are all zero/non-positive (unburned basin?); the "
            "model applies to recently burned areas only (Staley et al. 2017)."
        )
    accum_mm = (math.log(likelihood / (1.0 - likelihood)) - coef["beta"]) / denom
    return float(accum_mm * 60.0 / duration_min)


def cannon_id_threshold_mm_h(duration_h: float, region: str = "southern_california") -> float:
    """Cannon et al. (2008) burned-area I-D threshold intensity.

    Args:
        duration_h: Storm duration in hours (> 0).
        region: ``southern_california`` (I = 12.5 D^-0.4) or ``colorado``
            (I = 9.5 D^-0.7).

    Returns:
        Threshold mean intensity in mm/h.

    Raises:
        ValueError: On non-positive duration or unknown region.
    """
    if duration_h <= 0.0 or not np.isfinite(duration_h):
        raise ValueError(f"duration_h must be > 0, got {duration_h}")
    if region not in CANNON_ID_THRESHOLDS:
        raise ValueError(
            f"unknown region {region!r}; expected one of {sorted(CANNON_ID_THRESHOLDS)}"
        )
    a, b = CANNON_ID_THRESHOLDS[region]
    return float(a * duration_h**b)


def gartner_2014_volume_m3(
    i15_mm_h: float, burned_mh_km2: float, relief_m: float
) -> tuple[float, int]:
    """Gartner et al. (2014) debris-flow volume estimate + USGS class.

    ``ln V = 4.22 + 0.39 sqrt(i15) + 0.36 ln(Bmh) + 0.13 sqrt(relief)``

    Args:
        i15_mm_h: Peak 15-minute rainfall intensity, mm/h (>= 0).
        burned_mh_km2: Watershed area burned at moderate/high severity, km²
            (> 0; ln of zero is undefined and an unburned watershed is out
            of model scope).
        relief_m: Watershed relief, m (>= 0).

    Returns:
        Tuple of (volume in m³, volume class 1-4 per the USGS emergency
        assessment bins: 1 < 10³, 2 in 10³-10⁴, 3 in 10⁴-10⁵, 4 >= 10⁵).

    Raises:
        ValueError: On out-of-range inputs.
    """
    if not np.isfinite(i15_mm_h) or i15_mm_h < 0.0:
        raise ValueError(f"i15_mm_h must be >= 0, got {i15_mm_h}")
    if not np.isfinite(burned_mh_km2) or burned_mh_km2 <= 0.0:
        raise ValueError(
            f"burned_mh_km2 must be > 0 (model scope: burned watersheds), got {burned_mh_km2}"
        )
    if not np.isfinite(relief_m) or relief_m < 0.0:
        raise ValueError(f"relief_m must be >= 0, got {relief_m}")

    ln_v = (
        GARTNER_2014["intercept"]
        + GARTNER_2014["i15"] * math.sqrt(i15_mm_h)
        + GARTNER_2014["bmh"] * math.log(burned_mh_km2)
        + GARTNER_2014["relief"] * math.sqrt(relief_m)
    )
    volume = float(math.exp(ln_v))
    b1, b2, b3 = VOLUME_CLASS_BOUNDS_M3
    volume_class = 1 if volume < b1 else 2 if volume < b2 else 3 if volume < b3 else 4
    return volume, volume_class


def _m1_coefficients(duration_min: int) -> dict[str, float]:
    """Fetch M1 coefficients for a supported duration (fail loud)."""
    if duration_min not in M1_COEFFICIENTS:
        raise ValueError(
            f"M1 is published for durations {sorted(M1_COEFFICIENTS)} min only, "
            f"got {duration_min}"
        )
    return M1_COEFFICIENTS[duration_min]


def _validate_m1_terrain(t_proportion: float, dnbr_mean: float, kf_factor: float) -> None:
    """Validate the M1 terrain/severity terms (fail loud on nonsense)."""
    if not np.isfinite(t_proportion) or not 0.0 <= t_proportion <= 1.0:
        raise ValueError(f"t_proportion must be in [0, 1], got {t_proportion}")
    if not np.isfinite(dnbr_mean) or not -500.0 <= dnbr_mean <= 1400.0:
        raise ValueError(f"dnbr_mean must be a plausible raw dNBR in [-500, 1400], got {dnbr_mean}")
    if not np.isfinite(kf_factor) or not 0.0 <= kf_factor <= MAX_KF_FACTOR:
        raise ValueError(f"kf_factor must be in [0, {MAX_KF_FACTOR}], got {kf_factor}")


@dataclass
class RainfallAnalysis:
    """Peak rolling accumulations extracted from a rain series.

    Attributes:
        peak_accum_mm: Peak accumulation per duration (minutes -> mm).
        peak_intensity_mm_h: Peak intensity per duration (minutes -> mm/h).
        storm_duration_h: Wet-spell duration of the series (first to last
            wet sample), hours.
        storm_mean_intensity_mm_h: Mean intensity over the wet spell.
    """

    peak_accum_mm: dict[int, float]
    peak_intensity_mm_h: dict[int, float]
    storm_duration_h: float
    storm_mean_intensity_mm_h: float


@dataclass
class FireDebrisFlowResult:
    """Full cascade assessment.

    Attributes:
        likelihood: M1 likelihood at the peak accumulation for the primary
            duration.
        likelihood_class: Quartile bin label (documented in
            ``LIKELIHOOD_BINS``): low/moderate/high/very_high.
        primary_duration_min: Duration used for the headline likelihood.
        threshold_intensity_mm_h: M1 p=0.5 threshold intensity for the
            basin (mm/h, primary duration).
        threshold_exceeded: Peak observed intensity >= threshold intensity.
        cannon_threshold_mm_h: Cannon I-D threshold for the storm duration
            (None when the storm is entirely dry).
        cannon_exceeded: Storm mean intensity above the Cannon curve.
        volume_m3: Gartner 2014 volume (None when inputs were not
            supplied — omitted transparently, never guessed).
        volume_class: USGS volume class 1-4 (None with volume).
        volume_omitted_reason: Why the volume stage did not run ("" when it
            ran).
        rainfall: Rain-series reduction record.
        evidence: Ordered stage evidence with citations.
    """

    likelihood: float
    likelihood_class: str
    primary_duration_min: int
    threshold_intensity_mm_h: float
    threshold_exceeded: bool
    cannon_threshold_mm_h: float | None
    cannon_exceeded: bool | None
    volume_m3: float | None
    volume_class: int | None
    volume_omitted_reason: str
    rainfall: RainfallAnalysis
    evidence: list[dict[str, Any]] = field(default_factory=list)


class FireDebrisFlowCascadeDetector:
    """Staged wildfire → debris-flow cascade on the published USGS models.

    Args:
        primary_duration_min: Headline M1 duration (15 recommended by
            Staley et al. 2017 as the best-performing model). Default 15.
        cannon_region: Cannon et al. (2008) threshold family to apply.
            Default ``southern_california`` (the M1 training region).
    """

    def __init__(
        self,
        primary_duration_min: int = 15,
        cannon_region: str = "southern_california",
    ) -> None:
        """Initialize the instance."""
        _m1_coefficients(primary_duration_min)  # fail loud on bad duration
        if cannon_region not in CANNON_ID_THRESHOLDS:
            raise ValueError(
                f"unknown cannon_region {cannon_region!r}; expected one of "
                f"{sorted(CANNON_ID_THRESHOLDS)}"
            )
        self.primary_duration_min = primary_duration_min
        self.cannon_region = cannon_region
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Rain-series reduction
    # ------------------------------------------------------------------
    def analyze_rainfall(
        self, rain_mm: np.ndarray[Any, Any], step_minutes: float
    ) -> RainfallAnalysis:
        """Reduce a rain-gauge series to peak rolling accumulations.

        Args:
            rain_mm: Per-step rainfall depths, mm.
            step_minutes: Gauge step in minutes; must divide the model
                durations (15/30/60) so accumulations are exact rather than
                interpolated.

        Returns:
            RainfallAnalysis with peak accumulations/intensities for every
            M1 duration and the wet-spell summary.

        Raises:
            ValueError: On empty/negative/non-finite input, a step that
                does not divide 15 minutes, or an entirely dry series (a
                debris-flow assessment of a dry record would be meaningless
                — fail loud instead).
        """
        p = np.asarray(rain_mm, dtype=float)
        if p.ndim != 1 or p.size == 0:
            raise ValueError(f"rain series must be 1-D and non-empty, got shape {p.shape}")
        if np.any(p < 0) or not np.all(np.isfinite(p)):
            raise ValueError("rain series must be finite and non-negative")
        if step_minutes <= 0 or not np.isfinite(step_minutes):
            raise ValueError(f"step_minutes must be > 0, got {step_minutes}")
        if 15.0 % step_minutes != 0.0:
            raise ValueError(
                f"step_minutes must divide 15 so 15/30/60-min accumulations are "
                f"exact, got {step_minutes}"
            )
        wet_idx = np.flatnonzero(p > 0.0)
        if wet_idx.size == 0:
            raise ValueError("rain series is entirely dry; nothing to assess")

        peak_accum: dict[int, float] = {}
        peak_intensity: dict[int, float] = {}
        for duration in sorted(M1_COEFFICIENTS):
            window = round(duration / step_minutes)
            if p.size >= window:
                rolling = np.convolve(p, np.ones(window), mode="valid")
                accum = float(np.max(rolling))
            else:
                accum = float(np.sum(p))  # short record: total is the bound
            peak_accum[duration] = accum
            peak_intensity[duration] = accum * 60.0 / duration

        storm_span = int(wet_idx[-1] - wet_idx[0] + 1)
        storm_duration_h = storm_span * step_minutes / 60.0
        storm_mean = float(p[wet_idx[0] : wet_idx[-1] + 1].sum() / storm_duration_h)

        return RainfallAnalysis(
            peak_accum_mm=peak_accum,
            peak_intensity_mm_h=peak_intensity,
            storm_duration_h=storm_duration_h,
            storm_mean_intensity_mm_h=storm_mean,
        )

    # ------------------------------------------------------------------
    # Full cascade
    # ------------------------------------------------------------------
    def assess(
        self,
        t_proportion: float,
        dnbr_mean: float,
        kf_factor: float,
        rain_mm: np.ndarray[Any, Any],
        step_minutes: float,
        burned_mh_km2: float | None = None,
        relief_m: float | None = None,
        wildfire_context: Any | None = None,
    ) -> FireDebrisFlowResult:
        """Run the staged cascade on burn evidence + a rain series.

        Args:
            t_proportion: M1 T term (see :func:`staley_m1_likelihood`).
            dnbr_mean: Mean raw dNBR of the basin.
            kf_factor: Soil KF-factor.
            rain_mm: Rain-gauge series, mm per step.
            step_minutes: Gauge step, minutes.
            burned_mh_km2: Optional watershed moderate/high-severity burned
                area (km²) for the Gartner volume stage.
            relief_m: Optional watershed relief (m) for the volume stage.
            wildfire_context: Optional WildfirePredictionResult or dict
                recorded as burn-stage context. Never feeds the regressions
                (see module docstring).

        Returns:
            FireDebrisFlowResult with likelihood, thresholds, optional
            volume, and the full evidence trail.
        """
        evidence: list[dict[str, Any]] = []

        # Stage 1: burn evidence (validation happens inside the M1 calls).
        _validate_m1_terrain(t_proportion, dnbr_mean, kf_factor)
        burn_detail: dict[str, Any] = {
            "t_proportion": t_proportion,
            "dnbr_mean": dnbr_mean,
            "kf_factor": kf_factor,
        }
        if wildfire_context is not None:
            fire_detected = getattr(
                wildfire_context,
                "fire_detected",
                (
                    wildfire_context.get("fire_detected")
                    if isinstance(wildfire_context, dict)
                    else None
                ),
            )
            burn_detail["wildfire_context"] = {
                "fire_detected": fire_detected,
                "note": (
                    "context only; CNN confidence not used in the published "
                    "regressions (untrained network)"
                ),
            }
        evidence.append(
            {
                "stage": "burn_evidence",
                "detail": burn_detail,
                "citation": "Staley et al. 2017 input definitions (T, dNBR/1000, KF)",
            }
        )

        # Stage 2: rainfall reduction.
        rainfall = self.analyze_rainfall(rain_mm, step_minutes)
        evidence.append(
            {
                "stage": "rainfall",
                "detail": {
                    "peak_accum_mm": rainfall.peak_accum_mm,
                    "storm_duration_h": rainfall.storm_duration_h,
                    "storm_mean_intensity_mm_h": rainfall.storm_mean_intensity_mm_h,
                },
                "citation": "Kean et al. 2011 backwards-differencing storm metrics",
            }
        )

        # Stage 3: assessment.
        duration = self.primary_duration_min
        accum = rainfall.peak_accum_mm[duration]
        likelihood = staley_m1_likelihood(t_proportion, dnbr_mean, kf_factor, accum, duration)
        threshold_intensity = staley_m1_threshold_intensity(
            t_proportion, dnbr_mean, kf_factor, 0.5, duration
        )
        peak_intensity = rainfall.peak_intensity_mm_h[duration]
        threshold_exceeded = peak_intensity >= threshold_intensity

        cannon_threshold = cannon_id_threshold_mm_h(rainfall.storm_duration_h, self.cannon_region)
        cannon_exceeded = rainfall.storm_mean_intensity_mm_h >= cannon_threshold

        evidence.append(
            {
                "stage": "m1_likelihood",
                "detail": {
                    "duration_min": duration,
                    "rain_accum_mm": accum,
                    "likelihood": likelihood,
                    "threshold_intensity_mm_h": threshold_intensity,
                    "threshold_exceeded": threshold_exceeded,
                },
                "citation": "Staley et al. 2017 M1 (USGS OFR 2016-1106 Table 1)",
            }
        )
        evidence.append(
            {
                "stage": "cannon_id_threshold",
                "detail": {
                    "region": self.cannon_region,
                    "storm_duration_h": rainfall.storm_duration_h,
                    "threshold_mm_h": cannon_threshold,
                    "storm_mean_intensity_mm_h": rainfall.storm_mean_intensity_mm_h,
                    "exceeded": cannon_exceeded,
                },
                "citation": "Cannon et al. 2008 burned-area I-D thresholds",
            }
        )

        volume: float | None = None
        volume_class: int | None = None
        omitted = ""
        if burned_mh_km2 is not None and relief_m is not None:
            volume, volume_class = gartner_2014_volume_m3(
                rainfall.peak_intensity_mm_h[15], burned_mh_km2, relief_m
            )
            evidence.append(
                {
                    "stage": "gartner_volume",
                    "detail": {
                        "volume_m3": volume,
                        "volume_class": volume_class,
                        "i15_mm_h": rainfall.peak_intensity_mm_h[15],
                        "burned_mh_km2": burned_mh_km2,
                        "relief_m": relief_m,
                    },
                    "citation": "Gartner et al. 2014 emergency-assessment volume model",
                }
            )
        else:
            missing = [
                name
                for name, value in (
                    ("burned_mh_km2", burned_mh_km2),
                    ("relief_m", relief_m),
                )
                if value is None
            ]
            omitted = (
                f"volume stage omitted: missing {', '.join(missing)} (Gartner et al. "
                "2014 inputs); the cascade does not guess watershed properties"
            )
            evidence.append({"stage": "gartner_volume", "detail": {"omitted": omitted}})

        b1, b2, b3 = LIKELIHOOD_BINS
        likelihood_class = (
            "low"
            if likelihood < b1
            else "moderate" if likelihood < b2 else "high" if likelihood < b3 else "very_high"
        )

        return FireDebrisFlowResult(
            likelihood=likelihood,
            likelihood_class=likelihood_class,
            primary_duration_min=duration,
            threshold_intensity_mm_h=threshold_intensity,
            threshold_exceeded=threshold_exceeded,
            cannon_threshold_mm_h=cannon_threshold,
            cannon_exceeded=cannon_exceeded,
            volume_m3=volume,
            volume_class=volume_class,
            volume_omitted_reason=omitted,
            rainfall=rainfall,
            evidence=evidence,
        )
