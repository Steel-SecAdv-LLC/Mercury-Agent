# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Winter / ice-storm detector (physics core, works untrained).

Implements the literature-standard winter-precipitation formulations:

* **Partial-thickness precipitation-type inference** from a supplied thermal
  profile: the 1000-850 hPa and 850-700 hPa partial thicknesses with the
  canonical operational thresholds (1300 m low-level rain/snow discriminator;
  1540 m / 1560 m mid-level melting-layer discriminators), after the
  thickness-based precipitation-type scheme of Keeter & Cline (1991),
  *Wea. Forecasting*, 6, 456-469, as adapted in NWS operational forecasting
  references.
* **Surface wet-bulb method** when only surface data is available: the
  Stull (2011) wet-bulb approximation from temperature and relative
  humidity (Stull, R., 2011: "Wet-Bulb Temperature from Relative Humidity
  and Air Temperature", *J. Appl. Meteor. Climatol.*, 50, 2267-2269),
  enforced within its stated validity region (RH 5-99 %, T -20 to +50 deg C,
  fitted at sea-level pressure 101.325 kPa).
* **Flat-surface freezing-rain accretion (FRAM-like)** from precipitation
  rate and wet-bulb temperature, following the ice-to-liquid-ratio (ILR)
  findings of Sanders & Barjenbruch (2016): "Analysis of Ice-to-Liquid
  Ratios during Freezing Rain and the Development of an Ice Accumulation
  Model", *Wea. Forecasting*, 31, 1041-1060.  S&B report a mean flat-surface
  ILR near 0.72, decreasing both as the wet bulb approaches 0 deg C and at
  heavy precipitation rates (runoff / latent-heat limitation).  The
  piecewise-linear constants encoding those two documented reductions here
  are this module's own approximation of that published structure -- they
  are NOT the FRAM nomogram regressions, which S&B distribute graphically.
* **SPIA-style ice damage index tiers** after the Sperry-Piltz Ice
  Accumulation Index (Sperry & Piltz; operationally used by NWS offices;
  see https://www.spia-index.com): damage index 0-5 from radial-equivalent
  flat ice accretion and wind.
* **Blizzard criteria check** per the NWS Glossary definition: sustained
  wind or frequent gusts >= 35 mph (15.6 m/s) AND considerable falling
  and/or blowing snow reducing visibility below 1/4 mile (~400 m), both
  persisting for >= 3 hours (https://w1.weather.gov/glossary/, "Blizzard").
* **NWS winter alert wiring** (Winter Storm / Ice Storm / Blizzard Warning,
  Winter Weather Advisory, Winter Storm Watch) via the shared CAP helpers.

No neural network; every output derives from the supplied real inputs, and
missing / non-finite inputs raise instead of being silently defaulted.
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
    "BlizzardCheck",
    "IceAccretionResult",
    "PrecipType",
    "WinterStormDetector",
]

# --- Partial-thickness thresholds (Keeter & Cline 1991 lineage) --------------
_TH_1000_850_RAIN_SNOW_M = 1300.0  # low-level cold criterion (m)
_TH_850_700_SNOW_M = 1540.0  # below: no melting layer aloft (m)
_TH_850_700_WARM_M = 1560.0  # above: deep warm layer aloft (m)

# --- Stull (2011) validity region --------------------------------------------
_STULL_T_MIN_C = -20.0
_STULL_T_MAX_C = 50.0
_STULL_RH_MIN_PCT = 5.0
_STULL_RH_MAX_PCT = 99.0

#: Surface wet-bulb rain/snow discriminator (deg C).  Operational rule of
#: thumb: the rain/snow transition tracks the surface wet bulb near +1 C
#: far better than the dry bulb (snow commonly reaches the ground with
#: dry-bulb temperatures a few degrees above freezing when the wet bulb
#: is at or below about +1 C).
_WETBULB_SNOW_MAX_C = 1.0

# --- Sanders & Barjenbruch (2016) ILR structure -------------------------------
_ILR_FLAT_MEAN = 0.72  # S&B 2016 mean flat-surface ILR
_ILR_WARM_RAMP_C = 1.0  # own approximation: linear ILR onset over [-1, 0] C wet bulb
_ILR_RUNOFF_RATE_MM_HR = 2.5  # own approximation: runoff reduction beyond this rate
_ILR_RUNOFF_EXPONENT = 0.5  # own approximation: sqrt taper encodes S&B's decline

# --- NWS blizzard criteria -----------------------------------------------------
_BLIZZARD_WIND_MS = 15.6  # 35 mph sustained or frequent gusts
_BLIZZARD_VIS_M = 400.0  # 1/4 statute mile
_BLIZZARD_DURATION_S = 3.0 * 3600.0  # 3 hours

#: SPIA wind bands (mph) and ice (inches) breakpoints -- Sperry-Piltz matrix.
_SPIA_WIND_LOW_MPH = 15.0
_SPIA_WIND_HIGH_MPH = 25.0

_WINTER_ALERT_EVENTS = (
    "Winter Storm Warning",
    "Ice Storm Warning",
    "Blizzard Warning",
    "Winter Weather Advisory",
    "Winter Storm Watch",
)

_MM_PER_INCH = 25.4


class PrecipType:
    """Canonical precipitation-type labels emitted by this detector."""

    SNOW = "snow"
    SLEET = "sleet"
    FREEZING_RAIN = "freezing_rain"
    RAIN = "rain"


@dataclass(frozen=True)
class IceAccretionResult:
    """Flat-surface ice accretion estimate.

    Attributes:
        flat_ice_in: Accreted flat-surface ice (inches of ice thickness
            equivalent, liquid basis x ILR).
        liquid_equivalent_in: Total liquid precipitation input (inches).
        mean_ilr: Precipitation-weighted mean ice-to-liquid ratio applied.
        spia_index: SPIA-style damage index 0-5 (requires wind input).
        spia_description: Published damage description for the index tier.
    """

    flat_ice_in: float
    liquid_equivalent_in: float
    mean_ilr: float
    spia_index: int
    spia_description: str


@dataclass
class BlizzardCheck:
    """Result of the NWS blizzard-criteria evaluation.

    Attributes:
        blizzard: True when a contiguous >= 3 h window satisfied both the
            wind and visibility criteria.
        longest_qualifying_hours: Longest contiguous qualifying stretch (h).
        n_qualifying_samples: Number of samples meeting both criteria.
        notes: Derivation notes.
    """

    blizzard: bool
    longest_qualifying_hours: float
    n_qualifying_samples: int
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


class WinterStormDetector:
    """Winter / ice-storm detector built on cited operational formulations.

    Deterministic physics core -- no training, no neural network.  See the
    module docstring for the formulation citations.

    Example:
        >>> detector = WinterStormDetector()
        >>> detector.precip_type_partial_thickness(
        ...     thickness_1000_850_m=1285.0, thickness_850_700_m=1525.0
        ... )
        'snow'
    """

    #: Fixed fusion feature dimension (see :meth:`extract_features`).
    FEATURE_DIM = 16

    # ------------------------------------------------------------------
    # Precipitation type: partial thickness
    # ------------------------------------------------------------------

    def precip_type_partial_thickness(
        self,
        thickness_1000_850_m: float | None = None,
        thickness_850_700_m: float | None = None,
        z1000_m: float | None = None,
        z850_m: float | None = None,
        z700_m: float | None = None,
        surface_temp_c: float | None = None,
    ) -> str:
        """Infer precipitation type from 1000-850 / 850-700 hPa thicknesses.

        Decision matrix (thresholds per the Keeter & Cline 1991 lineage of
        operational partial-thickness forecasting; all values in metres):

        * ``TH850-700 < 1540``: no melting layer aloft -- ``snow`` if
          ``TH1000-850 < 1300`` (cold low levels), else ``rain`` (snow melts
          in the warm boundary layer).
        * ``1540 <= TH850-700 <= 1560``: partial melting aloft -- ``sleet``
          if ``TH1000-850 < 1300`` (refreezing layer), else ``rain``.
        * ``TH850-700 > 1560``: deep warm layer aloft -- ``freezing_rain``
          if ``TH1000-850 < 1300`` (subfreezing surface layer), else
          ``rain``.

        When ``surface_temp_c`` is supplied it disambiguates the
        freezing-rain branch: a surface temperature above 0 deg C downgrades
        ``freezing_rain`` to ``rain`` (no surface accretion possible).

        Args:
            thickness_1000_850_m: 1000-850 hPa thickness (m).  Either supply
                both thicknesses directly or supply the three geopotential
                heights below.
            thickness_850_700_m: 850-700 hPa thickness (m).
            z1000_m: 1000 hPa geopotential height (m), alternative input.
            z850_m: 850 hPa geopotential height (m), alternative input.
            z700_m: 700 hPa geopotential height (m), alternative input.
            surface_temp_c: Optional surface (2 m) temperature (deg C).

        Returns:
            One of :class:`PrecipType` (``snow`` / ``sleet`` /
            ``freezing_rain`` / ``rain``).

        Raises:
            ValueError: If neither thickness pair nor height triple is fully
                supplied, or any value is non-finite / non-physical.
        """
        if thickness_1000_850_m is None or thickness_850_700_m is None:
            if z1000_m is None or z850_m is None or z700_m is None:
                raise ValueError(
                    "Supply both partial thicknesses (thickness_1000_850_m, "
                    "thickness_850_700_m) or all three geopotential heights "
                    "(z1000_m, z850_m, z700_m); refusing to guess the thermal profile."
                )
            z1000 = _require_finite("z1000_m", z1000_m)
            z850 = _require_finite("z850_m", z850_m)
            z700 = _require_finite("z700_m", z700_m)
            thickness_1000_850_m = z850 - z1000
            thickness_850_700_m = z700 - z850

        th_low = _require_finite("thickness_1000_850_m", thickness_1000_850_m)
        th_mid = _require_finite("thickness_850_700_m", thickness_850_700_m)
        if not 1000.0 <= th_low <= 1600.0 or not 1300.0 <= th_mid <= 1800.0:
            raise ValueError(
                f"Partial thicknesses (low={th_low} m, mid={th_mid} m) outside "
                "plausible tropospheric ranges; check units (metres)."
            )

        cold_low_levels = th_low < _TH_1000_850_RAIN_SNOW_M

        if th_mid < _TH_850_700_SNOW_M:
            ptype = PrecipType.SNOW if cold_low_levels else PrecipType.RAIN
        elif th_mid <= _TH_850_700_WARM_M:
            ptype = PrecipType.SLEET if cold_low_levels else PrecipType.RAIN
        else:
            ptype = PrecipType.FREEZING_RAIN if cold_low_levels else PrecipType.RAIN

        if ptype == PrecipType.FREEZING_RAIN and surface_temp_c is not None:
            if _require_finite("surface_temp_c", surface_temp_c) > 0.0:
                ptype = PrecipType.RAIN

        return ptype

    # ------------------------------------------------------------------
    # Precipitation type: surface wet bulb (Stull 2011)
    # ------------------------------------------------------------------

    @staticmethod
    def wet_bulb_stull(temp_c: float, rh_pct: float) -> float:
        """Wet-bulb temperature via the Stull (2011) approximation.

        Formula (Stull 2011, Eq. 1; T in deg C, RH in percent)::

            Tw = T * atan(0.151977 * sqrt(RH + 8.313659))
                 + atan(T + RH) - atan(RH - 1.676331)
                 + 0.00391838 * RH**1.5 * atan(0.023101 * RH)
                 - 4.686035

        Stated validity: RH 5-99 %, T -20 to +50 deg C (excluding the
        cold/dry corner Stull marks invalid), fitted for sea-level pressure
        101.325 kPa with mean absolute error ~0.28 deg C.  The paper's
        worked example (T=20 C, RH=50 %) yields Tw=13.7 C.

        Args:
            temp_c: Air temperature (deg C).
            rh_pct: Relative humidity (percent, 5-99).

        Returns:
            Wet-bulb temperature (deg C).

        Raises:
            ValueError: Outside the published validity region -- the
                approximation degrades there and this module refuses to
                extrapolate.
        """
        t = _require_finite("temp_c", temp_c)
        rh = _require_finite("rh_pct", rh_pct)
        if not _STULL_T_MIN_C <= t <= _STULL_T_MAX_C:
            raise ValueError(
                f"temp_c={t} outside the Stull (2011) validity range "
                f"[{_STULL_T_MIN_C}, {_STULL_T_MAX_C}] C."
            )
        if not _STULL_RH_MIN_PCT <= rh <= _STULL_RH_MAX_PCT:
            raise ValueError(
                f"rh_pct={rh} outside the Stull (2011) validity range "
                f"[{_STULL_RH_MIN_PCT}, {_STULL_RH_MAX_PCT}] %."
            )
        return (
            t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(t + rh)
            - math.atan(rh - 1.676331)
            + 0.00391838 * rh**1.5 * math.atan(0.023101 * rh)
            - 4.686035
        )

    def precip_type_surface(self, temp_c: float, rh_pct: float) -> str:
        """Infer rain vs snow from surface data only (wet-bulb method).

        Uses :meth:`wet_bulb_stull`; classifies ``snow`` when the surface
        wet bulb is <= +1.0 deg C, else ``rain``.  The wet bulb tracks the
        rain/snow transition far better than the dry bulb because falling
        snow cools the air toward the wet bulb by evaporation/melting
        (operational wet-bulb rule of thumb).

        Transparent limitation: surface-only data cannot detect an elevated warm
        nose, so this method **cannot** discriminate freezing rain or sleet
        from snow -- those require the profile-based
        :meth:`precip_type_partial_thickness`.  This method therefore only
        ever returns ``snow`` or ``rain``.

        Args:
            temp_c: Surface air temperature (deg C).
            rh_pct: Surface relative humidity (percent).

        Returns:
            ``PrecipType.SNOW`` or ``PrecipType.RAIN``.

        Raises:
            ValueError: Propagated from :meth:`wet_bulb_stull` outside its
                validity region.
        """
        tw = self.wet_bulb_stull(temp_c, rh_pct)
        return PrecipType.SNOW if tw <= _WETBULB_SNOW_MAX_C else PrecipType.RAIN

    # ------------------------------------------------------------------
    # Freezing-rain accretion (FRAM-like) + SPIA tiers
    # ------------------------------------------------------------------

    @staticmethod
    def _ilr_flat(wet_bulb_c: float, precip_rate_mm_hr: float) -> float:
        """Ice-to-liquid ratio for a flat surface (S&B 2016 structure).

        Encodes the two documented reductions around the S&B mean flat ILR
        of 0.72: (a) accretion efficiency collapses as the wet bulb rises
        through -1..0 deg C (linear ramp, own approximation); (b) heavy
        precipitation rates shed liquid before it freezes (square-root taper
        beyond 2.5 mm/h, own approximation).  Wet bulbs above 0 deg C give
        zero accretion.

        Args:
            wet_bulb_c: Wet-bulb temperature (deg C).
            precip_rate_mm_hr: Liquid-equivalent precipitation rate (mm/h).

        Returns:
            Dimensionless ILR in [0, 0.72].
        """
        if wet_bulb_c > 0.0:
            return 0.0
        warm_factor = min(1.0, -wet_bulb_c / _ILR_WARM_RAMP_C)
        rate_factor = 1.0
        if precip_rate_mm_hr > _ILR_RUNOFF_RATE_MM_HR:
            rate_factor = (_ILR_RUNOFF_RATE_MM_HR / precip_rate_mm_hr) ** _ILR_RUNOFF_EXPONENT
        return _ILR_FLAT_MEAN * warm_factor * rate_factor

    def ice_accretion(
        self,
        precip_rate_mm_hr: Any,
        wet_bulb_c: Any,
        duration_hr: Any,
        wind_speed_mph: Any,
    ) -> IceAccretionResult:
        """Estimate flat-surface freezing-rain accretion and its SPIA tier.

        Accretion model (FRAM-like; Sanders & Barjenbruch 2016): flat ice =
        sum over intervals of (liquid rate x interval x ILR), with ILR from
        :meth:`_ilr_flat`.  Scalar or per-interval array inputs accepted;
        array inputs must share a common length and each interval spans
        ``duration_hr[i]`` hours.

        Args:
            precip_rate_mm_hr: Liquid-equivalent rate(s), mm/h, >= 0.
            wet_bulb_c: Wet-bulb temperature(s), deg C.
            duration_hr: Interval duration(s), hours, > 0.
            wind_speed_mph: Representative wind (mph) for the SPIA tier,
                >= 0 (use the event-mean or peak sustained wind).

        Returns:
            :class:`IceAccretionResult`.

        Raises:
            ValueError: On missing, non-finite, negative, or
                length-mismatched inputs.
        """
        rate = np.atleast_1d(np.asarray(precip_rate_mm_hr, dtype=np.float64))
        tw = np.atleast_1d(np.asarray(wet_bulb_c, dtype=np.float64))
        dur = np.atleast_1d(np.asarray(duration_hr, dtype=np.float64))
        wind = _require_finite("wind_speed_mph", wind_speed_mph)

        n = max(rate.size, tw.size, dur.size)
        if rate.size == 1:
            rate = np.full(n, rate[0])
        if tw.size == 1:
            tw = np.full(n, tw[0])
        if dur.size == 1:
            dur = np.full(n, dur[0])
        if not (rate.size == tw.size == dur.size == n):
            raise ValueError(
                f"Array-length mismatch: rate={rate.size}, wet_bulb={tw.size}, "
                f"duration={dur.size}."
            )
        for name, arr in (("precip_rate_mm_hr", rate), ("wet_bulb_c", tw), ("duration_hr", dur)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Input '{name}' contains non-finite values.")
        if np.any(rate < 0):
            raise ValueError("precip_rate_mm_hr must be >= 0.")
        if np.any(dur <= 0):
            raise ValueError("duration_hr must be > 0.")
        if wind < 0:
            raise ValueError("wind_speed_mph must be >= 0.")

        liquid_mm = rate * dur
        ilr = np.array([self._ilr_flat(t, r) for t, r in zip(tw, rate)])
        ice_mm = liquid_mm * ilr

        total_liquid_in = float(np.sum(liquid_mm) / _MM_PER_INCH)
        total_ice_in = float(np.sum(ice_mm) / _MM_PER_INCH)
        mean_ilr = float(np.sum(ice_mm) / np.sum(liquid_mm)) if np.sum(liquid_mm) > 0 else 0.0

        index, description = self.spia_tier(total_ice_in, wind)
        return IceAccretionResult(
            flat_ice_in=total_ice_in,
            liquid_equivalent_in=total_liquid_in,
            mean_ilr=mean_ilr,
            spia_index=index,
            spia_description=description,
        )

    @staticmethod
    def spia_tier(ice_in: float, wind_mph: float) -> tuple[int, str]:
        """SPIA-style damage index from flat ice accretion and wind.

        Matrix after the published Sperry-Piltz Ice Accumulation Index
        (wind bands < 15 / 15-25 / >= 25 mph against radial-ice
        breakpoints 0.10 / 0.25 / 0.50 / 0.75 / 1.00 / 1.50 in), with the
        published damage legend.

        Args:
            ice_in: Ice accretion (inches).
            wind_mph: Wind speed (mph).

        Returns:
            Tuple ``(index 0-5, damage description)``.

        Raises:
            ValueError: On non-finite or negative inputs.
        """
        ice = _require_finite("ice_in", ice_in)
        wind = _require_finite("wind_mph", wind_mph)
        if ice < 0 or wind < 0:
            raise ValueError("ice_in and wind_mph must be >= 0.")

        if wind < _SPIA_WIND_LOW_MPH:
            band = 0
        elif wind < _SPIA_WIND_HIGH_MPH:
            band = 1
        else:
            band = 2

        # index[band] thresholds: minimum ice (inches) to reach index 1..5.
        thresholds = {
            0: [0.25, 0.50, 0.75, 1.00, 1.50],
            1: [0.10, 0.25, 0.50, 0.75, 1.00],
            2: [0.10, 0.10, 0.25, 0.50, 0.75],
        }[band]

        index = 0
        for i, threshold in enumerate(thresholds, start=1):
            if ice >= threshold:
                index = i

        descriptions = {
            0: "Minimal risk of damage; slippery surfaces possible.",
            1: "Few utility interruptions; isolated outages up to a day.",
            2: "Scattered utility interruptions; outages 1-2 days possible.",
            3: "Numerous utility interruptions; outages 1-5 days.",
            4: "Prolonged, widespread utility interruptions; outages 5-10 days.",
            5: "Catastrophic damage to utility systems; outages 10+ days.",
        }
        return index, descriptions[index]

    # ------------------------------------------------------------------
    # Blizzard criteria (NWS)
    # ------------------------------------------------------------------

    def check_blizzard_criteria(
        self,
        times_s: Any,
        wind_speed_ms: Any,
        visibility_m: Any,
    ) -> BlizzardCheck:
        """Check the NWS blizzard definition over an observation series.

        NWS Glossary "Blizzard": sustained wind or frequent gusts >= 35 mph
        (15.6 m/s) accompanied by considerable falling and/or blowing snow
        frequently reducing visibility below 1/4 mile (~400 m), prevailing
        for 3 hours or longer.  A contiguous run of samples meeting both
        criteria must span >= 3 h; a gap (any non-qualifying sample) resets
        the run.

        Args:
            times_s: Sample epoch times (seconds), strictly increasing.
            wind_speed_ms: Sustained wind or frequent-gust speed (m/s).
            visibility_m: Visibility (m).

        Returns:
            :class:`BlizzardCheck`.

        Raises:
            ValueError: On missing/misaligned/non-finite series or
                non-monotonic timestamps.
        """
        t = np.asarray(times_s, dtype=np.float64)
        w = np.asarray(wind_speed_ms, dtype=np.float64)
        v = np.asarray(visibility_m, dtype=np.float64)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("times_s must be a 1-D series with >= 2 samples.")
        if not (t.size == w.size == v.size):
            raise ValueError(
                f"Series length mismatch: times={t.size}, wind={w.size}, visibility={v.size}."
            )
        for name, arr in (("times_s", t), ("wind_speed_ms", w), ("visibility_m", v)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Series '{name}' contains non-finite values.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times_s must be strictly increasing.")
        if np.any(w < 0) or np.any(v < 0):
            raise ValueError("wind_speed_ms and visibility_m must be >= 0.")

        qualifying = (w >= _BLIZZARD_WIND_MS) & (v < _BLIZZARD_VIS_M)

        longest_s = 0.0
        run_start: float | None = None
        for i in range(t.size):
            if qualifying[i]:
                if run_start is None:
                    run_start = t[i]
                longest_s = max(longest_s, t[i] - run_start)
            else:
                run_start = None

        blizzard = longest_s >= _BLIZZARD_DURATION_S
        notes = [
            f"criteria: wind >= {_BLIZZARD_WIND_MS} m/s (35 mph) and visibility < "
            f"{_BLIZZARD_VIS_M:.0f} m (1/4 mi) for >= 3 h (NWS Glossary)",
            f"longest qualifying stretch: {longest_s / 3600.0:.2f} h",
        ]
        return BlizzardCheck(
            blizzard=bool(blizzard),
            longest_qualifying_hours=float(longest_s / 3600.0),
            n_qualifying_samples=int(np.sum(qualifying)),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # NWS wiring
    # ------------------------------------------------------------------

    def cross_check_nws_alerts(self, alerts: Any) -> dict[str, Any]:
        """Cross-check against active NWS winter products.

        Args:
            alerts: Alert payload in any shape accepted by
                :func:`normalize_alert_records`.

        Returns:
            Dict with ``n_winter_alerts``, ``events`` (sorted distinct CAP
            event names found), and per-product booleans
            ``ice_storm_warned`` / ``blizzard_warned`` /
            ``winter_storm_warned``.

        Raises:
            TypeError: If the payload shape is unrecognized.
        """
        records = normalize_alert_records(alerts)
        winter = filter_alerts_by_event(records, _WINTER_ALERT_EVENTS)
        events = sorted({str(r.get("event")) for r in winter})
        return {
            "n_winter_alerts": len(winter),
            "events": events,
            "ice_storm_warned": "Ice Storm Warning" in events,
            "blizzard_warned": "Blizzard Warning" in events,
            "winter_storm_warned": "Winter Storm Warning" in events,
        }

    # ------------------------------------------------------------------
    # Fusion interface
    # ------------------------------------------------------------------

    def extract_features(self, data: Any) -> torch.Tensor:
        """Extract a fixed-width feature vector for the fusion registry.

        Dict input runs the real physics paths present in the dict:
        thickness keys -> precipitation-type one-hot; accretion keys
        (``precip_rate_mm_hr``, ``wet_bulb_c``, ``duration_hr``,
        ``wind_speed_mph``) -> accretion + SPIA features.  Array input
        yields documented robust summary statistics only.

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
            if "thickness_1000_850_m" in data and "thickness_850_700_m" in data:
                ptype = self.precip_type_partial_thickness(
                    thickness_1000_850_m=data["thickness_1000_850_m"],
                    thickness_850_700_m=data["thickness_850_700_m"],
                    surface_temp_c=data.get("surface_temp_c"),
                )
                order = [
                    PrecipType.SNOW,
                    PrecipType.SLEET,
                    PrecipType.FREEZING_RAIN,
                    PrecipType.RAIN,
                ]
                features.extend(1.0 if ptype == p else 0.0 for p in order)
                features.append(float(data["thickness_1000_850_m"]) / 1000.0)
                features.append(float(data["thickness_850_700_m"]) / 1000.0)
                used = True
            if all(
                k in data
                for k in ("precip_rate_mm_hr", "wet_bulb_c", "duration_hr", "wind_speed_mph")
            ):
                acc = self.ice_accretion(
                    data["precip_rate_mm_hr"],
                    data["wet_bulb_c"],
                    data["duration_hr"],
                    data["wind_speed_mph"],
                )
                features.extend(
                    [acc.flat_ice_in, acc.liquid_equivalent_in, acc.mean_ilr, float(acc.spia_index)]
                )
                used = True
            if not used:
                raise ValueError(
                    "Dict input carries no recognized winter-storm physics inputs "
                    "(need partial thicknesses and/or accretion inputs)."
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
