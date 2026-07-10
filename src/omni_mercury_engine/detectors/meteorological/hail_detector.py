# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hail / severe-convective environment detector (physics core, works untrained).

Assesses the large-hail potential of a supplied sounding-derived convective
environment using the literature-standard Storm Prediction Center (SPC)
formulations:

* **Significant Hail Parameter (SHIP)** -- the SPC composite for hail
  >= 2 in (5 cm), implemented with the SPC component clamps and low-end
  correction terms (SPC Mesoanalysis parameter description,
  https://www.spc.noaa.gov/exper/mesoanalysis/help/help_sigh.html).
* **CAPE / 0-6 km bulk-shear tiers** -- large-hail probability tiers from
  the SHIP value plus the deep-layer-shear supercell threshold documented in
  Thompson et al. (2003), *Wea. Forecasting*, 18, 1243-1261 (supercell
  proximity soundings cluster above ~35-40 kt of 0-6 km bulk shear;
  18 m/s ~= 35 kt is used here as the supercell-capable floor).
* **NWS Severe Thunderstorm Warning cross-check** -- corroborates the
  environmental assessment against official warnings (Impact-Based Warning
  ``maxHailSize`` / ``hailThreat`` tags when present), fed from
  :class:`~omni_mercury_engine.data_sources.earth_science.NWSWeatherAlertsSource`
  or raw ``api.weather.gov`` GeoJSON.

Scope note (honest input contract): this detector consumes **precomputed**
sounding-derived quantities (MUCAPE, most-unstable-parcel mixing ratio,
700-500 hPa lapse rate, 500 hPa temperature, 0-6 km bulk shear, freezing
level) as produced by SPC mesoanalysis, model soundings, or an upstream
sounding processor.  It deliberately does **not** integrate CAPE from a raw
temperature/dewpoint profile: a rigorous parcel-theory implementation
(pseudoadiabatic ascent, virtual-temperature correction, verification
against a published sounding) is a self-contained project, and a
half-implemented parcel ascent would silently corrupt every downstream
score.  Callers with only raw profiles must compute CAPE with a validated
tool (e.g. SHARPpy or MetPy) first.

This module contains no neural network: the physics core produces its full
output untrained, from real supplied inputs only, and raises on missing or
non-finite inputs rather than guessing.
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
    parse_max_hail_size_in,
    parse_threat_tag,
)

logger = logging.getLogger(__name__)

__all__ = ["HailAssessment", "HailDetector", "ShipComponents"]

# --- SPC SHIP formulation constants (help_sigh.html) -------------------------
_SHIP_DENOMINATOR = 44_000_000.0
_SHIP_SHEAR_FLOOR_MS = 7.0
_SHIP_SHEAR_CAP_MS = 27.0
_SHIP_MIXRATIO_FLOOR_GKG = 11.0
_SHIP_MIXRATIO_CAP_GKG = 13.6
_SHIP_T500_WARM_LIMIT_C = -5.5
_SHIP_LOW_CAPE_JKG = 1300.0
_SHIP_LOW_LAPSE_C_KM = 5.8
_SHIP_LOW_FZL_M = 2400.0

#: Required keys for :meth:`HailDetector.assess` (all sounding-derived).
_REQUIRED_KEYS = (
    "mucape_j_kg",
    "mu_mixing_ratio_g_kg",
    "lapse_rate_700_500_c_km",
    "temp_500_c",
    "shear_0_6km_ms",
    "freezing_level_m",
)

#: NWS CAP event types that carry hail threat information.
_HAIL_ALERT_EVENTS = ("Severe Thunderstorm Warning",)

#: NWS "severe" hail criterion (inches): 1 in diameter (NWS directive 10-511).
_NWS_SEVERE_HAIL_IN = 1.0
#: SPC "significant" hail threshold (inches): 2 in diameter.
_SIG_HAIL_IN = 2.0


@dataclass(frozen=True)
class ShipComponents:
    """SHIP value with the post-clamp component terms that produced it.

    Attributes:
        ship: Significant Hail Parameter (dimensionless).
        mucape_j_kg: MUCAPE as supplied (J/kg).
        mixing_ratio_g_kg: Most-unstable-parcel mixing ratio after the
            SPC 11-13.6 g/kg clamp.
        lapse_rate_c_km: 700-500 hPa lapse rate as supplied (deg C/km).
        temp_500_c: 500 hPa temperature after the -5.5 deg C warm limit.
        shear_ms: 0-6 km bulk shear after the SPC 7-27 m/s clamp.
        low_cape_factor: Multiplicative correction for MUCAPE < 1300 J/kg.
        low_lapse_factor: Multiplicative correction for lapse rate < 5.8 C/km.
        low_fzl_factor: Multiplicative correction for freezing level < 2400 m.
    """

    ship: float
    mucape_j_kg: float
    mixing_ratio_g_kg: float
    lapse_rate_c_km: float
    temp_500_c: float
    shear_ms: float
    low_cape_factor: float
    low_lapse_factor: float
    low_fzl_factor: float


@dataclass
class HailAssessment:
    """Full hail-environment assessment.

    Attributes:
        ship: SHIP components and value.
        tier: Large-hail probability tier (see
            :meth:`HailDetector.classify_tier` for the documented ladder).
        significant_hail_favorable: True when SHIP >= 1 (SPC threshold for
            environments supporting >= 2 in hail).
        supercell_capable_shear: True when 0-6 km bulk shear (unclamped)
            >= the Thompson et al. (2003) supercell floor (18 m/s).
        nws_cross_check: Result of :meth:`HailDetector.cross_check_nws_alerts`
            when alert data was supplied, else None.
        notes: Human-readable derivation notes.
    """

    ship: ShipComponents
    tier: str
    significant_hail_favorable: bool
    supercell_capable_shear: bool
    nws_cross_check: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def _require_finite(name: str, value: Any) -> float:
    """Coerce ``value`` to float, raising ``ValueError`` if missing/non-finite."""
    if value is None:
        raise ValueError(f"Required convective input '{name}' is missing (None).")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Convective input '{name}'={value!r} is not numeric.") from exc
    if not math.isfinite(out):
        raise ValueError(f"Convective input '{name}'={out} is not finite.")
    return out


class HailDetector:
    """Hail / severe-convective detector built on the SPC SHIP formulation.

    The detector is fully deterministic and physics-based; it requires no
    training and carries no neural network.  All thresholds are documented
    literature values (see module docstring for citations).

    Example:
        >>> detector = HailDetector()
        >>> result = detector.assess(
        ...     {
        ...         "mucape_j_kg": 3500.0,
        ...         "mu_mixing_ratio_g_kg": 13.6,
        ...         "lapse_rate_700_500_c_km": 8.0,
        ...         "temp_500_c": -12.0,
        ...         "shear_0_6km_ms": 25.0,
        ...         "freezing_level_m": 3800.0,
        ...     }
        ... )
        >>> result.significant_hail_favorable
        True
    """

    #: Fixed fusion feature dimension (see :meth:`extract_features`).
    FEATURE_DIM = 16

    def __init__(
        self,
        supercell_shear_floor_ms: float = 18.0,
        large_hail_cape_floor_j_kg: float = 1000.0,
        marginal_cape_floor_j_kg: float = 500.0,
    ) -> None:
        """Initialize the detector.

        Args:
            supercell_shear_floor_ms: 0-6 km bulk shear at/above which the
                environment is treated as supercell-capable.  Default 18 m/s
                (~35 kt), the operational floor consistent with the supercell
                proximity-sounding climatology of Thompson et al. (2003).
            large_hail_cape_floor_j_kg: MUCAPE floor for the
                ``large_hail_possible`` tier (1000 J/kg, SPC forecaster
                guidance for organized hail-bearing updrafts).
            marginal_cape_floor_j_kg: MUCAPE floor for the ``marginal`` tier
                (500 J/kg; below this, buoyancy is insufficient for hail
                growth to severe sizes in nearly all documented cases).
        """
        if supercell_shear_floor_ms <= 0 or large_hail_cape_floor_j_kg <= 0:
            raise ValueError("Tier thresholds must be positive.")
        if marginal_cape_floor_j_kg >= large_hail_cape_floor_j_kg:
            raise ValueError("marginal_cape_floor_j_kg must be < large_hail_cape_floor_j_kg.")
        self.supercell_shear_floor_ms = float(supercell_shear_floor_ms)
        self.large_hail_cape_floor_j_kg = float(large_hail_cape_floor_j_kg)
        self.marginal_cape_floor_j_kg = float(marginal_cape_floor_j_kg)

    # ------------------------------------------------------------------
    # SHIP (SPC formulation)
    # ------------------------------------------------------------------

    def compute_ship(
        self,
        mucape_j_kg: float,
        mu_mixing_ratio_g_kg: float,
        lapse_rate_700_500_c_km: float,
        temp_500_c: float,
        shear_0_6km_ms: float,
        freezing_level_m: float,
    ) -> ShipComponents:
        """Compute the SPC Significant Hail Parameter.

        Formulation (SPC Mesoanalysis, help_sigh.html)::

            SHIP = (MUCAPE * MUMR * LR75 * (-T500) * SHR6) / 44_000_000

        with component validity clamps applied before the product:

        * ``SHR6`` (0-6 km bulk shear) confined to 7-27 m/s;
        * ``MUMR`` (MU parcel mixing ratio) confined to 11-13.6 g/kg;
        * ``T500`` capped at -5.5 deg C (warmer values are set to -5.5);

        and three low-end correction factors applied after the product:

        * ``MUCAPE < 1300 J/kg``      -> SHIP *= MUCAPE / 1300
        * ``LR75 < 5.8 deg C/km``     -> SHIP *= LR75 / 5.8
        * ``FZL < 2400 m``            -> SHIP *= FZL / 2400

        SHIP >= 1 indicates an environment favorable for significant
        (>= 2 in) hail; the majority of significant-hail reports occur with
        SHIP between 1.5 and 2.0, and values >= 4 are considered extreme
        (all per the SPC parameter description).

        Args:
            mucape_j_kg: Most-unstable-parcel CAPE (J/kg), >= 0.
            mu_mixing_ratio_g_kg: MU parcel water-vapor mixing ratio (g/kg).
            lapse_rate_700_500_c_km: 700-500 hPa temperature lapse rate
                (deg C/km), > 0 for a physically meaningful mid-level layer.
            temp_500_c: 500 hPa temperature (deg C).
            shear_0_6km_ms: 0-6 km bulk wind difference magnitude (m/s), >= 0.
            freezing_level_m: Freezing-level height (m AGL), >= 0.

        Returns:
            :class:`ShipComponents` with the SHIP value and post-clamp terms.

        Raises:
            ValueError: If any input is missing, non-finite, or outside its
                physically meaningful sign/range.
        """
        mucape = _require_finite("mucape_j_kg", mucape_j_kg)
        mumr = _require_finite("mu_mixing_ratio_g_kg", mu_mixing_ratio_g_kg)
        lr75 = _require_finite("lapse_rate_700_500_c_km", lapse_rate_700_500_c_km)
        t500 = _require_finite("temp_500_c", temp_500_c)
        shr6 = _require_finite("shear_0_6km_ms", shear_0_6km_ms)
        fzl = _require_finite("freezing_level_m", freezing_level_m)

        if mucape < 0:
            raise ValueError(f"mucape_j_kg={mucape} must be >= 0 (CAPE is non-negative).")
        if mumr <= 0:
            raise ValueError(f"mu_mixing_ratio_g_kg={mumr} must be > 0.")
        if lr75 <= 0:
            raise ValueError(
                f"lapse_rate_700_500_c_km={lr75} must be > 0 (temperature decreasing with height)."
            )
        if not -60.0 <= t500 <= 10.0:
            raise ValueError(f"temp_500_c={t500} outside plausible 500 hPa range [-60, 10] C.")
        if shr6 < 0:
            raise ValueError(f"shear_0_6km_ms={shr6} must be >= 0 (vector magnitude).")
        if fzl < 0:
            raise ValueError(f"freezing_level_m={fzl} must be >= 0.")

        shr6_c = min(max(shr6, _SHIP_SHEAR_FLOOR_MS), _SHIP_SHEAR_CAP_MS)
        mumr_c = min(max(mumr, _SHIP_MIXRATIO_FLOOR_GKG), _SHIP_MIXRATIO_CAP_GKG)
        t500_c = min(t500, _SHIP_T500_WARM_LIMIT_C)

        ship = (mucape * mumr_c * lr75 * (-t500_c) * shr6_c) / _SHIP_DENOMINATOR

        low_cape = mucape / _SHIP_LOW_CAPE_JKG if mucape < _SHIP_LOW_CAPE_JKG else 1.0
        low_lapse = lr75 / _SHIP_LOW_LAPSE_C_KM if lr75 < _SHIP_LOW_LAPSE_C_KM else 1.0
        low_fzl = fzl / _SHIP_LOW_FZL_M if fzl < _SHIP_LOW_FZL_M else 1.0
        ship *= low_cape * low_lapse * low_fzl

        return ShipComponents(
            ship=float(ship),
            mucape_j_kg=mucape,
            mixing_ratio_g_kg=mumr_c,
            lapse_rate_c_km=lr75,
            temp_500_c=t500_c,
            shear_ms=shr6_c,
            low_cape_factor=float(low_cape),
            low_lapse_factor=float(low_lapse),
            low_fzl_factor=float(low_fzl),
        )

    # ------------------------------------------------------------------
    # Tier ladder
    # ------------------------------------------------------------------

    def classify_tier(self, ship: float, mucape_j_kg: float, shear_0_6km_ms: float) -> str:
        """Map SHIP + CAPE/shear onto a documented large-hail tier ladder.

        Ladder (highest matching tier wins):

        * ``extreme``                    -- SHIP >= 4 (SPC: "values greater
          than 4 are considered very high").
        * ``significant_hail_likely``    -- SHIP >= 1.5 (SPC: the majority of
          significant-hail reports occur with SHIP 1.5-2.0).
        * ``significant_hail_favorable`` -- SHIP >= 1 (SPC significant-hail
          discriminator).
        * ``large_hail_possible``        -- MUCAPE >= 1000 J/kg with 0-6 km
          shear >= 18 m/s (supercell-capable deep-layer shear per Thompson
          et al. 2003); supercells dominate >= 1 in hail climatology.
        * ``marginal``                   -- MUCAPE >= 500 J/kg (pulse-storm
          small hail possible; sub-severe in most documented cases).
        * ``none``                       -- insufficient buoyancy.

        Args:
            ship: Significant Hail Parameter value.
            mucape_j_kg: MUCAPE (J/kg), unclamped.
            shear_0_6km_ms: 0-6 km bulk shear (m/s), unclamped.

        Returns:
            Tier name string.
        """
        ship_v = _require_finite("ship", ship)
        mucape = _require_finite("mucape_j_kg", mucape_j_kg)
        shear = _require_finite("shear_0_6km_ms", shear_0_6km_ms)

        if ship_v >= 4.0:
            return "extreme"
        if ship_v >= 1.5:
            return "significant_hail_likely"
        if ship_v >= 1.0:
            return "significant_hail_favorable"
        if mucape >= self.large_hail_cape_floor_j_kg and shear >= self.supercell_shear_floor_ms:
            return "large_hail_possible"
        if mucape >= self.marginal_cape_floor_j_kg:
            return "marginal"
        return "none"

    # ------------------------------------------------------------------
    # Full assessment
    # ------------------------------------------------------------------

    def assess(self, convective_data: dict[str, Any]) -> HailAssessment:
        """Assess large-hail potential from a sounding-derived environment.

        Args:
            convective_data: Dict with the required keys ``mucape_j_kg``,
                ``mu_mixing_ratio_g_kg``, ``lapse_rate_700_500_c_km``,
                ``temp_500_c``, ``shear_0_6km_ms``, ``freezing_level_m``,
                plus optional ``nws_alerts`` (any payload accepted by
                :func:`normalize_alert_records`).

        Returns:
            :class:`HailAssessment`.

        Raises:
            ValueError: If any required key is absent or non-finite (the
                detector never fills in fabricated atmosphere values).
        """
        missing = [k for k in _REQUIRED_KEYS if k not in convective_data]
        if missing:
            raise ValueError(
                f"convective_data is missing required sounding-derived inputs: {missing}. "
                "Supply real values; this detector does not fabricate atmospheric state."
            )

        ship = self.compute_ship(
            mucape_j_kg=convective_data["mucape_j_kg"],
            mu_mixing_ratio_g_kg=convective_data["mu_mixing_ratio_g_kg"],
            lapse_rate_700_500_c_km=convective_data["lapse_rate_700_500_c_km"],
            temp_500_c=convective_data["temp_500_c"],
            shear_0_6km_ms=convective_data["shear_0_6km_ms"],
            freezing_level_m=convective_data["freezing_level_m"],
        )

        raw_shear = float(convective_data["shear_0_6km_ms"])
        raw_cape = float(convective_data["mucape_j_kg"])
        tier = self.classify_tier(ship.ship, raw_cape, raw_shear)

        notes = [
            f"SHIP={ship.ship:.3f} (clamped: MUMR={ship.mixing_ratio_g_kg:.1f} g/kg, "
            f"T500={ship.temp_500_c:.1f} C, SHR6={ship.shear_ms:.1f} m/s)",
            f"tier={tier}",
        ]

        cross_check: dict[str, Any] | None = None
        if "nws_alerts" in convective_data:
            cross_check = self.cross_check_nws_alerts(convective_data["nws_alerts"])
            notes.append(
                f"NWS cross-check: {cross_check['n_warnings']} severe thunderstorm warning(s), "
                f"max tagged hail {cross_check['max_hail_size_in']} in"
            )

        assessment = HailAssessment(
            ship=ship,
            tier=tier,
            significant_hail_favorable=ship.ship >= 1.0,
            supercell_capable_shear=raw_shear >= self.supercell_shear_floor_ms,
            nws_cross_check=cross_check,
            notes=notes,
        )
        logger.info("Hail assessment: %s", "; ".join(notes))
        return assessment

    # ------------------------------------------------------------------
    # NWS wiring
    # ------------------------------------------------------------------

    def cross_check_nws_alerts(self, alerts: Any) -> dict[str, Any]:
        """Cross-check against NWS Severe Thunderstorm Warnings.

        Args:
            alerts: Alert payload in any shape accepted by
                :func:`normalize_alert_records` (raw ``api.weather.gov``
                GeoJSON, flat CAP property dicts, or
                ``NWSWeatherAlertsSource`` DataPoints).

        Returns:
            Dict with:
                * ``n_warnings``: count of Severe Thunderstorm Warnings.
                * ``max_hail_size_in``: largest IBW ``maxHailSize`` tag
                  (inches) across the warnings, or ``None`` if untagged.
                * ``hail_threat_tags``: distinct ``hailThreat`` tag values.
                * ``severe_hail_warned``: True if any tag >= 1.0 in (the NWS
                  severe criterion).
                * ``significant_hail_warned``: True if any tag >= 2.0 in
                  (the SPC significant-hail size).

        Raises:
            TypeError: If the payload shape is unrecognized (fail-loud).
        """
        records = normalize_alert_records(alerts)
        warnings = filter_alerts_by_event(records, _HAIL_ALERT_EVENTS)

        sizes = [s for s in (parse_max_hail_size_in(r) for r in warnings) if s is not None]
        tags = sorted(
            {t for t in (parse_threat_tag(r, "hailThreat") for r in warnings) if t is not None}
        )
        max_size = max(sizes) if sizes else None

        return {
            "n_warnings": len(warnings),
            "max_hail_size_in": max_size,
            "hail_threat_tags": tags,
            "severe_hail_warned": bool(max_size is not None and max_size >= _NWS_SEVERE_HAIL_IN),
            "significant_hail_warned": bool(max_size is not None and max_size >= _SIG_HAIL_IN),
        }

    # ------------------------------------------------------------------
    # Fusion interface
    # ------------------------------------------------------------------

    def extract_features(self, data: Any) -> torch.Tensor:
        """Extract a fixed-width feature vector for the fusion registry.

        Two honest input paths:

        * ``dict`` with the required convective keys -> physics features
          derived from the real SHIP computation (SHIP value, clamped
          components, correction factors, tier ordinal).
        * array-like -> documented robust summary statistics of the supplied
          values (mean, std, min, max, median, IQR); no meteorological
          meaning is invented for anonymous arrays.

        Args:
            data: Convective-input dict or numeric array.

        Returns:
            ``torch.Tensor`` of shape ``(FEATURE_DIM,)``.

        Raises:
            ValueError: For dict input missing required keys (propagated
                from :meth:`assess`).
        """
        if isinstance(data, dict):
            assessment = self.assess(data)
            s = assessment.ship
            tier_ladder = [
                "none",
                "marginal",
                "large_hail_possible",
                "significant_hail_favorable",
                "significant_hail_likely",
                "extreme",
            ]
            features = [
                s.ship,
                s.mucape_j_kg / 1000.0,
                s.mixing_ratio_g_kg,
                s.lapse_rate_c_km,
                -s.temp_500_c,
                s.shear_ms,
                s.low_cape_factor,
                s.low_lapse_factor,
                s.low_fzl_factor,
                float(tier_ladder.index(assessment.tier)),
                1.0 if assessment.significant_hail_favorable else 0.0,
                1.0 if assessment.supercell_capable_shear else 0.0,
            ]
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
