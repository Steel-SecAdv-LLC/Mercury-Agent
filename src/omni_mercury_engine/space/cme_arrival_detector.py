# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""CME arrival-time prediction from coronagraph kinematics (DONKI CMEAnalysis).

Consumes NASA DONKI ``CMEAnalysis`` records (speed, half-angle, direction at
21.5 solar radii) and produces an Earth arrival-time window from two
independent, literature-anchored empirical models:

1. **Empirical shock-arrival (ESA) model** — Gopalswamy et al. (2000, GRL 27,
   145) effective interplanetary acceleration ``a [m s^-2] = 1.41 - 0.0035 u``
   (``u`` in km/s), with acceleration cessation at 0.76 AU per Gopalswamy et
   al. (2001, JGR 106, 29207, "Predicting the 1-AU arrival times of coronal
   mass ejections"). Two-phase constant-acceleration kinematics over the
   Sun-Earth distance.

2. **Drag-based model (DBM)** — Vršnak et al. (2013, Sol. Phys. 285, 295):
   ``dv/dt = -gamma |v - w| (v - w)`` with the analytic heliocentric-distance
   solution ``r(t) = r0 + w t ± (1/gamma) ln(1 ± gamma |v0 - w| t)``.
   The drag parameter range ``gamma = 0.2e-7 .. 2.0e-7 km^-1`` and ambient
   solar-wind speed ``w = 300 .. 500 km/s`` follow Vršnak et al. (2013) and
   the DBM ensemble practice of Dumbović et al. (2018, ApJ 854, 180).

Earth-directedness is assessed from cone geometry: the angular separation of
the CME apex direction (Stonyhurst latitude/longitude; Earth at 0°, 0°) from
the Sun-Earth line versus the reconstructed cone half-angle.

The arrival window (earliest / latest / most probable) spans the DBM ensemble
corners plus both central models; confidence combines inter-model spread with
the impact-geometry class. Physics only — no neural network is used or
justified here: the two calibrated empirical models are the operational state
of practice and work untrained.

Fail-loud contract: records missing any of the required kinematic fields
(``speed``, ``latitude``, ``longitude``, ``halfAngle``, ``time21_5``) raise
``ValueError``; no defaults are fabricated.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import product
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------
AU_KM: float = 1.495978707e8
AU_M: float = 1.495978707e11
R_SUN_KM: float = 6.957e5
#: DONKI CMEAnalysis kinematics are referenced to 21.5 solar radii
#: (the WSA-ENLIL inner boundary).
DONKI_R0_KM: float = 21.5 * R_SUN_KM

# --------------------------------------------------------------------------
# Gopalswamy empirical shock-arrival model coefficients
# --------------------------------------------------------------------------
#: Effective IP acceleration intercept, m s^-2 (Gopalswamy et al. 2000).
_G2000_A0: float = 1.41
#: Effective IP acceleration slope, m s^-2 per (km/s) (Gopalswamy et al. 2000).
_G2000_A1: float = 0.0035
#: Acceleration-cessation distance, AU (Gopalswamy et al. 2001).
_ACCEL_CESSATION_AU: float = 0.76
#: Calibration domain of the empirical model (km/s). The Gopalswamy sample
#: spans slow (~100 km/s) to extreme (~3000 km/s) CMEs; beyond that the
#: linear acceleration law is an extrapolation we refuse to make.
_ESA_SPEED_DOMAIN_KM_S: tuple[float, float] = (100.0, 4000.0)

# --------------------------------------------------------------------------
# Drag-based model documented parameter ranges (Vršnak et al. 2013)
# --------------------------------------------------------------------------
#: Documented gamma range, km^-1: 0.2e-7 (massive, fast CMEs) to 2.0e-7
#: (slow, low-mass CMEs). Vršnak et al. (2013), Sec. 3.
DBM_GAMMA_RANGE_PER_KM: tuple[float, float] = (0.2e-7, 2.0e-7)
#: Hard validity bounds accepted by :func:`dbm_transit_time_hours`.
_DBM_GAMMA_HARD_BOUNDS: tuple[float, float] = (0.05e-7, 5.0e-7)
#: Ambient solar-wind speed range, km/s (Vršnak et al. 2013; Dumbović et
#: al. 2018 ensemble practice).
DBM_WIND_RANGE_KM_S: tuple[float, float] = (300.0, 500.0)
_DBM_WIND_HARD_BOUNDS: tuple[float, float] = (200.0, 900.0)
_DBM_SPEED_DOMAIN_KM_S: tuple[float, float] = (50.0, 4000.0)


def gopalswamy_transit_time_hours(speed_km_s: float) -> float:
    """Sun-to-1-AU CME transit time from the Gopalswamy empirical model.

    Two-phase kinematics: constant effective acceleration
    ``a = 1.41 - 0.0035 u`` (m/s^2, ``u`` in km/s; Gopalswamy et al. 2000)
    from the Sun out to the cessation distance 0.76 AU (Gopalswamy et al.
    2001), then constant speed over the remaining 0.24 AU.

    Worked anchor (also locked in the unit tests): ``u = 1000 km/s`` gives
    ``a = -2.09 m/s^2``, phase-1 time 1.31865e5 s, arrival speed 724 km/s,
    total transit 50.4 h — consistent with the ~45-50 h band reported for
    1000 km/s CMEs in Gopalswamy et al. (2001), Fig. 4.

    Args:
        speed_km_s: Initial CME speed near the Sun (coronagraph plane-of-sky
            or reconstructed space speed), km/s.

    Returns:
        Transit time to 1 AU in hours.

    Raises:
        ValueError: If the speed lies outside the model calibration domain,
            or the CME would decelerate to rest before 0.76 AU (outside the
            empirical model's validity).
    """
    lo, hi = _ESA_SPEED_DOMAIN_KM_S
    if not math.isfinite(speed_km_s) or not lo <= speed_km_s <= hi:
        raise ValueError(
            f"CME speed {speed_km_s!r} km/s outside the Gopalswamy empirical "
            f"model calibration domain [{lo:.0f}, {hi:.0f}] km/s."
        )

    u_m_s = speed_km_s * 1000.0
    accel = _G2000_A0 - _G2000_A1 * speed_km_s  # m s^-2
    d1_m = _ACCEL_CESSATION_AU * AU_M

    if abs(accel) < 1e-12:
        v1_m_s = u_m_s
        t1_s = d1_m / u_m_s
    else:
        discriminant = u_m_s * u_m_s + 2.0 * accel * d1_m
        if discriminant <= 0.0:
            raise ValueError(
                f"CME at {speed_km_s:.0f} km/s would decelerate to rest before "
                f"{_ACCEL_CESSATION_AU} AU under a = {accel:.3f} m/s^2; outside "
                "the empirical model's validity domain."
            )
        v1_m_s = math.sqrt(discriminant)
        t1_s = (v1_m_s - u_m_s) / accel

    d2_m = (1.0 - _ACCEL_CESSATION_AU) * AU_M
    t2_s = d2_m / v1_m_s
    return (t1_s + t2_s) / 3600.0


def dbm_transit_time_hours(
    speed_km_s: float,
    gamma_per_km: float = 1.0e-7,
    wind_km_s: float = 400.0,
    r0_km: float = DONKI_R0_KM,
) -> float:
    """Transit time from ``r0`` to 1 AU under the drag-based model.

    Uses the analytic DBM solution of Vršnak et al. (2013, Sol. Phys. 285,
    295) for ``dv/dt = -gamma |v - w| (v - w)``::

        v(t) = w + (v0 - w) / (1 + gamma |v0 - w| t)
        r(t) = r0 + w t + sign(v0 - w) * (1/gamma) * ln(1 + gamma |v0 - w| t)

    and inverts ``r(T) = 1 AU`` for ``T`` by bisection (``r`` is strictly
    increasing since ``v(t) > min(v0, w) > 0``).

    Args:
        speed_km_s: CME speed at ``r0`` (for DONKI records: at 21.5 Rs), km/s.
        gamma_per_km: Drag parameter, km^-1. Documented range 0.2e-7..2.0e-7
            (Vršnak et al. 2013); values outside a small guard band raise.
        wind_km_s: Ambient solar-wind speed, km/s.
        r0_km: Initial heliocentric distance, km (default 21.5 Rs, the DONKI
            CMEAnalysis reference distance).

    Returns:
        Transit time to 1 AU in hours (measured from the epoch of ``r0``).

    Raises:
        ValueError: On non-finite/out-of-domain speed, gamma, wind, or r0.
    """
    lo_v, hi_v = _DBM_SPEED_DOMAIN_KM_S
    if not math.isfinite(speed_km_s) or not lo_v <= speed_km_s <= hi_v:
        raise ValueError(f"CME speed {speed_km_s!r} km/s outside DBM domain [{lo_v}, {hi_v}] km/s.")
    lo_g, hi_g = _DBM_GAMMA_HARD_BOUNDS
    if not math.isfinite(gamma_per_km) or not lo_g <= gamma_per_km <= hi_g:
        raise ValueError(
            f"DBM gamma {gamma_per_km!r} km^-1 outside accepted bounds "
            f"[{lo_g:.1e}, {hi_g:.1e}] km^-1 (documented range "
            f"{DBM_GAMMA_RANGE_PER_KM[0]:.1e}..{DBM_GAMMA_RANGE_PER_KM[1]:.1e}; "
            "Vršnak et al. 2013)."
        )
    lo_w, hi_w = _DBM_WIND_HARD_BOUNDS
    if not math.isfinite(wind_km_s) or not lo_w <= wind_km_s <= hi_w:
        raise ValueError(
            f"Solar-wind speed {wind_km_s!r} km/s outside accepted bounds "
            f"[{lo_w}, {hi_w}] km/s."
        )
    if not math.isfinite(r0_km) or not R_SUN_KM <= r0_km < 0.5 * AU_KM:
        raise ValueError(f"Initial distance r0 {r0_km!r} km outside [1 Rs, 0.5 AU).")

    target_km = AU_KM
    dv = speed_km_s - wind_km_s

    if abs(dv) < 1e-9:
        return (target_km - r0_km) / wind_km_s / 3600.0

    sign = 1.0 if dv > 0 else -1.0
    abs_dv = abs(dv)

    def r_of_t(t_s: float) -> float:
        return (
            r0_km + wind_km_s * t_s + sign / gamma_per_km * math.log1p(gamma_per_km * abs_dv * t_s)
        )

    # Bracket the root: r(0) = r0 < 1 AU; the asymptotic speed is w, so
    # 2x the slowest-plausible ballistic time always overshoots.
    t_lo = 0.0
    t_hi = 2.0 * (target_km - r0_km) / min(speed_km_s, wind_km_s)
    while r_of_t(t_hi) < target_km:  # pragma: no cover - defensive
        t_hi *= 2.0

    for _ in range(200):
        t_mid = 0.5 * (t_lo + t_hi)
        if r_of_t(t_mid) < target_km:
            t_lo = t_mid
        else:
            t_hi = t_mid
        if t_hi - t_lo < 1.0:  # 1 s precision
            break
    return 0.5 * (t_lo + t_hi) / 3600.0


def dbm_speed_at_1au_km_s(
    speed_km_s: float,
    gamma_per_km: float = 1.0e-7,
    wind_km_s: float = 400.0,
    r0_km: float = DONKI_R0_KM,
) -> float:
    """CME speed on arrival at 1 AU under the DBM (Vršnak et al. 2013).

    Args:
        speed_km_s: CME speed at ``r0``, km/s.
        gamma_per_km: Drag parameter, km^-1.
        wind_km_s: Ambient solar-wind speed, km/s.
        r0_km: Initial heliocentric distance, km.

    Returns:
        Arrival speed ``v(T)`` in km/s where ``T`` solves ``r(T) = 1 AU``.
    """
    t_s = dbm_transit_time_hours(speed_km_s, gamma_per_km, wind_km_s, r0_km) * 3600.0
    dv = speed_km_s - wind_km_s
    return wind_km_s + dv / (1.0 + gamma_per_km * abs(dv) * t_s)


def _parse_donki_time(value: str) -> datetime:
    """Parse a DONKI timestamp such as ``2024-05-08T09:30Z`` to aware UTC."""
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class CMEKinematics:
    """Kinematic inputs for a single CME arrival prediction.

    Attributes:
        speed_km_s: Reconstructed CME speed at ``time_21_5``, km/s.
        latitude_deg: Apex direction Stonyhurst latitude, degrees.
        longitude_deg: Apex direction Stonyhurst longitude, degrees
            (Earth at 0°; west positive per DONKI convention).
        half_angle_deg: Cone half-angle, degrees.
        time_21_5: UTC time the CME front crossed 21.5 solar radii.
        cme_id: DONKI ``associatedCMEID`` or other identifier.
    """

    speed_km_s: float
    latitude_deg: float
    longitude_deg: float
    half_angle_deg: float
    time_21_5: datetime
    cme_id: str = ""


@dataclass
class CMEArrivalPrediction:
    """Arrival-time window and Earth-directedness assessment for one CME.

    Attributes:
        cme_id: Identifier of the analysed CME.
        earth_directed: Whether the Sun-Earth line lies inside the CME cone.
        directedness: One of ``head_on`` / ``flank`` / ``unlikely`` / ``miss``.
        angular_separation_deg: Apex-direction offset from the Sun-Earth line.
        half_angle_deg: Cone half-angle used for the geometry test.
        earliest_arrival_hours: Hours after ``time_21_5`` (DBM/ESA minimum).
        latest_arrival_hours: Hours after ``time_21_5`` (DBM/ESA maximum).
        most_probable_arrival_hours: Median of the model ensemble, hours.
        earliest_arrival: UTC datetime of the earliest arrival.
        latest_arrival: UTC datetime of the latest arrival.
        most_probable_arrival: UTC datetime of the most probable arrival.
        arrival_speed_km_s: DBM arrival speed at 1 AU (typical parameters).
        model_spread_hours: ``latest - earliest`` in hours.
        confidence: Heuristic in [0, 1]: model agreement (spread vs. 72 h)
            scaled by the impact-geometry class weight. Documented heuristic,
            not a calibrated probability.
        model_predictions_hours: Per-model transit predictions (hours after
            ``time_21_5``) keyed by model label.
    """

    cme_id: str
    earth_directed: bool
    directedness: str
    angular_separation_deg: float
    half_angle_deg: float
    earliest_arrival_hours: float
    latest_arrival_hours: float
    most_probable_arrival_hours: float
    earliest_arrival: datetime
    latest_arrival: datetime
    most_probable_arrival: datetime
    arrival_speed_km_s: float
    model_spread_hours: float
    confidence: float
    model_predictions_hours: dict[str, float] = field(default_factory=dict)


#: Geometry-class weights applied to the model-agreement confidence.
#: Heuristic encoding that flank encounters are less certain to produce a
#: geoeffective impact than head-on hits (cf. the cone-geometry hit/miss
#: statistics in Dumbović et al. 2018).
_GEOMETRY_WEIGHTS: dict[str, float] = {
    "head_on": 1.0,
    "flank": 0.7,
    "unlikely": 0.3,
    "miss": 0.05,
}
#: Margin (deg) beyond the nominal cone half-angle within which an encounter
#: is "unlikely" rather than a clean miss — allows for CME expansion and
#: deflection uncertainty in the cone reconstruction.
_DIRECTEDNESS_MARGIN_DEG: float = 15.0


class CMEArrivalDetector:
    """Predicts Earth arrival windows for CMEs from DONKI-style kinematics.

    Combines the Gopalswamy et al. (2001) empirical shock-arrival model with
    a Vršnak et al. (2013) drag-based-model ensemble (documented gamma and
    solar-wind ranges) into an earliest / latest / most-probable arrival
    window, plus a cone-geometry Earth-directedness assessment.

    Example:
        >>> detector = CMEArrivalDetector()
        >>> prediction = detector.predict_from_donki(donki_cme_analysis_record)
        >>> prediction.earth_directed, prediction.most_probable_arrival
    """

    #: Fields a DONKI CMEAnalysis record must provide (fail-loud contract).
    REQUIRED_DONKI_FIELDS: tuple[str, ...] = (
        "speed",
        "latitude",
        "longitude",
        "halfAngle",
        "time21_5",
    )

    def __init__(
        self,
        gamma_range_per_km: tuple[float, float] = DBM_GAMMA_RANGE_PER_KM,
        wind_range_km_s: tuple[float, float] = DBM_WIND_RANGE_KM_S,
        gamma_typical_per_km: float = 1.0e-7,
        wind_typical_km_s: float = 400.0,
    ) -> None:
        """Initialize the detector.

        Args:
            gamma_range_per_km: (min, max) DBM drag parameter for the
                ensemble window, km^-1.
            wind_range_km_s: (min, max) ambient solar-wind speed for the
                ensemble window, km/s.
            gamma_typical_per_km: Central DBM gamma, km^-1.
            wind_typical_km_s: Central ambient wind speed, km/s.

        Raises:
            ValueError: If a range is inverted or a central value falls
                outside its range.
        """
        if gamma_range_per_km[0] > gamma_range_per_km[1]:
            raise ValueError(f"Inverted gamma range: {gamma_range_per_km!r}")
        if wind_range_km_s[0] > wind_range_km_s[1]:
            raise ValueError(f"Inverted wind range: {wind_range_km_s!r}")
        if not gamma_range_per_km[0] <= gamma_typical_per_km <= gamma_range_per_km[1]:
            raise ValueError(
                f"Typical gamma {gamma_typical_per_km!r} outside range {gamma_range_per_km!r}"
            )
        if not wind_range_km_s[0] <= wind_typical_km_s <= wind_range_km_s[1]:
            raise ValueError(
                f"Typical wind {wind_typical_km_s!r} outside range {wind_range_km_s!r}"
            )
        self.gamma_range_per_km = gamma_range_per_km
        self.wind_range_km_s = wind_range_km_s
        self.gamma_typical_per_km = gamma_typical_per_km
        self.wind_typical_km_s = wind_typical_km_s
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def angular_separation_deg(latitude_deg: float, longitude_deg: float) -> float:
        """Great-circle angle between the CME apex direction and Sun-Earth line.

        Earth sits at Stonyhurst (0°, 0°), so the separation is
        ``arccos(cos(lat) * cos(lon))``.

        Args:
            latitude_deg: Apex Stonyhurst latitude, degrees.
            longitude_deg: Apex Stonyhurst longitude, degrees.

        Returns:
            Angular separation in degrees, in [0, 180].
        """
        lat = math.radians(latitude_deg)
        lon = math.radians(longitude_deg)
        cos_sep = max(-1.0, min(1.0, math.cos(lat) * math.cos(lon)))
        return math.degrees(math.acos(cos_sep))

    @classmethod
    def classify_directedness(
        cls, latitude_deg: float, longitude_deg: float, half_angle_deg: float
    ) -> tuple[str, float]:
        """Classify Earth-directedness from cone geometry.

        Args:
            latitude_deg: Apex Stonyhurst latitude, degrees.
            longitude_deg: Apex Stonyhurst longitude, degrees.
            half_angle_deg: Cone half-angle, degrees.

        Returns:
            Tuple of (class, angular separation deg) where class is
            ``head_on`` (separation <= half-angle / 2), ``flank``
            (<= half-angle), ``unlikely`` (<= half-angle + 15° margin),
            or ``miss``.
        """
        sep = cls.angular_separation_deg(latitude_deg, longitude_deg)
        if sep <= 0.5 * half_angle_deg:
            return "head_on", sep
        if sep <= half_angle_deg:
            return "flank", sep
        if sep <= half_angle_deg + _DIRECTEDNESS_MARGIN_DEG:
            return "unlikely", sep
        return "miss", sep

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, kinematics: CMEKinematics) -> CMEArrivalPrediction:
        """Predict the Earth arrival window for one CME.

        Args:
            kinematics: Validated CME kinematic inputs.

        Returns:
            Arrival prediction with window, geometry, and confidence.

        Raises:
            ValueError: On non-finite or out-of-domain kinematics.
        """
        kin = kinematics
        if not math.isfinite(kin.speed_km_s) or kin.speed_km_s <= 0:
            raise ValueError(f"CME speed must be positive; got {kin.speed_km_s!r}.")
        if not math.isfinite(kin.latitude_deg) or abs(kin.latitude_deg) > 90.0:
            raise ValueError(f"CME latitude {kin.latitude_deg!r} outside [-90, 90].")
        if not math.isfinite(kin.longitude_deg) or abs(kin.longitude_deg) > 180.0:
            raise ValueError(f"CME longitude {kin.longitude_deg!r} outside [-180, 180].")
        if not math.isfinite(kin.half_angle_deg) or not 0.0 < kin.half_angle_deg <= 180.0:
            raise ValueError(f"CME half-angle {kin.half_angle_deg!r} outside (0, 180].")
        if kin.time_21_5.tzinfo is None:
            raise ValueError("time_21_5 must be timezone-aware (UTC).")

        directedness, separation = self.classify_directedness(
            kin.latitude_deg, kin.longitude_deg, kin.half_angle_deg
        )
        earth_directed = separation <= kin.half_angle_deg

        predictions: dict[str, float] = {}

        # DBM ensemble corners across documented gamma / wind ranges,
        # measured from time_21_5 with r0 = 21.5 Rs.
        for gamma, wind in product(self.gamma_range_per_km, self.wind_range_km_s):
            label = f"dbm_gamma={gamma:.1e}_w={wind:.0f}"
            predictions[label] = dbm_transit_time_hours(
                kin.speed_km_s, gamma_per_km=gamma, wind_km_s=wind
            )
        predictions["dbm_typical"] = dbm_transit_time_hours(
            kin.speed_km_s,
            gamma_per_km=self.gamma_typical_per_km,
            wind_km_s=self.wind_typical_km_s,
        )

        # ESA model transit is Sun-to-Earth; convert onto the time_21_5
        # clock by subtracting the (constant-speed approximated) travel
        # time from 1 Rs to 21.5 Rs. The empirical model does not resolve
        # sub-21.5 Rs acceleration, so this offset is approximate and
        # documented as such.
        esa_transit_h = gopalswamy_transit_time_hours(kin.speed_km_s)
        offset_h = (DONKI_R0_KM - R_SUN_KM) / kin.speed_km_s / 3600.0
        predictions["gopalswamy_esa"] = esa_transit_h - offset_h

        earliest_h = min(predictions.values())
        latest_h = max(predictions.values())
        most_probable_h = statistics.median(predictions.values())
        spread_h = latest_h - earliest_h

        model_agreement = max(0.0, min(1.0, 1.0 - spread_h / 72.0))
        confidence = model_agreement * _GEOMETRY_WEIGHTS[directedness]

        arrival_speed = dbm_speed_at_1au_km_s(
            kin.speed_km_s,
            gamma_per_km=self.gamma_typical_per_km,
            wind_km_s=self.wind_typical_km_s,
        )

        prediction = CMEArrivalPrediction(
            cme_id=kin.cme_id,
            earth_directed=earth_directed,
            directedness=directedness,
            angular_separation_deg=separation,
            half_angle_deg=kin.half_angle_deg,
            earliest_arrival_hours=earliest_h,
            latest_arrival_hours=latest_h,
            most_probable_arrival_hours=most_probable_h,
            earliest_arrival=kin.time_21_5 + timedelta(hours=earliest_h),
            latest_arrival=kin.time_21_5 + timedelta(hours=latest_h),
            most_probable_arrival=kin.time_21_5 + timedelta(hours=most_probable_h),
            arrival_speed_km_s=arrival_speed,
            model_spread_hours=spread_h,
            confidence=confidence,
            model_predictions_hours=predictions,
        )
        self.logger.info(
            "CME %s: %s (sep %.1f°, half-angle %.1f°), arrival %.1f..%.1f h "
            "(most probable %.1f h), confidence %.2f",
            kin.cme_id or "<unnamed>",
            directedness,
            separation,
            kin.half_angle_deg,
            earliest_h,
            latest_h,
            most_probable_h,
            confidence,
        )
        return prediction

    def predict_from_donki(self, record: dict[str, Any]) -> CMEArrivalPrediction:
        """Predict arrival from a raw DONKI ``CMEAnalysis`` record.

        Args:
            record: Parsed DONKI CMEAnalysis JSON object. Must contain
                non-null ``speed``, ``latitude``, ``longitude``,
                ``halfAngle`` and ``time21_5``.

        Returns:
            Arrival prediction.

        Raises:
            ValueError: If any required kinematic field is missing or null
                (fail-loud; nothing is defaulted), or out of domain.
        """
        missing = [key for key in self.REQUIRED_DONKI_FIELDS if record.get(key) is None]
        if missing:
            raise ValueError(
                "DONKI CMEAnalysis record is missing required kinematics "
                f"{missing}; refusing to predict from incomplete data "
                f"(record id: {record.get('associatedCMEID', '<unknown>')})."
            )
        kinematics = CMEKinematics(
            speed_km_s=float(record["speed"]),
            latitude_deg=float(record["latitude"]),
            longitude_deg=float(record["longitude"]),
            half_angle_deg=float(record["halfAngle"]),
            time_21_5=_parse_donki_time(str(record["time21_5"])),
            cme_id=str(record.get("associatedCMEID", "")),
        )
        return self.predict(kinematics)


__all__ = [
    "AU_KM",
    "DBM_GAMMA_RANGE_PER_KM",
    "DBM_WIND_RANGE_KM_S",
    "DONKI_R0_KM",
    "R_SUN_KM",
    "CMEArrivalDetector",
    "CMEArrivalPrediction",
    "CMEKinematics",
    "dbm_speed_at_1au_km_s",
    "dbm_transit_time_hours",
    "gopalswamy_transit_time_hours",
]
