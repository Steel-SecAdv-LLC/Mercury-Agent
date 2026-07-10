# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ionospheric scintillation / GNSS-degradation detection.

Honest scope, two strictly separated paths:

1. **Measurement path** — real S4 and sigma-phi computation from supplied
   receiver samples:

   * Amplitude scintillation index (Yeh & Liu 1982, Proc. IEEE 70, 324)::

         S4^2 = (<I^2> - <I>^2) / <I>^2

     where ``I`` is detrended signal intensity (power).
   * Phase scintillation index ``sigma_phi``: standard deviation of the
     detrended carrier phase. Detrending follows the GISTM receiver
     convention — 6th-order Butterworth high-pass at 0.1 Hz (Van
     Dierendonck, Klobuchar & Hua 1993, Proc. ION GPS-93; Yeh & Liu 1982).

   Classification against the standard weak / moderate / strong thresholds
   at 0.3 and 0.6 (S4 dimensionless; sigma-phi in radians), as used
   throughout the GNSS scintillation literature (e.g. Van Dierendonck et
   al. 1993; Basu et al. 2002, J. Atmos. Sol.-Terr. Phys. 64).

2. **Climatological-risk path** — when only geophysical indices are
   available (no receiver samples), a risk level driven by REAL proxies:

   * **Auroral / high-latitude risk**: phase scintillation tracks the
     auroral oval, whose equatorward boundary moves with Kp; we use the
     Gussenhoven, Hardy & Heinemann (1983, JGR 88, 5692) linear boundary
     model ``lambda_b ≈ 67.5° - 2.1° * Kp`` (corrected geomagnetic
     latitude, midnight sector). High-latitude scintillation climatology:
     Basu et al. (2002); Spogli et al. (2009, Ann. Geophys. 27).
   * **Post-sunset equatorial risk**: equatorial plasma-bubble
     scintillation peaks between local sunset and ~01 LT within ~20° of
     the magnetic equator, maximising around the equinoxes (Aarons 1982,
     Space Sci. Rev. 32, 169; Basu et al. 2002).

   Output from this path is labelled ``risk_basis="climatological"`` with
   ``is_measurement=False`` — it is an occurrence-likelihood assessment,
   NEVER a scintillation measurement, and no S4/sigma-phi value is ever
   fabricated from indices.

Physics/statistics only — no neural network.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

#: Standard scintillation-intensity class boundaries (S4 dimensionless,
#: sigma-phi radians): weak < 0.3, moderate 0.3-0.6, strong > 0.6.
WEAK_THRESHOLD: float = 0.3
STRONG_THRESHOLD: float = 0.6

#: GISTM-convention phase detrending high-pass cutoff (Hz) and filter order
#: (Van Dierendonck et al. 1993).
PHASE_DETREND_CUTOFF_HZ: float = 0.1
PHASE_DETREND_FILTER_ORDER: int = 6

#: Gussenhoven et al. (1983) equatorward auroral-boundary model
#: coefficients: lambda_b = 67.5 - 2.1 * Kp (deg CGM latitude).
_AURORAL_BOUNDARY_INTERCEPT_DEG: float = 67.5
_AURORAL_BOUNDARY_SLOPE_DEG_PER_KP: float = 2.1

#: Equatorial scintillation belt half-width (deg magnetic latitude): the
#: anomaly-crest region within ~+/-20 deg of the magnetic equator
#: (Aarons 1982; Basu et al. 2002).
_EQUATORIAL_BELT_DEG: float = 20.0
#: Post-sunset local-time window (hours) for equatorial plasma bubbles.
_EQUATORIAL_LT_START_H: float = 19.0
_EQUATORIAL_LT_END_H: float = 1.0
#: Equinox months in which equatorial scintillation climatologically peaks.
_EQUINOX_MONTHS: frozenset[int] = frozenset({3, 4, 9, 10})


def compute_s4(intensity: np.ndarray[Any, Any]) -> float:
    """Compute the amplitude scintillation index S4 from intensity samples.

    Standard definition (Yeh & Liu 1982)::

        S4 = sqrt((<I^2> - <I>^2) / <I>^2)

    Args:
        intensity: 1-D array of signal-intensity (power) samples over the
            evaluation interval (conventionally 60 s at >= 50 Hz).

    Returns:
        S4 index (dimensionless, >= 0).

    Raises:
        ValueError: On empty input, fewer than 2 samples, non-finite or
            negative samples, or zero mean intensity.
    """
    arr = np.asarray(intensity, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError(f"S4 needs >= 2 intensity samples; got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Intensity samples contain non-finite values; refusing to compute S4.")
    if np.any(arr < 0.0):
        raise ValueError("Intensity (signal power) samples must be non-negative.")
    mean_i = float(np.mean(arr))
    if mean_i <= 0.0:
        raise ValueError("Mean intensity is zero; S4 is undefined (no signal).")
    variance = float(np.mean(arr**2)) - mean_i**2
    # Numerical guard: variance can be a tiny negative number for a
    # constant signal; clamp at zero (exact-arithmetic value).
    return math.sqrt(max(0.0, variance) / mean_i**2)


def compute_sigma_phi(
    phase_rad: np.ndarray[Any, Any],
    sample_rate_hz: float,
    cutoff_hz: float = PHASE_DETREND_CUTOFF_HZ,
) -> float:
    """Compute the phase scintillation index sigma-phi (radians).

    Standard deviation of the high-pass detrended carrier phase, using the
    GISTM convention: 6th-order Butterworth high-pass at 0.1 Hz (Van
    Dierendonck et al. 1993). Detrending removes receiver-clock and
    geometric Doppler drift so only ionospheric phase fluctuations remain.

    Args:
        phase_rad: 1-D array of carrier-phase samples, radians.
        sample_rate_hz: Sampling rate, Hz. Must exceed twice the cutoff.
        cutoff_hz: High-pass cutoff frequency, Hz (default 0.1).

    Returns:
        sigma-phi in radians.

    Raises:
        ValueError: On too-short input, non-finite samples, or an invalid
            sample-rate / cutoff combination.
    """
    arr = np.asarray(phase_rad, dtype=np.float64).ravel()
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive and finite; got {sample_rate_hz!r}.")
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive and finite; got {cutoff_hz!r}.")
    if cutoff_hz >= sample_rate_hz / 2.0:
        raise ValueError(
            f"High-pass cutoff {cutoff_hz} Hz is not below the Nyquist rate "
            f"({sample_rate_hz / 2.0} Hz)."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Phase samples contain non-finite values; refusing to compute sigma-phi.")
    # filtfilt needs padding room: default padlen = 3 * (2*order + 1).
    min_samples = 3 * (2 * PHASE_DETREND_FILTER_ORDER + 1) + 1
    if arr.size < min_samples:
        raise ValueError(
            f"sigma-phi needs >= {min_samples} phase samples for the "
            f"{PHASE_DETREND_FILTER_ORDER}th-order zero-phase detrend filter; got {arr.size}."
        )
    b, a = butter(PHASE_DETREND_FILTER_ORDER, cutoff_hz, btype="highpass", fs=sample_rate_hz)
    detrended = filtfilt(b, a, arr)
    return float(np.std(detrended))


def classify_scintillation(index_value: float) -> str:
    """Classify a scintillation index against the standard 0.3 / 0.6 tiers.

    Applies to S4 (dimensionless) and sigma-phi (radians) alike, per the
    conventional weak / moderate / strong bands (Basu et al. 2002).

    Args:
        index_value: S4 or sigma-phi value.

    Returns:
        ``"weak"`` (< 0.3), ``"moderate"`` (0.3-0.6), or ``"strong"``
        (> 0.6).

    Raises:
        ValueError: On negative or non-finite input.
    """
    if not math.isfinite(index_value) or index_value < 0.0:
        raise ValueError(f"Scintillation index must be finite and >= 0; got {index_value!r}.")
    if index_value > STRONG_THRESHOLD:
        return "strong"
    if index_value >= WEAK_THRESHOLD:
        return "moderate"
    return "weak"


@dataclass
class ScintillationMeasurement:
    """A real scintillation measurement from receiver samples.

    Attributes:
        s4: Amplitude scintillation index (None if intensity not supplied).
        sigma_phi_rad: Phase scintillation index, radians (None if phase
            not supplied).
        amplitude_class: weak/moderate/strong for S4 (None if no S4).
        phase_class: weak/moderate/strong for sigma-phi (None if no phase).
        gnss_degradation: Worst of the two classes — the headline GNSS
            impact tier.
        n_intensity_samples: Sample count used for S4.
        n_phase_samples: Sample count used for sigma-phi.
        is_measurement: Always True on this path.
    """

    s4: float | None
    sigma_phi_rad: float | None
    amplitude_class: str | None
    phase_class: str | None
    gnss_degradation: str
    n_intensity_samples: int = 0
    n_phase_samples: int = 0
    is_measurement: bool = True


@dataclass
class ScintillationRisk:
    """A climatological scintillation-occurrence risk (NOT a measurement).

    Attributes:
        risk_level: ``low`` / ``moderate`` / ``high`` occurrence likelihood.
        risk_basis: Always ``"climatological"`` — index/geometry driven.
        is_measurement: Always False; no S4/sigma-phi value exists here.
        factors: Human-readable list of the climatology terms that fired.
        kp: Kp index used.
        magnetic_latitude_deg: Magnetic latitude assessed.
        local_time_hours: Local solar time assessed.
    """

    risk_level: str
    risk_basis: str
    is_measurement: bool
    factors: list[str] = field(default_factory=list)
    kp: float = 0.0
    magnetic_latitude_deg: float = 0.0
    local_time_hours: float = 0.0


class IonosphericScintillationDetector:
    """S4 / sigma-phi measurement plus climatological GNSS-degradation risk.

    The measurement path computes real indices from supplied receiver
    samples (Yeh & Liu 1982 definitions, GISTM detrending). The risk path
    maps real geophysical proxies (Kp, magnetic latitude, local time,
    season) onto an occurrence likelihood using published climatology, and
    is always labelled as climatological — never presented as a
    measurement. No scintillation value is ever fabricated.
    """

    def __init__(self) -> None:
        """Initialize the detector."""
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Measurement path
    # ------------------------------------------------------------------

    def measure(
        self,
        intensity: np.ndarray[Any, Any] | None = None,
        phase_rad: np.ndarray[Any, Any] | None = None,
        sample_rate_hz: float | None = None,
    ) -> ScintillationMeasurement:
        """Compute S4 and/or sigma-phi from real receiver samples.

        Args:
            intensity: Signal-intensity (power) samples for S4.
            phase_rad: Carrier-phase samples (radians) for sigma-phi.
            sample_rate_hz: Sampling rate for the phase series; REQUIRED
                when ``phase_rad`` is given (the detrend filter is defined
                in absolute frequency).

        Returns:
            Measurement with per-index classes and the worst-case
            ``gnss_degradation`` tier.

        Raises:
            ValueError: If neither input is supplied, if phase is supplied
                without a sample rate, or if any underlying computation
                rejects the samples.
        """
        if intensity is None and phase_rad is None:
            raise ValueError(
                "measure() requires intensity and/or phase samples; refusing "
                "to emit a scintillation measurement from no data."
            )

        s4: float | None = None
        amplitude_class: str | None = None
        n_intensity = 0
        if intensity is not None:
            s4 = compute_s4(intensity)
            amplitude_class = classify_scintillation(s4)
            n_intensity = int(np.asarray(intensity).size)

        sigma_phi: float | None = None
        phase_class: str | None = None
        n_phase = 0
        if phase_rad is not None:
            if sample_rate_hz is None:
                raise ValueError(
                    "sample_rate_hz is required with phase samples: the 0.1 Hz "
                    "GISTM detrend filter is undefined without it."
                )
            sigma_phi = compute_sigma_phi(phase_rad, sample_rate_hz)
            phase_class = classify_scintillation(sigma_phi)
            n_phase = int(np.asarray(phase_rad).size)

        order = {"weak": 0, "moderate": 1, "strong": 2}
        classes = [c for c in (amplitude_class, phase_class) if c is not None]
        headline = max(classes, key=order.__getitem__)

        measurement = ScintillationMeasurement(
            s4=s4,
            sigma_phi_rad=sigma_phi,
            amplitude_class=amplitude_class,
            phase_class=phase_class,
            gnss_degradation=headline,
            n_intensity_samples=n_intensity,
            n_phase_samples=n_phase,
        )
        self.logger.info(
            "Scintillation measurement: S4=%s (%s), sigma_phi=%s rad (%s) -> %s",
            f"{s4:.3f}" if s4 is not None else "n/a",
            amplitude_class or "n/a",
            f"{sigma_phi:.3f}" if sigma_phi is not None else "n/a",
            phase_class or "n/a",
            headline,
        )
        return measurement

    # ------------------------------------------------------------------
    # Climatological-risk path
    # ------------------------------------------------------------------

    @staticmethod
    def auroral_boundary_deg(kp: float) -> float:
        """Equatorward auroral-oval boundary latitude for a given Kp.

        Gussenhoven et al. (1983) midnight-sector linear model:
        ``lambda_b = 67.5 - 2.1 * Kp`` degrees corrected geomagnetic
        latitude.

        Args:
            kp: Planetary Kp index in [0, 9].

        Returns:
            Boundary latitude in degrees.

        Raises:
            ValueError: If Kp is outside [0, 9].
        """
        if not math.isfinite(kp) or not 0.0 <= kp <= 9.0:
            raise ValueError(f"Kp must be within [0, 9]; got {kp!r}.")
        return _AURORAL_BOUNDARY_INTERCEPT_DEG - _AURORAL_BOUNDARY_SLOPE_DEG_PER_KP * kp

    def climatological_risk(
        self,
        kp: float,
        magnetic_latitude_deg: float,
        local_time_hours: float,
        month: int | None = None,
    ) -> ScintillationRisk:
        """Assess scintillation-occurrence risk from real geophysical proxies.

        This is a climatological likelihood — explicitly NOT a measurement
        (``is_measurement=False``); no S4/sigma-phi value is produced.

        Logic (citations in the module docstring):

        * High latitude: if |mlat| is poleward of the Kp-driven auroral
          boundary (Gussenhoven et al. 1983), risk is ``high`` for Kp >= 5
          (storm-time oval) else ``moderate``.
        * Equatorial belt (|mlat| <= 20°): ``moderate`` risk inside the
          post-sunset 19-01 LT plasma-bubble window (Aarons 1982), raised
          to ``high`` in equinox months (scintillation season; Basu et
          al. 2002).
        * Mid latitude: ``low`` unless Kp >= 7, when storm-enhanced
          density gradients justify ``moderate`` (Basu et al. 2005,
          GRL 32, L12S05).

        Args:
            kp: Planetary Kp index in [0, 9].
            magnetic_latitude_deg: Magnetic latitude of the assessed
                location, degrees in [-90, 90].
            local_time_hours: Local solar time in [0, 24).
            month: Optional calendar month (1-12) for the equinox-season
                term.

        Returns:
            Climatological risk assessment.

        Raises:
            ValueError: On out-of-range Kp, latitude, local time, or month.
        """
        if not math.isfinite(magnetic_latitude_deg) or abs(magnetic_latitude_deg) > 90.0:
            raise ValueError(f"Magnetic latitude {magnetic_latitude_deg!r} outside [-90, 90].")
        if not math.isfinite(local_time_hours) or not 0.0 <= local_time_hours < 24.0:
            raise ValueError(f"Local time {local_time_hours!r} outside [0, 24).")
        if month is not None and month not in range(1, 13):
            raise ValueError(f"Month {month!r} outside 1..12.")
        boundary = self.auroral_boundary_deg(kp)  # validates kp

        abs_mlat = abs(magnetic_latitude_deg)
        factors: list[str] = []
        risk = "low"

        if abs_mlat >= boundary:
            risk = "high" if kp >= 5.0 else "moderate"
            factors.append(
                f"|mlat| {abs_mlat:.1f}° poleward of the Kp={kp:.1f} auroral "
                f"boundary {boundary:.1f}° (Gussenhoven et al. 1983)"
            )
            if kp >= 5.0:
                factors.append("storm-level Kp >= 5: expanded oval, phase-scintillation regime")
        elif abs_mlat <= _EQUATORIAL_BELT_DEG:
            in_window = (
                local_time_hours >= _EQUATORIAL_LT_START_H
                or local_time_hours < _EQUATORIAL_LT_END_H
            )
            if in_window:
                risk = "moderate"
                factors.append(
                    f"equatorial belt (|mlat| <= {_EQUATORIAL_BELT_DEG:.0f}°) in the "
                    f"post-sunset {_EQUATORIAL_LT_START_H:.0f}-"
                    f"{_EQUATORIAL_LT_END_H:.0f} LT plasma-bubble window (Aarons 1982)"
                )
                if month is not None and month in _EQUINOX_MONTHS:
                    risk = "high"
                    factors.append(
                        "equinox month: seasonal scintillation maximum (Basu et al. 2002)"
                    )
        elif kp >= 7.0:
            risk = "moderate"
            factors.append(
                "mid-latitude with Kp >= 7: storm-enhanced-density gradient "
                "risk (Basu et al. 2005)"
            )

        if not factors:
            factors.append("no climatological risk term fired")

        assessment = ScintillationRisk(
            risk_level=risk,
            risk_basis="climatological",
            is_measurement=False,
            factors=factors,
            kp=float(kp),
            magnetic_latitude_deg=float(magnetic_latitude_deg),
            local_time_hours=float(local_time_hours),
        )
        # Log the magnetic-latitude *regime* rather than the precise coordinate.
        # The regime (equatorial / mid-latitude / auroral) is what drives
        # scintillation climatology, and keeping a raw location out of routine
        # logs is sound log hygiene (CWE-532). The exact magnetic latitude
        # remains available on the returned ``ScintillationRisk``.
        if abs_mlat >= boundary:
            mlat_regime = "auroral"
        elif abs_mlat <= _EQUATORIAL_BELT_DEG:
            mlat_regime = "equatorial"
        else:
            mlat_regime = "mid-latitude"
        self.logger.info(
            "Climatological scintillation risk: %s (kp=%.1f, mlat_regime=%s, LT=%.1f h)",
            risk,
            kp,
            mlat_regime,
            local_time_hours,
        )
        return assessment


__all__ = [
    "PHASE_DETREND_CUTOFF_HZ",
    "PHASE_DETREND_FILTER_ORDER",
    "STRONG_THRESHOLD",
    "WEAK_THRESHOLD",
    "IonosphericScintillationDetector",
    "ScintillationMeasurement",
    "ScintillationRisk",
    "classify_scintillation",
    "compute_s4",
    "compute_sigma_phi",
]
