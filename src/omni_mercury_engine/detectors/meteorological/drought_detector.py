# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Drought Detector - Standardized Precipitation Index (SPI/SPEI) monitoring.

Implements the literature-standard drought indices on real precipitation
(and optionally temperature) time series:

- **SPI** (Standardized Precipitation Index): McKee, Doesken & Kleist (1993),
  "The relationship of drought frequency and duration to time scales",
  *Proc. 8th Conference on Applied Climatology*, 179-184.  Precipitation is
  aggregated over 1/3/6/12-month windows, fitted to a two-parameter gamma
  distribution via the Thom (1958) maximum-likelihood approximation
  (Thom, H.C.S., 1958: "A note on the gamma distribution", *Mon. Wea. Rev.*
  86, 117-122; see also Edwards & McKee 1997), and transformed to a standard
  normal deviate.  The zero-precipitation probability mass is handled with
  the mixed distribution H(x) = q + (1 - q) * G(x) where q is the empirical
  probability of a zero-total window.

- **SPEI** (Standardized Precipitation-Evapotranspiration Index):
  Vicente-Serrano, Begueria & Lopez-Moreno (2010), "A multiscalar drought
  index sensitive to global warming: the Standardized Precipitation
  Evapotranspiration Index", *J. Climate* 23, 1696-1718.  The climatic water
  balance D = P - PET is aggregated and fitted to the three-parameter
  log-logistic law in its generalized-logistic parameterization via
  L-moments (Hosking & Wallis 1997, Appendix A.7 - the estimator used by
  the reference SPEI implementation, valid for either skew sign), then
  transformed to a standard normal deviate.  PET is computed with the
  Thornthwaite (1948) temperature/latitude method (Thornthwaite, C.W., 1948:
  "An approach toward a rational classification of climate", *Geogr. Rev.*
  38, 55-94), with the Willmott et al. (1985) high-temperature
  parameterization for T > 26.5 degC.  The SPEI branch runs ONLY when
  temperature and latitude inputs are supplied - PET is never fabricated.

- **US Drought Monitor mapping**: SPI thresholds for D0-D4 categories follow
  Svoboda et al. (2002), "The Drought Monitor", *Bull. Amer. Meteor. Soc.*
  83, 1181-1190 (D0: -0.5 to -0.7, D1: -0.8 to -1.2, D2: -1.3 to -1.5,
  D3: -1.6 to -1.9, D4: <= -2.0).

The detector is a pure physics/statistics core: it works untrained, uses no
neural networks, and fails loudly on inadequate input (fewer than 30
aggregated samples per window, all-zero series, non-finite values) rather
than fabricating an index value.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)

#: Minimum number of aggregated samples required to fit a distribution for
#: one SPI/SPEI window.  McKee et al. (1993) recommend ~30 years of monthly
#: data; 30 aggregated values is the hard floor below which the gamma /
#: log-logistic fit is statistically meaningless.
MIN_SAMPLES_PER_WINDOW: int = 30

#: Probability clip applied before the inverse-normal transform.  Bounds the
#: index to roughly +/-4.75, mirroring the bounded lookup tables of the
#: original implementations (an SPI beyond +/-4 is far outside the resolvable
#: range of a 30-120 sample climatology).
_PROB_EPS: float = 1e-6

#: Days per calendar month (non-leap) used by the Thornthwaite computation.
_DAYS_IN_MONTH: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: Mid-month day-of-year (non-leap) used for the solar declination.
_MID_MONTH_DOY: tuple[int, ...] = (15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349)


class DroughtCategory(Enum):
    """US Drought Monitor drought categories (Svoboda et al. 2002)."""

    NONE = "none"
    D0_ABNORMALLY_DRY = "d0_abnormally_dry"
    D1_MODERATE_DROUGHT = "d1_moderate_drought"
    D2_SEVERE_DROUGHT = "d2_severe_drought"
    D3_EXTREME_DROUGHT = "d3_extreme_drought"
    D4_EXCEPTIONAL_DROUGHT = "d4_exceptional_drought"


#: USDM category boundaries on the SPI axis (Svoboda et al. 2002, Table 2).
#: A value at or below the threshold belongs to at least that category.
_USDM_THRESHOLDS: tuple[tuple[float, DroughtCategory], ...] = (
    (-2.0, DroughtCategory.D4_EXCEPTIONAL_DROUGHT),
    (-1.6, DroughtCategory.D3_EXTREME_DROUGHT),
    (-1.3, DroughtCategory.D2_SEVERE_DROUGHT),
    (-0.8, DroughtCategory.D1_MODERATE_DROUGHT),
    (-0.5, DroughtCategory.D0_ABNORMALLY_DRY),
)


@dataclass
class DroughtAssessmentResult:
    """Result of a multi-window drought assessment.

    Attributes:
        drought_detected: True when any window's latest SPI reaches D0 or
            worse (SPI <= -0.5).
        category: Worst (most severe) USDM category across all windows,
            as a :class:`DroughtCategory` value string.
        spi_latest: Latest SPI value per aggregation window (months -> SPI).
        spi_series: Full SPI series per aggregation window.
        spei_latest: Latest SPEI value per window; empty when PET inputs
            were not supplied.
        spei_series: Full SPEI series per window; empty without PET inputs.
        pet_mm: Thornthwaite PET series (mm/month) when computed, else None.
        categories_by_window: USDM category per window from the latest SPI.
        confidence: Fraction of requested windows that produced an index.
        warning_actions: Human-readable advisory strings.
    """

    drought_detected: bool
    category: str
    spi_latest: dict[int, float] = field(default_factory=dict)
    spi_series: dict[int, np.ndarray] = field(default_factory=dict)  # type: ignore[type-arg]
    spei_latest: dict[int, float] = field(default_factory=dict)
    spei_series: dict[int, np.ndarray] = field(default_factory=dict)  # type: ignore[type-arg]
    pet_mm: np.ndarray | None = None  # type: ignore[type-arg]
    categories_by_window: dict[int, str] = field(default_factory=dict)
    confidence: float = 0.0
    warning_actions: list[str] = field(default_factory=list)


def _validate_series(values: np.ndarray, name: str) -> np.ndarray:  # type: ignore[type-arg]
    """Validate a 1-D input series: finite, 1-D, non-empty.

    Args:
        values: Raw input array.
        name: Name used in error messages.

    Returns:
        The validated array as float64.

    Raises:
        ValueError: If the array is empty, not 1-D, or contains
            non-finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s); "
            "clean or gap-fill the series upstream - the index will not be "
            "computed on fabricated values"
        )
    return arr


def fit_gamma_thom(nonzero_values: np.ndarray) -> tuple[float, float]:  # type: ignore[type-arg]
    """Fit a two-parameter gamma distribution via the Thom (1958) approximation.

    The maximum-likelihood approximation used by the operational SPI
    (Edwards & McKee 1997)::

        A     = ln(mean(x)) - mean(ln(x))
        shape = (1 + sqrt(1 + 4A/3)) / (4A)
        scale = mean(x) / shape

    Args:
        nonzero_values: Strictly positive aggregated precipitation values.

    Returns:
        Tuple ``(shape, scale)`` of the fitted gamma distribution.

    Raises:
        ValueError: If any value is non-positive, or the sample is
            degenerate (all values identical, making A -> 0 and the
            shape estimate unbounded).
    """
    x = np.asarray(nonzero_values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot fit gamma distribution to an empty sample")
    if np.any(x <= 0.0):
        raise ValueError("gamma fit requires strictly positive values")

    log_stat = float(np.log(np.mean(x)) - np.mean(np.log(x)))
    if not math.isfinite(log_stat) or log_stat <= 0.0:
        # By Jensen's inequality A >= 0 with equality iff all values are
        # equal; A == 0 makes the Thom shape estimate divergent.
        raise ValueError(
            "degenerate precipitation sample (all aggregated values "
            "identical); gamma parameters are not identifiable"
        )
    shape = (1.0 + math.sqrt(1.0 + 4.0 * log_stat / 3.0)) / (4.0 * log_stat)
    scale = float(np.mean(x)) / shape
    return shape, scale


def _strata_for_windows(
    month_numbers: np.ndarray | None,  # type: ignore[type-arg]
    n_series: int,
    window_months: int,
) -> list[np.ndarray]:  # type: ignore[type-arg]
    """Build index groups over the aggregated series for distribution fitting.

    With ``month_numbers`` supplied, aggregated windows are stratified by
    the calendar month of the window's END - the McKee et al. (1993)
    operational convention (the k-month total ending in July is fitted
    against all k-month totals ending in July).  Without it, all windows
    form a single pooled stratum (documented variant for non-seasonal or
    deseasonalized series).

    Args:
        month_numbers: Optional calendar months (1-12) aligned to the
            ORIGINAL monthly series.
        n_series: Length of the original series.
        window_months: Aggregation window.

    Returns:
        List of index arrays into the aggregated series.

    Raises:
        ValueError: If ``month_numbers`` is malformed.
    """
    n_agg = n_series - window_months + 1
    if month_numbers is None:
        return [np.arange(n_agg)]
    months = np.asarray(month_numbers, dtype=np.int64)
    if months.ndim != 1 or months.size != n_series:
        raise ValueError(
            f"month_numbers must be 1-D with length {n_series}, got shape {months.shape}"
        )
    if np.any((months < 1) | (months > 12)):
        raise ValueError("month_numbers must contain calendar months 1-12")
    end_months = months[window_months - 1 :]
    return [np.where(end_months == m)[0] for m in range(1, 13) if np.any(end_months == m)]


def compute_spi(
    monthly_precip_mm: np.ndarray,  # type: ignore[type-arg]
    window_months: int,
    month_numbers: np.ndarray | None = None,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Compute the Standardized Precipitation Index (McKee et al. 1993).

    Aggregates the monthly series with a trailing moving sum of
    ``window_months``, fits a gamma distribution to the non-zero aggregated
    values (Thom 1958 approximation), applies the mixed-distribution
    correction H(x) = q + (1 - q) G(x) for the zero-precipitation
    probability mass q, and maps H through the inverse standard normal.

    When ``month_numbers`` is supplied, a separate gamma distribution is
    fitted for each calendar month of the window end - the operational
    McKee convention that removes the seasonal cycle.  Without it a single
    pooled fit is used (appropriate for non-seasonal or deseasonalized
    series; documented variant).

    Args:
        monthly_precip_mm: 1-D array of monthly precipitation totals (mm).
            Values must be finite and non-negative.
        window_months: Aggregation window in months (e.g. 1, 3, 6, 12).
        month_numbers: Optional calendar month (1-12) for each entry of
            the original series; enables per-calendar-month fitting.

    Returns:
        1-D array of SPI values of length ``len(series) - window + 1``
        (one value per complete trailing window).

    Raises:
        ValueError: If the series contains negative or non-finite values,
            any fitting stratum has fewer than
            :data:`MIN_SAMPLES_PER_WINDOW` samples, or every aggregated
            value in a stratum is zero (SPI undefined - McKee's gamma fit
            requires at least one wet window).
    """
    precip = _validate_series(monthly_precip_mm, "monthly_precip_mm")
    if np.any(precip < 0.0):
        raise ValueError("monthly_precip_mm contains negative values")
    if window_months < 1:
        raise ValueError(f"window_months must be >= 1, got {window_months}")

    kernel = np.ones(window_months, dtype=np.float64)
    aggregated = np.convolve(precip, kernel, mode="valid")
    spi = np.empty_like(aggregated)

    for stratum in _strata_for_windows(month_numbers, precip.size, window_months):
        values = aggregated[stratum]
        if values.size < MIN_SAMPLES_PER_WINDOW:
            raise ValueError(
                f"SPI-{window_months} needs >= {MIN_SAMPLES_PER_WINDOW} aggregated "
                f"samples per fitting stratum but one stratum has only "
                f"{values.size} (series length {precip.size}); supply a "
                "longer series or omit month_numbers for a pooled fit"
            )
        zero_mask = values <= 0.0
        n_zero = int(np.sum(zero_mask))
        if n_zero == values.size:
            raise ValueError(
                f"SPI-{window_months}: every aggregated window in a stratum "
                "is zero; the gamma distribution (and hence SPI) is "
                "undefined for an all-dry climatology"
            )
        q_zero = n_zero / values.size
        shape, scale = fit_gamma_thom(values[~zero_mask])
        cdf = np.empty_like(values)
        cdf[zero_mask] = q_zero
        cdf[~zero_mask] = q_zero + (1.0 - q_zero) * stats.gamma.cdf(
            values[~zero_mask], a=shape, scale=scale
        )
        spi[stratum] = stats.norm.ppf(np.clip(cdf, _PROB_EPS, 1.0 - _PROB_EPS))
    return spi


def thornthwaite_pet(
    monthly_temp_c: np.ndarray,  # type: ignore[type-arg]
    latitude_deg: float,
    month_numbers: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Compute monthly potential evapotranspiration (Thornthwaite 1948).

    Uses the classical annual heat index formulation::

        i_m = (T_m / 5) ^ 1.514          for months with T_m > 0
        I   = sum of the 12 climatological i_m
        a   = 6.75e-7 I^3 - 7.71e-5 I^2 + 1.7912e-2 I + 0.49239
        PET = 16 (L/12) (N/30) (10 T / I)^a          [mm/month, 0 < T <= 26.5]

    For T > 26.5 degC the Willmott, Rowe & Mintz (1985, *J. Climatol.* 5,
    589-606) parameterization of Thornthwaite's high-temperature table is
    used (-415.85 + 32.24 T - 0.43 T^2, scaled by the same day-length
    factors), and PET = 0 for T <= 0 degC.  Day length L is derived from
    the standard mid-month solar declination
    delta = 0.4093 sin(2 pi J / 365 - 1.405).

    Args:
        monthly_temp_c: 1-D array of monthly mean temperatures (degC).
        latitude_deg: Site latitude in decimal degrees, within [-66.5, 66.5]
            (the day-length formula degenerates poleward of the polar
            circles).
        month_numbers: 1-D integer array (same length) of calendar month
            numbers 1-12 for each entry.

    Returns:
        1-D array of PET values in mm/month.

    Raises:
        ValueError: If inputs are malformed, latitude is out of range, the
            series is shorter than 12 months, or not all 12 calendar months
            are represented (the annual heat index I requires a full
            monthly climatology).
    """
    temp = _validate_series(monthly_temp_c, "monthly_temp_c")
    months = np.asarray(month_numbers, dtype=np.int64)
    if months.shape != temp.shape:
        raise ValueError(
            f"month_numbers shape {months.shape} does not match "
            f"monthly_temp_c shape {temp.shape}"
        )
    if np.any((months < 1) | (months > 12)):
        raise ValueError("month_numbers must contain calendar months 1-12")
    if not -66.5 <= latitude_deg <= 66.5:
        raise ValueError(
            f"latitude_deg {latitude_deg} outside [-66.5, 66.5]; the "
            "Thornthwaite day-length term is undefined poleward of the "
            "polar circles"
        )
    if temp.size < 12 or len(set(months.tolist())) < 12:
        raise ValueError(
            "Thornthwaite PET needs at least one full year covering all 12 "
            "calendar months to form the annual heat index I"
        )

    # Annual heat index from the mean climatology of each calendar month.
    heat_index = 0.0
    for m in range(1, 13):
        t_clim = float(np.mean(temp[months == m]))
        if t_clim > 0.0:
            heat_index += (t_clim / 5.0) ** 1.514
    if heat_index <= 0.0:
        raise ValueError(
            "annual heat index I is zero (all monthly climatological "
            "temperatures <= 0 degC); Thornthwaite PET is undefined"
        )
    exp_a = 6.75e-7 * heat_index**3 - 7.71e-5 * heat_index**2 + 1.7912e-2 * heat_index + 0.49239

    lat_rad = math.radians(latitude_deg)
    pet = np.zeros_like(temp)
    for i in range(temp.size):
        t_i = float(temp[i])
        if t_i <= 0.0:
            continue
        m_idx = int(months[i]) - 1
        doy = _MID_MONTH_DOY[m_idx]
        declination = 0.4093 * math.sin(2.0 * math.pi * doy / 365.0 - 1.405)
        cos_omega = max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(declination)))
        day_length_h = 24.0 / math.pi * math.acos(cos_omega)
        n_days = _DAYS_IN_MONTH[m_idx]
        correction = (day_length_h / 12.0) * (n_days / 30.0)
        if t_i <= 26.5:
            pet_uncorrected = 16.0 * (10.0 * t_i / heat_index) ** exp_a
        else:
            # Willmott et al. (1985) high-temperature branch.
            pet_uncorrected = -415.85 + 32.24 * t_i - 0.43 * t_i * t_i
        pet[i] = max(0.0, pet_uncorrected * correction)
    return pet


def _generalized_logistic_lmom_fit(values: np.ndarray) -> tuple[float, float, float]:  # type: ignore[type-arg]
    """Fit a generalized logistic distribution by L-moments.

    The three-parameter log-logistic of Vicente-Serrano et al. (2010) is
    the generalized logistic (GLO) family in a different parameterization;
    the reference SPEI implementation (the SPEI R package) fits GLO by
    L-moments, which - unlike the raw log-logistic PWM equations - remains
    valid for water-balance strata of either skew sign.  Parameter
    estimators from Hosking & Wallis (1997, *Regional Frequency Analysis*,
    Appendix A.7)::

        kappa = -tau3
        alpha = lambda2 * sin(kappa pi) / (kappa pi)
        xi    = lambda1 - alpha (1/kappa - pi / sin(kappa pi))

    with sample L-moments derived from plotting-position PWMs
    ``b_s = mean(F_i^s x_i)``, ``F_i = (i - 0.35)/n`` (the plotting
    position used by Vicente-Serrano et al. 2010).

    Args:
        values: Aggregated climatic water-balance values (may be negative).

    Returns:
        Tuple ``(xi, alpha, kappa)`` - location, scale, shape.

    Raises:
        ValueError: If the sample L-scale is non-positive or the shape
            estimate falls outside the GLO validity domain |kappa| < 1.
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    plotting_pos = (ranks - 0.35) / n

    b0 = float(np.mean(x))
    b1 = float(np.mean(plotting_pos * x))
    b2 = float(np.mean(plotting_pos**2 * x))

    lambda1 = b0
    lambda2 = 2.0 * b1 - b0
    lambda3 = 6.0 * b2 - 6.0 * b1 + b0
    if lambda2 <= 0.0 or not math.isfinite(lambda2):
        raise ValueError(
            f"sample L-scale {lambda2:.4f} <= 0; the water-balance stratum "
            "is degenerate and no distribution can be fitted"
        )
    tau3 = lambda3 / lambda2
    kappa = -tau3
    if not math.isfinite(kappa) or abs(kappa) >= 1.0:
        raise ValueError(
            f"generalized-logistic shape estimate kappa={kappa:.4f} outside "
            "(-1, 1); the water-balance stratum cannot be fitted - supply a "
            "longer or better-conditioned series"
        )
    if abs(kappa) < 1e-8:
        # Logistic limit (kappa -> 0).
        return lambda1, lambda2, 0.0
    kp = kappa * math.pi
    alpha = lambda2 * math.sin(kp) / kp
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"generalized-logistic scale alpha={alpha:.4f} invalid (must be > 0)")
    xi = lambda1 - alpha * (1.0 / kappa - math.pi / math.sin(kp))
    return xi, alpha, kappa


def _generalized_logistic_cdf(
    values: np.ndarray,  # type: ignore[type-arg]
    xi: float,
    alpha: float,
    kappa: float,
) -> np.ndarray:  # type: ignore[type-arg]
    """Evaluate the generalized logistic CDF (Hosking & Wallis 1997, A.7).

    F(x) = 1 / (1 + exp(-y)) with y = -log(1 - kappa (x - xi)/alpha)/kappa
    for kappa != 0 and y = (x - xi)/alpha in the logistic limit.  Values
    outside the finite support bound (kappa != 0) saturate to 0 or 1.

    Args:
        values: Points of evaluation.
        xi: Location parameter.
        alpha: Scale parameter (> 0).
        kappa: Shape parameter, |kappa| < 1.

    Returns:
        CDF values in [0, 1].
    """
    z = (values - xi) / alpha
    if abs(kappa) < 1e-8:
        y = z
    else:
        arg = 1.0 - kappa * z
        y = np.empty_like(z)
        inside = arg > 0.0
        y[inside] = -np.log(arg[inside]) / kappa
        # Outside the support: kappa > 0 caps the upper tail (F -> 1),
        # kappa < 0 caps the lower tail (F -> 0).
        y[~inside] = np.inf if kappa > 0.0 else -np.inf
    return 1.0 / (1.0 + np.exp(-y))


def compute_spei(
    monthly_precip_mm: np.ndarray,  # type: ignore[type-arg]
    monthly_pet_mm: np.ndarray,  # type: ignore[type-arg]
    window_months: int,
    month_numbers: np.ndarray | None = None,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Compute the SPEI (Vicente-Serrano et al. 2010).

    The climatic water balance D = P - PET is aggregated over the trailing
    window, fitted to the three-parameter log-logistic law in its
    generalized-logistic parameterization via L-moments (see
    :func:`_generalized_logistic_lmom_fit`), and transformed through the
    inverse standard normal.  When ``month_numbers`` is supplied a separate
    distribution is fitted per calendar month of the window end - the
    convention of Vicente-Serrano et al. (2010), who standardize each
    monthly series separately; without it a single pooled fit is used
    (variant for non-seasonal series).

    Args:
        monthly_precip_mm: 1-D array of monthly precipitation totals (mm).
        monthly_pet_mm: 1-D array of monthly PET (mm), same length -
            typically from :func:`thornthwaite_pet`.
        window_months: Aggregation window in months.
        month_numbers: Optional calendar month (1-12) per original entry.

    Returns:
        1-D array of SPEI values (one per complete trailing window).

    Raises:
        ValueError: On malformed input, fewer than
            :data:`MIN_SAMPLES_PER_WINDOW` aggregated samples in a fitting
            stratum, or a degenerate log-logistic fit.
    """
    precip = _validate_series(monthly_precip_mm, "monthly_precip_mm")
    pet = _validate_series(monthly_pet_mm, "monthly_pet_mm")
    if precip.shape != pet.shape:
        raise ValueError(f"precipitation shape {precip.shape} does not match PET shape {pet.shape}")
    if window_months < 1:
        raise ValueError(f"window_months must be >= 1, got {window_months}")

    balance = precip - pet
    kernel = np.ones(window_months, dtype=np.float64)
    aggregated = np.convolve(balance, kernel, mode="valid")
    spei = np.empty_like(aggregated)

    for stratum in _strata_for_windows(month_numbers, balance.size, window_months):
        values = aggregated[stratum]
        if values.size < MIN_SAMPLES_PER_WINDOW:
            raise ValueError(
                f"SPEI-{window_months} needs >= {MIN_SAMPLES_PER_WINDOW} "
                f"aggregated samples per fitting stratum but one stratum has "
                f"only {values.size}"
            )
        if float(np.ptp(values)) <= 0.0:
            raise ValueError("aggregated water balance is constant; SPEI undefined")
        xi, alpha, kappa = _generalized_logistic_lmom_fit(values)
        cdf = _generalized_logistic_cdf(values, xi, alpha, kappa)
        spei[stratum] = stats.norm.ppf(np.clip(cdf, _PROB_EPS, 1.0 - _PROB_EPS))
    return spei


def classify_usdm(spi_value: float) -> DroughtCategory:
    """Map an SPI (or SPEI) value to a US Drought Monitor category.

    Thresholds from Svoboda et al. (2002), Table 2: D0 at SPI <= -0.5,
    D1 at <= -0.8, D2 at <= -1.3, D3 at <= -1.6, D4 at <= -2.0.

    Args:
        spi_value: SPI or SPEI value.

    Returns:
        The matching :class:`DroughtCategory` (``NONE`` for SPI > -0.5).

    Raises:
        ValueError: If ``spi_value`` is not finite.
    """
    if not math.isfinite(spi_value):
        raise ValueError(f"spi_value must be finite, got {spi_value}")
    for threshold, category in _USDM_THRESHOLDS:
        if spi_value <= threshold:
            return category
    return DroughtCategory.NONE


class DroughtDetector:
    """Multi-scale drought detector built on SPI/SPEI physics cores.

    Computes SPI at the configured aggregation windows from a monthly
    precipitation series and, when temperature + latitude are supplied,
    the Thornthwaite-PET SPEI as a water-balance cross-check.  Maps the
    latest index values to US Drought Monitor categories.

    The detector works untrained (pure distribution fitting) and never
    substitutes fabricated values for missing input.
    """

    #: Severity ordering used to pick the worst category across windows.
    _SEVERITY_ORDER: tuple[DroughtCategory, ...] = (
        DroughtCategory.NONE,
        DroughtCategory.D0_ABNORMALLY_DRY,
        DroughtCategory.D1_MODERATE_DROUGHT,
        DroughtCategory.D2_SEVERE_DROUGHT,
        DroughtCategory.D3_EXTREME_DROUGHT,
        DroughtCategory.D4_EXCEPTIONAL_DROUGHT,
    )

    def __init__(self, windows_months: tuple[int, ...] = (1, 3, 6, 12)) -> None:
        """Initialize the detector.

        Args:
            windows_months: SPI/SPEI aggregation windows in months
                (McKee et al. 1993 use 3/6/12/24/48; the operational
                default here is 1/3/6/12).

        Raises:
            ValueError: If no window is given or a window is < 1.
        """
        if not windows_months:
            raise ValueError("windows_months must not be empty")
        if any(w < 1 for w in windows_months):
            raise ValueError(f"all windows must be >= 1 month, got {windows_months}")
        self.windows_months = tuple(windows_months)
        self.logger = logging.getLogger(__name__)

    def assess(
        self,
        monthly_precip_mm: np.ndarray,  # type: ignore[type-arg]
        monthly_temp_c: np.ndarray | None = None,  # type: ignore[type-arg]
        latitude_deg: float | None = None,
        month_numbers: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> DroughtAssessmentResult:
        """Run a full multi-window drought assessment.

        Args:
            monthly_precip_mm: Monthly precipitation totals (mm).
            monthly_temp_c: Optional monthly mean temperatures (degC);
                enables the SPEI water-balance branch.
            latitude_deg: Site latitude (required with ``monthly_temp_c``).
            month_numbers: Calendar month (1-12) per entry.  When supplied,
                SPI/SPEI distributions are fitted per calendar month (the
                McKee / Vicente-Serrano operational convention, which
                needs >= 30 samples per calendar-month stratum, i.e.
                roughly 30 years of data).  Required for the SPEI branch.

        Returns:
            A :class:`DroughtAssessmentResult`.

        Raises:
            ValueError: Propagated from the index cores on inadequate
                input (short series, all-zero precipitation, non-finite
                values), or if the SPEI branch is requested with
                incomplete inputs.
        """
        spi_series: dict[int, np.ndarray] = {}  # type: ignore[type-arg]
        for window in self.windows_months:
            spi_series[window] = compute_spi(monthly_precip_mm, window, month_numbers)

        spei_series: dict[int, np.ndarray] = {}  # type: ignore[type-arg]
        pet: np.ndarray | None = None  # type: ignore[type-arg]
        if monthly_temp_c is not None:
            if latitude_deg is None or month_numbers is None:
                raise ValueError(
                    "SPEI requested (monthly_temp_c supplied) but latitude_deg "
                    "and month_numbers are required to compute Thornthwaite PET"
                )
            pet = thornthwaite_pet(monthly_temp_c, latitude_deg, month_numbers)
            for window in self.windows_months:
                spei_series[window] = compute_spei(monthly_precip_mm, pet, window, month_numbers)

        spi_latest = {w: float(s[-1]) for w, s in spi_series.items()}
        spei_latest = {w: float(s[-1]) for w, s in spei_series.items()}
        categories = {w: classify_usdm(v) for w, v in spi_latest.items()}
        worst = max(categories.values(), key=self._SEVERITY_ORDER.index)

        result = DroughtAssessmentResult(
            drought_detected=worst is not DroughtCategory.NONE,
            category=worst.value,
            spi_latest=spi_latest,
            spi_series=spi_series,
            spei_latest=spei_latest,
            spei_series=spei_series,
            pet_mm=pet,
            categories_by_window={w: c.value for w, c in categories.items()},
            confidence=len(spi_series) / len(self.windows_months),
            warning_actions=self._generate_warnings(worst),
        )
        self.logger.info(
            "Drought assessment: %s (SPI latest: %s)",
            result.category,
            {w: round(v, 2) for w, v in spi_latest.items()},
        )
        return result

    @staticmethod
    def _generate_warnings(category: DroughtCategory) -> list[str]:
        """Generate advisory strings for a USDM category."""
        warnings: list[str] = []
        if category is DroughtCategory.D4_EXCEPTIONAL_DROUGHT:
            warnings.append("D4 EXCEPTIONAL DROUGHT: widespread water emergencies likely")
            warnings.append("Exceptional and widespread crop/pasture losses expected")
        elif category is DroughtCategory.D3_EXTREME_DROUGHT:
            warnings.append("D3 EXTREME DROUGHT: major crop/pasture losses expected")
            warnings.append("Widespread water shortages and restrictions likely")
        elif category is DroughtCategory.D2_SEVERE_DROUGHT:
            warnings.append("D2 SEVERE DROUGHT: crop or pasture losses likely")
            warnings.append("Water shortages common; restrictions may be imposed")
        elif category is DroughtCategory.D1_MODERATE_DROUGHT:
            warnings.append("D1 MODERATE DROUGHT: some damage to crops and pastures")
            warnings.append("Voluntary water-use restrictions requested")
        elif category is DroughtCategory.D0_ABNORMALLY_DRY:
            warnings.append("D0 ABNORMALLY DRY: short-term dryness slowing growth")
        return warnings

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:  # type: ignore[type-arg]
        """Extract a fixed 20-dim feature vector for ML fusion.

        Treats the input as a (possibly multivariate) series whose first
        axis is time; robust summary statistics are always emitted, and the
        SPI features are appended only when the flattened series qualifies
        as a valid monthly precipitation input (non-negative, long enough).
        Windows that cannot be computed contribute zeros with a companion
        validity flag of 0 - the detector never emits an index it could
        not legitimately compute.

        Args:
            data: Input array or tensor.

        Returns:
            Feature tensor of shape (20,).
        """
        if isinstance(data, torch.Tensor):
            arr: np.ndarray = data.detach().cpu().numpy()  # type: ignore[type-arg]
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
            float(np.percentile(flat, 75) - np.percentile(flat, 25)),
            float(np.mean(flat <= 0.0)),
        ]
        for window in (1, 3, 6, 12):
            spi_value = 0.0
            valid = 0.0
            if np.all(flat >= 0.0) and flat.size - window + 1 >= MIN_SAMPLES_PER_WINDOW:
                try:
                    spi_value = float(compute_spi(flat, window)[-1])
                    valid = 1.0
                except ValueError:
                    spi_value = 0.0
            features.extend([spi_value, valid])

        while len(features) < 20:
            features.append(0.0)
        return torch.tensor(features[:20], dtype=torch.float32)
