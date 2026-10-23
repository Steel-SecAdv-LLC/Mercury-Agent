# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Climate Resilience Module - Anomaly detection for environmental data.

Supports SDG 13 (Climate Action) by detecting anomalies in:
- Temperature patterns (heatwaves, cold snaps)
- Precipitation anomalies (droughts, floods)
- Sea level rise acceleration
- Extreme weather events
- Carbon emission spikes

⚠️ SIMULATION-BASED: Uses simulated climate data. Real-world validation required.

Research sources:
- IPCC Assessment Reports
- NOAA Climate.gov
- NASA Earth Observatory
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.provenance import DecisionSource


class ClimateEvent(Enum):
    """Climate event."""

    NORMAL = "normal"
    HEATWAVE = "heatwave"
    COLD_SNAP = "cold_snap"
    DROUGHT = "drought"
    FLOOD = "flood"
    HURRICANE = "hurricane"
    WILDFIRE = "wildfire"


class ClimateResilienceDetector:
    """Detect climate anomalies for disaster prediction and resilience."""

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "Sea level: a local trend above the global-mean 3.4 mm/yr (satellite altimetry "
                "1993-2018) marks a faster-than-global coast, above twice that a hotspot. Wind: "
                "hurricane force from 64 kt and major-hurricane force from 96 kt on the "
                "Saffir-Simpson scale."
            ),
            citation=(
                "Nerem et al. 2018, PNAS 115:2022-2025 (doi:10.1073/pnas.1717312115), "
                "bibliographic record checked on Crossref; NHC Saffir-Simpson Hurricane Wind "
                "Scale, https://www.nhc.noaa.gov/aboutsshws.php (64-82 kt Category 1, 96-112 kt "
                "Category 3), page checked 2026-10-23."
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "Temperature and precipitation anomalies alarm at |z| of 2 (hand-set); the "
                "Angstrom fire-danger interpretation (index below 2.0 fire conditions very likely) "
                "is attributed in the code to Chandler et al. 1983 and was not re-read."
            ),
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.temp_baseline = {"mean": 15.0, "std": 10.0}
        # Measured monthly-precipitation climatology, NOT a guess: mean and
        # population std of the 360 CONUS monthly totals, 1991-2020 (the WMO
        # climate-normals period), from NOAA NCEI Climate at a Glance
        # (nClimGrid), converted from inches at 25.4 mm/in and rounded to
        # 0.1 mm. Source series:
        # https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/
        # national/time-series/110-pcp-all-1-1991-2020.csv, recorded at
        # tests/fixtures/humanitarian/noaa_cag_conus_pcp_199101_202012.csv
        # (sha256 70704320056d285b7abe62c098a6776f0f3e90901c3fa2d1fc2574b
        # 50684bff4); tests recompute these constants from that fixture.
        # The shipped 100 +/- 50 placeholder put the 2-sigma deficit at
        # exactly 0 mm, so only a fully rainless window could grade as
        # drought; with the measured baseline every drought grade is
        # reachable from positive rainfall (2-sigma deficit at 39.2 mm).
        self.precip_baseline = {"mean": 66.2, "std": 13.5}

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "temperature",
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect climate anomalies.

        Args:
            data: Environmental sensor data (see each detector for units)
            detection_type: 'temperature', 'precipitation', 'sea_level',
                'wind', 'fire_weather'
            historical_context: Historical climate data for comparison

        Returns:
            Climate anomaly detection results with disaster predictions
        """
        if detection_type == "temperature":
            return self.detect_temperature_anomaly(data, historical_context)
        elif detection_type == "precipitation":
            return self.detect_precipitation_anomaly(data, historical_context)
        elif detection_type == "sea_level":
            return self.detect_sea_level_anomaly(data, historical_context)
        elif detection_type == "wind":
            return self.detect_wind_anomaly(data, historical_context)
        elif detection_type == "fire_weather":
            return self.detect_fire_weather_anomaly(data, historical_context)
        else:
            return self.detect_temperature_anomaly(data, historical_context)

    def detect_temperature_anomaly(
        self,
        temperature_series: np.ndarray[Any, Any],
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect heatwaves, cold snaps, and abnormal warming trends.

        Raises:
            ValueError: Non-finite temperature values — a NaN window must
                never read as a normal climate.
        """
        if len(temperature_series) == 0:
            return {"anomaly_detected": False}
        if not np.all(np.isfinite(np.asarray(temperature_series, dtype=float))):
            raise ValueError("temperature_series contains non-finite values")

        mean_temp = np.mean(temperature_series)
        max_temp = np.max(temperature_series)
        min_temp = np.min(temperature_series)

        baseline_mean = self.temp_baseline["mean"]
        baseline_std = self.temp_baseline["std"]

        z_score = (mean_temp - baseline_mean) / baseline_std if baseline_std > 0 else 0

        event_type = ClimateEvent.NORMAL
        severity = "low"

        if max_temp > baseline_mean + 3 * baseline_std:
            event_type = ClimateEvent.HEATWAVE
            severity = "critical"
        elif min_temp < baseline_mean - 3 * baseline_std:
            event_type = ClimateEvent.COLD_SNAP
            severity = "high"
        elif abs(z_score) > 2:
            severity = "medium"

        return {
            "anomaly_detected": bool(abs(z_score) > 2 or severity in ["high", "critical"]),
            "event_type": event_type.value,
            "severity": severity,
            "metrics": {
                "mean_temperature_c": float(mean_temp),
                "max_temperature_c": float(max_temp),
                "min_temperature_c": float(min_temp),
                "z_score": float(z_score),
            },
            "disaster_risk": self._assess_disaster_risk(event_type, severity),
            "recommendations": self._generate_climate_recommendations(event_type, severity),
            "timestamp": datetime.now(UTC),
        }

    def detect_precipitation_anomaly(
        self,
        precipitation_mm: np.ndarray[Any, Any],
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect droughts and floods from precipitation patterns.

        Grading runs on the measured NOAA 1991-2020 monthly climatology
        (see ``__init__``): the two-sigma deficit sits at 39.2 mm, so a
        moderate ("medium") drought is reachable from positive rainfall
        and the "high" grade engages below three sigma (25.7 mm) or for a
        fully rainless window, the maximal physical drought. Both event
        boundaries stay inclusive at two sigma.

        Raises:
            ValueError: Non-finite or negative precipitation — negative
                rainfall is unmeasurable, and a NaN window must never read
                as normal rainfall.
        """
        if len(precipitation_mm) == 0:
            return {"anomaly_detected": False}
        precip = np.asarray(precipitation_mm, dtype=float)
        if not np.all(np.isfinite(precip)) or np.any(precip < 0):
            raise ValueError("precipitation_mm values must be finite and non-negative")

        total_precip = np.sum(precipitation_mm)
        baseline = self.precip_baseline["mean"]
        std = self.precip_baseline["std"]

        z_score = (total_precip - baseline) / std if std > 0 else 0

        event_type = ClimateEvent.NORMAL
        severity = "low"

        if total_precip <= baseline - 2 * std:
            event_type = ClimateEvent.DROUGHT
            severity = "high" if z_score < -3 or total_precip <= 0 else "medium"
        elif total_precip >= baseline + 2 * std:
            event_type = ClimateEvent.FLOOD
            severity = "critical" if z_score > 3 else "high"

        return {
            "anomaly_detected": bool(abs(z_score) >= 2),
            "event_type": event_type.value,
            "severity": severity,
            "metrics": {
                "total_precipitation_mm": float(total_precip),
                "z_score": float(z_score),
            },
            "disaster_risk": self._assess_disaster_risk(event_type, severity),
            "recommendations": self._generate_climate_recommendations(event_type, severity),
            "timestamp": datetime.now(UTC),
        }

    #: Global-mean sea-level rise, satellite altimetry 1993-2018: trend
    #: 3.4 mm/yr, acceleration 0.084 mm/yr^2 (Nerem et al. 2018, PNAS
    #: 115:2022-2025). A local trend above the global mean marks a
    #: faster-than-global coast; above twice the global mean, a hotspot.
    GLOBAL_MSL_TREND_MM_YR = 3.4
    GLOBAL_MSL_ACCEL_MM_YR2 = 0.084

    #: NHC/Saffir-Simpson wind thresholds: hurricane force 64 kt, major
    #: hurricane (Category 3) 96 kt, converted at 0.514444 m/s per knot
    #: (https://www.nhc.noaa.gov/aboutsshws.php).
    HURRICANE_FORCE_MS = 64 * 0.514444
    MAJOR_HURRICANE_MS = 96 * 0.514444

    #: Angström fire-danger index I = R/20 + (27 - T)/10 (R: relative
    #: humidity %, T: air temperature degC). Published interpretation
    #: (Chandler et al. 1983, "Fire in Forestry"): I < 2.0 fire conditions
    #: very likely; 2.0-2.5 favorable; above, progressively unlikely.
    ANGSTROM_VERY_LIKELY = 2.0
    ANGSTROM_FAVORABLE = 2.5

    def detect_sea_level_anomaly(
        self,
        monthly_msl_m: np.ndarray[Any, Any],
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sea-level-rise trend, acceleration and coastal flood risk.

        Implements the published tide-gauge trend method: ordinary least
        squares on the monthly mean sea level with annual and semiannual
        harmonics in the design (Zervas 2009, NOAA Technical Report NOS
        CO-OPS 053 — the method behind NOAA's station trends), plus a
        quadratic term for acceleration (Church & White 2011, Surv Geophys
        32:585-602; Nerem et al. 2018). The detection references are the
        measured global-mean altimetry values (:data:`GLOBAL_MSL_TREND_MM_YR`,
        :data:`GLOBAL_MSL_ACCEL_MM_YR2`): a coast rising faster than the
        global mean carries elevated coastal flood risk (Sweet et al. 2022,
        NOAA Technical Report NOS 01: higher local sea level drives
        high-tide-flood frequency).

        Args:
            monthly_msl_m: Monthly mean sea level in metres (any fixed
                datum), consecutive calendar months.
            historical_context: Optional ``{"minor_flood_threshold_m": x}``
                (same datum): when the latest 12-month mean reaches it, the
                event grades as a critical coastal flood condition.

        Returns:
            Detection result with trend (mm/yr), acceleration (mm/yr^2) and
            coastal flood risk.

        Raises:
            ValueError: Non-finite sea-level values — gaps must be handled
                by the caller, never interpolated away here.
        """
        if len(monthly_msl_m) == 0:
            return {"anomaly_detected": False}
        series = np.asarray(monthly_msl_m, dtype=float).ravel()
        if not np.all(np.isfinite(series)):
            raise ValueError("monthly_msl_m contains non-finite values; supply clean data")
        n = series.size
        if n < 24:
            return {
                "anomaly_detected": False,
                "note": f"sea-level trend needs at least 24 monthly means, got {n}",
            }

        t = np.arange(n, dtype=float) / 12.0  # years
        seasonal = [
            np.cos(2 * np.pi * t),
            np.sin(2 * np.pi * t),
            np.cos(4 * np.pi * t),
            np.sin(4 * np.pi * t),
        ]
        linear_design = np.column_stack([np.ones(n), t, *seasonal])
        linear_coef, *_ = np.linalg.lstsq(linear_design, series, rcond=None)
        trend_mm_yr = float(linear_coef[1] * 1000.0)

        quad_design = np.column_stack([np.ones(n), t, t**2, *seasonal])
        quad_coef, *_ = np.linalg.lstsq(quad_design, series, rcond=None)
        acceleration_mm_yr2 = float(2.0 * quad_coef[2] * 1000.0)

        above_global_trend = trend_mm_yr > self.GLOBAL_MSL_TREND_MM_YR
        above_global_accel = acceleration_mm_yr2 > self.GLOBAL_MSL_ACCEL_MM_YR2

        event_type = ClimateEvent.NORMAL
        severity = "low"
        if above_global_trend:
            severity = "high" if trend_mm_yr > 2 * self.GLOBAL_MSL_TREND_MM_YR else "medium"
        elif above_global_accel:
            severity = "medium"

        context = historical_context or {}
        threshold = context.get("minor_flood_threshold_m")
        latest_year_mean = float(np.mean(series[-12:]))
        if threshold is not None and latest_year_mean >= float(threshold):
            event_type = ClimateEvent.FLOOD
            severity = "critical"

        return {
            "anomaly_detected": bool(
                above_global_trend or above_global_accel or event_type is ClimateEvent.FLOOD
            ),
            "event_type": event_type.value,
            "severity": severity,
            "metrics": {
                "trend_mm_per_year": trend_mm_yr,
                "acceleration_mm_per_year2": acceleration_mm_yr2,
                "latest_12mo_mean_m": latest_year_mean,
                "n_months": int(n),
                "global_reference_trend_mm_per_year": self.GLOBAL_MSL_TREND_MM_YR,
            },
            "coastal_flood_risk": "elevated" if above_global_trend else "baseline",
            "disaster_risk": self._assess_disaster_risk(event_type, severity),
            "recommendations": self._generate_climate_recommendations(event_type, severity),
            "timestamp": datetime.now(UTC),
        }

    def detect_wind_anomaly(
        self,
        wind_speed_ms: np.ndarray[Any, Any],
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Hurricane-force wind detection on the NHC/Saffir-Simpson scale.

        A maximum sustained wind at or above 64 kt (:data:`HURRICANE_FORCE_MS`)
        is hurricane force; at or above 96 kt (:data:`MAJOR_HURRICANE_MS`,
        Category 3) it is a major hurricane and grades critical
        (https://www.nhc.noaa.gov/aboutsshws.php).

        Args:
            wind_speed_ms: Sustained wind speeds in m/s.
            historical_context: Unused; kept for signature parity.

        Returns:
            Detection result; ``event_type`` is ``hurricane`` at hurricane
            force.
        """
        if len(wind_speed_ms) == 0:
            return {"anomaly_detected": False}
        winds = np.asarray(wind_speed_ms, dtype=float).ravel()
        if not np.all(np.isfinite(winds)) or np.any(winds < 0):
            raise ValueError("wind_speed_ms must be finite, non-negative m/s values")

        max_wind = float(np.max(winds))
        hurricane_force = max_wind >= self.HURRICANE_FORCE_MS

        event_type = ClimateEvent.HURRICANE if hurricane_force else ClimateEvent.NORMAL
        if max_wind >= self.MAJOR_HURRICANE_MS:
            severity = "critical"
        elif hurricane_force:
            severity = "high"
        else:
            severity = "low"

        return {
            "anomaly_detected": bool(hurricane_force),
            "event_type": event_type.value,
            "severity": severity,
            "metrics": {
                "max_sustained_wind_ms": max_wind,
                "max_sustained_wind_kt": max_wind / 0.514444,
                "hurricane_force_threshold_ms": self.HURRICANE_FORCE_MS,
            },
            "disaster_risk": self._assess_disaster_risk(event_type, severity),
            "recommendations": self._generate_climate_recommendations(event_type, severity),
            "timestamp": datetime.now(UTC),
        }

    def detect_fire_weather_anomaly(
        self,
        temp_humidity: np.ndarray[Any, Any],
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fire-weather detection by the Angström index.

        ``I = R/20 + (27 - T)/10`` per observation (T air temperature degC,
        R relative humidity %); the worst (lowest) index over the window
        decides (Chandler et al. 1983): below
        :data:`ANGSTROM_VERY_LIKELY` fire conditions are very likely (high
        severity), below :data:`ANGSTROM_FAVORABLE` favorable (medium).

        Args:
            temp_humidity: ``(N, 2)`` array of ``[temperature_c,
                relative_humidity_pct]`` observations (a single ``(2,)``
                pair is accepted).
            historical_context: Unused; kept for signature parity.

        Returns:
            Detection result; ``event_type`` is ``wildfire`` when fire
            weather is favorable or worse.

        Raises:
            ValueError: Wrong shape, non-finite values, or humidity outside
                0-100 / temperature outside -90..60 degC.
        """
        if len(temp_humidity) == 0:
            return {"anomaly_detected": False}
        obs = np.asarray(temp_humidity, dtype=float)
        if obs.ndim == 1 and obs.size == 2:
            obs = obs.reshape(1, 2)
        if obs.ndim != 2 or obs.shape[1] != 2:
            raise ValueError(
                "temp_humidity takes [temperature_c, relative_humidity_pct] rows, "
                f"got shape {obs.shape}"
            )
        temps, humidity = obs[:, 0], obs[:, 1]
        if (
            not np.all(np.isfinite(obs))
            or np.any(humidity < 0)
            or np.any(humidity > 100)
            or np.any(temps < -90)
            or np.any(temps > 60)
        ):
            raise ValueError(
                "temperature must be finite in [-90, 60] degC and humidity in [0, 100] %"
            )

        index = humidity / 20.0 + (27.0 - temps) / 10.0
        worst = float(np.min(index))

        if worst < self.ANGSTROM_VERY_LIKELY:
            event_type, severity = ClimateEvent.WILDFIRE, "high"
        elif worst < self.ANGSTROM_FAVORABLE:
            event_type, severity = ClimateEvent.WILDFIRE, "medium"
        else:
            event_type, severity = ClimateEvent.NORMAL, "low"

        return {
            "anomaly_detected": bool(event_type is ClimateEvent.WILDFIRE),
            "event_type": event_type.value,
            "severity": severity,
            "metrics": {
                "worst_angstrom_index": worst,
                "very_likely_threshold": self.ANGSTROM_VERY_LIKELY,
                "favorable_threshold": self.ANGSTROM_FAVORABLE,
            },
            "disaster_risk": self._assess_disaster_risk(event_type, severity),
            "recommendations": self._generate_climate_recommendations(event_type, severity),
            "timestamp": datetime.now(UTC),
        }

    def _assess_disaster_risk(self, event_type: ClimateEvent, severity: str) -> str:
        """Assess disaster risk based on climate event."""
        if severity == "critical":
            return "imminent"
        elif severity == "high":
            return "elevated"
        elif event_type != ClimateEvent.NORMAL:
            return "moderate"
        return "low"

    def _generate_climate_recommendations(
        self, event_type: ClimateEvent, severity: str
    ) -> list[str]:
        """Generate climate resilience recommendations."""
        recommendations = []

        if event_type == ClimateEvent.HEATWAVE:
            recommendations.append("Activate heat emergency protocols")
            recommendations.append("Open cooling centers for vulnerable populations")
            if severity == "critical":
                recommendations.append("URGENT: Issue heat warnings, restrict outdoor activities")
        elif event_type == ClimateEvent.COLD_SNAP:
            recommendations.append("Activate cold weather emergency protocols")
            recommendations.append("Open warming centers for vulnerable populations")
            if severity == "critical":
                recommendations.append("URGENT: Issue cold warnings, check on isolated residents")
        elif event_type == ClimateEvent.DROUGHT:
            recommendations.append("Implement water conservation measures")
            recommendations.append("Monitor agricultural impacts and food security")
        elif event_type == ClimateEvent.FLOOD:
            recommendations.append("Activate flood response teams")
            recommendations.append("Issue evacuation advisories for flood zones")
            if severity == "critical":
                recommendations.append("CRITICAL: Begin emergency evacuations")
        elif event_type == ClimateEvent.HURRICANE:
            recommendations.append("Activate hurricane preparedness protocols")
            recommendations.append("Secure structures and pre-position emergency resources")
            if severity == "critical":
                recommendations.append(
                    "CRITICAL: Major hurricane winds — evacuate per official orders"
                )
        elif event_type == ClimateEvent.WILDFIRE:
            recommendations.append("Raise fire-weather readiness and restrict ignition sources")
            recommendations.append("Pre-position firefighting resources")
            if severity == "high":
                recommendations.append("Issue fire-weather warnings to exposed communities")

        if not recommendations:
            recommendations.append("Continue routine climate monitoring")

        return recommendations
