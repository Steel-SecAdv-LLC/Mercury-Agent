# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flood Detector - Multi-Parameter Flood Monitoring System.

Comprehensive flood detection for humanitarian early warning:
- Precipitation accumulation analysis
- River gauge monitoring
- Soil saturation models
- Soil-saturation runoff analysis
- Refactoring engine for dynamic model optimization
- Cross-domain fusion with hurricane/tornado detectors

Integrations:
- USGS stream gauge network
- NWS precipitation data
- Satellite soil moisture observations
- 3R mechanism for adaptive flood modeling
- Dam/reservoir level monitoring

Research sources:
- NOAA National Weather Service
- USGS Water Resources
- FEMA flood mapping
- Academic research on hydrological modeling

⚠️ SIMULATION-BASED: For research/development. NOT a replacement for official
flood warnings (NWS, USGS). Always defer to official flood warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from omni_mercury_engine.core.three_r_mechanism import (
    RecursionEngine,
    RefactoringEngine as CoreRefactoringEngine,
    ResonanceEngine,
)
from omni_mercury_engine.data_sources.base import DataSourceType
from omni_mercury_engine.data_sources.live_ingestion import (
    LiveDataError,
    LiveFetch,
    fetch_live_datapoints,
    require_live_client,
)
from omni_mercury_engine.detectors._hazard_input import (
    InvalidObservationError,
    finite_array,
    finite_scalar,
)
from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

if TYPE_CHECKING:
    from omni_mercury_engine.data_sources.earth_science import (
        NOAANWPSSource,
        NWSWeatherAlertsSource,
    )

#: NWS products issued when flooding is imminent or occurring (NWSI 10-320
#: and 10-922); the matching Watch, Advisory and Statement products are not.
FLOOD_WARNING_EVENTS = frozenset(
    {"Flood Warning", "Flash Flood Warning", "Coastal Flood Warning", "Lakeshore Flood Warning"}
)


class FloodSeverity(Enum):
    """Flood severity classification."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    RECORD = "record"
    NO_FLOOD = "no_flood"


class FloodType(Enum):
    """Types of flooding."""

    FLASH = "flash"
    RIVER = "river"
    COASTAL = "coastal"
    URBAN = "urban"
    DAM_FAILURE = "dam_failure"
    GROUNDWATER = "groundwater"
    NO_FLOOD = "no_flood"


@dataclass
class FloodPredictionResult:
    """Flood prediction results."""

    flood_likely: bool
    confidence: float
    severity: str
    flood_type: str

    river_stage_ft: float = 0.0
    flood_stage_ft: float = 0.0
    stage_trend: str = "stable"

    precipitation_24h_inches: float = 0.0
    precipitation_forecast_inches: float = 0.0
    soil_saturation_pct: float = 0.0

    runoff_coefficient: float = 0.0
    #: Lead time (h) to the crest of the supplied stage forecast; 0.0 when the
    #: forecast is no higher than the current stage. None without a forecast.
    time_to_peak_hours: float | None = None
    #: Largest supplied discharge (observed, history or forecast), cfs; None
    #: when no discharge was supplied (0.0 claimed a dry channel).
    peak_discharge_cfs: float | None = None

    refactoring_score: float = 0.0
    model_optimization_iterations: int = 0
    prediction_uncertainty: float = 0.0

    # Inundated area and exposed population need an inundation map and
    # census exposure, which no input carries: None means not assessed (the
    # former 0.0 / 0 read "nothing affected").
    affected_area_sq_mi: float | None = None
    population_at_risk: int | None = None

    warning_actions: list[str] = field(default_factory=list)
    evacuation_routes: list[str] = field(default_factory=list)
    shelter_locations: list[str] = field(default_factory=list)

    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None


class PrecipitationAnalyzer:
    """Precipitation accumulation analysis for flood potential.

    Monitors rainfall rates, accumulation, and forecast precipitation.
    Flash-flood potential is judged against the location's own NWS Flash
    Flood Guidance (FFG) when the caller supplies it (``ffg_1h_inches``,
    ``ffg_3h_inches``, ``ffg_6h_inches`` -- the River Forecast Centers'
    published definition is that flash flooding becomes possible when
    rainfall over the period exceeds the FFG value for that area); the
    fixed 2.0 in/h and 4.0 in/24 h lines remain only as a fallback screen
    and the result labels which basis graded it.
    """

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "With per-location NWS Flash Flood Guidance supplied, "
                "flash_flood_risk fires when any accumulation (1/3/6 h) "
                "reaches its FFG value -- the published operational "
                "definition of flash-flood potential."
            ),
            citation=(
                "NOAA/NWS River Forecast Center Flash Flood Guidance: FFG is "
                "the rain in a given duration needed to bring small streams "
                "to bankfull; rainfall exceeding FFG indicates flash-flood "
                "potential (NWS Manual 10-950 / RFC FFG product description)"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "Without FFG the fixed screen remains: 2.0 in/h flash and "
                "4.0 in/24 h flood lines with the intensity ladder; no "
                "published fixed national threshold exists (guidance is "
                "location-specific), and results carry "
                "basis='fixed_screen_unsourced'."
            ),
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.flash_flood_threshold_1h = 2.0  # inches/hour
        self.flood_threshold_24h = 4.0  # inches in 24 hours

    def analyze_precipitation(self, precip_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze precipitation for flood potential.

        Args:
            precip_data: Precipitation measurements and forecasts.  Optional
                ``ffg_1h_inches`` / ``ffg_3h_inches`` / ``ffg_6h_inches``
                carry the location's NWS Flash Flood Guidance; when any is
                supplied with its matching accumulation, flash-flood risk is
                graded against the guidance instead of the fixed screen.

        Returns:
            Precipitation analysis results.
        """
        # Upper bounds sit above the world records (about 12 in in 1 h and 72 in
        # in 24 h); anything larger is millimetres, not inches.
        precip_1h = finite_scalar(
            precip_data.get("precipitation_1h_inches", 0.0),
            "precipitation_1h_inches",
            unit="in",
            lo=0.0,
            hi=15.0,
        )
        precip_6h = finite_scalar(
            precip_data.get("precipitation_6h_inches", 0.0),
            "precipitation_6h_inches",
            unit="in",
            lo=0.0,
            hi=50.0,
        )
        precip_24h = finite_scalar(
            precip_data.get("precipitation_24h_inches", 0.0),
            "precipitation_24h_inches",
            unit="in",
            lo=0.0,
            hi=80.0,
        )
        forecast_24h = finite_scalar(
            precip_data.get("forecast_24h_inches", 0.0),
            "forecast_24h_inches",
            unit="in",
            lo=0.0,
            hi=80.0,
        )

        precip_3h = finite_scalar(
            precip_data.get("precipitation_3h_inches", 0.0),
            "precipitation_3h_inches",
            unit="in",
            lo=0.0,
            hi=40.0,
        )
        ffg_pairs: list[tuple[float, float]] = []
        for accumulation, key in (
            (precip_1h, "ffg_1h_inches"),
            (precip_3h, "ffg_3h_inches"),
            (precip_6h, "ffg_6h_inches"),
        ):
            guidance = precip_data.get(key)
            if guidance is not None:
                ffg_pairs.append(
                    (accumulation, finite_scalar(guidance, key, unit="in", lo=0.01, hi=30.0))
                )

        if ffg_pairs:
            # Published basis: rainfall at or above the location's NWS Flash
            # Flood Guidance for the matching duration.
            flash_flood_basis = "nws_flash_flood_guidance"
            flash_flood_risk = any(acc >= ffg for acc, ffg in ffg_pairs)
        else:
            flash_flood_basis = "fixed_screen_unsourced"
            flash_flood_risk = precip_1h >= self.flash_flood_threshold_1h
        flood_risk = precip_24h >= self.flood_threshold_24h

        if precip_1h >= 3.0 or precip_24h >= 8.0:
            intensity = "extreme"
        elif precip_1h >= 2.0 or precip_24h >= 6.0:
            intensity = "heavy"
        elif precip_1h >= 1.0 or precip_24h >= 4.0:
            intensity = "moderate"
        elif precip_1h >= 0.5 or precip_24h >= 2.0:
            intensity = "light"
        else:
            intensity = "trace"

        return {
            "precipitation_1h_inches": float(precip_1h),
            "precipitation_6h_inches": float(precip_6h),
            "precipitation_24h_inches": float(precip_24h),
            "forecast_24h_inches": float(forecast_24h),
            "flash_flood_risk": flash_flood_risk,
            "flash_flood_basis": flash_flood_basis,
            "flood_risk": flood_risk,
            "intensity": intensity,
            "total_expected_inches": float(precip_24h + forecast_24h),
        }


class RiverGaugeMonitor:
    """River gauge monitoring for flood stage detection.

    Tracks river levels relative to flood stages.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def analyze_river_stage(self, gauge_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze river gauge data for flood conditions.

        NWS defines flood categories per forecast point (NWSM 10-950), and not
        every point has every category, so the gauge's own stages are used and
        none are invented: ``current_stage_ft`` and ``flood_stage_ft`` are
        required, the others optional. The former defaults (action 10, flood
        15, moderate 20, major 25, record 30 ft) classified an 8 ft creek with
        a 6 ft flood stage as normal, and an empty record as at flood stage.

        Args:
            gauge_data: River gauge measurements (feet).

        Returns:
            River stage analysis results.

        Raises:
            InvalidObservationError: If the current or flood stage is missing
                or not finite, or the supplied categories are out of order.
        """
        if "current_stage_ft" not in gauge_data or "flood_stage_ft" not in gauge_data:
            raise InvalidObservationError(
                "gauge_data needs the observed 'current_stage_ft' and the gauge's own "
                "'flood_stage_ft' (NWS minor-flood stage); categories are never defaulted"
            )
        current_stage = finite_scalar(gauge_data["current_stage_ft"], "current_stage_ft", unit="ft")
        flood_stage = finite_scalar(gauge_data["flood_stage_ft"], "flood_stage_ft", unit="ft")
        categories = [("minor", flood_stage)]
        for key, status in (
            ("moderate_flood_stage_ft", "moderate"),
            ("major_flood_stage_ft", "major"),
            ("record_stage_ft", "record"),
        ):
            if gauge_data.get(key) is not None:
                categories.append((status, finite_scalar(gauge_data[key], key, unit="ft")))
        stages = [stage for _, stage in categories]
        if stages != sorted(stages):
            raise InvalidObservationError(
                f"gauge flood categories are out of order: {dict(categories)}; "
                "expected flood <= moderate <= major <= record stage"
            )
        action_stage = (
            finite_scalar(gauge_data["action_stage_ft"], "action_stage_ft", unit="ft")
            if gauge_data.get("action_stage_ft") is not None
            else None
        )

        stage_history = gauge_data.get("stage_history_ft", [])
        if len(stage_history):
            stage_history = finite_array(stage_history, "stage_history_ft", unit="ft", ndim=1)

        if len(stage_history) >= 2:
            recent_change = stage_history[-1] - stage_history[0]
            if recent_change > 2.0:
                trend = "rising_rapidly"
            elif recent_change > 0.5:
                trend = "rising"
            elif recent_change < -0.5:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        discharges = [
            finite_array(gauge_data[key], key, unit="cfs", ndim=(0, 1), lo=0.0)
            for key in ("discharge_cfs", "discharge_history_cfs", "forecast_discharge_cfs")
            if gauge_data.get(key) is not None
        ]
        peak_discharge = max(float(np.max(d)) for d in discharges) if discharges else None

        time_to_peak: float | None = None
        forecast_crest: float | None = None
        if gauge_data.get("forecast_stage_ft") is not None:
            forecast = finite_array(
                gauge_data["forecast_stage_ft"], "forecast_stage_ft", unit="ft", ndim=1
            )
            if "forecast_lead_h" not in gauge_data:
                raise InvalidObservationError(
                    "forecast_stage_ft needs forecast_lead_h (hours ahead of each value)"
                )
            lead = finite_array(
                gauge_data["forecast_lead_h"], "forecast_lead_h", unit="h", ndim=1, lo=0.0
            )
            if lead.shape != forecast.shape or np.any(np.diff(lead) <= 0):
                raise InvalidObservationError(
                    "forecast_lead_h must give one increasing lead time per forecast stage"
                )
            crest = int(np.argmax(forecast))
            forecast_crest = float(forecast[crest])
            time_to_peak = float(lead[crest]) if forecast_crest > current_stage else 0.0

        flood_status = "normal"
        if action_stage is not None and current_stage >= action_stage:
            flood_status = "action"
        for status, stage in categories:
            if current_stage >= stage:
                flood_status = status

        return {
            "current_stage_ft": float(current_stage),
            "flood_stage_ft": float(flood_stage),
            "stage_above_flood_ft": float(max(0, current_stage - flood_stage)),
            "flood_status": flood_status,
            "stage_trend": trend,
            "at_or_above_flood": current_stage >= flood_stage,
            "peak_discharge_cfs": peak_discharge,
            "time_to_peak_hours": time_to_peak,
            "forecast_crest_ft": forecast_crest,
        }


class SoilSaturationModel:
    """Soil saturation modeling for runoff prediction.

    Estimates soil moisture and runoff potential.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def analyze_soil_conditions(self, soil_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze soil saturation for runoff potential.

        Args:
            soil_data: Soil moisture and characteristics

        Returns:
            Soil analysis results
        """
        # A supplied-but-bad value is refused before anything else: a
        # percentage outside [0, 100] gave a runoff coefficient above 1.
        supplied_moisture = (
            finite_scalar(
                soil_data["soil_moisture_pct"], "soil_moisture_pct", unit="%", lo=0.0, hi=100.0
            )
            if "soil_moisture_pct" in soil_data
            else None
        )
        # Both the moisture and the soil class must be supplied: a 50%-moist
        # loam used to be invented for empty input and served as a measured
        # 0.8 runoff coefficient.
        missing = [k for k in ("soil_moisture_pct", "soil_type") if k not in soil_data]
        if missing:
            return {
                "soil_moisture_pct": None,
                "soil_type": soil_data.get("soil_type"),
                "infiltration_rate_in_hr": None,
                "runoff_coefficient": None,
                "saturation_status": "unknown",
                "high_runoff_potential": False,
                "missing_inputs": missing,
            }
        soil_moisture_pct = float(supplied_moisture)  # type: ignore[arg-type]
        soil_type = soil_data["soil_type"]
        # antecedent_moisture used for future soil memory modeling
        _ = soil_data.get("antecedent_moisture_pct")

        # Basic intake classes consistent with USDA NRCS ranges (in/hr):
        # sand > 0.75, loam ~ 0.35-0.5, clay < 0.15. An unrecognised class is
        # refused, not silently read as loam.
        infiltration_rates = {
            "sand": 0.8,
            "sandy_loam": 0.6,
            "loam": 0.4,
            "clay_loam": 0.25,
            "clay": 0.1,
        }
        if soil_type not in infiltration_rates:
            raise InvalidObservationError(
                f"soil_type must be one of {sorted(infiltration_rates)}, got {soil_type!r}"
            )
        base_infiltration = infiltration_rates[soil_type]

        saturation_factor = 1.0 - (soil_moisture_pct / 100.0)
        effective_infiltration = base_infiltration * saturation_factor

        runoff_coefficient = 1.0 - effective_infiltration

        if soil_moisture_pct >= 90:
            saturation_status = "saturated"
        elif soil_moisture_pct >= 70:
            saturation_status = "near_saturation"
        elif soil_moisture_pct >= 50:
            saturation_status = "moist"
        elif soil_moisture_pct >= 30:
            saturation_status = "moderate"
        else:
            saturation_status = "dry"

        return {
            "soil_moisture_pct": float(soil_moisture_pct),
            "soil_type": soil_type,
            "infiltration_rate_in_hr": float(effective_infiltration),
            "runoff_coefficient": float(runoff_coefficient),
            "saturation_status": saturation_status,
            "high_runoff_potential": runoff_coefficient > 0.6,
            "missing_inputs": [],
        }


class FloodPredictionOptimizer:
    """Dynamic model optimization engine for flood prediction.

    Implements iterative prediction refinement based on observed data for continuously improving
    flood prediction accuracy.
    """

    def __init__(self, max_iterations: int = 10, convergence_threshold: float = 0.01) -> None:
        """Initialize the instance."""
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.logger = logging.getLogger(__name__)

    def optimize_prediction(
        self,
        initial_prediction: dict[str, Any],
        observed_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Iteratively refactor prediction model based on observations.

        Args:
            initial_prediction: Initial flood prediction
            observed_data: Observed conditions for model refinement

        Returns:
            Optimized prediction with refactoring metrics
        """
        current_prediction = initial_prediction.copy()
        iterations = 0
        previous_error = float("inf")

        for i in range(self.max_iterations):
            iterations = i + 1

            error = self._compute_prediction_error(current_prediction, observed_data)

            if abs(previous_error - error) < self.convergence_threshold:
                break

            current_prediction = self._refactor_prediction(current_prediction, observed_data, error)

            previous_error = error

        uncertainty = self._estimate_uncertainty(current_prediction, observed_data)

        return {
            "optimized_prediction": current_prediction,
            "iterations": iterations,
            "final_error": float(previous_error),
            "uncertainty": float(uncertainty),
            "converged": iterations < self.max_iterations,
        }

    def _compute_prediction_error(
        self, prediction: dict[str, Any], observed: dict[str, Any]
    ) -> float:
        """Compute error between prediction and observations."""
        errors = []

        if "stage_ft" in prediction and "observed_stage_ft" in observed:
            stage_error = abs(prediction["stage_ft"] - observed["observed_stage_ft"])
            errors.append(stage_error / 10.0)

        if "discharge_cfs" in prediction and "observed_discharge_cfs" in observed:
            discharge_error = abs(prediction["discharge_cfs"] - observed["observed_discharge_cfs"])
            errors.append(discharge_error / 10000.0)

        return np.mean(errors) if errors else 0.5

    def _refactor_prediction(
        self,
        prediction: dict[str, Any],
        observed: dict[str, Any],
        error: float,
    ) -> dict[str, Any]:
        """Refactor prediction based on error."""
        refactored = prediction.copy()

        learning_rate = 0.3

        if "stage_ft" in refactored and "observed_stage_ft" in observed:
            correction = (observed["observed_stage_ft"] - refactored["stage_ft"]) * learning_rate
            refactored["stage_ft"] = refactored["stage_ft"] + correction

        if "discharge_cfs" in refactored and "observed_discharge_cfs" in observed:
            correction = (
                observed["observed_discharge_cfs"] - refactored["discharge_cfs"]
            ) * learning_rate
            refactored["discharge_cfs"] = refactored["discharge_cfs"] + correction

        return refactored

    def _estimate_uncertainty(self, prediction: dict[str, Any], observed: dict[str, Any]) -> float:
        """Estimate prediction uncertainty."""
        error = self._compute_prediction_error(prediction, observed)
        uncertainty = min(error * 2, 1.0)
        return uncertainty


class FloodDetector:
    """Comprehensive flood detection system.

    Integrates precipitation analysis, river gauge monitoring, soil saturation
    modeling (including its physics-based runoff coefficient), and the 3R
    mechanism for multi-parameter flood prediction.

    Deep 3R Integration:
    - RecursionEngine: Hierarchical multi-scale feature extraction
    - ResonanceEngine: FFT-based frequency-domain anomaly detection
    - FloodPredictionOptimizer: Iterative prediction refinement
    - CoreRefactoringEngine: Code complexity analysis (from three_r_mechanism)
    """

    decision_source = (
        # Rainfall legs: the FFG published rule and its unsourced fixed-screen
        # fallback, declared once on PrecipitationAnalyzer and reused here.
        *PrecipitationAnalyzer.decision_source,
        DecisionSource(
            kind="published",
            description=(
                "The river-gauge leg classifies flooding against the gauge's "
                "own published NWS flood categories (minor/moderate/major/"
                "record stages per forecast point; none are defaulted), and "
                "detect_live takes severity from the NWPS flood category of "
                "the worst gauge, treating an NWS Warning as imminent or "
                "occurring flooding."
            ),
            citation=(
                "NWS Manual 10-950 (flood categories defined per forecast "
                "point; action stage is preparation, not flooding); NWPS "
                "gauge flood categories; NWSI 10-320 (Warning = flooding "
                "imminent or occurring)"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The predict_flood aggregate is unsourced engineering: "
                "flood_likely fires at indicator count >= 2 (flash-flood "
                "risk = 2, gauge at/above flood stage = 2, 24 h flood risk "
                "= 1, stage rising > 2 ft = 1, high runoff potential = "
                "0.5) with confidence = indicators/5 and the moderate/minor "
                "severity split at 3 indicators -- in-code constants with "
                "no published source and no labelled calibration."
            ),
        ),
    )

    def __init__(
        self,
        enable_precipitation: bool = True,
        enable_river_gauge: bool = True,
        enable_soil: bool = True,
        enable_refactoring: bool = True,
        enable_recursion: bool = True,
        enable_resonance: bool = True,
        rng: DeterministicRNG | None = None,
        gauge_source: NOAANWPSSource | None = None,
        alerts_source: NWSWeatherAlertsSource | None = None,
    ):
        """Initialize the instance.

        Live-ingestion pattern (uniform across hazard detectors): pass an
        optional NWPS river-gauge client via ``gauge_source`` and/or an NWS
        weather-alerts client via ``alerts_source`` (dependency injection;
        default None = fully offline). :meth:`fetch_live_data` exposes a
        provenance-checked fetch and :meth:`detect_live` assesses the real
        observed gauge stages and active flood alerts -- official readings
        are never turned into synthetic precipitation or soil values.
        """
        self.enable_precipitation = enable_precipitation
        self.enable_river_gauge = enable_river_gauge
        self.enable_soil = enable_soil
        self.enable_refactoring = enable_refactoring
        self.enable_recursion = enable_recursion
        self.enable_resonance = enable_resonance
        self._rng = rng or get_global_rng()

        self.precip_analyzer = PrecipitationAnalyzer() if enable_precipitation else None
        self.gauge_monitor = RiverGaugeMonitor() if enable_river_gauge else None
        self.soil_model = SoilSaturationModel() if enable_soil else None
        self.prediction_optimizer = FloodPredictionOptimizer() if enable_refactoring else None

        self._gauge_source = gauge_source
        self._alerts_source = alerts_source

        self.recursion_engine = RecursionEngine(max_depth=5)
        self.resonance_engine = ResonanceEngine(sampling_rate=1.0)
        self.core_refactoring_engine = CoreRefactoringEngine()

        self.logger = logging.getLogger(__name__)

    def predict_flood(self, flood_data: dict[str, Any]) -> FloodPredictionResult:
        """Comprehensive flood prediction.

        Args:
            flood_data: Multi-parameter flood monitoring data including:
                - precip_data: Precipitation measurements and forecasts
                - gauge_data: River gauge measurements
                - soil_data: Soil moisture and characteristics
                - observed_data: Observed conditions for model refinement
                - metadata: Location info, timestamps

        Returns:
            Flood prediction with severity and recommendations
        """
        result = FloodPredictionResult(
            flood_likely=False,
            confidence=0.0,
            severity="no_flood",
            flood_type="no_flood",
        )

        indicators_detected: float = 0.0
        gauge_status: str | None = None

        if (
            self.enable_precipitation
            and "precip_data" in flood_data
            and self.precip_analyzer is not None
        ):
            precip_result = self.precip_analyzer.analyze_precipitation(flood_data["precip_data"])
            result.precipitation_24h_inches = precip_result["precipitation_24h_inches"]
            result.precipitation_forecast_inches = precip_result["forecast_24h_inches"]

            if precip_result["flash_flood_risk"]:
                indicators_detected += 2.0
                result.flood_type = "flash"
            elif precip_result["flood_risk"]:
                indicators_detected += 1.0

        if (
            self.enable_river_gauge
            and "gauge_data" in flood_data
            and self.gauge_monitor is not None
        ):
            gauge_result = self.gauge_monitor.analyze_river_stage(flood_data["gauge_data"])
            result.river_stage_ft = gauge_result["current_stage_ft"]
            result.flood_stage_ft = gauge_result["flood_stage_ft"]
            result.stage_trend = gauge_result["stage_trend"]
            result.peak_discharge_cfs = gauge_result["peak_discharge_cfs"]
            result.time_to_peak_hours = gauge_result["time_to_peak_hours"]
            gauge_status = gauge_result["flood_status"]

            if gauge_result["at_or_above_flood"]:
                indicators_detected += 2.0
                if result.flood_type == "no_flood":
                    result.flood_type = "river"

            if gauge_result["stage_trend"] == "rising_rapidly":
                indicators_detected += 1.0

        if self.enable_soil and "soil_data" in flood_data and self.soil_model is not None:
            soil_result = self.soil_model.analyze_soil_conditions(flood_data["soil_data"])
            if soil_result["soil_moisture_pct"] is not None:
                result.soil_saturation_pct = soil_result["soil_moisture_pct"]
                result.runoff_coefficient = soil_result["runoff_coefficient"]

            if soil_result["high_runoff_potential"]:
                indicators_detected += 0.5

        if (
            self.enable_refactoring
            and "observed_data" in flood_data
            and self.prediction_optimizer is not None
        ):
            initial_prediction = {"stage_ft": result.river_stage_ft}
            if result.peak_discharge_cfs is not None:
                initial_prediction["discharge_cfs"] = result.peak_discharge_cfs
            refactor_result = self.prediction_optimizer.optimize_prediction(
                initial_prediction, flood_data["observed_data"]
            )
            result.refactoring_score = 1.0 - refactor_result["final_error"]
            result.model_optimization_iterations = refactor_result["iterations"]
            result.prediction_uncertainty = refactor_result["uncertainty"]

        # ``signal_data`` no longer carries decision weight: two generic
        # spectral heuristics on an unspecified signal added 0.7 indicators, so
        # injected noise flipped ``flood_likely``.

        # The refactoring-engine integration happens above via
        # ``prediction_optimizer.optimize_prediction`` (same guard). A second
        # block here used to feed ``str(<result dict>)`` to
        # ``RefactoringEngine.detect_code_anomalies``, which requires a real
        # callable (``inspect.getsource``); it always failed internally,
        # logged a warning, and contributed exactly 0 — so it was removed.

        result.flood_likely = indicators_detected >= 2
        result.confidence = min(indicators_detected / 5.0, 1.0)
        result.severity = self._determine_severity(indicators_detected, result, gauge_status)

        result.warning_actions = self._generate_warnings(result)
        result.evacuation_routes = self._identify_evacuation_routes(result)
        result.shelter_locations = self._identify_shelters(result)

        self.logger.info(
            f"Flood prediction: {result.severity}, "
            f"stage={result.river_stage_ft:.1f}ft, confidence={result.confidence:.2f}"
        )

        return result

    def fetch_live_data(self, *, allow_simulated: bool = False, **kwargs: Any) -> LiveFetch:
        """Fetch live NWPS river-gauge readings through the injected client.

        Args:
            allow_simulated: Explicit opt-in for simulated sources (NWPS is a
                real feed, so this normally stays False).
            **kwargs: Passed to the client fetch.

        Returns:
            Provenance-checked LiveFetch of FLOOD data points.

        Raises:
            LiveDataError: No gauge client injected, or the fetch failed.
        """
        client = require_live_client(self._gauge_source, "FloodDetector", "NWPS river-gauge")
        return fetch_live_datapoints(
            client,
            allow_simulated=allow_simulated,
            source_types=[DataSourceType.FLOOD],
            **kwargs,
        )

    def detect_live(
        self, *, allow_simulated: bool = False, **fetch_kwargs: Any
    ) -> FloodPredictionResult:
        """Assess the live flood state from real gauge stages and NWS alerts.

        This is an OBSERVED-STATE assessment: severity is taken from the NWPS
        flood category of the worst gauge (the same major/moderate/minor
        vocabulary ``_determine_severity`` uses), the stage/forecast values on
        the result are that gauge's actual readings, and active NWS flood
        alerts corroborate. Precipitation and soil fields stay at their
        absent defaults -- gauge metadata is never converted into synthetic
        measurements. At least one of the two clients must be injected.

        Args:
            allow_simulated: Explicit opt-in for simulated sources.
            **fetch_kwargs: Extra client fetch parameters.

        Returns:
            FloodPredictionResult with ``source_id`` / ``data_provenance`` /
            ``live_context`` populated from the real observed state.

        Raises:
            LiveDataError: No client injected, or every wired fetch failed.
        """
        category_rank = {"major": 4, "moderate": 3, "minor": 2, "action": 1}
        live_context: dict[str, Any] = {}
        source_ids: list[str] = []
        provenance: list[str] = []

        worst_rank = 0
        worst_gauge: dict[str, Any] | None = None
        if self._gauge_source is not None:
            fetch = self.fetch_live_data(allow_simulated=allow_simulated, **fetch_kwargs)
            source_ids.append(fetch.source_id)
            provenance.append(fetch.data_provenance)
            categories: dict[str, int] = {}
            for dp in fetch.data_points:
                cat = str(dp.data.get("flood_category") or "").lower()
                categories[cat or "unknown"] = categories.get(cat or "unknown", 0) + 1
                rank = category_rank.get(cat, 0)
                # Only a gauge at action stage or above is "worst"; with none,
                # the first gauge used to be reported (a 70 ft reservoir pool
                # elevation came back as the river stage).
                if rank > worst_rank:
                    worst_rank = rank
                    worst_gauge = dp.data
            live_context["gauge_count"] = len(fetch.data_points)
            live_context["gauge_categories"] = categories
            if worst_gauge is not None:
                live_context["worst_gauge"] = {
                    "gauge_id": worst_gauge.get("gauge_id"),
                    "name": worst_gauge.get("name"),
                    "flood_category": worst_gauge.get("flood_category"),
                    "forecast_flood_category": worst_gauge.get("forecast_flood_category"),
                }
                # NWPS reports flow as the secondary value, usually in kcfs.
                flow = worst_gauge.get("observed_secondary")
                flow_scale = {"kcfs": 1000.0, "cfs": 1.0}.get(
                    str(worst_gauge.get("observed_secondary_unit", "")).lower()
                )
                if flow is not None and flow_scale is not None and float(flow) >= 0.0:
                    live_context["worst_gauge"]["observed_flow_cfs"] = float(flow) * flow_scale

        flood_alerts = 0
        if self._alerts_source is not None:
            alert_fetch = fetch_live_datapoints(
                self._alerts_source,
                allow_simulated=allow_simulated,
                source_types=[DataSourceType.WEATHER_ALERT],
            )
            source_ids.append(alert_fetch.source_id)
            provenance.append(alert_fetch.data_provenance)
            # NWS issues a Warning when flooding is imminent or occurring
            # (NWSI 10-320); watches, advisories and statements mean possible
            # or minor flooding and are reported but are not evidence of a
            # flood. Every alert containing "flood" used to count, so 23
            # Coastal Flood Advisories made flood_likely True.
            flood_named = [
                dp
                for dp in alert_fetch.data_points
                if "flood" in str(dp.data.get("event", "")).lower()
            ]
            warnings = [dp for dp in flood_named if dp.data.get("event") in FLOOD_WARNING_EVENTS]
            flood_alerts = len(warnings)
            warning_events = {str(dp.data.get("event")) for dp in warnings}
            live_context["alert_scope"] = alert_fetch.source_id
            live_context["flood_alerts"] = flood_alerts
            live_context["flood_alert_events"] = sorted(warning_events)
            live_context["other_flood_products"] = sorted(
                {str(dp.data.get("event", "")) for dp in flood_named} - warning_events
            )[:10]

        if not source_ids:
            raise LiveDataError(
                "FloodDetector: no gauge_source or alerts_source injected; construct "
                "the detector with a data_sources client instance to enable the "
                "optional live path."
            )

        # Action stage is preparation, not flooding (NWSM 10-950): it no
        # longer reads "minor".
        rank_to_severity = {4: "major", 3: "moderate", 2: "minor", 1: "no_flood", 0: "no_flood"}
        severity = rank_to_severity[worst_rank]
        flood_likely = worst_rank >= 2 or flood_alerts > 0
        if worst_rank < 2 and flood_alerts > 0:
            severity = "minor"  # A Warning is in effect but no gauge is in flood.

        # Confidence reflects observation quality: gauges are direct
        # measurements; alerts are official statements.
        if worst_rank >= 2:
            confidence = 0.9
        elif flood_alerts > 0:
            confidence = 0.7
        else:
            confidence = 0.0

        if worst_rank >= 2:
            flood_type = FloodType.RIVER.value
        elif any("Flash Flood" in event for event in live_context.get("flood_alert_events", [])):
            flood_type = FloodType.FLASH.value
        elif flood_alerts > 0 and all(
            event.startswith(("Coastal", "Lakeshore"))
            for event in live_context.get("flood_alert_events", [])
        ):
            flood_type = FloodType.COASTAL.value
        elif flood_alerts > 0:
            flood_type = FloodType.RIVER.value
        else:
            flood_type = FloodType.NO_FLOOD.value

        result = FloodPredictionResult(
            flood_likely=flood_likely,
            confidence=confidence,
            severity=severity,
            flood_type=flood_type,
            source_id="+".join(source_ids),
            data_provenance=("simulated" if any(p == "simulated" for p in provenance) else "live"),
            live_context=live_context,
        )
        # A gauge reporting in another unit (kcfs flow, or a reservoir in
        # metres) cannot fill a field in feet.
        if worst_gauge is not None and str(worst_gauge.get("observed_unit", "")).lower() == "ft":
            observed = worst_gauge.get("observed_value")
            forecast = worst_gauge.get("forecast_value")
            if observed is not None:
                result.river_stage_ft = float(observed)
            if forecast is not None and observed is not None:
                if float(forecast) > float(observed) + 0.1:
                    result.stage_trend = "rising"
                elif float(forecast) < float(observed) - 0.1:
                    result.stage_trend = "falling"
        return result

    def _determine_severity(
        self,
        indicators: float,
        result: FloodPredictionResult,
        gauge_status: str | None = None,
    ) -> str:
        """Determine flood severity.

        A gauge at or above its flood stage carries its own NWS category
        (minor/moderate/major/record), which is reported as measured. Without
        a gauge in flood, a likely flood (rain/soil evidence) is "moderate" at
        three or more indicators and "minor" otherwise: rain cannot place it
        on a river's category ladder. Nothing likely and no gauge in flood is
        "no_flood". The old rule added generic 10/5/2 ft offsets to an evidence
        count, so flash-flood rain plus a gauge at exactly minor read "record",
        and an empty record (stage 0 vs flood stage 0) read "minor".
        """
        if gauge_status in {"minor", "moderate", "major", "record"}:
            return str(gauge_status)
        if not result.flood_likely:
            return FloodSeverity.NO_FLOOD.value
        return FloodSeverity.MODERATE.value if indicators >= 3 else FloodSeverity.MINOR.value

    def _generate_warnings(self, result: FloodPredictionResult) -> list[str]:
        """Generate warning actions based on prediction."""
        warnings = []

        if result.severity == "record":
            warnings.append("RECORD FLOODING: Catastrophic conditions expected")
            warnings.append("Mandatory evacuation in effect")
            warnings.append("Do not attempt to drive through flooded areas")
        elif result.severity == "major":
            warnings.append("MAJOR FLOODING: Life-threatening conditions")
            warnings.append("Evacuate flood-prone areas immediately")
        elif result.severity == "moderate":
            warnings.append("MODERATE FLOODING: Significant impacts expected")
            warnings.append("Prepare for possible evacuation")
        elif result.severity == "minor":
            warnings.append("MINOR FLOODING: Some roads may be affected")
            warnings.append("Avoid low-lying areas")

        if result.flood_type == "flash":
            warnings.append("FLASH FLOOD: Move to higher ground immediately")

        if result.stage_trend == "rising_rapidly":
            warnings.append("River levels rising rapidly - monitor conditions")

        return warnings

    def _identify_evacuation_routes(self, result: FloodPredictionResult) -> list[str]:
        """Identify evacuation routes."""
        routes = []

        if result.severity in ["major", "record"]:
            routes.extend(
                [
                    "Primary: Interstate highways (elevated)",
                    "Secondary: State routes to higher elevation",
                    "Avoid: Low-lying roads and underpasses",
                ]
            )
        elif result.severity == "moderate":
            routes.extend(
                [
                    "Monitor road conditions",
                    "Identify alternate routes to higher ground",
                ]
            )

        return routes

    def _identify_shelters(self, result: FloodPredictionResult) -> list[str]:
        """Identify shelter locations."""
        shelters = []

        if result.severity in ["moderate", "major", "record"]:
            shelters.extend(
                [
                    "Local emergency shelters (check with authorities)",
                    "Schools and community centers on high ground",
                    "Hotels/motels outside flood zone",
                ]
            )

        return shelters

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        features = []

        features.extend(
            [
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
                np.median(data),
            ]
        )

        if len(data) > 1:
            trend = np.polyfit(np.arange(len(data.flatten())), data.flatten(), 1)[0]
            features.append(float(trend))
        else:
            features.append(0.0)

        features.extend(
            [
                np.percentile(data, 25),
                np.percentile(data, 75),
                np.percentile(data, 90),
                np.percentile(data, 95),
            ]
        )

        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features[:20], dtype=torch.float32)
