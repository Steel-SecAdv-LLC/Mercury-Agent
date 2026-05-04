"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Disaster Precursor Detector - Schumann Resonance + Multi-Source Correlation

Advanced disaster early warning using Schumann resonance anomalies:
- Earthquake precursor detection (electromagnetic signatures)
- Tsunami early warning (ionospheric perturbations)
- Volcanic eruption precursors
- Severe weather prediction (geomagnetic correlations)
- Climate pattern shifts

Integrates:
- Schumann resonance anomalies
- Seismic data correlation
- Geomagnetic indices
- Ionospheric disturbances
- Ancient pattern recognition (lunar/solar cycles)

⚠️ SIMULATION-BASED: For research. NOT a replacement for official warning systems.
Always defer to USGS, NOAA, and national seismological agencies.

Research sources:
- Electromagnetic earthquake precursor studies
- Ionospheric precursor research
- NOAA Space Weather Prediction Center
- USGS earthquake monitoring

"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector


@dataclass
class DisasterPrecursorResult:
    """Disaster precursor prediction results."""

    precursor_detected: bool
    confidence: float
    disaster_type: str
    risk_level: str

    time_to_event_hours: float | None = None
    estimated_magnitude: float | None = None
    affected_region: str | None = None

    schumann_anomaly: dict[str, Any] | None = None
    seismic_correlation: float | None = None
    geomagnetic_indicators: list[str] = field(default_factory=list)
    ionospheric_disturbance: bool = False

    early_warning_actions: list[str] = field(default_factory=list)
    monitoring_recommendations: list[str] = field(default_factory=list)


class EarthquakePrecursorAnalyzer(nn.Module):
    """
    Earthquake precursor detection using electromagnetic signatures.

    Analyzes Schumann+seismic correlations for earthquake prediction.
    """

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__()

        self.em_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.magnitude_predictor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.time_predictor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, em_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict earthquake from EM precursors.

        Args:
            em_features: Electromagnetic precursor features

        Returns:
            Tuple of (magnitude, time_to_event, confidence)
        """
        features = self.em_feature_extractor(em_features)

        magnitude = self.magnitude_predictor(features)
        time_to_event = self.time_predictor(features)
        confidence = self.confidence_head(features)

        return magnitude, time_to_event, confidence


class GeomageticCorrelator:
    """
    Correlate Schumann anomalies with geomagnetic indices.

    Uses Kp, Dst, and other indices for disaster correlation.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.kp_thresholds = {
            "quiet": (0, 3),
            "unsettled": (3, 4),
            "active": (4, 5),
            "minor_storm": (5, 6),
            "major_storm": (6, 7),
            "severe_storm": (7, 9),
        }

    def correlate_geomagnetic(
        self, schumann_anomaly: dict[str, Any], geomagnetic_data: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Correlate Schumann anomaly with geomagnetic activity.

        Args:
            schumann_anomaly: Schumann resonance anomaly data
            geomagnetic_data: Kp, Dst, and other indices

        Returns:
            Correlation analysis
        """
        if geomagnetic_data is None:
            geomagnetic_data = {"kp_index": 3.0, "dst_index": -20.0}

        kp = geomagnetic_data.get("kp_index", 3.0)
        dst = geomagnetic_data.get("dst_index", -20.0)

        geomagnetic_status = self._classify_geomagnetic_activity(kp)

        space_weather_factor = 1.0
        if kp > 7.0:
            space_weather_factor = 1.6
        elif kp > 5.0:
            space_weather_factor = 1.3

        dst_disturbance = dst < -50

        correlation_strength = 0.0

        if schumann_anomaly.get("frequency_anomaly") and geomagnetic_status in [
            "minor_storm",
            "major_storm",
            "severe_storm",
        ]:
            correlation_strength += 0.4

        if schumann_anomaly.get("amplitude_anomaly") and dst_disturbance:
            correlation_strength += 0.3

        if schumann_anomaly.get("power_spectrum_shift"):
            correlation_strength += 0.2

        correlation_strength = min(correlation_strength, 1.0)

        indicators = []
        if correlation_strength > 0.5:
            indicators.append("strong_geomagnetic_correlation")
        if dst_disturbance:
            indicators.append("ionospheric_current_disturbance")
        if kp > 6:
            indicators.append("severe_space_weather")

        return {
            "correlation_strength": correlation_strength,
            "geomagnetic_status": geomagnetic_status,
            "space_weather_factor": space_weather_factor,
            "indicators": indicators,
            "kp_index": kp,
            "dst_index": dst,
        }

    def _classify_geomagnetic_activity(self, kp: float) -> str:
        """Classify geomagnetic activity level."""
        for status, (min_kp, max_kp) in self.kp_thresholds.items():
            if min_kp <= kp < max_kp:
                return status
        return "severe_storm"


class IonosphericDisturbanceDetector:
    """
    Detect ionospheric disturbances from Schumann data.

    Ionospheric changes can precede earthquakes and tsunamis.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def detect_ionospheric_disturbance(
        self, schumann_data: dict[str, Any], tec_data: np.ndarray[Any, Any] | None = None
    ) -> dict[str, Any]:
        """
        Detect ionospheric disturbances.

        Args:
            schumann_data: Schumann resonance analysis
            tec_data: Total Electron Content measurements (optional)

        Returns:
            Ionospheric disturbance analysis
        """
        disturbance_detected = False
        indicators = []

        fundamental_deviation = schumann_data.get("fundamental_deviation", 0.0)

        if fundamental_deviation > 1.0:
            disturbance_detected = True
            indicators.append("significant_frequency_shift")

        if tec_data is not None and len(tec_data) > 1:
            tec_variation = np.std(tec_data)
            tec_trend = np.diff(tec_data)

            if tec_variation > 5.0:
                disturbance_detected = True
                indicators.append("high_tec_variability")

            if len(tec_trend) > 0 and np.mean(tec_trend) < -2.0:
                indicators.append("tec_depletion_observed")

        disturbance_level = len(indicators) / 3.0

        return {
            "disturbance_detected": disturbance_detected,
            "disturbance_level": min(disturbance_level, 1.0),
            "indicators": indicators,
            "potential_precursor": disturbance_detected
            and "significant_frequency_shift" in indicators,
        }


class SeismicCorrelator:
    """
    Correlate electromagnetic anomalies with seismic activity.

    Cross-references Schumann anomalies with seismic patterns.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def correlate_seismic(
        self, schumann_anomaly: dict[str, Any], seismic_data: np.ndarray[Any, Any] | None = None
    ) -> dict[str, Any]:
        """
        Correlate Schumann anomaly with seismic activity.

        Args:
            schumann_anomaly: Schumann anomaly data
            seismic_data: Seismic measurements (magnitude series)

        Returns:
            Seismic correlation analysis
        """
        if seismic_data is None or len(seismic_data) == 0:
            return {"correlation": 0.0, "significant": False}

        em_anomaly_strength = schumann_anomaly.get("risk_score", 0.0)

        recent_seismic = seismic_data[-10:] if len(seismic_data) > 10 else seismic_data

        seismic_activity = np.mean(recent_seismic)

        if em_anomaly_strength > 0.6 and seismic_activity > 4.0:
            correlation = min(em_anomaly_strength * (seismic_activity / 7.0), 1.0)
            significant = correlation > 0.7
        else:
            correlation = em_anomaly_strength * 0.5
            significant = False

        return {
            "correlation": correlation,
            "significant": significant,
            "recent_seismic_activity": float(seismic_activity),
            "precursor_likelihood": correlation * 0.8 if significant else 0.0,
        }


class DisasterPrecursorDetector:
    """
    Comprehensive disaster precursor detection system.

    Integrates Schumann resonance, geomagnetic, ionospheric, and seismic data for multi-modal
    disaster early warning.
    """

    def __init__(
        self,
        enable_earthquake: bool = True,
        enable_tsunami: bool = True,
        enable_geomagnetic: bool = True,
    ):
        self.enable_earthquake = enable_earthquake
        self.enable_tsunami = enable_tsunami
        self.enable_geomagnetic = enable_geomagnetic

        self.schumann_detector = SchumannResonanceDetector(
            sampling_rate=100.0, enable_ancient_correlation=True, golden_ratio_thresholds=True
        )

        self.earthquake_analyzer = EarthquakePrecursorAnalyzer() if enable_earthquake else None
        self.geomagnetic_correlator = GeomageticCorrelator() if enable_geomagnetic else None
        self.ionospheric_detector = IonosphericDisturbanceDetector()
        self.seismic_correlator = SeismicCorrelator()

        self.logger = logging.getLogger(__name__)

    def detect_disaster_precursor(self, precursor_data: dict[str, Any]) -> DisasterPrecursorResult:
        """
        Comprehensive disaster precursor detection.

        Args:
            precursor_data: Multi-source precursor data including:
                - elf_signal: Schumann resonance measurements
                - seismic_data: Optional seismic time series
                - geomagnetic_data: Optional Kp/Dst indices
                - tec_data: Optional ionospheric TEC data
                - temporal_history: Historical ELF measurements

        Returns:
            Disaster precursor prediction
        """
        result = DisasterPrecursorResult(
            precursor_detected=False,
            confidence=0.0,
            disaster_type="none",
            risk_level="low",
        )

        elf_signal = precursor_data.get("elf_signal")
        if elf_signal is None:
            self.logger.warning("No ELF signal provided")
            return result

        schumann_result = self.schumann_detector.detect_resonance_anomaly(
            elf_signal,
            temporal_history=precursor_data.get("temporal_history"),
            metadata=precursor_data.get("metadata"),
        )

        result.schumann_anomaly = {
            "anomaly_detected": schumann_result.anomaly_detected,
            "anomaly_type": schumann_result.anomaly_type,
            "fundamental_deviation": schumann_result.fundamental_deviation,
            "risk_score": schumann_result.risk_score,
        }

        if schumann_result.anomaly_detected and schumann_result.risk_score > 0.6:
            result.precursor_detected = True
            result.confidence = schumann_result.confidence

        if self.enable_geomagnetic and "geomagnetic_data" in precursor_data:
            if self.geomagnetic_correlator is None:
                raise RuntimeError("Geomagnetic correlator not initialized")
            geo_correlation = self.geomagnetic_correlator.correlate_geomagnetic(
                result.schumann_anomaly, precursor_data["geomagnetic_data"]
            )
            result.geomagnetic_indicators = geo_correlation["indicators"]

            if geo_correlation["correlation_strength"] > 0.6:
                result.confidence = max(result.confidence, geo_correlation["correlation_strength"])

        ionospheric_result = self.ionospheric_detector.detect_ionospheric_disturbance(
            result.schumann_anomaly, precursor_data.get("tec_data")
        )
        result.ionospheric_disturbance = ionospheric_result["disturbance_detected"]

        if ionospheric_result.get("potential_precursor"):
            result.precursor_detected = True
            result.disaster_type = "earthquake_or_volcanic"

        if "seismic_data" in precursor_data:
            seismic_correlation = self.seismic_correlator.correlate_seismic(
                result.schumann_anomaly, precursor_data["seismic_data"]
            )
            result.seismic_correlation = seismic_correlation["correlation"]

            if seismic_correlation["significant"]:
                result.precursor_detected = True
                result.disaster_type = "earthquake"
                result.time_to_event_hours = self._estimate_time_to_event(
                    schumann_result.risk_score, seismic_correlation["correlation"]
                )

        if self.enable_earthquake and "em_features" in precursor_data:
            eq_prediction = self._predict_earthquake(precursor_data["em_features"])
            result.estimated_magnitude = eq_prediction["magnitude"]
            result.time_to_event_hours = eq_prediction["time_to_event_hours"]
            result.confidence = max(result.confidence, eq_prediction["confidence"])

        if self.enable_tsunami and result.disaster_type == "earthquake":
            if result.estimated_magnitude and result.estimated_magnitude > 6.5:
                result.disaster_type = "earthquake_tsunami_risk"

        result.risk_level = self._assess_risk_level(result)
        result.early_warning_actions = self._generate_early_warning_actions(result)
        result.monitoring_recommendations = self._generate_monitoring_recommendations(result)

        self.logger.info(
            f"Disaster precursor: {result.disaster_type}, "
            f"confidence={result.confidence:.2f}, risk={result.risk_level}"
        )

        return result

    def _predict_earthquake(self, em_features: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Predict earthquake from EM features."""
        features_tensor = torch.tensor(em_features, dtype=torch.float32).unsqueeze(0)

        if self.earthquake_analyzer is None:
            raise RuntimeError("Earthquake analyzer not initialized")
        self.earthquake_analyzer.eval()
        with torch.no_grad():
            magnitude, time_to_event, confidence = self.earthquake_analyzer(features_tensor)

        magnitude_richter = float(magnitude[0].item()) * 9.0
        time_hours = float(time_to_event[0].item()) * 72.0

        return {
            "magnitude": magnitude_richter,
            "time_to_event_hours": time_hours,
            "confidence": float(confidence[0].item()),
        }

    def _estimate_time_to_event(self, risk_score: float, correlation: float) -> float:
        """Estimate time to potential event."""
        base_time = 48.0

        urgency_factor = (risk_score + correlation) / 2.0

        estimated_hours = base_time * (1.0 - urgency_factor * 0.8)

        return max(estimated_hours, 2.0)

    def _assess_risk_level(self, result: DisasterPrecursorResult) -> str:
        """Assess overall disaster risk level."""
        if not result.precursor_detected:
            return "low"

        if result.confidence > 0.8:
            return "critical"
        elif result.confidence > 0.6:
            return "high"
        elif result.confidence > 0.4:
            return "moderate"
        else:
            return "low"

    def _generate_early_warning_actions(self, result: DisasterPrecursorResult) -> list[str]:
        """Generate early warning actions."""
        actions = []

        if result.risk_level in ["critical", "high"]:
            actions.append("ALERT: Potential disaster precursor detected")
            actions.append("Notify emergency management agencies")
            actions.append("Activate early warning systems")

            if result.disaster_type == "earthquake":
                actions.append("Prepare for seismic event")
                actions.append("Review building safety protocols")

            if "tsunami" in result.disaster_type:
                actions.append("TSUNAMI WARNING: Coastal evacuation may be required")
                actions.append("Activate tsunami warning centers")

        elif result.risk_level == "moderate":
            actions.append("Enhanced monitoring recommended")
            actions.append("Inform disaster preparedness teams")

        return actions

    def _generate_monitoring_recommendations(self, result: DisasterPrecursorResult) -> list[str]:
        """Generate monitoring recommendations."""
        recs = []

        if result.ionospheric_disturbance:
            recs.append("Increase ionospheric monitoring frequency")
            recs.append("Deploy additional TEC measurement stations")

        if result.seismic_correlation and result.seismic_correlation > 0.5:
            recs.append("Intensify seismic network monitoring")
            recs.append("Check seismometer calibrations")

        if result.geomagnetic_indicators:
            recs.append("Monitor space weather conditions")
            recs.append("Track solar activity and CMEs")

        recs.append("Continue Schumann resonance monitoring")
        recs.append("Cross-reference with global monitoring networks")

        return recs
