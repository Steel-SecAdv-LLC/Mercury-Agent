"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Hurricane/Cyclone/Typhoon Detector - Tropical Cyclone Monitoring System

Comprehensive tropical cyclone detection for humanitarian early warning:
- Sea surface temperature (SST) monitoring
- Wind speed pattern analysis
- Barometric pressure tracking
- Saffir-Simpson scale classification
- Resonance engine for frequency amplification of storm signals
- Cross-domain fusion with flood/tornado detectors

Integrations:
- Satellite SST data processing (GOES, Himawari)
- Reconnaissance aircraft data (Hurricane Hunters)
- Buoy network observations
- 3R mechanism for adaptive storm tracking
- Storm surge prediction models

Research sources:
- NOAA National Hurricane Center (NHC)
- Joint Typhoon Warning Center (JTWC)
- World Meteorological Organization (WMO)
- Academic research on tropical cyclone intensification

⚠️ SIMULATION-BASED: For research/development. NOT a replacement for official
hurricane centers (NHC, JTWC). Always defer to official hurricane warnings.

Performance: Enhanced tracking via resonance frequency amplification + recursive analysis
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy.fft import fft
from torch import nn

from omni_anomaly_engine.core.three_r_mechanism import (
    RecursionEngine,
    RefactoringEngine,
    ResonanceEngine,
)
from omni_anomaly_engine.utils.rng import DeterministicRNG, get_global_rng


class SaffirSimpsonCategory(Enum):
    """Saffir-Simpson Hurricane Wind Scale categories"""

    TROPICAL_DEPRESSION = "tropical_depression"
    TROPICAL_STORM = "tropical_storm"
    CATEGORY_1 = "category_1"
    CATEGORY_2 = "category_2"
    CATEGORY_3 = "category_3"
    CATEGORY_4 = "category_4"
    CATEGORY_5 = "category_5"
    NO_CYCLONE = "no_cyclone"


class CycloneType(Enum):
    """Types of tropical cyclones by basin"""

    HURRICANE = "hurricane"  # Atlantic, Eastern Pacific
    TYPHOON = "typhoon"  # Western Pacific
    CYCLONE = "cyclone"  # Indian Ocean, South Pacific
    NO_CYCLONE = "no_cyclone"


@dataclass
class HurricanePredictionResult:
    """Hurricane/cyclone prediction results"""

    cyclone_detected: bool
    confidence: float
    category: str
    cyclone_type: str

    max_wind_speed_kt: float = 0.0
    min_pressure_mb: float = 1013.0
    eye_diameter_nm: float | None = None

    sst_anomaly_c: float = 0.0
    wind_shear_kt: float = 0.0
    ocean_heat_content: float = 0.0

    rapid_intensification: bool = False
    intensification_rate_kt_24h: float = 0.0

    resonance_score: float = 0.0
    frequency_amplification: float = 0.0
    harmonic_patterns: list[float] = field(default_factory=list)

    storm_surge_risk: str = "low"
    rainfall_potential_inches: float = 0.0

    track_forecast: list[dict] = field(default_factory=list)
    landfall_probability: float = 0.0
    time_to_landfall_hours: float | None = None

    warning_actions: list[str] = field(default_factory=list)
    evacuation_zones: list[str] = field(default_factory=list)


class SeaSurfaceTemperatureAnalyzer:
    """
    Sea surface temperature analysis for cyclone development potential.

    SST >= 26.5°C is generally required for tropical cyclone formation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.formation_threshold_c = 26.5
        self.intensification_threshold_c = 28.0

    def analyze_sst(self, sst_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze sea surface temperature for cyclone potential.

        Args:
            sst_data: SST measurements and climatology

        Returns:
            SST analysis results
        """
        current_sst = sst_data.get("sst_celsius", 25.0)
        climatology_sst = sst_data.get("climatology_celsius", 26.0)
        depth_26c = sst_data.get("depth_26c_m", 50.0)

        sst_anomaly = current_sst - climatology_sst

        favorable_formation = current_sst >= self.formation_threshold_c
        favorable_intensification = current_sst >= self.intensification_threshold_c

        ohc = self._estimate_ocean_heat_content(current_sst, depth_26c)

        if current_sst >= 29.0 and ohc > 80:
            potential = "very_high"
        elif current_sst >= 28.0 and ohc > 60:
            potential = "high"
        elif current_sst >= 26.5 and ohc > 40:
            potential = "moderate"
        elif current_sst >= 26.0:
            potential = "marginal"
        else:
            potential = "unfavorable"

        return {
            "current_sst_c": float(current_sst),
            "sst_anomaly_c": float(sst_anomaly),
            "ocean_heat_content": float(ohc),
            "favorable_formation": favorable_formation,
            "favorable_intensification": favorable_intensification,
            "development_potential": potential,
            "depth_26c_m": float(depth_26c),
        }

    def _estimate_ocean_heat_content(self, sst: float, depth_26c: float) -> float:
        """Estimate ocean heat content (kJ/cm²)."""
        if sst < 26.0:
            return 0.0
        ohc = (sst - 26.0) * depth_26c * 0.1
        return min(ohc, 150.0)


class WindPatternAnalyzer(nn.Module):
    """
    Wind pattern analyzer for cyclone structure detection.

    Uses CNN + LSTM to identify organized convection and eye formation.
    """

    def __init__(self, input_channels: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.lstm = nn.LSTM(
            input_size=64 * 8 * 8,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.intensity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, wind_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze wind field patterns.

        Args:
            wind_field: Wind field data (batch, time, channels, height, width)

        Returns:
            Tuple of (max_wind_speed, category_logits)
        """
        batch_size, seq_len = wind_field.shape[:2]

        encoded_frames = []
        for t in range(seq_len):
            frame = wind_field[:, t]
            encoded = self.conv_encoder(frame)
            encoded_frames.append(encoded.view(batch_size, -1))

        sequence = torch.stack(encoded_frames, dim=1)

        lstm_out, _ = self.lstm(sequence)
        final_state = lstm_out[:, -1]

        max_wind = self.intensity_predictor(final_state)
        category_logits = self.category_classifier(final_state)

        return max_wind, category_logits


class PressureTracker:
    """
    Central pressure tracking for cyclone intensity monitoring.

    Lower central pressure indicates stronger cyclone.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_pressure(self, pressure_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze central pressure for intensity assessment.

        Args:
            pressure_data: Central pressure measurements

        Returns:
            Pressure analysis results
        """
        central_pressure = pressure_data.get("central_pressure_mb", 1013.0)
        environmental_pressure = pressure_data.get("environmental_pressure_mb", 1013.0)
        pressure_history = pressure_data.get("pressure_history_mb", [])

        pressure_deficit = environmental_pressure - central_pressure

        if len(pressure_history) >= 2:
            pressure_change_24h = pressure_history[0] - pressure_history[-1]
            rapid_intensification = pressure_change_24h > 30
        else:
            pressure_change_24h = 0.0
            rapid_intensification = False

        estimated_wind = self._pressure_wind_relationship(central_pressure)

        return {
            "central_pressure_mb": float(central_pressure),
            "pressure_deficit_mb": float(pressure_deficit),
            "pressure_change_24h_mb": float(pressure_change_24h),
            "rapid_intensification": rapid_intensification,
            "estimated_max_wind_kt": float(estimated_wind),
        }

    def _pressure_wind_relationship(self, pressure: float) -> float:
        """
        Estimate max wind from central pressure using Dvorak relationship.

        V_max = 6.7 * (1013 - P)^0.644
        """
        if pressure >= 1013:
            return 0.0
        deficit = 1013 - pressure
        wind = 6.7 * (deficit**0.644)
        return min(wind, 200.0)


class ResonanceFrequencyAmplifier:
    """
    FFT-based resonance frequency amplifier for storm signal detection.

    Implements the Resonance Engine component of 3R for amplifying
    weak cyclone signatures in atmospheric data.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cyclone_frequencies = [0.05, 0.1, 0.2, 0.5]

    def amplify_signals(self, signal: np.ndarray) -> dict[str, Any]:
        """
        Amplify cyclone-characteristic frequency signals.

        Args:
            signal: Input signal array

        Returns:
            Amplification results
        """
        if len(signal) < 16:
            return {
                "resonance_score": 0.0,
                "amplification_factor": 1.0,
                "harmonic_patterns": [],
            }

        signal_flat = signal.flatten()

        fft_result = fft(signal_flat)
        power_spectrum = np.abs(fft_result) ** 2

        amplified_power = np.zeros_like(power_spectrum)
        harmonic_patterns = []

        for target_freq in self.cyclone_frequencies:
            idx = int(target_freq * len(signal_flat) / 10)
            if 0 <= idx < len(power_spectrum):
                window_start = max(0, idx - 2)
                window_end = min(len(power_spectrum), idx + 3)
                local_power = np.sum(power_spectrum[window_start:window_end])

                amplification = 2.0
                amplified_power[window_start:window_end] = (
                    power_spectrum[window_start:window_end] * amplification
                )

                if local_power > np.mean(power_spectrum) * 1.5:
                    harmonic_patterns.append(float(target_freq))

        total_original = np.sum(power_spectrum) + 1e-10
        total_amplified = np.sum(amplified_power) + 1e-10
        amplification_factor = total_amplified / total_original

        resonance_score = len(harmonic_patterns) / len(self.cyclone_frequencies)

        return {
            "resonance_score": float(resonance_score),
            "amplification_factor": float(amplification_factor),
            "harmonic_patterns": harmonic_patterns,
            "spectral_energy": float(total_original),
        }


class HurricaneDetector:
    """
    Comprehensive hurricane/cyclone/typhoon detection system.

    Integrates SST analysis, wind pattern detection, pressure tracking,
    and 3R mechanism for multi-parameter tropical cyclone prediction.

    Deep 3R Integration:
    - RecursionEngine: Hierarchical multi-scale feature extraction
    - ResonanceEngine: FFT-based frequency-domain anomaly detection
    - RefactoringEngine: Dynamic model optimization and code analysis
    """

    def __init__(
        self,
        enable_sst: bool = True,
        enable_wind: bool = True,
        enable_pressure: bool = True,
        enable_resonance: bool = True,
        enable_recursion: bool = True,
        enable_refactoring: bool = True,
        rng: DeterministicRNG | None = None,
    ):
        self.enable_sst = enable_sst
        self.enable_wind = enable_wind
        self.enable_pressure = enable_pressure
        self.enable_resonance = enable_resonance
        self.enable_recursion = enable_recursion
        self.enable_refactoring = enable_refactoring
        self._rng = rng or get_global_rng()

        self.sst_analyzer = SeaSurfaceTemperatureAnalyzer() if enable_sst else None
        self.wind_analyzer = WindPatternAnalyzer() if enable_wind else None
        self.pressure_tracker = PressureTracker() if enable_pressure else None
        self.resonance_amplifier = ResonanceFrequencyAmplifier() if enable_resonance else None

        self.recursion_engine = RecursionEngine(max_depth=5)
        self.resonance_engine = ResonanceEngine(sampling_rate=1.0)
        self.refactoring_engine = RefactoringEngine()

        self.logger = logging.getLogger(__name__)

    def predict_hurricane(self, cyclone_data: dict[str, Any]) -> HurricanePredictionResult:
        """
        Comprehensive tropical cyclone prediction.

        Args:
            cyclone_data: Multi-parameter cyclone monitoring data including:
                - sst_data: Sea surface temperature measurements
                - wind_field: Wind pattern data
                - pressure_data: Central pressure measurements
                - signal_data: Raw signal for resonance analysis
                - basin: Ocean basin for cyclone type classification
                - metadata: Storm info, timestamps

        Returns:
            Hurricane prediction with category and recommendations
        """
        result = HurricanePredictionResult(
            cyclone_detected=False,
            confidence=0.0,
            category="no_cyclone",
            cyclone_type="no_cyclone",
        )

        indicators_detected = 0

        if self.enable_sst and "sst_data" in cyclone_data:
            sst_result = self.sst_analyzer.analyze_sst(cyclone_data["sst_data"])
            result.sst_anomaly_c = sst_result["sst_anomaly_c"]
            result.ocean_heat_content = sst_result["ocean_heat_content"]
            if sst_result["favorable_formation"]:
                indicators_detected += 1
            if sst_result["favorable_intensification"]:
                indicators_detected += 0.5

        if self.enable_pressure and "pressure_data" in cyclone_data:
            pressure_result = self.pressure_tracker.analyze_pressure(cyclone_data["pressure_data"])
            result.min_pressure_mb = pressure_result["central_pressure_mb"]
            result.max_wind_speed_kt = pressure_result["estimated_max_wind_kt"]
            result.rapid_intensification = pressure_result["rapid_intensification"]
            result.intensification_rate_kt_24h = pressure_result["pressure_change_24h_mb"]

            if pressure_result["pressure_deficit_mb"] > 20:
                indicators_detected += 2
            elif pressure_result["pressure_deficit_mb"] > 10:
                indicators_detected += 1

            if pressure_result["rapid_intensification"]:
                indicators_detected += 1

        if self.enable_resonance and "signal_data" in cyclone_data:
            resonance_result = self.resonance_amplifier.amplify_signals(cyclone_data["signal_data"])
            result.resonance_score = resonance_result["resonance_score"]
            result.frequency_amplification = resonance_result["amplification_factor"]
            result.harmonic_patterns = resonance_result["harmonic_patterns"]
            if resonance_result["resonance_score"] > 0.5:
                indicators_detected += 0.5

        if self.enable_recursion and "signal_data" in cyclone_data:
            hierarchical_features = self.recursion_engine.hierarchical_feature_extraction(
                cyclone_data["signal_data"], num_levels=3
            )
            if len(hierarchical_features) > 0:
                multi_scale_variance = np.mean([np.var(f) for f in hierarchical_features])
                if multi_scale_variance > 0.5:
                    indicators_detected += 0.3

        if "signal_data" in cyclone_data:
            resonance_anomalies = self.resonance_engine.detect_resonance_anomalies(
                cyclone_data["signal_data"], threshold_std=2.5
            )
            if resonance_anomalies["is_anomalous"]:
                indicators_detected += 0.4
                result.harmonic_patterns.extend(
                    [float(f) for f in resonance_anomalies["anomalous_frequencies"][:3]]
                )

        if self.enable_refactoring and "observed_data" in cyclone_data:
            initial_prediction = {
                "confidence": result.confidence,
                "indicators": indicators_detected,
                "category": result.category,
            }
            refactor_result = self.refactoring_engine.detect_code_anomalies(str(initial_prediction))
            if refactor_result.get("anomaly_score", 0) > 0.5:
                indicators_detected += 0.2

        result.cyclone_detected = indicators_detected >= 2
        result.confidence = min(indicators_detected / 5.0, 1.0)
        result.category = self._classify_category(result.max_wind_speed_kt)
        result.cyclone_type = self._determine_cyclone_type(cyclone_data.get("basin", "atlantic"))

        result.storm_surge_risk = self._assess_storm_surge(result)
        result.rainfall_potential_inches = self._estimate_rainfall(result)

        result.warning_actions = self._generate_warnings(result)
        result.evacuation_zones = self._identify_evacuation_zones(result)

        self.logger.info(
            f"Hurricane prediction: {result.category}, "
            f"wind={result.max_wind_speed_kt:.0f}kt, confidence={result.confidence:.2f}"
        )

        return result

    def _classify_category(self, max_wind_kt: float) -> str:
        """Classify cyclone using Saffir-Simpson scale."""
        if max_wind_kt >= 157:
            return "category_5"
        elif max_wind_kt >= 130:
            return "category_4"
        elif max_wind_kt >= 111:
            return "category_3"
        elif max_wind_kt >= 96:
            return "category_2"
        elif max_wind_kt >= 74:
            return "category_1"
        elif max_wind_kt >= 39:
            return "tropical_storm"
        elif max_wind_kt >= 23:
            return "tropical_depression"
        else:
            return "no_cyclone"

    def _determine_cyclone_type(self, basin: str) -> str:
        """Determine cyclone type based on ocean basin."""
        basin_lower = basin.lower()
        if basin_lower in ["atlantic", "eastern_pacific", "central_pacific"]:
            return "hurricane"
        elif basin_lower in ["western_pacific", "south_china_sea"]:
            return "typhoon"
        elif basin_lower in ["indian", "south_pacific", "australian"]:
            return "cyclone"
        else:
            return "hurricane"

    def _assess_storm_surge(self, result: HurricanePredictionResult) -> str:
        """Assess storm surge risk based on intensity."""
        if result.category in ["category_4", "category_5"]:
            return "extreme"
        elif result.category == "category_3":
            return "high"
        elif result.category in ["category_1", "category_2"]:
            return "moderate"
        elif result.category == "tropical_storm":
            return "low"
        else:
            return "minimal"

    def _estimate_rainfall(self, result: HurricanePredictionResult) -> float:
        """Estimate potential rainfall in inches."""
        base_rainfall = {
            "category_5": 20.0,
            "category_4": 15.0,
            "category_3": 12.0,
            "category_2": 8.0,
            "category_1": 6.0,
            "tropical_storm": 4.0,
            "tropical_depression": 2.0,
            "no_cyclone": 0.0,
        }
        return base_rainfall.get(result.category, 0.0)

    def _generate_warnings(self, result: HurricanePredictionResult) -> list[str]:
        """Generate warning actions based on prediction."""
        warnings = []

        if result.category in ["category_4", "category_5"]:
            warnings.append("EXTREME DANGER: Life-threatening conditions expected")
            warnings.append("Mandatory evacuation may be ordered")
            warnings.append("Catastrophic damage expected")
        elif result.category == "category_3":
            warnings.append("MAJOR HURRICANE: Significant damage expected")
            warnings.append("Consider evacuation from surge-prone areas")
        elif result.category in ["category_1", "category_2"]:
            warnings.append("HURRICANE WARNING: Dangerous conditions expected")
            warnings.append("Secure property and prepare emergency supplies")
        elif result.category == "tropical_storm":
            warnings.append("TROPICAL STORM WARNING: Hazardous conditions possible")

        if result.rapid_intensification:
            warnings.append("RAPID INTENSIFICATION: Storm strengthening quickly")

        if result.storm_surge_risk in ["extreme", "high"]:
            warnings.append(f"STORM SURGE RISK: {result.storm_surge_risk.upper()}")

        return warnings

    def _identify_evacuation_zones(self, result: HurricanePredictionResult) -> list[str]:
        """Identify evacuation zones based on storm characteristics."""
        zones = []

        if result.category in ["category_3", "category_4", "category_5"]:
            zones.extend(["Zone A", "Zone B", "Zone C"])
        elif result.category in ["category_1", "category_2"]:
            zones.extend(["Zone A", "Zone B"])
        elif result.category == "tropical_storm":
            zones.append("Zone A")

        if result.storm_surge_risk == "extreme":
            zones.append("All coastal areas")

        return zones

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        features = []

        if self.enable_resonance:
            resonance = self.resonance_amplifier.amplify_signals(data)
            features.extend(
                [
                    resonance["resonance_score"],
                    resonance["amplification_factor"],
                    resonance["spectral_energy"] / 1e6,
                ]
            )

        features.extend(
            [
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
            ]
        )

        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features[:20], dtype=torch.float32)
