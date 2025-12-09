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
from __future__ import annotations

"""
Tornado Detector - Multi-Modal Severe Weather Monitoring

Comprehensive tornado detection for humanitarian early warning:
- Doppler radar pattern analysis (mesocyclone detection)
- Atmospheric pressure gradient monitoring
- Temperature/humidity anomaly detection
- Historical tornado alley correlation
- FFT-based resonance pattern analysis (3R integration)
- Recursive feature extraction for multi-scale patterns

Integrations:
- Doppler radar velocity data processing
- Surface observation networks (ASOS, AWOS)
- Upper-air soundings for atmospheric instability
- 3R mechanism for self-healing monitoring networks
- Cross-domain fusion with hurricane/flood detectors

Research sources:
- NOAA Storm Prediction Center
- National Weather Service
- NSSL (National Severe Storms Laboratory)
- Academic research on mesocyclone detection

⚠️ SIMULATION-BASED: For research/development. NOT a replacement for official
weather services (NWS, SPC). Always defer to official tornado warnings.

Performance: Enhanced detection via FFT resonance + recursive feature extraction
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


class TornadoIntensity(Enum):
    """Enhanced Fujita (EF) Scale tornado intensity levels"""

    EF0 = "ef0_weak"
    EF1 = "ef1_moderate"
    EF2 = "ef2_significant"
    EF3 = "ef3_severe"
    EF4 = "ef4_devastating"
    EF5 = "ef5_incredible"
    NO_TORNADO = "no_tornado"


class TornadoThreatLevel(Enum):
    """Tornado threat assessment levels"""

    NONE = "none"
    MARGINAL = "marginal"
    SLIGHT = "slight"
    ENHANCED = "enhanced"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class TornadoPredictionResult:
    """Tornado prediction results"""

    tornado_likely: bool
    confidence: float
    threat_level: str
    estimated_intensity: str

    mesocyclone_detected: bool = False
    rotation_velocity_ms: float = 0.0
    pressure_drop_mb: float = 0.0

    cape_value: float = 0.0  # Convective Available Potential Energy
    helicity_value: float = 0.0  # Storm Relative Helicity
    wind_shear_detected: bool = False

    resonance_score: float = 0.0
    recursion_depth: int = 0
    harmonic_anomalies: list[float] = field(default_factory=list)

    tornado_alley_correlation: float = 0.0
    time_to_touchdown_minutes: float | None = None

    warning_actions: list[str] = field(default_factory=list)
    shelter_recommendations: list[str] = field(default_factory=list)


class DopplerRadarAnalyzer(nn.Module):
    """
    Doppler radar velocity pattern analyzer for mesocyclone detection.

    Uses LSTM + attention to identify rotation signatures in radar data.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.mesocyclone_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.rotation_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(
        self, radar_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analyze Doppler radar data for mesocyclone signatures.

        Args:
            radar_sequence: Time series of radar velocity data (batch, seq_len, features)

        Returns:
            Tuple of (mesocyclone_probability, rotation_velocity, attention_weights)
        """
        lstm_out, _ = self.lstm(radar_sequence)

        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(lstm_out * attention_weights, dim=1)

        meso_prob = self.mesocyclone_classifier(context)
        rotation_vel = self.rotation_estimator(context)

        return meso_prob, rotation_vel, attention_weights.squeeze(-1)


class AtmosphericInstabilityAnalyzer:
    """
    Atmospheric instability analysis for tornado potential.

    Monitors CAPE, helicity, wind shear, and other severe weather parameters.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.cape_threshold = 1500.0  # J/kg for significant instability
        self.helicity_threshold = 150.0  # m²/s² for tornado potential
        self.shear_threshold = 40.0  # knots for significant shear

    def analyze_instability(self, atmospheric_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze atmospheric instability parameters.

        Args:
            atmospheric_data: Atmospheric sounding and surface data

        Returns:
            Instability analysis results
        """
        cape = atmospheric_data.get("cape_j_kg", 0.0)
        cin = atmospheric_data.get("cin_j_kg", 0.0)
        helicity = atmospheric_data.get("srh_m2_s2", 0.0)
        bulk_shear = atmospheric_data.get("bulk_shear_kt", 0.0)
        lcl_height = atmospheric_data.get("lcl_m", 2000.0)
        # lfc_height reserved for future convective initiation analysis
        _ = atmospheric_data.get("lfc_m", 3000.0)

        significant_cape = cape > self.cape_threshold
        significant_helicity = helicity > self.helicity_threshold
        significant_shear = bulk_shear > self.shear_threshold

        sig_tornado_param = self._compute_stp(cape, helicity, bulk_shear, lcl_height)

        if sig_tornado_param > 4.0:
            tornado_potential = "high"
        elif sig_tornado_param > 2.0:
            tornado_potential = "moderate"
        elif sig_tornado_param > 1.0:
            tornado_potential = "slight"
        else:
            tornado_potential = "low"

        return {
            "cape": float(cape),
            "cin": float(cin),
            "helicity": float(helicity),
            "bulk_shear": float(bulk_shear),
            "significant_cape": significant_cape,
            "significant_helicity": significant_helicity,
            "significant_shear": significant_shear,
            "sig_tornado_param": float(sig_tornado_param),
            "tornado_potential": tornado_potential,
            "lcl_height_m": float(lcl_height),
        }

    def _compute_stp(self, cape: float, helicity: float, shear: float, lcl: float) -> float:
        """
        Compute Significant Tornado Parameter (STP).

        STP = (CAPE/1500) * (SRH/150) * (Shear/40) * ((2000-LCL)/1000)
        """
        cape_term = min(cape / 1500.0, 2.0)
        helicity_term = min(helicity / 150.0, 2.0)
        shear_term = min(shear / 40.0, 2.0)
        lcl_term = max(0.0, min((2000.0 - lcl) / 1000.0, 1.0))

        stp = cape_term * helicity_term * shear_term * lcl_term

        return stp


class PressureGradientMonitor:
    """
    Atmospheric pressure gradient monitoring for tornado precursors.

    Detects rapid pressure drops associated with mesocyclone development.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.rapid_drop_threshold = 4.0  # mb in 15 minutes

    def analyze_pressure(self, pressure_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze pressure gradients for tornado signatures.

        Args:
            pressure_data: Time series of pressure measurements

        Returns:
            Pressure analysis results
        """
        pressure_series = pressure_data.get("pressure_mb", np.array([]))
        # timestamps reserved for future temporal correlation analysis
        _ = pressure_data.get("timestamps", np.array([]))

        if len(pressure_series) < 2:
            return {
                "rapid_drop_detected": False,
                "max_pressure_drop": 0.0,
                "pressure_trend": "stable",
            }

        pressure_changes = np.diff(pressure_series)
        max_drop = abs(min(pressure_changes)) if len(pressure_changes) > 0 else 0.0

        rapid_drop = max_drop > self.rapid_drop_threshold

        if np.mean(pressure_changes) < -0.5:
            trend = "falling_rapidly"
        elif np.mean(pressure_changes) < -0.1:
            trend = "falling"
        elif np.mean(pressure_changes) > 0.1:
            trend = "rising"
        else:
            trend = "stable"

        return {
            "rapid_drop_detected": rapid_drop,
            "max_pressure_drop": float(max_drop),
            "pressure_trend": trend,
            "current_pressure": float(pressure_series[-1]) if len(pressure_series) > 0 else 0.0,
        }


class ResonancePatternAnalyzer:
    """
    FFT-based resonance pattern analyzer for tornado signatures.

    Implements the Resonance Engine component of 3R for detecting
    characteristic frequency patterns in atmospheric data.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.tornado_frequencies = [0.1, 0.5, 1.0, 2.0]  # Hz characteristic frequencies

    def analyze_resonance(self, signal: np.ndarray[Any, Any]) -> dict[str, Any]:
        """
        Analyze signal for tornado-characteristic resonance patterns.

        Args:
            signal: Input signal array (radar, pressure, etc.)

        Returns:
            Resonance analysis results
        """
        if len(signal) < 16:
            return {
                "resonance_score": 0.0,
                "harmonic_anomalies": [],
                "dominant_frequency": 0.0,
            }

        signal_flat = signal.flatten()

        fft_result = fft(signal_flat)
        power_spectrum = np.abs(fft_result) ** 2
        frequencies = np.fft.fftfreq(len(signal_flat))

        harmonic_anomalies = []
        for target_freq in self.tornado_frequencies:
            idx = int(target_freq * len(signal_flat) / 10)
            if 0 <= idx < len(power_spectrum):
                local_power = power_spectrum[idx]
                mean_power = np.mean(power_spectrum)
                if local_power > 2 * mean_power:
                    harmonic_anomalies.append(float(target_freq))

        total_power = np.sum(power_spectrum) + 1e-10
        anomaly_power = sum(
            power_spectrum[int(f * len(signal_flat) / 10)]
            for f in harmonic_anomalies
            if 0 <= int(f * len(signal_flat) / 10) < len(power_spectrum)
        )

        resonance_score = min(anomaly_power / total_power * 10, 1.0)

        dominant_idx = np.argmax(power_spectrum[1:]) + 1 if len(power_spectrum) > 1 else 0
        dominant_freq = abs(frequencies[dominant_idx]) if dominant_idx < len(frequencies) else 0.0

        return {
            "resonance_score": float(resonance_score),
            "harmonic_anomalies": harmonic_anomalies,
            "dominant_frequency": float(dominant_freq),
            "spectral_energy": float(total_power),
        }


class RecursiveFeatureExtractor:
    """
    Recursive feature extraction for multi-scale tornado pattern detection.

    Implements the Recursion Engine component of 3R for hierarchical
    pattern analysis at progressively finer scales.
    """

    def __init__(self, max_depth: int = 4, decay_factor: float = 0.8) -> None:
        self.max_depth = max_depth
        self.decay_factor = decay_factor

    def extract_recursive_features(
        self, data: np.ndarray[Any, Any], depth: int = 0
    ) -> tuple[np.ndarray[Any, Any], int]:
        """
        Recursively extract features at multiple scales.

        Args:
            data: Input data array
            depth: Current recursion depth

        Returns:
            Tuple of (extracted_features, max_depth_reached)
        """
        if depth >= self.max_depth or len(data) < 4:
            return self._base_features(data), depth

        current_features = self._base_features(data)

        downsampled = data[::2] if len(data) > 1 else data
        recursive_features, max_depth = self.extract_recursive_features(downsampled, depth + 1)

        combined = current_features + self.decay_factor * recursive_features

        return combined, max_depth

    def _base_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract base statistical features."""
        if len(data) == 0:
            return np.zeros(6)

        features = np.array(
            [
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
                np.median(data),
                np.percentile(data, 75) - np.percentile(data, 25),
            ]
        )

        return features


class TornadoDetector:
    """
    Comprehensive tornado detection system.

    Integrates Doppler radar analysis, atmospheric instability monitoring,
    pressure gradient detection, and 3R mechanism (Resonance-Recursion-Refactoring)
    for multi-parameter tornado prediction.

    Deep 3R Integration:
    - RecursionEngine: Hierarchical multi-scale feature extraction
    - ResonanceEngine: FFT-based frequency-domain anomaly detection
    - RefactoringEngine: Dynamic model optimization and code analysis
    """

    def __init__(
        self,
        enable_radar: bool = True,
        enable_atmospheric: bool = True,
        enable_pressure: bool = True,
        enable_resonance: bool = True,
        enable_recursion: bool = True,
        enable_refactoring: bool = True,
        rng: DeterministicRNG | None = None,
    ):
        self.enable_radar = enable_radar
        self.enable_atmospheric = enable_atmospheric
        self.enable_pressure = enable_pressure
        self.enable_resonance = enable_resonance
        self.enable_recursion = enable_recursion
        self.enable_refactoring = enable_refactoring
        self._rng = rng or get_global_rng()

        self.radar_analyzer = DopplerRadarAnalyzer() if enable_radar else None
        self.atmospheric_analyzer = AtmosphericInstabilityAnalyzer() if enable_atmospheric else None
        self.pressure_monitor = PressureGradientMonitor() if enable_pressure else None
        self.resonance_analyzer = ResonancePatternAnalyzer() if enable_resonance else None
        self.recursive_extractor = RecursiveFeatureExtractor() if enable_recursion else None

        self.recursion_engine = RecursionEngine(max_depth=5)
        self.resonance_engine = ResonanceEngine(sampling_rate=1.0)
        self.refactoring_engine = RefactoringEngine()

        self.logger = logging.getLogger(__name__)

        self.tornado_alley_states = [
            "TX",
            "OK",
            "KS",
            "NE",
            "SD",
            "ND",
            "IA",
            "MO",
            "AR",
            "LA",
            "MS",
            "AL",
        ]

    def predict_tornado(self, weather_data: dict[str, Any]) -> TornadoPredictionResult:
        """
        Comprehensive tornado prediction.

        Args:
            weather_data: Multi-parameter weather monitoring data including:
                - radar_sequence: Doppler radar velocity data
                - atmospheric_data: CAPE, helicity, shear measurements
                - pressure_data: Surface pressure time series
                - location: State/region for tornado alley correlation
                - metadata: Station info, timestamps

        Returns:
            Tornado prediction with threat level and recommendations
        """
        result = TornadoPredictionResult(
            tornado_likely=False,
            confidence=0.0,
            threat_level="none",
            estimated_intensity="no_tornado",
        )

        indicators_detected = 0

        if self.enable_radar and "radar_sequence" in weather_data:
            radar_result = self._analyze_radar(weather_data["radar_sequence"])
            result.mesocyclone_detected = radar_result["mesocyclone_detected"]
            result.rotation_velocity_ms = radar_result["rotation_velocity"]
            if radar_result["mesocyclone_detected"]:
                indicators_detected += 2
                result.confidence = max(result.confidence, radar_result["confidence"])

        indicators_float: float = float(indicators_detected)

        if self.enable_atmospheric and "atmospheric_data" in weather_data:
            assert self.atmospheric_analyzer is not None
            atmos_result = self.atmospheric_analyzer.analyze_instability(
                weather_data["atmospheric_data"]
            )
            result.cape_value = atmos_result["cape"]
            result.helicity_value = atmos_result["helicity"]
            result.wind_shear_detected = atmos_result["significant_shear"]
            if atmos_result["tornado_potential"] in ["moderate", "high"]:
                indicators_float += 1

        if self.enable_pressure and "pressure_data" in weather_data:
            assert self.pressure_monitor is not None
            pressure_result = self.pressure_monitor.analyze_pressure(weather_data["pressure_data"])
            result.pressure_drop_mb = pressure_result["max_pressure_drop"]
            if pressure_result["rapid_drop_detected"]:
                indicators_float += 1

        if self.enable_resonance and "signal_data" in weather_data:
            assert self.resonance_analyzer is not None
            resonance_result = self.resonance_analyzer.analyze_resonance(
                weather_data["signal_data"]
            )
            result.resonance_score = resonance_result["resonance_score"]
            result.harmonic_anomalies = resonance_result["harmonic_anomalies"]
            if resonance_result["resonance_score"] > 0.6:
                indicators_float += 0.5

        if self.enable_recursion and "signal_data" in weather_data:
            assert self.recursive_extractor is not None
            _, depth = self.recursive_extractor.extract_recursive_features(
                weather_data["signal_data"]
            )
            result.recursion_depth = depth

            hierarchical_features = self.recursion_engine.hierarchical_feature_extraction(
                weather_data["signal_data"], num_levels=3
            )
            if len(hierarchical_features) > 0:
                multi_scale_variance = np.mean([np.var(f) for f in hierarchical_features])
                if multi_scale_variance > 0.5:
                    indicators_float += 0.3

        if "signal_data" in weather_data:
            resonance_anomalies = self.resonance_engine.detect_resonance_anomalies(
                weather_data["signal_data"], threshold_std=2.5
            )
            if resonance_anomalies["is_anomalous"]:
                indicators_float += 0.4
                result.harmonic_anomalies.extend(
                    [float(f) for f in resonance_anomalies["anomalous_frequencies"][:3]]
                )

        if self.enable_refactoring and "observed_data" in weather_data:
            initial_prediction_str = str({
                "confidence": result.confidence,
                "indicators": indicators_float,
            })
            # detect_code_anomalies expects a callable, so we pass a lambda
            refactor_result = self.refactoring_engine.detect_code_anomalies(
                lambda: initial_prediction_str
            )
            if refactor_result.get("anomaly_score", 0) > 0.5:
                indicators_float += 0.2

        location = weather_data.get("location", {})
        state = location.get("state", "")
        if state in self.tornado_alley_states:
            result.tornado_alley_correlation = 0.8
            indicators_float += 0.3

        result.tornado_likely = indicators_float >= 2
        result.confidence = min(indicators_float / 4.0, 1.0)
        result.threat_level = self._determine_threat_level(indicators_float, result)
        result.estimated_intensity = self._estimate_intensity(result)

        result.warning_actions = self._generate_warnings(result)
        result.shelter_recommendations = self._generate_shelter_advice(result)

        self.logger.info(
            f"Tornado prediction: {result.threat_level}, "
            f"indicators={indicators_float:.1f}, confidence={result.confidence:.2f}"
        )

        return result

    def _analyze_radar(self, radar_sequence: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Analyze Doppler radar data for mesocyclone signatures."""
        seq_tensor = torch.tensor(radar_sequence, dtype=torch.float32)
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)

        assert self.radar_analyzer is not None
        self.radar_analyzer.eval()
        with torch.no_grad():
            meso_prob, rotation_vel, _ = self.radar_analyzer(seq_tensor)

        mesocyclone_detected = float(meso_prob[0].item()) > 0.5

        return {
            "mesocyclone_detected": mesocyclone_detected,
            "confidence": float(meso_prob[0].item()),
            "rotation_velocity": float(rotation_vel[0].item()) * 50,
        }

    def _determine_threat_level(self, indicators: float, result: TornadoPredictionResult) -> str:
        """Determine tornado threat level."""
        if indicators >= 3.5 or (result.mesocyclone_detected and indicators >= 2.5):
            return "high"
        elif indicators >= 2.5:
            return "moderate"
        elif indicators >= 1.5:
            return "enhanced"
        elif indicators >= 1.0:
            return "slight"
        elif indicators >= 0.5:
            return "marginal"
        else:
            return "none"

    def _estimate_intensity(self, result: TornadoPredictionResult) -> str:
        """Estimate potential tornado intensity."""
        if not result.tornado_likely:
            return "no_tornado"

        score = (
            result.confidence * 0.3
            + (result.rotation_velocity_ms / 100) * 0.3
            + (result.cape_value / 5000) * 0.2
            + (result.helicity_value / 500) * 0.2
        )

        if score > 0.8:
            return "ef4_devastating"
        elif score > 0.6:
            return "ef3_severe"
        elif score > 0.4:
            return "ef2_significant"
        elif score > 0.2:
            return "ef1_moderate"
        else:
            return "ef0_weak"

    def _generate_warnings(self, result: TornadoPredictionResult) -> list[str]:
        """Generate warning actions based on prediction."""
        warnings = []

        if result.threat_level == "high":
            warnings.append("TORNADO WARNING: Take shelter immediately")
            warnings.append("Move to interior room on lowest floor")
            warnings.append("Stay away from windows")
        elif result.threat_level == "moderate":
            warnings.append("TORNADO WATCH: Be prepared to take shelter")
            warnings.append("Monitor weather updates closely")
            warnings.append("Identify nearest shelter location")
        elif result.threat_level in ["enhanced", "slight"]:
            warnings.append("Severe weather possible - stay weather aware")
            warnings.append("Review tornado safety procedures")

        if result.mesocyclone_detected:
            warnings.append("Mesocyclone detected - tornado formation possible")

        return warnings

    def _generate_shelter_advice(self, result: TornadoPredictionResult) -> list[str]:
        """Generate shelter recommendations."""
        advice = []

        if result.tornado_likely:
            advice.append("Seek shelter in basement or storm cellar if available")
            advice.append("If no basement, go to interior room on lowest floor")
            advice.append("Get under sturdy furniture and protect head/neck")
            advice.append("Avoid mobile homes - seek substantial structure")
            advice.append("If driving, do not try to outrun - find sturdy building")

        return advice

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion."""
        if isinstance(data, torch.Tensor):
            data_arr: np.ndarray[Any, Any] = data.cpu().numpy()
        else:
            data_arr = data

        features: list[float] = []

        if self.enable_resonance:
            assert self.resonance_analyzer is not None
            resonance = self.resonance_analyzer.analyze_resonance(data_arr)
            features.extend(
                [
                    resonance["resonance_score"],
                    resonance["dominant_frequency"],
                    resonance["spectral_energy"] / 1e6,
                ]
            )

        if self.enable_recursion:
            assert self.recursive_extractor is not None
            recursive_feat, depth = self.recursive_extractor.extract_recursive_features(data_arr)
            features.extend(recursive_feat.tolist())
            features.append(float(depth) / 4.0)

        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features[:20], dtype=torch.float32)
