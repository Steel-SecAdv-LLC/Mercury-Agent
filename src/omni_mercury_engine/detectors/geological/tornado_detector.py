# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tornado Detector - Multi-Modal Severe Weather Monitoring.

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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.fft import fft
from torch import nn

from omni_mercury_engine.core.three_r_mechanism import (
    RecursionEngine,
    RefactoringEngine,
    ResonanceEngine,
)
from omni_mercury_engine.data_sources.base import DataSourceType
from omni_mercury_engine.data_sources.live_ingestion import (
    LiveFetch,
    fetch_live_datapoints,
    require_live_client,
)
from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics
from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

if TYPE_CHECKING:
    from omni_mercury_engine.data_sources.earth_science import NWSWeatherAlertsSource

#: NWS CAP certainty -> confidence that the hazard is real, per the CAP 1.2
#: certainty vocabulary (Observed > Likely > Possible > Unlikely > Unknown).
_NWS_CERTAINTY_CONFIDENCE = {
    "observed": 0.95,
    "likely": 0.75,
    "possible": 0.45,
    "unlikely": 0.1,
    "unknown": 0.3,
}


class TornadoIntensity(Enum):
    """Enhanced Fujita (EF) Scale tornado intensity levels."""

    EF0 = "ef0_weak"
    EF1 = "ef1_moderate"
    EF2 = "ef2_significant"
    EF3 = "ef3_severe"
    EF4 = "ef4_devastating"
    EF5 = "ef5_incredible"
    NO_TORNADO = "no_tornado"


class TornadoThreatLevel(Enum):
    """Tornado threat assessment levels."""

    NONE = "none"
    MARGINAL = "marginal"
    SLIGHT = "slight"
    ENHANCED = "enhanced"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class TornadoPredictionResult:
    """Tornado prediction results."""

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

    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None

    # Populated only when the detector was built with keep_diagnostics=True.
    diagnostics: HazardDiagnostics | None = None


class DopplerRadarAnalyzer(nn.Module):
    """Doppler radar velocity pattern analyzer for mesocyclone detection.

    Uses LSTM + attention to identify rotation signatures in radar data.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128) -> None:
        """Initialize the instance."""
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
        """Analyze Doppler radar data for mesocyclone signatures.

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
    """Atmospheric instability analysis for tornado potential.

    Monitors CAPE, helicity, wind shear, and other severe weather parameters.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

        self.cape_threshold = 1500.0  # J/kg for significant instability
        self.helicity_threshold = 150.0  # m²/s² for tornado potential
        self.shear_threshold = 40.0  # knots for significant shear

    def analyze_instability(self, atmospheric_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze atmospheric instability parameters.

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
        """Compute Significant Tornado Parameter (STP).

        STP = (CAPE/1500) * (SRH/150) * (Shear/40) * ((2000-LCL)/1000)
        """
        cape_term = min(cape / 1500.0, 2.0)
        helicity_term = min(helicity / 150.0, 2.0)
        shear_term = min(shear / 40.0, 2.0)
        lcl_term = max(0.0, min((2000.0 - lcl) / 1000.0, 1.0))

        stp = cape_term * helicity_term * shear_term * lcl_term

        return stp


class PressureGradientMonitor:
    """Atmospheric pressure gradient monitoring for tornado precursors.

    Detects rapid pressure drops associated with mesocyclone development.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.rapid_drop_threshold = 4.0  # mb in 15 minutes

    def analyze_pressure(self, pressure_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze pressure gradients for tornado signatures.

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
    """FFT-based resonance pattern analyzer for tornado signatures.

    Implements the Resonance Engine component of 3R for detecting characteristic frequency patterns
    in atmospheric data.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.tornado_frequencies = [0.1, 0.5, 1.0, 2.0]  # Hz characteristic frequencies

    def analyze_resonance(self, signal: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Analyze signal for tornado-characteristic resonance patterns.

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
    """Recursive feature extraction for multi-scale tornado pattern detection.

    Implements the Recursion Engine component of 3R for hierarchical pattern analysis at
    progressively finer scales.
    """

    def __init__(self, max_depth: int = 4, decay_factor: float = 0.8) -> None:
        """Initialize the instance."""
        self.max_depth = max_depth
        self.decay_factor = decay_factor

    def extract_recursive_features(
        self, data: np.ndarray[Any, Any], depth: int = 0
    ) -> tuple[np.ndarray[Any, Any], int]:
        """Recursively extract features at multiple scales.

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
    """Comprehensive tornado detection system.

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
        data_source: NWSWeatherAlertsSource | None = None,
        keep_diagnostics: bool = False,
    ):
        """Initialize the instance.

        Live-ingestion pattern (uniform across hazard detectors): pass an
        optional NWS weather-alerts client via ``data_source`` (dependency
        injection; default None = fully offline). :meth:`fetch_live_data`
        exposes a provenance-checked fetch and :meth:`detect_live` assesses
        the real active tornado warning/watch state -- official NWS alert
        metadata is never turned into synthetic radar or CAPE values.

        Args:
            enable_radar: Enable Doppler radar mesocyclone analysis.
            enable_atmospheric: Enable CAPE/helicity/shear instability analysis.
            enable_pressure: Enable pressure-gradient monitoring.
            enable_resonance: Enable FFT resonance pattern analysis.
            enable_recursion: Enable recursive multi-scale feature extraction.
            enable_refactoring: Enable the 3R refactoring engine.
            rng: Deterministic RNG for reproducibility.
            data_source: Optional NWS weather-alerts client.
            keep_diagnostics: When True and radar data is supplied, each
                prediction result carries the Doppler velocity field the LSTM
                consumed, its attention weights, and the located velocity
                couplet (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Default False keeps memory behavior unchanged.
        """
        self.enable_radar = enable_radar
        self.enable_atmospheric = enable_atmospheric
        self.enable_pressure = enable_pressure
        self.enable_resonance = enable_resonance
        self.enable_recursion = enable_recursion
        self.enable_refactoring = enable_refactoring
        self.keep_diagnostics = keep_diagnostics
        self._rng = rng or get_global_rng()

        self.radar_analyzer = DopplerRadarAnalyzer() if enable_radar else None
        self.atmospheric_analyzer = AtmosphericInstabilityAnalyzer() if enable_atmospheric else None
        self.pressure_monitor = PressureGradientMonitor() if enable_pressure else None
        self.resonance_analyzer = ResonancePatternAnalyzer() if enable_resonance else None
        self.recursive_extractor = RecursiveFeatureExtractor() if enable_recursion else None

        # Anti-theater guard (mirrors SchumannResonanceDetector): the
        # DopplerRadarAnalyzer LSTM ships with random weights and no labelled
        # mesocyclone corpus exists to train it. Until real weights are loaded
        # via load_neural_weights(), its probability/rotation outputs are noise,
        # so _analyze_radar derives both from the OBSERVED Doppler velocity
        # field instead: the rotational (couplet) velocity is
        # (Vmax - Vmin) / 2 -- the standard mesocyclone strength measure -- and
        # detection follows the operational threshold (~15 m/s).
        self._neural_trained = False
        self._warned_untrained = False

        self.recursion_engine = RecursionEngine(max_depth=5)
        self.resonance_engine = ResonanceEngine(sampling_rate=1.0)
        self.refactoring_engine = RefactoringEngine()

        self._alerts_source = data_source

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
        """Comprehensive tornado prediction.

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
            if self.keep_diagnostics and "velocity_field" in radar_result:
                # Both paths capture the consumed Doppler field; only the
                # trained-LSTM path additionally carries attention weights.
                result.diagnostics = self._build_radar_diagnostics(radar_result)

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
            initial_prediction_str = str(
                {
                    "confidence": result.confidence,
                    "indicators": indicators_float,
                }
            )
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

    def load_neural_weights(self, checkpoint_path: str) -> None:
        """Load trained weights for the Doppler radar analyzer.

        Until this is called the network is untrained and mesocyclone detection
        runs on the deterministic velocity-couplet physics.

        Args:
            checkpoint_path: Path to a torch checkpoint containing a
                ``radar_analyzer`` state dict.
        """
        if self.radar_analyzer is None:
            raise RuntimeError("radar analysis is disabled on this detector")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.radar_analyzer.load_state_dict(checkpoint["radar_analyzer"])
        self._neural_trained = True
        self.logger.info(
            "Tornado radar neural weights loaded from %s; using learned analyzer", checkpoint_path
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the untrained NN is bypassed for physics."""
        if not self._warned_untrained:
            self.logger.warning(
                "TornadoDetector's DopplerRadarAnalyzer is untrained (no checkpoint "
                "loaded); detecting mesocyclones from the Doppler velocity-couplet "
                "physics instead of the NN. Call load_neural_weights() once a "
                "trained checkpoint exists."
            )
            self._warned_untrained = True

    def _analyze_radar(self, radar_sequence: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Analyze Doppler radar data for mesocyclone signatures.

        Uses the trained LSTM only when real weights have been loaded
        (:meth:`load_neural_weights`); otherwise falls back to the deterministic
        velocity-couplet physics so an untrained network can never fabricate (or
        mask) a mesocyclone.
        """
        radar_result: dict[str, Any]
        if not self._neural_trained:
            self._warn_untrained_once()
            radar_result = self._analyze_radar_physics(radar_sequence)
            if self.keep_diagnostics:
                # The physics couplet analysis consumes the same Doppler
                # velocity field the LSTM would; capturing it is capturing
                # real input data, not fabricating an intermediate. (There
                # are no attention weights on this path — none are drawn.)
                arr = np.asarray(radar_sequence, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                radar_result["velocity_field"] = arr
            return radar_result

        seq_tensor = torch.tensor(radar_sequence, dtype=torch.float32)
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)

        assert self.radar_analyzer is not None
        self.radar_analyzer.eval()
        with torch.no_grad():
            meso_prob, rotation_vel, attention = self.radar_analyzer(seq_tensor)

        mesocyclone_detected = float(meso_prob[0].item()) > 0.5

        radar_result = {
            "mesocyclone_detected": mesocyclone_detected,
            "confidence": float(meso_prob[0].item()),
            "rotation_velocity": float(rotation_vel[0].item()) * 50,
        }
        if self.keep_diagnostics:
            # Capture the field the LSTM consumed and its attention weights
            # (previously discarded) exactly as computed.
            radar_result["velocity_field"] = seq_tensor[0].cpu().numpy().astype(float)
            radar_result["attention_weights"] = attention[0].cpu().numpy().astype(float)
        return radar_result

    @staticmethod
    def _build_radar_diagnostics(radar_result: dict[str, Any]) -> HazardDiagnostics:
        """Assemble the radar diagnostics payload, locating the velocity couplet.

        The couplet is the classic mesocyclone signature: the strongest
        adjacent-gate velocity shear in the consumed Doppler field. It is
        located deterministically on the captured field
        (``argmax |v[:, j+1] - v[:, j]|``); nothing is fabricated.

        Args:
            radar_result: Output of :meth:`_analyze_radar` with diagnostics kept.

        Returns:
            The tornado :class:`HazardDiagnostics` payload.
        """
        field_2d: np.ndarray[Any, Any] = radar_result["velocity_field"]
        context: dict[str, Any] = {
            "mesocyclone_detected": bool(radar_result["mesocyclone_detected"]),
            "rotation_velocity_ms": float(radar_result["rotation_velocity"]),
        }
        if field_2d.ndim == 2 and field_2d.shape[1] >= 2:
            gate_shear = np.abs(np.diff(field_2d, axis=1))
            row, col = np.unravel_index(int(np.argmax(gate_shear)), gate_shear.shape)
            context["couplet_row"] = int(row)
            context["couplet_col"] = int(col)
            context["couplet_shear"] = float(gate_shear[row, col])
        else:
            # A single-gate field has no adjacent-gate shear: say so transparently.
            context["couplet_row"] = None
            context["couplet_col"] = None
            context["couplet_shear"] = None
        arrays: dict[str, np.ndarray[Any, Any]] = {"doppler_velocity_field": field_2d}
        # Attention weights exist only on the trained-LSTM path; the physics
        # fallback consumes the same field but computes no attention, so none
        # is included (transparently absent rather than fabricated).
        if "attention_weights" in radar_result:
            arrays["radar_attention"] = radar_result["attention_weights"]
        return HazardDiagnostics(
            hazard="tornado",
            arrays=arrays,
            context=context,
        )

    @staticmethod
    def _analyze_radar_physics(radar_sequence: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Deterministic mesocyclone detection from the Doppler velocity couplet.

        A mesocyclone appears in Doppler velocity data as a couplet of inbound /
        outbound velocities; its strength is the rotational velocity
        ``V_rot = (V_max - V_min) / 2``. Operational (WSR-88D-style) practice
        flags a mesocyclone at roughly ``V_rot >= 15 m/s``. The rotational
        velocity is taken per time step and the median over the sequence is used
        so a single noisy frame can neither trigger nor mask a detection.
        Deterministic: identical input → identical output.
        """
        arr = np.asarray(radar_sequence, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        finite = np.where(np.isfinite(arr), arr, 0.0)
        # V_rot per frame over the leading (time) axis, then the median.
        per_frame = [(np.max(frame) - np.min(frame)) / 2.0 for frame in finite]
        v_rot = float(np.median(per_frame)) if per_frame else 0.0
        # 15 m/s = operational mesocyclone threshold; 30 m/s saturates (strong).
        confidence = float(np.clip((v_rot - 5.0) / 25.0, 0.0, 1.0))
        return {
            "mesocyclone_detected": v_rot >= 15.0,
            "confidence": confidence,
            "rotation_velocity": v_rot,
            "method": "physics_velocity_couplet",
        }

    def fetch_live_data(self, *, allow_simulated: bool = False, **kwargs: Any) -> LiveFetch:
        """Fetch live NWS weather alerts through the injected client.

        Args:
            allow_simulated: Explicit opt-in for simulated sources (NWS alerts
                are a real feed, so this normally stays False).
            **kwargs: Passed to the client fetch.

        Returns:
            Provenance-checked LiveFetch of WEATHER_ALERT data points.

        Raises:
            LiveDataError: No alerts client injected, or the fetch failed.
        """
        client = require_live_client(self._alerts_source, "TornadoDetector", "NWS weather-alerts")
        return fetch_live_datapoints(
            client,
            allow_simulated=allow_simulated,
            source_types=[DataSourceType.WEATHER_ALERT],
            **kwargs,
        )

    def detect_live(
        self, *, allow_simulated: bool = False, **fetch_kwargs: Any
    ) -> TornadoPredictionResult:
        """Assess the live tornado threat from active official NWS alerts.

        This is an ALERT-STATE assessment: it reports what the National
        Weather Service has actually issued (tornado warnings/watches and
        severe-thunderstorm warnings), with confidence taken from the CAP
        certainty field of the most severe tornado product. Radar, CAPE,
        helicity and pressure fields stay at their absent defaults -- alert
        text is never converted into synthetic measurements.

        Args:
            allow_simulated: Explicit opt-in for simulated sources.
            **fetch_kwargs: Extra client fetch parameters.

        Returns:
            TornadoPredictionResult with ``source_id`` / ``data_provenance`` /
            ``live_context`` populated from the real alert state.

        Raises:
            LiveDataError: No alerts client injected, or the fetch failed.
        """
        fetch = self.fetch_live_data(allow_simulated=allow_simulated, **fetch_kwargs)

        warnings: list[Any] = []
        watches: list[Any] = []
        severe_tstorm: list[Any] = []
        for dp in fetch.data_points:
            event = str(dp.data.get("event", "")).lower()
            if event == "tornado warning":
                warnings.append(dp)
            elif event == "tornado watch":
                watches.append(dp)
            elif event == "severe thunderstorm warning":
                severe_tstorm.append(dp)

        live_context: dict[str, Any] = {
            "tornado_warnings": len(warnings),
            "tornado_watches": len(watches),
            "severe_thunderstorm_warnings": len(severe_tstorm),
            "total_alerts": len(fetch.data_points),
            "warned_areas": [str(dp.data.get("area_desc", ""))[:120] for dp in warnings[:10]],
        }

        # Alert-state mapping: an active Tornado Warning is an NWS statement
        # that a tornado is occurring or imminent; a Watch means conditions
        # are favorable. Threat vocabulary matches _determine_threat_level.
        if warnings:
            strongest = max(
                warnings,
                key=lambda dp: _NWS_CERTAINTY_CONFIDENCE.get(
                    str(dp.data.get("certainty", "unknown")).lower(), 0.3
                ),
            )
            certainty = str(strongest.data.get("certainty", "unknown")).lower()
            confidence = _NWS_CERTAINTY_CONFIDENCE.get(certainty, 0.3)
            tornado_likely = certainty in ("observed", "likely")
            threat_level = "high"
            intensity = "warned_unrated"  # NWS warnings carry no EF rating pre-event.
            actions = [
                a
                for a in (str(strongest.data.get("instruction", "")).strip(),)
                if a  # Real NWS instruction text, when present.
            ]
        elif watches:
            confidence = 0.4
            tornado_likely = False
            threat_level = "moderate"
            intensity = "watch_conditions"
            actions = ["Tornado Watch in effect: monitor conditions and be ready to shelter."]
        elif severe_tstorm:
            confidence = 0.2
            tornado_likely = False
            threat_level = "slight"
            intensity = "no_tornado"
            actions = []
        else:
            confidence = 0.0
            tornado_likely = False
            threat_level = "none"
            intensity = "no_tornado"
            actions = []

        return TornadoPredictionResult(
            tornado_likely=tornado_likely,
            confidence=confidence,
            threat_level=threat_level,
            estimated_intensity=intensity,
            warning_actions=actions,
            source_id=fetch.source_id,
            data_provenance=fetch.data_provenance,
            live_context=live_context,
        )

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
