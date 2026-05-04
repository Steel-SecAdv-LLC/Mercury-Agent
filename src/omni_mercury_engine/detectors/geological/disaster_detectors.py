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
Comprehensive Disaster Detectors for Humanitarian Early Warning

Implements detection systems for:
- Tsunami: Oceanic waveform FFT analysis with Resonance integration
- Earthquake: P/S-wave spectrogram analysis via Scipy.signal
- Meteor: Optical/radar Bayesian filter with NOAA stub
- Solar Flare: X-ray flux predictors with geomagnetic HMM

All detectors integrate with the 3R mechanism:
- Recursion: Multi-scale hierarchical feature extraction
- Resonance: FFT-based frequency domain analysis
- Refactoring: Adaptive threshold optimization

Research sources:
- NOAA National Weather Service
- USGS Earthquake Hazards Program
- NASA Space Weather Prediction Center
- Pacific Tsunami Warning Center

Performance: Synaptic integration with GOSNN for ethical gating
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from scipy import signal
from scipy.fft import fft, fftfreq

from omni_mercury_engine.resilience.api_circuit_breakers import get_data_loader_breaker
from omni_mercury_engine.security.input_validation import TrustedEndpoints
from omni_mercury_engine.utils.rng import get_global_rng

logger = logging.getLogger(__name__)

# Feature dimension for fusion pipeline
FEATURE_DIM = 20


class TsunamiSeverity(Enum):
    """Tsunami severity classification based on wave height."""

    NONE = "none"
    ADVISORY = "advisory"  # < 0.3m
    WATCH = "watch"  # 0.3-1m
    WARNING = "warning"  # 1-3m
    MAJOR = "major"  # > 3m


class EarthquakeMagnitude(Enum):
    """Earthquake magnitude classification (Richter scale)"""

    MICRO = "micro"  # < 2.0
    MINOR = "minor"  # 2.0-3.9
    LIGHT = "light"  # 4.0-4.9
    MODERATE = "moderate"  # 5.0-5.9
    STRONG = "strong"  # 6.0-6.9
    MAJOR = "major"  # 7.0-7.9
    GREAT = "great"  # >= 8.0


class MeteorThreatLevel(Enum):
    """Meteor/asteroid threat classification."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SolarFlareClass(Enum):
    """Solar flare classification (GOES X-ray flux)"""

    A = "a_class"  # < 10^-7 W/m^2
    B = "b_class"  # 10^-7 to 10^-6
    C = "c_class"  # 10^-6 to 10^-5
    M = "m_class"  # 10^-5 to 10^-4
    X = "x_class"  # >= 10^-4


@dataclass
class TsunamiPredictionResult:
    """Tsunami prediction results."""

    tsunami_detected: bool
    confidence: float
    severity: str
    estimated_wave_height_m: float

    arrival_time_minutes: float | None = None
    source_distance_km: float | None = None
    source_magnitude: float | None = None

    resonance_score: float = 0.0
    dominant_frequencies: list[float] = field(default_factory=list)
    waveform_anomaly_score: float = 0.0

    warning_actions: list[str] = field(default_factory=list)
    evacuation_zones: list[str] = field(default_factory=list)


@dataclass
class EarthquakePredictionResult:
    """Earthquake prediction results."""

    earthquake_detected: bool
    confidence: float
    estimated_magnitude: float
    magnitude_class: str

    p_wave_detected: bool = False
    s_wave_detected: bool = False
    p_wave_arrival_time: float | None = None
    s_wave_arrival_time: float | None = None

    epicenter_distance_km: float | None = None
    depth_km: float | None = None

    resonance_score: float = 0.0
    spectral_anomalies: list[float] = field(default_factory=list)

    warning_actions: list[str] = field(default_factory=list)
    aftershock_probability: float = 0.0


@dataclass
class MeteorPredictionResult:
    """Meteor/asteroid prediction results."""

    meteor_detected: bool
    confidence: float
    threat_level: str

    estimated_size_m: float | None = None
    estimated_velocity_kms: float | None = None
    impact_probability: float = 0.0

    optical_detection: bool = False
    radar_detection: bool = False
    bayesian_posterior: float = 0.0

    trajectory_confidence: float = 0.0
    time_to_closest_approach_hours: float | None = None

    warning_actions: list[str] = field(default_factory=list)


@dataclass
class SolarFlarePredictionResult:
    """Solar flare prediction results."""

    flare_detected: bool
    confidence: float
    flare_class: str

    x_ray_flux: float = 0.0
    proton_flux: float = 0.0
    geomagnetic_storm_probability: float = 0.0

    kp_index_predicted: float = 0.0
    dst_index_predicted: float = 0.0

    hmm_state: int = 0
    transition_probability: float = 0.0

    warning_actions: list[str] = field(default_factory=list)
    affected_systems: list[str] = field(default_factory=list)


if TORCH_AVAILABLE:

    class WaveformFFTAnalyzer(nn.Module):
        """
        FFT-based waveform analyzer for tsunami detection.

        Analyzes oceanic waveform patterns using frequency domain analysis integrated with 3R
        Resonance mechanism.
        """

        def __init__(self, input_dim: int = 256, hidden_dim: int = 64) -> None:
            super().__init__()

            self.conv1d = nn.Conv1d(1, 16, kernel_size=7, padding=3)
            self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)

            self.lstm = nn.LSTM(
                input_size=32,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

            self.wave_height_estimator = nn.Sequential(
                nn.Linear(hidden_dim * 2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass for waveform analysis.

            Args:
                x: Waveform tensor [batch, seq_len]

            Returns:
                Tuple of (tsunami_prob, estimated_wave_height)
            """
            if x.dim() == 2:
                x = x.unsqueeze(1)

            x = torch.relu(self.conv1d(x))
            x = torch.relu(self.conv1d_2(x))

            x = x.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x)

            pooled = lstm_out.mean(dim=1)

            tsunami_prob = self.classifier(pooled)
            wave_height = self.wave_height_estimator(pooled)

            return tsunami_prob.squeeze(-1), wave_height.squeeze(-1)

else:

    def WaveformFFTAnalyzer(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        """Stub: WaveformFFTAnalyzer requires PyTorch."""
        raise ImportError("WaveformFFTAnalyzer requires PyTorch. Install with: pip install torch")


if TORCH_AVAILABLE:

    class SeismicWaveAnalyzer(nn.Module):
        """
        P/S-wave spectrogram analyzer for earthquake detection.

        Uses scipy.signal for spectrogram computation and neural network for classification.
        """

        def __init__(self, n_freq_bins: int = 64, hidden_dim: int = 128) -> None:
            super().__init__()

            self.conv2d = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )

            self.classifier = nn.Sequential(
                nn.Linear(64 * 4 * 4, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

            self.magnitude_estimator = nn.Sequential(
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            self.p_wave_detector = nn.Sequential(
                nn.Linear(64 * 4 * 4, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

            self.s_wave_detector = nn.Sequential(
                nn.Linear(64 * 4 * 4, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(
            self, spectrogram: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass for seismic analysis.

            Args:
                spectrogram: Spectrogram tensor [batch, 1, freq, time]

            Returns:
                Tuple of (earthquake_prob, magnitude, p_wave_prob, s_wave_prob)
            """
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)

            features = self.conv2d(spectrogram)
            features_flat = features.view(features.size(0), -1)

            earthquake_prob = self.classifier(features_flat)
            magnitude = self.magnitude_estimator(features_flat)
            p_wave_prob = self.p_wave_detector(features_flat)
            s_wave_prob = self.s_wave_detector(features_flat)

            return (
                earthquake_prob.squeeze(-1),
                magnitude.squeeze(-1),
                p_wave_prob.squeeze(-1),
                s_wave_prob.squeeze(-1),
            )

else:

    def SeismicWaveAnalyzer(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        """Stub: SeismicWaveAnalyzer requires PyTorch."""
        raise ImportError("SeismicWaveAnalyzer requires PyTorch. Install with: pip install torch")


class BayesianMeteorFilter:
    """Bayesian filter for meteor detection combining optical and radar data."""

    def __init__(
        self,
        prior_probability: float = 1e-6,
        optical_sensitivity: float = 0.8,
        radar_sensitivity: float = 0.9,
    ):
        self.prior = prior_probability
        self.optical_sens = optical_sensitivity
        self.radar_sens = radar_sensitivity

        self.optical_false_positive = 0.01
        self.radar_false_positive = 0.005

    def update(
        self,
        optical_detection: bool,
        radar_detection: bool,
        prior: float | None = None,
    ) -> float:
        """
        Update posterior probability using Bayes' theorem.

        Args:
            optical_detection: Whether optical sensor detected object
            radar_detection: Whether radar detected object
            prior: Optional prior probability override

        Returns:
            Posterior probability of meteor
        """
        p_meteor = prior if prior is not None else self.prior

        if optical_detection:
            p_optical_given_meteor = self.optical_sens
            p_optical_given_no_meteor = self.optical_false_positive
        else:
            p_optical_given_meteor = 1 - self.optical_sens
            p_optical_given_no_meteor = 1 - self.optical_false_positive

        if radar_detection:
            p_radar_given_meteor = self.radar_sens
            p_radar_given_no_meteor = self.radar_false_positive
        else:
            p_radar_given_meteor = 1 - self.radar_sens
            p_radar_given_no_meteor = 1 - self.radar_false_positive

        p_evidence_given_meteor = p_optical_given_meteor * p_radar_given_meteor
        p_evidence_given_no_meteor = p_optical_given_no_meteor * p_radar_given_no_meteor

        p_evidence = p_evidence_given_meteor * p_meteor + p_evidence_given_no_meteor * (
            1 - p_meteor
        )

        if p_evidence > 0:
            posterior = (p_evidence_given_meteor * p_meteor) / p_evidence
        else:
            posterior = p_meteor

        return posterior


class GeomagneticHMM:
    """Hidden Markov Model for solar flare and geomagnetic storm prediction."""

    def __init__(self, n_states: int = 5) -> None:
        self.n_states = n_states

        self.transition_matrix = np.array(
            [
                [0.90, 0.08, 0.02, 0.00, 0.00],
                [0.05, 0.85, 0.08, 0.02, 0.00],
                [0.02, 0.05, 0.83, 0.08, 0.02],
                [0.01, 0.02, 0.05, 0.82, 0.10],
                [0.00, 0.01, 0.02, 0.07, 0.90],
            ]
        )

        self.emission_means = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
        self.emission_stds = np.array([5e-9, 5e-8, 5e-7, 5e-6, 5e-5])

        self.state_names = ["Quiet", "B-class", "C-class", "M-class", "X-class"]
        self.current_state = 0

    def predict_next_state(self, current_state: int | None = None) -> tuple[int, float]:
        """
        Predict most likely next state.

        Args:
            current_state: Current state index (uses internal if None)

        Returns:
            Tuple of (predicted_state, transition_probability)
        """
        state = current_state if current_state is not None else self.current_state
        probs = self.transition_matrix[state]
        predicted_state = int(np.argmax(probs))
        return predicted_state, probs[predicted_state]

    def update_state(self, x_ray_flux: float) -> int:
        """
        Update state based on observed X-ray flux.

        Args:
            x_ray_flux: Observed X-ray flux in W/m^2

        Returns:
            Most likely current state
        """
        log_likelihoods = np.zeros(self.n_states)

        for i in range(self.n_states):
            diff = np.log10(max(x_ray_flux, 1e-10)) - np.log10(self.emission_means[i])
            log_likelihoods[i] = -0.5 * (diff / 0.5) ** 2

        log_likelihoods += np.log(self.transition_matrix[self.current_state] + 1e-10)

        self.current_state = int(np.argmax(log_likelihoods))
        return self.current_state

    def get_storm_probability(self, state: int | None = None) -> float:
        """
        Get probability of geomagnetic storm given current state.

        Args:
            state: State index (uses current if None)

        Returns:
            Storm probability [0, 1]
        """
        s = state if state is not None else self.current_state
        storm_probs = [0.01, 0.05, 0.15, 0.45, 0.85]
        return storm_probs[s]


class TsunamiDetector:
    """
    Tsunami detector using oceanic waveform FFT analysis.

    Integrates with 3R Resonance mechanism for frequency-domain anomaly detection in oceanic sensor
    data.
    """

    def __init__(
        self,
        sampling_rate: float = 1.0,
        detection_threshold: float = 0.96,
        device: str = "cpu",
    ):
        self.sampling_rate = sampling_rate
        self.detection_threshold = detection_threshold
        self.device = torch.device(device)
        self.rng = get_global_rng()

        self.waveform_analyzer = WaveformFFTAnalyzer().to(self.device)
        self.waveform_analyzer.eval()

        self.tsunami_frequencies = [0.001, 0.005, 0.01, 0.02]

        logger.info(f"TsunamiDetector initialized: threshold={detection_threshold}")

    def predict_tsunami(
        self,
        waveform_data: np.ndarray[Any, Any] | torch.Tensor,
        source_info: dict[str, Any] | None = None,
    ) -> TsunamiPredictionResult:
        """
        Predict tsunami from oceanic waveform data.

        Args:
            waveform_data: Sea level or pressure waveform [seq_len] or [batch, seq_len]
            source_info: Optional source information (earthquake magnitude, location)

        Returns:
            TsunamiPredictionResult with detection details
        """
        if isinstance(waveform_data, np.ndarray):
            waveform_data = torch.from_numpy(waveform_data).float()

        if waveform_data.dim() == 1:
            waveform_data = waveform_data.unsqueeze(0)

        waveform_data = waveform_data.to(self.device)

        fft_result = fft(waveform_data.cpu().numpy()[0])
        freqs = fftfreq(len(fft_result), 1.0 / self.sampling_rate)
        power_spectrum = np.abs(fft_result) ** 2

        resonance_score = 0.0
        dominant_freqs = []

        for target_freq in self.tsunami_frequencies:
            idx = np.argmin(np.abs(freqs - target_freq))
            if idx < len(power_spectrum):
                local_power = power_spectrum[max(0, idx - 2) : idx + 3].mean()  # type: ignore[misc, unused-ignore]
                global_power = power_spectrum.mean() + 1e-10
                if local_power / global_power > 2.0:
                    resonance_score += 0.25
                    dominant_freqs.append(float(freqs[idx]))

        with torch.no_grad():
            tsunami_prob, wave_height = self.waveform_analyzer(waveform_data)

        confidence = float(tsunami_prob[0].item())
        confidence = min(1.0, confidence + resonance_score * 0.3)

        tsunami_detected = confidence > self.detection_threshold

        severity = self._determine_severity(float(wave_height[0].item()))

        arrival_time = None
        if source_info and "distance_km" in source_info:
            tsunami_speed_kmh = 700
            arrival_time = source_info["distance_km"] / tsunami_speed_kmh * 60

        warnings = self._generate_warnings(tsunami_detected, severity)
        zones = self._generate_evacuation_zones(severity)

        return TsunamiPredictionResult(
            tsunami_detected=tsunami_detected,
            confidence=confidence,
            severity=severity,
            estimated_wave_height_m=float(wave_height[0].item()),
            arrival_time_minutes=arrival_time,
            source_distance_km=source_info.get("distance_km") if source_info else None,
            source_magnitude=source_info.get("magnitude") if source_info else None,
            resonance_score=resonance_score,
            dominant_frequencies=dominant_freqs,
            waveform_anomaly_score=float(np.std(power_spectrum)),
            warning_actions=warnings,
            evacuation_zones=zones,
        )

    def _determine_severity(self, wave_height: float) -> str:
        """Determine tsunami severity from wave height."""
        if wave_height < 0.1:
            return TsunamiSeverity.NONE.value
        elif wave_height < 0.3:
            return TsunamiSeverity.ADVISORY.value
        elif wave_height < 1.0:
            return TsunamiSeverity.WATCH.value
        elif wave_height < 3.0:
            return TsunamiSeverity.WARNING.value
        else:
            return TsunamiSeverity.MAJOR.value

    def _generate_warnings(self, detected: bool, severity: str) -> list[str]:
        """Generate warning actions based on detection."""
        if not detected:
            return []

        warnings = ["Monitor official tsunami warning centers"]

        if severity in [TsunamiSeverity.WARNING.value, TsunamiSeverity.MAJOR.value]:
            warnings.extend(
                [
                    "EVACUATE coastal areas immediately",
                    "Move to high ground (30m+ elevation)",
                    "Stay away from beaches and harbors",
                    "Do not return until all-clear issued",
                ]
            )
        elif severity == TsunamiSeverity.WATCH.value:
            warnings.extend(
                [
                    "Prepare for possible evacuation",
                    "Stay informed via emergency broadcasts",
                    "Avoid coastal areas",
                ]
            )

        return warnings

    def _generate_evacuation_zones(self, severity: str) -> list[str]:
        """Generate evacuation zone recommendations."""
        if severity == TsunamiSeverity.MAJOR.value:
            return ["All coastal areas within 5km of shore", "Low-lying areas below 30m elevation"]
        elif severity == TsunamiSeverity.WARNING.value:
            return ["Coastal areas within 2km of shore", "Areas below 15m elevation"]
        elif severity == TsunamiSeverity.WATCH.value:
            return ["Immediate beach areas", "Harbor facilities"]
        return []

    def extract_features(
        self, waveform_data: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any]:
        """
        Extract features for fusion pipeline.

        Args:
            waveform_data: Waveform data

        Returns:
            Feature array [FEATURE_DIM]
        """
        if isinstance(waveform_data, torch.Tensor):
            waveform_data = waveform_data.cpu().numpy()

        if waveform_data.ndim > 1:
            waveform_data = waveform_data.flatten()

        features = np.zeros(FEATURE_DIM)

        features[0] = np.mean(waveform_data)
        features[1] = np.std(waveform_data)
        features[2] = np.max(waveform_data) - np.min(waveform_data)

        fft_result = fft(waveform_data)
        power = np.abs(fft_result) ** 2
        features[3:8] = power[: min(5, len(power))] / (power.sum() + 1e-10)

        features[8] = float(np.argmax(power[: len(power) // 2])) / len(power)

        result = self.predict_tsunami(waveform_data)
        features[9] = result.confidence
        features[10] = result.resonance_score
        features[11] = result.estimated_wave_height_m

        return features


class EarthquakeDetector:
    """
    Earthquake detector using P/S-wave spectrogram analysis.

    Uses scipy.signal for spectrogram computation and integrates with 3R Resonance for frequency-
    domain analysis.
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        detection_threshold: float = 0.96,
        device: str = "cpu",
    ):
        self.sampling_rate = sampling_rate
        self.detection_threshold = detection_threshold
        self.device = torch.device(device)
        self.rng = get_global_rng()

        self.seismic_analyzer = SeismicWaveAnalyzer().to(self.device)
        self.seismic_analyzer.eval()

        self.p_wave_velocity = 6.0
        self.s_wave_velocity = 3.5

        logger.info(f"EarthquakeDetector initialized: threshold={detection_threshold}")

    def predict_earthquake(
        self,
        seismic_data: np.ndarray[Any, Any] | torch.Tensor,
        station_info: dict[str, Any] | None = None,
    ) -> EarthquakePredictionResult:
        """
        Predict earthquake from seismic waveform data.

        Args:
            seismic_data: Seismic waveform [seq_len] or [batch, seq_len]
            station_info: Optional station information

        Returns:
            EarthquakePredictionResult with detection details
        """
        if isinstance(seismic_data, torch.Tensor):
            seismic_data = seismic_data.cpu().numpy()

        if seismic_data.ndim == 1:
            seismic_data = seismic_data.reshape(1, -1)

        f, t, Sxx = signal.spectrogram(
            seismic_data[0],
            fs=self.sampling_rate,
            nperseg=min(256, len(seismic_data[0]) // 4),
            noverlap=min(128, len(seismic_data[0]) // 8),
        )

        Sxx_log = np.log10(Sxx + 1e-10)
        Sxx_norm = (Sxx_log - Sxx_log.mean()) / (Sxx_log.std() + 1e-10)

        spectrogram_tensor = torch.from_numpy(Sxx_norm).float().unsqueeze(0).unsqueeze(0)
        spectrogram_tensor = spectrogram_tensor.to(self.device)

        with torch.no_grad():
            eq_prob, magnitude, p_prob, s_prob = self.seismic_analyzer(spectrogram_tensor)

        confidence = float(eq_prob[0].item())
        estimated_mag = float(magnitude[0].item()) * 4 + 2

        p_wave_detected = float(p_prob[0].item()) > 0.5
        s_wave_detected = float(s_prob[0].item()) > 0.5

        resonance_score = self._compute_resonance_score(Sxx, f)

        confidence = min(1.0, confidence + resonance_score * 0.2)

        earthquake_detected = confidence > self.detection_threshold

        magnitude_class = self._classify_magnitude(estimated_mag)

        epicenter_distance = None
        if p_wave_detected and s_wave_detected:
            p_arrival = self._detect_wave_arrival(seismic_data[0], "p")
            s_arrival = self._detect_wave_arrival(seismic_data[0], "s")
            if p_arrival is not None and s_arrival is not None:
                time_diff = (s_arrival - p_arrival) / self.sampling_rate
                epicenter_distance = (
                    time_diff
                    * (self.p_wave_velocity * self.s_wave_velocity)
                    / (self.p_wave_velocity - self.s_wave_velocity)
                )

        warnings = self._generate_warnings(earthquake_detected, magnitude_class)

        return EarthquakePredictionResult(
            earthquake_detected=earthquake_detected,
            confidence=confidence,
            estimated_magnitude=estimated_mag,
            magnitude_class=magnitude_class,
            p_wave_detected=p_wave_detected,
            s_wave_detected=s_wave_detected,
            epicenter_distance_km=epicenter_distance,
            resonance_score=resonance_score,
            spectral_anomalies=self._find_spectral_anomalies(Sxx, f),
            warning_actions=warnings,
            aftershock_probability=min(0.9, estimated_mag / 10),
        )

    def _compute_resonance_score(
        self, Sxx: np.ndarray[Any, Any], freqs: np.ndarray[Any, Any]
    ) -> float:
        """Compute resonance score from spectrogram."""
        power_by_freq = Sxx.mean(axis=1)

        seismic_bands = [(0.1, 1.0), (1.0, 5.0), (5.0, 20.0)]
        score = 0.0

        for low, high in seismic_bands:
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                band_power = power_by_freq[mask].mean()
                total_power = power_by_freq.mean() + 1e-10
                if band_power / total_power > 1.5:
                    score += 0.33

        return min(1.0, score)

    def _classify_magnitude(self, magnitude: float) -> str:
        """Classify earthquake magnitude."""
        if magnitude < 2.0:
            return EarthquakeMagnitude.MICRO.value
        elif magnitude < 4.0:
            return EarthquakeMagnitude.MINOR.value
        elif magnitude < 5.0:
            return EarthquakeMagnitude.LIGHT.value
        elif magnitude < 6.0:
            return EarthquakeMagnitude.MODERATE.value
        elif magnitude < 7.0:
            return EarthquakeMagnitude.STRONG.value
        elif magnitude < 8.0:
            return EarthquakeMagnitude.MAJOR.value
        else:
            return EarthquakeMagnitude.GREAT.value

    def _detect_wave_arrival(self, data: np.ndarray[Any, Any], wave_type: str) -> int | None:
        """Detect P or S wave arrival time using STA/LTA."""
        sta_len = int(0.5 * self.sampling_rate)
        lta_len = int(5.0 * self.sampling_rate)

        if len(data) < lta_len + sta_len:
            return None

        sta_lta = np.zeros(len(data))
        for i in range(lta_len, len(data) - sta_len):
            sta = np.mean(np.abs(data[i : i + sta_len]))
            lta = np.mean(np.abs(data[i - lta_len : i]))
            sta_lta[i] = sta / (lta + 1e-10)

        threshold = 3.0 if wave_type == "p" else 2.0
        arrivals = np.where(sta_lta > threshold)[0]

        return int(arrivals[0]) if len(arrivals) > 0 else None

    def _find_spectral_anomalies(
        self, Sxx: np.ndarray[Any, Any], freqs: np.ndarray[Any, Any]
    ) -> list[float]:
        """Find anomalous frequencies in spectrogram."""
        power_by_freq = Sxx.mean(axis=1)
        mean_power = power_by_freq.mean()
        std_power = power_by_freq.std()

        anomalies = []
        for i, (f, p) in enumerate(zip(freqs, power_by_freq)):
            if p > mean_power + 2 * std_power:
                anomalies.append(float(f))

        return anomalies[:5]

    def _generate_warnings(self, detected: bool, magnitude_class: str) -> list[str]:
        """Generate warning actions."""
        if not detected:
            return []

        warnings = ["Monitor official earthquake information"]

        if magnitude_class in [EarthquakeMagnitude.MAJOR.value, EarthquakeMagnitude.GREAT.value]:
            warnings.extend(
                [
                    "DROP, COVER, and HOLD ON",
                    "Move away from windows and heavy objects",
                    "Expect aftershocks",
                    "Check for gas leaks after shaking stops",
                ]
            )
        elif magnitude_class in [
            EarthquakeMagnitude.STRONG.value,
            EarthquakeMagnitude.MODERATE.value,
        ]:
            warnings.extend(
                [
                    "Take protective action",
                    "Be prepared for aftershocks",
                ]
            )

        return warnings

    def extract_features(
        self, seismic_data: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any]:
        """Extract features for fusion pipeline."""
        if isinstance(seismic_data, torch.Tensor):
            seismic_data = seismic_data.cpu().numpy()

        if seismic_data.ndim > 1:
            seismic_data = seismic_data.flatten()

        features = np.zeros(FEATURE_DIM)

        features[0] = np.mean(seismic_data)
        features[1] = np.std(seismic_data)
        features[2] = np.max(np.abs(seismic_data))

        f, t, Sxx = signal.spectrogram(
            seismic_data,
            fs=self.sampling_rate,
            nperseg=min(256, len(seismic_data) // 4),
        )
        features[3:8] = Sxx.mean(axis=1)[: min(5, len(f))]

        result = self.predict_earthquake(seismic_data)
        features[8] = result.confidence
        features[9] = result.estimated_magnitude / 10
        features[10] = result.resonance_score

        return features


class MeteorDetector:
    """
    Meteor detector using optical/radar Bayesian filter with NASA CNEOS integration.

    Combines optical and radar observations with Bayesian inference
    for meteor/asteroid detection and trajectory estimation.

    Production Features:
        - NASA CNEOS Fireball API integration for real atmospheric impact data
        - NASA Close Approach Data (CAD) for near-Earth object tracking
        - NASA Sentry impact monitoring for potential future impacts
        - Bayesian sensor fusion for optical/radar observations
    """

    def __init__(
        self,
        detection_threshold: float = 0.7,
        prior_probability: float = 1e-6,
        use_nasa_data: bool = True,
    ):
        """
        Initialize MeteorDetector.

        Args:
            detection_threshold: Confidence threshold for meteor detection (0-1)
            prior_probability: Prior probability of meteor occurrence for Bayesian filter
            use_nasa_data: Whether to fetch real-time data from NASA CNEOS APIs
        """
        self.detection_threshold = detection_threshold
        self.bayesian_filter = BayesianMeteorFilter(prior_probability=prior_probability)
        self.rng = get_global_rng()
        self.use_nasa_data = use_nasa_data

        # Cached NASA data
        self._fireball_cache: list[Any] | None = None
        self._close_approach_cache: list[Any] | None = None
        self._sentry_cache: list[Any] | None = None
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_hours: int = 6  # Refresh NASA data every 6 hours

        logger.info(
            f"MeteorDetector initialized: threshold={detection_threshold}, "
            f"nasa_data={use_nasa_data}"
        )

    def _refresh_nasa_cache(self) -> None:
        """Refresh NASA data cache if expired or empty."""
        now = datetime.now()

        # Check if cache needs refresh
        if (
            self._cache_timestamp is not None
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_hours * 3600
        ):
            return  # Cache is still valid

        logger.info("Refreshing NASA CNEOS data cache...")

        # Load all NASA data sources
        self._fireball_cache = load_nasa_fireball_data(days_back=30)
        self._close_approach_cache = load_nasa_close_approach_data(days_forward=30)
        self._sentry_cache = load_nasa_sentry_data()
        self._cache_timestamp = now

    def get_recent_fireballs(self, days: int = 7) -> list[Any]:
        """
        Get recent fireball events from NASA CNEOS.

        Args:
            days: Number of days back to look

        Returns:
            List of FireballEvent objects from the last N days
        """
        if not self.use_nasa_data:
            return []

        self._refresh_nasa_cache()

        if self._fireball_cache is None:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        return [fb for fb in self._fireball_cache if fb.date >= cutoff]

    def get_upcoming_close_approaches(self) -> list[Any]:
        """
        Get upcoming near-Earth object close approaches.

        Returns:
            List of CloseApproachEvent objects
        """
        if not self.use_nasa_data:
            return []

        self._refresh_nasa_cache()
        return self._close_approach_cache or []

    def get_impact_risks(self) -> list[Any]:
        """
        Get current impact risk assessments from NASA Sentry.

        Returns:
            List of SentryImpactRisk objects sorted by Palermo scale
        """
        if not self.use_nasa_data:
            return []

        self._refresh_nasa_cache()

        if self._sentry_cache is None:
            return []

        # Sort by Palermo scale (higher = more concerning)
        return sorted(self._sentry_cache, key=lambda x: x.palermo_scale, reverse=True)

    def predict_meteor(
        self,
        optical_data: np.ndarray[Any, Any] | None = None,
        radar_data: np.ndarray[Any, Any] | None = None,
        noaa_stub: dict[str, Any] | None = None,
    ) -> MeteorPredictionResult:
        """
        Predict meteor from optical and radar data with NASA CNEOS integration.

        This method combines local sensor data with real-time NASA CNEOS data
        for comprehensive meteor/NEO detection.

        Args:
            optical_data: Optical sensor data (brightness measurements)
            radar_data: Radar return data
            noaa_stub: Optional external data dict with keys:
                - optical_alert: bool - External optical detection
                - radar_alert: bool - External radar detection
                - fireball_energy_kt: float - Observed impact energy in kilotons
                - velocity_km_s: float - Observed velocity
                - size_estimate_m: float - Estimated size in meters

        Returns:
            MeteorPredictionResult with detection details including NASA CNEOS data
        """
        optical_detection = False
        radar_detection = False

        # Process local optical sensor data
        if optical_data is not None:
            optical_threshold = np.percentile(optical_data, 99)
            optical_detection = np.max(optical_data) > optical_threshold * 1.5

        # Process local radar sensor data
        if radar_data is not None:
            radar_threshold = np.percentile(radar_data, 99)
            radar_detection = np.max(radar_data) > radar_threshold * 1.5

        # Integrate external data (NOAA stub or other sources)
        if noaa_stub is not None:
            optical_detection = optical_detection or noaa_stub.get("optical_alert", False)
            radar_detection = radar_detection or noaa_stub.get("radar_alert", False)

        # Check NASA CNEOS data for recent significant events
        nasa_fireball_alert = False
        nasa_close_approach_alert = False
        nasa_size_estimate = None
        nasa_velocity_estimate = None
        nasa_impact_probability = 0.0

        if self.use_nasa_data:
            self._refresh_nasa_cache()

            # Check for recent fireballs (last 24 hours with significant energy)
            if self._fireball_cache:
                recent_cutoff = datetime.now() - timedelta(hours=24)
                recent_fireballs = [
                    fb
                    for fb in self._fireball_cache
                    if fb.date >= recent_cutoff
                    and fb.calculated_total_impact_energy_kt is not None
                    and fb.calculated_total_impact_energy_kt > 0.1  # > 100 tons TNT
                ]
                if recent_fireballs:
                    nasa_fireball_alert = True
                    # Use the most energetic recent fireball for estimates
                    biggest = max(
                        recent_fireballs,
                        key=lambda x: x.calculated_total_impact_energy_kt or 0,
                    )
                    nasa_size_estimate = biggest.estimated_size_m
                    nasa_velocity_estimate = biggest.velocity_km_s

            # Check for imminent close approaches (within 1 lunar distance)
            if self._close_approach_cache:
                lunar_distance_km = 384400
                imminent = [
                    ca
                    for ca in self._close_approach_cache
                    if ca.nominal_distance_km < lunar_distance_km
                    and ca.close_approach_date <= datetime.now() + timedelta(days=7)
                ]
                if imminent:
                    nasa_close_approach_alert = True
                    # Estimate impact probability from closest approach
                    closest = min(imminent, key=lambda x: x.nominal_distance_km)
                    # Very rough heuristic: closer = higher concern
                    nasa_impact_probability = (
                        max(0, 1 - closest.nominal_distance_km / lunar_distance_km) * 0.001
                    )

            # Check Sentry for elevated impact risks
            if self._sentry_cache:
                high_risk = [s for s in self._sentry_cache if s.palermo_scale > -3]
                if high_risk:
                    # Highest risk object
                    max_risk = max(high_risk, key=lambda x: x.palermo_scale)
                    nasa_impact_probability = max(
                        nasa_impact_probability, max_risk.impact_probability
                    )

        # Update Bayesian posterior with all detection sources
        # NASA data provides additional evidence
        combined_optical = optical_detection or nasa_fireball_alert
        combined_radar = radar_detection or nasa_close_approach_alert

        posterior = self.bayesian_filter.update(combined_optical, combined_radar)

        # Boost posterior if we have NASA confirmation
        if nasa_fireball_alert:
            posterior = min(1.0, posterior + 0.2)
        if nasa_close_approach_alert:
            posterior = min(1.0, posterior + 0.15)

        meteor_detected = posterior > self.detection_threshold

        threat_level = self._assess_threat(posterior, optical_data, radar_data)

        # Prefer NASA size/velocity estimates if available
        size_estimate = nasa_size_estimate
        velocity_estimate = nasa_velocity_estimate

        # Fall back to local radar estimates
        if size_estimate is None and meteor_detected and radar_data is not None:
            size_estimate = self._estimate_size(radar_data)
        if velocity_estimate is None and meteor_detected and radar_data is not None:
            velocity_estimate = self._estimate_velocity(radar_data)

        # Use external stub estimates if provided
        if noaa_stub is not None:
            if size_estimate is None:
                size_estimate = noaa_stub.get("size_estimate_m")
            if velocity_estimate is None:
                velocity_estimate = noaa_stub.get("velocity_km_s")

        # Compute final impact probability
        final_impact_prob = max(
            nasa_impact_probability,
            posterior * 0.001 if meteor_detected else 0.0,
        )

        warnings = self._generate_warnings(meteor_detected, threat_level)

        # Add NASA-specific warnings
        if nasa_fireball_alert:
            warnings.insert(0, "NASA CNEOS: Recent significant fireball detected")
        if nasa_close_approach_alert:
            warnings.insert(0, "NASA CNEOS: Imminent near-Earth object close approach")

        return MeteorPredictionResult(
            meteor_detected=meteor_detected,
            confidence=posterior,
            threat_level=threat_level,
            estimated_size_m=size_estimate,
            estimated_velocity_kms=velocity_estimate,
            impact_probability=final_impact_prob,
            optical_detection=combined_optical,
            radar_detection=combined_radar,
            bayesian_posterior=posterior,
            trajectory_confidence=(
                0.9 if nasa_close_approach_alert else (0.8 if combined_radar else 0.3)
            ),
            warning_actions=warnings,
        )

    def _assess_threat(
        self,
        posterior: float,
        optical_data: np.ndarray[Any, Any] | None,
        radar_data: np.ndarray[Any, Any] | None,
    ) -> str:
        """Assess meteor threat level."""
        if posterior < 0.1:
            return MeteorThreatLevel.NONE.value
        elif posterior < 0.3:
            return MeteorThreatLevel.MINIMAL.value
        elif posterior < 0.5:
            return MeteorThreatLevel.LOW.value
        elif posterior < 0.7:
            return MeteorThreatLevel.MODERATE.value
        elif posterior < 0.9:
            return MeteorThreatLevel.HIGH.value
        else:
            return MeteorThreatLevel.CRITICAL.value

    def _estimate_size(self, radar_data: np.ndarray[Any, Any]) -> float:
        """Estimate meteor size from radar cross-section."""
        rcs = np.max(radar_data)
        size = np.sqrt(rcs / np.pi) * 10
        return float(size)

    def _estimate_velocity(self, radar_data: np.ndarray[Any, Any]) -> float:
        """Estimate meteor velocity from Doppler shift."""
        if len(radar_data) < 2:
            return 20.0

        doppler_shift = np.diff(radar_data).mean()
        velocity = abs(doppler_shift) * 0.1 + 10
        return float(velocity)

    def _generate_warnings(self, detected: bool, threat_level: str) -> list[str]:
        """Generate warning actions."""
        if not detected:
            return []

        warnings = ["Monitor official space weather alerts"]

        if threat_level in [MeteorThreatLevel.HIGH.value, MeteorThreatLevel.CRITICAL.value]:
            warnings.extend(
                [
                    "Potential impact event detected",
                    "Follow emergency management guidance",
                    "Prepare for possible evacuation",
                ]
            )
        elif threat_level == MeteorThreatLevel.MODERATE.value:
            warnings.append("Elevated meteor activity detected")

        return warnings

    def extract_features(
        self,
        optical_data: np.ndarray[Any, Any] | None = None,
        radar_data: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Extract features for fusion pipeline."""
        features = np.zeros(FEATURE_DIM)

        if optical_data is not None:
            features[0] = np.mean(optical_data)
            features[1] = np.std(optical_data)
            features[2] = np.max(optical_data)

        if radar_data is not None:
            features[3] = np.mean(radar_data)
            features[4] = np.std(radar_data)
            features[5] = np.max(radar_data)

        result = self.predict_meteor(optical_data, radar_data)
        features[6] = result.confidence
        features[7] = 1.0 if result.optical_detection else 0.0
        features[8] = 1.0 if result.radar_detection else 0.0

        return features


class SolarFlareDetector:
    """
    Solar flare detector using X-ray flux and geomagnetic HMM.

    Predicts solar flares and geomagnetic storms using Hidden Markov Model for state transitions and
    X-ray flux analysis.
    """

    def __init__(
        self,
        detection_threshold: float = 0.7,
        proton_flux_agg_method: str = "max",
    ):
        """
        Initialize SolarFlareDetector.

        Args:
            detection_threshold: Confidence threshold for flare detection (0-1)
            proton_flux_agg_method: Aggregation method for proton flux arrays.
                'max' (default) - Use peak value for detecting flare threats
                'mean' - Use average value for general monitoring
                'median' - Use median for robust estimation
        """
        self.detection_threshold = detection_threshold
        self.proton_flux_agg_method = proton_flux_agg_method
        self.hmm = GeomagneticHMM()
        self.rng = get_global_rng()

        self.flux_thresholds = {
            "A": 1e-8,
            "B": 1e-7,
            "C": 1e-6,
            "M": 1e-5,
            "X": 1e-4,
        }

        # Aggregation function mapping
        self._agg_funcs: dict[str, Any] = {
            "max": np.max,
            "mean": np.mean,
            "median": np.median,
        }

        logger.info(
            f"SolarFlareDetector initialized: threshold={detection_threshold}, "
            f"proton_flux_agg={proton_flux_agg_method}"
        )

    def predict_solar_flare(
        self,
        x_ray_flux: float | np.ndarray[Any, Any],
        proton_flux: float | None = None,
        magnetometer_data: np.ndarray[Any, Any] | None = None,
    ) -> SolarFlarePredictionResult:
        """
        Predict solar flare from X-ray and proton flux data.

        Args:
            x_ray_flux: X-ray flux in W/m^2 (scalar or time series)
            proton_flux: Optional proton flux
            magnetometer_data: Optional magnetometer readings

        Returns:
            SolarFlarePredictionResult with detection details
        """
        if isinstance(x_ray_flux, np.ndarray):
            current_flux = float(x_ray_flux[-1])
            flux_trend = np.diff(x_ray_flux).mean() if len(x_ray_flux) > 1 else 0
        else:
            current_flux = float(x_ray_flux)
            flux_trend = 0

        current_state = self.hmm.update_state(current_flux)
        next_state, transition_prob = self.hmm.predict_next_state()

        flare_class = self._classify_flare(current_flux)

        confidence = self._compute_confidence(current_flux, current_state, flux_trend)

        flare_detected = confidence > self.detection_threshold

        storm_prob = self.hmm.get_storm_probability(current_state)

        kp_predicted = self._predict_kp_index(current_state, storm_prob)
        dst_predicted = self._predict_dst_index(current_state, storm_prob)

        warnings = self._generate_warnings(flare_detected, flare_class, storm_prob)
        affected = self._identify_affected_systems(flare_class, storm_prob)

        return SolarFlarePredictionResult(
            flare_detected=flare_detected,
            confidence=confidence,
            flare_class=flare_class,
            x_ray_flux=current_flux,
            proton_flux=self._aggregate_proton_flux(proton_flux),
            geomagnetic_storm_probability=storm_prob,
            kp_index_predicted=kp_predicted,
            dst_index_predicted=dst_predicted,
            hmm_state=current_state,
            transition_probability=transition_prob,
            warning_actions=warnings,
            affected_systems=affected,
        )

    def _aggregate_proton_flux(self, proton_flux: float | np.ndarray[Any, Any] | None) -> float:
        """
        Aggregate proton flux using configured method.

        For time-series threats like solar flares, peak detection (max) is
        recommended as it captures the most dangerous flux levels. Mean is
        suitable for general monitoring, while median provides robust estimation.

        Args:
            proton_flux: Proton flux value(s) - scalar, array, or None

        Returns:
            Aggregated proton flux value (0.0 if None)
        """
        if proton_flux is None:
            return 0.0

        if isinstance(proton_flux, np.ndarray):
            agg_func = self._agg_funcs.get(self.proton_flux_agg_method, np.max)
            return float(agg_func(proton_flux))
        else:
            return float(proton_flux)

    def _classify_flare(self, flux: float) -> str:
        """Classify solar flare based on X-ray flux."""
        if flux >= self.flux_thresholds["X"]:
            return SolarFlareClass.X.value
        elif flux >= self.flux_thresholds["M"]:
            return SolarFlareClass.M.value
        elif flux >= self.flux_thresholds["C"]:
            return SolarFlareClass.C.value
        elif flux >= self.flux_thresholds["B"]:
            return SolarFlareClass.B.value
        else:
            return SolarFlareClass.A.value

    def _compute_confidence(self, flux: float, state: int, trend: float) -> float:
        """Compute detection confidence."""
        base_confidence = state / 4.0

        if flux >= self.flux_thresholds["M"]:
            base_confidence += 0.3
        elif flux >= self.flux_thresholds["C"]:
            base_confidence += 0.1

        if trend > 0:
            base_confidence += min(0.2, trend * 1e6)

        return min(1.0, base_confidence)

    def _predict_kp_index(self, state: int, storm_prob: float) -> float:
        """Predict Kp geomagnetic index."""
        base_kp = [1, 2, 4, 6, 8]
        return base_kp[state] + storm_prob * 2

    def _predict_dst_index(self, state: int, storm_prob: float) -> float:
        """Predict Dst geomagnetic index."""
        base_dst = [0, -10, -30, -100, -300]
        return base_dst[state] - storm_prob * 50

    def _generate_warnings(self, detected: bool, flare_class: str, storm_prob: float) -> list[str]:
        """Generate warning actions."""
        if not detected:
            return []

        warnings = ["Monitor NOAA Space Weather Prediction Center"]

        if flare_class in [SolarFlareClass.X.value, SolarFlareClass.M.value]:
            warnings.extend(
                [
                    "Significant solar flare detected",
                    "Possible HF radio blackouts",
                    "Satellite operators: monitor for anomalies",
                ]
            )

        if storm_prob > 0.5:
            warnings.extend(
                [
                    "Geomagnetic storm likely",
                    "Power grid operators: prepare for GIC",
                    "Aviation: possible GPS/communication issues",
                ]
            )

        return warnings

    def _identify_affected_systems(self, flare_class: str, storm_prob: float) -> list[str]:
        """Identify systems potentially affected."""
        affected = []

        if flare_class in [SolarFlareClass.X.value, SolarFlareClass.M.value]:
            affected.extend(["HF Radio", "Satellites", "GPS"])

        if storm_prob > 0.3:
            affected.extend(["Power Grids", "Pipelines"])

        if storm_prob > 0.6:
            affected.extend(["Aviation Navigation", "Spacecraft Operations"])

        return affected

    def extract_features(
        self,
        x_ray_flux: float | np.ndarray[Any, Any],
        proton_flux: float | None = None,
    ) -> np.ndarray[Any, Any]:
        """Extract features for fusion pipeline."""
        features = np.zeros(FEATURE_DIM)

        if isinstance(x_ray_flux, np.ndarray):
            features[0] = np.mean(x_ray_flux)
            features[1] = np.std(x_ray_flux)
            features[2] = np.max(x_ray_flux)
            features[3] = np.log10(np.max(x_ray_flux) + 1e-10) + 10
        else:
            features[0] = x_ray_flux
            features[3] = np.log10(x_ray_flux + 1e-10) + 10

        if proton_flux is not None:
            if isinstance(proton_flux, np.ndarray):
                features[4] = np.mean(proton_flux)
            else:
                features[4] = proton_flux

        result = self.predict_solar_flare(x_ray_flux, proton_flux)
        features[5] = result.confidence
        features[6] = result.hmm_state / 4.0
        features[7] = result.geomagnetic_storm_probability
        features[8] = result.kp_index_predicted / 9.0

        return features


# =============================================================================
# Synthetic Data Generation and Training for Disaster Neural Networks
# =============================================================================


def generate_synthetic_tsunami_data(
    n_samples: int = 1000,
    seq_len: int = 256,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Generate synthetic tsunami waveform data for training.

    Creates realistic oceanic waveform patterns:
    - Normal waves: Sinusoidal with noise
    - Tsunami waves: Long-period waves with characteristic frequency (0.001-0.01 Hz)

    Args:
        n_samples: Number of samples to generate
        seq_len: Sequence length for each sample
        rng: Random number generator for reproducibility

    Returns:
        Tuple of (waveforms, labels, wave_heights)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    waveforms = np.zeros((n_samples, seq_len), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.float32)
    wave_heights = np.zeros(n_samples, dtype=np.float32)

    t = np.linspace(0, 10, seq_len)

    for i in range(n_samples):
        is_tsunami = rng.random() > 0.5
        labels[i] = float(is_tsunami)

        if is_tsunami:
            # Tsunami: Long-period wave (0.001-0.01 Hz) with high amplitude
            freq = rng.uniform(0.001, 0.01)
            amplitude = rng.uniform(2.0, 10.0)  # meters
            wave_heights[i] = amplitude
            waveform = amplitude * np.sin(2 * np.pi * freq * t * 100)
            # Add characteristic tsunami signature: rapid rise
            rise_idx = rng.integers(seq_len // 4, seq_len // 2)
            waveform[rise_idx:] += amplitude * 0.5 * np.exp(-0.1 * np.arange(seq_len - rise_idx))
        else:
            # Normal ocean waves: Higher frequency, lower amplitude
            freq = rng.uniform(0.05, 0.2)
            amplitude = rng.uniform(0.1, 1.0)
            wave_heights[i] = amplitude
            waveform = amplitude * np.sin(2 * np.pi * freq * t * 100)

        # Add noise
        noise = rng.normal(0, 0.1, seq_len)
        waveforms[i] = waveform + noise

    return waveforms, labels, wave_heights


def generate_synthetic_earthquake_data(
    n_samples: int = 1000,
    n_freq_bins: int = 64,
    n_time_bins: int = 64,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Generate synthetic earthquake spectrogram data for training.

    Creates realistic seismic spectrograms:
    - Normal: Background seismic noise
    - Earthquake: P-wave (1-10 Hz) followed by S-wave (0.1-1 Hz) patterns

    Args:
        n_samples: Number of samples to generate
        n_freq_bins: Number of frequency bins in spectrogram
        n_time_bins: Number of time bins in spectrogram
        rng: Random number generator for reproducibility

    Returns:
        Tuple of (spectrograms, labels, magnitudes)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    spectrograms = np.zeros((n_samples, 1, n_freq_bins, n_time_bins), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.float32)
    magnitudes = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        is_earthquake = rng.random() > 0.5
        labels[i] = float(is_earthquake)

        if is_earthquake:
            magnitude = rng.uniform(3.0, 8.0)
            magnitudes[i] = magnitude

            # Create spectrogram with P-wave and S-wave signatures
            spectrogram = rng.normal(0, 0.1, (n_freq_bins, n_time_bins))

            # P-wave: High frequency (upper half of spectrogram), early arrival
            p_wave_start = rng.integers(5, 15)
            p_wave_duration = rng.integers(5, 15)
            p_wave_intensity = magnitude / 8.0
            spectrogram[
                n_freq_bins // 2 :, p_wave_start : p_wave_start + p_wave_duration
            ] += p_wave_intensity * rng.uniform(0.5, 1.0, (n_freq_bins // 2, p_wave_duration))

            # S-wave: Lower frequency (lower half), later arrival
            s_wave_start = p_wave_start + p_wave_duration + rng.integers(5, 15)
            s_wave_duration = rng.integers(10, 25)
            s_wave_intensity = magnitude / 6.0
            if s_wave_start + s_wave_duration < n_time_bins:
                spectrogram[
                    : n_freq_bins // 2, s_wave_start : s_wave_start + s_wave_duration
                ] += s_wave_intensity * rng.uniform(0.5, 1.0, (n_freq_bins // 2, s_wave_duration))
        else:
            magnitudes[i] = rng.uniform(0.0, 2.0)
            # Background noise only
            spectrogram = rng.normal(0, 0.1, (n_freq_bins, n_time_bins))

        spectrograms[i, 0] = spectrogram

    return spectrograms, labels, magnitudes


# =============================================================================
# Real-World Dataset Loaders for Disaster Detection Training
# =============================================================================

# NOAA DART Buoy API for tsunami detection (via TrustedEndpoints for SSRF prevention)
DART_BUOY_API_URL = TrustedEndpoints.NOAA_NDBC_REALTIME

# NOAA Tsunami Events API (via TrustedEndpoints for SSRF prevention)
NOAA_TSUNAMI_API_URL = TrustedEndpoints.NOAA_TSUNAMI_EVENTS

# USGS Earthquake Catalog API (via TrustedEndpoints for SSRF prevention)
USGS_EARTHQUAKE_API_URL = TrustedEndpoints.USGS_EARTHQUAKE

# NASA CNEOS Fireball API (via TrustedEndpoints for SSRF prevention)
NASA_CNEOS_FIREBALL_URL = TrustedEndpoints.NASA_CNEOS_FIREBALL

# NASA CNEOS Close Approach Data API
NASA_CNEOS_CAD_URL = TrustedEndpoints.NASA_CNEOS_CAD

# NASA Sentry Impact Monitoring API
NASA_SENTRY_URL = TrustedEndpoints.NASA_SENTRY


def load_dart_buoy_data(
    station_id: str = "46419",
    days_back: int = 30,
    seq_len: int = 256,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """
    Load real tsunami waveform data from NOAA DART buoy network.

    DART (Deep-ocean Assessment and Reporting of Tsunamis) buoys provide
    real-time sea level measurements for tsunami detection.

    Data source: NOAA National Data Buoy Center
    https://www.ndbc.noaa.gov/dart.shtml

    Args:
        station_id: DART buoy station ID (default: 46419 - Pacific)
        days_back: Number of days of historical data to fetch
        seq_len: Sequence length for waveform samples

    Returns:
        Tuple of (waveforms, labels, wave_heights) or None if API unavailable
    """
    circuit_breaker = get_data_loader_breaker("dart_buoy")

    def _fetch_dart_data() -> (
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ):
        url = f"{DART_BUOY_API_URL}/{station_id}.dart"
        if not url.startswith("https://"):
            raise RuntimeError("DART API URL must use HTTPS")

        # Validate URL before opening (SSRF protection via domain allowlist)
        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        TrustedEndpoints.validate_url(DART_BUOY_API_URL)
        req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
        with urlopen(req, timeout=30) as response:  # nosec B310
            raw_data = response.read().decode()

        lines = raw_data.strip().split("\n")
        water_levels = []

        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    water_level = float(parts[6])
                    water_levels.append(water_level)
                except (ValueError, IndexError):
                    continue

        if len(water_levels) < seq_len * 10:
            raise RuntimeError(
                f"Insufficient DART data: {len(water_levels)} samples. "
                "Need at least {seq_len * 10} for training."
            )

        water_levels_arr = np.array(water_levels, dtype=np.float32)
        n_samples = len(water_levels_arr) // seq_len
        waveforms = np.zeros((n_samples, seq_len), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.float32)
        wave_heights = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            start_idx = i * seq_len
            waveform = water_levels_arr[start_idx : start_idx + seq_len]
            waveforms[i] = waveform

            amplitude = np.max(waveform) - np.min(waveform)
            wave_heights[i] = amplitude

            fft_result = np.abs(fft(waveform))
            low_freq_power = np.sum(fft_result[1:10])
            high_freq_power = np.sum(fft_result[10:])
            is_tsunami = low_freq_power > high_freq_power * 2 and amplitude > 0.5
            labels[i] = float(is_tsunami)

        return waveforms, labels, wave_heights

    try:
        result: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] = (
            circuit_breaker.call(_fetch_dart_data)
        )
        return result
    except Exception as e:
        logger.warning(f"Failed to load DART buoy data: {e}. Using synthetic fallback.")
        return None


def load_noaa_tsunami_records(
    min_year: int = 2000,
    max_records: int = 1000,
) -> list[dict[str, Any]] | None:
    """
    Load historical tsunami event records from NOAA NGDC.

    Data source: NOAA National Centers for Environmental Information
    https://www.ngdc.noaa.gov/hazel/view/hazards/tsunami/event-search

    Args:
        min_year: Minimum year for historical records
        max_records: Maximum number of records to fetch

    Returns:
        List of tsunami event dictionaries or None if API unavailable
    """
    circuit_breaker = get_data_loader_breaker("noaa_tsunami")

    def _fetch_tsunami_records() -> list[dict[str, Any]]:
        url = f"{NOAA_TSUNAMI_API_URL}?minYear={min_year}&maxSize={max_records}"
        if not url.startswith("https://"):
            raise RuntimeError("NOAA Tsunami API URL must use HTTPS")

        # Validate URL before opening (SSRF protection via domain allowlist)
        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        TrustedEndpoints.validate_url(NOAA_TSUNAMI_API_URL)
        req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
        with urlopen(req, timeout=30) as response:  # nosec B310
            data = json.loads(response.read().decode())

        events: list[dict[str, Any]] = data.get("items", [])
        if not events:
            raise RuntimeError("NOAA Tsunami API returned no events")

        return events

    try:
        result: list[dict[str, Any]] = circuit_breaker.call(_fetch_tsunami_records)
        return result
    except Exception as e:
        logger.warning(f"Failed to load NOAA tsunami records: {e}. Using synthetic fallback.")
        return None


def load_usgs_earthquake_catalog(
    days_back: int = 365,
    min_magnitude: float = 4.0,
    n_freq_bins: int = 64,
    n_time_bins: int = 64,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """
    Load earthquake data from USGS Earthquake Catalog API.

    Converts earthquake metadata into synthetic spectrograms based on
    magnitude, depth, and location for training seismic analyzers.

    Data source: USGS Earthquake Hazards Program
    https://earthquake.usgs.gov/fdsnws/event/1/

    Args:
        days_back: Number of days of historical data
        min_magnitude: Minimum earthquake magnitude
        n_freq_bins: Number of frequency bins for spectrograms
        n_time_bins: Number of time bins for spectrograms

    Returns:
        Tuple of (spectrograms, labels, magnitudes) or None if API unavailable
    """
    circuit_breaker = get_data_loader_breaker("usgs_earthquake_catalog")

    def _fetch_earthquake_data() -> (
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ):
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)

        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%d"),
            "endtime": end_time.strftime("%Y-%m-%d"),
            "minmagnitude": str(min_magnitude),
            "limit": "1000",
        }

        url = f"{USGS_EARTHQUAKE_API_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
        if not url.startswith("https://"):
            raise RuntimeError("USGS API URL must use HTTPS")

        # Validate URL before opening (SSRF protection via domain allowlist)
        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        TrustedEndpoints.validate_url(USGS_EARTHQUAKE_API_URL)
        req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
        with urlopen(req, timeout=30) as response:  # nosec B310
            data = json.loads(response.read().decode())

        features = data.get("features", [])
        if not features:
            raise RuntimeError("USGS Earthquake API returned no events")

        n_samples = len(features)
        spectrograms = np.zeros((n_samples, 1, n_freq_bins, n_time_bins), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.float32)
        magnitudes = np.zeros(n_samples, dtype=np.float32)

        rng = np.random.default_rng(42)

        for i, feature in enumerate(features):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {}).get("coordinates", [0, 0, 0])

            mag = float(props.get("mag", 0) or 0)
            depth = float(geom[2]) if len(geom) > 2 else 10.0

            magnitudes[i] = mag
            labels[i] = 1.0 if mag >= min_magnitude else 0.0

            spectrogram = rng.normal(0, 0.1, (n_freq_bins, n_time_bins))

            p_wave_start = int(5 + depth / 100)
            p_wave_duration = int(5 + mag)
            p_wave_intensity = mag / 8.0

            if p_wave_start + p_wave_duration < n_time_bins:
                spectrogram[
                    n_freq_bins // 2 :, p_wave_start : p_wave_start + p_wave_duration
                ] += p_wave_intensity * rng.uniform(0.5, 1.0, (n_freq_bins // 2, p_wave_duration))

            s_wave_start = p_wave_start + p_wave_duration + int(depth / 50)
            s_wave_duration = int(10 + mag * 2)
            s_wave_intensity = mag / 6.0

            if s_wave_start + s_wave_duration < n_time_bins:
                spectrogram[
                    : n_freq_bins // 2, s_wave_start : s_wave_start + s_wave_duration
                ] += s_wave_intensity * rng.uniform(0.5, 1.0, (n_freq_bins // 2, s_wave_duration))

            spectrograms[i, 0] = spectrogram

        return spectrograms, labels, magnitudes

    try:
        result: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] = (
            circuit_breaker.call(_fetch_earthquake_data)
        )
        return result
    except Exception as e:
        logger.warning(f"Failed to load USGS earthquake catalog: {e}. Using synthetic fallback.")
        return None


# =============================================================================
# NASA CNEOS Fireball and Near-Earth Object Data Loaders
# =============================================================================


@dataclass
class FireballEvent:
    """
    NASA CNEOS Fireball event data.

    Represents a bolide (fireball) detected by US Government sensors.
    Data source: NASA JPL Center for Near Earth Object Studies (CNEOS)
    https://cneos.jpl.nasa.gov/fireballs/
    """

    date: datetime
    latitude: float | None
    longitude: float | None
    altitude_km: float | None
    velocity_km_s: float | None
    total_radiated_energy_j: float | None
    calculated_total_impact_energy_kt: float | None

    @property
    def estimated_size_m(self) -> float | None:
        """Estimate size from impact energy using empirical relation.

        Based on: E = 4.185 × 10^10 × D^3 (Brown et al., 2002)
        Where E is energy in Joules and D is diameter in meters.
        """
        if self.calculated_total_impact_energy_kt is None:
            return None
        # Convert kt TNT to Joules (1 kt = 4.184e12 J)
        energy_j = self.calculated_total_impact_energy_kt * 4.184e12
        # Solve for diameter: D = (E / 4.185e10)^(1/3)
        diameter = (energy_j / 4.185e10) ** (1 / 3)
        return float(diameter)


@dataclass
class CloseApproachEvent:
    """
    NASA CNEOS Close Approach event data.

    Represents a near-Earth object (NEO) close approach event.
    Data source: NASA JPL CNEOS Close Approach Data API
    https://ssd-api.jpl.nasa.gov/doc/cad.html
    """

    designation: str
    close_approach_date: datetime
    nominal_distance_au: float
    nominal_distance_km: float
    relative_velocity_km_s: float
    absolute_magnitude_h: float | None
    estimated_diameter_km: float | None


@dataclass
class SentryImpactRisk:
    """
    NASA Sentry impact monitoring data.

    Represents a potential future Earth impact event monitored by Sentry.
    Data source: NASA JPL Sentry Impact Monitoring System
    https://cneos.jpl.nasa.gov/sentry/
    """

    designation: str
    potential_impacts: int
    impact_probability: float
    palermo_scale: float
    torino_scale: int
    estimated_diameter_km: float | None
    next_impact_date: datetime | None


def load_nasa_fireball_data(
    days_back: int = 365,
    min_energy_kt: float = 0.0,
) -> list[FireballEvent] | None:
    """
    Load fireball/bolide data from NASA CNEOS Fireball API.

    Data source: NASA JPL Center for Near Earth Object Studies (CNEOS)
    https://ssd-api.jpl.nasa.gov/doc/fireball.html

    This API provides data on fireballs and bolides detected by US Government
    sensors. The data includes location, velocity, and energy estimates.

    Args:
        days_back: Number of days of historical data to fetch
        min_energy_kt: Minimum impact energy in kilotons TNT

    Returns:
        List of FireballEvent objects or None if API unavailable
    """
    circuit_breaker = get_data_loader_breaker("nasa_fireball")

    def _fetch_fireball_data() -> list[FireballEvent]:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        url = (
            f"{NASA_CNEOS_FIREBALL_URL}"
            f"?date-min={start_date.strftime('%Y-%m-%d')}"
            f"&date-max={end_date.strftime('%Y-%m-%d')}"
            f"&req-loc=true"
        )

        if not url.startswith("https://"):
            raise RuntimeError("NASA Fireball API URL must use HTTPS")

        # Validate URL before opening (SSRF protection via domain allowlist)
        TrustedEndpoints.validate_url(NASA_CNEOS_FIREBALL_URL)
        req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
        with urlopen(req, timeout=30) as response:  # nosec B310
            data = json.loads(response.read().decode())

        if "data" not in data or not data["data"]:
            logger.info("NASA Fireball API returned no events")
            return []

        # Parse field indices from response
        fields = {f: i for i, f in enumerate(data.get("fields", []))}

        events: list[FireballEvent] = []
        for row in data["data"]:
            try:
                # Parse date
                date_str = row[fields.get("date", 0)]
                event_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

                # Parse location (may be None if not available)
                lat = float(row[fields.get("lat", 1)]) if row[fields.get("lat", 1)] else None
                lon = float(row[fields.get("lon", 2)]) if row[fields.get("lon", 2)] else None
                lat_dir = row[fields.get("lat-dir", 3)]
                lon_dir = row[fields.get("lon-dir", 4)]

                if lat is not None and lat_dir == "S":
                    lat = -lat
                if lon is not None and lon_dir == "W":
                    lon = -lon

                # Parse other fields
                alt = float(row[fields.get("alt", 5)]) if row[fields.get("alt", 5)] else None
                vel = float(row[fields.get("vel", 6)]) if row[fields.get("vel", 6)] else None
                energy_rad = (
                    float(row[fields.get("energy", 7)]) if row[fields.get("energy", 7)] else None
                )
                energy_impact = (
                    float(row[fields.get("impact-e", 8)])
                    if row[fields.get("impact-e", 8)]
                    else None
                )

                # Filter by minimum energy
                if min_energy_kt > 0 and (energy_impact is None or energy_impact < min_energy_kt):
                    continue

                events.append(
                    FireballEvent(
                        date=event_date,
                        latitude=lat,
                        longitude=lon,
                        altitude_km=alt,
                        velocity_km_s=vel,
                        total_radiated_energy_j=energy_rad,
                        calculated_total_impact_energy_kt=energy_impact,
                    )
                )
            except (ValueError, IndexError, KeyError) as e:
                logger.debug(f"Skipping malformed fireball record: {e}")
                continue

        return events

    try:
        result: list[FireballEvent] = circuit_breaker.call(_fetch_fireball_data)
        logger.info(f"Loaded {len(result)} NASA CNEOS fireball events")
        return result
    except Exception as e:
        logger.warning(f"Failed to load NASA fireball data: {e}")
        return None


def load_nasa_close_approach_data(
    days_forward: int = 60,
    distance_max_au: float = 0.05,
) -> list[CloseApproachEvent] | None:
    """Load close approach data from NASA CNEOS Close Approach API.

    Data source: NASA JPL CNEOS Close Approach Data API
    https://ssd-api.jpl.nasa.gov/doc/cad.html

    This API provides predicted close approach data for near-Earth objects
    (asteroids and comets) with Earth.

    Args:
        days_forward: Number of days to look ahead for close approaches
        distance_max_au: Maximum close approach distance in AU (1 AU = ~150M km)

    Returns:
        List of CloseApproachEvent objects or None if API unavailable
    """
    circuit_breaker = get_data_loader_breaker("nasa_cad")

    def _fetch_cad_data() -> list[CloseApproachEvent]:
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=days_forward)

        url = (
            f"{NASA_CNEOS_CAD_URL}"
            f"?date-min={start_date.strftime('%Y-%m-%d')}"
            f"&date-max={end_date.strftime('%Y-%m-%d')}"
            f"&dist-max={distance_max_au}"
            f"&body=Earth"
        )

        if not url.startswith("https://"):
            raise RuntimeError("NASA CAD API URL must use HTTPS")

        TrustedEndpoints.validate_url(NASA_CNEOS_CAD_URL)
        req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
        with urlopen(req, timeout=30) as response:  # nosec B310
            data = json.loads(response.read().decode())

        if "data" not in data or not data["data"]:
            logger.info("NASA CAD API returned no close approaches")
            return []

        fields = {f: i for i, f in enumerate(data.get("fields", []))}

        events: list[CloseApproachEvent] = []
        for row in data["data"]:
            try:
                designation = row[fields.get("des", 0)]
                date_str = row[fields.get("cd", 3)]
                ca_date = datetime.strptime(date_str, "%Y-%b-%d %H:%M")

                dist_au = float(row[fields.get("dist", 4)])
                dist_km = dist_au * 149597870.7  # AU to km

                v_rel = float(row[fields.get("v_rel", 7)])
                h_mag = float(row[fields.get("h", 10)]) if row[fields.get("h", 10)] else None

                # Estimate diameter from absolute magnitude H
                # D = 1329 / sqrt(albedo) * 10^(-H/5)
                # Assuming albedo = 0.15 (typical for rocky asteroids)
                diameter_km = None
                if h_mag is not None:
                    diameter_km = 1329 / (0.15**0.5) * (10 ** (-h_mag / 5))

                events.append(
                    CloseApproachEvent(
                        designation=designation,
                        close_approach_date=ca_date,
                        nominal_distance_au=dist_au,
                        nominal_distance_km=dist_km,
                        relative_velocity_km_s=v_rel,
                        absolute_magnitude_h=h_mag,
                        estimated_diameter_km=diameter_km,
                    )
                )
            except (ValueError, IndexError, KeyError) as e:
                logger.debug(f"Skipping malformed CAD record: {e}")
                continue

        return events

    try:
        result: list[CloseApproachEvent] = circuit_breaker.call(_fetch_cad_data)
        logger.info(f"Loaded {len(result)} NASA CNEOS close approach events")
        return result
    except Exception as e:
        logger.warning(f"Failed to load NASA close approach data: {e}")
        return None


def load_nasa_sentry_data() -> list[SentryImpactRisk] | None:
    """
    Load potential impact data from NASA Sentry Impact Monitoring API.

    Data source: NASA JPL Sentry Impact Monitoring System
    https://ssd-api.jpl.nasa.gov/doc/sentry.html

    Sentry is a highly automated collision monitoring system that continually
    scans the most current asteroid catalog for possibilities of future impact
    with Earth over the next 100 years.

    Returns:
        List of SentryImpactRisk objects or None if API unavailable
    """
    circuit_breaker = get_data_loader_breaker("nasa_sentry")

    def _fetch_sentry_data() -> list[SentryImpactRisk]:
        url = f"{NASA_SENTRY_URL}?all=1"

        if not url.startswith("https://"):
            raise RuntimeError("NASA Sentry API URL must use HTTPS")

        TrustedEndpoints.validate_url(NASA_SENTRY_URL)
        req = Request(url, headers={"User-Agent": "Mercury-Agent/1.0"})
        with urlopen(req, timeout=30) as response:  # nosec B310
            data = json.loads(response.read().decode())

        if "data" not in data or not data["data"]:
            logger.info("NASA Sentry API returned no impact risks (good news!)")
            return []

        risks: list[SentryImpactRisk] = []
        for obj in data["data"]:
            try:
                designation = obj.get("des", "Unknown")
                n_imp = int(obj.get("n_imp", 0))
                ip = float(obj.get("ip", 0))
                ps = float(obj.get("ps", -10))
                ts = int(obj.get("ts", 0))
                diameter = float(obj.get("diameter", 0)) if obj.get("diameter") else None

                # Parse next impact date if available
                next_impact = None
                if obj.get("range"):
                    try:
                        date_str = obj["range"].split("-")[0].strip()
                        next_impact = datetime.strptime(date_str, "%Y")
                    except (ValueError, AttributeError):
                        pass

                risks.append(
                    SentryImpactRisk(
                        designation=designation,
                        potential_impacts=n_imp,
                        impact_probability=ip,
                        palermo_scale=ps,
                        torino_scale=ts,
                        estimated_diameter_km=diameter,
                        next_impact_date=next_impact,
                    )
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping malformed Sentry record: {e}")
                continue

        return risks

    try:
        result: list[SentryImpactRisk] = circuit_breaker.call(_fetch_sentry_data)
        logger.info(f"Loaded {len(result)} NASA Sentry impact risk objects")
        return result
    except Exception as e:
        logger.warning(f"Failed to load NASA Sentry data: {e}")
        return None


def train_waveform_analyzer(
    model: WaveformFFTAnalyzer,
    n_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    n_samples: int = 1000,
    device: str = "cpu",
    use_real_data: bool = True,
) -> dict[str, list[float]]:
    """
    Train WaveformFFTAnalyzer on tsunami data.

    Attempts to load real-world data from NOAA DART buoy network first,
    falling back to synthetic data if the API is unavailable.

    Real-world data sources:
    - NOAA DART buoy network (https://www.ndbc.noaa.gov/dart.shtml)
    - NOAA tsunami event records (https://www.ngdc.noaa.gov/hazel/)

    Args:
        model: WaveformFFTAnalyzer model to train
        n_epochs: Number of training epochs (default 10)
        batch_size: Training batch size
        learning_rate: Adam optimizer learning rate
        n_samples: Number of synthetic samples to generate (fallback)
        device: Training device ('cpu' or 'cuda')
        use_real_data: Whether to attempt loading real-world data first

    Returns:
        Training history with loss and accuracy per epoch
    """
    model = model.to(device)
    model.train()

    data_source = "synthetic"
    waveforms = None
    labels = None
    wave_heights = None

    if use_real_data:
        logger.info("Attempting to load real DART buoy data for tsunami training...")
        real_data = load_dart_buoy_data()
        if real_data is not None:
            waveforms, labels, wave_heights = real_data
            data_source = "real (DART buoy)"
            logger.info(f"Loaded {len(waveforms)} real tsunami waveform samples")

    if waveforms is None:
        logger.info(f"Using synthetic tsunami data ({n_samples} samples)")
        waveforms, labels, wave_heights = generate_synthetic_tsunami_data(n_samples)

    logger.info(f"Training WaveformFFTAnalyzer for {n_epochs} epochs on {data_source} data")

    # Convert to tensors
    waveforms_tensor = torch.tensor(waveforms, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    wave_heights_tensor = torch.tensor(wave_heights, dtype=torch.float32).to(device)

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    history: dict[str, list[float]] = {"loss": [], "accuracy": [], "height_mse": []}

    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_height_mse = 0.0

        # Shuffle data
        indices = torch.randperm(n_samples)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_waveforms = waveforms_tensor[batch_indices]
            batch_labels = labels_tensor[batch_indices]
            batch_heights = wave_heights_tensor[batch_indices]

            optimizer.zero_grad()

            # Forward pass
            pred_prob, pred_height = model(batch_waveforms)

            # Compute losses
            classification_loss = bce_loss(pred_prob, batch_labels)
            height_loss = mse_loss(pred_height, batch_heights)
            total_loss = classification_loss + 0.1 * height_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_correct += ((pred_prob > 0.5).float() == batch_labels).sum().item()
            epoch_height_mse += height_loss.item()

        avg_loss = epoch_loss / n_batches
        accuracy = epoch_correct / n_samples
        avg_height_mse = epoch_height_mse / n_batches

        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        history["height_mse"].append(avg_height_mse)

        logger.info(
            f"Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}, "
            f"accuracy={accuracy:.4f}, height_mse={avg_height_mse:.4f}"
        )

    model.eval()
    logger.info(
        f"WaveformFFTAnalyzer training complete. Final accuracy: {history['accuracy'][-1]:.4f}"
    )

    return history


def train_seismic_analyzer(
    model: SeismicWaveAnalyzer,
    n_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    n_samples: int = 1000,
    device: str = "cpu",
    use_real_data: bool = True,
) -> dict[str, list[float]]:
    """
    Train SeismicWaveAnalyzer on earthquake data.

    Attempts to load real-world data from USGS Earthquake Catalog first,
    falling back to synthetic data if the API is unavailable.

    Real-world data sources:
    - USGS Earthquake Hazards Program (https://earthquake.usgs.gov/)
    - USGS FDSN Event Web Service (https://earthquake.usgs.gov/fdsnws/event/1/)

    Args:
        model: SeismicWaveAnalyzer model to train
        n_epochs: Number of training epochs (default 10)
        batch_size: Training batch size
        learning_rate: Adam optimizer learning rate
        n_samples: Number of synthetic samples to generate (fallback)
        device: Training device ('cpu' or 'cuda')
        use_real_data: Whether to attempt loading real-world data first

    Returns:
        Training history with loss and accuracy per epoch
    """
    model = model.to(device)
    model.train()

    data_source = "synthetic"
    spectrograms = None
    labels = None
    magnitudes = None

    if use_real_data:
        logger.info("Attempting to load real USGS earthquake catalog data...")
        real_data = load_usgs_earthquake_catalog()
        if real_data is not None:
            spectrograms, labels, magnitudes = real_data
            data_source = "real (USGS catalog)"
            logger.info(f"Loaded {len(spectrograms)} real earthquake samples")

    if spectrograms is None:
        logger.info(f"Using synthetic earthquake data ({n_samples} samples)")
        spectrograms, labels, magnitudes = generate_synthetic_earthquake_data(n_samples)

    logger.info(f"Training SeismicWaveAnalyzer for {n_epochs} epochs on {data_source} data")

    # Convert to tensors
    spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    magnitudes_tensor = torch.tensor(magnitudes, dtype=torch.float32).to(device)

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    history: dict[str, list[float]] = {"loss": [], "accuracy": [], "magnitude_mse": []}

    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_mag_mse = 0.0

        # Shuffle data
        indices = torch.randperm(n_samples)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_spectrograms = spectrograms_tensor[batch_indices]
            batch_labels = labels_tensor[batch_indices]
            batch_magnitudes = magnitudes_tensor[batch_indices]

            optimizer.zero_grad()

            # Forward pass
            pred_prob, pred_magnitude = model(batch_spectrograms)

            # Compute losses
            classification_loss = bce_loss(pred_prob, batch_labels)
            magnitude_loss = mse_loss(pred_magnitude, batch_magnitudes)
            total_loss = classification_loss + 0.1 * magnitude_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_correct += ((pred_prob > 0.5).float() == batch_labels).sum().item()
            epoch_mag_mse += magnitude_loss.item()

        avg_loss = epoch_loss / n_batches
        accuracy = epoch_correct / n_samples
        avg_mag_mse = epoch_mag_mse / n_batches

        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        history["magnitude_mse"].append(avg_mag_mse)

        logger.info(
            f"Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}, "
            f"accuracy={accuracy:.4f}, magnitude_mse={avg_mag_mse:.4f}"
        )

    model.eval()
    logger.info(
        f"SeismicWaveAnalyzer training complete. Final accuracy: {history['accuracy'][-1]:.4f}"
    )

    return history


def train_all_disaster_networks(
    device: str = "cpu",
    n_epochs: int = 10,
) -> dict[str, dict[str, list[float]]]:
    """
    Train all disaster detection neural networks on synthetic data.

    This function initializes and trains:
    - WaveformFFTAnalyzer for tsunami detection
    - SeismicWaveAnalyzer for earthquake detection

    Args:
        device: Training device ('cpu' or 'cuda')
        n_epochs: Number of training epochs per model

    Returns:
        Dictionary mapping model names to training histories
    """
    logger.info("Training all disaster detection neural networks...")

    results = {}

    # Train WaveformFFTAnalyzer
    waveform_model = WaveformFFTAnalyzer()
    results["WaveformFFTAnalyzer"] = train_waveform_analyzer(
        waveform_model, n_epochs=n_epochs, device=device
    )

    # Train SeismicWaveAnalyzer
    seismic_model = SeismicWaveAnalyzer()
    results["SeismicWaveAnalyzer"] = train_seismic_analyzer(
        seismic_model, n_epochs=n_epochs, device=device
    )

    logger.info("All disaster detection networks trained successfully.")

    return results
