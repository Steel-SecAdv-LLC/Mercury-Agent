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

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy import signal
from scipy.fft import fft, fftfreq
from torch import nn

from omni_anomaly_engine.utils.rng import get_global_rng

logger = logging.getLogger(__name__)

# Feature dimension for fusion pipeline
FEATURE_DIM = 20


class TsunamiSeverity(Enum):
    """Tsunami severity classification based on wave height"""

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
    """Meteor/asteroid threat classification"""

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
    """Tsunami prediction results"""

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
    """Earthquake prediction results"""

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
    """Meteor/asteroid prediction results"""

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
    """Solar flare prediction results"""

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


class WaveformFFTAnalyzer(nn.Module):
    """FFT-based waveform analyzer for tsunami detection.

    Analyzes oceanic waveform patterns using frequency domain analysis
    integrated with 3R Resonance mechanism.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64):
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
        """Forward pass for waveform analysis.

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


class SeismicWaveAnalyzer(nn.Module):
    """P/S-wave spectrogram analyzer for earthquake detection.

    Uses scipy.signal for spectrogram computation and neural network
    for classification.
    """

    def __init__(self, n_freq_bins: int = 64, hidden_dim: int = 128):
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
        """Forward pass for seismic analysis.

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
        """Update posterior probability using Bayes' theorem.

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

    def __init__(self, n_states: int = 5):
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
        """Predict most likely next state.

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
        """Update state based on observed X-ray flux.

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
        """Get probability of geomagnetic storm given current state.

        Args:
            state: State index (uses current if None)

        Returns:
            Storm probability [0, 1]
        """
        s = state if state is not None else self.current_state
        storm_probs = [0.01, 0.05, 0.15, 0.45, 0.85]
        return storm_probs[s]


class TsunamiDetector:
    """Tsunami detector using oceanic waveform FFT analysis.

    Integrates with 3R Resonance mechanism for frequency-domain
    anomaly detection in oceanic sensor data.
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
        waveform_data: np.ndarray | torch.Tensor,
        source_info: dict[str, Any] | None = None,
    ) -> TsunamiPredictionResult:
        """Predict tsunami from oceanic waveform data.

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
                local_power = power_spectrum[max(0, idx - 2) : idx + 3].mean()
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

    def extract_features(self, waveform_data: np.ndarray | torch.Tensor) -> np.ndarray:
        """Extract features for fusion pipeline.

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
    """Earthquake detector using P/S-wave spectrogram analysis.

    Uses scipy.signal for spectrogram computation and integrates
    with 3R Resonance for frequency-domain analysis.
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
        seismic_data: np.ndarray | torch.Tensor,
        station_info: dict[str, Any] | None = None,
    ) -> EarthquakePredictionResult:
        """Predict earthquake from seismic waveform data.

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

    def _compute_resonance_score(self, Sxx: np.ndarray, freqs: np.ndarray) -> float:
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

    def _detect_wave_arrival(self, data: np.ndarray, wave_type: str) -> int | None:
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

    def _find_spectral_anomalies(self, Sxx: np.ndarray, freqs: np.ndarray) -> list[float]:
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

    def extract_features(self, seismic_data: np.ndarray | torch.Tensor) -> np.ndarray:
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
    """Meteor detector using optical/radar Bayesian filter.

    Combines optical and radar observations with Bayesian inference
    for meteor/asteroid detection and trajectory estimation.
    """

    def __init__(
        self,
        detection_threshold: float = 0.7,
        prior_probability: float = 1e-6,
    ):
        self.detection_threshold = detection_threshold
        self.bayesian_filter = BayesianMeteorFilter(prior_probability=prior_probability)
        self.rng = get_global_rng()

        logger.info(f"MeteorDetector initialized: threshold={detection_threshold}")

    def predict_meteor(
        self,
        optical_data: np.ndarray | None = None,
        radar_data: np.ndarray | None = None,
        noaa_stub: dict[str, Any] | None = None,
    ) -> MeteorPredictionResult:
        """Predict meteor from optical and radar data.

        Args:
            optical_data: Optical sensor data (brightness measurements)
            radar_data: Radar return data
            noaa_stub: Optional NOAA data stub for external integration

        Returns:
            MeteorPredictionResult with detection details
        """
        optical_detection = False
        radar_detection = False

        if optical_data is not None:
            optical_threshold = np.percentile(optical_data, 99)
            optical_detection = np.max(optical_data) > optical_threshold * 1.5

        if radar_data is not None:
            radar_threshold = np.percentile(radar_data, 99)
            radar_detection = np.max(radar_data) > radar_threshold * 1.5

        if noaa_stub is not None:
            optical_detection = optical_detection or noaa_stub.get("optical_alert", False)
            radar_detection = radar_detection or noaa_stub.get("radar_alert", False)

        posterior = self.bayesian_filter.update(optical_detection, radar_detection)

        meteor_detected = posterior > self.detection_threshold

        threat_level = self._assess_threat(posterior, optical_data, radar_data)

        size_estimate = None
        velocity_estimate = None
        if meteor_detected and radar_data is not None:
            size_estimate = self._estimate_size(radar_data)
            velocity_estimate = self._estimate_velocity(radar_data)

        warnings = self._generate_warnings(meteor_detected, threat_level)

        return MeteorPredictionResult(
            meteor_detected=meteor_detected,
            confidence=posterior,
            threat_level=threat_level,
            estimated_size_m=size_estimate,
            estimated_velocity_kms=velocity_estimate,
            impact_probability=posterior * 0.001 if meteor_detected else 0.0,
            optical_detection=optical_detection,
            radar_detection=radar_detection,
            bayesian_posterior=posterior,
            trajectory_confidence=0.8 if radar_detection else 0.3,
            warning_actions=warnings,
        )

    def _assess_threat(
        self,
        posterior: float,
        optical_data: np.ndarray | None,
        radar_data: np.ndarray | None,
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

    def _estimate_size(self, radar_data: np.ndarray) -> float:
        """Estimate meteor size from radar cross-section."""
        rcs = np.max(radar_data)
        size = np.sqrt(rcs / np.pi) * 10
        return float(size)

    def _estimate_velocity(self, radar_data: np.ndarray) -> float:
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
        optical_data: np.ndarray | None = None,
        radar_data: np.ndarray | None = None,
    ) -> np.ndarray:
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
    """Solar flare detector using X-ray flux and geomagnetic HMM.

    Predicts solar flares and geomagnetic storms using Hidden Markov
    Model for state transitions and X-ray flux analysis.
    """

    def __init__(
        self,
        detection_threshold: float = 0.7,
    ):
        self.detection_threshold = detection_threshold
        self.hmm = GeomagneticHMM()
        self.rng = get_global_rng()

        self.flux_thresholds = {
            "A": 1e-8,
            "B": 1e-7,
            "C": 1e-6,
            "M": 1e-5,
            "X": 1e-4,
        }

        logger.info(f"SolarFlareDetector initialized: threshold={detection_threshold}")

    def predict_solar_flare(
        self,
        x_ray_flux: float | np.ndarray,
        proton_flux: float | None = None,
        magnetometer_data: np.ndarray | None = None,
    ) -> SolarFlarePredictionResult:
        """Predict solar flare from X-ray and proton flux data.

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
            proton_flux=float(np.mean(proton_flux)) if proton_flux is not None else 0.0,
            geomagnetic_storm_probability=storm_prob,
            kp_index_predicted=kp_predicted,
            dst_index_predicted=dst_predicted,
            hmm_state=current_state,
            transition_probability=transition_prob,
            warning_actions=warnings,
            affected_systems=affected,
        )

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
        x_ray_flux: float | np.ndarray,
        proton_flux: float | None = None,
    ) -> np.ndarray:
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
