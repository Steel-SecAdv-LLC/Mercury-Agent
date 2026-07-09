# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive Disaster Detectors for Humanitarian Early Warning.

Implements detection systems for:
- Tsunami: Oceanic waveform FFT analysis with Resonance integration
- Earthquake: P/S-wave spectrogram analysis via Scipy.signal
- Meteor: Optical/radar Bayesian filter fused with NASA/JPL data_sources
  clients (JPLFireballSource, NASANeoWsSource, JPLSentrySource)

The SolarFlareDetector that used to live here was a name-only duplicate of
:class:`omni_mercury_engine.space.solar_storm_detector.SolarFlareDetector`;
the canonical class (and its SolarFlarePredictionResult) is re-exported from
this module for import compatibility. Likewise the private NASA CNEOS HTTP
loaders were consolidated into ``data_sources/jpl_ssd.py`` and their
dataclasses are re-exported here.

Live ingestion follows the uniform pattern in
:mod:`omni_mercury_engine.data_sources.live_ingestion`: constructors accept an
optional data_sources client (default None = offline), ``fetch_live_data``
fails loud, and the ``*_live`` conveniences stamp ``source_id`` /
``data_provenance`` / ``live_context`` on the native result dataclasses.

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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        from torch import nn

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

from scipy import signal
from scipy.fft import fft, fftfreq

from omni_mercury_engine.data_sources.jpl_ssd import (
    CloseApproachEvent,
    FireballEvent,
    JPLFireballSource,
    JPLSentrySource,
    SentryImpactRisk,
    close_approaches_from_neows_datapoints,
    fireball_events_from_datapoints,
    sentry_risks_from_datapoints,
)
from omni_mercury_engine.data_sources.live_ingestion import (
    LiveDataError,
    fetch_live_datapoints,
    haversine_km,
    require_live_client,
)
from omni_mercury_engine.data_sources.space_weather import NASANeoWsSource
from omni_mercury_engine.resilience.api_circuit_breakers import get_data_loader_breaker
from omni_mercury_engine.security.input_validation import TrustedEndpoints
from omni_mercury_engine.security.safe_http import SafeHTTPClient

# Canonical solar-flare detector (deduplicated): re-exported from here for
# import compatibility. The class that used to live in this module carried
# fabricated per-HMM-state Kp/Dst lookup tables; the canonical implementation
# derives storm fields only from a REAL observed planetary Kp (see its
# docstring and DEPRECATION.md).
# ``__all__`` below re-exports the two imported names explicitly: pyflakes,
# ruff and mypy all treat ``__all__`` membership as an intentional re-export,
# which is the one mechanism the three tools agree on (a self-alias trips
# ruff PLC0414; a bare import without ``__all__`` trips flake8 F401 and
# mypy attr-defined).
from omni_mercury_engine.space.solar_storm_detector import (
    SolarFlareDetector,
    SolarFlarePredictionResult,
)
from omni_mercury_engine.utils.rng import get_global_rng

__all__ = [
    "BayesianMeteorFilter",
    "EarthquakeDetector",
    "EarthquakeMagnitude",
    "EarthquakePredictionResult",
    "MeteorDetector",
    "MeteorPredictionResult",
    "MeteorThreatLevel",
    "SeismicWaveAnalyzer",
    "SolarFlareClass",
    "SolarFlareDetector",
    "SolarFlarePredictionResult",
    "TsunamiDetector",
    "TsunamiPredictionResult",
    "TsunamiSeverity",
    "WaveformFFTAnalyzer",
    "generate_synthetic_earthquake_data",
    "generate_synthetic_tsunami_data",
    "load_dart_buoy_data",
    "load_usgs_earthquake_catalog",
    "train_all_disaster_networks",
    "train_seismic_analyzer",
    "train_waveform_analyzer",
]

if TYPE_CHECKING:
    from omni_mercury_engine.data_sources.earth_science import USGSEarthquakeSource
    from omni_mercury_engine.data_sources.live_ingestion import LiveFetch

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
    """Earthquake magnitude classification (Richter scale)."""

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
    """Solar flare classification (GOES X-ray flux) -- LEGACY LABELS.

    Deprecated: the canonical
    :class:`omni_mercury_engine.space.solar_storm_detector.SolarFlareDetector`
    reports NOAA letter labels ("A".."X", matching the space-module
    ``SolarFlareClass`` enum). This "a_class"-style enum is preserved for
    import compatibility only (see DEPRECATION.md).
    """

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

    # Live-ingestion provenance (populated only by predict_tsunami_live()).
    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None


@dataclass
class EarthquakePredictionResult:
    """Earthquake prediction results."""

    earthquake_detected: bool
    confidence: float
    # None when no trained model is loaded: an uncalibrated single station has
    # no honest Richter estimate (magnitude_class is "undetermined" then).
    estimated_magnitude: float | None
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

    # Live-ingestion provenance (populated only by detect_live()).
    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None


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

    # Live-ingestion provenance (populated only when live clients were used).
    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None


if TYPE_CHECKING or TORCH_AVAILABLE:

    class WaveformFFTAnalyzer(nn.Module):
        """FFT-based waveform analyzer for tsunami detection.

        Analyzes oceanic waveform patterns using frequency domain analysis integrated with 3R
        Resonance mechanism.
        """

        def __init__(self, input_dim: int = 256, hidden_dim: int = 64) -> None:
            """Initialize the instance."""
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

else:

    class WaveformFFTAnalyzer:
        """Stub: WaveformFFTAnalyzer requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the instance."""
            raise ImportError(
                "WaveformFFTAnalyzer requires PyTorch. Install with: pip install torch"
            )


if TYPE_CHECKING or TORCH_AVAILABLE:

    class SeismicWaveAnalyzer(nn.Module):
        """P/S-wave spectrogram analyzer for earthquake detection.

        Uses scipy.signal for spectrogram computation and neural network for classification.
        """

        def __init__(self, n_freq_bins: int = 64, hidden_dim: int = 128) -> None:
            """Initialize the instance."""
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

else:

    class SeismicWaveAnalyzer:
        """Stub: SeismicWaveAnalyzer requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the instance."""
            raise ImportError(
                "SeismicWaveAnalyzer requires PyTorch. Install with: pip install torch"
            )


class BayesianMeteorFilter:
    """Bayesian filter for meteor detection combining optical and radar data."""

    def __init__(
        self,
        prior_probability: float = 1e-6,
        optical_sensitivity: float = 0.8,
        radar_sensitivity: float = 0.9,
    ):
        """Initialize the instance."""
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


class TsunamiDetector:
    """Tsunami detector using oceanic waveform FFT analysis.

    Integrates with 3R Resonance mechanism for frequency-domain anomaly detection in oceanic sensor
    data.
    """

    def __init__(
        self,
        sampling_rate: float = 1.0,
        detection_threshold: float = 0.96,
        device: str = "cpu",
        data_source: USGSEarthquakeSource | None = None,
    ):
        """Initialize the instance.

        Live-ingestion pattern (uniform across hazard detectors): pass an
        optional USGS earthquake-catalog client via ``data_source``
        (dependency injection; default None = fully offline).
        :meth:`fetch_live_data` exposes a provenance-checked fetch and
        :meth:`predict_tsunami_live` enriches the waveform physics with a
        live candidate source event (magnitude + epicentral distance), never
        inventing a waveform.

        Args:
            sampling_rate: Waveform sampling rate (Hz).
            detection_threshold: Confidence threshold for detection.
            device: Torch device for the (optional) neural analyzer.
            data_source: Optional USGS earthquake-catalog client.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("TsunamiDetector requires PyTorch. Install with: pip install torch")
        self.sampling_rate = sampling_rate
        self.detection_threshold = detection_threshold
        self.device = torch.device(device)
        self.rng = get_global_rng()
        self._catalog_source = data_source

        self.waveform_analyzer = WaveformFFTAnalyzer().to(self.device)
        self.waveform_analyzer.eval()

        # Anti-theater guard (mirrors SchumannResonanceDetector): the
        # WaveformFFTAnalyzer ships with random weights and no labelled tsunami
        # corpus exists to train it. Until real weights are loaded via
        # load_neural_weights(), its probability/wave-height outputs are noise,
        # so predict_tsunami derives both from the OBSERVED record instead: the
        # wave height is the peak sea-level deviation from the median baseline
        # (what a DART bottom-pressure recorder actually measures) and the
        # confidence is a noisy-OR of that amplitude severity with the
        # tsunami-band FFT resonance score.
        self._neural_trained = False
        self._warned_untrained = False

        self.tsunami_frequencies = [0.001, 0.005, 0.01, 0.02]

        logger.info(f"TsunamiDetector initialized: threshold={detection_threshold}")

    def load_neural_weights(self, checkpoint_path: str) -> None:
        """Load trained weights for the waveform analyzer.

        Until this is called the network is untrained and detection runs on the
        deterministic amplitude + resonance physics of the observed waveform.

        Args:
            checkpoint_path: Path to a torch checkpoint containing a
                ``waveform_analyzer`` state dict.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.waveform_analyzer.load_state_dict(checkpoint["waveform_analyzer"])
        self._neural_trained = True
        logger.info(
            "Tsunami neural weights loaded from %s; using learned analyzer", checkpoint_path
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the untrained NN is bypassed for physics."""
        if not self._warned_untrained:
            logger.warning(
                "TsunamiDetector's WaveformFFTAnalyzer is untrained (no checkpoint "
                "loaded); deriving wave height and confidence from the observed "
                "waveform amplitude + tsunami-band resonance instead of the NN. "
                "Call load_neural_weights() once a trained checkpoint exists."
            )
            self._warned_untrained = True

    def predict_tsunami(
        self,
        waveform_data: np.ndarray[Any, Any] | torch.Tensor,
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
                local_power = power_spectrum[max(0, idx - 2) : idx + 3].mean()  # type: ignore[misc, unused-ignore]
                global_power = power_spectrum.mean() + 1e-10
                if local_power / global_power > 2.0:
                    resonance_score += 0.25
                    dominant_freqs.append(float(freqs[idx]))

        if self._neural_trained:
            with torch.no_grad():
                tsunami_prob, wave_height = self.waveform_analyzer(waveform_data)
            confidence = float(tsunami_prob[0].item())
            confidence = min(1.0, confidence + resonance_score * 0.3)
            wave_height_m = float(wave_height[0].item())
        else:
            # Physics path: the wave height IS the observed peak sea-level
            # deviation from the median baseline, and the confidence is a
            # noisy-OR of the robust amplitude severity with the tsunami-band
            # resonance score. Deterministic; nothing is fabricated.
            self._warn_untrained_once()
            record = waveform_data.cpu().numpy()[0]
            median = float(np.median(record))
            wave_height_m = float(np.max(np.abs(record - median)))
            # Noise floor from the QUIETEST segment of the record, not the whole
            # record -- a long-period tsunami excursion would otherwise inflate
            # its own baseline (self-masking). Deterministic.
            n_segments = max(1, min(8, len(record) // 64))
            segment_scales = []
            for seg in np.array_split(record, n_segments):
                seg_mad = float(np.median(np.abs(seg - np.median(seg))))
                if seg_mad > 0:
                    segment_scales.append(1.4826 * seg_mad)
            scale = min(segment_scales) if segment_scales else (float(np.std(record)) or 1.0)
            z_peak = wave_height_m / scale
            # z 5 (ordinary extreme of noise) → 0; z 20 (unambiguous long-period
            # excursion) saturates. Resonance compounds via noisy-OR.
            amplitude_severity = float(np.clip((z_peak - 5.0) / 15.0, 0.0, 1.0))
            confidence = 1.0 - (1.0 - amplitude_severity) * (1.0 - resonance_score)

        tsunami_detected = confidence > self.detection_threshold

        severity = self._determine_severity(wave_height_m)

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
            estimated_wave_height_m=wave_height_m,
            arrival_time_minutes=arrival_time,
            source_distance_km=source_info.get("distance_km") if source_info else None,
            source_magnitude=source_info.get("magnitude") if source_info else None,
            resonance_score=resonance_score,
            dominant_frequencies=dominant_freqs,
            waveform_anomaly_score=float(np.std(power_spectrum)),
            warning_actions=warnings,
            evacuation_zones=zones,
        )

    def fetch_live_data(self, *, allow_simulated: bool = False, **kwargs: Any) -> LiveFetch:
        """Fetch live USGS earthquake-catalog events through the injected client.

        Args:
            allow_simulated: Explicit opt-in for simulated sources (the USGS
                catalog is a real feed, so this normally stays False).
            **kwargs: Passed to the client fetch (e.g. ``min_magnitude=``).

        Returns:
            Provenance-checked LiveFetch.

        Raises:
            LiveDataError: No catalog client injected, or the fetch failed.
        """
        client = require_live_client(self._catalog_source, "TsunamiDetector", "USGS earthquake")
        return fetch_live_datapoints(client, allow_simulated=allow_simulated, **kwargs)

    def predict_tsunami_live(
        self,
        waveform_data: np.ndarray[Any, Any] | torch.Tensor,
        station_lat: float,
        station_lon: float,
        *,
        min_magnitude: float = 6.5,
        allow_simulated: bool = False,
    ) -> TsunamiPredictionResult:
        """Predict tsunami from an observed waveform + live USGS catalog context.

        Maps live catalog events onto the existing :meth:`predict_tsunami`
        input contract: the waveform physics is unchanged (the waveform must
        come from the caller's gauge -- it is NEVER invented from catalog
        metadata), while the ``source_info`` argument (source magnitude +
        epicentral distance, which drive the arrival-time estimate) is
        populated from the most tsunamigenic recent catalog event: a
        tsunami-flagged event if any, else the largest magnitude event at or
        above ``min_magnitude``.

        Args:
            waveform_data: Sea level / bottom-pressure waveform from the
                caller's instrument.
            station_lat: Gauge latitude (degrees) for epicentral distance.
            station_lon: Gauge longitude (degrees) for epicentral distance.
            min_magnitude: Catalog magnitude floor for candidate sources.
            allow_simulated: Explicit opt-in for simulated sources.

        Returns:
            TsunamiPredictionResult with ``source_id`` / ``data_provenance`` /
            ``live_context`` populated.

        Raises:
            LiveDataError: No catalog client injected, or the fetch failed.
        """
        fetch = self.fetch_live_data(allow_simulated=allow_simulated, min_magnitude=min_magnitude)

        candidates = [dp for dp in fetch.data_points if dp.location is not None]
        source_info: dict[str, Any] | None = None
        candidate_context: dict[str, Any] | None = None
        if candidates:
            tsunami_flagged = [dp for dp in candidates if dp.data.get("tsunami")]
            pool = tsunami_flagged or candidates
            strongest = max(pool, key=lambda dp: float(dp.data.get("magnitude", 0.0)))
            lat, lon, _depth = strongest.location  # type: ignore[misc]
            distance_km = haversine_km(station_lat, station_lon, lat, lon)
            source_info = {
                "distance_km": distance_km,
                "magnitude": float(strongest.data.get("magnitude", 0.0)),
            }
            candidate_context = {
                "event_id": strongest.event_id,
                "place": strongest.data.get("place"),
                "magnitude": strongest.data.get("magnitude"),
                "tsunami_flagged": bool(strongest.data.get("tsunami")),
                "distance_km": distance_km,
                "event_time": strongest.timestamp.isoformat(),
            }

        result = self.predict_tsunami(waveform_data, source_info)
        result.source_id = fetch.source_id
        result.data_provenance = fetch.data_provenance
        result.live_context = {
            "catalog_events": len(fetch.data_points),
            "min_magnitude": min_magnitude,
            "candidate_source_event": candidate_context,
        }
        return result

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

    Uses scipy.signal for spectrogram computation and integrates with 3R Resonance for frequency-
    domain analysis.
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        detection_threshold: float = 0.96,
        device: str = "cpu",
        data_source: USGSEarthquakeSource | None = None,
    ):
        """Initialize the instance.

        Live-ingestion pattern (uniform across hazard detectors): pass an
        optional USGS earthquake-catalog client via ``data_source``
        (dependency injection; default None = fully offline).
        :meth:`fetch_live_data` exposes a provenance-checked fetch and
        :meth:`detect_live` builds an event-stream assessment (observed
        catalog magnitudes, rate and clustering features) -- catalog metadata
        is never turned into synthetic waveforms.

        Args:
            sampling_rate: Waveform sampling rate (Hz).
            detection_threshold: Confidence threshold for detection.
            device: Torch device for the (optional) neural analyzer.
            data_source: Optional USGS earthquake-catalog client.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "EarthquakeDetector requires PyTorch. Install with: pip install torch"
            )
        self.sampling_rate = sampling_rate
        self.detection_threshold = detection_threshold
        self.device = torch.device(device)
        self.rng = get_global_rng()
        self._catalog_source = data_source

        self.seismic_analyzer = SeismicWaveAnalyzer().to(self.device)
        self.seismic_analyzer.eval()

        # Anti-theater guard (mirrors SchumannResonanceDetector): the
        # SeismicWaveAnalyzer ships with random weights and no labelled seismic
        # corpus exists to train it. Worse than fabricating, the untrained
        # network previously GATED the real physics: P/S-wave detection came
        # from its random p_prob/s_prob heads, deciding whether the genuine
        # STA/LTA arrival picker even ran. Until real weights are loaded via
        # load_neural_weights(), detection now runs directly on the
        # field-standard physics -- STA/LTA triggering, S-P epicenter distance,
        # band resonance -- and NO magnitude is estimated (a single uncalibrated
        # station cannot honestly produce a Richter magnitude).
        self._neural_trained = False
        self._warned_untrained = False

        self.p_wave_velocity = 6.0
        self.s_wave_velocity = 3.5

        logger.info(f"EarthquakeDetector initialized: threshold={detection_threshold}")

    def load_neural_weights(self, checkpoint_path: str) -> None:
        """Load trained weights for the seismic analyzer.

        Until this is called the network is untrained and detection runs on the
        deterministic STA/LTA + spectral physics of the observed waveform.

        Args:
            checkpoint_path: Path to a torch checkpoint containing a
                ``seismic_analyzer`` state dict.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.seismic_analyzer.load_state_dict(checkpoint["seismic_analyzer"])
        self._neural_trained = True
        logger.info(
            "Earthquake neural weights loaded from %s; using learned analyzer", checkpoint_path
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the untrained NN is bypassed for physics."""
        if not self._warned_untrained:
            logger.warning(
                "EarthquakeDetector's SeismicWaveAnalyzer is untrained (no "
                "checkpoint loaded); detecting from STA/LTA + spectral physics and "
                "emitting no magnitude estimate (estimated_magnitude=None) -- an "
                "uncalibrated single station cannot honestly produce one. Call "
                "load_neural_weights() once a trained checkpoint exists."
            )
            self._warned_untrained = True

    def predict_earthquake(
        self,
        seismic_data: np.ndarray[Any, Any] | torch.Tensor,
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

        resonance_score = self._compute_resonance_score(Sxx, f)

        # The STA/LTA arrival picker is the field-standard trigger; run it
        # unconditionally on both paths (previously the untrained network's
        # random p_prob/s_prob heads gated whether it ran at all). The S pick
        # searches after the P trigger plus one second -- on a single trace the
        # lower S threshold would otherwise fire at-or-before the P pick.
        p_arrival = self._detect_wave_arrival(seismic_data[0], "p")
        s_min_index = (p_arrival + int(self.sampling_rate)) if p_arrival is not None else 0
        s_arrival = self._detect_wave_arrival(seismic_data[0], "s", min_index=s_min_index)

        estimated_mag: float | None
        if self._neural_trained:
            spectrogram_tensor = torch.from_numpy(Sxx_norm).float().unsqueeze(0).unsqueeze(0)
            spectrogram_tensor = spectrogram_tensor.to(self.device)

            with torch.no_grad():
                eq_prob, magnitude, p_prob, s_prob = self.seismic_analyzer(spectrogram_tensor)

            confidence = float(eq_prob[0].item())
            estimated_mag = float(magnitude[0].item()) * 4 + 2
            p_wave_detected = float(p_prob[0].item()) > 0.5
            s_wave_detected = float(s_prob[0].item()) > 0.5
            confidence = min(1.0, confidence + resonance_score * 0.2)
            magnitude_class = self._classify_magnitude(estimated_mag)
            aftershock_probability = min(0.9, estimated_mag / 10)
        else:
            # Physics path: detection strength is the peak STA/LTA trigger ratio
            # blended with the seismic-band resonance; P/S detection is the
            # picker itself. No magnitude is fabricated -- an uncalibrated
            # single station has no honest Richter estimate, so
            # estimated_magnitude stays None ("undetermined").
            self._warn_untrained_once()
            p_wave_detected = p_arrival is not None
            s_wave_detected = s_arrival is not None and (p_arrival is None or s_arrival > p_arrival)
            peak_ratio = self._peak_sta_lta(seismic_data[0])
            trigger_severity = float(np.clip((peak_ratio - 2.5) / 7.5, 0.0, 1.0))
            confidence = min(1.0, trigger_severity + resonance_score * 0.2)
            estimated_mag = None
            magnitude_class = "undetermined"
            aftershock_probability = 0.0

        earthquake_detected = confidence > self.detection_threshold

        epicenter_distance = None
        if p_arrival is not None and s_arrival is not None and s_arrival > p_arrival:
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
            p_wave_arrival_time=(p_arrival / self.sampling_rate if p_arrival is not None else None),
            s_wave_arrival_time=(s_arrival / self.sampling_rate if s_arrival is not None else None),
            epicenter_distance_km=epicenter_distance,
            resonance_score=resonance_score,
            spectral_anomalies=self._find_spectral_anomalies(Sxx, f),
            warning_actions=warnings,
            aftershock_probability=aftershock_probability,
        )

    def fetch_live_data(self, *, allow_simulated: bool = False, **kwargs: Any) -> LiveFetch:
        """Fetch live USGS earthquake-catalog events through the injected client.

        Args:
            allow_simulated: Explicit opt-in for simulated sources (the USGS
                catalog is a real feed, so this normally stays False).
            **kwargs: Passed to the client fetch (e.g. ``min_magnitude=``,
                geographic bounds).

        Returns:
            Provenance-checked LiveFetch.

        Raises:
            LiveDataError: No catalog client injected, or the fetch failed.
        """
        client = require_live_client(self._catalog_source, "EarthquakeDetector", "USGS earthquake")
        return fetch_live_datapoints(client, allow_simulated=allow_simulated, **kwargs)

    def detect_live(
        self,
        *,
        min_magnitude: float | None = None,
        station_lat: float | None = None,
        station_lon: float | None = None,
        allow_simulated: bool = False,
        **fetch_kwargs: Any,
    ) -> EarthquakePredictionResult:
        """Assess live seismicity from the USGS event catalog.

        This is an EVENT-STREAM assessment, feeding what the physics actually
        consumes from a catalog: observed magnitudes, occurrence rate and
        space-time clustering. Catalog metadata is never converted into
        synthetic waveforms, so the waveform fields (P/S picks, resonance,
        spectral anomalies) are absent/zero here and ``estimated_magnitude``
        is the LARGEST OBSERVED catalog magnitude -- a real USGS measurement,
        not a model estimate (the untrained-network magnitude fabrication was
        removed in the honesty wave). ``aftershock_probability`` stays 0.0:
        no calibrated Reasenberg-Jones parameters are available, so no
        forecast is fabricated. Rate/clustering features (events/day,
        maximum-likelihood b-value per Aki 1965, clustered-event fraction)
        are reported in ``live_context``.

        Args:
            min_magnitude: Optional catalog magnitude floor.
            station_lat: Optional station latitude for epicentral distance.
            station_lon: Optional station longitude for epicentral distance.
            allow_simulated: Explicit opt-in for simulated sources.
            **fetch_kwargs: Extra client fetch parameters (e.g. bounds).

        Returns:
            EarthquakePredictionResult with ``source_id`` /
            ``data_provenance`` / ``live_context`` populated.

        Raises:
            LiveDataError: No catalog client injected, or the fetch failed.
        """
        if min_magnitude is not None:
            fetch_kwargs["min_magnitude"] = min_magnitude
        fetch = self.fetch_live_data(allow_simulated=allow_simulated, **fetch_kwargs)

        events = [dp for dp in fetch.data_points if dp.data.get("magnitude") is not None]
        events.sort(key=lambda dp: dp.timestamp)

        if not events:
            return EarthquakePredictionResult(
                earthquake_detected=False,
                confidence=0.0,
                estimated_magnitude=None,
                magnitude_class="undetermined",
                source_id=fetch.source_id,
                data_provenance=fetch.data_provenance,
                live_context={"event_count": 0},
            )

        magnitudes = np.array([float(dp.data["magnitude"]) for dp in events])
        strongest = events[int(np.argmax(magnitudes))]
        max_magnitude = float(magnitudes.max())

        # Occurrence rate over the observed window (>= 1 h to avoid a
        # single-event zero-span blowup).
        span_days = max(
            (events[-1].timestamp - events[0].timestamp).total_seconds() / 86400.0,
            1.0 / 24.0,
        )
        events_per_day = len(events) / span_days

        # Maximum-likelihood b-value (Aki 1965): b = log10(e) / (mean(M) -
        # (Mc - dM/2)) with completeness Mc taken as the smallest observed
        # magnitude and 0.1-unit binning. Only meaningful with enough events.
        b_value: float | None = None
        if len(magnitudes) >= 10:
            mc = float(magnitudes.min())
            mean_excess = float(magnitudes.mean()) - (mc - 0.05)
            if mean_excess > 0:
                b_value = float(np.log10(np.e) / mean_excess)

        # Clustered fraction: events with a preceding event within 100 km and
        # 72 h (a deterministic space-time clustering measure, not a forecast).
        clustered = 0
        for i, dp in enumerate(events):
            if dp.location is None:
                continue
            for prior in events[:i]:
                if prior.location is None:
                    continue
                dt_hours = (dp.timestamp - prior.timestamp).total_seconds() / 3600.0
                if dt_hours > 72.0:
                    continue
                dist = haversine_km(
                    dp.location[0], dp.location[1], prior.location[0], prior.location[1]
                )
                if dist <= 100.0:
                    clustered += 1
                    break
        clustered_fraction = clustered / len(events)

        epicenter_distance = None
        if station_lat is not None and station_lon is not None and strongest.location is not None:
            epicenter_distance = haversine_km(
                station_lat, station_lon, strongest.location[0], strongest.location[1]
            )

        magnitude_class = self._classify_magnitude(max_magnitude)
        result = EarthquakePredictionResult(
            earthquake_detected=True,
            confidence=float(strongest.confidence),
            estimated_magnitude=max_magnitude,  # observed catalog magnitude
            magnitude_class=magnitude_class,
            epicenter_distance_km=epicenter_distance,
            depth_km=(strongest.location[2] if strongest.location is not None else None),
            warning_actions=self._generate_warnings(True, magnitude_class),
            source_id=fetch.source_id,
            data_provenance=fetch.data_provenance,
            live_context={
                "event_count": len(events),
                "events_per_day": events_per_day,
                "window_days": span_days,
                "max_magnitude": max_magnitude,
                "mean_magnitude": float(magnitudes.mean()),
                "b_value": b_value,
                "clustered_fraction": clustered_fraction,
                "strongest_event": {
                    "event_id": strongest.event_id,
                    "place": strongest.data.get("place"),
                    "magnitude": max_magnitude,
                    "time": strongest.timestamp.isoformat(),
                    "tsunami_flagged": bool(strongest.data.get("tsunami")),
                },
            },
        )
        return result

    def _peak_sta_lta(self, data: np.ndarray[Any, Any]) -> float:
        """Peak STA/LTA trigger ratio over the record (0.0 if too short)."""
        sta_len = int(0.5 * self.sampling_rate)
        lta_len = int(5.0 * self.sampling_rate)
        if len(data) < lta_len + sta_len:
            return 0.0
        peak = 0.0
        for i in range(lta_len, len(data) - sta_len):
            sta = np.mean(np.abs(data[i : i + sta_len]))
            lta = np.mean(np.abs(data[i - lta_len : i]))
            peak = max(peak, float(sta / (lta + 1e-10)))
        return peak

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

    def _detect_wave_arrival(
        self, data: np.ndarray[Any, Any], wave_type: str, min_index: int = 0
    ) -> int | None:
        """Detect P or S wave arrival time using STA/LTA.

        Args:
            data: The seismic trace.
            wave_type: ``"p"`` (threshold 3.0) or ``"s"`` (threshold 2.0).
            min_index: Ignore triggers before this sample. Required for a
                meaningful S pick on a single trace: the lower S threshold
                otherwise always fires at-or-before the P trigger, so the S
                search must start after the P arrival.
        """
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
        arrivals = np.where(sta_lta[min_index:] > threshold)[0]

        return int(arrivals[0]) + min_index if len(arrivals) > 0 else None

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
        features[9] = (result.estimated_magnitude or 0.0) / 10
        features[10] = result.resonance_score

        return features


class MeteorDetector:
    """Meteor detector using optical/radar Bayesian filter with NASA/JPL integration.

    Combines optical and radar observations with Bayesian inference
    for meteor/asteroid detection and trajectory estimation.

    Production Features:
        - JPLFireballSource (data_sources) for real atmospheric impact data
        - NASANeoWsSource (data_sources) for near-Earth object close approaches
        - JPLSentrySource (data_sources) for potential future impact risks
        - Bayesian sensor fusion for optical/radar observations

    Live-ingestion pattern (uniform across hazard detectors): the constructor
    accepts optional data_sources client instances (dependency injection).
    ``use_nasa_data=True`` (the default, historical behaviour) constructs the
    default clients when none are injected; ``use_nasa_data=False`` with no
    injected clients means fully offline. The former private module-level HTTP
    loaders are gone -- ALL network access flows through the clients, whose
    own ``CacheConfig`` (6 h TTL) preserves the historical refresh cadence.
    :meth:`get_recent_fireballs` / :meth:`get_upcoming_close_approaches` /
    :meth:`get_impact_risks` fail loud (:class:`LiveDataError`) when their
    fetch fails; :meth:`predict_meteor` treats NASA/JPL data as optional
    corroborating evidence, logs fetch failures and stamps ``source_id`` /
    ``data_provenance`` / ``live_context`` on the result when live data was
    actually consulted.
    """

    def __init__(
        self,
        detection_threshold: float = 0.7,
        prior_probability: float = 1e-6,
        use_nasa_data: bool = True,
        fireball_source: JPLFireballSource | None = None,
        neo_source: NASANeoWsSource | None = None,
        sentry_source: JPLSentrySource | None = None,
    ):
        """Initialize MeteorDetector.

        Args:
            detection_threshold: Confidence threshold for meteor detection (0-1)
            prior_probability: Prior probability of meteor occurrence for Bayesian filter
            use_nasa_data: Construct default NASA/JPL clients when none are
                injected (historical knob; False + no clients = offline)
            fireball_source: Optional injected JPL Fireball client
            neo_source: Optional injected NASA NeoWs close-approach client
            sentry_source: Optional injected JPL Sentry impact-risk client
        """
        self.detection_threshold = detection_threshold
        self.bayesian_filter = BayesianMeteorFilter(prior_probability=prior_probability)
        self.rng = get_global_rng()

        any_injected = any(s is not None for s in (fireball_source, neo_source, sentry_source))
        self.use_nasa_data = use_nasa_data or any_injected

        self._fireball_source: JPLFireballSource | None
        self._neo_source: NASANeoWsSource | None
        self._sentry_source: JPLSentrySource | None
        if self.use_nasa_data:
            self._fireball_source = fireball_source or JPLFireballSource(days_back=30)
            self._neo_source = neo_source or NASANeoWsSource(days_forward=7)
            self._sentry_source = sentry_source or JPLSentrySource()
        else:
            self._fireball_source = None
            self._neo_source = None
            self._sentry_source = None

        logger.info(
            f"MeteorDetector initialized: threshold={detection_threshold}, "
            f"nasa_data={self.use_nasa_data}"
        )

    def fetch_live_data(
        self, client_name: str = "fireball", *, allow_simulated: bool = False, **kwargs: Any
    ) -> LiveFetch:
        """Fetch live data points through one of the injected clients.

        Args:
            client_name: One of ``"fireball"``, ``"neo"``, ``"sentry"``.
            allow_simulated: Explicit opt-in for simulated sources (all three
                NASA/JPL clients are real feeds, so this normally stays False).
            **kwargs: Passed through to the client fetch.

        Returns:
            Provenance-checked LiveFetch.

        Raises:
            LiveDataError: Unknown/unwired client, or the fetch failed.
        """
        clients = {
            "fireball": self._fireball_source,
            "neo": self._neo_source,
            "sentry": self._sentry_source,
        }
        if client_name not in clients:
            raise LiveDataError(
                f"MeteorDetector: unknown live client {client_name!r} "
                f"(expected one of {sorted(clients)})"
            )
        client = require_live_client(
            clients[client_name], "MeteorDetector", f"NASA/JPL {client_name}"
        )
        return fetch_live_datapoints(client, allow_simulated=allow_simulated, **kwargs)

    def get_recent_fireballs(self, days: int = 7) -> list[FireballEvent]:
        """Get recent fireball events from the JPL Fireball API.

        Args:
            days: Number of days back to look

        Returns:
            List of FireballEvent objects from the last N days

        Raises:
            LiveDataError: The fireball client is unwired or its fetch failed
                (fail-loud: no silent empty list on network failure).
        """
        if not self.use_nasa_data:
            return []

        fetch = self.fetch_live_data("fireball", start_time=datetime.now(UTC) - timedelta(days=30))
        events = fireball_events_from_datapoints(fetch.data_points)
        cutoff = datetime.now(UTC) - timedelta(days=days)
        return [fb for fb in events if fb.date >= cutoff]

    def get_upcoming_close_approaches(self) -> list[CloseApproachEvent]:
        """Get upcoming near-Earth object close approaches via NASA NeoWs.

        Returns:
            List of CloseApproachEvent objects sorted by approach date.

        Raises:
            LiveDataError: The NeoWs client is unwired or its fetch failed.
        """
        if not self.use_nasa_data:
            return []

        fetch = self.fetch_live_data("neo")
        return close_approaches_from_neows_datapoints(fetch.data_points)

    def get_impact_risks(self) -> list[SentryImpactRisk]:
        """Get current impact risk assessments from JPL Sentry.

        Returns:
            List of SentryImpactRisk objects sorted by Palermo scale
            (higher = more concerning).

        Raises:
            LiveDataError: The Sentry client is unwired or its fetch failed.
        """
        if not self.use_nasa_data:
            return []

        fetch = self.fetch_live_data("sentry")
        risks = sentry_risks_from_datapoints(fetch.data_points)
        return sorted(risks, key=lambda x: x.palermo_scale, reverse=True)

    def predict_meteor(
        self,
        optical_data: np.ndarray[Any, Any] | None = None,
        radar_data: np.ndarray[Any, Any] | None = None,
        noaa_stub: dict[str, Any] | None = None,
    ) -> MeteorPredictionResult:
        """Predict meteor from optical and radar data with NASA CNEOS integration.

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

        # Check NASA/JPL data_sources clients for recent significant events.
        # Live data here is optional corroborating evidence for the local
        # sensor fusion: a failed fetch is logged (never silently faked) and
        # the detector proceeds sensor-only with data_provenance left None.
        nasa_fireball_alert = False
        nasa_close_approach_alert = False
        nasa_size_estimate = None
        nasa_velocity_estimate = None
        nasa_impact_probability = 0.0
        live_sources_used: list[str] = []
        live_context: dict[str, Any] = {}

        if self.use_nasa_data:
            # Recent fireballs (last 24 hours with significant energy).
            try:
                fireballs = self.get_recent_fireballs(days=1)
                if self._fireball_source is not None:
                    live_sources_used.append(self._fireball_source.source_id)
                significant = [
                    fb
                    for fb in fireballs
                    if fb.calculated_total_impact_energy_kt is not None
                    and fb.calculated_total_impact_energy_kt > 0.1  # > 100 tons TNT
                ]
                live_context["recent_fireballs_24h"] = len(fireballs)
                if significant:
                    nasa_fireball_alert = True
                    # Use the most energetic recent fireball for estimates
                    biggest = max(
                        significant,
                        key=lambda x: x.calculated_total_impact_energy_kt or 0,
                    )
                    nasa_size_estimate = biggest.estimated_size_m
                    nasa_velocity_estimate = biggest.velocity_km_s
            except LiveDataError as e:
                logger.warning(f"MeteorDetector: fireball feed unavailable: {e}")
                live_context["fireball_error"] = str(e)

            # Imminent close approaches (within 1 lunar distance, next 7 days).
            try:
                approaches = self.get_upcoming_close_approaches()
                if self._neo_source is not None:
                    live_sources_used.append(self._neo_source.source_id)
                lunar_distance_km = 384400
                imminent = [
                    ca
                    for ca in approaches
                    if ca.nominal_distance_km < lunar_distance_km
                    and ca.close_approach_date <= datetime.now(UTC) + timedelta(days=7)
                ]
                live_context["upcoming_close_approaches"] = len(approaches)
                live_context["imminent_close_approaches"] = len(imminent)
                if imminent:
                    nasa_close_approach_alert = True
                    # Estimate impact probability from closest approach
                    closest = min(imminent, key=lambda x: x.nominal_distance_km)
                    # Very rough heuristic: closer = higher concern
                    nasa_impact_probability = (
                        max(0, 1 - closest.nominal_distance_km / lunar_distance_km) * 0.001
                    )
            except LiveDataError as e:
                logger.warning(f"MeteorDetector: close-approach feed unavailable: {e}")
                live_context["close_approach_error"] = str(e)

            # Sentry elevated impact risks.
            try:
                risks = self.get_impact_risks()
                if self._sentry_source is not None:
                    live_sources_used.append(self._sentry_source.source_id)
                high_risk = [s for s in risks if s.palermo_scale > -3]
                live_context["sentry_high_risk_objects"] = len(high_risk)
                if high_risk:
                    # Never understate a published cumulative impact probability:
                    # the highest-Palermo object (energy/time weighted) is not
                    # necessarily the one with the largest raw probability, so
                    # floor at the max probability across all elevated objects.
                    nasa_impact_probability = max(
                        nasa_impact_probability,
                        *(s.impact_probability for s in high_risk),
                    )
            except LiveDataError as e:
                logger.warning(f"MeteorDetector: Sentry feed unavailable: {e}")
                live_context["sentry_error"] = str(e)

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
            source_id=",".join(live_sources_used) if live_sources_used else None,
            data_provenance="live" if live_sources_used else None,
            live_context=live_context or None,
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


# =============================================================================
# Synthetic Data Generation and Training for Disaster Neural Networks
# =============================================================================


def generate_synthetic_tsunami_data(
    n_samples: int = 1000,
    seq_len: int = 256,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Generate synthetic tsunami waveform data for training.

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
    """Generate synthetic earthquake spectrogram data for training.

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


def load_dart_buoy_data(
    station_id: str = "46419",
    days_back: int = 30,
    seq_len: int = 256,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Load real tsunami waveform data from NOAA DART buoy network.

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
        raw_data = SafeHTTPClient.get_text(
            f"{DART_BUOY_API_URL}/{station_id}.dart",
            headers={"User-Agent": "Mercury-Agent/1.0"},
            timeout=30,
        )

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
    """Load historical tsunami event records from NOAA NGDC.

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
        data = SafeHTTPClient.get_json(
            NOAA_TSUNAMI_API_URL,
            params={"minYear": str(min_year), "maxSize": str(max_records)},
            headers={"User-Agent": "Mercury-Agent/1.0"},
            timeout=30,
        )

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
    """Load earthquake data from USGS Earthquake Catalog API.

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

        data = SafeHTTPClient.get_json(
            USGS_EARTHQUAKE_API_URL,
            params=params,
            headers={"User-Agent": "Mercury-Agent/1.0"},
            timeout=30,
        )

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


def train_waveform_analyzer(
    model: WaveformFFTAnalyzer,
    n_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    n_samples: int = 1000,
    device: str = "cpu",
    use_real_data: bool = True,
    allow_synthetic_fallback: bool = False,
) -> dict[str, list[float]]:
    """Train WaveformFFTAnalyzer on tsunami data.

    Attempts to load real-world data from the NOAA DART buoy network first.
    When real data is unavailable this FAILS LOUD by default: silently
    training on synthetic waveforms produces weights indistinguishable from
    real-trained ones downstream. Synthetic training requires the explicit
    ``allow_synthetic_fallback=True`` opt-in (demo/experiment use only —
    never ship such weights; the merit-gated pipeline in
    ``ml/hazard_training`` is the only shipping path).

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
        allow_synthetic_fallback: Explicit opt-in to train on synthetic
            waveforms when real data is unavailable (default False: raise).

    Returns:
        Training history with loss and accuracy per epoch

    Raises:
        RuntimeError: Real data unavailable and synthetic fallback not
            explicitly allowed.
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
        if not allow_synthetic_fallback:
            raise RuntimeError(
                "real DART buoy tsunami data is unavailable and synthetic "
                "fallback was not explicitly allowed. Pass "
                "allow_synthetic_fallback=True ONLY for demo/experiment "
                "training; synthetic-trained weights must never ship."
            )
        logger.warning(
            f"SYNTHETIC-FALLBACK OPT-IN: training on {n_samples} synthetic "
            "tsunami samples; the resulting weights are demo-grade and must "
            "never ship."
        )
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
    allow_synthetic_fallback: bool = False,
) -> dict[str, list[float]]:
    """Train SeismicWaveAnalyzer on earthquake data.

    Attempts to load real-world data from the USGS Earthquake Catalog first.
    When real data is unavailable this FAILS LOUD by default; synthetic
    training requires the explicit ``allow_synthetic_fallback=True`` opt-in
    (demo/experiment use only — synthetic-trained weights must never ship;
    the merit-gated pipeline in ``ml/hazard_training`` is the only shipping
    path).

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
        allow_synthetic_fallback: Explicit opt-in to train on synthetic
            spectrograms when real data is unavailable (default False: raise).

    Returns:
        Training history with loss and accuracy per epoch

    Raises:
        RuntimeError: Real data unavailable and synthetic fallback not
            explicitly allowed.
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
        if not allow_synthetic_fallback:
            raise RuntimeError(
                "real USGS earthquake-catalog data is unavailable and "
                "synthetic fallback was not explicitly allowed. Pass "
                "allow_synthetic_fallback=True ONLY for demo/experiment "
                "training; synthetic-trained weights must never ship."
            )
        logger.warning(
            f"SYNTHETIC-FALLBACK OPT-IN: training on {n_samples} synthetic "
            "earthquake spectrograms; the resulting weights are demo-grade "
            "and must never ship."
        )
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
    """Train all disaster detection neural networks (DEMO; synthetic fallback).

    This demo helper initializes and trains:
    - WaveformFFTAnalyzer for tsunami detection
    - SeismicWaveAnalyzer for earthquake detection

    It passes ``allow_synthetic_fallback=True`` explicitly: when the real
    data sources are unreachable the models train on synthetic samples and
    the resulting weights are demo-grade — they must never ship. The
    merit-gated pipeline in ``ml/hazard_training`` is the only shipping path.

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
        waveform_model, n_epochs=n_epochs, device=device, allow_synthetic_fallback=True
    )

    # Train SeismicWaveAnalyzer
    seismic_model = SeismicWaveAnalyzer()
    results["SeismicWaveAnalyzer"] = train_seismic_analyzer(
        seismic_model, n_epochs=n_epochs, device=device, allow_synthetic_fallback=True
    )

    logger.info("All disaster detection networks trained successfully.")

    return results
