# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.detectors._hazard_input import (
    InputGrader,
    InvalidObservationError,
    finite_scalar,
)
from omni_mercury_engine.detectors.geological.sea_level import (
    detide,
    quietest_segment_sigma,
    remove_isolated_spikes,
)
from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.security.safe_torch import safe_torch_load

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
from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics
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
    "CloseApproachEvent",
    "EarthquakeDetector",
    "EarthquakeMagnitude",
    "EarthquakePredictionResult",
    "FireballEvent",
    "MeteorDetector",
    "MeteorPredictionResult",
    "MeteorThreatLevel",
    "SeismicWaveAnalyzer",
    "SentryImpactRisk",
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
    """Observed-height category of a detected tsunami at the recording instrument.

    Heights follow the NWS observed-height criteria (NWSI 10-701 §3.4,
    https://www.weather.gov/media/directives/010_pdfs/pd01007001curr.pdf):
    Warning above 1 m, Advisory 0.5-1 m; PTWC's top threat band (> 3 m) is
    MAJOR, and a detected wave below 0.5 m is INFORMATION ("minor waves at
    most"). A Watch has no height criterion (an event that may later impact
    the area), so no height maps to WATCH. These describe the wave at this
    gauge: a deep-ocean DART amplitude understates the coastal height, which
    warning centres forecast with propagation models.
    """

    NONE = "none"
    INFORMATION = "information"  # detected, < 0.5 m
    ADVISORY = "advisory"  # 0.5-1 m
    WATCH = "watch"  # no height criterion
    WARNING = "warning"  # 1-3 m
    MAJOR = "major"  # > 3 m


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
    # Peak detided residual in noise units (|residual| max / noise scale). It
    # was the standard deviation of the raw power spectrum: unbounded and in
    # the square of whatever unit the record used.
    waveform_anomaly_score: float = 0.0

    # Diagnostic-only NN wave-height estimate (populated only when a trained
    # checkpoint is loaded). ``estimated_wave_height_m`` is ALWAYS the
    # measured peak sea-level deviation of the record itself -- a measured
    # quantity is never replaced by a model estimate (the trained head's MAE
    # on held-out DART arrivals is ~4x the measurement's).
    nn_wave_height_m: float | None = None

    warning_actions: list[str] = field(default_factory=list)
    evacuation_zones: list[str] = field(default_factory=list)

    # Live-ingestion provenance (populated only by predict_tsunami_live()).
    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None

    # Populated only when the detector was built with keep_diagnostics=True.
    diagnostics: HazardDiagnostics | None = None

    # Partial-input contract: True when part of the record was set aside;
    # ``set_aside`` names each piece and why.
    partial: bool = False
    set_aside: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EarthquakePredictionResult:
    """Earthquake prediction results."""

    earthquake_detected: bool
    confidence: float
    # None when no trained model is loaded: an uncalibrated single station has
    # no transparent Richter estimate (magnitude_class is "undetermined" then).
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
    # 0.0 means "not estimated": no calibrated aftershock model exists.
    aftershock_probability: float = 0.0

    # Live-ingestion provenance (populated only by detect_live()).
    source_id: str | None = None
    data_provenance: str | None = None
    live_context: dict[str, Any] | None = None

    # Populated only when the detector was built with keep_diagnostics=True.
    diagnostics: HazardDiagnostics | None = None

    # Partial-input contract: True when part of the record was set aside;
    # ``set_aside`` names each piece and why.
    partial: bool = False
    set_aside: list[dict[str, Any]] = field(default_factory=list)

    # Which path decided: "learned_seismic_stead", "physics_sta_lta", or
    # "usgs_catalog" (detect_live). Added after the released fields so
    # positional construction keeps their order.
    detection_method: str = "physics_sta_lta"


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

    # Populated only when the detector was built with keep_diagnostics=True.
    diagnostics: HazardDiagnostics | None = None

    # Partial-input contract: True when part of the record was set aside;
    # ``set_aside`` names each piece and why.
    partial: bool = False
    set_aside: list[dict[str, Any]] = field(default_factory=list)


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

        def detection_logit(self, spectrogram: torch.Tensor) -> torch.Tensor:
            """Pre-sigmoid earthquake-detection logit (the unsaturated alert score).

            The classifier head ends in a ``Sigmoid`` that pins confident traces
            to *exactly* 1.0 in float32, collapsing the score tail a deployable
            alert threshold needs (empirically ~79% of true events saturate to
            1.0). This returns the logit feeding that final ``Sigmoid`` -- strictly
            monotonic in the probability (identical ROC / AUC) but unbounded and
            tie-free, so the deployed rule can threshold a temperature-calibrated
            ``sigmoid(logit / T)`` without the saturation floor. Weights are shared
            with :meth:`forward` (``classifier`` minus its terminal activation).
            """
            if spectrogram.dim() == 3:
                spectrogram = spectrogram.unsqueeze(1)
            features = self.conv2d(spectrogram)
            features_flat = features.view(features.size(0), -1)
            logit = self.classifier[:-1](features_flat)
            return logit.squeeze(-1)

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


#: DART bottom-pressure trigger level in the North Pacific (30 mm; Mofjeld,
#: https://nctr.pmel.noaa.gov/tda_documentation.html): the physics path does
#: not call a tsunami on a smaller residual.
DART_TRIGGER_M = 0.03

#: Largest tidal range a record in metres can carry (the largest on Earth,
#: Bay of Fundy, is about 16 m).
MAX_TIDE_RANGE_M = 20.0

#: Largest detided residual a record in metres can carry.
MAX_RESIDUAL_M = 30.0


class TsunamiDetector:
    """Tsunami detector using oceanic waveform FFT analysis.

    Integrates with 3R Resonance mechanism for frequency-domain anomaly detection in oceanic sensor
    data.

    A checkpoint for the :class:`WaveformFFTAnalyzer` trains as
    ``tsunami_dart`` (see
    :mod:`omni_mercury_engine.ml.hazard_training.tsunami_waveform`) on real
    NOAA NDBC DART bottom-pressure records labeled with NCEI HazEL tsunami
    arrivals, and ships only if it passes the merit gate (the 2026-07
    candidate won held-out AUC decisively but regressed the deployed-rule
    false-alarm rate, so it was refused and the physics path stays in
    charge -- see the candidate's evaluation record). Its input contract is
    a detided 96-sample 15-minute window (24 h of water-column-height
    residual, metres; ``sampling_rate=1/900`` Hz);
    :meth:`load_neural_weights` with no argument loads the shipped default
    with provenance logging once one exists. Until weights are loaded the
    deterministic amplitude + resonance physics path decides.

    Trained checkpoints carry a ratified ``operating_point`` (the
    validation-selected confidence threshold governing the learned path's
    ``tsunami_detected`` decision -- part of the deployed rule the merit
    gate evaluated). ``estimated_wave_height_m`` is always the record's own
    measured peak deviation on both paths; the NN wave-height head is
    surfaced only as the diagnostic ``nn_wave_height_m``.
    """

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "The physics path cannot call a tsunami unless the detided "
                "water-column residual reaches the NOAA DART bottom-pressure "
                "trigger level (DART_TRIGGER_M = 0.03 m), the North Pacific "
                "amplitude at which the operational DART algorithm triggers."
            ),
            citation=(
                "Mofjeld, NOAA PMEL/NCTR Tsunami Detection Algorithm "
                "documentation, nctr.pmel.noaa.gov/tda_documentation.html "
                "(30 mm North Pacific trigger level)"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "Above that gate the detection confidence is a noisy-OR of an "
                "amplitude severity mapping (robust-z 5 maps to 0, z 20 "
                "saturates) with a tsunami-band FFT resonance score (0.25 per "
                "band with >2x local power), thresholded at the constructor's "
                "detection_threshold (default 0.96); these constants are "
                "in-code monitoring defaults with no published source and no "
                "labelled calibration. No trained leg ships: the 2026-07 "
                "tsunami_dart checkpoint candidate was refused by the merit "
                "gate, so the physics path stays in charge."
            ),
        ),
    )

    def __init__(
        self,
        sampling_rate: float = 1.0 / 900.0,
        detection_threshold: float = 0.96,
        device: str = "cpu",
        data_source: USGSEarthquakeSource | None = None,
        keep_diagnostics: bool = False,
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
            sampling_rate: Waveform sampling rate (Hz). The default is the
                DART standard-mode rate, one sample per 15 min, which is also
                the trained checkpoint's contract; pass the record's own rate
                (NOS tide gauges: 1/360 or 1/60 Hz). The former 1 Hz default
                matched no sea-level product.
            detection_threshold: Confidence threshold for detection.
            device: Torch device for the (optional) neural analyzer.
            data_source: Optional USGS earthquake-catalog client.
            keep_diagnostics: When True, each prediction result carries the FFT
                power spectrum the resonance scan computed (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Default False keeps memory behavior unchanged.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("TsunamiDetector requires PyTorch. Install with: pip install torch")
        self.sampling_rate = sampling_rate
        self.detection_threshold = detection_threshold
        self.device = torch.device(device)
        self.keep_diagnostics = keep_diagnostics
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

        # Input contract metadata of a loaded trained checkpoint (see
        # load_neural_weights); None until a checkpoint that declares it loads.
        self._feature_spec: str | None = None
        self._window_samples: int | None = None
        self._sampling_period_s: float | None = None

        # Ratified detection operating point carried by trained checkpoints
        # (see tsunami_waveform.py _select_operating_point, mirroring the
        # solar-storm dual-rule machinery): the learned path's
        # ``tsunami_detected`` decision thresholds the confidence at a
        # VALIDATION-selected tau, because the constructor default (0.96) was
        # chosen for the physics confidence scale and the NN probability
        # lives on its own scale. The loaded threshold is part of the
        # ratified deployed rule -- it ships inside the merit-gated
        # checkpoint and is selected against the same recall/false-alarm
        # constraints the ship gate enforces. None until a checkpoint that
        # declares one is loaded (then the learned path uses it; the physics
        # path always keeps ``detection_threshold``).
        self._operating_point: dict[str, float] | None = None

        if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError(f"sampling_rate must be a positive frequency, got {sampling_rate}")
        # Resonance targets inside the tsunami band (periods 60, 30, 15 and
        # 7.5 min; IOC: 5-60 min typical). Three of the former four (3.3 min,
        # 100 s, 50 s) were shorter than tsunami periods.
        self.tsunami_frequencies = [1.0 / 3600.0, 1.0 / 1800.0, 1.0 / 900.0, 1.0 / 450.0]

        logger.info(f"TsunamiDetector initialized: threshold={detection_threshold}")

    def load_neural_weights(self, checkpoint_path: str | None = None) -> None:
        """Load trained weights for the waveform analyzer.

        Until this is called the network is untrained and detection runs on the
        deterministic amplitude + resonance physics of the observed waveform.

        Args:
            checkpoint_path: Path to a torch checkpoint containing a
                ``waveform_analyzer`` state dict. ``None`` loads the shipped
                default checkpoint (``tsunami_dart``), whose provenance
                sidecar is logged; missing or corrupt files raise instead of
                degrading silently.

        Raises:
            ValueError: If the checkpoint carries an ``operating_point``
                whose detection threshold is not a probability in (0, 1) --
                a nonsensical deployed rule must refuse, not load.
        """
        if checkpoint_path is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            checkpoint, _provenance = load_shipped_checkpoint("tsunami_dart")
            source = "shipped default 'tsunami_dart'"
        else:
            checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
            source = checkpoint_path
        self.waveform_analyzer.load_state_dict(checkpoint["waveform_analyzer"])
        # Input contract carried by trained checkpoints (see
        # omni_mercury_engine.ml.hazard_training.tsunami_waveform): detided
        # fixed-length windows at a fixed sampling period. None when an
        # explicit checkpoint predates the feature-spec convention.
        self._feature_spec = (
            str(checkpoint["feature_spec"]) if "feature_spec" in checkpoint else None
        )
        self._window_samples = (
            int(checkpoint["window_samples"]) if "window_samples" in checkpoint else None
        )
        self._sampling_period_s = (
            float(checkpoint["sampling_period_s"]) if "sampling_period_s" in checkpoint else None
        )
        # Ratified operating point (validation-selected detection threshold
        # for the learned path -- part of the deployed decision rule the
        # merit gate evaluated). Explicit checkpoints that predate the
        # convention simply leave the constructor threshold in charge.
        op = checkpoint.get("operating_point")
        if op is not None:
            tau = float(op["detection_threshold"])
            if not np.isfinite(tau) or not (0.0 < tau < 1.0):
                raise ValueError(
                    f"checkpoint operating point threshold {tau} is not a "
                    "probability; refusing a nonsensical detection rule"
                )
            self._operating_point = {"detection_threshold": tau}
        else:
            self._operating_point = None
        self._neural_trained = True
        logger.info(
            "Tsunami neural weights loaded from %s (feature spec: %s%s); using learned analyzer",
            source,
            self._feature_spec or "unspecified",
            (
                f", detection operating point tau="
                f"{self._operating_point['detection_threshold']:.4f}"
                if self._operating_point is not None
                else ", no operating point (constructor threshold governs)"
            ),
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

        The record is detided first (:mod:`.sea_level`: harmonic for 12 h or
        more, cubic below, refused under 30 min), isolated single-sample
        glitches are removed, and the wave height is the peak of that
        residual. A pure tide, a constant offset and a lone spike therefore
        score as no tsunami; each read "watch" or "major" before.

        Args:
            waveform_data: One sea-level or bottom-pressure record in metres,
                ``[seq_len]`` or ``[1, seq_len]``, sampled at
                ``self.sampling_rate``.
            source_info: Optional source information (earthquake magnitude, location)

        Returns:
            TsunamiPredictionResult with detection details. ``severity`` and
            the evacuation zones are set only for a detected tsunami.

        Raises:
            InvalidObservationError: On non-finite samples, more than one
                record, a record shorter than 30 min, or values that cannot be
                metres (tidal range above 20 m or a residual above 30 m).
        """
        grader = InputGrader()
        record = grader.usable_run(waveform_data, "waveform_data", unit="m", ndim=(1, 2))
        if record.ndim == 2:
            if record.shape[0] != 1:
                raise InvalidObservationError(
                    f"predict_tsunami scores one record per call, got {record.shape[0]}; "
                    "only the first was read before"
                )
            record = record[0]
        period_s = 1.0 / self.sampling_rate

        residual, tide = detide(record, period_s)
        # The largest tides on Earth span about 16 m and no gauge has recorded
        # a tsunami residual near 30 m; values beyond either are not metres.
        tide_range = float(tide.max() - tide.min())
        if tide_range > MAX_TIDE_RANGE_M:
            raise InvalidObservationError(
                f"record's tidal range is {tide_range:.1f}, above {MAX_TIDE_RANGE_M:g} m; "
                "waveform_data must be in metres"
            )
        noise_m = quietest_segment_sigma(residual)
        cleaned, n_spikes = remove_isolated_spikes(residual, noise_m, period_s)
        if n_spikes:
            # The tide fit absorbed part of each glitch; refit without them.
            residual, tide = detide(tide + cleaned, period_s)
            noise_m = quietest_segment_sigma(residual)
        wave_height_m = float(np.max(np.abs(residual)))
        if wave_height_m > MAX_RESIDUAL_M:
            raise InvalidObservationError(
                f"detided residual peaks at {wave_height_m:.1f}, above {MAX_RESIDUAL_M:g} m; "
                "waveform_data must be in metres"
            )
        z_peak = wave_height_m / noise_m

        n = residual.size
        fft_result = fft(residual)
        freqs = fftfreq(n, period_s)
        power_spectrum = np.abs(fft_result) ** 2
        # Resonance on the residual, excluding the DC bin: on the raw record a
        # constant 4000 m column snapped every target onto DC and read
        # resonance 1.0. A target is scored only if it lies below Nyquist and
        # the record holds at least two of its cycles.
        resolution_hz = 1.0 / (n * period_s)
        nyquist_hz = 0.5 / period_s
        positive = slice(1, n // 2 + 1)
        global_power = float(power_spectrum[positive].mean()) + 1e-30
        resonance_score = 0.0
        dominant_freqs = []
        for target_freq in self.tsunami_frequencies:
            if target_freq >= nyquist_hz or target_freq < 2.0 * resolution_hz:
                continue
            idx = round(target_freq / resolution_hz)
            local_power = float(power_spectrum[max(1, idx - 2) : idx + 3].mean())
            if local_power / global_power > 2.0:
                resonance_score += 0.25
                dominant_freqs.append(float(freqs[idx]))

        # The reported wave height is a MEASUREMENT on both paths: the peak
        # detided residual is what the record itself shows. The trained
        # network's wave-height head is surfaced only as the diagnostic
        # ``nn_wave_height_m`` (its held-out MAE is ~4x the measurement's).
        nn_wave_height_m: float | None = None
        amplitude_severity = float(np.clip((z_peak - 5.0) / 15.0, 0.0, 1.0))

        if self._neural_trained:
            # The network's contract is a detided window, so it receives the
            # residual (identical to the training preprocessing for a 24 h
            # DART window; see sea_level).
            tensor = torch.from_numpy(residual.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                tsunami_prob, nn_wave_height = self.waveform_analyzer(tensor)
            confidence = float(tsunami_prob[0].item())
            confidence = min(1.0, confidence + resonance_score * 0.3)
            nn_wave_height_m = float(nn_wave_height[0].item())
        else:
            # Physics path: a noisy-OR of the amplitude severity (z 5, an
            # ordinary noise extreme, maps to 0; z 20 saturates) with the
            # tsunami-band resonance score. Deterministic.
            self._warn_untrained_once()
            confidence = 1.0 - (1.0 - amplitude_severity) * (1.0 - resonance_score)

        # Deployed decision rule: the learned path uses the checkpoint's
        # ratified operating point when it carries one (see
        # tsunami_waveform._select_operating_point). The physics path also
        # needs amplitude evidence -- resonance alone cannot detect -- and a
        # residual of at least the DART trigger level.
        if self._neural_trained and self._operating_point is not None:
            tsunami_detected = confidence > self._operating_point["detection_threshold"]
        elif self._neural_trained:
            tsunami_detected = confidence > self.detection_threshold
        else:
            tsunami_detected = (
                confidence > self.detection_threshold
                and amplitude_severity > 0.0
                and wave_height_m >= DART_TRIGGER_M
            )

        severity = (
            self._determine_severity(wave_height_m)
            if tsunami_detected
            else TsunamiSeverity.NONE.value
        )

        arrival_time = None
        if source_info and "distance_km" in source_info:
            tsunami_speed_kmh = 700
            arrival_time = source_info["distance_km"] / tsunami_speed_kmh * 60

        warnings = self._generate_warnings(tsunami_detected, severity)
        zones = self._generate_evacuation_zones(severity) if tsunami_detected else []

        diagnostics: HazardDiagnostics | None = None
        if self.keep_diagnostics:
            # Capture the spectrum arrays the resonance scan above ALREADY built
            # (two-sided FFT ordering, exactly as computed) -- no recomputation.
            diagnostics = HazardDiagnostics(
                hazard="tsunami",
                arrays={
                    "fft_freqs_hz": np.asarray(freqs, dtype=float),
                    "fft_power": np.asarray(power_spectrum, dtype=float),
                },
                context={
                    "sampling_rate_hz": float(self.sampling_rate),
                    "resonance_score": float(resonance_score),
                    "dominant_frequencies_hz": [float(f) for f in dominant_freqs],
                },
            )

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
            waveform_anomaly_score=float(z_peak),
            nn_wave_height_m=nn_wave_height_m,
            warning_actions=warnings,
            evacuation_zones=zones,
            diagnostics=diagnostics,
            partial=grader.partial,
            set_aside=list(grader.set_aside),
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
            # strongest is drawn from candidates/pool, both filtered to
            # dp.location is not None above, so the unpack is total.
            assert strongest.location is not None
            lat, lon, _depth = strongest.location
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
        """Observed-height category of a detected tsunami (see :class:`TsunamiSeverity`)."""
        if wave_height > 3.0:
            return TsunamiSeverity.MAJOR.value
        if wave_height > 1.0:
            return TsunamiSeverity.WARNING.value
        if wave_height >= 0.5:
            return TsunamiSeverity.ADVISORY.value
        return TsunamiSeverity.INFORMATION.value

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
        elif severity == TsunamiSeverity.ADVISORY.value:
            warnings.extend(
                [
                    "Strong currents likely: stay out of the water",
                    "Stay away from beaches, harbors and waterways",
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
        # The same gap-free run predict_tsunami grades: a NaN sample used to
        # make every summary feature NaN.
        waveform_data = InputGrader().usable_run(waveform_data, "waveform_data", unit="m")

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


#: STA/LTA trigger band (Hz). STA/LTA runs on filtered traces: the filter
#: removes the DC offset, which otherwise makes |x| meaningless, and the
#: microseism (Trnkoczy, NMSOP IS 8.1). A +1000-count offset hid a clear event
#: from the unfiltered ratio.
STA_LTA_BAND_HZ = (1.0, 45.0)

#: STA and LTA window lengths (s) of the detector's trigger.
STA_WINDOW_S = 0.5
LTA_WINDOW_S = 5.0


def local_magnitude(amplitude_nm: float, hypocentral_distance_km: float) -> float:
    """IASPEI standard local magnitude.

    ``ML = log10(A) + 1.11 log10(R) + 0.00189 R - 2.09`` (IASPEI Working Group
    on Magnitudes, 2013; Hutton & Boore 1987 with the measured Wood-Anderson
    gain of 2080), for crustal events with southern-California-like
    attenuation; other regions need their own calibration.

    Args:
        amplitude_nm: Maximum horizontal trace amplitude, nm, on a record
            filtered to the Wood-Anderson response with static magnification 1.
        hypocentral_distance_km: Hypocentral distance, km (at most 1000).

    Returns:
        ML.

    Raises:
        InvalidObservationError: On a non-positive amplitude or a distance
            outside (0, 1000] km.
    """
    amplitude = finite_scalar(amplitude_nm, "wood_anderson_amplitude_nm", unit="nm", lo=0.0)
    distance = finite_scalar(
        hypocentral_distance_km, "hypocentral_distance_km", unit="km", lo=0.0, hi=1000.0
    )
    if amplitude == 0.0 or distance == 0.0:
        raise InvalidObservationError("ML needs a positive amplitude and hypocentral distance")
    return math.log10(amplitude) + 1.11 * math.log10(distance) + 0.00189 * distance - 2.09


#: Reasenberg & Jones (1989) generic California aftershock sequence: rate of
#: M >= M_min aftershocks 10^(a + b (M_main - M_min)) (t + c)^-p per day, t
#: and c in days (USGS Open-File Report 96-266). Regional: other regions need
#: their own parameters (Page et al. 2016), so a detector uses them only when
#: they are passed as ``aftershock_parameters``.
REASENBERG_JONES_GENERIC_CALIFORNIA: dict[str, float] = {
    "a": -1.67,
    "b": 0.91,
    "p": 1.08,
    "c": 0.05,
}

#: Aftershock forecast reported by the detector: at least one M >= 5 within 7 days.
AFTERSHOCK_MIN_MAGNITUDE = 5.0
AFTERSHOCK_WINDOW_DAYS = 7.0


def reasenberg_jones_probability(
    mainshock_magnitude: float,
    parameters: dict[str, float],
    *,
    min_magnitude: float = AFTERSHOCK_MIN_MAGNITUDE,
    start_days: float = 0.0,
    duration_days: float = AFTERSHOCK_WINDOW_DAYS,
) -> float:
    """Probability of at least one aftershock of ``min_magnitude`` or more in a window.

    The expected number is the Reasenberg-Jones rate integrated over
    ``[start_days, start_days + duration_days]`` after the mainshock, and
    the probability is ``1 - exp(-N)`` (Poisson).

    Args:
        mainshock_magnitude: Mainshock magnitude.
        parameters: ``a``, ``b``, ``p``, ``c`` (c in days).
        min_magnitude: Smallest aftershock magnitude counted.
        start_days: Window start, days after the mainshock.
        duration_days: Window length, days.

    Returns:
        Probability in [0, 1).
    """
    a, b, p, c = (
        finite_scalar(parameters[k], f"aftershock parameter {k}", unit="-") for k in "abpc"
    )
    t1 = finite_scalar(start_days, "start_days", unit="days", lo=0.0)
    t2 = t1 + finite_scalar(duration_days, "duration_days", unit="days", lo=0.0)
    if c <= 0.0:
        raise InvalidObservationError("aftershock parameter c must be positive (days)")
    if abs(p - 1.0) < 1e-9:
        decay = math.log((t2 + c) / (t1 + c))
    else:
        decay = ((t1 + c) ** (1.0 - p) - (t2 + c) ** (1.0 - p)) / (p - 1.0)
    expected = 10.0 ** (a + b * (mainshock_magnitude - min_magnitude)) * decay
    return 1.0 - math.exp(-expected)


def gutenberg_richter_b(
    magnitudes: np.ndarray[Any, Any], bin_width: float = 0.1, min_events: int = 50
) -> tuple[float | None, tuple[float, float] | None, float | None, int]:
    """Maximum-likelihood Gutenberg-Richter b-value above the completeness magnitude.

    Completeness ``Mc`` is the maximum-curvature estimate (the mode of the
    binned frequency-magnitude distribution, Wiemer & Wyss 2000) plus 0.2,
    the usual correction for its underestimate (Woessner & Wiemer 2005).
    ``b = log10(e) / (mean(M >= Mc) - (Mc - bin_width / 2))`` (Aki 1965), with
    the ``(N - 1) / N`` small-sample correction and a 95% interval
    ``b (1 +/- 1.96 / sqrt(N))``. Fewer than ``min_events`` above ``Mc`` gives
    no estimate (Marzocchi et al. 2020, GJI 220: N = 50 is the common floor).
    Taking the smallest observed magnitude as ``Mc`` gave b = 0.37 on a live
    100-event global feed.

    Returns:
        ``(b, ci95, Mc, n_above_mc)``; ``b``, ``ci95`` and ``Mc`` are None
        when there are too few events.
    """
    mags = np.asarray(magnitudes, dtype=np.float64)
    mags = mags[np.isfinite(mags)]
    if mags.size < min_events:
        return None, None, None, int(mags.size)
    binned = np.round(mags / bin_width) * bin_width
    values, counts = np.unique(np.round(binned, 6), return_counts=True)
    mc = float(values[int(np.argmax(counts))]) + 0.2
    above = mags[binned >= mc - 1e-9]
    n = int(above.size)
    excess = float(above.mean()) - (mc - bin_width / 2.0) if n else 0.0
    if n < min_events or excess <= 0.0:
        return None, None, mc, n
    b = float(np.log10(np.e) / excess) * (n - 1) / n
    half = 1.96 / np.sqrt(n)
    return b, (b * (1.0 - half), b * (1.0 + half)), mc, n


def condition_trace(trace: np.ndarray[Any, Any], sampling_rate: float) -> np.ndarray[Any, Any]:
    """Detrend and band-pass a seismic trace for STA/LTA triggering.

    Linear detrend (which removes the mean), then a zero-phase 4th-order
    Butterworth band-pass over :data:`STA_LTA_BAND_HZ`, its upper edge capped
    at 0.45 x the sampling rate. Below about 2.2 Hz sampling there is no band
    left and only the detrend applies. Shared with the ``seismic_stead``
    training pipeline, which mirrors the detector's physics.

    Args:
        trace: 1-D finite waveform.
        sampling_rate: Sampling rate (Hz).

    Returns:
        The conditioned trace, ``float64``.
    """
    x = signal.detrend(np.asarray(trace, dtype=np.float64), type="linear")
    low, high = STA_LTA_BAND_HZ[0], min(STA_LTA_BAND_HZ[1], 0.45 * sampling_rate)
    if high > low:
        sos = signal.butter(4, [low, high], btype="bandpass", fs=sampling_rate, output="sos")
        if x.size > 3 * (2 * sos.shape[0] + 1):
            x = signal.sosfiltfilt(sos, x)
    return np.asarray(x, dtype=np.float64)


def sta_lta_ratio(
    conditioned: np.ndarray[Any, Any], sampling_rate: float
) -> np.ndarray[Any, Any] | None:
    """STA/LTA ratio of ``|x|`` per sample (0 outside the computable span).

    ``STA`` is the mean over the next :data:`STA_WINDOW_S`, ``LTA`` the mean
    over the previous :data:`LTA_WINDOW_S`, both by cumulative sums (the former
    per-sample Python loop took 0.84 s per 60,000 samples).

    Returns:
        The ratio series, or ``None`` when the trace is shorter than one
        STA + LTA window.
    """
    sta_len = int(STA_WINDOW_S * sampling_rate)
    lta_len = int(LTA_WINDOW_S * sampling_rate)
    x = np.abs(conditioned)
    if sta_len < 1 or len(x) < lta_len + sta_len:
        return None
    csum = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(lta_len, len(x) - sta_len)
    ratio = np.zeros(len(x))
    sta = (csum[idx + sta_len] - csum[idx]) / sta_len
    lta = (csum[idx] - csum[idx - lta_len]) / lta_len
    ratio[idx] = sta / (lta + 1e-10)
    return ratio


class EarthquakeDetector:
    """Earthquake detector using P/S-wave spectrogram analysis.

    Uses scipy.signal for spectrogram computation and integrates with 3R Resonance for frequency-
    domain analysis.

    Trained checkpoints carry a ratified ``operating_point`` (the alert
    threshold selected on the training pipeline's validation year -- part of
    the deployed rule the merit gate evaluated). The learned path's
    ``earthquake_detected`` decision uses it; the physics path always keeps
    the constructor's ``detection_threshold``, and the emitted ``confidence``
    is identical either way.
    """

    decision_source = (
        DecisionSource(
            kind="trained",
            description=(
                "A default-constructed detector serves the shipped "
                "merit-gated 'seismic_stead' checkpoint (trained on real "
                "STEAD waveforms); on-contract input the alert decision "
                "thresholds the (temperature-calibrated) detection head at "
                "the checkpoint's ratified, validation-selected tau."
            ),
            operating_point=(
                "shipped checkpoint 'seismic_stead' "
                "(models/checkpoints/seismic_stead.pt + provenance sidecar): "
                "operating_point.detection_threshold and calibration "
                "temperature selected by "
                "ml.hazard_training.seismic_wave._select_operating_point"
            ),
        ),
        DecisionSource(
            kind="published",
            description=(
                "The physics fallback path picks arrivals with an STA/LTA "
                "trigger (0.5 s / 5 s windows) on a trace band-passed to "
                "1-45 Hz; ML, when a Wood-Anderson amplitude and hypocentral "
                "distance are supplied, is the IASPEI standard local "
                "magnitude, and the aftershock probability integrates the "
                "Reasenberg-Jones rate with caller-supplied regional "
                "parameters."
            ),
            citation=(
                "Trnkoczy, NMSOP IS 8.1 (STA/LTA trigger on filtered "
                "traces); IASPEI Working Group on Magnitudes (2013) and "
                "Hutton & Boore (1987) (ML = log10(A) + 1.11 log10(R) + "
                "0.00189 R - 2.09); Reasenberg & Jones (1989), USGS "
                "Open-File Report 96-266 (aftershock rate); Page et al. "
                "(2016) (regional parameters)"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The physics-path ALERT bar itself is unsourced: confidence "
                "= clip((peak STA/LTA - 2.5) / 7.5, 0, 1) + 0.2 * "
                "seismic-band resonance score, thresholded at the "
                "constructor's detection_threshold (default 0.96) -- the "
                "2.5/7.5 mapping, the 0.2 resonance bump and the 0.96 bar "
                "are in-code constants with no published source and no "
                "labelled calibration."
            ),
        ),
    )

    def __init__(
        self,
        sampling_rate: float = 100.0,
        detection_threshold: float = 0.96,
        device: str = "cpu",
        data_source: USGSEarthquakeSource | None = None,
        keep_diagnostics: bool = False,
        load_shipped_weights: bool = True,
        aftershock_parameters: dict[str, float] | None = None,
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
            keep_diagnostics: When True, each prediction result carries the
                normalized spectrogram fed to the CNN plus the STA/LTA
                arrival-detection series (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Default False keeps memory behavior unchanged.
            load_shipped_weights: Load the shipped merit-gated
                ``seismic_stead`` checkpoint at construction (default). Pass
                False for the pure STA/LTA physics configuration (the hazard
                regression guard's physics lane, and the honesty-contract
                tests). Absence of the checkpoint falls open to physics; an
                invalid checkpoint still fails loud.
            aftershock_parameters: Reasenberg-Jones ``a``, ``b``, ``p``, ``c``
                for the region monitored (e.g.
                :data:`REASENBERG_JONES_GENERIC_CALIFORNIA`). With them,
                ``aftershock_probability`` is the probability of an M >= 5
                aftershock within 7 days of a known mainshock magnitude;
                without them it stays 0.0 (not estimated), because generic
                parameters from one region miscalibrate another.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "EarthquakeDetector requires PyTorch. Install with: pip install torch"
            )
        if aftershock_parameters is not None:
            missing = set("abpc") - set(aftershock_parameters)
            if missing:
                raise ValueError(f"aftershock_parameters lacks {sorted(missing)}")
            reasenberg_jones_probability(6.0, aftershock_parameters)  # validates values
        self.aftershock_parameters = aftershock_parameters
        self.sampling_rate = sampling_rate
        self.detection_threshold = detection_threshold
        self.device = torch.device(device)
        self.keep_diagnostics = keep_diagnostics
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
        # station cannot transparently produce a Richter magnitude).
        self._neural_trained = False
        self._warned_untrained = False

        # Ratified alert operating point carried by trained checkpoints (see
        # seismic_wave.py _select_operating_point, mirroring the solar-storm
        # machinery): the learned path's ``earthquake_detected`` decision
        # thresholds the confidence at a VALIDATION-selected tau, because the
        # constructor default (0.96) was calibrated for the physics STA/LTA
        # confidence scale and the CNN's sigmoid lives on its own scale. The
        # loaded threshold is part of the deployed rule the merit gate
        # evaluated; it changes the ALERT decision only, never the emitted
        # confidence. None until a checkpoint that declares one is loaded
        # (the physics path always keeps ``detection_threshold``).
        self._operating_point: dict[str, float] | None = None
        self._contract_rate_hz: float | None = None
        self._contract_window: int | None = None
        self._warned_off_contract = False

        self.p_wave_velocity = 6.0
        self.s_wave_velocity = 3.5

        # The seismic_stead checkpoint cleared the hazard merit gate on real
        # held-out STEAD data (learned beats physics at the deployed alert
        # rule), so a default-constructed detector serves the shipped winner.
        # Absence of the checkpoint (e.g. a stripped install) falls open to
        # the disclosed STA/LTA physics; a present-but-invalid checkpoint
        # still fails loud inside load_neural_weights (sha256/operating-point
        # validation is a supply-chain control, not a fallback case).
        if load_shipped_weights:
            try:
                self.load_neural_weights()
            except FileNotFoundError:
                logger.debug(
                    "No shipped 'seismic_stead' checkpoint available; running "
                    "on STA/LTA + spectral physics."
                )

        logger.info(f"EarthquakeDetector initialized: threshold={detection_threshold}")

    def load_neural_weights(self, checkpoint_path: str | None = None) -> None:
        """Load trained weights for the seismic analyzer.

        Until this is called the network is untrained and detection runs on the
        deterministic STA/LTA + spectral physics of the observed waveform.

        Args:
            checkpoint_path: Path to a torch checkpoint containing a
                ``seismic_analyzer`` state dict. ``None`` loads the shipped
                default checkpoint (``seismic_stead``, trained on real STEAD
                waveforms -- see its provenance sidecar, which is logged at
                load time); missing or corrupt files raise instead of
                degrading silently.

        Raises:
            ValueError: If the checkpoint carries an ``operating_point``
                whose detection threshold is not a probability in (0, 1) --
                a nonsensical alert rule must refuse, not load.
        """
        if checkpoint_path is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            checkpoint, _provenance = load_shipped_checkpoint("seismic_stead")
            source = "shipped default 'seismic_stead'"
        else:
            checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
            source = checkpoint_path
        # Ratified operating point (validation-selected alert threshold for
        # the learned path -- part of the deployed decision rule the merit
        # gate evaluated). Validated BEFORE any state mutates so a bad rule
        # cannot half-load. Checkpoints that predate the convention simply
        # leave the constructor threshold in charge.
        op = checkpoint.get("operating_point")
        temperature: float | None = None
        if op is not None:
            tau = float(op["detection_threshold"])
            if not np.isfinite(tau) or not (0.0 < tau < 1.0):
                raise ValueError(
                    f"checkpoint operating point detection threshold {tau} is not a "
                    "probability; refusing a nonsensical alert rule"
                )
            if op.get("temperature") is not None:
                temperature = float(op["temperature"])
                if not np.isfinite(temperature) or temperature <= 0.0:
                    raise ValueError(
                        f"checkpoint operating point temperature {temperature} is not a "
                        "positive scale; refusing a nonsensical alert rule"
                    )
        self.seismic_analyzer.load_state_dict(checkpoint["seismic_analyzer"])
        # Input contract the checkpoint was trained and evaluated on; the
        # learned path runs only on input that matches it.
        self._contract_rate_hz = (
            float(checkpoint["sampling_rate_hz"]) if "sampling_rate_hz" in checkpoint else None
        )
        self._contract_window = (
            int(checkpoint["window_samples"]) if "window_samples" in checkpoint else None
        )
        if op is not None:
            self._operating_point = {"detection_threshold": float(op["detection_threshold"])}
            if temperature is not None:
                self._operating_point["temperature"] = temperature
        else:
            self._operating_point = None
        self._neural_trained = True
        logger.info(
            "Earthquake neural weights loaded from %s; using learned analyzer%s",
            source,
            (
                f" (alert operating point tau="
                f"{self._operating_point['detection_threshold']:.4f})"
                if self._operating_point is not None
                else " (no operating point; constructor threshold governs)"
            ),
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the untrained NN is bypassed for physics."""
        if not self._warned_untrained:
            logger.warning(
                "EarthquakeDetector's SeismicWaveAnalyzer is untrained (no "
                "checkpoint loaded); detecting from STA/LTA + spectral physics and "
                "emitting no magnitude estimate (estimated_magnitude=None) -- an "
                "uncalibrated single station cannot transparently produce one. Call "
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
            seismic_data: One seismic waveform, ``[seq_len]`` or
                ``[1, seq_len]``, sampled at ``self.sampling_rate``.
            station_info: Optional station information. With
                ``wood_anderson_amplitude_nm`` (peak horizontal amplitude on
                the Wood-Anderson response, magnification 1) and
                ``hypocentral_distance_km``, the IASPEI local magnitude is
                reported (:func:`local_magnitude`).

        Returns:
            EarthquakePredictionResult with detection details.
            ``estimated_magnitude`` comes only from that calibrated amplitude
            and distance: the learned head reads a per-trace z-normalised
            spectrogram, which carries no amplitude (on real Alaska events it
            scored worse than a constant), so the trace alone gives None.
            ``aftershock_probability`` follows from a magnitude and the
            constructor's ``aftershock_parameters``. ``detection_method``
            names the path that ran.

        Raises:
            InputModalityError: If the input is not a 1-D or 2-D waveform.
            InvalidObservationError: On non-finite samples, more than one
                trace, or a trace shorter than one STA + LTA window.
        """
        if isinstance(seismic_data, torch.Tensor):
            seismic_data = seismic_data.detach().cpu().numpy()
        seismic_data = np.asarray(seismic_data)
        if seismic_data.ndim not in (1, 2):
            # Fail loud with the documented contract instead of the opaque
            # downstream crash a 0-d scalar or a rank>=3 stack used to
            # produce: every path below needs a 1-D trace.
            raise InputModalityError(
                "predict_earthquake expects a seismic waveform shaped "
                f"[seq_len] or [batch, seq_len]; got ndim={seismic_data.ndim} "
                f"(shape {seismic_data.shape})"
            )
        grader = InputGrader()
        record = grader.usable_run(seismic_data, "seismic_data", unit="counts", ndim=(1, 2))
        if record.ndim == 2:
            if record.shape[0] != 1:
                raise InvalidObservationError(
                    f"predict_earthquake scores one trace per call, got {record.shape[0]}; "
                    "only the first was read before"
                )
            record = record[0]
        min_samples = int((STA_WINDOW_S + LTA_WINDOW_S) * self.sampling_rate)
        if record.size < min_samples:
            raise InvalidObservationError(
                f"seismic trace has {record.size} samples; one STA + LTA window "
                f"({STA_WINDOW_S + LTA_WINDOW_S:g} s at {self.sampling_rate:g} Hz) "
                f"needs {min_samples}"
            )
        seismic_data = record.reshape(1, -1)

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
        # Compute the STA/LTA ratio series ONCE and derive both arrivals from it
        # (previously it was rebuilt per wave type and then discarded); with
        # keep_diagnostics the series itself is persisted on the result.
        sta_lta = self._sta_lta_series(seismic_data[0])
        p_arrival: int | None = None
        s_arrival: int | None = None
        if sta_lta is not None:
            p_arrival = self._first_arrival(sta_lta, "p")
            s_min_index = (p_arrival + int(self.sampling_rate)) if p_arrival is not None else 0
            s_arrival = self._first_arrival(sta_lta, "s", min_index=s_min_index)

        # The learned path runs only on the input the checkpoint was trained
        # and evaluated on (rate and window length); a 40 Hz trace used to go
        # through the 100 Hz CNN with its frequency axis silently rescaled.
        on_contract = self._neural_trained and (
            (self._contract_rate_hz is None or self._contract_rate_hz == self.sampling_rate)
            and (self._contract_window is None or self._contract_window == record.size)
        )
        if self._neural_trained and not on_contract and not self._warned_off_contract:
            logger.warning(
                "EarthquakeDetector: input (%g Hz, %d samples) is off the checkpoint's "
                "contract (%s Hz, %s samples); using the STA/LTA physics path",
                self.sampling_rate,
                record.size,
                self._contract_rate_hz,
                self._contract_window,
            )
            self._warned_off_contract = True

        estimated_mag: float | None
        if on_contract:
            spectrogram_tensor = torch.from_numpy(Sxx_norm).float().unsqueeze(0).unsqueeze(0)
            spectrogram_tensor = spectrogram_tensor.to(self.device)

            with torch.no_grad():
                eq_prob, magnitude, p_prob, s_prob = self.seismic_analyzer(spectrogram_tensor)
                # Deployed alert score. When the checkpoint carries a calibration
                # temperature (seismic-stead-v2 semantics) the alert rule
                # thresholds sigmoid(logit / T): this preserves the head's ROC
                # while undoing the terminal-sigmoid saturation that pinned
                # confident traces to exactly 1.0 (a threshold cannot split a 1.0
                # pile-up). The physics-path 0.2*resonance bump is NOT added on the
                # learned head -- it is an STA/LTA-confidence device that only
                # manufactures noise-driven false alarms here. Checkpoints without
                # a temperature keep the legacy (eq_prob + 0.2*resonance, capped)
                # semantics unchanged.
                temperature = (self._operating_point or {}).get("temperature")
                if temperature is not None:
                    eq_logit = self.seismic_analyzer.detection_logit(spectrogram_tensor)
                    confidence = float(torch.sigmoid(eq_logit[0] / temperature).item())
                else:
                    confidence = min(1.0, float(eq_prob[0].item()) + resonance_score * 0.2)

            # The magnitude head reads an amplitude-free spectrogram (its output
            # was the same for a trace scaled by 1 and by 1e6), and the
            # aftershock "probability" was min(0.9, M / 10): neither is served.
            del magnitude
            estimated_mag = None
            p_wave_detected = float(p_prob[0].item()) > 0.5
            s_wave_detected = float(s_prob[0].item()) > 0.5
            magnitude_class = "undetermined"
            aftershock_probability = 0.0
            detection_method = "learned_seismic_stead"
        else:
            # Physics path: detection strength is the peak STA/LTA trigger ratio
            # blended with the seismic-band resonance; P/S detection is the
            # picker itself. No magnitude is fabricated -- an uncalibrated
            # single station has no transparent Richter estimate, so
            # estimated_magnitude stays None ("undetermined").
            if not self._neural_trained:
                self._warn_untrained_once()
            p_wave_detected = p_arrival is not None
            s_wave_detected = s_arrival is not None and (p_arrival is None or s_arrival > p_arrival)
            peak_ratio = self._peak_sta_lta(seismic_data[0])
            trigger_severity = float(np.clip((peak_ratio - 2.5) / 7.5, 0.0, 1.0))
            confidence = min(1.0, trigger_severity + resonance_score * 0.2)
            estimated_mag = None
            magnitude_class = "undetermined"
            aftershock_probability = 0.0
            detection_method = "physics_sta_lta"

        if station_info and station_info.get("wood_anderson_amplitude_nm") is not None:
            if station_info.get("hypocentral_distance_km") is None:
                raise InvalidObservationError(
                    "wood_anderson_amplitude_nm needs hypocentral_distance_km for ML"
                )
            estimated_mag = local_magnitude(
                station_info["wood_anderson_amplitude_nm"],
                station_info["hypocentral_distance_km"],
            )
            magnitude_class = self._classify_magnitude(estimated_mag)

        # Deployed alert rule: the physics path thresholds at the
        # constructor's ``detection_threshold``; the learned path uses the
        # checkpoint's ratified operating point when it carries one (the
        # validation-selected tau IS part of the deployed rule the merit gate
        # evaluated -- see seismic_wave._select_operating_point). Alert
        # decision only; the confidence estimate above is never altered.
        if on_contract and self._operating_point is not None:
            earthquake_detected = confidence > self._operating_point["detection_threshold"]
        else:
            earthquake_detected = confidence > self.detection_threshold

        if (
            earthquake_detected
            and estimated_mag is not None
            and self.aftershock_parameters is not None
        ):
            aftershock_probability = reasenberg_jones_probability(
                estimated_mag, self.aftershock_parameters
            )

        epicenter_distance = None
        if p_arrival is not None and s_arrival is not None and s_arrival > p_arrival:
            time_diff = (s_arrival - p_arrival) / self.sampling_rate
            epicenter_distance = (
                time_diff
                * (self.p_wave_velocity * self.s_wave_velocity)
                / (self.p_wave_velocity - self.s_wave_velocity)
            )

        warnings = self._generate_warnings(earthquake_detected, magnitude_class)

        diagnostics: HazardDiagnostics | None = None
        if self.keep_diagnostics:
            # Capture the spectrogram exactly as fed to the CNN (f, t, Sxx_norm)
            # plus the STA/LTA series computed above -- no recomputation.
            arrays: dict[str, np.ndarray[Any, Any]] = {
                "spectrogram_freqs_hz": np.asarray(f, dtype=float),
                "spectrogram_times_s": np.asarray(t, dtype=float),
                "spectrogram_norm": np.asarray(Sxx_norm, dtype=float),
            }
            if sta_lta is not None:
                arrays["sta_lta_ratio"] = np.asarray(sta_lta, dtype=float)
            diagnostics = HazardDiagnostics(
                hazard="earthquake",
                arrays=arrays,
                context={
                    "sampling_rate_hz": float(self.sampling_rate),
                    "p_arrival_index": p_arrival,
                    "s_arrival_index": s_arrival,
                    "sta_lta_available": sta_lta is not None,
                },
            )

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
            detection_method=detection_method,
            diagnostics=diagnostics,
            partial=grader.partial,
            set_aside=list(grader.set_aside),
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
        as_of: datetime | None = None,
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
        removed in the transparency wave). ``aftershock_probability`` is the
        Reasenberg-Jones probability of an M >= 5 aftershock of the largest
        event in the 7 days after ``as_of`` when the detector was given
        ``aftershock_parameters`` for the region; otherwise it stays 0.0 (not
        estimated). Rate/clustering features (events/day,
        maximum-likelihood b-value per Aki 1965, clustered-event fraction)
        are reported in ``live_context``.

        Args:
            min_magnitude: Optional catalog magnitude floor.
            station_lat: Optional station latitude for epicentral distance.
            station_lon: Optional station longitude for epicentral distance.
            allow_simulated: Explicit opt-in for simulated sources.
            as_of: Forecast start for the aftershock window (default: now, UTC).
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

        b_value, b_value_ci95, completeness_mc, n_above_mc = gutenberg_richter_b(magnitudes)

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

        aftershock_probability = 0.0
        if self.aftershock_parameters is not None:
            elapsed_days = max(
                ((as_of or datetime.now(UTC)) - strongest.timestamp).total_seconds() / 86400.0,
                0.0,
            )
            aftershock_probability = reasenberg_jones_probability(
                max_magnitude, self.aftershock_parameters, start_days=elapsed_days
            )

        magnitude_class = self._classify_magnitude(max_magnitude)
        result = EarthquakePredictionResult(
            earthquake_detected=True,
            confidence=float(strongest.confidence),
            estimated_magnitude=max_magnitude,  # observed catalog magnitude
            magnitude_class=magnitude_class,
            epicenter_distance_km=epicenter_distance,
            depth_km=(strongest.location[2] if strongest.location is not None else None),
            # A catalog says an earthquake happened, not that the caller will
            # feel it: protective actions need a location-specific shaking
            # estimate (ShakeMap), so a global M6 no longer tells everyone to
            # "Take protective action".
            warning_actions=["Monitor official earthquake information"],
            aftershock_probability=aftershock_probability,
            detection_method="usgs_catalog",
            source_id=fetch.source_id,
            data_provenance=fetch.data_provenance,
            live_context={
                "event_count": len(events),
                "events_per_day": events_per_day,
                "window_days": span_days,
                "max_magnitude": max_magnitude,
                "mean_magnitude": float(magnitudes.mean()),
                "b_value": b_value,
                "b_value_ci95": b_value_ci95,
                "completeness_magnitude": completeness_mc,
                "events_above_completeness": n_above_mc,
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
        """Peak STA/LTA trigger ratio of the conditioned trace (0.0 if too short)."""
        ratio = sta_lta_ratio(condition_trace(data, self.sampling_rate), self.sampling_rate)
        return float(ratio.max()) if ratio is not None else 0.0

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
        """Classify earthquake magnitude.

        Raises:
            InvalidObservationError: If ``magnitude`` is not finite (NaN fell
                through every ``<`` to "great" before).
        """
        magnitude = finite_scalar(magnitude, "magnitude", unit="magnitude units")
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

    def _sta_lta_series(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
        """Compute the STA/LTA ratio series used for P/S arrival picking.

        Args:
            data: 1-D seismic waveform.

        Returns:
            The per-sample short-term/long-term average ratio of the
            conditioned trace (:func:`condition_trace`), or ``None`` when the
            record is shorter than one STA+LTA window (no ratio exists).
        """
        return sta_lta_ratio(condition_trace(data, self.sampling_rate), self.sampling_rate)

    @staticmethod
    def _first_arrival(
        sta_lta: np.ndarray[Any, Any], wave_type: str, min_index: int = 0
    ) -> int | None:
        """Pick the first sample where the STA/LTA ratio crosses the wave threshold.

        Args:
            sta_lta: STA/LTA ratio series from :meth:`_sta_lta_series`.
            wave_type: ``"p"`` (threshold 3.0) or ``"s"`` (threshold 2.0).
            min_index: Ignore triggers before this sample. Required for a
                meaningful S pick on a single trace: the lower S threshold
                otherwise always fires at-or-before the P trigger, so the S
                search must start after the P arrival.

        Returns:
            The first crossing index, or ``None`` if the threshold is never crossed.
        """
        threshold = 3.0 if wave_type == "p" else 2.0
        arrivals = np.where(sta_lta[min_index:] > threshold)[0]

        return int(arrivals[0]) + min_index if len(arrivals) > 0 else None

    def _detect_wave_arrival(
        self, data: np.ndarray[Any, Any], wave_type: str, min_index: int = 0
    ) -> int | None:
        """Detect P or S wave arrival time using STA/LTA.

        Args:
            data: The seismic trace.
            wave_type: ``"p"`` (threshold 3.0) or ``"s"`` (threshold 2.0).
            min_index: Ignore triggers before this sample (see
                :meth:`_first_arrival`).
        """
        sta_lta = self._sta_lta_series(data)
        if sta_lta is None:
            return None
        return self._first_arrival(sta_lta, wave_type, min_index=min_index)

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
        # The same gap-free run predict_earthquake grades: a NaN sample used
        # to make every summary feature NaN.
        seismic_data = InputGrader().usable_run(seismic_data, "seismic_data", unit="counts")

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

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "Live corroboration legs consume operational NASA/JPL "
                "products as published: a recent significant fireball from "
                "the JPL CNEOS Fireball feed, upcoming close approaches from "
                "NASA NeoWs, and Sentry impact risks whose published "
                "cumulative impact probability floors the reported "
                "impact_probability (never understated)."
            ),
            citation=(
                "NASA/JPL CNEOS SSD APIs (ssd-api.jpl.nasa.gov): Fireball "
                "API, Sentry impact-risk API (cumulative impact probability, "
                "Palermo/Torino ratings); NASA NeoWs close-approach feed "
                "(api.nasa.gov)"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The detection rule itself is unsourced engineering: a "
                "Bayesian update over optical/radar detections with assumed "
                "sensitivities (0.8 optical, 0.9 radar) and false-positive "
                "rates (0.01/0.005), posterior boosts for NASA corroboration "
                "(+0.2 fireball, +0.15 close approach), thresholded at the "
                "constructor's detection_threshold (default 0.7); the Sentry "
                "screen keeps objects with palermo_scale > -3 and the "
                "close-approach screen uses < 1 lunar distance within 7 "
                "days. None of these constants has a published source or a "
                "labelled calibration."
            ),
        ),
    )

    def __init__(
        self,
        detection_threshold: float = 0.7,
        prior_probability: float = 1e-6,
        use_nasa_data: bool = True,
        fireball_source: JPLFireballSource | None = None,
        neo_source: NASANeoWsSource | None = None,
        sentry_source: JPLSentrySource | None = None,
        keep_diagnostics: bool = False,
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
            keep_diagnostics: When True and radar data is supplied, the prediction
                result carries the first-difference Doppler shift profile whose
                mean drives the velocity estimate (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Default False keeps memory behavior unchanged.
        """
        self.detection_threshold = detection_threshold
        self.bayesian_filter = BayesianMeteorFilter(prior_probability=prior_probability)
        self.rng = get_global_rng()
        self.keep_diagnostics = keep_diagnostics

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
            MeteorPredictionResult with detection details including NASA CNEOS
            data.  An unusable sensor series or stub is set aside and reported
            (``partial``, ``set_aside``); a NaN sample used to make the
            percentile NaN and every comparison False, a silent "no meteor".

        Raises:
            InvalidObservationError: When nothing usable was fed and no live
                feed answered (an unobserved sky is not a quiet one).
        """
        grader = InputGrader()
        if optical_data is not None:
            try:
                optical_data = grader.usable_run(optical_data, "optical_data", unit="brightness")
            except InvalidObservationError as exc:
                grader.discard("optical_data", str(exc))
                optical_data = None
        if radar_data is not None:
            try:
                radar_data = grader.usable_run(radar_data, "radar_data", unit="radar return")
            except InvalidObservationError as exc:
                grader.discard("radar_data", str(exc))
                radar_data = None
        if noaa_stub is not None and not isinstance(noaa_stub, dict):
            grader.discard(
                "noaa_stub", f"must be a mapping, got {type(noaa_stub).__name__}", noaa_stub
            )
            noaa_stub = None
        optical_detection = False
        radar_detection = False

        # Process local optical sensor data
        if optical_data is not None:
            optical_threshold = np.percentile(optical_data, 99)
            # bool(): the numpy comparison yields np.bool_, which leaked into
            # the bool-typed result field.
            optical_detection = bool(np.max(optical_data) > optical_threshold * 1.5)

        # Process local radar sensor data
        if radar_data is not None:
            radar_threshold = np.percentile(radar_data, 99)
            radar_detection = bool(np.max(radar_data) > radar_threshold * 1.5)

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

        grader.require_usable(
            optical_data is not None
            or radar_data is not None
            or noaa_stub is not None
            or bool(live_sources_used),
            "no observation: supply optical_data, radar_data or noaa_stub, or enable a "
            "live feed that answers; an unobserved sky is not a quiet one",
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

        # Capture the Doppler shift profile ONCE; the velocity estimate consumes
        # its mean and (with keep_diagnostics) the profile itself is persisted.
        doppler_profile: np.ndarray[Any, Any] | None = None
        if radar_data is not None and len(radar_data) >= 2:
            doppler_profile = np.diff(np.asarray(radar_data, dtype=float))

        # Fall back to local radar estimates
        if size_estimate is None and meteor_detected and radar_data is not None:
            size_estimate = self._estimate_size(radar_data)
        if velocity_estimate is None and meteor_detected and radar_data is not None:
            velocity_estimate = self._estimate_velocity(radar_data, profile=doppler_profile)

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

        diagnostics: HazardDiagnostics | None = None
        if self.keep_diagnostics and doppler_profile is not None:
            diagnostics = HazardDiagnostics(
                hazard="meteor",
                arrays={"doppler_shift_profile": doppler_profile},
                context={
                    "n_radar_samples": len(radar_data) if radar_data is not None else 0,
                    "mean_doppler_shift": float(doppler_profile.mean()),
                },
            )

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
            diagnostics=diagnostics,
            partial=grader.partial,
            set_aside=list(grader.set_aside),
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

    def _estimate_velocity(
        self,
        radar_data: np.ndarray[Any, Any],
        profile: np.ndarray[Any, Any] | None = None,
    ) -> float:
        """Estimate meteor velocity from Doppler shift.

        Args:
            radar_data: Radar return series.
            profile: Optional precomputed first-difference Doppler profile
                (``np.diff(radar_data)``) so the caller can capture it without
                recomputation.

        Returns:
            Velocity estimate in km/s.
        """
        if len(radar_data) < 2:
            return 20.0

        if profile is None:
            profile = np.diff(radar_data)
        doppler_shift = profile.mean()
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
        """Extract features for fusion pipeline.

        The sensor series are read as their gap-free runs, as
        :meth:`predict_meteor` grades them (a NaN sample used to make the
        summary features NaN); an unusable series contributes no features.
        """
        features = np.zeros(FEATURE_DIM)
        grader = InputGrader()
        series: dict[str, np.ndarray[Any, Any] | None] = {}
        for name, raw in (("optical_data", optical_data), ("radar_data", radar_data)):
            series[name] = None
            if raw is not None:
                try:
                    series[name] = grader.usable_run(raw, name, unit="sensor counts")
                except InvalidObservationError:
                    pass
        optical_data, radar_data = series["optical_data"], series["radar_data"]

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

    # Batch/shuffle over the ACTUAL number of loaded samples, not the
    # ``n_samples`` synthetic-fallback parameter: on the real-data path
    # ``load_dart_buoy_data`` returns a different count, so using ``n_samples``
    # (default 1000) would make ``torch.randperm`` index the shorter real
    # tensors out of bounds and divide the accuracy by the wrong denominator.
    n_train = waveforms_tensor.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_height_mse = 0.0

        # Shuffle data
        indices = torch.randperm(n_train)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
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
        accuracy = epoch_correct / n_train
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

    # Batch/shuffle over the ACTUAL number of loaded samples, not the
    # ``n_samples`` synthetic-fallback parameter: on the real-data path
    # ``load_usgs_earthquake_catalog`` returns a different count, so using
    # ``n_samples`` (default 1000) would make ``torch.randperm`` index the
    # shorter real tensors out of bounds and divide accuracy by the wrong
    # denominator.
    n_train = spectrograms_tensor.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_mag_mse = 0.0

        # Shuffle data
        indices = torch.randperm(n_train)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = indices[start_idx:end_idx]

            batch_spectrograms = spectrograms_tensor[batch_indices]
            batch_labels = labels_tensor[batch_indices]
            batch_magnitudes = magnitudes_tensor[batch_indices]

            optimizer.zero_grad()

            # Forward pass. SeismicWaveAnalyzer.forward returns a 4-tuple
            # (earthquake_prob, magnitude, p_wave_prob, s_wave_prob); this
            # trainer only supervises detection + magnitude, so the phase-pick
            # heads are unpacked and left unsupervised (unpacking only two values
            # raised ValueError and made this function crash on every batch).
            pred_prob, pred_magnitude, _p_wave_prob, _s_wave_prob = model(batch_spectrograms)

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
        accuracy = epoch_correct / n_train
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


def load_noaa_tsunami_records(
    min_year: int = 2000,
    max_records: int = 1000,
) -> list[dict[str, Any]] | None:
    """Load historical tsunami event records from NOAA NCEI HazEL.

    Data source: NOAA National Centers for Environmental Information
    https://www.ngdc.noaa.gov/hazel/view/hazards/tsunami/event-search

    HazEL pages its results (``itemsPerPage`` up to 200, ``page`` from 1, with
    ``totalPages`` in each response) and ignores any other size parameter, so
    the records are read page by page until ``max_records`` or the last page.

    Args:
        min_year: Minimum year for historical records
        max_records: Maximum number of records to fetch

    Returns:
        Up to ``max_records`` tsunami event dictionaries, or None if the
        service is unavailable or returned no events.

    Raises:
        OfflineModeError: ``MERCURY_OFFLINE`` is set. The air-gap refusal is
            raised, not reported as an unavailable service.
        ValueError: ``max_records`` is not positive.
    """
    from omni_mercury_engine.security.safe_http import enforce_offline_egress

    if max_records < 1:
        raise ValueError(f"max_records must be positive, got {max_records}")
    enforce_offline_egress(NOAA_TSUNAMI_API_URL)
    circuit_breaker = get_data_loader_breaker("noaa_tsunami")
    per_page = min(200, max_records)

    def _fetch_tsunami_records() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        page, total_pages = 1, 1
        while page <= total_pages and len(events) < max_records:
            data = SafeHTTPClient.get_json(
                NOAA_TSUNAMI_API_URL,
                params={
                    "minYear": str(min_year),
                    "itemsPerPage": str(per_page),
                    "page": str(page),
                },
                headers={"User-Agent": "Mercury-Agent/1.0"},
                timeout=30,
            )
            events.extend(data.get("items", []))
            total_pages = int(data.get("totalPages", 0))
            page += 1
        if not events:
            raise RuntimeError("NOAA Tsunami API returned no events")
        return events[:max_records]

    try:
        result: list[dict[str, Any]] = circuit_breaker.call(_fetch_tsunami_records)
        return result
    except Exception as e:
        logger.warning(f"Failed to load NOAA tsunami records: {e}; returning None.")
        return None
