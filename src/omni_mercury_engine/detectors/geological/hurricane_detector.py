# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hurricane/Cyclone/Typhoon Detector - Tropical Cyclone Monitoring System.

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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy.fft import fft
from torch import nn

from omni_mercury_engine.core.three_r_mechanism import (
    RecursionEngine,
    RefactoringEngine,
    ResonanceEngine,
)
from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics
from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


class SaffirSimpsonCategory(Enum):
    """Saffir-Simpson Hurricane Wind Scale categories."""

    TROPICAL_DEPRESSION = "tropical_depression"
    TROPICAL_STORM = "tropical_storm"
    CATEGORY_1 = "category_1"
    CATEGORY_2 = "category_2"
    CATEGORY_3 = "category_3"
    CATEGORY_4 = "category_4"
    CATEGORY_5 = "category_5"
    NO_CYCLONE = "no_cyclone"


class CycloneType(Enum):
    """Types of tropical cyclones by basin."""

    HURRICANE = "hurricane"  # Atlantic, Eastern Pacific
    TYPHOON = "typhoon"  # Western Pacific
    CYCLONE = "cyclone"  # Indian Ocean, South Pacific
    NO_CYCLONE = "no_cyclone"


@dataclass
class HurricanePredictionResult:
    """Hurricane/cyclone prediction results."""

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

    # Observed wind-field kinematics (populated when a wind field is supplied).
    max_relative_vorticity_s1: float | None = None
    closed_circulation: bool = False

    # NOTE: track_forecast / landfall_probability / time_to_landfall_hours were
    # removed deliberately: they were declared but never computed anywhere, and
    # a transparent track forecast requires steering-flow data and a track model
    # this detector does not have. Advertising uncomputed skill is theater.

    warning_actions: list[str] = field(default_factory=list)
    evacuation_zones: list[str] = field(default_factory=list)

    # Populated only when the detector was built with keep_diagnostics=True.
    # Carries the wind speed field, u/v components, and the finite-difference
    # vorticity field. There is deliberately NO track cone: the storm-track
    # model was removed as uncomputed, and none is fabricated here.
    diagnostics: HazardDiagnostics | None = None


class SeaSurfaceTemperatureAnalyzer:
    """Sea surface temperature analysis for cyclone development potential.

    SST >= 26.5°C is generally required for tropical cyclone formation.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.formation_threshold_c = 26.5
        self.intensification_threshold_c = 28.0

    def analyze_sst(self, sst_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze sea surface temperature for cyclone potential.

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


#: Class order of the WindPatternAnalyzer's 8-way category head. This is the
#: single source of truth shared with the training pipeline
#: (``ml/hazard_training/hurricane_wind.py`` imports it): index 0 is the
#: no-cyclone class trained on far-from-storm patches; 1..7 follow the
#: Saffir-Simpson progression used everywhere else in this detector.
NEURAL_CATEGORY_ORDER: tuple[str, ...] = (
    "no_cyclone",
    "tropical_depression",
    "tropical_storm",
    "category_1",
    "category_2",
    "category_3",
    "category_4",
    "category_5",
)


class WindPatternAnalyzer(nn.Module):
    """Wind pattern analyzer for cyclone structure detection.

    Uses CNN + LSTM to identify organized convection and eye formation.
    """

    def __init__(self, input_channels: int = 3, hidden_dim: int = 128) -> None:
        """Initialize the instance."""
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
        """Analyze wind field patterns.

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
    """Central pressure tracking for cyclone intensity monitoring.

    Lower central pressure indicates stronger cyclone.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def analyze_pressure(self, pressure_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze central pressure for intensity assessment.

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
        """Estimate max wind from central pressure using Dvorak relationship.

        V_max = 6.7 * (1013 - P)^0.644
        """
        if pressure >= 1013:
            return 0.0
        deficit = 1013 - pressure
        wind = 6.7 * (deficit**0.644)
        return float(min(wind, 200.0))


class ResonanceFrequencyAmplifier:
    """FFT-based resonance frequency amplifier for storm signal detection.

    Implements the Resonance Engine component of 3R for amplifying weak cyclone signatures in
    atmospheric data.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.cyclone_frequencies = [0.05, 0.1, 0.2, 0.5]

    def amplify_signals(self, signal: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Amplify cyclone-characteristic frequency signals.

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
    """Comprehensive hurricane/cyclone/typhoon detection system.

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
        keep_diagnostics: bool = False,
        load_shipped_weights: bool = True,
    ):
        """Initialize the instance.

        Args:
            enable_sst: Enable sea-surface-temperature analysis.
            enable_wind: Enable the wind-pattern analyzer components.
            enable_pressure: Enable central-pressure tracking.
            enable_resonance: Enable resonance frequency amplification.
            enable_recursion: Enable recursive multi-scale feature extraction.
            enable_refactoring: Enable the 3R refactoring engine.
            rng: Deterministic RNG for reproducibility.
            keep_diagnostics: When True and ``wind_field`` data is supplied, each
                prediction result carries the wind speed field and (when u/v
                components are given) the finite-difference relative-vorticity
                field plus circulation metrics (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Detection scalars stay pressure/SST-driven either way. Default
                False keeps memory behavior unchanged.
            load_shipped_weights: Load the shipped merit-gated ``hurricane_era5``
                checkpoint at construction (default), so a default-constructed
                detector serves the ratified winner. Pass False for the pure
                observed-kinematics physics configuration (the hazard regression
                guard's physics lane and the honesty-contract tests). Absence of
                the checkpoint falls open to physics; an invalid checkpoint still
                fails loud.
        """
        self.enable_sst = enable_sst
        self.enable_wind = enable_wind
        self.enable_pressure = enable_pressure
        self.enable_resonance = enable_resonance
        self.enable_recursion = enable_recursion
        self.enable_refactoring = enable_refactoring
        self.keep_diagnostics = keep_diagnostics
        self._rng = rng or get_global_rng()

        self.sst_analyzer = SeaSurfaceTemperatureAnalyzer() if enable_sst else None
        self.wind_analyzer = WindPatternAnalyzer() if enable_wind else None
        self.pressure_tracker = PressureTracker() if enable_pressure else None
        self.resonance_amplifier = ResonanceFrequencyAmplifier() if enable_resonance else None

        self.recursion_engine = RecursionEngine(max_depth=5)
        self.resonance_engine = ResonanceEngine(sampling_rate=1.0)
        self.refactoring_engine = RefactoringEngine()

        # Anti-theater guard (mirrors SchumannResonanceDetector): the
        # WindPatternAnalyzer CNN+LSTM ships with random weights, no labelled
        # cyclone corpus exists to train it -- and it was never even called, so
        # supplied wind data was silently ignored. Wind fields are now analysed
        # with deterministic kinematics (observed maximum wind + relative
        # vorticity, the standard measure of organized cyclonic circulation);
        # the network is consulted only after load_neural_weights().
        self._neural_trained = False
        self._warned_untrained = False
        self._feature_spec: str | None = None

        self.logger = logging.getLogger(__name__)

        # The hurricane_era5 checkpoint cleared the hazard merit gate on real
        # held-out data, so a default-constructed detector serves the shipped
        # winner. Absence (e.g. a stripped install) falls open to the disclosed
        # observed-kinematics physics; a present-but-invalid checkpoint still
        # fails loud inside load_neural_weights (sha256/state-dict validation).
        if load_shipped_weights and self.wind_analyzer is not None:
            try:
                self.load_neural_weights()
            except FileNotFoundError:
                self.logger.debug(
                    "No shipped 'hurricane_era5' checkpoint available; analysing "
                    "wind fields with observed-kinematics physics."
                )

    def load_neural_weights(self, checkpoint_path: str | None = None) -> None:
        """Load trained weights for the wind-pattern analyzer.

        Until this is called the network is untrained and wind fields are
        analysed with the deterministic kinematics of
        :meth:`_analyze_wind_field` only.

        Args:
            checkpoint_path: Path to a torch checkpoint containing a
                ``wind_analyzer`` state dict. ``None`` loads the shipped
                default checkpoint (``hurricane_era5``, trained on ERA5 10 m
                wind patches labeled with IBTrACS best-track intensities),
                whose provenance sidecar is logged; missing or corrupt files
                raise instead of degrading silently.
        """
        if self.wind_analyzer is None:
            raise RuntimeError("wind analysis is disabled on this detector")
        if checkpoint_path is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            checkpoint, _provenance = load_shipped_checkpoint("hurricane_era5")
            source = "shipped default 'hurricane_era5'"
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            source = checkpoint_path
        self.wind_analyzer.load_state_dict(checkpoint["wind_analyzer"])
        self.wind_analyzer.eval()
        self._feature_spec = str(checkpoint.get("feature_spec", "unknown"))
        self._neural_trained = True
        self.logger.info(
            "Hurricane wind neural weights loaded from %s (feature spec: %s); "
            "using learned analyzer",
            source,
            self._feature_spec,
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the untrained NN is bypassed for physics."""
        if not self._warned_untrained:
            self.logger.warning(
                "HurricaneDetector's WindPatternAnalyzer is untrained (no checkpoint "
                "loaded); analysing wind fields with the deterministic observed-"
                "kinematics physics instead of the NN. Call load_neural_weights() "
                "once a trained checkpoint exists."
            )
            self._warned_untrained = True

    def predict_hurricane(self, cyclone_data: dict[str, Any]) -> HurricanePredictionResult:
        """Comprehensive tropical cyclone prediction.

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

        indicators_detected: float = 0.0

        if self.enable_sst and "sst_data" in cyclone_data:
            assert self.sst_analyzer is not None, "SST analyzer must be initialized"
            sst_result = self.sst_analyzer.analyze_sst(cyclone_data["sst_data"])
            result.sst_anomaly_c = sst_result["sst_anomaly_c"]
            result.ocean_heat_content = sst_result["ocean_heat_content"]
            if sst_result["favorable_formation"]:
                indicators_detected += 1
            if sst_result["favorable_intensification"]:
                indicators_detected += 0.5

        if self.enable_pressure and "pressure_data" in cyclone_data:
            assert self.pressure_tracker is not None, "Pressure tracker must be initialized"
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

        if self.enable_wind and "wind_field" in cyclone_data:
            if self._neural_trained:
                wind_result = self._analyze_wind_field_neural(cyclone_data["wind_field"])
            else:
                self._warn_untrained_once()
                wind_result = self._analyze_wind_field(cyclone_data["wind_field"])
            result.max_relative_vorticity_s1 = wind_result["max_relative_vorticity_s1"]
            result.closed_circulation = wind_result["closed_circulation"]
            # The observed field can only raise the wind estimate, never mask a
            # stronger pressure-derived value.
            result.max_wind_speed_kt = max(
                result.max_wind_speed_kt, wind_result["max_wind_speed_kt"]
            )
            if wind_result["closed_circulation"]:
                indicators_detected += 1
            elif wind_result["max_relative_vorticity_s1"] > 5e-4:
                indicators_detected += 0.5
            # Learned path only: the category head's tropical-cyclone
            # probability (physics results carry no such key).
            if wind_result.get("neural_tc_probability", 0.0) >= 0.5:
                indicators_detected += 1

        if self.enable_resonance and "signal_data" in cyclone_data:
            assert self.resonance_amplifier is not None, "Resonance amplifier must be initialized"
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
            # Skip refactoring engine code anomaly detection for non-callable data
            # The refactoring engine expects callable functions, not string data
            pass

        if self.keep_diagnostics and self.enable_wind and "wind_field" in cyclone_data:
            # Rebuilds the speed/vorticity fields directly from the caller's own
            # wind field (independent of the detection path above), failing loud
            # on malformed u/v rather than capturing anything imputed.
            result.diagnostics = self._build_wind_diagnostics(cyclone_data["wind_field"])

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

    def _analyze_wind_field(
        self, wind_field: dict[str, Any] | np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Deterministic cyclone kinematics from an observed wind field.

        Previously the supplied wind data was silently ignored (the untrained
        WindPatternAnalyzer was never called). This computes the two standard
        kinematic measures directly from the field:

        * the observed maximum wind speed ``max sqrt(u² + v²)`` (m/s → kt), and
        * the peak relative vorticity ``ζ = ∂v/∂x − ∂u/∂y`` -- organized
          cyclonic circulation shows |ζ| of order 1e-3 s⁻¹ and above at the
          core; 2e-3 s⁻¹ is treated as a closed circulation.

        Args:
            wind_field: ``{"u": 2-D array, "v": 2-D array}`` wind components in
                m/s, plus optional ``grid_spacing_m`` (defaults to 4000 m, a
                typical analysis-grid resolution). A bare array is treated as a
                wind-speed field in m/s: it can still raise the observed maximum
                wind, but no vorticity is derivable from speed alone and none is
                imputed.

        Returns:
            ``max_wind_speed_kt``, ``max_relative_vorticity_s1``, and
            ``closed_circulation``. Deterministic; missing/1-D components yield
            zero vorticity rather than anything imputed.
        """
        if not isinstance(wind_field, dict):
            speed_ms = np.asarray(wind_field, dtype=float)
            return {
                "max_wind_speed_kt": (
                    float(np.nanmax(speed_ms) * 1.9438) if speed_ms.size else 0.0
                ),
                "max_relative_vorticity_s1": 0.0,
                "closed_circulation": False,
            }

        u = np.asarray(wind_field.get("u", []), dtype=float)
        v = np.asarray(wind_field.get("v", []), dtype=float)
        spacing = float(wind_field.get("grid_spacing_m", 4000.0))

        if u.size == 0 or v.size == 0 or u.shape != v.shape:
            return {
                "max_wind_speed_kt": 0.0,
                "max_relative_vorticity_s1": 0.0,
                "closed_circulation": False,
            }

        speed_ms = np.sqrt(u**2 + v**2)
        max_wind_kt = float(np.nanmax(speed_ms) * 1.9438)

        max_vorticity = 0.0
        if u.ndim == 2 and min(u.shape) >= 2:
            dv_dx = np.gradient(v, spacing, axis=1)
            du_dy = np.gradient(u, spacing, axis=0)
            zeta = dv_dx - du_dy
            max_vorticity = float(np.nanmax(np.abs(zeta)))

        return {
            "max_wind_speed_kt": max_wind_kt,
            "max_relative_vorticity_s1": max_vorticity,
            "closed_circulation": max_vorticity >= 2e-3,
        }

    def _analyze_wind_field_neural(
        self, wind_field: dict[str, Any] | np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Learned wind-field analysis (only reached after trained weights load).

        The observed kinematics of :meth:`_analyze_wind_field` stay authoritative
        for what was *measured* -- vorticity, closed circulation, and the
        observed maximum wind -- computed from the identical input. The trained
        WindPatternAnalyzer then contributes what the coarse field cannot show
        directly: its intensity head is trained in **knots** against IBTrACS
        best-track maximum sustained winds (feature spec ``hurricane-era5-v1``,
        channels u10/v10/speed in m/s, no input standardization -- the raw
        caller field is the network input, guaranteeing train/serve parity),
        so its output is used as ``max_wind_speed_kt`` directly, floored at the
        observed patch maximum: real measured wind can only ever raise the
        estimate, mirroring how this method's caller treats pressure-derived
        winds.

        Args:
            wind_field: ``{"u": array, "v": array}`` wind components in m/s.
                2-D ``(H, W)`` arrays are a single-time field; 3-D
                ``(T, H, W)`` arrays are a time sequence, oldest first (the
                shipped checkpoint is trained on T=2 frames at t-6h and t).
                Inputs the network cannot consume (bare speed arrays,
                mismatched or non-finite components) fall back to the
                deterministic physics analysis of the same data -- nothing is
                imputed.

        Returns:
            The physics keys (``max_wind_speed_kt``,
            ``max_relative_vorticity_s1``, ``closed_circulation``) plus
            ``neural_max_wind_kt``, ``neural_tc_probability`` (1 minus the
            no-cyclone class probability), and ``neural_category``.
        """
        physics = self._analyze_wind_field(wind_field)
        if not isinstance(wind_field, dict):
            return physics
        u = np.asarray(wind_field.get("u", []), dtype=np.float32)
        v = np.asarray(wind_field.get("v", []), dtype=np.float32)
        if u.ndim == 2:
            u, v = u[None], v[None]
        if (
            u.ndim != 3
            or u.shape != v.shape
            or u.size == 0
            or not (np.isfinite(u).all() and np.isfinite(v).all())
            # The conv encoder's MaxPool2d(2) needs at least a 2x2 spatial
            # field; a degenerate single-row/column field -- e.g. a (1, W)
            # transect promoted to (T=1, H=1, W) above -- pools to height 0
            # and crashes. The physics analysis handles the same input, so
            # off-contract spatial dims fall back there (solar geomag guard
            # pattern).
            or u.shape[1] < 2
            or u.shape[2] < 2
        ):
            return physics

        frames = np.stack([u, v, np.hypot(u, v)], axis=1)  # (T, 3, H, W)
        tensor = torch.from_numpy(frames).unsqueeze(0)  # (1, T, 3, H, W)
        assert self.wind_analyzer is not None, "wind analyzer must be initialized"
        self.wind_analyzer.eval()
        with torch.no_grad():
            max_wind, category_logits = self.wind_analyzer(tensor)
            probs = torch.softmax(category_logits[0], dim=-1)
        neural_kt = float(max_wind[0, 0])
        tc_probability = float(1.0 - probs[0])
        category = NEURAL_CATEGORY_ORDER[int(torch.argmax(probs))]
        return {
            **physics,
            "max_wind_speed_kt": max(neural_kt, physics["max_wind_speed_kt"]),
            "neural_max_wind_kt": neural_kt,
            "neural_tc_probability": tc_probability,
            "neural_category": category,
        }

    @staticmethod
    def _build_wind_diagnostics(wind_field: Any) -> HazardDiagnostics:
        """Compute the wind-speed and vorticity fields from a supplied wind field.

        Everything here is a deterministic derivation of the caller's own field:
        speed is ``hypot(u, v)`` and relative vorticity is the central
        finite-difference curl ``dv/dx - du/dy`` (``numpy.gradient`` with the
        supplied grid spacing). No storm track is produced -- the track model
        was removed as uncomputed and nothing is fabricated in its place.

        Args:
            wind_field: Either a mapping with 2-D ``u`` and ``v`` component
                arrays (m/s) plus optional ``grid_spacing_m``, or a single 2-D
                array treated as a wind-speed field (no vorticity is derivable
                from speed alone, and none is emitted).

        Returns:
            The hurricane :class:`HazardDiagnostics` payload.

        Raises:
            ValueError: If the field is not 2-D or u/v shapes disagree.
        """
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        context: dict[str, Any] = {}

        if isinstance(wind_field, dict):
            if "u" not in wind_field or "v" not in wind_field:
                raise ValueError("wind_field mapping requires 'u' and 'v' 2-D component arrays")
            u = np.asarray(wind_field["u"], dtype=float)
            v = np.asarray(wind_field["v"], dtype=float)
            if u.ndim != 2 or u.shape != v.shape:
                raise ValueError(
                    f"wind_field u/v must be matching 2-D arrays, got {u.shape} and {v.shape}"
                )
            spacing = float(wind_field.get("grid_spacing_m", 1.0))
            if spacing <= 0:
                raise ValueError("wind_field grid_spacing_m must be positive")
            speed = np.hypot(u, v)
            # Relative vorticity: dv/dx - du/dy. Rows are y, columns are x.
            dv_dx = np.gradient(v, spacing, axis=1)
            du_dy = np.gradient(u, spacing, axis=0)
            vorticity = dv_dx - du_dy
            arrays["wind_u"] = u
            arrays["wind_v"] = v
            arrays["vorticity_field"] = vorticity
            context["grid_spacing_m"] = spacing
            context["max_abs_vorticity"] = float(np.max(np.abs(vorticity)))
            context["mean_vorticity"] = float(np.mean(vorticity))
        else:
            speed = np.asarray(wind_field, dtype=float)
            if speed.ndim != 2:
                raise ValueError(f"wind_field must be a 2-D speed field, got ndim={speed.ndim}")
            context["note"] = (
                "speed-only field supplied; vorticity requires u/v components and is not emitted"
            )

        arrays["wind_speed_field"] = speed
        context["max_wind_speed"] = float(np.max(speed)) if speed.size else 0.0
        context["mean_wind_speed"] = float(np.mean(speed)) if speed.size else 0.0
        return HazardDiagnostics(hazard="hurricane", arrays=arrays, context=context)

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

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        features = []

        if self.enable_resonance and self.resonance_amplifier is not None:
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
