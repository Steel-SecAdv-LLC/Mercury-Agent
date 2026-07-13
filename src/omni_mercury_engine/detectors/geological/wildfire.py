# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wildfire Detector - Ignition, Spread & Risk Assessment.

Advanced wildfire detection and prediction for humanitarian early warning:
- Ignition detection (satellite thermal + ground sensors)
- Fire spread modeling (weather fusion + vegetation fuel)
- Risk assessment (drought + temperature + wind)
- Smoke plume tracking
- Controlled burn optimization
- Cascade detection (fire → mudslide → flooding)

Integrations:
- Thermal satellite data (MODIS, VIIRS)
- Weather data fusion (wind, humidity, temperature)
- Vegetation indices (NDVI for fuel load)
- Resilience framework for post-fire hazards

Research sources:
- NOAA/NASA GOES fire detection
- USFS wildfire science
- FIRMS (Fire Information for Resource Management System)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from scipy.fft import fft, fftfreq
from torch import nn

from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics

# Brightness temperature (Kelvin) above which a pixel counts as a thermal hotspot.
HOTSPOT_THRESHOLD_K = 350.0

# Default decision threshold on the CNN fire probability (the detector's
# historical fixed alert rule). A trained checkpoint may carry a
# validation-selected operating point that replaces it for the ALERT decision
# only (see load_neural_weights); the reported confidence always stays the raw
# sigmoid probability.
FIRE_PROB_THRESHOLD_DEFAULT = 0.6


class FireRiskLevel(Enum):
    """Fire risk classifications."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class WildfirePredictionResult:
    """Wildfire prediction results.

    Coordinate transparency: ``ignition_locations`` are PIXEL-SPACE ``(row, col)``
    centroids of connected hotspot regions in the supplied thermal image. The
    detector has no geotransform, so it never fabricates lat/lon; callers that
    know the image georeferencing map pixels to WGS84 themselves (see
    ``build_hazard_geojson``). ``fire_perimeter_km2`` is the hotspot-area
    estimate ``hotspot_pixel_count * pixel_size_km**2`` and is populated only
    when the caller supplies ``pixel_size_km`` in the wildfire data -- otherwise
    it stays ``None`` rather than inventing a ground resolution.
    """

    fire_detected: bool
    confidence: float
    risk_level: str

    ignition_locations: list[tuple[float, float]] = field(default_factory=list)
    fire_perimeter_km2: float | None = None
    spread_rate_km_hr: float | None = None
    spread_direction_deg: float | None = None

    thermal_hotspots: int = 0
    smoke_detected: bool = False

    weather_factors: dict[str, float] = field(default_factory=dict)
    fuel_moisture: float | None = None

    evacuation_zones: list[str] = field(default_factory=list)
    containment_strategy: list[str] = field(default_factory=list)
    early_warning_actions: list[str] = field(default_factory=list)

    # Populated only when the detector was built with keep_diagnostics=True.
    diagnostics: HazardDiagnostics | None = None


class FireIgnitionDetector(nn.Module):
    """CNN scoring fire activity from satellite-derived thermal rasters.

    The shipped ``wildfire_firms`` checkpoint trains this network for
    NEXT-DAY fire activity forecasting: inputs are 3-channel 32x32-cell
    (0.04-degree) rasters built from NASA FIRMS VIIRS detections over days
    ``[t-6..t]`` (today's max brightness temperature in Kelvin, log1p 3-day
    FRP sum, log1p 7-day detection count -- detection-derived fields, NOT
    full thermal imagery), and the sigmoid output is the probability of any
    confirmed vegetation-fire detection in the center 2x2 cells on day t+1.
    See ``omni_mercury_engine.ml.hazard_training.wildfire_ignition``.
    """

    def __init__(self, input_channels: int = 3) -> None:
        """Initialize the instance."""
        super().__init__()

        self.thermal_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fire_classifier = nn.Sequential(
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, thermal_image: torch.Tensor) -> torch.Tensor:
        """Detect fire ignition from thermal imagery."""
        features = self.thermal_cnn(thermal_image)
        features = features.view(features.size(0), -1)
        fire_prob = self.fire_classifier(features)

        return fire_prob


class FireSpreadModel:
    """Fire spread rate and direction prediction.

    Incorporates weather (wind), terrain, and fuel load.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def predict_spread(
        self, fire_data: dict[str, Any], weather_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Predict fire spread dynamics.

        Args:
            fire_data: Current fire parameters
            weather_data: Wind speed, direction, humidity, temperature

        Returns:
            Fire spread prediction
        """
        wind_speed_kmh = weather_data.get("wind_speed_kmh", 10.0)
        wind_direction_deg = weather_data.get("wind_direction_deg", 0.0)
        relative_humidity = weather_data.get("relative_humidity_pct", 50.0)
        temperature_c = weather_data.get("temperature_c", 25.0)

        fuel_load = fire_data.get("fuel_load_tons_ha", 20.0)
        fuel_moisture = fire_data.get("fuel_moisture_pct", 10.0)

        base_spread_rate = 0.5  # km/hr

        wind_factor = 1.0 + (wind_speed_kmh / 20.0)
        humidity_factor = 1.0 - (relative_humidity / 200.0)
        temp_factor = 1.0 + ((temperature_c - 20.0) / 50.0)
        fuel_factor = 1.0 + (fuel_load / 40.0)
        moisture_factor = 1.0 - (fuel_moisture / 100.0)

        spread_rate_km_hr = (
            base_spread_rate
            * wind_factor
            * humidity_factor
            * temp_factor
            * fuel_factor
            * moisture_factor
        )

        spread_rate_km_hr = max(spread_rate_km_hr, 0.1)

        return {
            "spread_rate_km_hr": float(spread_rate_km_hr),
            "spread_direction_deg": float(wind_direction_deg),
            "wind_driven": wind_speed_kmh > 20.0,
        }


class NDVIProcessor:
    """NDVI (Normalized Difference Vegetation Index) processor for fuel load estimation.

    NDVI = (NIR - Red) / (NIR + Red)
    Higher NDVI indicates denser vegetation (more fuel for fires).
    """

    def __init__(
        self,
        fuel_threshold: float = 0.4,
        drought_threshold: float = 0.2,
    ):
        """Initialize NDVI processor.

        Args:
            fuel_threshold: NDVI threshold for high fuel load (default: 0.4)
            drought_threshold: NDVI threshold for drought conditions (default: 0.2)
        """
        self.fuel_threshold = fuel_threshold
        self.drought_threshold = drought_threshold
        self.logger = logging.getLogger(__name__)

    def compute_ndvi(
        self,
        nir_band: np.ndarray[Any, Any],
        red_band: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Compute NDVI from satellite bands.

        Args:
            nir_band: Near-infrared band data
            red_band: Red band data

        Returns:
            NDVI array with values in [-1, 1]
        """
        # Avoid division by zero
        denominator = nir_band.astype(float) + red_band.astype(float)
        denominator = np.where(denominator == 0, 1e-10, denominator)

        ndvi = (nir_band.astype(float) - red_band.astype(float)) / denominator
        return np.clip(ndvi, -1.0, 1.0)

    def estimate_fuel_load(
        self,
        ndvi: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Estimate fuel load from NDVI.

        Args:
            ndvi: NDVI array

        Returns:
            Fuel load estimation with metrics
        """
        mean_ndvi = float(np.mean(ndvi))
        max_ndvi = float(np.max(ndvi))

        # Fuel load estimation (tons/hectare)
        # Based on empirical relationship: fuel_load ≈ 50 * NDVI^2 for vegetated areas
        fuel_load = 50.0 * (mean_ndvi**2) if mean_ndvi > 0 else 0.0

        # Drought assessment
        drought_fraction = float(np.mean(ndvi < self.drought_threshold))
        is_drought = drought_fraction > 0.3

        # High fuel areas
        high_fuel_fraction = float(np.mean(ndvi > self.fuel_threshold))

        return {
            "mean_ndvi": mean_ndvi,
            "max_ndvi": max_ndvi,
            "fuel_load_tons_ha": fuel_load,
            "drought_fraction": drought_fraction,
            "is_drought": is_drought,
            "high_fuel_fraction": high_fuel_fraction,
        }

    def compute_ndvi_change(
        self,
        ndvi_current: np.ndarray[Any, Any],
        ndvi_previous: np.ndarray[Any, Any],
    ) -> dict[str, float]:
        """Compute NDVI change for fire damage assessment.

        Args:
            ndvi_current: Current NDVI
            ndvi_previous: Previous NDVI (pre-fire)

        Returns:
            NDVI change metrics
        """
        delta_ndvi = ndvi_current - ndvi_previous
        mean_change = float(np.mean(delta_ndvi))

        # Burned area detection (significant NDVI decrease)
        burned_fraction = float(np.mean(delta_ndvi < -0.2))

        return {
            "mean_ndvi_change": mean_change,
            "burned_fraction": burned_fraction,
            "recovery_potential": 1.0 - burned_fraction,
        }


class ResonanceFrequencyAnalyzer:
    """3R Resonance mechanism for smoke pattern frequency analysis.

    Analyzes temporal patterns in thermal and smoke data to detect
    fire behavior resonances (e.g., diurnal fire activity cycles,
    wind-driven spread patterns).

    Synapse: Integrates with GOSNN for ethical gating and scalar registration.
    """

    def __init__(
        self,
        sample_rate_hz: float = 1.0,
        phi: float = 1.618033988749895,
    ):
        """Initialize resonance analyzer.

        Args:
            sample_rate_hz: Sampling rate in Hz
            phi: Golden ratio for harmonic weighting
        """
        self.sample_rate_hz = sample_rate_hz
        self.phi = phi
        self.logger = logging.getLogger(__name__)

        # Key fire behavior frequencies (cycles per hour)
        self.fire_frequencies = {
            "diurnal": 1.0 / 24.0,  # Daily cycle
            "wind_gust": 1.0 / 0.5,  # 30-minute wind gusts
            "convective": 1.0 / 2.0,  # 2-hour convective cycles
            "spotting": 1.0 / 0.25,  # 15-minute spotting events
        }

    def analyze_thermal_resonance(
        self,
        thermal_time_series: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Analyze thermal time series for fire behavior patterns.

        Args:
            thermal_time_series: Temperature/thermal readings over time

        Returns:
            Resonance analysis with frequency components
        """
        n = len(thermal_time_series)
        if n < 4:
            return {
                "resonance_score": 0.0,
                "dominant_frequency": 0.0,
                "fire_behavior_detected": False,
            }

        # Compute FFT
        fft_vals = fft(thermal_time_series - np.mean(thermal_time_series))
        freqs = fftfreq(n, d=1.0 / self.sample_rate_hz)

        # Get positive frequencies only
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_power = np.abs(fft_vals[pos_mask]) ** 2

        if len(pos_power) == 0:
            return {
                "resonance_score": 0.0,
                "dominant_frequency": 0.0,
                "fire_behavior_detected": False,
            }

        # Find dominant frequency
        dominant_idx = np.argmax(pos_power)
        dominant_freq = float(pos_freqs[dominant_idx])
        dominant_power = float(pos_power[dominant_idx])

        # Compute resonance score based on fire behavior frequencies
        resonance_scores = []
        for name, target_freq in self.fire_frequencies.items():
            # Find power near target frequency
            freq_mask = np.abs(pos_freqs - target_freq) < target_freq * 0.2
            if np.any(freq_mask):
                power_at_freq = float(np.max(pos_power[freq_mask]))
                resonance_scores.append(power_at_freq / (dominant_power + 1e-10))

        # Aggregate resonance score using phi-weighting
        if resonance_scores:
            weights = np.array([self.phi ** (-i) for i in range(len(resonance_scores))])
            weights /= weights.sum()
            resonance_score = float(np.sum(np.array(resonance_scores) * weights))
        else:
            resonance_score = 0.0

        # Detect fire behavior patterns
        fire_behavior_detected = resonance_score > 0.3 or dominant_power > np.var(
            thermal_time_series
        )

        return {
            "resonance_score": min(1.0, resonance_score),
            "dominant_frequency": dominant_freq,
            "dominant_power": dominant_power,
            "fire_behavior_detected": fire_behavior_detected,
            "frequency_components": dict(
                zip(
                    self.fire_frequencies.keys(),
                    resonance_scores[:4] if resonance_scores else [0] * 4,
                )
            ),
        }

    def analyze_smoke_patterns(
        self,
        smoke_density_series: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Analyze smoke density patterns for fire spread prediction.

        Args:
            smoke_density_series: Smoke density readings over time

        Returns:
            Smoke pattern analysis
        """
        if len(smoke_density_series) < 4:
            return {
                "smoke_trend": "stable",
                "spread_indicator": 0.0,
            }

        # Compute trend
        diff = np.diff(smoke_density_series)
        mean_diff = float(np.mean(diff))

        if mean_diff > 0.1:
            trend = "increasing"
            spread_indicator = min(1.0, mean_diff * 2)
        elif mean_diff < -0.1:
            trend = "decreasing"
            spread_indicator = 0.0
        else:
            trend = "stable"
            spread_indicator = 0.3

        # Detect rapid changes (potential crown fire or spotting)
        max_change = float(np.max(np.abs(diff)))
        rapid_change_detected = max_change > 0.5

        return {
            "smoke_trend": trend,
            "spread_indicator": spread_indicator,
            "rapid_change_detected": rapid_change_detected,
            "max_change_rate": max_change,
        }


class WildfireCNN(nn.Module):
    """Enhanced CNN for wildfire detection from thermal/NDVI satellite inputs.

    Architecture:
    - Multi-channel input (thermal, NIR, Red, NDVI)
    - Conv layers with batch normalization
    - Adaptive pooling for variable input sizes
    - Dual output: fire probability + severity classification
    """

    def __init__(
        self,
        input_channels: int = 4,
        num_severity_classes: int = 5,
    ):
        """Initialize wildfire CNN.

        Args:
            input_channels: Number of input channels (default: 4 for thermal, NIR, Red, NDVI)
            num_severity_classes: Number of severity classes (default: 5)
        """
        super().__init__()

        phi = 1.618

        # Feature extraction backbone
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, int(128 * phi), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(128 * phi)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Feature dimension after conv blocks
        feature_dim = int(128 * phi) * 16

        # Fire detection head
        self.fire_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Severity classification head
        self.severity_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_severity_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Tuple of (fire_probability, severity_logits)
        """
        features = self.conv_block1(x)
        features = self.conv_block2(features)
        features = self.conv_block3(features)

        features = features.view(features.size(0), -1)

        fire_prob = self.fire_head(features)
        severity_logits = self.severity_head(features)

        return fire_prob, severity_logits


class WildfireDetector:
    """Comprehensive wildfire detection and prediction system.

    Enhanced with:
    - NDVI processing for fuel load estimation
    - Enhanced CNN for multi-channel satellite analysis
    - 3R Resonance mechanism for smoke pattern frequency analysis
    - GOSNN synapse for ethical gating and scalar registration
    """

    def __init__(
        self,
        enable_ignition_detection: bool = True,
        enable_spread_modeling: bool = True,
        enable_ndvi_processing: bool = True,
        enable_resonance: bool = True,
        enable_enhanced_cnn: bool = True,
        keep_diagnostics: bool = False,
    ):
        """Initialize the instance.

        Args:
            enable_ignition_detection: Enable thermal-CNN ignition detection.
            enable_spread_modeling: Enable weather-driven spread modeling.
            enable_ndvi_processing: Enable NDVI fuel-load processing.
            enable_resonance: Enable smoke-pattern resonance analysis.
            enable_enhanced_cnn: Enable the multi-channel wildfire CNN.
            keep_diagnostics: When True, each prediction result carries the
                thermal array, the boolean hotspot mask, and the hotspot pixel
                coordinates the ignition path computed (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Default False keeps memory behavior unchanged.
        """
        self.enable_ignition = enable_ignition_detection
        self.enable_spread = enable_spread_modeling
        self.enable_ndvi = enable_ndvi_processing
        self.enable_resonance = enable_resonance
        self.enable_enhanced_cnn = enable_enhanced_cnn
        self.keep_diagnostics = keep_diagnostics

        self.ignition_detector = FireIgnitionDetector() if self.enable_ignition else None
        self.spread_model = FireSpreadModel() if self.enable_spread else None

        # Enhanced components
        self.ndvi_processor = NDVIProcessor() if enable_ndvi_processing else None
        self.resonance_analyzer = ResonanceFrequencyAnalyzer() if enable_resonance else None
        self.enhanced_cnn = WildfireCNN() if enable_enhanced_cnn else None

        # Anti-theater guard (mirrors SchumannResonanceDetector): the
        # FireIgnitionDetector CNN ships with random weights and no labelled fire
        # corpus exists to train it. Until real weights are loaded via
        # load_neural_weights(), its fire probability is noise, so
        # _detect_ignition derives detection from the OBSERVED thermal field
        # instead, using VIIRS/MODIS-style brightness-temperature criteria
        # (absolute threshold + contextual contrast against the background).
        self._neural_trained = False
        self._warned_untrained = False

        # Input contract of a loaded trained checkpoint (see
        # omni_mercury_engine.ml.hazard_training.wildfire_ignition): the
        # shipped weights consume FIRMS-detection-derived rasters and emit a
        # NEXT-DAY fire-activity probability. None until a checkpoint that
        # declares a spec is loaded.
        self._feature_spec: str | None = None

        # Deployed alert threshold on the CNN fire probability. A trained
        # checkpoint may carry a validation-selected operating point (see
        # wildfire_ignition._select_operating_point) because a pos-weighted
        # BCE sigmoid's scale is not calibrated to the historical fixed 0.6;
        # the threshold governs the ALERT decision only -- confidence remains
        # the raw sigmoid probability.
        self._fire_prob_threshold: float = FIRE_PROB_THRESHOLD_DEFAULT

        self.logger = logging.getLogger(__name__)

    def load_neural_weights(self, checkpoint_path: str | None = None) -> None:
        """Load trained weights for the ignition detector (and optional CNN).

        Until this is called the networks are untrained and ignition detection
        runs on the deterministic thermal physics.

        The shipped default (``wildfire_firms``) is trained for next-day fire
        activity FORECASTING: its input is a 3-channel 32x32 raster of NASA
        FIRMS VIIRS detection-derived fields from days <= t (today's max
        brightness temperature, log1p 3-day FRP, log1p 7-day count), and its
        confidence is the probability of a confirmed vegetation-fire detection
        in the center 2x2 cells on day t+1 -- not a same-scene imagery
        classification. The checkpoint's ``feature_spec`` records this
        contract.

        A checkpoint may also carry an ``operating_point`` with a
        validation-selected ``fire_prob_threshold`` replacing the default
        alert rule ``probability > 0.6`` (the training sigmoid's scale is not
        calibrated to that fixed constant). The threshold changes the ALERT
        decision only; the reported confidence stays the raw probability. An
        operating point that is not a probability in (0, 1) raises rather
        than installing a nonsensical alert rule.

        Args:
            checkpoint_path: Path to a torch checkpoint containing an
                ``ignition_detector`` (and optionally ``enhanced_cnn``) state
                dict. ``None`` loads the shipped default checkpoint
                (``wildfire_firms``), whose provenance sidecar is logged;
                missing or corrupt files raise instead of degrading silently.
        """
        if self.ignition_detector is None:
            raise RuntimeError("ignition detection is disabled on this detector")
        if checkpoint_path is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            checkpoint, _provenance = load_shipped_checkpoint("wildfire_firms")
            source = "shipped default 'wildfire_firms'"
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            source = checkpoint_path
        self.ignition_detector.load_state_dict(checkpoint["ignition_detector"])
        if self.enhanced_cnn is not None and "enhanced_cnn" in checkpoint:
            self.enhanced_cnn.load_state_dict(checkpoint["enhanced_cnn"])
        self._feature_spec = (
            str(checkpoint["feature_spec"]) if "feature_spec" in checkpoint else None
        )
        op = checkpoint.get("operating_point")
        if op is not None:
            tau = float(op["fire_prob_threshold"])
            if not np.isfinite(tau) or not (0.0 < tau < 1.0):
                raise ValueError(
                    f"checkpoint operating-point threshold {tau} is not a "
                    "probability in (0, 1); refusing a nonsensical fire-alert rule"
                )
            self._fire_prob_threshold = tau
        else:
            self._fire_prob_threshold = FIRE_PROB_THRESHOLD_DEFAULT
        self._neural_trained = True
        self.logger.info(
            "Wildfire neural weights loaded from %s (feature spec: %s; alert "
            "threshold: %.4f%s); using learned detector",
            source,
            self._feature_spec or "unspecified",
            self._fire_prob_threshold,
            " from checkpoint operating point" if op is not None else " default",
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the untrained NN is bypassed for physics."""
        if not self._warned_untrained:
            self.logger.warning(
                "WildfireDetector's FireIgnitionDetector is untrained (no checkpoint "
                "loaded); detecting from VIIRS-style brightness-temperature physics "
                "instead of the CNN. Call load_neural_weights() once a trained "
                "checkpoint exists."
            )
            self._warned_untrained = True

    def predict_wildfire(self, wildfire_data: dict[str, Any]) -> WildfirePredictionResult:
        """Comprehensive wildfire prediction.

        Args:
            wildfire_data: Thermal imagery, weather, vegetation data

        Returns:
            Wildfire prediction with risk assessment
        """
        result = WildfirePredictionResult(
            fire_detected=False,
            confidence=0.0,
            risk_level="low",
        )

        if self.enable_ignition and "thermal_image" in wildfire_data:
            ignition_result = self._detect_ignition(wildfire_data["thermal_image"])
            result.fire_detected = ignition_result["fire_detected"]
            result.confidence = ignition_result["confidence"]
            result.thermal_hotspots = ignition_result["hotspot_count"]

            # Populate the (previously never-filled) ignition fields from the
            # REAL hotspot mask: pixel-space (row, col) component centroids, and
            # the hotspot-area estimate only when a ground resolution is known.
            # (.get: tolerate an ignition path that computed no pixel geometry.)
            centroids = ignition_result.get("ignition_centroids", ())
            result.ignition_locations = [(float(r), float(c)) for r, c in centroids]
            pixel_size_km = wildfire_data.get("pixel_size_km")
            if pixel_size_km is not None:
                result.fire_perimeter_km2 = float(
                    ignition_result["hotspot_count"] * float(pixel_size_km) ** 2
                )

            # Capture only when the ignition path actually produced the spatial
            # intermediates; diagnostics stay transparently absent otherwise instead
            # of crashing or fabricating empty arrays.
            if self.keep_diagnostics and "hotspot_mask" in ignition_result:
                context: dict[str, Any] = {
                    "hotspot_threshold_k": HOTSPOT_THRESHOLD_K,
                    "hotspot_count": int(ignition_result["hotspot_count"]),
                    "coordinate_space": "pixel",
                }
                if pixel_size_km is not None:
                    context["pixel_size_km"] = float(pixel_size_km)
                result.diagnostics = HazardDiagnostics(
                    hazard="wildfire",
                    arrays={
                        "thermal_image_k": ignition_result["thermal_image_k"],
                        "hotspot_mask": ignition_result["hotspot_mask"],
                        "ignition_pixels": ignition_result["ignition_pixels"],
                        "ignition_centroids": ignition_result["ignition_centroids"],
                        "ignition_component_sizes": ignition_result["ignition_component_sizes"],
                    },
                    context=context,
                )

        if self.enable_spread and "weather_data" in wildfire_data and self.spread_model is not None:
            spread_result = self.spread_model.predict_spread(
                wildfire_data.get("fire_data", {}), wildfire_data["weather_data"]
            )
            result.spread_rate_km_hr = spread_result["spread_rate_km_hr"]
            result.spread_direction_deg = spread_result["spread_direction_deg"]

        result.risk_level = self._assess_fire_risk(wildfire_data, result)
        result.early_warning_actions = self._generate_warnings(result)

        return result

    def _detect_ignition(self, thermal_image: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect fire ignition and locate hotspot pixels.

        Uses the trained CNN only when real weights have been loaded
        (:meth:`load_neural_weights`); otherwise falls back to the deterministic
        brightness-temperature physics so an untrained network can never
        fabricate (or mask) a fire. Both paths emit the same REAL per-pixel
        intermediates.

        Args:
            thermal_image: Brightness-temperature image in Kelvin, ``(H, W)`` or
                ``(C, H, W)``.

        Returns:
            Dict with the fire decision plus the REAL per-pixel intermediates:
            ``thermal_image_k`` (channel-max 2-D Kelvin map), ``hotspot_mask``
            (2-D bool, > :data:`HOTSPOT_THRESHOLD_K` in any channel),
            ``ignition_pixels`` (K, 2) int (row, col), ``ignition_centroids``
            (C, 2) connected-component centroids, and
            ``ignition_component_sizes`` (C,) pixel counts.
        """
        empty = {
            "fire_detected": False,
            "confidence": 0.0,
            "hotspot_count": 0,
            "thermal_image_k": np.zeros((0, 0)),
            "hotspot_mask": np.zeros((0, 0), dtype=bool),
            "ignition_pixels": np.zeros((0, 2), dtype=np.intp),
            "ignition_centroids": np.zeros((0, 2)),
            "ignition_component_sizes": np.zeros((0,), dtype=np.intp),
        }
        if self.ignition_detector is None:
            return empty

        if not self._neural_trained:
            self._warn_untrained_once()
            return self._detect_ignition_physics(thermal_image)

        if len(thermal_image.shape) == 2:
            thermal_image = thermal_image.reshape(1, 1, *thermal_image.shape)
        elif len(thermal_image.shape) == 3:
            thermal_image = thermal_image.reshape(1, *thermal_image.shape)

        thermal_tensor = torch.tensor(thermal_image, dtype=torch.float32)

        self.ignition_detector.eval()
        with torch.no_grad():
            fire_prob = self.ignition_detector(thermal_tensor)

        # Alert decision at the deployed threshold: the checkpoint's
        # validation-selected operating point when one was loaded, else the
        # historical fixed default. Confidence stays the raw probability.
        fire_detected = float(fire_prob[0].item()) > self._fire_prob_threshold

        # Spatial hotspot mask over the (1, C, H, W) array: a pixel is a hotspot
        # candidate when ANY channel exceeds the threshold; the 2-D thermal map
        # is the channel-max brightness temperature.
        thermal_2d = np.asarray(thermal_image, dtype=float).max(axis=(0, 1))
        hotspot_mask = np.asarray(thermal_image > HOTSPOT_THRESHOLD_K).any(axis=(0, 1))
        ignition_pixels = np.argwhere(hotspot_mask)
        # SPATIAL hotspot pixel count from the mask. A channel-summed
        # exceedance count (the previous formula) counts one pixel up to C
        # times when several bands exceed the threshold, which inflated the
        # ground-area estimate fire_perimeter_km2 = count * pixel_size_km^2.
        hotspot_count = int(hotspot_mask.sum())

        # Connected hotspot regions (8-connectivity) -> one ignition location
        # per region: its pixel-space centroid, with the region's pixel count.
        if hotspot_mask.any():
            labels, n_components = ndimage.label(hotspot_mask, structure=np.ones((3, 3)))
            component_ids = list(range(1, n_components + 1))
            centroids = np.asarray(
                ndimage.center_of_mass(hotspot_mask, labels, component_ids), dtype=float
            ).reshape(-1, 2)
            sizes = np.asarray(
                ndimage.sum_labels(hotspot_mask, labels, component_ids), dtype=np.intp
            )
        else:
            centroids = np.zeros((0, 2))
            sizes = np.zeros((0,), dtype=np.intp)

        return {
            "fire_detected": fire_detected,
            "confidence": float(fire_prob[0].item()),
            "hotspot_count": hotspot_count,
            "thermal_image_k": thermal_2d,
            "hotspot_mask": hotspot_mask,
            "ignition_pixels": ignition_pixels,
            "ignition_centroids": centroids,
            "ignition_component_sizes": sizes,
        }

    @staticmethod
    def _detect_ignition_physics(thermal_image: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Deterministic active-fire detection from brightness temperature (K).

        VIIRS/MODIS-style contextual criteria: a pixel is fire either absolutely
        (brightness temperature above ~360 K, unambiguous combustion) or
        contextually (above ~330 K AND standing far above the scene background,
        median + 4·MAD-scale). Confidence grows with both the peak temperature
        above 330 K and the fraction of hot pixels. Deterministic: identical
        input → identical output.

        Emits the same REAL per-pixel intermediates as the CNN path (channel-max
        Kelvin map, any-channel hotspot mask, hotspot pixels, connected-region
        centroids and sizes) so the ignition fields and the keep_diagnostics
        capture work identically on the physics fallback.
        """
        arr = np.asarray(thermal_image, dtype=float)
        values = arr[np.isfinite(arr)] if arr.size else arr.ravel()
        empty_arrays: dict[str, np.ndarray[Any, Any]] = {
            "thermal_image_k": np.zeros((0, 0)),
            "hotspot_mask": np.zeros((0, 0), dtype=bool),
            "ignition_pixels": np.zeros((0, 2), dtype=np.intp),
            "ignition_centroids": np.zeros((0, 2)),
            "ignition_component_sizes": np.zeros((0,), dtype=np.intp),
        }
        if values.size == 0:
            return {
                "fire_detected": False,
                "confidence": 0.0,
                "hotspot_count": 0,
                **empty_arrays,
            }
        max_temp = float(np.max(values))
        background = float(np.median(values))
        mad = float(np.median(np.abs(values - background)))
        scale = 1.4826 * mad if mad > 0 else max(float(np.std(values)), 1.0)
        contextual = max_temp > 330.0 and (max_temp - background) > 4.0 * scale
        absolute = max_temp > 360.0
        fire_detected = absolute or contextual
        temp_severity = float(np.clip((max_temp - 330.0) / 60.0, 0.0, 1.0))
        hot_fraction = float(np.clip(np.mean(values > HOTSPOT_THRESHOLD_K) * 20.0, 0.0, 1.0))
        confidence = float(1.0 - (1.0 - temp_severity) * (1.0 - hot_fraction))

        # Spatial intermediates, mirroring the CNN path: the 2-D thermal map is
        # the channel-max brightness temperature and a pixel is a hotspot when
        # ANY channel exceeds the threshold. Non-2-D inputs (bare series) carry
        # no pixel geometry, so the spatial arrays stay transparently empty.
        thermal_2d = empty_arrays["thermal_image_k"]
        hotspot_mask = empty_arrays["hotspot_mask"]
        if arr.ndim >= 2:
            stacked = arr.reshape(-1, *arr.shape[-2:])
            thermal_2d = stacked.max(axis=0)
            hotspot_mask = np.asarray((stacked > HOTSPOT_THRESHOLD_K).any(axis=0))
        ignition_pixels = np.argwhere(hotspot_mask)
        # SPATIAL hotspot pixel count from the mask (a pixel hot in several
        # bands is still ONE ground pixel; a channel-summed exceedance count
        # inflated fire_perimeter_km2 by up to a factor of C). A bare 1-D
        # series has no pixel geometry, so the flat exceedance count is the
        # only meaningful number there.
        hotspot_count = (
            int(hotspot_mask.sum()) if arr.ndim >= 2 else int(np.sum(values > HOTSPOT_THRESHOLD_K))
        )

        # Connected hotspot regions (8-connectivity) -> one ignition location
        # per region: its pixel-space centroid, with the region's pixel count.
        if hotspot_mask.any():
            labels, n_components = ndimage.label(hotspot_mask, structure=np.ones((3, 3)))
            component_ids = list(range(1, n_components + 1))
            centroids = np.asarray(
                ndimage.center_of_mass(hotspot_mask, labels, component_ids), dtype=float
            ).reshape(-1, 2)
            sizes = np.asarray(
                ndimage.sum_labels(hotspot_mask, labels, component_ids), dtype=np.intp
            )
        else:
            centroids = np.zeros((0, 2))
            sizes = np.zeros((0,), dtype=np.intp)

        return {
            "fire_detected": fire_detected,
            "confidence": confidence if fire_detected else min(confidence, 0.5),
            "hotspot_count": hotspot_count,
            "method": "physics_brightness_temperature",
            "thermal_image_k": thermal_2d,
            "hotspot_mask": hotspot_mask,
            "ignition_pixels": ignition_pixels,
            "ignition_centroids": centroids,
            "ignition_component_sizes": sizes,
        }

    def _assess_fire_risk(
        self, wildfire_data: dict[str, Any], result: WildfirePredictionResult
    ) -> str:
        """Assess fire risk level."""
        risk_score = 0.0

        if result.fire_detected:
            risk_score += 0.4

        weather = wildfire_data.get("weather_data", {})
        if weather.get("wind_speed_kmh", 0) > 30:
            risk_score += 0.2
        if weather.get("relative_humidity_pct", 100) < 20:
            risk_score += 0.2
        if weather.get("temperature_c", 0) > 35:
            risk_score += 0.2

        if risk_score > 0.8:
            return FireRiskLevel.EXTREME.value
        elif risk_score > 0.6:
            return FireRiskLevel.VERY_HIGH.value
        elif risk_score > 0.4:
            return FireRiskLevel.HIGH.value
        elif risk_score > 0.2:
            return FireRiskLevel.MODERATE.value
        else:
            return FireRiskLevel.LOW.value

    def _generate_warnings(self, result: WildfirePredictionResult) -> list[str]:
        """Generate early warnings."""
        warnings = []

        if result.risk_level in ["extreme", "very_high"]:
            warnings.append("EXTREME FIRE DANGER - Evacuations may be required")
            warnings.append("Activate emergency response teams")
        elif result.risk_level == "high":
            warnings.append("High fire danger - Prepare for rapid response")

        return warnings
