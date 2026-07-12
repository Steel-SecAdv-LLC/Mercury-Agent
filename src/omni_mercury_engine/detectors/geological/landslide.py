# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Landslide Detector - Soil Slope Instability Analysis.

Comprehensive soil slope failure detection for humanitarian early warning:
- Slope stability monitoring (rainfall-triggered, seismic-triggered)
- Debris flow prediction
- Soil saturation analysis
- Ground displacement tracking
- Multi-hazard cascade detection (earthquake → landslide → dam failure)

Scope note (avalanche carve-out):
    Snow avalanche forecasting no longer lives here. Snowpack failure is
    governed by weak-layer shear strength, skier/overburden stress and snow
    metamorphism — not by the soil mechanics this module models — so the
    dedicated snow-stability physics (SK38 skier stability index, critical
    new-snow loading, temperature-gradient metamorphism, rain-on-snow) is in
    :mod:`omni_mercury_engine.detectors.geological.avalanche_detector`.
    ``LandslideType.SNOW_AVALANCHE`` is retained only as a legacy class label
    of the neural type classifier; new avalanche assessments must use
    :class:`~omni_mercury_engine.detectors.geological.avalanche_detector.
    AvalancheDetector`.

Integrations:
- Weather data (rainfall intensity, snowmelt)
- Seismic triggers (earthquake-induced failures)
- Topographic analysis (slope angle, aspect)
- Soil moisture sensors
- InSAR deformation monitoring
- Resilience framework for cascade hazards

Research sources:
- USGS Landslide Hazards Program
- NASA Landslide Viewer
- Swiss Federal Institute for Snow and Avalanche Research (SLF)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy import signal
from torch import nn

from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics


class LandslideRiskLevel(Enum):
    """Landslide risk classifications."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class LandslideType(Enum):
    """Types of slope failures."""

    DEBRIS_FLOW = "debris_flow"
    ROCK_SLIDE = "rock_slide"
    EARTH_FLOW = "earth_flow"
    SNOW_AVALANCHE = "snow_avalanche"
    MUD_FLOW = "mud_flow"
    ROTATIONAL_SLIDE = "rotational_slide"


@dataclass
class LandslidePredictionResult:
    """Landslide prediction results."""

    landslide_imminent: bool
    confidence: float
    risk_level: str
    landslide_type: str

    slope_failure_probability: float = 0.0
    time_to_failure_hours: float | None = None

    rainfall_trigger: bool = False
    seismic_trigger: bool = False
    snowmelt_trigger: bool = False

    soil_saturation_pct: float | None = None
    slope_angle_deg: float | None = None
    displacement_rate_mm_day: float | None = None

    affected_area_km2: float | None = None
    runout_distance_km: float | None = None

    evacuation_zones: list[str] = field(default_factory=list)
    early_warning_actions: list[str] = field(default_factory=list)
    cascade_risks: list[str] = field(default_factory=list)

    # Populated only when the detector was built with keep_diagnostics=True.
    # NOTE: the landslide detector computes no zonal/geographic output (its
    # evacuation zones are string labels); the diagnostics carry the failure
    # TYPE probability distribution the argmax previously discarded.
    diagnostics: HazardDiagnostics | None = None


class RainfallTriggerModel:
    """Rainfall-induced landslide trigger analysis.

    Uses intensity-duration thresholds and antecedent rainfall.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def assess_rainfall_trigger(self, rainfall_data: dict[str, Any]) -> dict[str, Any]:
        """Assess rainfall-induced landslide risk.

        Args:
            rainfall_data: Rainfall intensity, duration, antecedent precipitation

        Returns:
            Rainfall trigger assessment
        """
        intensity_mm_hr = rainfall_data.get("intensity_mm_hr", 0.0)
        duration_hours = rainfall_data.get("duration_hours", 0.0)
        antecedent_7day_mm = rainfall_data.get("antecedent_7day_mm", 0.0)

        critical_intensity = 10.0
        critical_duration = 6.0

        id_threshold = intensity_mm_hr * (duration_hours**0.5)
        critical_id = critical_intensity * (critical_duration**0.5)

        rainfall_trigger = id_threshold > critical_id

        saturation_boost = antecedent_7day_mm / 100.0
        trigger_probability = min((id_threshold / critical_id) * (1 + saturation_boost), 1.0)

        if trigger_probability > 0.8:
            severity = "extreme"
        elif trigger_probability > 0.6:
            severity = "high"
        elif trigger_probability > 0.4:
            severity = "moderate"
        else:
            severity = "low"

        return {
            "rainfall_trigger": rainfall_trigger,
            "trigger_probability": float(trigger_probability),
            "severity": severity,
            "id_threshold": float(id_threshold),
        }


class SeismicTriggerModel:
    """Earthquake-induced landslide trigger analysis.

    Uses peak ground acceleration (PGA) and slope characteristics.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def assess_seismic_trigger(self, seismic_data: dict[str, Any]) -> dict[str, Any]:
        """Assess earthquake-induced landslide risk.

        Args:
            seismic_data: PGA, magnitude, epicentral distance

        Returns:
            Seismic trigger assessment
        """
        pga_g = seismic_data.get("pga_g", 0.0)
        magnitude = seismic_data.get("magnitude", 0.0)
        distance_km = seismic_data.get("distance_km", 100.0)

        critical_pga = 0.15

        seismic_trigger = pga_g > critical_pga

        magnitude_factor = max(magnitude - 5.0, 0) / 3.0
        distance_factor = max(1.0 - (distance_km / 50.0), 0)

        trigger_probability = min(
            (pga_g / critical_pga) * (1 + magnitude_factor) * (1 + distance_factor), 1.0
        )

        return {
            "seismic_trigger": seismic_trigger,
            "trigger_probability": float(trigger_probability),
            "pga_g": float(pga_g),
        }


class SlopeStabilityModel(nn.Module):
    """Neural network for slope stability assessment.

    Integrates topography, soil properties, and hydrological conditions.
    """

    def __init__(self, input_dim: int = 64) -> None:
        """Initialize the instance."""
        super().__init__()

        phi = 1.618

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, int(128 * phi)),
            nn.BatchNorm1d(int(128 * phi)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(128 * phi), int(64 * phi)),
            nn.BatchNorm1d(int(64 * phi)),
            nn.ReLU(),
            nn.Linear(int(64 * phi), 64),
        )

        self.stability_predictor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        self.type_classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))

    def forward(self, slope_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict slope failure probability and type.

        Args:
            slope_features: Integrated slope characteristics

        Returns:
            Tuple of (failure_probability, type_logits)
        """
        features = self.feature_encoder(slope_features)

        failure_prob = self.stability_predictor(features)
        type_logits = self.type_classifier(features)

        return failure_prob, type_logits


class RecursionMultiScaleAnalyzer:
    """3R Recursion mechanism for multi-scale landslide analysis.

    Implements hierarchical feature extraction at multiple temporal scales
    to capture both rapid onset (debris flows) and slow-moving (earth flows)
    landslide patterns.

    Synapse: Integrates with GOSNN for ethical gating and scalar registration.
    """

    def __init__(
        self,
        scales: list[int] | None = None,
        phi: float = 1.618033988749895,
    ):
        """Initialize multi-scale analyzer.

        Args:
            scales: Temporal scales for analysis (default: [1, 4, 16, 64] hours)
            phi: Golden ratio for scale weighting
        """
        self.scales = scales or [1, 4, 16, 64]
        self.phi = phi
        self.logger = logging.getLogger(__name__)

        # Scale weights using golden ratio decay
        self._scale_weights = np.array([phi ** (-i) for i in range(len(self.scales))])
        self._scale_weights /= self._scale_weights.sum()

    def extract_multi_scale_features(
        self,
        time_series: np.ndarray[Any, Any],
        sample_rate_hz: float = 1.0,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Extract features at multiple temporal scales.

        Args:
            time_series: Input time series data (e.g., displacement, rainfall)
            sample_rate_hz: Sampling rate in Hz

        Returns:
            Dictionary with scale-specific features and aggregated recursion score
        """
        features = {}
        scale_scores = []

        for i, scale in enumerate(self.scales):
            # Downsample to scale
            window_size = max(1, int(scale * sample_rate_hz * 3600))
            if len(time_series) < window_size:
                downsampled = time_series
            else:
                downsampled = signal.resample(time_series, len(time_series) // window_size)

            # Extract scale-specific features
            scale_features = self._compute_scale_features(downsampled, scale)
            features[f"scale_{scale}h"] = scale_features

            # Compute variance-based recursion score for this scale
            variance = np.var(scale_features) if len(scale_features) > 1 else 0.0
            recursion_score = 1.0 - variance / (variance + 1.0)
            scale_scores.append(recursion_score * self._scale_weights[i])

        # Aggregate recursion score across scales
        features["recursion_score"] = float(np.sum(scale_scores))  # type: ignore[assignment, unused-ignore]
        features["scale_weights"] = self._scale_weights

        return features

    def _compute_scale_features(
        self,
        data: np.ndarray[Any, Any],
        scale: int,
    ) -> np.ndarray[Any, Any]:
        """Compute features for a specific scale.

        Args:
            data: Downsampled time series
            scale: Temporal scale in hours

        Returns:
            Feature vector for this scale
        """
        if len(data) < 2:
            return np.zeros(8)

        features = np.zeros(8)
        features[0] = np.mean(data)
        features[1] = np.std(data)
        features[2] = np.max(data) - np.min(data)  # Range
        features[3] = np.percentile(data, 90) - np.percentile(data, 10)  # IQR-like

        # Trend features
        if len(data) > 2:
            diff = np.diff(data)
            features[4] = np.mean(diff)  # Average rate of change
            features[5] = np.std(diff)  # Volatility
            features[6] = np.sum(diff > 0) / len(diff)  # Fraction increasing
            features[7] = np.max(np.abs(diff))  # Max change

        return features


class TemporalLagFeatureExtractor:
    """Extract temporal lag features for landslide prediction.

    Captures delayed effects of rainfall and seismic events on slope stability.
    """

    def __init__(
        self,
        lag_hours: list[int] | None = None,
    ):
        """Initialize temporal lag extractor.

        Args:
            lag_hours: Lag periods in hours (default: [1, 6, 12, 24, 48, 72])
        """
        self.lag_hours = lag_hours or [1, 6, 12, 24, 48, 72]
        self.logger = logging.getLogger(__name__)

    def extract_lag_features(
        self,
        time_series: np.ndarray[Any, Any],
        sample_rate_hz: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Extract lag features from time series.

        Args:
            time_series: Input time series
            sample_rate_hz: Sampling rate in Hz

        Returns:
            Feature vector with lag correlations and cumulative values
        """
        n_lags = len(self.lag_hours)
        features = np.zeros(n_lags * 3)  # 3 features per lag

        for i, lag_h in enumerate(self.lag_hours):
            lag_samples = int(lag_h * 3600 * sample_rate_hz)

            if lag_samples >= len(time_series):
                continue

            # Lagged value
            features[i * 3] = time_series[-lag_samples] if lag_samples > 0 else time_series[-1]

            # Cumulative sum over lag period
            features[i * 3 + 1] = np.sum(time_series[-lag_samples:])

            # Correlation with current
            if lag_samples > 1:
                current = time_series[-lag_samples:]
                lagged = time_series[:-lag_samples]
                min_len = min(len(current), len(lagged))
                if min_len > 1:
                    corr = np.corrcoef(current[:min_len], lagged[:min_len])[0, 1]
                    features[i * 3 + 2] = corr if not np.isnan(corr) else 0.0

        return features


class SVMRFEnsembleClassifier:
    """Ensemble classifier combining SVM and Random Forest for landslide detection.

    Provides robust classification by combining:
    - SVM: Good for high-dimensional feature spaces
    - Random Forest: Handles non-linear relationships and provides feature importance
    """

    def __init__(
        self,
        svm_kernel: str = "rbf",
        rf_n_estimators: int = 100,
        ensemble_weights: tuple[float, float] = (0.4, 0.6),
    ):
        """Initialize ensemble classifier.

        Args:
            svm_kernel: SVM kernel type ('rbf', 'linear', 'poly')
            rf_n_estimators: Number of trees in Random Forest
            ensemble_weights: Weights for (SVM, RF) predictions
        """
        try:
            from omni_mercury_engine.ml.mercury_ml import (
                SVC,
                RandomForestClassifier,
                StandardScaler,
            )
        except ImportError as e:
            raise ImportError(
                "Mercury ML utilities not available. Ensure mercury_ml.py is installed."
            ) from e

        self.svm = SVC(kernel=svm_kernel, probability=True, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42)
        self.scaler = StandardScaler()
        self.ensemble_weights = ensemble_weights
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fit(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
    ) -> SVMRFEnsembleClassifier:
        """Fit both classifiers on training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for chaining
        """
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self.rf.fit(X_scaled, y)
        self.is_fitted = True
        self.logger.info(f"SVMRFEnsembleClassifier fitted on {len(y)} samples")
        return self

    def predict_proba(
        self,
        X: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Predict class probabilities using ensemble.

        Args:
            X: Input features

        Returns:
            Ensemble probability predictions
        """
        if not self.is_fitted:
            # Return default probabilities if not fitted
            return np.array([[0.5, 0.5]] * len(X))

        X_scaled = self.scaler.transform(X)

        svm_proba = self.svm.predict_proba(X_scaled)
        rf_proba = self.rf.predict_proba(X_scaled)

        # Weighted ensemble
        ensemble_proba = self.ensemble_weights[0] * svm_proba + self.ensemble_weights[1] * rf_proba

        return ensemble_proba

    def get_feature_importance(self) -> np.ndarray[Any, Any]:
        """Get feature importance from Random Forest.

        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            return np.array([])
        return self.rf.feature_importances_


class LandslideDetector:
    """Comprehensive landslide and avalanche detection system.

    Integrates rainfall, seismic, snowmelt triggers with slope stability analysis.

    Enhanced with:
    - SVM/RF ensemble classifiers for robust prediction
    - Temporal lag features for delayed trigger effects
    - 3R Recursion mechanism for multi-scale analysis
    - GOSNN synapse for ethical gating and scalar registration
    """

    def __init__(
        self,
        enable_rainfall_trigger: bool = True,
        enable_seismic_trigger: bool = True,
        enable_stability_model: bool = True,
        enable_ml_ensemble: bool = True,
        enable_recursion: bool = True,
        keep_diagnostics: bool = False,
    ):
        """Initialize the instance.

        Args:
            enable_rainfall_trigger: Enable rainfall intensity-duration triggers.
            enable_seismic_trigger: Enable earthquake-induced trigger analysis.
            enable_stability_model: Enable the neural slope-stability model.
            enable_ml_ensemble: Enable the SVM/RF ensemble classifier.
            enable_recursion: Enable multi-scale recursion analysis.
            keep_diagnostics: When True and slope features are supplied, each
                prediction result carries the failure-type softmax distribution
                the argmax previously discarded (see
                :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`).
                Default False keeps memory behavior unchanged.
        """
        self.enable_rainfall = enable_rainfall_trigger
        self.enable_seismic = enable_seismic_trigger
        self.enable_stability = enable_stability_model
        self.enable_ml_ensemble = enable_ml_ensemble
        self.enable_recursion = enable_recursion
        self.keep_diagnostics = keep_diagnostics

        self.rainfall_model = RainfallTriggerModel() if enable_rainfall_trigger else None
        self.seismic_model = SeismicTriggerModel() if enable_seismic_trigger else None
        self.stability_model = SlopeStabilityModel() if enable_stability_model else None

        # Enhanced ML components
        self.ml_ensemble = SVMRFEnsembleClassifier() if enable_ml_ensemble else None
        self.recursion_analyzer = RecursionMultiScaleAnalyzer() if enable_recursion else None
        self.lag_extractor = TemporalLagFeatureExtractor() if enable_recursion else None

        # Anti-theater guard (mirrors SchumannResonanceDetector): the
        # SlopeStabilityModel ships with random weights and no labelled failure
        # corpus exists to train it. Until real weights are loaded via
        # load_neural_weights(), its failure probability / type outputs are
        # noise, so slope stability is assessed from the OBSERVED geotechnical
        # fields instead -- slope angle, soil saturation, displacement rate, and
        # the trigger states (see _assess_slope_stability_physics).
        self._neural_trained = False
        self._warned_untrained = False

        # Ratified alert operating point carried by trained checkpoints (see
        # landslide_stability.py _select_operating_point, mirroring the
        # tsunami/seismic machinery): the learned path's landslide_imminent
        # decision thresholds the slope-failure probability at a
        # VALIDATION-selected tau, because the fixed 0.6 bar was chosen for
        # the physics probability scale and the BCE-trained stability head
        # lives on its own scale. The loaded threshold is part of the
        # ratified deployed rule -- it ships inside the merit-gated
        # checkpoint and is selected against the same recall/false-alarm
        # constraints the ship gate enforces. None until a checkpoint that
        # declares one is loaded (then the learned path uses it,
        # decision-only; the physics path always keeps the 0.6 bar).
        self._operating_point: dict[str, float] | None = None

        self.logger = logging.getLogger(__name__)

    def load_neural_weights(self, checkpoint_path: str | None = None) -> None:
        """Load trained weights for the slope-stability model.

        Until this is called the network is untrained and stability is assessed
        from the deterministic geotechnical physics.

        Trained checkpoints define the ``slope_features`` input contract:
        the shipped ``landslide_coolr`` checkpoint consumes the RAW 64-dim
        ``landslide-coolr-v1`` vector (train-year standardization is folded
        into its first encoder layer), documented dimension-by-dimension in
        :mod:`omni_mercury_engine.ml.hazard_training.landslide_stability`.

        Trained checkpoints may carry a ratified ``operating_point`` (the
        validation-selected slope-failure probability threshold governing
        the learned path's ``landslide_imminent`` decision -- part of the
        deployed rule the merit gate evaluated). It is consumed
        decision-only: the reported probability is never rescaled.

        Args:
            checkpoint_path: Path to a torch checkpoint containing a
                ``stability_model`` state dict. None loads the shipped
                ``landslide_coolr`` default (sha256-verified against its
                provenance sidecar; missing/corrupt files raise).

        Raises:
            ValueError: If the checkpoint carries an ``operating_point``
                whose detection threshold is not a probability in (0, 1) --
                a nonsensical alert rule must refuse, not load.
        """
        if self.stability_model is None:
            raise RuntimeError("slope-stability modelling is disabled on this detector")
        if checkpoint_path is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            checkpoint, _provenance = load_shipped_checkpoint("landslide_coolr")
            source = "shipped default 'landslide_coolr'"
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            source = checkpoint_path
        # Ratified operating point (validation-selected alert threshold for
        # the learned path -- part of the deployed decision rule the merit
        # gate evaluated). Validated BEFORE any state mutates so a bad rule
        # cannot half-load. Checkpoints that predate the convention simply
        # leave the fixed 0.6 bar in charge.
        op = checkpoint.get("operating_point")
        if op is not None:
            tau = float(op["detection_threshold"])
            if not np.isfinite(tau) or not (0.0 < tau < 1.0):
                raise ValueError(
                    f"checkpoint operating point detection threshold {tau} is not a "
                    "probability; refusing a nonsensical alert rule"
                )
        self.stability_model.load_state_dict(checkpoint["stability_model"])
        if op is not None:
            self._operating_point = {"detection_threshold": float(op["detection_threshold"])}
        else:
            self._operating_point = None
        self._neural_trained = True
        self.logger.info(
            "Landslide neural weights loaded from %s (feature spec: %s); using learned "
            "stability model%s",
            source,
            checkpoint.get("feature_spec", "unspecified"),
            (
                f" (alert operating point tau="
                f"{self._operating_point['detection_threshold']:.4f})"
                if self._operating_point is not None
                else " (no operating point; fixed 0.6 alert bar governs)"
            ),
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the untrained NN is bypassed for physics."""
        if not self._warned_untrained:
            self.logger.warning(
                "LandslideDetector's SlopeStabilityModel is untrained (no checkpoint "
                "loaded); assessing stability from geotechnical physics (slope angle, "
                "saturation, displacement, triggers) instead of the NN. Call "
                "load_neural_weights() once a trained checkpoint exists."
            )
            self._warned_untrained = True

    def predict_landslide(self, landslide_data: dict[str, Any]) -> LandslidePredictionResult:
        """Comprehensive landslide prediction.

        Args:
            landslide_data: Multi-parameter slope monitoring data including:
                - rainfall_data: Intensity, duration, antecedent rainfall
                - seismic_data: PGA, magnitude, distance
                - slope_data: Angle, aspect, soil properties
                - sensor_data: Soil moisture, displacement
                - weather_data: Snowmelt, temperature

        Returns:
            Landslide prediction with risk level and evacuation recommendations
        """
        result = LandslidePredictionResult(
            landslide_imminent=False,
            confidence=0.0,
            risk_level="low",
            landslide_type="debris_flow",
        )

        triggers_detected: float = 0

        if (
            self.enable_rainfall
            and "rainfall_data" in landslide_data
            and self.rainfall_model is not None
        ):
            rainfall_result = self.rainfall_model.assess_rainfall_trigger(
                landslide_data["rainfall_data"]
            )
            result.rainfall_trigger = rainfall_result["rainfall_trigger"]
            if rainfall_result["rainfall_trigger"]:
                triggers_detected += 1
                result.confidence = max(result.confidence, rainfall_result["trigger_probability"])

        if (
            self.enable_seismic
            and "seismic_data" in landslide_data
            and self.seismic_model is not None
        ):
            seismic_result = self.seismic_model.assess_seismic_trigger(
                landslide_data["seismic_data"]
            )
            result.seismic_trigger = seismic_result["seismic_trigger"]
            if seismic_result["seismic_trigger"]:
                triggers_detected += 1
                result.confidence = max(result.confidence, seismic_result["trigger_probability"])

        if "weather_data" in landslide_data:
            snowmelt_rate = landslide_data["weather_data"].get("snowmelt_mm_day", 0.0)
            result.snowmelt_trigger = snowmelt_rate > 20.0
            if result.snowmelt_trigger:
                triggers_detected += 0.5

        # Parse the observed geotechnical fields BEFORE the stability assessment
        # so the physics path can consume them.
        if "sensor_data" in landslide_data:
            result.soil_saturation_pct = landslide_data["sensor_data"].get("soil_saturation_pct")
            result.displacement_rate_mm_day = landslide_data["sensor_data"].get(
                "displacement_rate_mm_day"
            )

        if "slope_data" in landslide_data:
            result.slope_angle_deg = landslide_data["slope_data"].get("slope_angle_deg")

        neural_path_used = False
        if self.enable_stability:
            if self._neural_trained and "slope_features" in landslide_data:
                stability_result = self._assess_slope_stability(landslide_data["slope_features"])
                neural_path_used = True
            else:
                # Physics path: works from the real geotechnical fields, so it
                # runs even without an opaque slope_features vector (previously
                # landslide_imminent could NEVER fire without one).
                self._warn_untrained_once()
                stability_result = self._assess_slope_stability_physics(result)
            result.slope_failure_probability = stability_result["failure_probability"]
            result.landslide_type = stability_result["landslide_type"]
            result.confidence = max(result.confidence, stability_result["failure_probability"])
            # Only the (trained) NN path produces a failure-type softmax; the
            # physics path emits no "type_probs" key, so the .get() guard keeps
            # keep_diagnostics from crashing there and diagnostics stay honestly
            # absent (None) rather than fabricated.
            if self.keep_diagnostics and stability_result.get("type_probs") is not None:
                result.diagnostics = HazardDiagnostics(
                    hazard="landslide",
                    arrays={"failure_type_probs": stability_result["type_probs"]},
                    context={
                        "failure_type_labels": stability_result["type_labels"],
                        "failure_probability": float(stability_result["failure_probability"]),
                    },
                )

        # Deployed alert decision. The learned path thresholds at the
        # checkpoint's ratified operating point when one was loaded (the
        # deployed rule the merit gate evaluated -- see
        # landslide_stability._select_operating_point); the physics path and
        # operating-point-free checkpoints keep the fixed 0.6 bar. The
        # threshold is decision-only: slope_failure_probability is reported
        # unchanged either way.
        alert_bar = 0.6
        if neural_path_used and self._operating_point is not None:
            alert_bar = self._operating_point["detection_threshold"]
        result.landslide_imminent = (
            triggers_detected >= 1 and result.slope_failure_probability > alert_bar
        )

        result.risk_level = self._determine_risk_level(triggers_detected, result)
        result.evacuation_zones = self._identify_evacuation_zones(result)
        result.early_warning_actions = self._generate_warnings(result)
        result.cascade_risks = self._assess_cascade_risks(result, landslide_data)

        return result

    def _assess_slope_stability(self, slope_features: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Assess slope stability using ML model."""
        landslide_types = [
            "debris_flow",
            "rock_slide",
            "earth_flow",
            "snow_avalanche",
            "mud_flow",
            "rotational_slide",
        ]
        if self.stability_model is None:
            return {
                "failure_probability": 0.0,
                "landslide_type": "debris_flow",
                "type_probs": None,
                "type_labels": landslide_types,
            }

        features_tensor = torch.tensor(slope_features, dtype=torch.float32).unsqueeze(0)

        self.stability_model.eval()
        with torch.no_grad():
            failure_prob, type_logits = self.stability_model(features_tensor)

        failure_probability = float(failure_prob[0].item())

        type_probs = torch.softmax(type_logits[0], dim=0)
        type_idx = int(torch.argmax(type_probs).item())

        landslide_type = landslide_types[type_idx]

        return {
            "failure_probability": failure_probability,
            "landslide_type": landslide_type,
            # The full softmax distribution over failure types -- previously
            # reduced to its argmax and discarded.
            "type_probs": type_probs.cpu().numpy().astype(float),
            "type_labels": landslide_types,
        }

    @staticmethod
    def _assess_slope_stability_physics(result: LandslidePredictionResult) -> dict[str, Any]:
        """Deterministic stability assessment from observed geotechnical fields.

        Grounded in standard slope-failure mechanics rather than an untrained
        network: an accelerating displacement rate is the strongest single
        precursor (saturating around 50 mm/day of creep), and the interaction of
        slope angle with soil saturation (both are required -- a saturated flat
        field does not slide, a dry moderate slope rarely does) forms the
        geotechnical term. The two combine with a noisy-OR. Missing fields
        contribute zero severity -- nothing is imputed or fabricated. The failure
        type follows the observed trigger: snowmelt → snow avalanche, seismic on
        a steep slope → rock slide, rainfall on saturated ground → mud/debris
        flow, otherwise earth flow / rotational slide by saturation.
        """
        slope = result.slope_angle_deg or 0.0
        saturation = (result.soil_saturation_pct or 0.0) / 100.0
        displacement = result.displacement_rate_mm_day or 0.0

        slope_severity = float(np.clip((slope - 15.0) / 30.0, 0.0, 1.0))
        displacement_severity = float(np.clip(displacement / 50.0, 0.0, 1.0))
        geotechnical = slope_severity * float(np.clip(saturation, 0.0, 1.0))

        failure_probability = float(1.0 - (1.0 - displacement_severity) * (1.0 - geotechnical))

        if result.snowmelt_trigger:
            landslide_type = "snow_avalanche"
        elif result.seismic_trigger and slope >= 35.0:
            landslide_type = "rock_slide"
        elif result.rainfall_trigger and saturation >= 0.8:
            landslide_type = "mud_flow"
        elif result.rainfall_trigger:
            landslide_type = "debris_flow"
        elif saturation >= 0.6:
            landslide_type = "earth_flow"
        else:
            landslide_type = "rotational_slide"

        return {
            "failure_probability": failure_probability,
            "landslide_type": landslide_type,
            "method": "physics_geotechnical",
        }

    def _determine_risk_level(self, triggers: float, result: LandslidePredictionResult) -> str:
        """Determine overall risk level."""
        if triggers >= 2 and result.slope_failure_probability > 0.8:
            return LandslideRiskLevel.EXTREME.value
        elif triggers >= 1 and result.slope_failure_probability > 0.6:
            return LandslideRiskLevel.VERY_HIGH.value
        elif triggers >= 1 or result.slope_failure_probability > 0.4:
            return LandslideRiskLevel.HIGH.value
        elif result.slope_failure_probability > 0.2:
            return LandslideRiskLevel.MODERATE.value
        else:
            return LandslideRiskLevel.LOW.value

    def _identify_evacuation_zones(self, result: LandslidePredictionResult) -> list[str]:
        """Identify evacuation zones."""
        zones = []

        if result.risk_level in ["extreme", "very_high"]:
            zones.append("immediate_downslope_area")
            zones.append("potential_runout_path")

            if result.landslide_type == "debris_flow":
                zones.append("drainage_channels")
            elif result.landslide_type == "snow_avalanche":
                zones.append("avalanche_path")

        return zones

    def _generate_warnings(self, result: LandslidePredictionResult) -> list[str]:
        """Generate early warnings."""
        warnings = []

        if result.risk_level == "extreme":
            warnings.append("LANDSLIDE WARNING: Immediate evacuation required")
            warnings.append("Close roads in affected areas")
        elif result.risk_level == "very_high":
            warnings.append("LANDSLIDE WATCH: Prepare for evacuation")
            warnings.append("Monitor conditions continuously")
        elif result.risk_level == "high":
            warnings.append("Landslide Advisory: Heightened awareness")

        return warnings

    def _assess_cascade_risks(
        self, result: LandslidePredictionResult, data: dict[str, Any]
    ) -> list[str]:
        """Assess cascade hazard risks."""
        cascades = []

        if result.landslide_imminent:
            if "infrastructure_data" in data:
                if data["infrastructure_data"].get("dam_present"):
                    cascades.append("dam_failure_risk")

            if result.landslide_type == "debris_flow":
                cascades.append("river_blockage_flooding")

            if "population_data" in data:
                if data["population_data"].get("population_density", 0) > 100:
                    cascades.append("high_casualty_potential")

        return cascades
