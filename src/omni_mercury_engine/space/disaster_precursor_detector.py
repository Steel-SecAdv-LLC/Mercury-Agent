# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Disaster Precursor Detector - Schumann Resonance + Multi-Source Correlation.

Advanced disaster early warning using Schumann resonance anomalies:
- Earthquake seismicity-rate forecasting (probabilistic, catalog-statistical;
  see below -- the former "electromagnetic precursor" framing failed review)
- Tsunami early warning (ionospheric perturbations)
- Volcanic eruption precursors
- Severe weather prediction (geomagnetic correlations)
- Climate pattern shifts

Integrates:
- Schumann resonance anomalies
- Seismic data correlation
- Geomagnetic indices
- Ionospheric disturbances
- Cyclic geophysical correlation (lunar tidal / solar cycle, exploratory)

Earthquake path (reframed per docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md):
the ``EarthquakePrecursorAnalyzer`` is a regional seismicity-rate forecaster
trained on real USGS catalog features (``seismicity-catalog-v2``). Its primary
output is P(M>=5.0 within 30 days in a 0.5-degree California cell) -- a
probabilistic rate forecast, dominated by honest Omori/ETAS-style aftershock
clustering, NOT a validated precursor signal and NOT a deterministic
prediction of individual earthquakes (no magnitude, location or
time-of-occurrence claims; Geller et al. 1997; Jordan et al. 2011). The
electromagnetic/Schumann earthquake-prediction interpretation did not survive
peer review and is retired; no EM feature feeds this model.

⚠️ SIMULATION-BASED: For research. NOT a replacement for official warning systems.
Always defer to USGS, NOAA, and national seismological agencies.

Research sources:
- USGS ComCat earthquake catalog (FDSN event service)
- Operational statistical seismicity forecasting (Reasenberg & Jones 1989;
  Gerstenberger et al. 2005; CSEP testing standards)
- Ionospheric precursor research
- NOAA Space Weather Prediction Center
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector


@dataclass
class DisasterPrecursorResult:
    """Disaster precursor prediction results."""

    precursor_detected: bool
    confidence: float
    disaster_type: str
    risk_level: str

    time_to_event_hours: float | None = None
    estimated_magnitude: float | None = None
    affected_region: str | None = None

    schumann_anomaly: dict[str, Any] | None = None
    seismic_correlation: float | None = None
    geomagnetic_indicators: list[str] = field(default_factory=list)
    ionospheric_disturbance: bool = False

    early_warning_actions: list[str] = field(default_factory=list)
    monitoring_recommendations: list[str] = field(default_factory=list)


class EarthquakePrecursorAnalyzer(nn.Module):
    """Regional seismicity-rate forecaster over catalog statistics.

    Reframed per ``docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md``:
    the input is the 128-dim ``seismicity-catalog-v2`` feature vector (USGS
    catalog rates, Reasenberg-Jones triggered rate, b-values, clustering
    statistics, and -- since v2 -- the stacked RJ baseline's own causal
    30-day forecast, dims 32-35; NO electromagnetic/Schumann inputs, which
    failed review).

    Heads (architecture fixed; semantics per the review):

    * ``confidence_head`` -- PRIMARY: P(M>=5.0 event within 30 days in a
      0.5-degree cell), a probabilistic rate forecast.
    * ``magnitude_predictor`` -- DIAGNOSTIC regression of the observed
      maximum magnitude in positive windows (scaled /9). Never a prediction
      of a specific future event's magnitude.
    * ``time_predictor`` -- DIAGNOSTIC regression of observed days to the
      first in-cell M>=4 (scaled /30 d). Never a time-to-event prediction;
      the literature supports no such capability.
    """

    def __init__(self, input_dim: int = 128) -> None:
        """Initialize the instance."""
        super().__init__()

        self.em_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.magnitude_predictor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.time_predictor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forecast from catalog seismicity features (``seismicity-catalog-v2``).

        Note: the extractor attribute keeps its historical name
        ``em_feature_extractor`` so existing checkpoint state-dict keys stay
        stable; its inputs are catalog statistics, never EM measurements.

        Args:
            features: Standardized 128-dim catalog seismicity feature batch.

        Returns:
            Tuple of (diagnostic magnitude regression, diagnostic time
            regression, event probability) -- the probability is the primary
            output; the other heads are diagnostics, never predictions.
        """
        features = self.em_feature_extractor(features)

        magnitude = self.magnitude_predictor(features)
        time_to_event = self.time_predictor(features)
        confidence = self.confidence_head(features)

        return magnitude, time_to_event, confidence


class GeomageticCorrelator:
    """Correlate Schumann anomalies with geomagnetic indices.

    Uses Kp, Dst, and other indices for disaster correlation.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

        self.kp_thresholds = {
            "quiet": (0, 3),
            "unsettled": (3, 4),
            "active": (4, 5),
            "minor_storm": (5, 6),
            "major_storm": (6, 7),
            "severe_storm": (7, 9),
        }

    def correlate_geomagnetic(
        self, schumann_anomaly: dict[str, Any], geomagnetic_data: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """Correlate Schumann anomaly with geomagnetic activity.

        Args:
            schumann_anomaly: Schumann resonance anomaly data
            geomagnetic_data: Observed Kp/Dst indices. When absent (or an
                index is missing) the analysis reports honestly: status
                ``"unknown"``, ``kp_index``/``dst_index`` ``None``, and a
                ``geomagnetic_data_unavailable`` indicator. No quiet-time
                default is ever invented — a fabricated Kp 3.0 / Dst −20
                "quiet" reading is indistinguishable from a measurement
                downstream (this replaced exactly that behavior).

        Returns:
            Correlation analysis
        """
        if geomagnetic_data is None:
            geomagnetic_data = {}

        kp = geomagnetic_data.get("kp_index")
        dst = geomagnetic_data.get("dst_index")

        geomagnetic_status = (
            self._classify_geomagnetic_activity(kp) if kp is not None else "unknown"
        )

        space_weather_factor = 1.0
        if kp is not None:
            if kp > 7.0:
                space_weather_factor = 1.6
            elif kp > 5.0:
                space_weather_factor = 1.3

        dst_disturbance = dst is not None and dst < -50

        correlation_strength = 0.0

        if schumann_anomaly.get("frequency_anomaly") and geomagnetic_status in [
            "minor_storm",
            "major_storm",
            "severe_storm",
        ]:
            correlation_strength += 0.4

        if schumann_anomaly.get("amplitude_anomaly") and dst_disturbance:
            correlation_strength += 0.3

        if schumann_anomaly.get("power_spectrum_shift"):
            correlation_strength += 0.2

        correlation_strength = min(correlation_strength, 1.0)

        indicators = []
        if kp is None and dst is None:
            indicators.append("geomagnetic_data_unavailable")
        if correlation_strength > 0.5:
            indicators.append("strong_geomagnetic_correlation")
        if dst_disturbance:
            indicators.append("ionospheric_current_disturbance")
        if kp is not None and kp > 6:
            indicators.append("severe_space_weather")

        return {
            "correlation_strength": correlation_strength,
            "geomagnetic_status": geomagnetic_status,
            "space_weather_factor": space_weather_factor,
            "indicators": indicators,
            "kp_index": kp,
            "dst_index": dst,
        }

    def _classify_geomagnetic_activity(self, kp: float) -> str:
        """Classify geomagnetic activity level."""
        for status, (min_kp, max_kp) in self.kp_thresholds.items():
            if min_kp <= kp < max_kp:
                return status
        return "severe_storm"


class IonosphericDisturbanceDetector:
    """Detect ionospheric disturbances from Schumann data.

    Ionospheric changes can precede earthquakes and tsunamis.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_ionospheric_disturbance(
        self, schumann_data: dict[str, Any], tec_data: np.ndarray[Any, Any] | None = None
    ) -> dict[str, Any]:
        """Detect ionospheric disturbances.

        Args:
            schumann_data: Schumann resonance analysis
            tec_data: Total Electron Content measurements (optional)

        Returns:
            Ionospheric disturbance analysis
        """
        disturbance_detected = False
        indicators = []

        fundamental_deviation = schumann_data.get("fundamental_deviation", 0.0)

        if fundamental_deviation > 1.0:
            disturbance_detected = True
            indicators.append("significant_frequency_shift")

        if tec_data is not None and len(tec_data) > 1:
            tec_variation = np.std(tec_data)
            tec_trend = np.diff(tec_data)

            if tec_variation > 5.0:
                disturbance_detected = True
                indicators.append("high_tec_variability")

            if len(tec_trend) > 0 and np.mean(tec_trend) < -2.0:
                indicators.append("tec_depletion_observed")

        disturbance_level = len(indicators) / 3.0

        return {
            "disturbance_detected": disturbance_detected,
            "disturbance_level": min(disturbance_level, 1.0),
            "indicators": indicators,
            "potential_precursor": disturbance_detected
            and "significant_frequency_shift" in indicators,
        }


class SeismicCorrelator:
    """Correlate electromagnetic anomalies with seismic activity.

    Cross-references Schumann anomalies with seismic patterns.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def correlate_seismic(
        self, schumann_anomaly: dict[str, Any], seismic_data: np.ndarray[Any, Any] | None = None
    ) -> dict[str, Any]:
        """Correlate Schumann anomaly with seismic activity.

        Args:
            schumann_anomaly: Schumann anomaly data
            seismic_data: Seismic measurements (magnitude series)

        Returns:
            Seismic correlation analysis
        """
        if seismic_data is None or len(seismic_data) == 0:
            return {"correlation": 0.0, "significant": False}

        em_anomaly_strength = schumann_anomaly.get("risk_score", 0.0)

        recent_seismic = seismic_data[-10:] if len(seismic_data) > 10 else seismic_data

        seismic_activity = np.mean(recent_seismic)

        if em_anomaly_strength > 0.6 and seismic_activity > 4.0:
            correlation = min(em_anomaly_strength * (seismic_activity / 7.0), 1.0)
            significant = correlation > 0.7
        else:
            correlation = em_anomaly_strength * 0.5
            significant = False

        return {
            "correlation": correlation,
            "significant": significant,
            "recent_seismic_activity": float(seismic_activity),
            "precursor_likelihood": correlation * 0.8 if significant else 0.0,
        }


class DisasterPrecursorDetector:
    """Comprehensive disaster precursor detection system.

    Integrates Schumann resonance, geomagnetic, ionospheric, and seismic data for multi-modal
    disaster early warning.
    """

    def __init__(
        self,
        enable_earthquake: bool = True,
        enable_tsunami: bool = True,
        enable_geomagnetic: bool = True,
    ):
        """Initialize the instance."""
        self.enable_earthquake = enable_earthquake
        self.enable_tsunami = enable_tsunami
        self.enable_geomagnetic = enable_geomagnetic

        self.schumann_detector = SchumannResonanceDetector(
            sampling_rate=100.0, enable_cycle_correlation=True, golden_ratio_thresholds=True
        )

        self.earthquake_analyzer = EarthquakePrecursorAnalyzer() if enable_earthquake else None
        self.geomagnetic_correlator = GeomageticCorrelator() if enable_geomagnetic else None
        self.ionospheric_detector = IonosphericDisturbanceDetector()
        self.seismic_correlator = SeismicCorrelator()

        # Anti-theater guard (mirrors SchumannResonanceDetector): the
        # EarthquakePrecursorAnalyzer initializes with random weights. Until a
        # real catalog-trained checkpoint is loaded via load_neural_weights(),
        # its outputs are noise, so the earthquake path abstains: no event
        # probability is emitted and estimated_magnitude stays None. Even when
        # trained, this path emits P(M>=5.0 within 30 days in a 0.5-deg cell)
        # -- never a magnitude or time-to-event for a specific future quake
        # (per docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md). The
        # real correlation paths (Schumann risk, geomagnetic, ionospheric,
        # seismic) keep working either way.
        self._neural_trained = False
        self._warned_untrained = False

        # Feature contract carried by trained checkpoints (see
        # omni_mercury_engine.ml.hazard_training.earthquake_precursor):
        # spec name + standardization statistics from the training years.
        # None until a checkpoint that declares them is loaded.
        self._feature_spec: str | None = None
        self._feature_mean: np.ndarray[Any, Any] | None = None
        self._feature_std: np.ndarray[Any, Any] | None = None

        self.logger = logging.getLogger(__name__)

    def load_neural_weights(self, checkpoint_path: str | None = None) -> None:
        """Load trained weights for the earthquake seismicity-rate forecaster.

        Until this is called, the earthquake neural path abstains entirely
        (loud warning, no fabricated outputs). Once loaded, the path emits
        P(M>=5.0 within 30 days in a 0.5-degree cell) as its confidence --
        never a magnitude or time-to-event for a specific future earthquake
        (per ``docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md``).

        Args:
            checkpoint_path: Path to a torch checkpoint containing an
                ``earthquake_analyzer`` state dict. ``None`` loads the shipped
                default checkpoint (``earthquake_precursor_ca``), whose
                provenance sidecar is verified and logged; missing or corrupt
                files raise instead of degrading silently.
        """
        if self.earthquake_analyzer is None:
            raise RuntimeError("earthquake precursor analysis is disabled on this detector")
        if checkpoint_path is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            checkpoint, _provenance = load_shipped_checkpoint("earthquake_precursor_ca")
            source = "shipped default 'earthquake_precursor_ca'"
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            source = checkpoint_path
        self.earthquake_analyzer.load_state_dict(checkpoint["earthquake_analyzer"])
        if "feature_mean" in checkpoint and "feature_std" in checkpoint:
            self._feature_spec = str(checkpoint.get("feature_spec", "unknown"))
            self._feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            self._feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self._neural_trained = True
        self.logger.info(
            "Earthquake seismicity-rate weights loaded from %s (feature spec: %s); "
            "probabilistic 30-day M>=5 forecasting enabled",
            source,
            self._feature_spec or "raw features, no standardization",
        )

    def _warn_untrained_once(self) -> None:
        """Emit a single WARNING that the earthquake neural path is abstaining."""
        if not self._warned_untrained:
            self.logger.warning(
                "EarthquakePrecursorAnalyzer is untrained (no checkpoint loaded); "
                "the earthquake seismicity-rate path abstains -- no event "
                "probability is emitted and estimated_magnitude remains None, "
                "so nothing is fabricated from random weights. Precursor "
                "detection continues from the real Schumann/geomagnetic/"
                "ionospheric/seismic correlations. Call load_neural_weights() "
                "to load the shipped catalog-trained checkpoint."
            )
            self._warned_untrained = True

    def detect_disaster_precursor(self, precursor_data: dict[str, Any]) -> DisasterPrecursorResult:
        """Comprehensive disaster precursor detection.

        Args:
            precursor_data: Multi-source precursor data including:
                - elf_signal: Schumann resonance measurements
                - seismic_data: Optional seismic time series
                - geomagnetic_data: Optional Kp/Dst indices
                - tec_data: Optional ionospheric TEC data
                - temporal_history: Historical ELF measurements
                - seismicity_features: Optional 128-dim catalog feature vector
                  (``seismicity-catalog-v2``); drives the probabilistic
                  earthquake rate forecast and requires no ELF signal.
                  ``em_features`` is accepted as a legacy alias -- the vector
                  was never electromagnetic data once the EM framing was
                  retired (see the module docstring).

        Returns:
            Disaster precursor prediction. When the trained earthquake path
            runs, ``confidence`` carries P(M>=5.0 within 30 days in the
            feature vector's 0.5-degree cell); ``estimated_magnitude`` is
            NEVER populated from the neural model (review-prohibited claim).
        """
        result = DisasterPrecursorResult(
            precursor_detected=False,
            confidence=0.0,
            disaster_type="none",
            risk_level="low",
        )

        seismicity_features = precursor_data.get(
            "seismicity_features", precursor_data.get("em_features")
        )

        elf_signal = precursor_data.get("elf_signal")
        if elf_signal is None:
            # Catalog-only earthquake forecasting needs no ELF signal (the
            # reviewed model consumes catalog statistics exclusively).
            if self.enable_earthquake and seismicity_features is not None:
                if self._neural_trained:
                    eq_forecast = self._predict_earthquake(seismicity_features)
                    result.confidence = eq_forecast["event_probability"]
                    result.disaster_type = "earthquake"
                    result.precursor_detected = eq_forecast["event_probability"] >= 0.5
                    result.risk_level = self._assess_risk_level(result)
                else:
                    self._warn_untrained_once()
                return result
            self.logger.warning("No ELF signal provided")
            return result

        schumann_result = self.schumann_detector.detect_resonance_anomaly(
            elf_signal,
            temporal_history=precursor_data.get("temporal_history"),
            metadata=precursor_data.get("metadata"),
        )

        result.schumann_anomaly = {
            "anomaly_detected": schumann_result.anomaly_detected,
            "anomaly_type": schumann_result.anomaly_type,
            "fundamental_deviation": schumann_result.fundamental_deviation,
            "risk_score": schumann_result.risk_score,
        }

        if schumann_result.anomaly_detected and schumann_result.risk_score > 0.6:
            result.precursor_detected = True
            result.confidence = schumann_result.confidence

        if self.enable_geomagnetic and "geomagnetic_data" in precursor_data:
            if self.geomagnetic_correlator is None:
                raise RuntimeError("Geomagnetic correlator not initialized")
            geo_correlation = self.geomagnetic_correlator.correlate_geomagnetic(
                result.schumann_anomaly, precursor_data["geomagnetic_data"]
            )
            result.geomagnetic_indicators = geo_correlation["indicators"]

            if geo_correlation["correlation_strength"] > 0.6:
                result.confidence = max(result.confidence, geo_correlation["correlation_strength"])

        ionospheric_result = self.ionospheric_detector.detect_ionospheric_disturbance(
            result.schumann_anomaly, precursor_data.get("tec_data")
        )
        result.ionospheric_disturbance = ionospheric_result["disturbance_detected"]

        if ionospheric_result.get("potential_precursor"):
            result.precursor_detected = True
            result.disaster_type = "earthquake_or_volcanic"

        if "seismic_data" in precursor_data:
            seismic_correlation = self.seismic_correlator.correlate_seismic(
                result.schumann_anomaly, precursor_data["seismic_data"]
            )
            result.seismic_correlation = seismic_correlation["correlation"]

            if seismic_correlation["significant"]:
                result.precursor_detected = True
                result.disaster_type = "earthquake"
                result.time_to_event_hours = self._estimate_time_to_event(
                    schumann_result.risk_score, seismic_correlation["correlation"]
                )

        if self.enable_earthquake and seismicity_features is not None:
            if self._neural_trained:
                eq_forecast = self._predict_earthquake(seismicity_features)
                # Probability framing only: confidence carries P(M>=5.0, 30 d).
                # estimated_magnitude / time_to_event_hours are deliberately
                # NOT populated from the neural model -- the review forbids
                # magnitude and time-to-event claims for specific events.
                result.confidence = max(result.confidence, eq_forecast["event_probability"])
                if eq_forecast["event_probability"] >= 0.5:
                    result.precursor_detected = True
                    result.disaster_type = "earthquake"
            else:
                # Fail honest: an untrained network must not fabricate a
                # forecast. Detection still proceeds from the real correlations.
                self._warn_untrained_once()

        if self.enable_tsunami and result.disaster_type == "earthquake":
            if result.estimated_magnitude and result.estimated_magnitude > 6.5:
                result.disaster_type = "earthquake_tsunami_risk"

        result.risk_level = self._assess_risk_level(result)
        result.early_warning_actions = self._generate_early_warning_actions(result)
        result.monitoring_recommendations = self._generate_monitoring_recommendations(result)

        self.logger.info(
            f"Disaster precursor: {result.disaster_type}, "
            f"confidence={result.confidence:.2f}, risk={result.risk_level}"
        )

        return result

    def _predict_earthquake(self, features: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Forecast P(M>=5.0 within 30 days in a 0.5-degree cell).

        Consumes a ``seismicity-catalog-v2`` feature vector (real USGS
        catalog statistics; see
        ``omni_mercury_engine.ml.hazard_training.earthquake_precursor``),
        standardized with the loaded checkpoint's training-years statistics.

        Returns a dict with:

        * ``event_probability`` (also mirrored as ``confidence``) -- PRIMARY:
          the probabilistic 30-day M>=5.0 rate forecast. Skill is expected to
          be dominated by aftershock/foreshock clustering (honest ETAS-style
          skill), not novel precursor detection.
        * ``diagnostic_max_magnitude`` / ``diagnostic_days_to_m4`` --
          DIAGNOSTIC regressions of observables (window max magnitude;
          days to first in-cell M>=4). Per the literature review these are
          NEVER predictions of a specific future event's magnitude or timing
          and must not populate ``estimated_magnitude``/
          ``time_to_event_hours``.

        Raises:
            RuntimeError: If called before :meth:`load_neural_weights` -- an
                untrained network's output would be fabrication.
        """
        if not self._neural_trained:
            raise RuntimeError(
                "EarthquakePrecursorAnalyzer is untrained; refusing to fabricate "
                "a forecast. Call load_neural_weights() first."
            )
        if self.earthquake_analyzer is None:
            raise RuntimeError("Earthquake analyzer not initialized")

        vec = np.asarray(features, dtype=np.float32)
        if self._feature_mean is not None and self._feature_std is not None:
            vec = (vec - self._feature_mean) / self._feature_std
        features_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

        self.earthquake_analyzer.eval()
        with torch.no_grad():
            magnitude, time_to_event, confidence = self.earthquake_analyzer(features_tensor)

        probability = float(confidence[0].item())
        return {
            "event_probability": probability,
            "confidence": probability,
            "diagnostic_max_magnitude": float(magnitude[0].item()) * 9.0,
            "diagnostic_days_to_m4": float(time_to_event[0].item()) * 30.0,
        }

    def _estimate_time_to_event(self, risk_score: float, correlation: float) -> float:
        """Estimate time to potential event."""
        base_time = 48.0

        urgency_factor = (risk_score + correlation) / 2.0

        estimated_hours = base_time * (1.0 - urgency_factor * 0.8)

        return max(estimated_hours, 2.0)

    def _assess_risk_level(self, result: DisasterPrecursorResult) -> str:
        """Assess overall disaster risk level."""
        if not result.precursor_detected:
            return "low"

        if result.confidence > 0.8:
            return "critical"
        elif result.confidence > 0.6:
            return "high"
        elif result.confidence > 0.4:
            return "moderate"
        else:
            return "low"

    def _generate_early_warning_actions(self, result: DisasterPrecursorResult) -> list[str]:
        """Generate early warning actions."""
        actions = []

        if result.risk_level in ["critical", "high"]:
            actions.append("ALERT: Potential disaster precursor detected")
            actions.append("Notify emergency management agencies")
            actions.append("Activate early warning systems")

            if result.disaster_type == "earthquake":
                actions.append("Prepare for seismic event")
                actions.append("Review building safety protocols")

            if "tsunami" in result.disaster_type:
                actions.append("TSUNAMI WARNING: Coastal evacuation may be required")
                actions.append("Activate tsunami warning centers")

        elif result.risk_level == "moderate":
            actions.append("Enhanced monitoring recommended")
            actions.append("Inform disaster preparedness teams")

        return actions

    def _generate_monitoring_recommendations(self, result: DisasterPrecursorResult) -> list[str]:
        """Generate monitoring recommendations."""
        recs = []

        if result.ionospheric_disturbance:
            recs.append("Increase ionospheric monitoring frequency")
            recs.append("Deploy additional TEC measurement stations")

        if result.seismic_correlation and result.seismic_correlation > 0.5:
            recs.append("Intensify seismic network monitoring")
            recs.append("Check seismometer calibrations")

        if result.geomagnetic_indicators:
            recs.append("Monitor space weather conditions")
            recs.append("Track solar activity and CMEs")

        recs.append("Continue Schumann resonance monitoring")
        recs.append("Cross-reference with global monitoring networks")

        return recs
