# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Meteor / near-Earth-object hazard detector (planetary defense).

Bolide and NEO threat assessment from optical and radar channels, graded by a
Bayesian track filter, with live close-approach, fireball and Sentry impact-risk
context from NASA NeoWs and JPL SSD feeds.

This is space science, not geology: the detector moved here from
``detectors/geological/disaster_detectors.py`` (where it sat with the seismic
and tsunami detectors for historical reasons) so the module layout matches the
discipline, joining the solar-storm, CME, SEP, GIC and interstellar-object
detectors. The old module re-exports every public name, so existing imports
keep working; the detector manifest categorises ``meteor`` under ``space``.

Classes:
    MeteorThreatLevel: Torino-style threat ladder for the fused assessment.
    MeteorPredictionResult: Structured per-assessment result.
    BayesianMeteorFilter: Recursive detection-probability filter over the
        optical/radar evidence stream.
    MeteorDetector: The detector: channel physics, the Bayesian filter and the
        live NASA/JPL context, fused into one graded assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

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
    require_live_client,
)

if TYPE_CHECKING:
    from omni_mercury_engine.data_sources.live_ingestion import LiveFetch
from omni_mercury_engine.data_sources.space_weather import NASANeoWsSource
from omni_mercury_engine.detectors._hazard_input import (
    InputGrader,
    InvalidObservationError,
)
from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics
from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.utils.rng import get_global_rng

logger = logging.getLogger(__name__)

#: Feature dimension of the fusion-pipeline vector :meth:`MeteorDetector.extract_features` emits
#: (kept equal to the other disaster detectors' so the fusion layer sees one width).
FEATURE_DIM = 20

__all__ = [
    "BayesianMeteorFilter",
    "MeteorDetector",
    "MeteorPredictionResult",
    "MeteorThreatLevel",
]


class MeteorThreatLevel(Enum):
    """Meteor/asteroid threat classification."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


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
