# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unified Detector Registry - Bridge connecting detectors across domains for fusion.

This module provides a central registry for all anomaly detectors and models,
enabling:
- Unified detector discovery and registration
- Parallel feature extraction across all detectors
- Feature aggregation for the fusion pipeline
- Detector health monitoring and statistics

Architecture:
    DetectorRegistry connects to the following detector categories:

    Base Detectors (detectors/):
        - MercuryAnomalyDetector
        - TemporalAnomalyDetector
        - SpatialAnomalyDetector
        - DimensionalAnalyzer
        - SigmaDirectiveDetector
        - GraphAnomalyDetector

    Domain-Specific Detectors:
        - Economic: FinancialCrisisDetector, FraudDetector
        - Energy: EMPDetector
        - Geological: LandslideDetector, VolcanicEruptionDetector, WildfireDetector
        - Marine: MarineBiodiversityDetector
        - Space: SolarStormDetector, SchumannResonanceDetector, DisasterPrecursorDetector
        - Medical: SepsisDetector, CardiologyPredictor, NeurocriticalCare
        - Security: ThreatDetector, TEMPESTDetector, PSYOPAnalyzer
        - Emergent: EmergentLifeDetector

    Specialized Models (models/):
        - QuantumAnomalyModel
        - AstrophysicalAnomalyModel
        - BiometricAnomalyModel
        - AffectiveAnomalyModel
        - NeuralCognitiveModel
        - ConsciousnessPreservationModel
        - MultiverseOmniEngine
        - NeurosymbolicEngine
        - ChemistryAnomalyDetector
        - ParapsychologyDetector

    Advanced Physics-Inspired Detectors (v1.4.0):
        - SpectralVibrationDetector (GNN/CNN spectral, phonon interactions)
        - AccelerationDynamicsDetector (kinematics, Lyapunov, phase space)
        - UIUXAnomalyDetector (user behavior, engagement, bot detection)
        - PhysicsIntegratedDetector (unified with 3R and GOSNN)

Example:
    >>> from omni_mercury_engine.core.detector_registry import DetectorRegistry
    >>> registry = DetectorRegistry()
    >>> registry.auto_discover()
    >>> features = registry.extract_all_features(data)
    >>> print(f"Collected features from {len(features)} detectors")
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from omni_mercury_engine._compat import HAS_TORCH
from omni_mercury_engine.core.parallel_detectors import (
    DetectorTask,
    OutcomeStatus,
    ParallelDetectorExecutor,
)
from omni_mercury_engine.resilience.api_circuit_breakers import get_detector_breaker
from omni_mercury_engine.resilience.circuit_breaker import CircuitState

if TYPE_CHECKING or HAS_TORCH:
    import torch
else:
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class _CallTicket:
    """Agreement between a registry call's worker and its caller on who reports it.

    The executor abandons a call that overruns its budget but cannot stop the
    thread running it. Before this ticket, the abandoned call was reported as
    a plain failure result with no circuit-breaker failure and no
    ``error_count``, and when its thread finally returned, ``breaker.call``
    counted it as a *success*. A hung detector therefore never tripped its
    breaker: it was relaunched on every call, each launch stranding a thread,
    until the executor saturated and every healthy parallel-safe detector
    failed with ``ExecutorSaturatedError``.

    Exactly one side reports each call: the worker when it finishes first, the
    caller (as a timeout) when it abandons a started call first. A call
    abandoned before it started is skipped if its worker ever picks it up.
    """

    _PENDING, _RUNNING, _FINISHED, _ABANDONED = range(4)

    def __init__(self) -> None:
        """Create a ticket for a call that has not started."""
        self._lock = threading.Lock()
        self._state = self._PENDING

    def start(self) -> bool:
        """Mark the call running; ``False`` when it was abandoned before starting."""
        with self._lock:
            if self._state == self._ABANDONED:
                return False
            self._state = self._RUNNING
            return True

    def finish(self) -> bool:
        """Mark the call finished; ``False`` when the caller already abandoned it."""
        with self._lock:
            if self._state == self._ABANDONED:
                return False
            self._state = self._FINISHED
            return True

    def abandon(self) -> bool:
        """Abandon the call at the deadline.

        Returns:
            ``True`` when the call was running (a timeout the caller must
            report); ``False`` when it had finished (the worker reported it) or
            never started (nothing ran, so nothing failed).
        """
        with self._lock:
            previous = self._state
            if previous == self._FINISHED:
                return False
            self._state = self._ABANDONED
            return previous == self._RUNNING


def _record_breaker_outcome(breaker: Any, error: Exception | None) -> None:
    """Feed one call's outcome to ``breaker`` through its public ``call`` API.

    The detector runs outside ``breaker.call`` so that an abandoned call is
    counted once, as a failure, at the deadline, and its late completion not
    at all. Replaying the outcome through ``call`` records it exactly as a
    wrapped call would. A breaker that opened meanwhile (another thread's
    failures) rejects the replay; the outcome is then dropped, which is what
    an open breaker does with any call.

    Args:
        breaker: The detector's circuit breaker.
        error: The exception the call raised, or ``None`` for a success.
    """

    def _replay() -> None:
        if error is not None:
            raise error

    with contextlib.suppress(Exception):
        breaker.call(_replay)


class DetectorCategory(Enum):
    """Categories for organizing detectors."""

    BASE = "base"
    ECONOMIC = "economic"
    ENERGY = "energy"
    GEOLOGICAL = "geological"
    MARINE = "marine"
    SPACE = "space"
    MEDICAL = "medical"
    SECURITY = "security"
    EMERGENT = "emergent"
    MODEL = "model"
    NEUROSYMBOLIC = "neurosymbolic"
    INTELLIGENCE = "intelligence"
    # New SOTA categories
    VISUAL = "visual"  # Visual anomaly detection (PatchCore, PaDiM, etc.)
    VLM = "vlm"  # Vision-Language Models (AnyAnomaly, LAVAD)
    FOUNDATION = "foundation"  # Foundation model adapters (TimeGPT, Chronos)
    DISTILLATION = "distillation"  # Knowledge distillation methods
    # Advanced Physics-Inspired categories (v1.4.0)
    PHYSICS = "physics"  # Physics-inspired detectors (spectral, dynamics, kinematics)
    UIUX = "uiux"  # UI/UX behavioral anomaly detection
    METEOROLOGICAL = "meteorological"  # Hydro-climate hazards (drought, heatwave, AR, lightning)
    UAV = "uav"  # Uncrewed aerial vehicle telemetry (RADD rules + Mercury ensemble)


class DetectorProtocol(Protocol):
    """Protocol for detector interface compatibility."""

    def extract_features(self, data: Any) -> Any:
        """Extract features from input data."""
        ...

    def predict(self, data: Any) -> dict[str, Any]:
        """Make predictions on input data."""
        ...


@dataclass
class DetectorInfo:
    """Metadata about a registered detector."""

    name: str
    category: DetectorCategory
    instance: Any
    feature_dim: int | None = None
    is_fitted: bool = False
    last_execution_time_ms: float = 0.0
    total_invocations: int = 0
    error_count: int = 0
    description: str = ""
    tags: list[str] = field(default_factory=list)
    modality: str = "record"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "feature_dim": self.feature_dim,
            "is_fitted": self.is_fitted,
            "last_execution_time_ms": self.last_execution_time_ms,
            "total_invocations": self.total_invocations,
            "error_count": self.error_count,
            "description": self.description,
            "tags": self.tags,
            "modality": self.modality,
        }


@dataclass
class FeatureExtractionResult:
    """Result from feature extraction."""

    detector_name: str
    features: np.ndarray[Any, Any] | torch.Tensor | None
    scores: np.ndarray[Any, Any] | None
    execution_time_ms: float
    success: bool
    error: str | None = None

    def to_tensor(self, device: str = "cpu") -> torch.Tensor | None:
        """Convert features to PyTorch tensor."""
        if self.features is None:
            return None
        if isinstance(self.features, torch.Tensor):
            return self.features.to(device)
        return torch.tensor(self.features, dtype=torch.float32, device=device)


#: Seams a manifest entry may declare.  ``"engine"`` means the class is a
#: :class:`~omni_mercury_engine.core.base.BaseDetector` reachable through
#: ``OmniMercuryEngine.enable_detector``; ``"registry"`` means its
#: ``extract_features``/``predict`` yield real output through this registry
#: when fed the entry's declared modality.  An entry with no seams is
#: library-only: its domain API is called directly, and both bridges refuse
#: it loudly instead of wrapping it in theater.
MANIFEST_SEAMS: frozenset[str] = frozenset({"engine", "registry"})

#: Input payloads a detector's declared seams accept.  Measured, not aspired:
#: every entry's seams were driven end-to-end with its modality's data.
MANIFEST_MODALITIES: frozenset[str] = frozenset(
    {
        "series_1d",  # 1-D numeric time series
        "matrix_2d",  # 2-D numeric batch [n, d]
        "rows",  # named physical columns (HazardRowAdapter.COLUMNS)
        "window",  # one time-ordered window over named columns (HazardWindowAdapter)
        "image_batch",  # [B, C, H, W] / [B, H, W, C] images
        "video_frames",  # [T, C, H, W] / [T, H, W, C] frames
        "graph",  # networkx graph
        "trajectory",  # [n, 3] (lat, lon, epoch_s)
        "windows_3d",  # [n, window, features] pre-windowed series batches
        "record",  # typed domain mapping/dataclass
        "event",  # discrete domain events fed over time
        "text",  # string payloads
        "tensor",  # torch tensors on a model-specific contract
    }
)


@dataclass(frozen=True)
class DetectorManifestEntry:
    """Declarative entry describing a discoverable detector.

    Each entry maps a detector name to the module/class that provides it, along with registration
    metadata.  The manifest is iterated by ``auto_discover_detectors`` so new detectors can be added
    as data rather than code.

    ``seams`` declares where the detector is *measured* to work: ``"engine"``
    (``enable_detector`` -> ``fit``/``detect``) and/or ``"registry"``
    (``extract_features``/``predict`` with real output).  The default is no
    seams -- fail closed: an undeclared entry is library-only and neither
    bridge will serve it until its seams are declared and proven.
    ``modality`` names the input its seams were proven with; ``fit_required``
    marks detectors whose registry methods demand ``fit`` first; ``external``
    marks detectors that need an external service or model backend to produce
    output; ``covered_by`` names the manifest adapter entry serving the same
    domain through the numeric seams (see ``detectors/hazard_adapter.py``).
    ``wraps`` names (``module:Class``) the domain detector an adapter entry
    serves, so the catalog shows what is behind every seam.
    """

    name: str
    module_path: str
    class_name: str
    category: DetectorCategory
    description: str
    feature_dim: int | None = None
    tags: list[str] = field(default_factory=list)
    seams: tuple[str, ...] = ()
    modality: str = "record"
    fit_required: bool = False
    external: bool = False
    covered_by: str | None = None
    wraps: str | None = None

    def __post_init__(self) -> None:
        """Refuse unknown seams or modalities instead of mis-routing later."""
        unknown = set(self.seams) - MANIFEST_SEAMS
        if unknown:
            raise ValueError(
                f"Manifest entry {self.name!r} declares unknown seams {sorted(unknown)}; "
                f"known seams: {sorted(MANIFEST_SEAMS)}"
            )
        if self.modality not in MANIFEST_MODALITIES:
            raise ValueError(
                f"Manifest entry {self.name!r} declares unknown modality {self.modality!r}; "
                f"known modalities: {sorted(MANIFEST_MODALITIES)}"
            )


# ---------------------------------------------------------------------------
# Detector Manifest -- single source of truth for auto-discovery
# ---------------------------------------------------------------------------
# To register a new detector, add an entry here.  No other code changes are
# required; ``auto_discover_detectors`` will pick it up automatically.
# ---------------------------------------------------------------------------
DETECTOR_MANIFEST: list[DetectorManifestEntry] = [
    # -- Base detectors -------------------------------------------------------
    DetectorManifestEntry(
        "statistical",
        "omni_mercury_engine.detectors.statistical",
        "MercuryAnomalyDetector",
        DetectorCategory.BASE,
        "Z-score, percentile, MAD-based detection",
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "temporal",
        "omni_mercury_engine.detectors.temporal",
        "TemporalAnomalyDetector",
        DetectorCategory.BASE,
        "Time-series patterns and seasonal anomalies",
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "spatial",
        "omni_mercury_engine.detectors.spatial",
        "SpatialAnomalyDetector",
        DetectorCategory.BASE,
        "Geographic and spatial relationship anomalies",
        seams=("engine", "registry"),
        modality="matrix_2d",
    ),
    DetectorManifestEntry(
        "geo_movement",
        "omni_mercury_engine.detectors.geo_movement",
        "GeoMovementAnomalyDetector",
        DetectorCategory.BASE,
        "Trajectory movement plausibility: velocity, jump, and silence channels",
        feature_dim=8,
        tags=["geospatial", "trajectory"],
        seams=("engine", "registry"),
        modality="trajectory",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "dimensional",
        "omni_mercury_engine.detectors.dimensional",
        "DimensionalAnalyzer",
        DetectorCategory.BASE,
        "High-dimensional data anomalies",
        seams=("engine",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "directive",
        "omni_mercury_engine.detectors.directive",
        "SigmaDirectiveDetector",
        DetectorCategory.BASE,
        "Rule-based sigma detection",
        seams=("engine", "registry"),
        modality="matrix_2d",
    ),
    DetectorManifestEntry(
        "graph_based",
        "omni_mercury_engine.detectors.graph_based",
        "GraphAnomalyDetector",
        DetectorCategory.BASE,
        "Graph structure anomaly detection",
        seams=("engine", "registry"),
        modality="graph",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "kmeans_distance",
        "omni_mercury_engine.detectors.kmeans_distance",
        "KMeansDistanceDetector",
        DetectorCategory.BASE,
        "Distance-to-nearest-centroid unsupervised anomaly signal (revived "
        "cognitive clusterer; ROC-AUC ~0.8-0.98 on ADBench)",
        feature_dim=9,
        tags=["unsupervised", "clustering"],
        seams=("engine", "registry"),
        modality="series_1d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "spectral_residual",
        "omni_mercury_engine.detectors.spectral_residual",
        "SpectralResidualDetector",
        DetectorCategory.BASE,
        "Spectral-Residual saliency detector for streaming time series "
        "(training-free FFT saliency; Ren et al., KDD 2019)",
        feature_dim=1,
        tags=["streaming", "temporal", "spectral"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "bocpd",
        "omni_mercury_engine.detectors.bocpd",
        "BOCPDDetector",
        DetectorCategory.BASE,
        "Bayesian Online Change-Point Detection via run-length posterior "
        "(Gaussian/NIG conjugate model; Adams & MacKay, 2007)",
        feature_dim=1,
        tags=["streaming", "temporal", "change-point", "probabilistic"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "spot_evt",
        "omni_mercury_engine.detectors.spot_evt",
        "SPOTDetector",
        DetectorCategory.BASE,
        "SPOT/DSPOT Peaks-Over-Threshold EVT dynamic-threshold detector with "
        "risk-budgeted false-positive control (Siffer et al., KDD 2017)",
        feature_dim=1,
        tags=["streaming", "statistical", "evt", "thresholding"],
        seams=("engine", "registry"),
        modality="series_1d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "hawkes",
        "omni_mercury_engine.detectors.hawkes",
        "HawkesBurstDetector",
        DetectorCategory.BASE,
        "Hawkes self-exciting point-process event-rate / burst detector for "
        "count streams (exponential-kernel intensity residuals)",
        feature_dim=1,
        tags=["streaming", "temporal", "event-rate", "burst"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "particle_filter",
        "omni_mercury_engine.detectors.particle_filter",
        "ParticleFilterDetector",
        DetectorCategory.BASE,
        "Bootstrap particle-filter state-space residual detector scoring "
        "normalised one-step-ahead predictive innovations",
        feature_dim=1,
        tags=["streaming", "state-space", "tracking", "residual"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "imm",
        "omni_mercury_engine.detectors.imm",
        "IMMDetector",
        DetectorCategory.BASE,
        "Interacting-Multiple-Model switching state-space residual detector "
        "(quiet + manoeuvring Kalman bank; Blom & Bar-Shalom, 1988)",
        feature_dim=1,
        tags=["streaming", "state-space", "tracking", "switching", "residual"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "gaussian_process",
        "omni_mercury_engine.detectors.gaussian_process",
        "GaussianProcessDetector",
        DetectorCategory.BASE,
        "Windowed RBF Gaussian-Process one-step-ahead residual detector with "
        "calibrated predictive variance (Rasmussen & Williams, 2006)",
        feature_dim=1,
        tags=["streaming", "probabilistic", "regression", "uncertainty"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "echo_state",
        "omni_mercury_engine.detectors.echo_state",
        "EchoStateDetector",
        DetectorCategory.BASE,
        "Echo-State-Network reservoir-computing predictive residual detector "
        "(fixed spectral-radius reservoir + ridge readout; Jaeger, 2001)",
        feature_dim=1,
        tags=["streaming", "neuromorphic", "reservoir", "residual"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "spiking",
        "omni_mercury_engine.detectors.spiking",
        "SpikingNetworkDetector",
        DetectorCategory.BASE,
        "Spiking-neural-network (leaky integrate-and-fire) novelty detector "
        "scoring spike-rate divergence from the learned normal regime",
        feature_dim=1,
        tags=["streaming", "neuromorphic", "spiking", "novelty"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "digital_twin",
        "omni_mercury_engine.detectors.digital_twin",
        "DigitalTwinResidualDetector",
        DetectorCategory.BASE,
        "Digital-twin simulation-residual detector: observed-vs-simulated "
        "divergence of an identified AR forward model (Grieves & Vickers, 2017)",
        feature_dim=1,
        tags=["state-space", "simulation", "digital-twin", "residual"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "survival",
        "omni_mercury_engine.detectors.survival",
        "SurvivalHazardDetector",
        DetectorCategory.BASE,
        "Survival / time-to-event hazard detector (Kaplan-Meier baseline + Cox "
        "proportional-hazards deviation) for inter-event-time streams",
        feature_dim=1,
        tags=["probabilistic", "survival", "time-to-event", "hazard"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "energy_based",
        "omni_mercury_engine.detectors.energy_based",
        "EnergyBasedDetector",
        DetectorCategory.BASE,
        "Explicit energy-based model detector: delay-embedding quadratic "
        "(Gaussian-family) energy fitted by score matching; anomaly score is the "
        "calibrated free energy",
        feature_dim=1,
        tags=["generative", "energy-based", "representation", "density"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "deep_svdd",
        "omni_mercury_engine.detectors.deep_svdd",
        "DeepSVDDDetector",
        DetectorCategory.BASE,
        "One-class SVDD hypersphere detector on a fixed random tanh-feature "
        "embedding (a saturating nonlinear lift, not a Fourier map); "
        "distance-to-centre is the anomaly signal (Tax & Duin, 2004)",
        feature_dim=1,
        tags=["generative", "one-class", "representation", "embedding"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "rca",
        "omni_mercury_engine.detectors.rca",
        "RootCauseGraphDetector",
        DetectorCategory.BASE,
        "Root-cause localisation detector: propagates per-node residuals over a "
        "causal/service graph to rank candidate root causes with attribution",
        feature_dim=1,
        tags=["systems", "rca", "causal-graph", "attribution"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "deeplog_sequence",
        "omni_mercury_engine.detectors.deeplog_sequence",
        "DeepLogSequenceDetector",
        DetectorCategory.BASE,
        "DeepLog-style log/sequence detector: next-key surprisal from an n-gram "
        "transition model flags sequences leaving the learned template grammar",
        feature_dim=1,
        tags=["systems", "sequence", "log", "surprisal"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "frequent_pattern",
        "omni_mercury_engine.detectors.frequent_pattern",
        "FrequentPatternDetector",
        DetectorCategory.BASE,
        "Frequent-pattern / association-rule mining detector: flags transactions "
        "violating high-confidence rules mined from normal traces (Apriori)",
        feature_dim=1,
        tags=["systems", "association-rules", "frequent-pattern", "tabular"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "srcnn",
        "omni_mercury_engine.detectors.srcnn",
        "SRCNNDetector",
        DetectorCategory.BASE,
        "SR-CNN: CNN discriminator over Spectral-Residual saliency, trained on "
        "synthetic-anomaly-augmented series (Ren et al., KDD 2019; requires torch)",
        feature_dim=1,
        tags=["streaming", "temporal", "spectral", "cnn", "torch"],
        seams=("engine", "registry"),
        modality="series_1d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "diffusion_ad",
        "omni_mercury_engine.detectors.diffusion_ad",
        "DiffusionReconstructionDetector",
        DetectorCategory.BASE,
        "DDPM-AD: denoising-diffusion reconstruction-error detector; anomaly score "
        "is off-manifold noise-prediction error (Ho et al., 2020; requires torch)",
        feature_dim=1,
        tags=["generative", "diffusion", "reconstruction", "torch"],
        seams=("engine", "registry"),
        modality="series_1d",
        fit_required=True,
    ),
    # -- Specialized models ---------------------------------------------------
    DetectorManifestEntry(
        "quantum",
        "omni_mercury_engine.models.quantum",
        "QuantumAnomalyModel",
        DetectorCategory.MODEL,
        "Quantum state anomalies",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "astrophysical",
        "omni_mercury_engine.models.astrophysical",
        "AstrophysicalAnomalyModel",
        DetectorCategory.MODEL,
        "Cosmic signal anomalies",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "biometric",
        "omni_mercury_engine.models.biometric",
        "BiometricAnomalyModel",
        DetectorCategory.MODEL,
        "Face/biometric anomalies",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "affective",
        "omni_mercury_engine.models.affective",
        "AffectiveAnomalyModel",
        DetectorCategory.MODEL,
        "Emotional state anomalies",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "neural_cognitive",
        "omni_mercury_engine.models.neural",
        "NeuralCognitiveModel",
        DetectorCategory.MODEL,
        "Brain activity anomalies",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "consciousness",
        "omni_mercury_engine.models.consciousness",
        "ConsciousnessPreservationModel",
        DetectorCategory.MODEL,
        "Consciousness preservation analysis",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "neurosymbolic",
        "omni_mercury_engine.models.neurosymbolic",
        "NeurosymbolicEngine",
        DetectorCategory.NEUROSYMBOLIC,
        "Hybrid neural-symbolic reasoning",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "chemistry",
        "omni_mercury_engine.models.chemistry",
        "ChemistryAnomalyDetector",
        DetectorCategory.MODEL,
        "Chemical anomaly detection",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "parapsychology",
        "omni_mercury_engine.models.parapsychology",
        "ParapsychologyDetector",
        DetectorCategory.MODEL,
        "Psi phenomena detection",
        seams=("registry",),
        modality="series_1d",
    ),
    # -- Security detectors ---------------------------------------------------
    DetectorManifestEntry(
        "threat_detection",
        "omni_mercury_engine.security.threat_detection",
        "ThreatDetector",
        DetectorCategory.SECURITY,
        "Security threat detection",
        seams=(),
        modality="text",
    ),
    DetectorManifestEntry(
        "psyop",
        "omni_mercury_engine.security.psyop",
        "PSYOPAnalyzer",
        DetectorCategory.INTELLIGENCE,
        "Psychological operations analysis",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "intelligence_fusion",
        "omni_mercury_engine.security.intelligence_fusion",
        "IntelligenceFusionEngine",
        DetectorCategory.INTELLIGENCE,
        "Multi-source intelligence fusion",
        seams=("registry",),
        modality="record",
    ),
    # -- Space detectors ------------------------------------------------------
    DetectorManifestEntry(
        "schumann_resonance",
        "omni_mercury_engine.space.schumann_resonance",
        "SchumannResonanceDetector",
        DetectorCategory.SPACE,
        "Earth resonance detection",
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "solar_storm",
        "omni_mercury_engine.detectors.adapters.space",
        "SolarStormSnapshotAdapter",
        DetectorCategory.SPACE,
        "Solar storm prediction",
        seams=("engine", "registry"),
        modality="rows",
        wraps="omni_mercury_engine.space.solar_storm_detector:SolarStormDetector",
    ),
    DetectorManifestEntry(
        "solar_flare",
        "omni_mercury_engine.space.solar_storm_detector",
        "SolarFlareDetector",
        DetectorCategory.SPACE,
        "GOES X-ray flare classification with observed-Kp geomagnetic context",
        tags=["disaster", "space-weather", "live-wiring"],
        seams=("registry",),
        modality="series_1d",
        covered_by="xray_flare",
    ),
    # -- Medical detectors ----------------------------------------------------
    DetectorManifestEntry(
        "medical_abms",
        "omni_mercury_engine.medical.abms_disciplines",
        "ABMSDisciplineDetector",
        DetectorCategory.MEDICAL,
        "Medical discipline detection",
        seams=("registry",),
        modality="record",
    ),
    # -- Geological detectors -------------------------------------------------
    DetectorManifestEntry(
        "volcanic",
        "omni_mercury_engine.detectors.geological.volcanic",
        "VolcanicEruptionDetector",
        DetectorCategory.GEOLOGICAL,
        "Volcanic eruption prediction",
        seams=(),
        modality="record",
    ),
    DetectorManifestEntry(
        "landslide",
        "omni_mercury_engine.detectors.geological.landslide",
        "LandslideDetector",
        DetectorCategory.GEOLOGICAL,
        "Landslide prediction",
        seams=(),
        modality="record",
    ),
    DetectorManifestEntry(
        "wildfire",
        "omni_mercury_engine.detectors.geological.wildfire",
        "WildfireDetector",
        DetectorCategory.GEOLOGICAL,
        "Wildfire prediction",
        seams=(),
        modality="record",
    ),
    DetectorManifestEntry(
        "tornado",
        "omni_mercury_engine.detectors.geological.tornado_detector",
        "TornadoDetector",
        DetectorCategory.GEOLOGICAL,
        "Tornado prediction with Doppler radar and FFT resonance analysis",
        feature_dim=20,
        tags=["disaster", "weather", "3r-resonance"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "hurricane",
        "omni_mercury_engine.detectors.geological.hurricane_detector",
        "HurricaneDetector",
        DetectorCategory.GEOLOGICAL,
        "Hurricane/cyclone/typhoon prediction with SST and resonance amplification",
        feature_dim=20,
        tags=["disaster", "weather", "3r-resonance"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "flood",
        "omni_mercury_engine.detectors.geological.flood_detector",
        "FloodDetector",
        DetectorCategory.GEOLOGICAL,
        "Flood prediction with refactoring engine optimization",
        feature_dim=20,
        tags=["disaster", "weather", "3r-refactoring"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "earthquake",
        "omni_mercury_engine.detectors.geological.disaster_detectors",
        "EarthquakeDetector",
        DetectorCategory.GEOLOGICAL,
        "P/S-wave spectrogram + STA/LTA earthquake detection (USGS live catalog wiring)",
        tags=["disaster", "seismic", "live-wiring"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "tsunami",
        "omni_mercury_engine.detectors.geological.disaster_detectors",
        "TsunamiDetector",
        DetectorCategory.GEOLOGICAL,
        "Waveform-spectrum tsunami detection from observed sea-level series",
        tags=["disaster", "marine", "live-wiring"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "meteor",
        "omni_mercury_engine.detectors.geological.disaster_detectors",
        "MeteorDetector",
        DetectorCategory.GEOLOGICAL,
        "Bayesian meteor/NEO threat assessment over NASA NeoWs + JPL fireball/Sentry",
        tags=["disaster", "space", "live-wiring"],
        seams=("registry",),
        modality="series_1d",
    ),
    # -- Meteorological / hydro-climate detectors -----------------------------
    DetectorManifestEntry(
        "drought",
        "omni_mercury_engine.detectors.meteorological.drought_detector",
        "DroughtDetector",
        DetectorCategory.METEOROLOGICAL,
        "SPI/SPEI drought index detector (McKee et al. 1993; Vicente-Serrano et al. 2010)",
        feature_dim=20,
        tags=["disaster", "weather", "drought", "hydro-climate"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "heatwave",
        "omni_mercury_engine.detectors.meteorological.heatwave_detector",
        "HeatwaveDetector",
        DetectorCategory.METEOROLOGICAL,
        "Percentile-climatology heatwave detector with Excess Heat Factor "
        "(Perkins & Alexander 2013; Nairn & Fawcett 2015)",
        feature_dim=20,
        tags=["disaster", "weather", "heatwave", "hydro-climate"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "atmospheric_river",
        "omni_mercury_engine.detectors.meteorological.atmospheric_river_detector",
        "AtmosphericRiverDetector",
        DetectorCategory.METEOROLOGICAL,
        "IVT computation and Ralph et al. (2019) AR-scale classification",
        feature_dim=20,
        tags=["disaster", "weather", "atmospheric-river", "hydro-climate"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "lightning",
        "omni_mercury_engine.detectors.meteorological.lightning_detector",
        "LightningDetector",
        DetectorCategory.METEOROLOGICAL,
        "2-sigma lightning jump severe-weather precursor (Schultz et al. 2009)",
        feature_dim=20,
        tags=["disaster", "weather", "lightning", "hydro-climate"],
        seams=("registry",),
        modality="series_1d",
    ),
    # -- Severe-storm / winter detectors --------------------------------------
    DetectorManifestEntry(
        "hail",
        "omni_mercury_engine.detectors.meteorological.hail_detector",
        "HailDetector",
        DetectorCategory.METEOROLOGICAL,
        "SHIP significant-hail parameter detector (SPC formulation)",
        tags=["disaster", "weather", "hail", "severe-storm"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "winter_storm",
        "omni_mercury_engine.detectors.meteorological.winter_storm_detector",
        "WinterStormDetector",
        DetectorCategory.METEOROLOGICAL,
        "Winter/ice-storm detector (precip-type + FRAM ice accretion)",
        tags=["disaster", "weather", "winter-storm", "severe-storm"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "derecho",
        "omni_mercury_engine.detectors.meteorological.derecho_detector",
        "DerechoDetector",
        DetectorCategory.METEOROLOGICAL,
        "Derecho swath detector (Johns & Hirt 1987 criteria)",
        tags=["disaster", "weather", "derecho", "severe-storm"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "dust_storm",
        "omni_mercury_engine.detectors.meteorological.dust_storm_detector",
        "DustStormDetector",
        DetectorCategory.METEOROLOGICAL,
        "Dust-storm/haboob detector (emission potential + visibility class)",
        tags=["disaster", "weather", "dust-storm", "severe-storm"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "surge_flood_cascade",
        "omni_mercury_engine.detectors.meteorological.surge_flood_cascade",
        "SurgeFloodCascade",
        DetectorCategory.METEOROLOGICAL,
        "Hurricane -> storm-surge -> coastal-flood cascade (CO-OPS residual)",
        tags=["disaster", "weather", "cascade", "surge", "flood"],
        seams=(),
        modality="event",
    ),
    # -- Geophysical detectors -------------------------------------------------
    DetectorManifestEntry(
        "avalanche",
        "omni_mercury_engine.detectors.geological.avalanche_detector",
        "AvalancheDetector",
        DetectorCategory.GEOLOGICAL,
        "Snowpack stability / SK38 skier-stability avalanche detector",
        tags=["disaster", "geological", "avalanche"],
        seams=(),
        modality="record",
    ),
    DetectorManifestEntry(
        "rockfall",
        "omni_mercury_engine.detectors.geological.rockfall_detector",
        "RockfallDetector",
        DetectorCategory.GEOLOGICAL,
        "Rockfall detector with Fukuzono inverse-velocity failure forecasting",
        tags=["disaster", "geological", "rockfall"],
        seams=(),
        modality="record",
    ),
    DetectorManifestEntry(
        "subsidence",
        "omni_mercury_engine.detectors.geological.subsidence_detector",
        "SubsidenceDetector",
        DetectorCategory.GEOLOGICAL,
        "Subsidence/sinkhole detector over InSAR-style displacement series",
        tags=["disaster", "geological", "subsidence", "insar"],
        seams=(),
        modality="record",
    ),
    DetectorManifestEntry(
        "eq_tsunami_cascade",
        "omni_mercury_engine.detectors.geological.eq_tsunami_cascade",
        "EqTsunamiCascadeDetector",
        DetectorCategory.GEOLOGICAL,
        "Earthquake -> tsunami cascade (PTWC-style screening on real events)",
        tags=["disaster", "geological", "cascade", "tsunami"],
        seams=(),
        modality="event",
    ),
    DetectorManifestEntry(
        "fire_debris_flow_cascade",
        "omni_mercury_engine.detectors.geological.fire_debris_flow_cascade",
        "FireDebrisFlowCascadeDetector",
        DetectorCategory.GEOLOGICAL,
        "Wildfire -> post-fire debris-flow cascade (USGS Staley 2017 M1 + Cannon I-D)",
        tags=["disaster", "geological", "cascade", "debris-flow"],
        seams=(),
        modality="record",
    ),
    # -- Space-weather detectors (T4) ------------------------------------------
    DetectorManifestEntry(
        "cme_arrival",
        "omni_mercury_engine.detectors.adapters.space",
        "CMEArrivalAdapter",
        DetectorCategory.SPACE,
        "CME Earth-arrival forecaster (Gopalswamy ESA + Vrsnak DBM, DONKI-fed)",
        tags=["disaster", "space-weather", "cme"],
        seams=("engine", "registry"),
        modality="rows",
        wraps="omni_mercury_engine.space.cme_arrival_detector:CMEArrivalDetector",
    ),
    DetectorManifestEntry(
        "sep_storm",
        "omni_mercury_engine.detectors.adapters.space",
        "SEPProtonWindowAdapter",
        DetectorCategory.SPACE,
        "Solar energetic-particle / NOAA S-scale radiation-storm detector",
        tags=["disaster", "space-weather", "sep", "radiation"],
        seams=("engine", "registry"),
        modality="window",
        wraps="omni_mercury_engine.space.sep_storm_detector:SEPStormDetector",
    ),
    DetectorManifestEntry(
        "ionospheric_scintillation",
        "omni_mercury_engine.detectors.adapters.space",
        "ScintillationAdapter",
        DetectorCategory.SPACE,
        "GNSS ionospheric scintillation detector (S4 / sigma-phi indices)",
        tags=["disaster", "space-weather", "gnss", "scintillation"],
        seams=("engine", "registry"),
        modality="record",
        wraps="omni_mercury_engine.space.ionospheric_scintillation_detector:IonosphericScintillationDetector",
    ),
    DetectorManifestEntry(
        "gic",
        "omni_mercury_engine.detectors.adapters.space",
        "GICWindowAdapter",
        DetectorCategory.SPACE,
        "Geomagnetically induced current detector (dB/dt + plane-wave geoelectric proxy)",
        tags=["disaster", "space-weather", "gic", "grid"],
        seams=("engine", "registry"),
        modality="window",
        wraps="omni_mercury_engine.space.gic_detector:GICDetector",
    ),
    DetectorManifestEntry(
        "solar_gic_cascade",
        "omni_mercury_engine.detectors.adapters.space",
        "SolarGICCascadeAdapter",
        DetectorCategory.SPACE,
        "Solar eruption -> geomagnetic storm -> GIC grid-impact cascade",
        tags=["disaster", "space-weather", "cascade", "gic"],
        seams=("engine", "registry"),
        modality="event",
        wraps="omni_mercury_engine.space.solar_gic_cascade:SolarGICCascadeDetector",
    ),
    # -- Economic detectors ---------------------------------------------------
    DetectorManifestEntry(
        "financial_crisis",
        "omni_mercury_engine.detectors.economic.financial_crisis_detector",
        "FinancialCrisisDetector",
        DetectorCategory.ECONOMIC,
        "Financial crisis prediction",
        seams=(),
        modality="record",
        covered_by="financial_stress",
    ),
    # -- Energy detectors -----------------------------------------------------
    DetectorManifestEntry(
        "emp",
        "omni_mercury_engine.detectors.energy.emp_detector",
        "EMPDetector",
        DetectorCategory.ENERGY,
        "Electromagnetic pulse detection",
        seams=(),
        modality="record",
    ),
    # -- Marine detectors -----------------------------------------------------
    DetectorManifestEntry(
        "marine_biodiversity",
        "omni_mercury_engine.detectors.marine.biodiversity_detector",
        "MarineBiodiversityDetector",
        DetectorCategory.MARINE,
        "Marine biodiversity threat detection",
        seams=(),
        modality="record",
    ),
    # -- Trained sequence / security ensembles --------------------------------
    DetectorManifestEntry(
        "lstm_autoencoder",
        "omni_mercury_engine.detectors.lstm_ae",
        "LSTMAutoencoderDetector",
        DetectorCategory.MODEL,
        "Trained LSTM-autoencoder multivariate sequence reconstruction",
        feature_dim=1,
        tags=["time-series", "trained", "multivariate", "reconstruction"],
        seams=("registry",),
        modality="series_1d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "realtime_threat",
        "omni_mercury_engine.security.realtime_threat_detection",
        "RealTimeThreatDetector",
        DetectorCategory.SECURITY,
        "Projection + kNN + robust-covariance ensemble, training-calibrated",
        feature_dim=3,
        tags=["security", "ensemble", "calibrated"],
        seams=("registry",),
        modality="series_1d",
    ),
    # -- UAV detectors --------------------------------------------------------
    DetectorManifestEntry(
        "drone",
        "omni_mercury_engine.detectors.drone.detector",
        "DroneAnomalyDetector",
        DetectorCategory.UAV,
        "Drone fault detection: RADD phase invariants + single-row Mercury ensemble",
        feature_dim=16,
        tags=["drone", "uav", "telemetry", "radd"],
        seams=("registry",),
        modality="series_1d",
    ),
    # -- Safeguards -----------------------------------------------------------
    DetectorManifestEntry(
        "nano_safeguard",
        "omni_mercury_engine.safeguards.nano_safeguards",
        "NanoSafeguardDetector",
        DetectorCategory.BASE,
        "Nano-safeguard micro-anomaly detection with hierarchical scanning",
        feature_dim=20,
        tags=["safeguard", "micro-anomaly", "3r-recursion", "lyapunov"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    # -- SOTA Visual Anomaly Detection ----------------------------------------
    DetectorManifestEntry(
        "patchcore",
        "omni_mercury_engine.detectors.visual",
        "PatchCoreDetector",
        DetectorCategory.VISUAL,
        "Memory bank + coreset subsampling anomaly detection",
        tags=["visual", "sota", "memory-bank"],
        seams=("engine", "registry"),
        modality="image_batch",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "padim",
        "omni_mercury_engine.detectors.visual",
        "PaDiMDetector",
        DetectorCategory.VISUAL,
        "Patch-wise Mahalanobis distance anomaly detection",
        tags=["visual", "sota", "mahalanobis"],
        seams=("engine", "registry"),
        modality="image_batch",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "stfpm",
        "omni_mercury_engine.detectors.visual",
        "STFPMDetector",
        DetectorCategory.VISUAL,
        "Student-Teacher Feature Pyramid Matching",
        tags=["visual", "sota", "teacher-student"],
        seams=("engine", "registry"),
        modality="image_batch",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "reverse_distillation",
        "omni_mercury_engine.detectors.visual",
        "ReverseDistillationDetector",
        DetectorCategory.VISUAL,
        "Reverse knowledge distillation with OCE bottleneck",
        tags=["visual", "sota", "distillation"],
        seams=("engine", "registry"),
        modality="image_batch",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "cflow",
        "omni_mercury_engine.detectors.visual",
        "CFlowDetector",
        DetectorCategory.VISUAL,
        "Conditional normalizing flow anomaly detection",
        tags=["visual", "sota", "normalizing-flow"],
        seams=("engine", "registry"),
        modality="image_batch",
        fit_required=True,
    ),
    # -- Vision-Language Model (VLM) Detectors --------------------------------
    DetectorManifestEntry(
        "anyanomaly",
        "omni_mercury_engine.detectors.vlm",
        "AnyAnomalyDetector",
        DetectorCategory.VLM,
        "Zero-shot customizable VAD with LVLM (WACV 2026)",
        tags=["vlm", "zero-shot", "lvlm", "sota"],
        seams=("engine",),
        modality="video_frames",
        fit_required=True,
        external=True,
    ),
    DetectorManifestEntry(
        "lavad",
        "omni_mercury_engine.detectors.vlm",
        "LAVADDetector",
        DetectorCategory.VLM,
        "Training-free LLM-based VAD (CVPR 2024)",
        tags=["vlm", "training-free", "llm", "sota"],
        seams=("engine", "registry"),
        modality="video_frames",
        fit_required=True,
    ),
    # -- Foundation Model Adapters --------------------------------------------
    DetectorManifestEntry(
        "timegpt",
        "omni_mercury_engine.models.foundation",
        "TimeGPTAdapter",
        DetectorCategory.FOUNDATION,
        "Nixtla TimeGPT foundation model adapter",
        tags=["foundation", "time-series", "api"],
        seams=(),
        modality="series_1d",
        external=True,
    ),
    DetectorManifestEntry(
        "chronos",
        "omni_mercury_engine.models.foundation",
        "ChronosAdapter",
        DetectorCategory.FOUNDATION,
        "Amazon Chronos time-series foundation model",
        tags=["foundation", "time-series", "local"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "matrix_profile",
        "omni_mercury_engine.models.foundation",
        "MatrixProfileAdapter",
        DetectorCategory.FOUNDATION,
        "STUMPY Matrix Profile for time-series anomalies",
        tags=["foundation", "time-series", "matrix-profile"],
        seams=("registry",),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "foundation_ensemble",
        "omni_mercury_engine.models.foundation",
        "FoundationEnsemble",
        DetectorCategory.FOUNDATION,
        "Ensemble of foundation model adapters",
        tags=["foundation", "ensemble"],
        seams=("registry",),
        modality="series_1d",
    ),
    # -- Knowledge Distillation -----------------------------------------------
    DetectorManifestEntry(
        "dual_student",
        "omni_mercury_engine.ml.distillation",
        "DualStudentDistillation",
        DetectorCategory.DISTILLATION,
        "Dual-student knowledge distillation for AD",
        tags=["distillation", "sota", "dual-student"],
        seams=("registry",),
        modality="image_batch",
        fit_required=True,
    ),
    # -- Advanced Physics-Inspired Detectors (v1.4.0) -------------------------
    DetectorManifestEntry(
        "spectral_vibration",
        "omni_mercury_engine.detectors.spectral_vibration",
        "SpectralVibrationDetector",
        DetectorCategory.PHYSICS,
        "GNN/CNN spectral analysis with phonon interactions",
        tags=["physics", "spectral", "vibration", "maintenance"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "acceleration_dynamics",
        "omni_mercury_engine.detectors.acceleration_dynamics",
        "AccelerationDynamicsDetector",
        DetectorCategory.PHYSICS,
        "Kinematic analysis with Lyapunov stability and phase space",
        tags=["physics", "kinematics", "chaos", "energy"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "uiux_anomaly",
        "omni_mercury_engine.detectors.uiux_anomaly",
        "UIUXAnomalyDetector",
        DetectorCategory.UIUX,
        "User interaction and behavior anomaly detection",
        tags=["uiux", "behavior", "engagement", "bot-detection"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    DetectorManifestEntry(
        "physics_integrated",
        "omni_mercury_engine.detectors.physics_integration",
        "PhysicsIntegratedDetector",
        DetectorCategory.PHYSICS,
        "Unified physics detector with 3R and GOSNN integration",
        tags=["physics", "integrated", "3r", "gosnn", "fusion"],
        seams=("engine", "registry"),
        modality="series_1d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "spectral_domain_frequency",
        "omni_mercury_engine.detectors.spectral_domain_frequency",
        "SpectralDomainFrequency",
        DetectorCategory.PHYSICS,
        "Neuro-symbolic spectral-domain anomaly detection with per-band scoring",
        tags=["physics", "spectral", "frequency", "neuro-symbolic", "humanitarian"],
        seams=("engine", "registry"),
        modality="series_1d",
    ),
    # -- Hazard adapters: the engine contract over the hazard detectors ------
    # Each wraps its domain detector's own API in named-column batches; the
    # scores and decisions are the wrapped detector's own (see
    # detectors/hazard_adapter.py). The "adapter" tag drives the manifest
    # contract test that holds these to the full BaseDetector contract.
    DetectorManifestEntry(
        "tornado_conditions",
        "omni_mercury_engine.detectors.hazard_adapter",
        "TornadoConditionsAdapter",
        DetectorCategory.GEOLOGICAL,
        "TornadoDetector over (cape_j_kg, cin_j_kg, srh_m2_s2, bulk_shear_kt, lcl_m) rows",
        feature_dim=4,
        tags=["adapter", "tornado", "atmospheric"],
        seams=("engine", "registry"),
        modality="rows",
    ),
    DetectorManifestEntry(
        "hurricane_pressure",
        "omni_mercury_engine.detectors.hazard_adapter",
        "HurricanePressureAdapter",
        DetectorCategory.GEOLOGICAL,
        "HurricaneDetector over (central_pressure_mb, environmental_pressure_mb, sst_celsius)",
        feature_dim=4,
        tags=["adapter", "hurricane", "pressure"],
        seams=("engine", "registry"),
        modality="rows",
    ),
    DetectorManifestEntry(
        "flood_precipitation",
        "omni_mercury_engine.detectors.hazard_adapter",
        "FloodPrecipitationAdapter",
        DetectorCategory.GEOLOGICAL,
        "FloodDetector over measured and forecast precipitation depth rows (inches)",
        feature_dim=3,
        tags=["adapter", "flood", "precipitation"],
        seams=("engine", "registry"),
        modality="rows",
    ),
    DetectorManifestEntry(
        "financial_stress",
        "omni_mercury_engine.detectors.hazard_adapter",
        "FinancialStressAdapter",
        DetectorCategory.ECONOMIC,
        "FinancialCrisisDetector over measured market and banking metric rows",
        feature_dim=4,
        tags=["adapter", "financial", "market", "banking"],
        seams=("engine", "registry"),
        modality="rows",
    ),
    DetectorManifestEntry(
        "solar_wind_geomag",
        "omni_mercury_engine.detectors.hazard_adapter",
        "SolarWindGeomagAdapter",
        DetectorCategory.SPACE,
        "SolarStormDetector over (solar_wind_speed_km_s, bz_imf_nt); score is NOAA G/5",
        feature_dim=3,
        tags=["adapter", "geomagnetic", "solar-wind"],
        seams=("engine", "registry"),
        modality="rows",
    ),
    DetectorManifestEntry(
        "xray_flare",
        "omni_mercury_engine.detectors.hazard_adapter",
        "XrayFlareAdapter",
        DetectorCategory.SPACE,
        "SolarFlareDetector over GOES long-band flux rows; score is NOAA R/5",
        feature_dim=3,
        tags=["adapter", "flare", "xray"],
        seams=("engine", "registry"),
        modality="rows",
    ),
    DetectorManifestEntry(
        "gic_minute_series",
        "omni_mercury_engine.detectors.hazard_adapter",
        "GICMinuteSeriesAdapter",
        DetectorCategory.SPACE,
        "GICDetector over a (bx_nt, by_nt) minute series; score is the dB/dt tier over 3",
        feature_dim=3,
        tags=["adapter", "gic", "magnetometer"],
        seams=("engine", "registry"),
        modality="rows",
    ),
    # -- Statistical / geometric heads integrated from library-only modules --
    DetectorManifestEntry(
        "extended_statistical",
        "omni_mercury_engine.detectors.statistical_extended",
        "ExtendedStatisticalDetector",
        DetectorCategory.BASE,
        "Ensemble over MAD, LOF, DBSCAN, MCD, CUSUM, Grubbs and GESD statistical methods",
        seams=("engine", "registry"),
        modality="series_1d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "adaptive_anomaly",
        "omni_mercury_engine.detectors.adaptive_detector",
        "AdaptiveAnomalyDetector",
        DetectorCategory.BASE,
        "Profile-calibrated detector routing to temporal/covariance/high-dim backends",
        seams=("registry",),
        modality="matrix_2d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "topological",
        "omni_mercury_engine.detectors.topological_analysis",
        "TopologicalAnomalyDetector",
        DetectorCategory.BASE,
        "Persistent-homology profile scoring (Betti, entropy, landscapes, Wasserstein)",
        seams=("registry",),
        modality="matrix_2d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "multivariate_ts",
        "omni_mercury_engine.detectors.multivariate_timeseries",
        "MultivariateTSDetector",
        DetectorCategory.BASE,
        "LTG multivariate windowed time-series detector (LSTM/temporal-conv/graph features)",
        seams=("registry",),
        modality="windows_3d",
        fit_required=True,
    ),
    DetectorManifestEntry(
        "info_geometry",
        "omni_mercury_engine.detectors.info_geometry",
        "InformationGeometryDetector",
        DetectorCategory.BASE,
        "Fisher-Rao information-geometry OOD detector over a fitted reference",
        seams=("registry",),
        modality="matrix_2d",
        fit_required=True,
    ),
]


class DetectorRegistry:
    """Central registry for all anomaly detectors and models.

    Provides unified interface for:
    - Detector registration and discovery
    - Parallel feature extraction
    - Feature aggregation for fusion pipeline
    - Detector health monitoring

    This is the bridge that connects detectors across domains to the fusion model.
    """

    # Target feature dimension for fusion
    FUSION_FEATURE_DIM = 768

    def __init__(
        self,
        max_workers: int = 8,
        timeout_seconds: float = 30.0,
        auto_discover: bool = False,
    ):
        """Initialize DetectorRegistry.

        Args:
            max_workers: Maximum parallel workers for feature extraction
            timeout_seconds: Timeout for detector operations
            auto_discover: Automatically discover and register detectors
        """
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds

        self._detectors: dict[str, DetectorInfo] = {}
        self._category_index: dict[DetectorCategory, list[str]] = {
            cat: [] for cat in DetectorCategory
        }
        # Dedicated detector executor: keeps finished results when a detector
        # overruns ``timeout_seconds``, returns results in request order, and
        # runs only ``parallel_safe`` detectors concurrently.
        self._executor: ParallelDetectorExecutor | None = None
        # Opt-in result cache (enable_feature_cache); the tensor-vs-ndarray
        # provenance of a hit is encoded in the cache key itself (|ft / |f),
        # so the registry keeps no side state that could outlive eviction.
        self._feature_cache: Any = None

        if auto_discover:
            self.auto_discover_detectors()

    # -- opt-in feature cache ------------------------------------------------

    def enable_feature_cache(
        self,
        max_memory_mb: float = 256.0,
        max_entries: int = 1000,
        ttl_seconds: float = 0.0,
    ) -> None:
        """Cache (features, scores) for repeated identical payloads.

        Only detectors exposing ``feature_cache_token()`` take part: the
        token must change whenever their fitted state changes, so a refit
        can never serve stale features, and a detector without one (e.g. a
        stateful per-tick detector) is never cached. Hits and misses are
        recorded in ``omni_feature_cache_hits_total`` /
        ``omni_feature_cache_misses_total``. Storage is lossless.
        """
        from omni_mercury_engine.utils.feature_cache import (
            CacheConfig,
            MemoryEfficientFeatureCache,
        )

        self._feature_cache = MemoryEfficientFeatureCache(
            CacheConfig(max_size=max_entries, max_memory_mb=max_memory_mb, ttl_seconds=ttl_seconds)
        )

    def disable_feature_cache(self) -> None:
        """Drop the feature cache and its entries."""
        self._feature_cache = None

    def feature_cache_stats(self) -> dict[str, Any] | None:
        """Cache statistics, or ``None`` when the cache is off."""
        return None if self._feature_cache is None else self._feature_cache.get_stats()

    def _feature_cache_key(self, name: str, detector: Any, data: Any) -> str | None:
        """``name|token|sha3(data)`` for a cacheable call, else ``None``."""
        if self._feature_cache is None or isinstance(data, dict):
            return None
        token_fn = getattr(detector, "feature_cache_token", None)
        if not callable(token_fn):
            return None
        token = token_fn()
        if token is None:
            return None
        if torch is not None and isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)
        if arr.dtype == object:
            return None
        import hashlib

        digest = hashlib.sha3_256(
            np.ascontiguousarray(arr).tobytes() + repr((arr.shape, arr.dtype.str)).encode()
        ).hexdigest()
        return f"{name}|{token!r}|{digest}"

    def _cached_result(self, key: str) -> tuple[Any, Any] | None:
        """Coherent cache hit, or ``None``.

        A hit requires the features entry AND its scores companion: the
        underlying cache evicts entries independently (LRU / memory / TTL),
        and serving features whose scores entry was evicted would silently
        strip ``anomaly_scores`` from the result. ``|ft`` marks features
        stored from a torch tensor (the flag lives in the key, so no side
        state can outlive or outgrow the cache itself); a zero-length
        scores array is the sentinel for "this result had no scores".
        """
        cache = self._feature_cache
        if cache is None:
            return None
        tensor_hit = cache.contains(f"{key}|ft")
        features_key = f"{key}|ft" if tensor_hit else f"{key}|f"
        if not cache.contains(features_key) or not cache.contains(f"{key}|s"):
            return None
        features = cache.get(features_key)
        scores = cache.get(f"{key}|s")
        if features is None or scores is None:
            return None
        if tensor_hit and torch is not None:
            features = torch.as_tensor(features)
        return features, (None if scores.size == 0 else scores)

    def _store_result(self, key: str, features: Any, scores: Any) -> None:
        cache = self._feature_cache
        if cache is None or features is None:
            return
        is_tensor = torch is not None and isinstance(features, torch.Tensor)
        stored = features.detach().cpu().numpy() if is_tensor else np.asarray(features)
        if stored.dtype == object:
            return
        scores_arr = np.empty(0, dtype=np.float64) if scores is None else np.asarray(scores)
        if scores_arr.dtype == object:
            return
        features_key = f"{key}|ft" if is_tensor else f"{key}|f"
        if cache.put(features_key, stored):
            # If this put fails the lone features entry is unreadable by
            # design (_cached_result demands both), so it degrades to a
            # miss instead of a scores-less hit.
            cache.put(f"{key}|s", scores_arr)

    def register(
        self,
        name: str,
        detector: Any,
        category: DetectorCategory = DetectorCategory.BASE,
        feature_dim: int | None = None,
        description: str = "",
        tags: list[str] | None = None,
        modality: str = "record",
    ) -> None:
        """Register a detector with the registry.

        Args:
            name: Unique detector name
            detector: Detector instance (must have extract_features and predict methods)
            category: Detector category for organization
            feature_dim: Output feature dimension (auto-detected if None)
            description: Human-readable description
            tags: Optional tags for filtering
            modality: Input payload the detector's registry methods accept
                (one of ``MANIFEST_MODALITIES``); lets ``extract_all_features``
                run only the detectors that can read a given payload.

        Raises:
            TypeError: If ``detector`` has neither ``extract_features`` nor
                ``predict``.  Such a detector can never produce output through
                the registry; registering it anyway used to make every
                extraction report ``success=True`` with no features and no
                scores -- a silent no-op in the fusion path.  Call its domain
                API directly, or register a real adapter
                (see ``detectors/hazard_adapter.py``).
        """
        if not hasattr(detector, "extract_features") and not hasattr(detector, "predict"):
            raise TypeError(
                f"Detector {name!r} has neither 'extract_features' nor 'predict' and can "
                "never produce output through the registry. Call its domain API directly, "
                "or register a real adapter (see detectors/hazard_adapter.py)."
            )

        info = DetectorInfo(
            name=name,
            category=category,
            instance=detector,
            feature_dim=feature_dim,
            is_fitted=getattr(detector, "_is_fitted", False),
            description=description,
            tags=tags or [],
            modality=modality,
        )

        self._detectors[name] = info
        self._category_index[category].append(name)

        logger.debug(f"Registered detector: {name} (category={category.value})")

    def unregister(self, name: str) -> bool:
        """Unregister a detector.

        Args:
            name: Detector name to remove

        Returns:
            True if removed, False if not found
        """
        if name not in self._detectors:
            return False

        info = self._detectors.pop(name)
        self._category_index[info.category].remove(name)
        return True

    def get(self, name: str) -> DetectorInfo | None:
        """Get detector info by name."""
        return self._detectors.get(name)

    def get_by_category(self, category: DetectorCategory) -> list[DetectorInfo]:
        """Get all detectors in a category."""
        return [
            self._detectors[name]
            for name in self._category_index[category]
            if name in self._detectors
        ]

    def list_all(self) -> list[str]:
        """List all registered detector names."""
        return list(self._detectors.keys())

    def list_by_tags(self, tags: list[str]) -> list[str]:
        """List detectors matching any of the given tags."""
        matching = []
        for name, info in self._detectors.items():
            if any(tag in info.tags for tag in tags):
                matching.append(name)
        return matching

    def _get_executor(self) -> ParallelDetectorExecutor:
        """Get or create thread pool executor."""
        if self._executor is None:
            self._executor = ParallelDetectorExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="mercury-registry",
            )
        return self._executor

    def extract_features(
        self,
        name: str,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
    ) -> FeatureExtractionResult:
        """Extract features from a single detector with circuit breaker protection.

        Uses circuit breaker pattern to prevent cascade failures when a detector
        repeatedly fails. If the circuit is open, returns a failed result immediately
        without invoking the detector.

        Args:
            name: Detector name
            data: Input data

        Returns:
            FeatureExtractionResult with features and metadata
        """
        return self._extract_features(name, data, _CallTicket())

    def _extract_features(
        self,
        name: str,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
        ticket: _CallTicket,
    ) -> FeatureExtractionResult:
        """:meth:`extract_features` under a ticket the caller may abandon at a deadline.

        Args:
            name: Detector name
            data: Input data
            ticket: Shared with the caller; an abandoned call neither updates
                the breaker nor the detector's statistics when it ends.

        Returns:
            FeatureExtractionResult with features and metadata
        """
        if name not in self._detectors:
            return FeatureExtractionResult(
                detector_name=name,
                features=None,
                scores=None,
                execution_time_ms=0,
                success=False,
                error=f"Detector '{name}' not found",
            )

        breaker = get_detector_breaker(name)

        if breaker.state == CircuitState.OPEN:
            logger.debug(f"Circuit breaker OPEN for detector '{name}', skipping")
            return FeatureExtractionResult(
                detector_name=name,
                features=None,
                scores=None,
                execution_time_ms=0,
                success=False,
                error=f"Circuit breaker open for detector '{name}'",
            )

        info = self._detectors[name]
        detector = info.instance
        if not ticket.start():
            return FeatureExtractionResult(
                detector_name=name,
                features=None,
                scores=None,
                execution_time_ms=0,
                success=False,
                error=f"Detector '{name}' abandoned at the deadline before it started",
            )
        start_time = time.perf_counter()

        def _execute_detector() -> FeatureExtractionResult:
            """Run the detector; the caller reports the outcome to the breaker."""
            # Try to fit if needed. ``is_fitted`` is a method on the core
            # detector contract (core/base.py), but this registry is a
            # plugin boundary: registered detectors may expose it as a
            # read-only property instead, so resolve both forms rather
            # than crash calling a bool.
            fitted_attr = getattr(detector, "is_fitted", None)
            if fitted_attr is not None:
                fitted = fitted_attr() if callable(fitted_attr) else bool(fitted_attr)
                if not fitted and hasattr(detector, "fit") and not isinstance(data, dict):
                    detector.fit(data)

            cache_key = self._feature_cache_key(name, detector, data)
            cached = self._cached_result(cache_key) if cache_key is not None else None
            if cached is not None:
                from omni_mercury_engine.core.metrics import record_cache_hit

                record_cache_hit(name)
                features, scores = cached
            else:
                # Extract features
                features = None
                if hasattr(detector, "extract_features"):
                    features = detector.extract_features(data)

                # Get predictions/scores
                scores = None
                if hasattr(detector, "predict"):
                    result = detector.predict(data)
                    if isinstance(result, dict):
                        scores = result.get("anomaly_scores", result.get("scores"))
                        if scores is not None:
                            scores = np.atleast_1d(scores)
                if features is None and scores is None:
                    # The detector ran but yielded nothing the registry can
                    # harvest.  Raising routes this through the normal failure
                    # path (success=False, error_count, circuit breaker), so a
                    # detector that never produces output stops being paid for
                    # instead of reporting success forever.
                    raise RuntimeError(
                        "detector produced no features and no scores under the "
                        "registry contract (extract_features/predict)"
                    )
                if cache_key is not None:
                    from omni_mercury_engine.core.metrics import record_cache_miss

                    record_cache_miss(name)
                    self._store_result(cache_key, features, scores)

            return FeatureExtractionResult(
                detector_name=name,
                features=features,
                scores=scores,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                success=True,
            )

        try:
            result = _execute_detector()
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            if ticket.finish():
                _record_breaker_outcome(breaker, e)
                info.error_count += 1
                info.last_execution_time_ms = execution_time
            # else: abandoned at the deadline, already counted as a timeout.
            logger.debug(f"Feature extraction failed for '{name}': {e}")
            return FeatureExtractionResult(
                detector_name=name,
                features=None,
                scores=None,
                execution_time_ms=execution_time,
                success=False,
                error=str(e),
            )

        if ticket.finish():
            _record_breaker_outcome(breaker, None)
            info.last_execution_time_ms = result.execution_time_ms
            info.total_invocations += 1
            info.is_fitted = getattr(detector, "_is_fitted", True)
        # else: a late completion of a call abandoned (and counted as a
        # timeout) at the deadline. Its result is discarded by the executor,
        # and it must not count as a success that resets the breaker.
        return result

    def _record_timeout(self, name: str, duration_s: float) -> None:
        """Count a call abandoned at the deadline as a detector failure.

        Args:
            name: Detector name.
            duration_s: How long the call had been running at the deadline.
        """
        info = self._detectors.get(name)
        if info is not None:
            info.error_count += 1
            info.last_execution_time_ms = duration_s * 1000.0
        _record_breaker_outcome(
            get_detector_breaker(name),
            TimeoutError(f"{name} exceeded the {self.timeout_seconds:g}s detector budget"),
        )
        logger.warning(
            "Detector '%s' timed out after %gs; counted as a circuit-breaker failure",
            name,
            self.timeout_seconds,
        )

    def extract_all_features(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
        parallel: bool = True,
        categories: list[DetectorCategory] | None = None,
        detector_names: list[str] | None = None,
        modality: str | None = None,
    ) -> dict[str, FeatureExtractionResult]:
        """Extract features from all registered detectors.

        Args:
            data: Input data
            parallel: Use parallel execution
            categories: Filter by categories (None = all)
            detector_names: Filter by specific names (None = all)
            modality: Run only detectors registered for this input modality
                (None = all).  ``data`` is broadcast to every detector run, so
                passing the payload's modality keeps detectors that cannot
                read it from being driven into typed errors.

        Returns:
            Dictionary mapping detector names to extraction results
        """
        # Determine which detectors to run
        if detector_names is not None:
            names = [n for n in detector_names if n in self._detectors]
        elif categories is not None:
            names = []
            for cat in categories:
                names.extend(self._category_index[cat])
        else:
            names = list(self._detectors.keys())
        if modality is not None:
            names = [n for n in names if self._detectors[n].modality == modality]

        results: dict[str, FeatureExtractionResult] = {}

        if parallel and len(names) > 1:
            # The old loop used ``as_completed(..., timeout=...)``: on expiry it
            # raised TimeoutError out of the loop and lost every result that had
            # already finished, and it returned results in completion order.
            #
            # When the executor is saturated by stranded overrun threads it
            # refuses parallel work rather than running it in this thread:
            # here nothing could bound a hung detector, and the caller would
            # hang with it. Counting timeouts as breaker failures (below) is
            # what keeps a hung detector from stranding threads until then.
            tickets = {name: _CallTicket() for name in dict.fromkeys(names)}
            outcomes = self._get_executor().run(
                [
                    DetectorTask(
                        name,
                        functools.partial(self._extract_features, name, data, tickets[name]),
                        parallel_safe=getattr(
                            self._detectors[name].instance, "parallel_safe", False
                        )
                        is True,
                    )
                    for name in tickets
                ],
                timeout_s=self.timeout_seconds,
            )
            for outcome in outcomes:
                if outcome.ok:
                    results[outcome.name] = outcome.value
                    continue
                if outcome.status is OutcomeStatus.TIMEOUT and tickets[outcome.name].abandon():
                    self._record_timeout(outcome.name, outcome.duration_s)
                error = (
                    f"timed out after {self.timeout_seconds:g}s"
                    if outcome.status is OutcomeStatus.TIMEOUT
                    else str(outcome.error)
                )
                results[outcome.name] = FeatureExtractionResult(
                    detector_name=outcome.name,
                    features=None,
                    scores=None,
                    execution_time_ms=outcome.duration_s * 1000.0,
                    success=False,
                    error=error,
                )
        else:
            for name in names:
                results[name] = self.extract_features(name, data)

        return results

    def aggregate_features(
        self,
        extraction_results: dict[str, FeatureExtractionResult],
        target_dim: int | None = None,
        device: str = "cpu",
        enable_feature_engineering: bool = False,
        enable_interaction_features: bool = False,
        enable_temporal_features: bool = False,
        enable_rolling_stats: bool = False,
        enable_golden_ratio_scaling: bool = False,
        rolling_window_size: int = 5,
        temporal_lags: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate features from multiple detectors for fusion.

        Enhanced with feature engineering capabilities for improved anomaly detection:
        - Interaction features: Cross-detector correlation features
        - Temporal lag features: Multi-scale temporal dependencies
        - Rolling window statistics: Mean, std, min, max over windows
        - Golden ratio scaling: Apply phi-optimization to feature dimensions

        Args:
            extraction_results: Results from extract_all_features
            target_dim: Target dimension for each feature vector (default: 128)
            device: PyTorch device
            enable_feature_engineering: Enable all feature engineering (master switch)
            enable_interaction_features: Add cross-detector correlation features
            enable_temporal_features: Add temporal lag features
            enable_rolling_stats: Add rolling window statistics
            enable_golden_ratio_scaling: Apply golden ratio scaling to features
            rolling_window_size: Window size for rolling statistics
            temporal_lags: List of lag values for temporal features

        Returns:
            Dictionary mapping detector names to normalized tensors
        """
        target_dim = target_dim or 128
        aggregated: dict[str, torch.Tensor] = {}

        for name, result in extraction_results.items():
            if not result.success or result.features is None:
                continue

            features = result.to_tensor(device)
            if features is None:
                continue

            if features.dim() == 1:
                features = features.unsqueeze(0)

            current_dim = features.shape[-1]
            if current_dim != target_dim:
                if current_dim > target_dim:
                    features = features[..., :target_dim]
                else:
                    padding = torch.zeros(
                        *features.shape[:-1],
                        target_dim - current_dim,
                        device=device,
                        dtype=features.dtype,
                    )
                    features = torch.cat([features, padding], dim=-1)

            aggregated[name] = features

        if enable_feature_engineering or enable_golden_ratio_scaling:
            aggregated = self._apply_golden_ratio_scaling(aggregated)

        if enable_feature_engineering or enable_rolling_stats:
            aggregated = self._add_rolling_statistics(aggregated, rolling_window_size, device)

        if enable_feature_engineering or enable_temporal_features:
            aggregated = self._add_temporal_lag_features(
                aggregated, temporal_lags or [1, 2, 5], device
            )

        if enable_feature_engineering or enable_interaction_features:
            aggregated = self._add_interaction_features(aggregated, device)

        return aggregated

    def _apply_golden_ratio_scaling(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply golden ratio (phi) scaling to feature dimensions.

        The golden ratio (1.618...) has been shown to optimize information distribution in neural
        networks and signal processing.
        """
        phi = (1 + 5**0.5) / 2
        scaled: dict[str, torch.Tensor] = {}

        for name, feat in features.items():
            dim = feat.shape[-1]
            scale_factors = torch.tensor(
                [phi ** (i / dim - 0.5) for i in range(dim)],
                device=feat.device,
                dtype=feat.dtype,
            )
            scaled[name] = feat * scale_factors

        return scaled

    def _add_rolling_statistics(
        self,
        features: dict[str, torch.Tensor],
        window_size: int,
        device: str,
    ) -> dict[str, torch.Tensor]:
        """Add rolling window statistics to features.

        Computes mean, std, min, max over sliding windows for each feature.
        """
        enhanced: dict[str, torch.Tensor] = {}

        for name, feat in features.items():
            enhanced[name] = feat

            if feat.dim() < 2 or feat.shape[-1] < window_size:
                continue

            batch_size = feat.shape[0]
            feat_dim = feat.shape[-1]

            rolling_mean = torch.zeros(batch_size, feat_dim, device=feat.device)
            rolling_std = torch.zeros(batch_size, feat_dim, device=feat.device)

            for i in range(feat_dim):
                start = max(0, i - window_size // 2)
                end = min(feat_dim, i + window_size // 2 + 1)
                window = feat[:, start:end]
                rolling_mean[:, i] = window.mean(dim=-1)
                rolling_std[:, i] = window.std(dim=-1) if window.shape[-1] > 1 else 0

            enhanced[f"{name}_rolling_mean"] = rolling_mean
            enhanced[f"{name}_rolling_std"] = rolling_std

        return enhanced

    def _add_temporal_lag_features(
        self,
        features: dict[str, torch.Tensor],
        lags: list[int],
        device: str,
    ) -> dict[str, torch.Tensor]:
        """Add temporal lag features for multi-scale dependency analysis."""
        enhanced: dict[str, torch.Tensor] = {}

        for name, feat in features.items():
            enhanced[name] = feat

            if feat.dim() < 2:
                continue

            feat_dim = feat.shape[-1]

            for lag in lags:
                if lag >= feat_dim:
                    continue

                lagged = torch.zeros_like(feat)
                lagged[:, lag:] = feat[:, :-lag]
                lagged[:, :lag] = feat[:, 0:1].expand(-1, lag)

                diff = feat - lagged

                enhanced[f"{name}_lag{lag}"] = lagged
                enhanced[f"{name}_diff{lag}"] = diff

        return enhanced

    def _add_interaction_features(
        self,
        features: dict[str, torch.Tensor],
        device: str,
    ) -> dict[str, torch.Tensor]:
        """Add interaction features between detector outputs.

        Creates cross-correlation and product features for enhanced detection.
        """
        enhanced = dict(features)
        feature_names = list(features.keys())

        base_names = [
            n
            for n in feature_names
            if not any(suffix in n for suffix in ["_rolling_", "_lag", "_diff", "_x_", "_corr_"])
        ]

        for i, name1 in enumerate(base_names):
            for name2 in base_names[i + 1 :]:
                feat1 = features.get(name1)
                feat2 = features.get(name2)

                if feat1 is None or feat2 is None:
                    continue

                if feat1.shape != feat2.shape:
                    continue

                enhanced[f"{name1}_x_{name2}"] = feat1 * feat2

                if feat1.dim() >= 2 and feat1.shape[-1] > 1:
                    corr_vals = []
                    for b in range(feat1.shape[0]):
                        f1_flat = feat1[b].flatten()
                        f2_flat = feat2[b].flatten()
                        f1_centered = f1_flat - f1_flat.mean()
                        f2_centered = f2_flat - f2_flat.mean()
                        denom = f1_centered.norm() * f2_centered.norm()
                        if denom > 1e-8:
                            corr = (f1_centered * f2_centered).sum() / denom
                        else:
                            corr = torch.tensor(0.0, device=feat1.device)
                        corr_vals.append(corr)

                    corr_tensor = torch.stack(corr_vals).unsqueeze(-1)
                    corr_expanded = corr_tensor.expand(-1, feat1.shape[-1])
                    enhanced[f"{name1}_corr_{name2}"] = corr_expanded

        return enhanced

    def auto_discover_detectors(self) -> int:
        """Auto-discover and register available detectors.

        Iterates over DETECTOR_MANIFEST entries, dynamically importing each
        detector class and registering it. Missing optional dependencies are
        logged at debug level and skipped gracefully.

        Returns:
            Number of detectors registered
        """
        registered_count = 0

        for entry in DETECTOR_MANIFEST:
            if "registry" not in entry.seams:
                # Library-only (or engine-only) entry: its registry methods are
                # absent or yield nothing, so registering it would re-create the
                # silent no-op this seam field exists to end.
                logger.debug(
                    "Manifest entry '%s' does not declare the registry seam; skipping",
                    entry.name,
                )
                continue
            try:
                # Only allow imports from the omni_mercury_engine package tree
                # to prevent arbitrary module loading from manifest entries.
                if not entry.module_path.startswith("omni_mercury_engine."):
                    logger.warning(
                        "Blocked detector import from untrusted module path: %s",
                        entry.module_path,
                    )
                    continue
                module = __import__(entry.module_path, fromlist=[entry.class_name])
                cls = getattr(module, entry.class_name)
                self.register(
                    entry.name,
                    cls(),
                    entry.category,
                    feature_dim=entry.feature_dim,
                    description=entry.description,
                    tags=entry.tags or [],
                    modality=entry.modality,
                )
                registered_count += 1
            except ImportError as e:
                logger.debug("Optional detector '%s' not available: %s", entry.name, e)
            except Exception as e:
                logger.warning("Failed to load detector '%s': %s", entry.name, e)

        logger.info("Auto-discovered and registered %d detectors", registered_count)
        return registered_count

    def aggregate_enhanced_geological_features(
        self,
        extraction_results: dict[str, FeatureExtractionResult],
        device: str = "cpu",
        enable_gosnn_synapse: bool = True,
    ) -> dict[str, Any]:
        """Aggregate features from enhanced geological detectors with 128D normalization.

        This method specifically handles the enhanced Landslide, Wildfire, and Volcanic
        detectors with their 3R mechanism synapses (Recursion, Resonance, Refactoring).

        Features are normalized to 128D for fusion pipeline compatibility and optionally
        passed through GOSNN ethical gating for scalar registration.

        Args:
            extraction_results: Results from extract_all_features for geological detectors
            device: PyTorch device for tensor operations
            enable_gosnn_synapse: Enable GOSNN ethical gating and scalar registration

        Returns:
            Dictionary containing:
                - aggregated_features: 128D normalized feature tensors per detector
                - combined_features: Concatenated features from all geological detectors
                - gosnn_scalars: Registered omni-scalars (if GOSNN enabled)
                - detector_coverage: Coverage statistics
        """
        target_dim = 128  # Standard 128D normalization for fusion
        phi = 1.618033988749895  # Golden ratio for weighting

        # Filter to geological detectors only
        geological_names = ["landslide", "wildfire", "volcanic"]
        geo_results = {
            name: result
            for name, result in extraction_results.items()
            if name in geological_names and result.success and result.features is not None
        }

        aggregated: dict[str, torch.Tensor] = {}
        omni_scalars: dict[str, float] = {}

        for name, result in geo_results.items():
            features = result.to_tensor(device)
            if features is None:
                continue

            # Ensure 2D tensor
            if features.dim() == 1:
                features = features.unsqueeze(0)

            # Normalize to 128D
            current_dim = features.shape[-1]
            if current_dim != target_dim:
                if current_dim > target_dim:
                    # Truncate with importance weighting (keep most significant)
                    features = features[..., :target_dim]
                else:
                    # Pad with zeros
                    padding = torch.zeros(
                        *features.shape[:-1],
                        target_dim - current_dim,
                        device=device,
                        dtype=features.dtype,
                    )
                    features = torch.cat([features, padding], dim=-1)

            # Apply L2 normalization for stable fusion
            norm = torch.norm(features, p=2, dim=-1, keepdim=True)
            features = features / (norm + 1e-8)

            # Apply golden ratio scaling for phi-optimization
            scale_factors = torch.tensor(
                [phi ** (i / target_dim - 0.5) for i in range(target_dim)],
                device=device,
                dtype=features.dtype,
            )
            features = features * scale_factors

            aggregated[name] = features

            # Register omni-scalars for GOSNN synapse
            if enable_gosnn_synapse:
                # Compute detector-specific scalars
                feature_mean = float(features.mean().item())
                feature_std = float(features.std().item())
                feature_max = float(features.max().item())

                omni_scalars[f"omni_{name}_mean"] = feature_mean
                omni_scalars[f"omni_{name}_std"] = feature_std
                omni_scalars[f"omni_{name}_max"] = feature_max
                omni_scalars[f"omni_{name}_coverage"] = 1.0 if features.numel() > 0 else 0.0

        # Combine all geological features
        combined_features: torch.Tensor | None = None
        if aggregated:
            feature_list = list(aggregated.values())
            # Stack and mean-pool across detectors
            stacked = torch.stack(feature_list, dim=0)
            combined_features = stacked.mean(dim=0)

            # Add combined scalars
            if enable_gosnn_synapse:
                omni_scalars["omni_geological_combined_mean"] = float(
                    combined_features.mean().item()
                )
                omni_scalars["omni_geological_detector_count"] = float(len(aggregated))

        # Compute coverage statistics
        coverage = {
            "total_geological_detectors": len(geological_names),
            "active_detectors": len(aggregated),
            "coverage_ratio": len(aggregated) / len(geological_names) if geological_names else 0.0,
            "detectors_active": list(aggregated.keys()),
        }

        return {
            "aggregated_features": aggregated,
            "combined_features": combined_features,
            "gosnn_scalars": omni_scalars if enable_gosnn_synapse else {},
            "detector_coverage": coverage,
            "target_dimension": target_dim,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics."""
        total_invocations = sum(d.total_invocations for d in self._detectors.values())
        total_errors = sum(d.error_count for d in self._detectors.values())

        return {
            "total_detectors": len(self._detectors),
            "categories": {
                cat.value: len(names) for cat, names in self._category_index.items() if names
            },
            "total_invocations": total_invocations,
            "total_errors": total_errors,
            "error_rate": total_errors / total_invocations if total_invocations > 0 else 0,
            "detectors": {name: info.to_dict() for name, info in self._detectors.items()},
        }

    def get_feature_dimensions(self) -> dict[str, int | None]:
        """Get feature dimensions for all detectors."""
        return {name: info.feature_dim for name, info in self._detectors.items()}

    def health_check(self) -> dict[str, Any]:
        """Perform health check on all detectors.

        Returns:
            Dictionary with health status per detector
        """
        health = {}
        for name, info in self._detectors.items():
            detector = info.instance

            # Check if detector has health_check method
            if hasattr(detector, "health_check"):
                try:
                    health[name] = detector.health_check()
                except Exception as e:
                    health[name] = {"healthy": False, "error": str(e)}
            else:
                # Basic health: has required methods
                health[name] = {
                    "healthy": True,
                    "has_extract_features": hasattr(detector, "extract_features"),
                    "has_predict": hasattr(detector, "predict"),
                    "error_count": info.error_count,
                }

        return health

    def __del__(self) -> None:
        """Cleanup resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)


# Singleton instance for global access
_global_registry: DetectorRegistry | None = None


def get_global_registry() -> DetectorRegistry:
    """Get or create global detector registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = DetectorRegistry(auto_discover=True)
    return _global_registry
