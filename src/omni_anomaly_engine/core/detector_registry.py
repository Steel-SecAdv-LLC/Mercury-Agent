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
Unified Detector Registry - Bridge connecting detectors across domains for fusion.

This module provides a central registry for all anomaly detectors and models,
enabling:
- Unified detector discovery and registration
- Parallel feature extraction across all detectors
- Feature aggregation for the fusion pipeline
- Detector health monitoring and statistics

Architecture:
    DetectorRegistry connects to the following detector categories:

    Base Detectors (detectors/):
        - StatisticalAnomalyDetector
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

Example:
    >>> from omni_anomaly_engine.core.detector_registry import DetectorRegistry
    >>> registry = DetectorRegistry()
    >>> registry.auto_discover()
    >>> features = registry.extract_all_features(data)
    >>> print(f"Collected features from {len(features)} detectors")
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
        }


@dataclass
class FeatureExtractionResult:
    """Result from feature extraction."""

    detector_name: str
    features: np.ndarray | torch.Tensor | None
    scores: np.ndarray | None
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


class DetectorRegistry:
    """
    Central registry for all anomaly detectors and models.

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
        self._executor: ThreadPoolExecutor | None = None

        if auto_discover:
            self.auto_discover_detectors()

    def register(
        self,
        name: str,
        detector: Any,
        category: DetectorCategory = DetectorCategory.BASE,
        feature_dim: int | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Register a detector with the registry.

        Args:
            name: Unique detector name
            detector: Detector instance (must have extract_features and predict methods)
            category: Detector category for organization
            feature_dim: Output feature dimension (auto-detected if None)
            description: Human-readable description
            tags: Optional tags for filtering
        """
        if not hasattr(detector, "extract_features") and not hasattr(detector, "predict"):
            logger.warning(f"Detector '{name}' does not have extract_features or predict methods")

        info = DetectorInfo(
            name=name,
            category=category,
            instance=detector,
            feature_dim=feature_dim,
            is_fitted=getattr(detector, "_is_fitted", False),
            description=description,
            tags=tags or [],
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

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def extract_features(
        self,
        name: str,
        data: np.ndarray | torch.Tensor | dict[str, Any],
    ) -> FeatureExtractionResult:
        """Extract features from a single detector.

        Args:
            name: Detector name
            data: Input data

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

        info = self._detectors[name]
        detector = info.instance
        start_time = time.perf_counter()

        try:
            # Try to fit if needed
            if hasattr(detector, "is_fitted") and not detector.is_fitted():
                if hasattr(detector, "fit") and not isinstance(data, dict):
                    detector.fit(data)

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

            execution_time = (time.perf_counter() - start_time) * 1000

            # Update stats
            info.last_execution_time_ms = execution_time
            info.total_invocations += 1
            info.is_fitted = getattr(detector, "_is_fitted", True)

            return FeatureExtractionResult(
                detector_name=name,
                features=features,
                scores=scores,
                execution_time_ms=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            info.error_count += 1
            info.last_execution_time_ms = execution_time

            logger.debug(f"Feature extraction failed for '{name}': {e}")

            return FeatureExtractionResult(
                detector_name=name,
                features=None,
                scores=None,
                execution_time_ms=execution_time,
                success=False,
                error=str(e),
            )

    def extract_all_features(
        self,
        data: np.ndarray | torch.Tensor | dict[str, Any],
        parallel: bool = True,
        categories: list[DetectorCategory] | None = None,
        detector_names: list[str] | None = None,
    ) -> dict[str, FeatureExtractionResult]:
        """Extract features from all registered detectors.

        Args:
            data: Input data
            parallel: Use parallel execution
            categories: Filter by categories (None = all)
            detector_names: Filter by specific names (None = all)

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

        results: dict[str, FeatureExtractionResult] = {}

        if parallel and len(names) > 1:
            executor = self._get_executor()
            futures = {executor.submit(self.extract_features, name, data): name for name in names}

            for future in as_completed(futures, timeout=self.timeout_seconds):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = FeatureExtractionResult(
                        detector_name=name,
                        features=None,
                        scores=None,
                        execution_time_ms=0,
                        success=False,
                        error=str(e),
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

        The golden ratio (1.618...) has been shown to optimize information
        distribution in neural networks and signal processing.
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

        Returns:
            Number of detectors registered
        """
        registered_count = 0

        # Base detectors
        try:
            from omni_anomaly_engine.detectors.statistical import StatisticalAnomalyDetector

            self.register(
                "statistical",
                StatisticalAnomalyDetector(),
                DetectorCategory.BASE,
                description="Z-score, percentile, MAD-based detection",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.temporal import TemporalAnomalyDetector

            self.register(
                "temporal",
                TemporalAnomalyDetector(),
                DetectorCategory.BASE,
                description="Time-series patterns and seasonal anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.spatial import SpatialAnomalyDetector

            self.register(
                "spatial",
                SpatialAnomalyDetector(),
                DetectorCategory.BASE,
                description="Geographic and spatial relationship anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.dimensional import DimensionalAnalyzer

            self.register(
                "dimensional",
                DimensionalAnalyzer(),
                DetectorCategory.BASE,
                description="High-dimensional data anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.directive import SigmaDirectiveDetector

            self.register(
                "directive",
                SigmaDirectiveDetector(),
                DetectorCategory.BASE,
                description="Rule-based sigma detection",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.graph_based import GraphAnomalyDetector

            self.register(
                "graph_based",
                GraphAnomalyDetector(),
                DetectorCategory.BASE,
                description="Graph structure anomaly detection",
            )
            registered_count += 1
        except ImportError:
            pass

        # Specialized models
        try:
            from omni_anomaly_engine.models.quantum import QuantumAnomalyModel

            self.register(
                "quantum",
                QuantumAnomalyModel(),
                DetectorCategory.MODEL,
                description="Quantum state anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.astrophysical import AstrophysicalAnomalyModel

            self.register(
                "astrophysical",
                AstrophysicalAnomalyModel(),
                DetectorCategory.MODEL,
                description="Cosmic signal anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.biometric import BiometricAnomalyModel

            self.register(
                "biometric",
                BiometricAnomalyModel(),
                DetectorCategory.MODEL,
                description="Face/biometric anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.affective import AffectiveAnomalyModel

            self.register(
                "affective",
                AffectiveAnomalyModel(),
                DetectorCategory.MODEL,
                description="Emotional state anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.neural import NeuralCognitiveModel

            self.register(
                "neural_cognitive",
                NeuralCognitiveModel(),
                DetectorCategory.MODEL,
                description="Brain activity anomalies",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.consciousness import ConsciousnessPreservationModel

            self.register(
                "consciousness",
                ConsciousnessPreservationModel(),
                DetectorCategory.MODEL,
                description="Consciousness preservation analysis",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.neurosymbolic import NeurosymbolicEngine

            self.register(
                "neurosymbolic",
                NeurosymbolicEngine(),
                DetectorCategory.NEUROSYMBOLIC,
                description="Hybrid neural-symbolic reasoning",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.chemistry import ChemistryAnomalyDetector

            self.register(
                "chemistry",
                ChemistryAnomalyDetector(),
                DetectorCategory.MODEL,
                description="Chemical anomaly detection",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.parapsychology import ParapsychologyDetector

            self.register(
                "parapsychology",
                ParapsychologyDetector(),
                DetectorCategory.MODEL,
                description="Psi phenomena detection",
            )
            registered_count += 1
        except ImportError:
            pass

        # Security detectors
        try:
            from omni_anomaly_engine.security.threat_detection import ThreatDetector

            self.register(
                "threat_detection",
                ThreatDetector(),
                DetectorCategory.SECURITY,
                description="Security threat detection",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.security.psyop import PSYOPAnalyzer

            self.register(
                "psyop",
                PSYOPAnalyzer(),
                DetectorCategory.INTELLIGENCE,
                description="Psychological operations analysis",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.security.intelligence_fusion import IntelligenceFusionEngine

            self.register(
                "intelligence_fusion",
                IntelligenceFusionEngine(),
                DetectorCategory.INTELLIGENCE,
                description="Multi-source intelligence fusion",
            )
            registered_count += 1
        except ImportError:
            pass

        # Space detectors
        try:
            from omni_anomaly_engine.space.schumann_resonance import SchumannResonanceDetector

            self.register(
                "schumann_resonance",
                SchumannResonanceDetector(),
                DetectorCategory.SPACE,
                description="Earth resonance detection",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.space.solar_storm_detector import SolarStormDetector

            self.register(
                "solar_storm",
                SolarStormDetector(),
                DetectorCategory.SPACE,
                description="Solar storm prediction",
            )
            registered_count += 1
        except ImportError:
            pass

        # Medical detectors
        try:
            from omni_anomaly_engine.medical.abms_disciplines import ABMSDisciplineDetector

            self.register(
                "medical_abms",
                ABMSDisciplineDetector(),
                DetectorCategory.MEDICAL,
                description="Medical discipline detection",
            )
            registered_count += 1
        except ImportError:
            pass

        # Geological detectors
        try:
            from omni_anomaly_engine.detectors.geological.volcanic import VolcanicEruptionDetector

            self.register(
                "volcanic",
                VolcanicEruptionDetector(),
                DetectorCategory.GEOLOGICAL,
                description="Volcanic eruption prediction",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.geological.landslide import LandslideDetector

            self.register(
                "landslide",
                LandslideDetector(),
                DetectorCategory.GEOLOGICAL,
                description="Landslide prediction",
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.geological.wildfire import WildfireDetector

            self.register(
                "wildfire",
                WildfireDetector(),
                DetectorCategory.GEOLOGICAL,
                description="Wildfire prediction",
            )
            registered_count += 1
        except ImportError:
            pass

        # Economic detectors
        try:
            from omni_anomaly_engine.detectors.economic.financial_crisis_detector import (
                FinancialCrisisDetector,
            )

            self.register(
                "financial_crisis",
                FinancialCrisisDetector(),
                DetectorCategory.ECONOMIC,
                description="Financial crisis prediction",
            )
            registered_count += 1
        except ImportError:
            pass

        # Energy detectors
        try:
            from omni_anomaly_engine.detectors.energy.emp_detector import EMPDetector

            self.register(
                "emp",
                EMPDetector(),
                DetectorCategory.ENERGY,
                description="Electromagnetic pulse detection",
            )
            registered_count += 1
        except ImportError:
            pass

        # Marine detectors
        try:
            from omni_anomaly_engine.detectors.marine.biodiversity_detector import (
                MarineBiodiversityDetector,
            )

            self.register(
                "marine_biodiversity",
                MarineBiodiversityDetector(),
                DetectorCategory.MARINE,
                description="Marine biodiversity threat detection",
            )
            registered_count += 1
        except ImportError:
            pass

        # =====================================================================
        # SOTA Visual Anomaly Detection (PatchCore, PaDiM, STFPM, etc.)
        # =====================================================================
        try:
            from omni_anomaly_engine.detectors.visual import PatchCoreDetector

            self.register(
                "patchcore",
                PatchCoreDetector(),
                DetectorCategory.VISUAL,
                description="Memory bank + coreset subsampling anomaly detection",
                tags=["visual", "sota", "memory-bank"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.visual import PaDiMDetector

            self.register(
                "padim",
                PaDiMDetector(),
                DetectorCategory.VISUAL,
                description="Patch-wise Mahalanobis distance anomaly detection",
                tags=["visual", "sota", "mahalanobis"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.visual import STFPMDetector

            self.register(
                "stfpm",
                STFPMDetector(),
                DetectorCategory.VISUAL,
                description="Student-Teacher Feature Pyramid Matching",
                tags=["visual", "sota", "teacher-student"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.visual import ReverseDistillationDetector

            self.register(
                "reverse_distillation",
                ReverseDistillationDetector(),
                DetectorCategory.VISUAL,
                description="Reverse knowledge distillation with OCE bottleneck",
                tags=["visual", "sota", "distillation"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.visual import CFlowDetector

            self.register(
                "cflow",
                CFlowDetector(),
                DetectorCategory.VISUAL,
                description="Conditional normalizing flow anomaly detection",
                tags=["visual", "sota", "normalizing-flow"],
            )
            registered_count += 1
        except ImportError:
            pass

        # =====================================================================
        # Vision-Language Model (VLM) Detectors
        # =====================================================================
        try:
            from omni_anomaly_engine.detectors.vlm import AnyAnomalyDetector

            self.register(
                "anyanomaly",
                AnyAnomalyDetector(),
                DetectorCategory.VLM,
                description="Zero-shot customizable VAD with LVLM (WACV 2026)",
                tags=["vlm", "zero-shot", "lvlm", "sota"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.detectors.vlm import LAVADDetector

            self.register(
                "lavad",
                LAVADDetector(),
                DetectorCategory.VLM,
                description="Training-free LLM-based VAD (CVPR 2024)",
                tags=["vlm", "training-free", "llm", "sota"],
            )
            registered_count += 1
        except ImportError:
            pass

        # =====================================================================
        # Foundation Model Adapters (TimeGPT, Chronos, Matrix Profile)
        # =====================================================================
        try:
            from omni_anomaly_engine.models.foundation import TimeGPTAdapter

            self.register(
                "timegpt",
                TimeGPTAdapter(),
                DetectorCategory.FOUNDATION,
                description="Nixtla TimeGPT foundation model adapter",
                tags=["foundation", "time-series", "api"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.foundation import ChronosAdapter

            self.register(
                "chronos",
                ChronosAdapter(),
                DetectorCategory.FOUNDATION,
                description="Amazon Chronos time-series foundation model",
                tags=["foundation", "time-series", "local"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.foundation import MatrixProfileAdapter

            self.register(
                "matrix_profile",
                MatrixProfileAdapter(),
                DetectorCategory.FOUNDATION,
                description="STUMPY Matrix Profile for time-series anomalies",
                tags=["foundation", "time-series", "matrix-profile"],
            )
            registered_count += 1
        except ImportError:
            pass

        try:
            from omni_anomaly_engine.models.foundation import FoundationEnsemble

            self.register(
                "foundation_ensemble",
                FoundationEnsemble(),
                DetectorCategory.FOUNDATION,
                description="Ensemble of foundation model adapters",
                tags=["foundation", "ensemble"],
            )
            registered_count += 1
        except ImportError:
            pass

        # =====================================================================
        # Knowledge Distillation Methods
        # =====================================================================
        try:
            from omni_anomaly_engine.ml.distillation import DualStudentDistillation

            self.register(
                "dual_student",
                DualStudentDistillation(),
                DetectorCategory.DISTILLATION,
                description="Dual-student knowledge distillation for AD",
                tags=["distillation", "sota", "dual-student"],
            )
            registered_count += 1
        except ImportError:
            pass

        logger.info(f"Auto-discovered and registered {registered_count} detectors")
        return registered_count

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

    def __del__(self):
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
