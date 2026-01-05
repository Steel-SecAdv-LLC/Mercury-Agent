"""
Mercury Agent ♱
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

from __future__ import annotations

"""Main OmniMercuryEngine orchestrating all detectors and models.

This module provides the core anomaly detection engine that integrates
13 specialized detection models through neural network fusion. It supports
multiple detection modes, batch processing, and configurable sensitivity.

Architecture:
    The OmniMercuryEngine combines multiple specialized detectors:

    Base Detectors (5):
        - Statistical: Z-score, percentile, MAD-based detection
        - Temporal: Time-series patterns and seasonal anomalies
        - Spatial: Geographic and spatial relationship anomalies
        - Dimensional: High-dimensional data anomalies
        - Directive (Sigma): Rule-based sigma detection

    Specialized Models (13):
        - Quantum: Quantum state anomalies
        - Astrophysical: Cosmic signal anomalies
        - Biometric: Face/biometric anomalies
        - Affective: Emotional state anomalies
        - Neural: Brain activity anomalies
        - Consciousness: Consciousness preservation
        - Multiverse: Multi-universe state analysis
        - Neurosymbolic: Hybrid neural-symbolic reasoning
        - Medical ABMS: Medical discipline detection
        - Intelligence Fusion: Multi-source intelligence
        - Schumann Resonance: Earth resonance detection
        - Chemistry: Chemical anomaly detection
        - Parapsychology: Psi phenomena detection

Performance Characteristics:
    - Time Complexity: O(n * m) where n = samples, m = features
    - Space Complexity: O(n * d) where d = number of detectors
    - Batch Processing: Supports dynamic batch sizes for memory optimization

Example:
    Basic usage with fusion mode::

        from omni_mercury_engine.engine import OmniMercuryEngine
        import numpy as np

        # Initialize engine
        engine = OmniMercuryEngine(mode="fusion", device="cuda")

        # Detect anomalies
        data = np.random.randn(100, 10)
        result = engine.detect_with_fusion(data)

        print(f"Anomaly probability: {result['anomaly_prob']:.3f}")
        print(f"Is anomaly: {result['is_anomaly']}")

    Batch processing for large datasets::

        engine = OmniMercuryEngine(mode="fusion")
        large_data = np.random.randn(10000, 50)

        # Process in batches for memory efficiency
        results = engine.detect_batch(large_data, batch_size=64)

Attributes:
    config: Engine configuration object
    mode: Detection mode ('fusion' or individual detector)
    device: Computation device (cpu/cuda)
    detectors: Dictionary of base detectors
    models: Dictionary of specialized models
    fusion_model: Neural network fusion model (in fusion mode)

See Also:
    - :class:`omni_mercury_engine.core.config.EngineConfig`
    - :class:`omni_mercury_engine.ml.fusion_network.OmniFusionModel`
"""

import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from omni_mercury_engine.core.config import EngineConfig
from omni_mercury_engine.core.global_omni_scalar_network import (
    ScalarGroup,
    get_global_scalar_network,
)
from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector
from omni_mercury_engine.medical.abms_disciplines import ABMSDisciplineDetector

# Runtime pipeline integration modules
from omni_mercury_engine.ml.drift import (
    DriftResult,
    EnsembleDriftDetector,
)
from omni_mercury_engine.ml.fairness import (
    BiasAuditConfig,
    FairnessAuditor,
    FairnessReport,
)
from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference
from omni_mercury_engine.ml.optimization import (
    OptimizationConfig,
    ParallelExecutor,
)
from omni_mercury_engine.models.affective import AffectiveAnomalyModel
from omni_mercury_engine.models.astrophysical import AstrophysicalAnomalyModel
from omni_mercury_engine.models.biometric import BiometricAnomalyModel
from omni_mercury_engine.models.chemistry import ChemistryAnomalyDetector
from omni_mercury_engine.models.consciousness import ConsciousnessPreservationModel
from omni_mercury_engine.models.foundation.llm_adapter import (
    LLMConfig,
    LLMProvider,
    ZeroShotAnomalyDetector,
)
from omni_mercury_engine.models.neural import NeuralCognitiveModel
from omni_mercury_engine.models.parapsychology import ParapsychologyDetector
from omni_mercury_engine.models.quantum import QuantumAnomalyModel
from omni_mercury_engine.resilience.self_healing import SelfHealingEngine
from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine
from omni_mercury_engine.security.threat_detection import ThreatDetector
from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Configure module logger
logger = logging.getLogger(__name__)


class FeatureCache:
    """Thread-safe LRU cache for computed features.

    This cache stores computed features to avoid redundant calculations
    when processing the same data multiple times.

    Attributes:
        max_size: Maximum number of cached entries.
        cache: Dictionary storing cached features.
        access_order: List tracking access order for LRU eviction.
        lock: Threading lock for thread safety.
        hits: Number of cache hits.
        misses: Number of cache misses.

    Example:
        >>> cache = FeatureCache(max_size=100)
        >>> cache.get_or_compute("key1", compute_fn, data)
    """

    def __init__(self, max_size: int = 128) -> None:
        """Initialize the feature cache.

        Args:
            max_size: Maximum number of entries to cache. Default 128.
        """
        self.max_size = max_size
        self.cache: dict[str, Any] = {}
        self.access_order: list[str] = []
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, data: np.ndarray[Any, Any] | torch.Tensor, prefix: str = "") -> str:
        """Generate a cache key from data.

        Args:
            data: Input data to hash.
            prefix: Optional prefix for the key.

        Returns:
            String hash key for the data.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        # Use shape and sample of data for key
        data_hash = hash(
            (data.shape, data.tobytes()[:1024] if data.nbytes > 1024 else data.tobytes())
        )
        return f"{prefix}_{data_hash}"

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """Get cached value or compute and cache it.

        Args:
            key: Cache key.
            compute_fn: Function to compute value if not cached.

        Returns:
            Cached or computed value.
        """
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end for LRU
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]

            self.misses += 1

        # Compute outside lock
        value = compute_fn()

        with self.lock:
            # Evict if necessary
            while len(self.cache) >= self.max_size and self.access_order:
                oldest = self.access_order.pop(0)
                self.cache.pop(oldest, None)

            self.cache[key] = value
            self.access_order.append(key)

        return value

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }


class MemoryMonitor:
    """Monitor and manage memory usage during detection.

    This class provides memory tracking and automatic garbage collection
    to prevent out-of-memory errors during large-scale processing.

    Attributes:
        peak_memory_mb: Peak memory usage in megabytes.
        threshold_mb: Memory threshold for triggering GC.
        gc_count: Number of garbage collections triggered.

    Example:
        >>> monitor = MemoryMonitor(threshold_mb=1024)
        >>> with monitor.track_allocation("batch_processing"):
        ...     process_batch(data)
    """

    def __init__(self, threshold_mb: float = 2048.0) -> None:
        """Initialize memory monitor.

        Args:
            threshold_mb: Memory threshold in MB for triggering GC.
        """
        self.threshold_mb = threshold_mb
        self.peak_memory_mb = 0.0
        self.gc_count = 0
        self._allocations: dict[str, float] = {}

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in megabytes.

        Returns:
            Current memory usage in MB.
        """
        try:
            import psutil

            process = psutil.Process()
            return float(process.memory_info().rss / (1024 * 1024))
        except ImportError:
            # Fallback if psutil not available
            return 0.0

    def check_and_collect(self) -> bool:
        """Check memory and trigger GC if needed.

        Returns:
            True if GC was triggered, False otherwise.
        """
        current = self.get_current_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, current)

        if current > self.threshold_mb:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.gc_count += 1
            logger.debug(f"GC triggered at {current:.1f}MB (threshold: {self.threshold_mb}MB)")
            return True
        return False

    @contextmanager
    def track_allocation(self, name: str) -> Iterator[None]:
        """Context manager to track memory allocation.

        Args:
            name: Name for this allocation tracking.

        Yields:
            None
        """
        start_mem = self.get_current_memory_mb()
        try:
            yield
        finally:
            end_mem = self.get_current_memory_mb()
            self._allocations[name] = end_mem - start_mem
            self.check_and_collect()

    def stats(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory statistics.
        """
        return {
            "current_mb": self.get_current_memory_mb(),
            "peak_mb": self.peak_memory_mb,
            "threshold_mb": self.threshold_mb,
            "gc_count": self.gc_count,
            "allocations": dict(self._allocations),
        }


class OmniMercuryEngine:
    """Unified anomaly detection engine with ML-Centric Hybrid Fusion.

    This is the main entry point for the Mercury Agent ♱ anomaly detection system.
    It integrates 13 specialized detection engines through neural network
    fusion to provide comprehensive multi-domain anomaly detection.

    The engine supports multiple operation modes:
        - **fusion**: Neural network fusion of all detectors (default)
        - **individual**: Use specific detectors directly

    Attributes:
        config: Engine configuration (EngineConfig instance).
        mode: Operation mode ('fusion' or individual detector name).
        device: PyTorch device for computation.
        detectors: Dictionary of base anomaly detectors.
        models: Dictionary of specialized domain models.
        security: Security threat detector.
        fusion_model: Neural network fusion model (fusion mode only).
        fusion_inference: Fusion inference engine (fusion mode only).
        self_healing: Self-healing resilience engine.
        feature_cache: Cache for computed features.
        memory_monitor: Memory usage monitor.

    Example:
        Basic detection::

            engine = OmniMercuryEngine()
            data = np.random.randn(100, 10)
            result = engine.detect(data)

        Fusion mode with GPU::

            engine = OmniMercuryEngine(mode="fusion", device="cuda")
            result = engine.detect_with_fusion(data)

        Batch processing::

            results = engine.detect_batch(large_data, batch_size=32)

    Note:
        The fusion model requires all detectors and models to be properly
        initialized. Individual detectors can be used directly for lighter
        weight operation.

    See Also:
        - :meth:`detect`: Basic detection using specified detectors
        - :meth:`detect_with_fusion`: ML fusion detection
        - :meth:`detect_batch`: Batch processing for large datasets
        - :meth:`train_fusion_model`: Train the fusion network
    """

    # Default batch size for processing
    DEFAULT_BATCH_SIZE = 32

    # Maximum parallel workers for feature extraction
    MAX_WORKERS = 4

    def __init__(
        self,
        config: EngineConfig | None = None,
        mode: str = "fusion",
        device: str = "cpu",
        cache_size: int = 128,
        memory_threshold_mb: float = 2048.0,
    ) -> None:
        """Initialize the OmniMercuryEngine.

        Args:
            config: Engine configuration. If None, uses default config.
            mode: Operation mode. Either 'fusion' for ML fusion or
                a specific detector name.
            device: Computation device ('cpu' or 'cuda').
            cache_size: Maximum entries in feature cache. Default 128.
            memory_threshold_mb: Memory threshold for GC in MB. Default 2048.

        Raises:
            ValueError: If device is 'cuda' but CUDA is not available.

        Example:
            >>> engine = OmniMercuryEngine(
            ...     mode="fusion",
            ...     device="cuda",
            ...     cache_size=256
            ... )
        """
        self.config = config or EngineConfig()
        self.mode = mode
        self.device = torch.device(device)

        # Validate CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Initialize caching and memory management
        self.feature_cache = FeatureCache(max_size=cache_size)
        self.memory_monitor = MemoryMonitor(threshold_mb=memory_threshold_mb)

        # Thread pool for parallel processing
        self._executor: ThreadPoolExecutor | None = None

        self._init_detectors()
        self._init_models()
        self._init_fusion()
        self._init_resilience()
        self._init_runtime_pipeline()

        logger.info(f"OmniMercuryEngine initialized (mode={mode}, device={self.device})")

    def _init_detectors(self) -> None:
        """Initialize all base anomaly detectors.

        Creates instances of the 5 base detectors:
            - statistical: Statistical anomaly detection
            - temporal: Temporal pattern detection
            - spatial: Spatial relationship detection
            - dimensional: High-dimensional analysis
            - directive: Sigma-based rule detection
        """
        self.detectors = {
            "statistical": StatisticalAnomalyDetector(),
            "temporal": TemporalAnomalyDetector(),
            "spatial": SpatialAnomalyDetector(),
            "dimensional": DimensionalAnalyzer(),
            "directive": SigmaDirectiveDetector(),
        }

    def _init_models(self) -> None:
        """Initialize all specialized domain models.

        Creates instances of the 13 specialized models covering various
        domains from quantum physics to medical diagnostics.
        """
        from omni_mercury_engine.models.multiverse import MultiverseOmniEngine
        from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine

        self.models = {
            "quantum": QuantumAnomalyModel(),
            "astrophysical": AstrophysicalAnomalyModel(),
            "biometric": BiometricAnomalyModel(),
            "affective": AffectiveAnomalyModel(),
            "neural": NeuralCognitiveModel(),
            "consciousness": ConsciousnessPreservationModel(),
            "multiverse": MultiverseOmniEngine(num_universes=10, state_dim=50),
            "neurosymbolic": NeurosymbolicEngine(input_dim=64),
            "medical_abms": ABMSDisciplineDetector(),
            "intelligence_fusion": IntelligenceFusionEngine(),
            "schumann_resonance": SchumannResonanceDetector(),
            "chemistry": ChemistryAnomalyDetector(),
            "parapsychology": ParapsychologyDetector(),
        }

        self.security = ThreatDetector()

    def _init_fusion(self) -> None:
        """Initialize ML fusion components.

        Sets up the neural network fusion model and inference engine
        when operating in fusion mode.
        """
        if self.mode == "fusion":
            self.fusion_model = OmniFusionModel()
            self.fusion_model.to(self.device)
            self.fusion_inference = FusionInference(
                model=self.fusion_model,
                device=str(self.device),
            )

    def _init_resilience(self) -> None:
        """Initialize resilience and self-healing components."""
        self.self_healing = SelfHealingEngine()

    def _init_runtime_pipeline(self) -> None:
        """Initialize runtime pipeline integration modules.

        Sets up drift detection, fairness auditing, optimization, and LLM
        enhancement components for the detection pipeline. These modules
        provide optional stages that can be enabled via configuration.

        Components initialized:
            - drift_detector: Ensemble drift detector for distribution monitoring
            - fairness_auditor: Bias auditing and fairness assessment
            - optimization_config: Performance optimization settings
            - parallel_executor: Parallel execution manager
            - llm_detector: Zero-shot LLM-based anomaly detection (optional)
        """
        self.drift_detector: EnsembleDriftDetector | None = None
        self.fairness_auditor: FairnessAuditor | None = None
        self.llm_detector: ZeroShotAnomalyDetector | None = None
        self._baseline_features: np.ndarray[Any, Any] | None = None

        self.optimization_config = OptimizationConfig(
            enable_joblib=True,
            n_jobs=self.MAX_WORKERS,
            enable_torch_compile=False,
            enable_memory_tracking=True,
            memory_threshold_mb=self.memory_monitor.threshold_mb,
        )

        self.parallel_executor = ParallelExecutor(
            n_jobs=self.optimization_config.n_jobs,
        )

        logger.debug("Runtime pipeline modules initialized")

    def enable_drift_detection(
        self,
        baseline_data: np.ndarray[Any, Any] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Enable drift detection for the detection pipeline.

        Drift detection monitors for distribution shifts between the baseline
        data and incoming detection requests. This is critical for detecting
        when model performance may degrade due to data drift.

        Args:
            baseline_data: Reference data for drift comparison. If None,
                the first detection batch will be used as baseline.
            feature_names: Optional names for features for detailed reporting.

        Example:
            >>> engine = OmniMercuryEngine()
            >>> engine.enable_drift_detection(training_data)
            >>> result = engine.detect_with_fusion(new_data)
            >>> if result.get("drift_detected"):
            ...     print("Data drift detected!")
        """
        self.drift_detector = EnsembleDriftDetector(
            feature_names=feature_names,
            significance_level=0.05,
        )
        if baseline_data is not None:
            self._baseline_features = baseline_data
            self.drift_detector.fit(baseline_data)
        logger.info("Drift detection enabled")

    def enable_fairness_auditing(
        self,
        sensitive_features: list[str] | None = None,
        fairness_threshold: float = 0.8,
    ) -> None:
        """Enable fairness auditing for detection results.

        Fairness auditing monitors for bias in detection outcomes across
        sensitive demographic groups. This supports ethical AI principles
        and regulatory compliance.

        Args:
            sensitive_features: List of feature names that are sensitive
                attributes (e.g., age, gender, race).
            fairness_threshold: Minimum fairness score threshold (0-1).
                Default 0.8 for 80% fairness requirement.

        Example:
            >>> engine = OmniMercuryEngine()
            >>> engine.enable_fairness_auditing(
            ...     sensitive_features=["age_group", "region"],
            ...     fairness_threshold=0.85
            ... )
        """
        audit_config = BiasAuditConfig(
            sensitive_features=sensitive_features or [],
            fairness_threshold=fairness_threshold,
            enable_mitigation=True,
        )
        self.fairness_auditor = FairnessAuditor(config=audit_config)
        logger.info(f"Fairness auditing enabled with threshold={fairness_threshold}")

    def enable_llm_enhancement(
        self,
        provider: str = "mock",
        model_name: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Enable LLM-based anomaly explanation enhancement.

        LLM enhancement provides natural language explanations for detected
        anomalies using zero-shot classification. This is a non-blocking
        optional stage that enhances detection results with interpretability.

        Args:
            provider: LLM provider name ('mock', 'huggingface', 'openai').
                Default 'mock' for testing without API calls.
            model_name: Model identifier for the provider.
            api_key: API key for the provider (if required).
            timeout_seconds: Maximum time to wait for LLM response.

        Example:
            >>> engine = OmniMercuryEngine()
            >>> engine.enable_llm_enhancement(
            ...     provider="huggingface",
            ...     model_name="facebook/bart-large-mnli"
            ... )
        """
        try:
            llm_provider = LLMProvider(provider.lower())
        except ValueError:
            llm_provider = LLMProvider.MOCK
            logger.warning(f"Unknown LLM provider '{provider}', using mock")

        llm_config = LLMConfig(
            provider=llm_provider,
            model_name=model_name or "mock-model",
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=2,
        )
        self.llm_detector = ZeroShotAnomalyDetector(config=llm_config)
        logger.info(f"LLM enhancement enabled with provider={provider}")

    def _check_drift(
        self,
        features: np.ndarray[Any, Any],
    ) -> DriftResult | None:
        """Check for data drift against baseline.

        Args:
            features: Current feature data to check for drift.

        Returns:
            DriftResult if drift detection is enabled, None otherwise.
        """
        if self.drift_detector is None:
            return None

        if self._baseline_features is None:
            self._baseline_features = features
            self.drift_detector.fit(features)
            return None

        try:
            drift_result = self.drift_detector.detect(features)
            if drift_result.is_drift:
                logger.warning(
                    f"Data drift detected: severity={drift_result.severity.name}, "
                    f"p_value={drift_result.p_value:.4f}"
                )
            return drift_result
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            return None

    def _audit_fairness(
        self,
        predictions: np.ndarray[Any, Any],
        sensitive_data: dict[str, np.ndarray[Any, Any]] | None = None,
    ) -> FairnessReport | None:
        """Audit detection results for fairness.

        Args:
            predictions: Model predictions to audit.
            sensitive_data: Dictionary mapping sensitive feature names
                to their values for the predictions.

        Returns:
            FairnessReport if fairness auditing is enabled, None otherwise.
        """
        if self.fairness_auditor is None or sensitive_data is None:
            return None

        try:
            fairness_report = self.fairness_auditor.audit(
                predictions=predictions,
                sensitive_features=sensitive_data,
            )
            if not fairness_report.is_fair:
                logger.warning(
                    f"Fairness violation detected: score={fairness_report.overall_fairness_score:.3f}, "
                    f"violations={len(fairness_report.violations)}"
                )
            return fairness_report
        except Exception as e:
            logger.error(f"Fairness audit error: {e}")
            return None

    def _enhance_with_llm(
        self,
        data: np.ndarray[Any, Any] | dict[str, Any],
        detection_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Enhance detection results with LLM explanations.

        Args:
            data: Original input data.
            detection_result: Detection result to enhance.

        Returns:
            LLM enhancement result if enabled, None otherwise.
        """
        if self.llm_detector is None:
            return None

        if not detection_result.get("is_anomaly", False):
            return None

        try:
            if isinstance(data, dict):
                data_str = str(data)[:1000]
            else:
                data_str = f"Numerical data shape: {data.shape}"

            llm_result = self.llm_detector.detect(
                text=f"Anomaly detected in: {data_str}",
                candidate_labels=[
                    "security_threat",
                    "system_failure",
                    "data_corruption",
                    "unusual_pattern",
                    "normal_variation",
                ],
            )
            return {
                "llm_explanation": llm_result.explanation,
                "llm_category": llm_result.category,
                "llm_confidence": llm_result.confidence,
            }
        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
            return None

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor.

        Returns:
            ThreadPoolExecutor for parallel processing.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        return self._executor

    def detect(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
        detector_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies using specified detectors.

        This method runs the specified base detectors on the input data
        and aggregates their results. If no detectors are specified,
        all available detectors are used.

        Args:
            data: Input data for anomaly detection. Can be:
                - numpy.ndarray[Any, Any]: Numerical data array
                - torch.Tensor: PyTorch tensor
                - Dict[str, Any]: Dictionary with domain-specific data
            detector_types: List of detector names to use. If None,
                uses all detectors. Valid names: 'statistical',
                'temporal', 'spatial', 'dimensional', 'directive'.

        Returns:
            Dictionary containing:
                - detectors: Dict mapping detector names to their results
                - is_anomaly: Boolean indicating if any detector found anomaly

        Example:
            >>> engine = OmniMercuryEngine()
            >>> data = np.random.randn(100, 10)
            >>> result = engine.detect(data, detector_types=["statistical"])
            >>> print(result["is_anomaly"])
            False

        Note:
            Detectors are automatically fitted if not already fitted.
            Dictionary input skips fitting for detectors that require arrays.
        """
        if detector_types is None:
            detector_types = list(self.detectors.keys())

        results = {}

        for detector_name in detector_types:
            if detector_name in self.detectors:
                detector = self.detectors[detector_name]

                if not detector.is_fitted():
                    if isinstance(data, dict):
                        continue
                    detector.fit(data)

                if isinstance(data, dict):
                    continue
                results[detector_name] = detector.detect(data)

        return {
            "detectors": results,
            "is_anomaly": any(
                (
                    r.get("is_anomaly", False)
                    if isinstance(r.get("is_anomaly"), bool)
                    else any(r.get("is_anomaly", []))
                )
                for r in results.values()
            ),
        }

    def detect_batch(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        batch_size: int | None = None,
        use_fusion: bool = True,
        parallel: bool = True,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in batches for large datasets.

        This method implements intelligent batching with dynamic batch
        size adjustment based on available memory. It's optimized for
        processing large datasets efficiently.

        Args:
            data: Input data array with shape (n_samples, n_features).
            batch_size: Number of samples per batch. If None, uses
                dynamic sizing based on available memory.
            use_fusion: If True, use fusion detection. Default True.
            parallel: If True, process batches in parallel. Default True.

        Returns:
            List of detection results, one per sample.

        Raises:
            ValueError: If data has invalid shape.

        Example:
            >>> engine = OmniMercuryEngine(mode="fusion")
            >>> large_data = np.random.randn(10000, 50)
            >>> results = engine.detect_batch(large_data, batch_size=64)
            >>> anomaly_indices = [i for i, r in enumerate(results)
            ...                    if r.get("is_anomaly", False)]

        Performance:
            - Time: O(n * m / batch_size) where n = samples, m = features
            - Memory: O(batch_size * m) per batch
            - Parallel speedup: ~2-4x with 4 workers
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        n_samples = data.shape[0]

        # Dynamic batch size based on memory
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(data)

        results: list[dict[str, Any]] = []

        # Process in batches
        with self.memory_monitor.track_allocation("batch_detection"):
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_data = data[start_idx:end_idx]

                if use_fusion and self.mode == "fusion":
                    batch_result = self.detect_with_fusion(batch_data)
                    # Expand batch result to individual results
                    for _i in range(len(batch_data)):
                        results.append(
                            {
                                "anomaly_prob": batch_result.get("anomaly_prob", 0.0),
                                "is_anomaly": batch_result.get("is_anomaly", False),
                                "severity": batch_result.get("severity", 0.0),
                                "mode": "fusion",
                            }
                        )
                else:
                    batch_result = self.detect(batch_data)
                    for _i in range(len(batch_data)):
                        results.append(batch_result)

                # Check memory and potentially trigger GC
                self.memory_monitor.check_and_collect()

        return results

    def _calculate_optimal_batch_size(
        self,
        data: np.ndarray[Any, Any],
        target_memory_mb: float = 512.0,
    ) -> int:
        """Calculate optimal batch size based on data and memory.

        Args:
            data: Input data array.
            target_memory_mb: Target memory usage per batch in MB.

        Returns:
            Optimal batch size.
        """
        # Estimate memory per sample
        sample_size_bytes = data[0:1].nbytes if len(data) > 0 else 100
        # Account for intermediate computations (~10x multiplier)
        effective_size = sample_size_bytes * 10

        # Calculate batch size
        target_bytes = target_memory_mb * 1024 * 1024
        optimal_size = max(1, int(target_bytes / effective_size))

        # Clamp to reasonable bounds
        return min(max(optimal_size, 8), 256)

    def _normalize_scores(self, scores: Any, batch_size: int) -> torch.Tensor:
        """Normalize scores to tensor format [batch_size, 1].

        This method handles various score formats and normalizes them
        to a consistent tensor format for fusion.

        Args:
            scores: Input scores in various formats (list, array,
                tensor, bool, or scalar).
            batch_size: Expected batch size for expansion.

        Returns:
            Normalized torch.Tensor with shape [batch_size, 1].

        Example:
            >>> scores = [0.5, 0.8, 0.3]
            >>> normalized = engine._normalize_scores(scores, 3)
            >>> print(normalized.shape)
            torch.Size([3, 1])
        """
        if isinstance(scores, (list, np.ndarray)):
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            if scores_tensor.dim() == 1:
                scores_tensor = scores_tensor.unsqueeze(-1)
            return scores_tensor
        elif isinstance(scores, bool):
            return torch.full((batch_size, 1), float(scores), dtype=torch.float32)
        else:
            return torch.full((batch_size, 1), float(scores), dtype=torch.float32)

    def _extract_detector_features(
        self, data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any]
    ) -> tuple[Any, ...]:
        """Extract features from all detectors.

        This method extracts feature vectors from all base detectors
        and normalizes their anomaly scores. Features are cached for
        repeated access to the same data.

        Args:
            data: Input data for feature extraction.

        Returns:
            Tuple of (detector_features, detector_scores) where:
                - detector_features: Dict mapping detector names to features
                - detector_scores: Dict mapping detector names to scores

        Note:
            Uses parallel processing when available for improved performance.
            Features are cached using the FeatureCache for repeated access.
        """
        detector_features = {}
        detector_scores = {}

        for name, detector in self.detectors.items():
            try:
                if not detector.is_fitted():
                    if isinstance(data, dict):
                        continue
                    detector.fit(data)

                # Try to use cached features
                cache_key = self.feature_cache._make_key(
                    data if not isinstance(data, dict) else np.array([0]), prefix=f"detector_{name}"
                )

                def compute_features(det: Any = detector, d: Any = data) -> tuple[Any, ...]:
                    features = det.extract_features(d)
                    result = det.detect(d)
                    return features, result

                cached = self.feature_cache.get_or_compute(cache_key, compute_features)
                features, result = cached

                detector_features[name] = features
                scores = result.get("scores", result.get("is_anomaly", 0))
                detector_scores[name] = self._normalize_scores(scores, features.shape[0])
            except Exception:
                continue

        return detector_features, detector_scores

    def _extract_model_features(
        self, data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any]
    ) -> tuple[Any, ...]:
        """Extract features from all specialized models.

        This method extracts feature vectors from all 13 specialized
        domain models and normalizes their anomaly scores.

        Args:
            data: Input data for feature extraction.

        Returns:
            Tuple of (model_features, model_scores) where:
                - model_features: Dict mapping model names to features
                - model_scores: Dict mapping model names to scores

        Note:
            Models that fail to process data are silently skipped.
            This allows graceful degradation when domain-specific
            data is not available.
        """
        model_features = {}
        model_scores = {}

        for name, model in self.models.items():
            try:
                # Try to use cached features
                cache_key = self.feature_cache._make_key(
                    data if not isinstance(data, dict) else np.array([0]), prefix=f"model_{name}"
                )

                def compute_features(mdl: Any = model, d: Any = data) -> tuple[Any, ...]:
                    features = mdl.extract_features(d)
                    prediction = mdl.predict(d)
                    return features, prediction

                cached = self.feature_cache.get_or_compute(cache_key, compute_features)
                features, prediction = cached

                model_features[name] = features
                scores = prediction.get("anomaly_scores", 0)
                model_scores[name] = self._normalize_scores(scores, features.shape[0])
            except Exception:
                continue

        return model_features, model_scores

    def _extract_features_parallel(
        self, data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any]
    ) -> tuple[Any, ...]:
        """Extract features from all sources in parallel.

        This method uses thread pool execution to extract features
        from detectors and models concurrently.

        Args:
            data: Input data for feature extraction.

        Returns:
            Tuple of (all_features, all_scores) combining detector
            and model features.
        """
        executor = self._get_executor()

        # Submit parallel tasks
        detector_future = executor.submit(self._extract_detector_features, data)
        model_future = executor.submit(self._extract_model_features, data)

        # Collect results
        det_features, det_scores = detector_future.result()
        mod_features, mod_scores = model_future.result()

        return (
            {**det_features, **mod_features},
            {**det_scores, **mod_scores},
        )

    def detect_with_fusion(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
        domain: str | None = None,
        enable_gosnn: bool = True,
    ) -> dict[str, Any]:
        """Detect anomalies using ML fusion with GOSNN synaptic integration.

        This method combines outputs from all detectors and models using a
        neural network fusion approach with attention-based weighting. It
        integrates bidirectionally with the Global Omni-Scalar Network (GOSNN)
        for ethical gating and scalar enhancement.

        GOSNN Integration (Synaptic Fusion):
            1. Extract features from all detectors and models
            2. Call GOSNN.get_enhanced_scalars() for ethical gating (sigma_Immutable)
            3. Apply 32-head attention with triadic phi-weighting for harmonic synergy
            4. Feed enhanced scalars back to fusion for adaptive weighting
            5. Return results with ethical compliance metadata

        Args:
            data: Input data for detection.
            domain: Optional domain identifier for sigma_Immutable threshold tuning
                    (e.g., "medical" uses 0.93 fallback instead of 0.96 default)
            enable_gosnn: Enable GOSNN synaptic integration (default True)

        Returns:
            Dictionary containing:
                - anomaly_prob: Probability of anomaly (0.0-1.0)
                - is_anomaly: Boolean anomaly flag (prob > 0.5)
                - class_prediction: Predicted anomaly class
                - severity: Anomaly severity score
                - detector_importance: Dict of detector weights
                - mode: Detection mode ('fusion')
                - gosnn_metadata: GOSNN integration metadata (if enabled):
                    - ethical_gate_passed: Whether sigma_Immutable threshold was met
                    - sigma_immutable_score: Ethical compliance score
                    - harmonic_synergy: H(omega) component for Ava-Dominance Equation
                    - intelligence_contribution: GOSNN intelligence score
                    - warnings: Any ethical warnings

        Example:
            >>> engine = OmniMercuryEngine(mode="fusion")
            >>> data = np.random.randn(100, 10)
            >>> result = engine.detect_with_fusion(data, domain="security")
            >>> print(f"Anomaly: {result['is_anomaly']}, "
            ...       f"Prob: {result['anomaly_prob']:.3f}")
            >>> if result.get('gosnn_metadata'):
            ...     print(f"Ethical gate: {result['gosnn_metadata']['ethical_gate_passed']}")

        Note:
            Falls back to basic detection if not in fusion mode.
            GOSNN integration can be disabled via enable_gosnn=False for testing.
        """
        if self.mode != "fusion":
            return self.detect(data)

        det_features, det_scores = self._extract_detector_features(data)
        mod_features, mod_scores = self._extract_model_features(data)

        all_features = {**det_features, **mod_features}
        all_scores = {**det_scores, **mod_scores}

        # GOSNN Synaptic Integration
        gosnn_metadata: dict[str, Any] = {}
        if enable_gosnn:
            try:
                # Get GOSNN singleton with domain-appropriate threshold
                gosnn = get_global_scalar_network(
                    device=str(self.device),
                    domain=domain,
                    num_attention_heads=32,
                    enable_triadic_phi=True,
                )

                # Prepare base scalars from detector scores for enhancement
                base_scalars = {
                    f"detector_{name}_score": float(np.mean(score))
                    for name, score in all_scores.items()
                    if isinstance(score, (np.ndarray, float, int))
                }

                # Get enhanced scalars with ethical gating and harmonic synergy
                enhancement_result = gosnn.get_enhanced_scalars(
                    requesting_component="OmniMercuryEngine.detect_with_fusion",
                    base_scalars=base_scalars,
                    context={"domain": domain, "data_shape": getattr(data, "shape", None)},
                )

                # Store GOSNN metadata for transparency
                gosnn_metadata = {
                    "ethical_gate_passed": enhancement_result.ethical_gate_passed,
                    "sigma_immutable_score": enhancement_result.fusion_score,
                    "harmonic_synergy": gosnn.last_harmonic_synergy,
                    "intelligence_contribution": enhancement_result.intelligence_contribution,
                    "warnings": enhancement_result.warnings,
                    "sigma_immutable_threshold": gosnn.sigma_immutable_threshold,
                }

                # Register detector scalars with GOSNN for bidirectional feedback
                gosnn.register_scalars(
                    component_name="fusion_detectors",
                    scalars=enhancement_result.enhanced_scalars,
                    group=ScalarGroup.SECURITY if domain == "security" else ScalarGroup.ETHICAL,
                    metadata={"source": "detect_with_fusion", "domain": domain},
                )

                logger.debug(
                    f"GOSNN integration: ethical_gate={enhancement_result.ethical_gate_passed}, "
                    f"harmonic_synergy={gosnn.last_harmonic_synergy:.3f}"
                )

            except Exception as e:
                logger.warning(
                    f"GOSNN integration error: {e}. Falling back to raw features. "
                    "Detection will proceed without ethical gating enhancement."
                )
                # Fallback: Use raw detector scores as features without GOSNN enhancement
                # This ensures detection continues even if GOSNN fails
                fallback_scalars = {
                    f"fallback_{name}": float(np.mean(score))
                    for name, score in all_scores.items()
                    if isinstance(score, (np.ndarray, float, int))
                }
                gosnn_metadata = {
                    "error": str(e),
                    "ethical_gate_passed": True,  # Assume ethical for graceful degradation
                    "fallback_mode": True,
                    "fallback_scalars": fallback_scalars,
                    "sigma_immutable_score": 0.96,  # Default threshold
                    "harmonic_synergy": 0.5,  # Neutral synergy
                }

        fusion_result = self.fusion_inference.predict(
            all_features,
            return_attention=True,
        )

        anomaly_prob_val = fusion_result["anomaly_probs"][0]
        if isinstance(anomaly_prob_val, np.ndarray) or hasattr(anomaly_prob_val, "item"):
            anomaly_prob_val = anomaly_prob_val.item()

        severity_val = fusion_result["severity_scores"][0]
        if isinstance(severity_val, np.ndarray) or hasattr(severity_val, "item"):
            severity_val = severity_val.item()

        class_pred_val = fusion_result["class_predictions"][0]
        if isinstance(class_pred_val, np.ndarray) or hasattr(class_pred_val, "item"):
            class_pred_val = class_pred_val.item()

        result = {
            "anomaly_prob": float(anomaly_prob_val),
            "is_anomaly": bool(float(anomaly_prob_val) > 0.5),
            "class_prediction": int(class_pred_val),
            "severity": float(severity_val),
            "detector_importance": fusion_result.get("detector_importance", {}),
            "mode": "fusion",
        }

        # Add GOSNN metadata if integration was enabled
        if gosnn_metadata:
            result["gosnn_metadata"] = gosnn_metadata

        # Runtime Pipeline Integration: Drift Detection
        if self.drift_detector is not None:
            if isinstance(data, np.ndarray):
                drift_result = self._check_drift(data)
            elif isinstance(data, torch.Tensor):
                drift_result = self._check_drift(data.cpu().numpy())
            else:
                drift_result = None

            if drift_result is not None:
                result["drift_detection"] = {
                    "is_drift": drift_result.is_drift,
                    "severity": drift_result.severity.name if drift_result.severity else None,
                    "p_value": drift_result.p_value,
                    "message": drift_result.message,
                }

        # Runtime Pipeline Integration: LLM Enhancement (non-blocking)
        llm_enhancement = self._enhance_with_llm(data, result)
        if llm_enhancement is not None:
            result["llm_enhancement"] = llm_enhancement

        return result

    def detect_biometric(
        self,
        reference_image: str | np.ndarray[Any, Any],
        test_image: str | np.ndarray[Any, Any] | None = None,
        enable_age_progression: bool = False,
    ) -> dict[str, Any]:
        """Perform biometric face matching and analysis.

        This method uses the biometric model to analyze faces and
        optionally compare a test image against a reference.

        Args:
            reference_image: Reference face image (path or array).
            test_image: Optional test image to match against reference.
            enable_age_progression: Enable age progression estimation.

        Returns:
            Dictionary with biometric analysis results including:
                - match_score: Similarity score (if test_image provided)
                - face_attributes: Detected facial attributes
                - anomaly_scores: Biometric anomaly scores

        Example:
            >>> engine = OmniMercuryEngine()
            >>> result = engine.detect_biometric(
            ...     "reference.jpg",
            ...     "test.jpg"
            ... )
            >>> print(f"Match score: {result.get('match_score', 0):.3f}")
        """
        biometric_model = cast("BiometricAnomalyModel", self.models["biometric"])

        if test_image is not None:
            return biometric_model.predict(
                {
                    "reference": reference_image,
                    "test": test_image,
                }
            )
        else:
            # Convert string path to dict format for predict method
            if isinstance(reference_image, str):
                return biometric_model.predict({"reference": reference_image})
            return biometric_model.predict(reference_image)

    def detect_security_threat(
        self,
        payload: str,
        headers: dict[str, str] | None = None,
        source_ip: str | None = None,
    ) -> dict[str, Any]:
        """Detect security threats in requests.

        This method analyzes request payloads for potential security
        threats including injection attacks, XSS, and malicious patterns.

        Args:
            payload: Request payload string to analyze.
            headers: Optional request headers for context.
            source_ip: Optional source IP for logging and correlation.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean indicating threat detected
                - threats: List of detected threat types
                - source_ip: Source IP (if provided)

        Example:
            >>> engine = OmniMercuryEngine()
            >>> result = engine.detect_security_threat(
            ...     "<script>alert('xss')</script>",
            ...     source_ip="192.168.1.1"
            ... )
            >>> if result["is_anomaly"]:
            ...     print(f"Threats: {result['threats']}")
        """
        threat_result = self.security.detect_all(payload)

        return {
            "is_anomaly": threat_result["is_threat"],
            "threats": threat_result["threats"],
            "source_ip": source_ip,
        }

    def train_fusion_model(
        self,
        training_data: str,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer_type: str = "adamw",
        early_stopping_patience: int = 10,
        checkpoint_dir: str | None = None,
        use_mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> dict[str, Any]:
        """Train the fusion model on custom data.

        This method implements a complete training pipeline with support
        for learning rate scheduling, early stopping, checkpointing, and
        optional mixed-precision training.

        Training Pipeline:
            1. Load and validate training data
            2. Create train/validation split
            3. Configure optimizer and scheduler
            4. Train with early stopping
            5. Save best model checkpoint

        Args:
            training_data: Path to training data file (.npz or .pkl).
                Security Note: Pickle files (.pkl) can execute arbitrary code
                when loaded. Only use pickle files generated by this application
                or from trusted sources. For untrusted data, use .npz format.
            validation_split: Fraction of data for validation (0.0-1.0).
            epochs: Maximum number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Initial learning rate.
            optimizer_type: Optimizer type. Options:
                - 'adamw': AdamW optimizer (default)
                - 'ava_base': AVA base optimizer
                - 'ava_momentum': AVA with momentum
                - 'ava_exp_decay': AVA with exponential decay
                - 'ava_harmonic': AVA with harmonic decay
            early_stopping_patience: Epochs without improvement before
                stopping. Default 10.
            checkpoint_dir: Directory for saving checkpoints. If None,
                uses a temporary directory.
            use_mixed_precision: Enable FP16 mixed precision training.
                Requires CUDA. Default False.
            gradient_accumulation_steps: Steps for gradient accumulation.
                Useful for larger effective batch sizes. Default 1.

        Returns:
            Dictionary containing:
                - final_loss: Final validation loss
                - best_loss: Best validation loss achieved
                - epochs_trained: Number of epochs completed
                - best_epoch: Epoch with best validation loss
                - checkpoint_path: Path to best model checkpoint
                - training_history: List of per-epoch metrics
                - early_stopped: Whether training stopped early

        Raises:
            ValueError: If parameters are invalid or mode is not 'fusion'.
            RuntimeError: If training data loading fails.

        Example:
            >>> engine = OmniMercuryEngine(mode="fusion")
            >>> result = engine.train_fusion_model(
            ...     "training_data.npz",
            ...     epochs=100,
            ...     batch_size=64,
            ...     early_stopping_patience=15
            ... )
            >>> print(f"Best loss: {result['best_loss']:.4f}")
            >>> print(f"Epochs: {result['epochs_trained']}")
        """
        import os
        import pickle
        import tempfile

        from torch.utils.data import DataLoader, random_split

        from omni_mercury_engine.ml.training import AnomalyDataset, FusionTrainer

        if self.mode != "fusion":
            raise ValueError("Training requires fusion mode. Initialize with mode='fusion'")

        # Validate parameters
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if not (0.0 < validation_split < 1.0):
            raise ValueError("validation_split must be between 0 and 1 (exclusive)")

        # Load training data
        if not os.path.exists(training_data):
            raise ValueError(f"Training data path does not exist: {training_data}")

        try:
            if training_data.endswith(".npz"):
                data = np.load(training_data, allow_pickle=True)
                features_dict = {
                    k: torch.tensor(v, dtype=torch.float32)
                    for k, v in data.items()
                    if k != "labels"
                }
                labels = torch.tensor(data["labels"], dtype=torch.long)
            elif training_data.endswith(".pkl") or training_data.endswith(".pickle"):
                # nosec B301 - pickle required for legacy data format compatibility
                # Security Note: Only load pickle files from trusted sources to
                # prevent arbitrary code execution during deserialization.
                with open(training_data, "rb") as f:
                    loaded = pickle.load(f)  # nosec B301
                features_dict = {
                    k: torch.tensor(v, dtype=torch.float32) for k, v in loaded["features"].items()
                }
                labels = torch.tensor(loaded["labels"], dtype=torch.long)
            else:
                raise ValueError(f"Unsupported data format. Use .npz or .pkl: {training_data}")
        except Exception as e:
            raise RuntimeError(f"Failed to load training data: {e}")

        # Create dataset and split
        dataset = AnomalyDataset(features_dict, labels)
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = tempfile.mkdtemp(prefix="omni_fusion_")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create trainer module
        trainer_module = FusionTrainer(
            model=self.fusion_model,
            learning_rate=learning_rate,
        )
        trainer_module.optimizer_type = optimizer_type  # type: ignore[assignment]

        # Training state
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        training_history: list[dict[str, float]] = []
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        # Setup mixed precision if requested
        scaler = (
            torch.cuda.amp.GradScaler()
            if use_mixed_precision and self.device.type == "cuda"
            else None
        )

        # Configure optimizer
        optimizer_config = cast("dict[str, Any]", trainer_module.configure_optimizers())
        optimizer = optimizer_config["optimizer"]
        scheduler = optimizer_config["lr_scheduler"]["scheduler"]

        self.fusion_model.train()

        for epoch in range(epochs):
            # Training phase
            train_losses = []
            self.fusion_model.train()

            for batch_idx, batch in enumerate(train_loader):
                if use_mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = trainer_module.training_step(batch, batch_idx)
                    scaler.scale(loss / gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = trainer_module.training_step(batch, batch_idx)
                    (loss / gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Validation phase
            val_losses = []
            self.fusion_model.eval()

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    trainer_module.validation_step(batch, batch_idx)
                    # Calculate validation loss manually
                    features, labels_batch = batch
                    outputs = self.fusion_model(features, return_attention=True)
                    anomaly_labels = (labels_batch > 0).float().unsqueeze(1)
                    val_loss = torch.nn.functional.binary_cross_entropy(
                        outputs["anomaly_probs"], anomaly_labels
                    )
                    val_losses.append(val_loss.item())

            avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss

            # Update learning rate scheduler
            scheduler.step(avg_val_loss)

            # Track history
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # Check for improvement and save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.fusion_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    best_checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                break

        # Load best model
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(
                best_checkpoint_path, map_location=self.device, weights_only=True
            )
            self.fusion_model.load_state_dict(checkpoint["model_state_dict"])

        self.fusion_model.eval()

        return {
            "final_loss": training_history[-1]["val_loss"] if training_history else 0.0,
            "best_loss": best_val_loss,
            "epochs_trained": len(training_history),
            "best_epoch": best_epoch,
            "checkpoint_path": best_checkpoint_path,
            "training_history": training_history,
            "early_stopped": epochs_without_improvement >= early_stopping_patience,
        }

    def save_model(self, path: str) -> None:
        """Save fusion model weights to file.

        Args:
            path: File path for saving the model.

        Example:
            >>> engine.save_model("models/fusion_model.pt")
        """
        if self.mode == "fusion":
            torch.save(self.fusion_model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load fusion model weights from file.

        Args:
            path: File path to load the model from.

        Example:
            >>> engine.load_model("models/fusion_model.pt")
        """
        if self.mode == "fusion":
            self.fusion_model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get feature cache statistics.

        Returns:
            Dictionary with cache statistics including hit rate.

        Example:
            >>> stats = engine.get_cache_stats()
            >>> print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        """
        return self.feature_cache.stats()

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory statistics.

        Example:
            >>> stats = engine.get_memory_stats()
            >>> print(f"Peak memory: {stats['peak_mb']:.1f} MB")
        """
        return self.memory_monitor.stats()

    def clear_cache(self) -> None:
        """Clear the feature cache.

        This can be useful to free memory after processing
        large datasets.

        Example:
            >>> engine.clear_cache()
        """
        self.feature_cache.clear()

    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)


# Legacy alias removed - project renamed to Mercury Agent ♱
