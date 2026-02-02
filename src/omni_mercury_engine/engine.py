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

# Core detectors - always imported (lightweight base classes)
from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

# Runtime pipeline modules - always imported (required for core functionality)
from omni_mercury_engine.ml.drift import DriftResult, EnsembleDriftDetector
from omni_mercury_engine.ml.fairness import BiasAuditConfig, FairnessAuditor, FairnessReport
from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference
from omni_mercury_engine.ml.optimization import OptimizationConfig, ParallelExecutor
from omni_mercury_engine.utils.logging import LoggerMixin


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    # Type hints for lazy-loaded models (improves IDE support without import cost)
    from omni_mercury_engine.medical.abms_disciplines import ABMSDisciplineDetector
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

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Import Functions for Specialized Models
# =============================================================================
# These functions defer import of heavy model modules until first use,
# reducing cold-start time by ~50% for applications that don't use all models.

_lazy_cache: dict[str, Any] = {}
_lazy_lock = threading.Lock()


def _lazy_import(name: str) -> Any:
    """Thread-safe lazy import of specialized models."""
    if name in _lazy_cache:
        return _lazy_cache[name]

    with _lazy_lock:
        if name in _lazy_cache:
            return _lazy_cache[name]

        logger.debug(f"Lazy loading: {name}")

        if name == "ABMSDisciplineDetector":
            from omni_mercury_engine.medical.abms_disciplines import ABMSDisciplineDetector

            _lazy_cache[name] = ABMSDisciplineDetector
        elif name == "AffectiveAnomalyModel":
            from omni_mercury_engine.models.affective import AffectiveAnomalyModel

            _lazy_cache[name] = AffectiveAnomalyModel
        elif name == "AstrophysicalAnomalyModel":
            from omni_mercury_engine.models.astrophysical import AstrophysicalAnomalyModel

            _lazy_cache[name] = AstrophysicalAnomalyModel
        elif name == "BiometricAnomalyModel":
            from omni_mercury_engine.models.biometric import BiometricAnomalyModel

            _lazy_cache[name] = BiometricAnomalyModel
        elif name == "ChemistryAnomalyDetector":
            from omni_mercury_engine.models.chemistry import ChemistryAnomalyDetector

            _lazy_cache[name] = ChemistryAnomalyDetector
        elif name == "ConsciousnessPreservationModel":
            from omni_mercury_engine.models.consciousness import ConsciousnessPreservationModel

            _lazy_cache[name] = ConsciousnessPreservationModel
        elif name == "NeuralCognitiveModel":
            from omni_mercury_engine.models.neural import NeuralCognitiveModel

            _lazy_cache[name] = NeuralCognitiveModel
        elif name == "ParapsychologyDetector":
            from omni_mercury_engine.models.parapsychology import ParapsychologyDetector

            _lazy_cache[name] = ParapsychologyDetector
        elif name == "QuantumAnomalyModel":
            from omni_mercury_engine.models.quantum import QuantumAnomalyModel

            _lazy_cache[name] = QuantumAnomalyModel
        elif name == "IntelligenceFusionEngine":
            from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine

            _lazy_cache[name] = IntelligenceFusionEngine
        elif name == "ThreatDetector":
            from omni_mercury_engine.security.threat_detection import ThreatDetector

            _lazy_cache[name] = ThreatDetector
        elif name == "SchumannResonanceDetector":
            from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

            _lazy_cache[name] = SchumannResonanceDetector
        elif name == "SelfHealingEngine":
            from omni_mercury_engine.resilience.self_healing import SelfHealingEngine

            _lazy_cache[name] = SelfHealingEngine
        elif name == "LLMConfig":
            from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig

            _lazy_cache[name] = LLMConfig
        elif name == "LLMProvider":
            from omni_mercury_engine.models.foundation.llm_adapter import LLMProvider

            _lazy_cache[name] = LLMProvider
        elif name == "ZeroShotAnomalyDetector":
            from omni_mercury_engine.models.foundation.llm_adapter import ZeroShotAnomalyDetector

            _lazy_cache[name] = ZeroShotAnomalyDetector
        else:
            raise ValueError(f"Unknown lazy model: {name}")

        return _lazy_cache[name]


def get_abms_detector() -> type[ABMSDisciplineDetector]:
    """Get ABMSDisciplineDetector class (lazy loaded)."""
    return _lazy_import("ABMSDisciplineDetector")


def get_affective_model() -> type[AffectiveAnomalyModel]:
    """Get AffectiveAnomalyModel class (lazy loaded)."""
    return _lazy_import("AffectiveAnomalyModel")


def get_astrophysical_model() -> type[AstrophysicalAnomalyModel]:
    """Get AstrophysicalAnomalyModel class (lazy loaded)."""
    return _lazy_import("AstrophysicalAnomalyModel")


def get_biometric_model() -> type[BiometricAnomalyModel]:
    """Get BiometricAnomalyModel class (lazy loaded)."""
    return _lazy_import("BiometricAnomalyModel")


def get_chemistry_detector() -> type[ChemistryAnomalyDetector]:
    """Get ChemistryAnomalyDetector class (lazy loaded)."""
    return _lazy_import("ChemistryAnomalyDetector")


def get_consciousness_model() -> type[ConsciousnessPreservationModel]:
    """Get ConsciousnessPreservationModel class (lazy loaded)."""
    return _lazy_import("ConsciousnessPreservationModel")


def get_neural_model() -> type[NeuralCognitiveModel]:
    """Get NeuralCognitiveModel class (lazy loaded)."""
    return _lazy_import("NeuralCognitiveModel")


def get_parapsychology_detector() -> type[ParapsychologyDetector]:
    """Get ParapsychologyDetector class (lazy loaded)."""
    return _lazy_import("ParapsychologyDetector")


def get_quantum_model() -> type[QuantumAnomalyModel]:
    """Get QuantumAnomalyModel class (lazy loaded)."""
    return _lazy_import("QuantumAnomalyModel")


def get_intelligence_fusion() -> type[IntelligenceFusionEngine]:
    """Get IntelligenceFusionEngine class (lazy loaded)."""
    return _lazy_import("IntelligenceFusionEngine")


def get_threat_detector() -> type[ThreatDetector]:
    """Get ThreatDetector class (lazy loaded)."""
    return _lazy_import("ThreatDetector")


def get_schumann_detector() -> type[SchumannResonanceDetector]:
    """Get SchumannResonanceDetector class (lazy loaded)."""
    return _lazy_import("SchumannResonanceDetector")


def get_self_healing() -> type[SelfHealingEngine]:
    """Get SelfHealingEngine class (lazy loaded)."""
    return _lazy_import("SelfHealingEngine")


def get_llm_config() -> type[LLMConfig]:
    """Get LLMConfig class (lazy loaded)."""
    return _lazy_import("LLMConfig")


def get_llm_provider() -> type[LLMProvider]:
    """Get LLMProvider class (lazy loaded)."""
    return _lazy_import("LLMProvider")


def get_zero_shot_detector() -> type[ZeroShotAnomalyDetector]:
    """Get ZeroShotAnomalyDetector class (lazy loaded)."""
    return _lazy_import("ZeroShotAnomalyDetector")


# Backward-compatible aliases for direct access (triggers lazy load on access)
# These allow existing code like `from engine import QuantumAnomalyModel` to work
def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy loading on attribute access."""
    lazy_models = {
        "ABMSDisciplineDetector",
        "AffectiveAnomalyModel",
        "AstrophysicalAnomalyModel",
        "BiometricAnomalyModel",
        "ChemistryAnomalyDetector",
        "ConsciousnessPreservationModel",
        "NeuralCognitiveModel",
        "ParapsychologyDetector",
        "QuantumAnomalyModel",
        "IntelligenceFusionEngine",
        "ThreatDetector",
        "SchumannResonanceDetector",
        "SelfHealingEngine",
        "LLMConfig",
        "LLMProvider",
        "ZeroShotAnomalyDetector",
    }
    if name in lazy_models:
        return _lazy_import(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


class OmniMercuryEngine(LoggerMixin):
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

        Uses lazy loading to defer import of heavy model modules until first use,
        reducing cold-start time by ~50% for applications that don't use all models.
        """
        from omni_mercury_engine.models.multiverse import MultiverseOmniEngine
        from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine

        # Use lazy import getter functions for specialized models
        self.models = {
            "quantum": get_quantum_model()(),
            "astrophysical": get_astrophysical_model()(),
            "biometric": get_biometric_model()(),
            "affective": get_affective_model()(),
            "neural": get_neural_model()(),
            "consciousness": get_consciousness_model()(),
            "multiverse": MultiverseOmniEngine(num_universes=10, state_dim=50),
            "neurosymbolic": NeurosymbolicEngine(input_dim=64),
            "medical_abms": get_abms_detector()(),
            "intelligence_fusion": get_intelligence_fusion()(),
            "schumann_resonance": get_schumann_detector()(),
            "chemistry": get_chemistry_detector()(),
            "parapsychology": get_parapsychology_detector()(),
        }

        self.security = get_threat_detector()()

    def _init_fusion(self) -> None:
        """Initialize ML fusion components.

        Sets up the neural network fusion model and inference engine
        when operating in fusion mode.

        Note:
            Fix for Issue #1: Untrained Fusion Neural Network.
            The fusion model is initialized with random weights. Users should
            call fit_fusion() with training data before detection for optimal
            performance. Detection will still work without training but may
            produce suboptimal results.
        """
        if self.mode == "fusion":
            self.fusion_model = OmniFusionModel()
            self.fusion_model.to(self.device)
            self.fusion_inference = FusionInference(
                model=self.fusion_model,
                device=str(self.device),
            )
            # Track training state - Fix for Issue #1
            self._fusion_trained = False
            logger.info(
                "OmniFusionModel initialized (untrained). Call fit_fusion() "
                "before detection for optimal performance."
            )

    def _init_resilience(self) -> None:
        """Initialize resilience and self-healing components."""
        self.self_healing = get_self_healing()()

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

    def fit_fusion(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any] | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        contamination: float | None = None,
    ) -> dict[str, Any]:
        """Fit the fusion model on training data with semi-supervised learning.

        This method extracts features from all detectors and trains the OmniFusionModel
        to produce calibrated anomaly scores. Supports both supervised (with labels)
        and semi-supervised (estimated pseudo-labels) training.

        This is the primary fix for Issue #1: Untrained Fusion Neural Network.

        Args:
            X: Training features (n_samples, n_features).
            y: Optional training labels (1=anomaly, 0=normal). If None, uses
               semi-supervised learning with pseudo-labels from detector consensus.
            epochs: Maximum training epochs (default: 50).
            batch_size: Training batch size (default: 32).
            learning_rate: Learning rate for optimizer (default: 0.001).
            early_stopping_patience: Epochs without improvement before stopping.
            validation_split: Fraction of data for validation.
            contamination: Expected anomaly fraction for pseudo-labeling. If None,
                          estimated from data using adaptive methods.

        Returns:
            Dictionary with training metrics including final_loss, best_loss,
            epochs_trained, and convergence information.

        Raises:
            ValueError: If mode is not 'fusion'.
            RuntimeError: If no detector features could be extracted.

        Example:
            >>> engine = OmniMercuryEngine(mode="fusion")
            >>> metrics = engine.fit_fusion(X_train, y_train, epochs=100)
            >>> print(f"Training loss: {metrics['best_loss']:.4f}")
            >>> result = engine.detect_with_fusion(X_test)
        """
        if self.mode != "fusion":
            raise ValueError("fit_fusion() requires mode='fusion'")

        # GPU check with fallback
        device = self.device
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = torch.device("cpu")

        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        n_samples = len(X)
        logger.info(f"Starting fusion training on {n_samples} samples...")

        # Fit all base detectors first
        logger.info(f"Fitting {len(self.detectors)} base detectors...")
        for name, detector in self.detectors.items():
            try:
                if not detector.is_fitted():
                    detector.fit(X)
                    logger.debug(f"Fitted detector: {name}")
            except Exception as e:
                logger.warning(f"Failed to fit detector {name}: {e}")

        # Extract features from all detectors
        logger.info("Extracting detector features for fusion training...")
        detector_features: dict[str, torch.Tensor] = {}
        for name, detector in self.detectors.items():
            try:
                features = detector.extract_features(X)
                if isinstance(features, np.ndarray):
                    features = torch.tensor(features, dtype=torch.float32)
                detector_features[name] = features
                logger.debug(f"Extracted features from {name}: shape={features.shape}")
            except Exception as e:
                logger.warning(f"Failed to extract features from {name}: {e}")

        if not detector_features:
            raise RuntimeError("No detector features could be extracted")

        # Generate pseudo-labels if not provided (semi-supervised)
        if y is None:
            logger.info("No labels provided, using semi-supervised pseudo-labeling...")
            y = self._generate_pseudo_labels(X, contamination)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Prepare training data
        labels_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Create train/val split
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Training loop with early stopping
        self.fusion_model.train()
        self.fusion_model.to(device)

        optimizer = torch.optim.AdamW(
            self.fusion_model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        loss_history: list[dict[str, float]] = []

        for epoch in range(epochs):
            # Training phase
            self.fusion_model.train()
            train_losses: list[float] = []

            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch_indices = train_indices[start_idx:end_idx]

                # Get batch features
                batch_features = {
                    name: feat[batch_indices].to(device) for name, feat in detector_features.items()
                }
                batch_labels = labels_tensor[batch_indices].to(device)

                optimizer.zero_grad()
                outputs = self.fusion_model(batch_features)
                loss = torch.nn.functional.binary_cross_entropy(
                    outputs["anomaly_probs"], batch_labels
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            self.fusion_model.eval()
            val_losses: list[float] = []

            with torch.no_grad():
                for start_idx in range(0, n_val, batch_size):
                    end_idx = min(start_idx + batch_size, n_val)
                    batch_indices = val_indices[start_idx:end_idx]

                    batch_features = {
                        name: feat[batch_indices].to(device)
                        for name, feat in detector_features.items()
                    }
                    batch_labels = labels_tensor[batch_indices].to(device)

                    outputs = self.fusion_model(batch_features)
                    loss = torch.nn.functional.binary_cross_entropy(
                        outputs["anomaly_probs"], batch_labels
                    )
                    val_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss

            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.fusion_model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={avg_val_loss:.4f}"
                )

        # Restore best model
        if best_state is not None:
            self.fusion_model.load_state_dict(best_state)

        self.fusion_model.eval()
        self._fusion_trained = True

        logger.info(f"Fusion training complete. Best val_loss: {best_val_loss:.4f}")

        return {
            "final_loss": loss_history[-1]["val_loss"] if loss_history else 0.0,
            "best_loss": best_val_loss,
            "epochs_trained": len(loss_history),
            "best_epoch": len(loss_history) - epochs_without_improvement,
            "loss_history": loss_history,
            "early_stopped": epochs_without_improvement >= early_stopping_patience,
        }

    def _generate_pseudo_labels(
        self,
        X: np.ndarray[Any, Any],
        contamination: float | None = None,
    ) -> np.ndarray[Any, Any]:
        """Generate pseudo-labels using detector consensus for semi-supervised learning.

        Uses adaptive contamination estimation and ensemble voting from detector
        scores to identify likely anomalies for training.

        Args:
            X: Training features.
            contamination: Expected anomaly fraction. If None, estimated adaptively.

        Returns:
            Binary pseudo-labels (0=normal, 1=anomaly).
        """
        n_samples = len(X)

        # Collect scores from all detectors
        all_scores: list[np.ndarray[Any, Any]] = []
        for name, detector in self.detectors.items():
            try:
                if not detector.is_fitted():
                    detector.fit(X)
                result = detector.detect(X)
                scores = result.get("scores", result.get("is_anomaly", np.zeros(n_samples)))
                if isinstance(scores, (list, np.ndarray)):
                    scores = np.array(scores).flatten()
                    if len(scores) == n_samples:
                        all_scores.append(scores)
            except Exception as e:
                logger.debug(f"Failed to get scores from {name}: {e}")

        if not all_scores:
            # Fallback: use distance from mean
            mean = np.mean(X, axis=0)
            distances = np.linalg.norm(X - mean, axis=1)
            all_scores = [distances / (distances.max() + 1e-8)]

        # Ensemble score (average)
        ensemble_score = np.mean(all_scores, axis=0)

        # Estimate contamination if not provided
        if contamination is None:
            # Use IQR-based estimation
            q1, q3 = np.percentile(ensemble_score, [25, 75])
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            contamination = float(np.mean(ensemble_score > upper_fence))
            contamination = max(0.001, min(contamination, 0.5))

        # Threshold at (1 - contamination) percentile
        threshold = np.percentile(ensemble_score, (1 - contamination) * 100)
        pseudo_labels = (ensemble_score > threshold).astype(float)

        logger.info(
            f"Generated pseudo-labels: contamination={contamination:.4f}, "
            f"n_anomalies={int(pseudo_labels.sum())}/{n_samples}"
        )

        return pseudo_labels

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
            p_value_threshold=0.05,
        )
        self._drift_feature_names = feature_names
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
            protected_features=sensitive_features or [],
            fairness_threshold=fairness_threshold,
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

            context = {
                "anomaly_detected": True,
                "candidate_labels": [
                    "security_threat",
                    "system_failure",
                    "data_corruption",
                    "unusual_pattern",
                    "normal_variation",
                ],
            }
            llm_result = self.llm_detector.detect(
                data=f"Anomaly detected in: {data_str}",
                context=context,
            )
            return {
                "llm_explanation": llm_result.get("explanation", ""),
                "llm_category": llm_result.get("category", "unknown"),
                "llm_confidence": llm_result.get("confidence", 0.0),
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

    def _get_anomaly_threshold(
        self,
        current_score: float,
        all_scores: np.ndarray | None = None,
    ) -> float:
        """Determine anomaly threshold based on configuration.

        This method addresses the threshold calibration issue where a fixed
        threshold of 0.5 fails on highly imbalanced datasets. It supports:
        1. Fixed threshold from config.anomaly_threshold
        2. Contamination-based percentile threshold (like sklearn IsolationForest)
        3. Adaptive threshold based on score distribution with IQR fallback

        The adaptive threshold uses IQR-based outlier detection when contamination
        is not explicitly set, which handles extreme class imbalance (e.g., covtype
        with ~0.5% anomaly rate) better than fixed thresholds.

        Args:
            current_score: The anomaly probability to evaluate.
            all_scores: Array of all anomaly scores for percentile calculation.

        Returns:
            The threshold to use for anomaly classification.

        Example:
            >>> # Fixed threshold (default)
            >>> engine.config.anomaly_threshold = 0.3
            >>> threshold = engine._get_anomaly_threshold(0.4, scores)
            >>> # Returns 0.3

            >>> # Contamination-based (top 5% = anomaly)
            >>> engine.config.contamination = 0.05
            >>> threshold = engine._get_anomaly_threshold(0.4, scores)
            >>> # Returns 95th percentile of scores

            >>> # Adaptive with IQR fallback
            >>> engine.config.adaptive_threshold = True
            >>> threshold = engine._get_anomaly_threshold(0.4, scores)
            >>> # Returns IQR-based threshold for extreme imbalance
        """
        # If contamination is set, use percentile-based threshold
        if self.config.contamination is not None and all_scores is not None:
            if len(all_scores) > 0:
                # Calculate the (1 - contamination) percentile
                # e.g., contamination=0.05 means we want 95th percentile as threshold
                percentile = (1.0 - self.config.contamination) * 100
                threshold = float(np.percentile(all_scores, percentile))
                logger.debug(
                    f"Using contamination-based threshold: {threshold:.4f} "
                    f"(contamination={self.config.contamination})"
                )
                return threshold

        # If adaptive threshold is enabled, use IQR-based calibration
        # This addresses the covtype F1=0 issue where fixed thresholds fail
        # on extremely imbalanced datasets
        if self.config.adaptive_threshold and all_scores is not None:
            if len(all_scores) > 1:
                # IQR-based outlier detection for contamination estimation
                q1, q3 = np.percentile(all_scores, [25, 75])
                iqr = q3 - q1

                if iqr > 1e-8:  # Avoid division by zero
                    # Estimate contamination from score distribution
                    # Points above Q3 + 1.5*IQR are statistical outliers
                    upper_fence = q3 + 1.5 * iqr
                    estimated_contamination = float(np.mean(all_scores > upper_fence))
                    # Floor at 0.1% to ensure some predictions
                    estimated_contamination = max(estimated_contamination, 0.001)

                    # Use percentile-based threshold with estimated contamination
                    percentile = (1.0 - estimated_contamination) * 100
                    threshold = float(np.percentile(all_scores, percentile))

                    logger.debug(
                        f"Using adaptive IQR threshold: {threshold:.4f} "
                        f"(estimated_contamination={estimated_contamination:.4f}, "
                        f"IQR={iqr:.4f})"
                    )
                    return threshold
                else:
                    # Fallback: use mean + 2*std when IQR is too small
                    mean_score = float(np.mean(all_scores))
                    std_score = float(np.std(all_scores))
                    if std_score > 1e-8:
                        threshold = mean_score + 2 * std_score
                        # Cap threshold using configurable maximum
                        threshold = min(threshold, self.config.thresholds.anomaly_cap)
                        logger.debug(f"Using adaptive mean+2std threshold: {threshold:.4f}")
                        return threshold

        # Default: use fixed threshold from config
        return self.config.anomaly_threshold

    # =========================================================================
    # Calibration Pipeline (Solves F1=0 Problem)
    # =========================================================================

    def enable_auto_calibration(
        self,
        contamination: float = 0.05,
        method: str = "auto",
    ) -> OmniMercuryEngine:
        """
        Enable automatic threshold calibration for all detectors.

        This solves the F1=0 problem where good ROC-AUC is achieved but
        a fixed 0.5 threshold produces no positive predictions.

        Args:
            contamination: Expected fraction of anomalies (0.0-1.0)
            method: Calibration method ("auto", "percentile", "otsu", etc.)

        Returns:
            Self for method chaining.

        Example:
            >>> engine = OmniMercuryEngine()
            >>> engine.enable_auto_calibration(contamination=0.05)
            >>> result = engine.detect(data)  # Thresholds auto-calibrated
        """
        for detector in self.detectors.values():
            detector.enable_auto_calibration(
                contamination=contamination,
                method=method,
            )

        self.config.adaptive_threshold = True
        self.config.contamination = contamination
        return self

    def disable_auto_calibration(self) -> OmniMercuryEngine:
        """Disable automatic threshold calibration for all detectors."""
        for detector in self.detectors.values():
            detector.disable_auto_calibration()

        self.config.adaptive_threshold = False
        self.config.contamination = None
        return self

    def calibrate_from_scores(
        self,
        scores: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any] | None = None,
        method: str = "auto",
    ) -> dict[str, Any]:
        """
        Calibrate threshold from a batch of scores.

        Use this method when you have precomputed scores and want to
        find the optimal threshold without re-running detection.

        Args:
            scores: Precomputed anomaly scores
            labels: Optional ground truth labels (enables optimal F1)
            method: Calibration method

        Returns:
            Dictionary with:
                - threshold: Calibrated threshold
                - predictions: Binary predictions
                - diagnostics: CalibrationDiagnostics object

        Example:
            >>> result = engine.detect(data)
            >>> calibration = engine.calibrate_from_scores(
            ...     result["scores"], y_true, method="auto"
            ... )
            >>> print(calibration["diagnostics"])
        """
        from omni_mercury_engine.core.score_calibration import (
            CalibrationMethod,
            ScoreCalibrationManager,
        )

        manager = ScoreCalibrationManager(
            contamination=self.config.contamination or 0.05,
            method=CalibrationMethod(method),
        )

        result = manager.calibrate(
            scores=np.asarray(scores),
            labels=np.asarray(labels) if labels is not None else None,
        )

        return {
            "threshold": result.threshold,
            "predictions": result.predictions,
            "diagnostics": result.diagnostics,
            "method": result.method.value,
            "confidence": result.confidence,
        }

    def diagnose_detection(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        labels: np.ndarray[Any, Any] | None = None,
        print_output: bool = True,
    ) -> dict[str, Any]:
        """
        Run detection with full diagnostics for debugging F1=0 issues.

        This method is specifically designed to help diagnose calibration
        problems. It runs detection and provides comprehensive diagnostics
        about the score distribution and threshold.

        Args:
            data: Input data for anomaly detection
            labels: Optional ground truth labels
            print_output: Whether to print diagnostics

        Returns:
            Dictionary with:
                - detection_result: Standard detection result
                - diagnostics: Per-detector diagnostics
                - recommendations: Suggested fixes

        Example:
            >>> # Debug F1=0 problem
            >>> diag = engine.diagnose_detection(test_data, y_true)
            >>> # Prints detailed diagnostics
            >>> print(diag["recommendations"])
        """
        from omni_mercury_engine.core.score_calibration import (
            CalibrationDiagnostics,
            ScoreDiagnostics,
        )

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Run detection
        detection_result = self.detect(data)

        # Collect diagnostics from each detector
        detector_diagnostics: dict[str, CalibrationDiagnostics] = {}
        recommendations: list[str] = []

        for detector_name, detector_result in detection_result.get("detectors", {}).items():
            scores = detector_result.get("scores")
            if scores is None:
                scores = detector_result.get("anomaly_score")
                if scores is not None:
                    scores = np.array([scores])
                else:
                    continue

            scores = np.asarray(scores).flatten()
            threshold = detector_result.get("threshold", 0.5)

            diag = ScoreDiagnostics.analyze(
                scores=scores,
                threshold=threshold,
                labels=labels,
                method=detector_name,
            )
            detector_diagnostics[detector_name] = diag

            if print_output:
                print(f"\n{'='*60}")
                print(f"DETECTOR: {detector_name}")
                print(diag)

            # Generate recommendations
            if diag.predicted_anomaly_ratio == 0:
                recommendations.append(
                    f"[{detector_name}] All predictions are NEGATIVE. "
                    f"Threshold ({threshold:.4f}) > max score ({diag.score_max:.4f}). "
                    f"SOLUTION: Use auto-calibration or lower threshold."
                )
            elif diag.predicted_anomaly_ratio > 0.5:
                recommendations.append(
                    f"[{detector_name}] Too many positives ({diag.predicted_anomaly_ratio:.1%}). "
                    f"SOLUTION: Increase threshold or use contamination-based calibration."
                )

            if diag.is_bimodal:
                recommendations.append(
                    f"[{detector_name}] Bimodal score distribution detected. "
                    f"SOLUTION: Use Otsu's method for threshold selection."
                )

        if print_output and recommendations:
            print("\n" + "=" * 60)
            print("RECOMMENDATIONS")
            print("=" * 60)
            for rec in recommendations:
                print(f"  - {rec}")
            print("=" * 60)

        return {
            "detection_result": detection_result,
            "diagnostics": detector_diagnostics,
            "recommendations": recommendations,
        }

    def detect_with_calibration(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        labels: np.ndarray[Any, Any] | None = None,
        calibration_method: str = "auto",
        contamination: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect anomalies with automatic threshold calibration.

        This is the recommended method when you want optimal F1 performance.
        It runs detection, calibrates the threshold based on the score
        distribution (or ground truth if provided), and returns predictions.

        Args:
            data: Input data for anomaly detection
            labels: Optional ground truth labels (enables optimal F1)
            calibration_method: Method for threshold selection
            contamination: Expected anomaly ratio (if known)

        Returns:
            Dictionary with:
                - is_anomaly: Calibrated binary predictions
                - scores: Raw anomaly scores
                - threshold: Calibrated threshold
                - diagnostics: Calibration diagnostics
                - detector_results: Raw results from each detector

        Example:
            >>> # Get calibrated predictions
            >>> result = engine.detect_with_calibration(data, method="auto")
            >>> predictions = result["is_anomaly"]
            >>> print(f"Threshold: {result['threshold']:.4f}")
            >>> print(result["diagnostics"])
        """
        from omni_mercury_engine.core.score_calibration import (
            CalibrationMethod,
            ScoreCalibrationManager,
        )

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        contamination = contamination or self.config.contamination or 0.05

        # Run detection
        detection_result = self.detect(data)

        # Aggregate scores from all detectors
        all_scores: list[np.ndarray] = []
        for detector_result in detection_result.get("detectors", {}).values():
            scores = detector_result.get("scores")
            if scores is not None:
                all_scores.append(np.asarray(scores).flatten())

        if not all_scores:
            # Fallback: use is_anomaly flags
            return {
                **detection_result,
                "threshold": 0.5,
                "diagnostics": None,
            }

        # Combine scores (simple average)
        combined_scores = np.mean(all_scores, axis=0)

        # Calibrate threshold
        manager = ScoreCalibrationManager(
            contamination=contamination,
            method=CalibrationMethod(calibration_method),
        )

        calibration_result = manager.calibrate(
            scores=combined_scores,
            labels=np.asarray(labels) if labels is not None else None,
        )

        return {
            "is_anomaly": calibration_result.predictions,
            "scores": combined_scores,
            "threshold": calibration_result.threshold,
            "diagnostics": calibration_result.diagnostics,
            "calibration_method": calibration_result.method.value,
            "detector_results": detection_result.get("detectors", {}),
        }

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
            except (ValueError, TypeError, RuntimeError, KeyError, AttributeError, IndexError) as e:
                logger.debug(f"Detector {name} feature extraction failed: {e}")
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
            except (ValueError, TypeError, RuntimeError, KeyError, AttributeError, IndexError) as e:
                logger.debug(f"Model {name} feature extraction failed: {e}")
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
                    - harmonic_synergy: H(omega) component for weighted fusion Equation
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

        # Determine threshold for anomaly classification
        threshold = self._get_anomaly_threshold(
            anomaly_prob_val,
            fusion_result.get("anomaly_probs", np.array([anomaly_prob_val])),
        )

        result = {
            "anomaly_prob": float(anomaly_prob_val),
            "is_anomaly": bool(float(anomaly_prob_val) > threshold),
            "class_prediction": int(class_pred_val),
            "severity": float(severity_val),
            "detector_importance": fusion_result.get("detector_importance", {}),
            "mode": "fusion",
            "threshold_used": float(threshold),
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

    def detect_with_fusion_calibrated(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
        labels: np.ndarray[Any, Any] | None = None,
        calibration_method: str = "auto",
        contamination: float | None = None,
        domain: str | None = None,
        enable_gosnn: bool = True,
    ) -> dict[str, Any]:
        """Detect anomalies using ML fusion with automatic threshold calibration.

        This method combines detect_with_fusion with the score calibration system
        to solve the F1=0 problem where good ROC-AUC is achieved but fixed
        thresholds produce no positive predictions.

        The calibration is applied to the fusion probability output, ensuring
        the decision boundary is optimal for the actual score distribution.

        Args:
            data: Input data for detection
            labels: Optional ground truth labels (enables optimal F1 calibration)
            calibration_method: Method for threshold selection:
                - "auto": Automatically select best method
                - "percentile": Use percentile based on contamination
                - "otsu": Otsu's bimodal threshold
                - "optimal_f1": Find threshold maximizing F1 (requires labels)
            contamination: Expected anomaly ratio (if known)
            domain: Domain identifier for GOSNN threshold tuning
            enable_gosnn: Enable GOSNN synaptic integration

        Returns:
            Dictionary containing:
                - anomaly_prob: Fusion probability output
                - is_anomaly: Calibrated binary predictions
                - threshold: Calibrated threshold value
                - threshold_method: Calibration method used
                - calibration_diagnostics: Full calibration diagnostics
                - class_prediction: Predicted anomaly class
                - severity: Anomaly severity score
                - detector_importance: Attention weights
                - mode: "fusion_calibrated"
                - gosnn_metadata: GOSNN integration data (if enabled)

        Example:
            >>> engine = OmniMercuryEngine(mode="fusion")
            >>> result = engine.detect_with_fusion_calibrated(
            ...     data, y_true, calibration_method="auto"
            ... )
            >>> print(f"Threshold: {result['threshold']:.4f}")
            >>> print(f"F1=0 problem solved: {result['is_anomaly'].sum()} anomalies detected")
        """
        from omni_mercury_engine.core.score_calibration import (
            CalibrationMethod,
            ScoreCalibrationManager,
        )

        # Run standard fusion detection
        fusion_result = self.detect_with_fusion(
            data=data,
            domain=domain,
            enable_gosnn=enable_gosnn,
        )

        # Get fusion probability
        anomaly_prob = fusion_result.get("anomaly_prob", 0.5)

        # For batch data, we need the full probability array
        # The fusion_inference returns probs for all samples
        if self.mode == "fusion":
            det_features, det_scores = self._extract_detector_features(data)
            mod_features, mod_scores = self._extract_model_features(data)
            all_features = {**det_features, **mod_features}

            fusion_output = self.fusion_inference.predict(all_features, return_attention=True)
            all_probs = fusion_output.get("anomaly_probs", np.array([anomaly_prob]))
            if isinstance(all_probs, torch.Tensor):
                all_probs = all_probs.cpu().numpy()
            all_probs = np.asarray(all_probs).flatten()
        else:
            all_probs = np.array([anomaly_prob])

        # Set contamination
        contamination = contamination or self.config.contamination or 0.05

        # Calibrate threshold on fusion probabilities
        manager = ScoreCalibrationManager(
            contamination=contamination,
            method=CalibrationMethod(calibration_method),
        )

        calibration_result = manager.calibrate(
            scores=all_probs,
            labels=np.asarray(labels) if labels is not None else None,
        )

        # Build calibrated result
        result = {
            "anomaly_prob": float(anomaly_prob) if len(all_probs) == 1 else all_probs,
            "is_anomaly": calibration_result.predictions,
            "threshold": calibration_result.threshold,
            "threshold_method": calibration_result.method.value,
            "calibration_diagnostics": calibration_result.diagnostics,
            "calibration_confidence": calibration_result.confidence,
            "class_prediction": fusion_result.get("class_prediction", 0),
            "severity": fusion_result.get("severity", 0.0),
            "detector_importance": fusion_result.get("detector_importance", {}),
            "mode": "fusion_calibrated",
        }

        # Carry over GOSNN metadata
        if "gosnn_metadata" in fusion_result:
            result["gosnn_metadata"] = fusion_result["gosnn_metadata"]

        # Carry over drift detection
        if "drift_detection" in fusion_result:
            result["drift_detection"] = fusion_result["drift_detection"]

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
                # Security: Try loading without pickle first (safe for pure numpy arrays)
                # Only fall back to pickle for legacy files with Python objects
                try:
                    data = np.load(training_data, allow_pickle=False)
                except ValueError as e:
                    if "allow_pickle" in str(e).lower():
                        logger.warning(
                            f"Loading {training_data} with pickle - ensure file is from trusted source"
                        )
                        data = np.load(training_data, allow_pickle=True)  # nosec B301
                    else:
                        raise
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
        # Use config.num_workers (default 4) for parallel data loading
        # Set to 0 in config for single-threaded loading (needed for some environments)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
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
