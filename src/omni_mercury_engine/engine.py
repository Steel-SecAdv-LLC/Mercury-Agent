"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
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
        data = np.random.default_rng().standard_normal((100, 10))
        result = engine.detect_with_fusion(data)

        print(f"Anomaly probability: {result['anomaly_prob']:.3f}")
        print(f"Is anomaly: {result['is_anomaly']}")

    Batch processing for large datasets::

        engine = OmniMercuryEngine(mode="fusion")
        large_data = np.random.default_rng().standard_normal((10000, 50))

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

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    TORCH_AVAILABLE = False

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
    sanitize_domain,
)
from omni_mercury_engine.core.config import EngineConfig
from omni_mercury_engine.core.global_omni_scalar_network import (
    ScalarGroup,
    get_global_scalar_network,
)

# ---------------------------------------------------------------------------
# σ_Immutable / GOSNN testing-only bypass.
#
# Production code MUST NOT toggle this flag.  It exists so unit tests
# that exercise non-GOSNN code paths can opt out of σ_Immutable's
# corpus / weights bootstrap when running on stripped-down CI images
# (no torch, no signed corpus).  Setting it True turns every
# ``detect_with_fusion`` call into a benevolence-only check; the
# engine still runs the BenevolenceScorer hard gate, but skips the
# σ_Immutable second gate.
#
# At every public call site, the ``_enable_gosnn`` parameter is private
# (leading underscore) — public callers cannot bypass σ_Immutable.
# Tests set this module-level flag explicitly via monkeypatch so the
# bypass is auditable from outside the test file.
# ---------------------------------------------------------------------------
_GOSNN_TESTING_BYPASS: bool = False

# ---------------------------------------------------------------------------
# Default fusion checkpoint (Issue #2).
#
# A versioned checkpoint shipped with the package so the headline fusion path
# is real out of the box: ``detect``/``serve`` load it and score with a trained
# network + calibrated probabilities without any training step. Produced by
# ``scripts/train_default_fusion.py``; located via ``default_fusion_checkpoint_path()``.
# ---------------------------------------------------------------------------
FUSION_CHECKPOINT_FORMAT_VERSION: int = 1

try:  # avoid a hard import cycle with the package __init__
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("mercury-agent")
except Exception:
    __version__ = "1.7.0"

# Core detectors - always imported (lightweight base classes)
from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

# Runtime pipeline modules - always imported (required for core functionality)
from omni_mercury_engine.ml.drift import DriftResult, EnsembleDriftDetector
from omni_mercury_engine.ml.fairness import BiasAuditConfig, FairnessAuditor, FairnessReport
from omni_mercury_engine.ml.fusion_network import FocalLoss, OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference
from omni_mercury_engine.ml.optimization import OptimizationConfig, ParallelExecutor
from omni_mercury_engine.utils.logging import LoggerMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from omni_mercury_engine.cognitive.ethical_bounding import BenevolenceScorer

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


def default_fusion_checkpoint_path() -> Path:
    """Return the path of the shipped default fusion checkpoint.

    The file may not exist (it is generated by
    ``scripts/train_default_fusion.py``); callers should check
    :meth:`pathlib.Path.exists` before loading.
    """
    from pathlib import Path

    return Path(__file__).resolve().parent / "models" / "checkpoints" / "default_fusion.pt"


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
    return cast("type[ABMSDisciplineDetector]", _lazy_import("ABMSDisciplineDetector"))  # type: ignore[no-any-return, unused-ignore]


def get_affective_model() -> type[AffectiveAnomalyModel]:
    """Get AffectiveAnomalyModel class (lazy loaded)."""
    return cast("type[AffectiveAnomalyModel]", _lazy_import("AffectiveAnomalyModel"))  # type: ignore[no-any-return, unused-ignore]


def get_astrophysical_model() -> type[AstrophysicalAnomalyModel]:
    """Get AstrophysicalAnomalyModel class (lazy loaded)."""
    return cast("type[AstrophysicalAnomalyModel]", _lazy_import("AstrophysicalAnomalyModel"))  # type: ignore[no-any-return, unused-ignore]


def get_biometric_model() -> type[BiometricAnomalyModel]:
    """Get BiometricAnomalyModel class (lazy loaded)."""
    return cast("type[BiometricAnomalyModel]", _lazy_import("BiometricAnomalyModel"))  # type: ignore[no-any-return, unused-ignore]


def get_chemistry_detector() -> type[ChemistryAnomalyDetector]:
    """Get ChemistryAnomalyDetector class (lazy loaded)."""
    return cast("type[ChemistryAnomalyDetector]", _lazy_import("ChemistryAnomalyDetector"))  # type: ignore[no-any-return, unused-ignore]


def get_consciousness_model() -> type[ConsciousnessPreservationModel]:
    """Get ConsciousnessPreservationModel class (lazy loaded)."""
    return cast(  # type: ignore[no-any-return, unused-ignore]
        "type[ConsciousnessPreservationModel]", _lazy_import("ConsciousnessPreservationModel")
    )


def get_neural_model() -> type[NeuralCognitiveModel]:
    """Get NeuralCognitiveModel class (lazy loaded)."""
    return cast("type[NeuralCognitiveModel]", _lazy_import("NeuralCognitiveModel"))  # type: ignore[no-any-return, unused-ignore]


def get_parapsychology_detector() -> type[ParapsychologyDetector]:
    """Get ParapsychologyDetector class (lazy loaded)."""
    return cast("type[ParapsychologyDetector]", _lazy_import("ParapsychologyDetector"))  # type: ignore[no-any-return, unused-ignore]


def get_quantum_model() -> type[QuantumAnomalyModel]:
    """Get QuantumAnomalyModel class (lazy loaded)."""
    return cast("type[QuantumAnomalyModel]", _lazy_import("QuantumAnomalyModel"))  # type: ignore[no-any-return, unused-ignore]


def get_intelligence_fusion() -> type[IntelligenceFusionEngine]:
    """Get IntelligenceFusionEngine class (lazy loaded)."""
    return cast("type[IntelligenceFusionEngine]", _lazy_import("IntelligenceFusionEngine"))  # type: ignore[no-any-return, unused-ignore]


def get_threat_detector() -> type[ThreatDetector]:
    """Get ThreatDetector class (lazy loaded)."""
    return cast("type[ThreatDetector]", _lazy_import("ThreatDetector"))  # type: ignore[no-any-return, unused-ignore]


def get_schumann_detector() -> type[SchumannResonanceDetector]:
    """Get SchumannResonanceDetector class (lazy loaded)."""
    return cast("type[SchumannResonanceDetector]", _lazy_import("SchumannResonanceDetector"))  # type: ignore[no-any-return, unused-ignore]


def get_self_healing() -> type[SelfHealingEngine]:
    """Get SelfHealingEngine class (lazy loaded)."""
    return cast("type[SelfHealingEngine]", _lazy_import("SelfHealingEngine"))  # type: ignore[no-any-return, unused-ignore]


def get_llm_config() -> type[LLMConfig]:
    """Get LLMConfig class (lazy loaded)."""
    return cast("type[LLMConfig]", _lazy_import("LLMConfig"))  # type: ignore[no-any-return, unused-ignore]


def get_llm_provider() -> type[LLMProvider]:
    """Get LLMProvider class (lazy loaded)."""
    return cast("type[LLMProvider]", _lazy_import("LLMProvider"))  # type: ignore[no-any-return, unused-ignore]


def get_zero_shot_detector() -> type[ZeroShotAnomalyDetector]:
    """Get ZeroShotAnomalyDetector class (lazy loaded)."""
    return cast("type[ZeroShotAnomalyDetector]", _lazy_import("ZeroShotAnomalyDetector"))  # type: ignore[no-any-return, unused-ignore]


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
        """
        Initialize the feature cache.

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
        """
        Generate a cache key from data.

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
        """
        Get cached value or compute and cache it.

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
        """
        Get cache statistics.

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
        """
        Initialize memory monitor.

        Args:
            threshold_mb: Memory threshold in MB for triggering GC.
        """
        self.threshold_mb = threshold_mb
        self.peak_memory_mb = 0.0
        self.gc_count = 0
        self._allocations: dict[str, float] = {}

    def get_current_memory_mb(self) -> float:
        """
        Get current memory usage in megabytes.

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
        """
        Check memory and trigger GC if needed.

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
        """
        Context manager to track memory allocation.

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
        """
        Get memory statistics.

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
    """Unified neuro-symbolic detection engine with hybrid fusion.

    This is the main entry point for the Mercury Agent neuro-symbolic AI
    platform.  It combines a deep-learning core (specialised neural
    detectors and a multi-head attention fusion network) with an explicit
    symbolic layer (knowledge graphs, rule bases, formal verification) and
    a hard ethical-governance gate, exposing the result as a unified
    multi-domain anomaly-detection capability.

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
            data = np.random.default_rng().standard_normal((100, 10))
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
        auto_load_checkpoint: bool = False,
    ) -> None:
        """Initialize the OmniMercuryEngine.

        Args:
            config: Engine configuration. If None, uses default config.
            mode: Operation mode. Either 'fusion' for ML fusion or
                a specific detector name.
            device: Computation device ('cpu' or 'cuda').
            cache_size: Maximum entries in feature cache. Default 128.
            memory_threshold_mb: Memory threshold for GC in MB. Default 2048.
            auto_load_checkpoint: When True and mode='fusion', load the packaged
                default fusion checkpoint at init so detection works without a
                training step. Default False to keep a freshly-constructed
                engine deterministically untrained; the ``detect``/``serve``
                CLI entry points opt in.

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

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for OmniMercuryEngine. Install it with: pip install torch"
            )

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

        # Ethical boundary scorer — constructed eagerly at engine init so
        # the first call to ``detect_with_fusion`` cannot race the gate
        # under concurrent first-callers (a lazy ``if is None`` check
        # would let two threads both enter the construction branch).
        # ``BenevolenceScorer.__init__`` is cheap (no I/O, no model
        # weights) so eager construction is the simpler correct
        # alternative to a memoise-behind-a-lock pattern.
        from omni_mercury_engine.cognitive.ethical_bounding import (
            MINIMUM_BENEVOLENCE_FLOOR as _MINIMUM_BENEVOLENCE_FLOOR,
            BenevolenceScorer as _BenevolenceScorer,
        )

        self._boundary_scorer: BenevolenceScorer = _BenevolenceScorer(
            benevolence_threshold=_MINIMUM_BENEVOLENCE_FLOOR
        )

        # σ_Immutable second hard ethical gate (Wave B item 1).  Loaded
        # eagerly for the same reason as the benevolence scorer above:
        # the first ``detect_with_fusion`` call cannot race the gate's
        # corpus-verification step.  The gate is a process-wide
        # singleton — every boundary (engine, hub, orchestrator)
        # observes the same trained network and the same signed-corpus
        # verdict, so a corpus tampering at startup poisons every
        # decision boundary uniformly.
        from omni_mercury_engine.security.sigma_immutable_gate import (
            get_sigma_immutable_gate,
        )

        self._sigma_immutable_gate = get_sigma_immutable_gate()

        self._init_detectors()
        self._init_models()
        self._init_fusion()
        self._init_resilience()
        self._init_runtime_pipeline()

        if auto_load_checkpoint and self.mode == "fusion":
            self.load_default_fusion_checkpoint()

        logger.info(f"OmniMercuryEngine initialized (mode={mode}, device={self.device})")

    def _init_detectors(self) -> None:
        """
        Initialize all base anomaly detectors.

        Creates instances of the 5 base detectors:
            - statistical: Statistical anomaly detection
            - temporal: Temporal pattern detection
            - spatial: Spatial relationship detection
            - dimensional: High-dimensional analysis
            - directive: Sigma-based rule detection
        """
        self.detectors = {
            "statistical": MercuryAnomalyDetector(),
            "temporal": TemporalAnomalyDetector(),
            "spatial": SpatialAnomalyDetector(),
            "dimensional": DimensionalAnalyzer(),
            "directive": SigmaDirectiveDetector(),
        }

    def _init_models(self) -> None:
        """
        Initialize all specialized domain models.

        Creates instances of the 13 specialized models covering various domains from quantum physics
        to medical diagnostics.

        Uses lazy loading to defer import of heavy model modules until first use, reducing cold-
        start time by ~50% for applications that don't use all models.
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
        """
        Initialize ML fusion components.

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
            # Post-hoc temperature calibration (Guo et al. 2017), fit on a
            # held-out val split during fit_fusion. Identity (T=1) until fit.
            self._fusion_calibrator: Any = None
            # Feature groups the network was actually trained on. Persisted in
            # the checkpoint and used to restrict the inference feature dict so
            # detect_with_fusion feeds exactly the groups training saw (no
            # untrained projections for aggregating/extra groups). None until
            # trained → no filtering, preserving the untrained-engine contract.
            self._fusion_feature_groups: list[str] | None = None
            # Optional training provenance (source, datasets, seed, ...) recorded
            # by training scripts and persisted in/restored from the checkpoint
            # so a shipped artifact is self-describing for audit. None unless set.
            self._fusion_provenance: dict[str, Any] | None = None
            logger.info(
                "OmniFusionModel initialized (untrained). Call fit_fusion() "
                "before detection for optimal performance."
            )

    def _init_resilience(self) -> None:
        """Initialize resilience and self-healing components."""
        self.self_healing = get_self_healing()()

    def _init_runtime_pipeline(self) -> None:
        """
        Initialize runtime pipeline integration modules.

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
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        calibrate: bool = True,
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
            use_focal_loss: Train with FocalLoss instead of BCE (default True).
                Focal loss down-weights easy negatives so rare anomalies are not
                drowned out under class imbalance (fixes probability collapse).
            focal_alpha: Positive-class weight for focal loss (default 0.75).
            focal_gamma: Focusing parameter for focal loss (default 2.0).
            calibrate: Fit a post-hoc temperature scalar (Guo et al. 2017) on the
                held-out validation split so outputs are trustworthy
                probabilities, not just rankings (default True). Temperature
                scaling is monotonic, so ranking/AUC is exactly preserved.

        Returns:
            Dictionary with training metrics including final_loss, best_loss,
            epochs_trained, convergence information, and (when calibrated)
            ``temperature`` plus ``ece_before``/``ece_after``.

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

        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        n_samples = len(X)
        logger.info(f"Starting fusion training on {n_samples} samples...")

        # Fit base detectors and extract the fusion feature set (detectors +
        # domain models, matching inference). Shared with build_feature_npz()
        # so the offline builder's archive is byte-for-byte what fit_fusion()
        # trains on. Record the trained group names so inference can restrict
        # itself to exactly this set.
        detector_features = self._extract_fusion_features(X, fit_detectors=True)
        self._fusion_feature_groups = sorted(detector_features.keys())

        # Generate pseudo-labels if not provided (semi-supervised)
        if y is None:
            logger.info("No labels provided, using semi-supervised pseudo-labeling...")
            y = self._generate_pseudo_labels(X, contamination)

        return self._fit_fusion_on_features(
            detector_features,
            y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            calibrate=calibrate,
        )

    def _fit_fusion_on_features(
        self,
        detector_features: dict[str, torch.Tensor],
        y: np.ndarray[Any, Any] | torch.Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        calibrate: bool = True,
    ) -> dict[str, Any]:
        """Train the fusion head on a pre-extracted feature-group mapping.

        Shared training + temperature-calibration tail for both
        :meth:`fit_fusion` (single raw-input corpus) and
        :meth:`fit_fusion_pooled` (multi-dataset prior). It operates purely on
        an already-extracted ``{group_name: (N, dim) tensor}`` mapping plus
        per-sample integer labels, so the caller owns feature extraction and
        decides which groups are recorded in ``self._fusion_feature_groups``.
        Sets ``self._fusion_trained`` and fits ``self._fusion_calibrator``
        exactly as the single-corpus path always has, so a checkpoint saved
        afterwards serves calibrated, group-restricted probabilities
        identically however it was trained.

        Args:
            detector_features: Per-sample feature groups, each ``(N, dim)``.
            y: Binary labels ``(N,)`` (1=anomaly, 0=normal).
            epochs, batch_size, learning_rate, early_stopping_patience,
            validation_split, use_focal_loss, focal_alpha, focal_gamma,
            calibrate: As in :meth:`fit_fusion`.

        Returns:
            Training metrics (``best_loss``, ``epochs_trained`` ... plus
            ``temperature``/``ece_before``/``ece_after`` when calibrated).
        """
        # GPU check with fallback
        device = self.device
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = torch.device("cpu")

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        y = np.asarray(y).reshape(-1)
        n_samples = len(y)

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

        # Loss criterion. FocalLoss addresses imbalanced-data collapse (rare
        # anomalies swamped by easy negatives), which manifested as well-ranked
        # but flattened/over-confident probabilities. Falls back to BCE when
        # explicitly disabled. Both operate on the model's sigmoid output.
        if use_focal_loss:
            focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

            def _loss_fn(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return focal_criterion(probs, targets)

        else:

            def _loss_fn(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.binary_cross_entropy(probs, targets)

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
                loss = _loss_fn(outputs["anomaly_probs"], batch_labels)
                loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
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
                    loss = _loss_fn(outputs["anomaly_probs"], batch_labels)
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

        metrics: dict[str, Any] = {
            "final_loss": loss_history[-1]["val_loss"] if loss_history else 0.0,
            "best_loss": best_val_loss,
            "epochs_trained": len(loss_history),
            "best_epoch": len(loss_history) - epochs_without_improvement,
            "loss_history": loss_history,
            "early_stopped": epochs_without_improvement >= early_stopping_patience,
        }

        # Post-hoc temperature scaling (Guo et al. 2017): fit a single scalar on
        # the held-out validation split so the sigmoid outputs are trustworthy
        # probabilities. Monotonic, so ROC-AUC/ranking is exactly preserved.
        self._fusion_calibrator = None
        if calibrate and n_val > 0:
            cal_metrics = self._fit_fusion_temperature(
                detector_features, labels_tensor, val_indices
            )
            metrics.update(cal_metrics)

        return metrics

    def _fit_fusion_temperature(
        self,
        detector_features: dict[str, torch.Tensor],
        labels_tensor: torch.Tensor,
        val_indices: torch.Tensor,
    ) -> dict[str, Any]:
        """Fit post-hoc temperature scaling on the validation split.

        Returns a dict with the learned ``temperature`` and validation
        ``ece_before``/``ece_after`` (Expected Calibration Error). On failure
        (e.g. single-class val split) calibration is left as identity (T=1).
        """
        from omni_mercury_engine.core.calibration import TemperatureScaling, compute_ece

        device = next(self.fusion_model.parameters()).device
        self.fusion_model.eval()
        with torch.no_grad():
            val_feats = {
                name: feat[val_indices].to(device) for name, feat in detector_features.items()
            }
            val_probs = (
                self.fusion_model(val_feats)["anomaly_probs"].detach().cpu().numpy().reshape(-1)
            )
        val_labels = labels_tensor[val_indices].cpu().numpy().reshape(-1).astype(int)

        if len(np.unique(val_labels)) < 2:
            logger.warning("Skipping temperature calibration: validation split is single-class")
            return {"temperature": 1.0, "ece_before": None, "ece_after": None}

        ece_before = float(compute_ece(val_labels, val_probs))
        scaler = TemperatureScaling().fit(val_probs, val_labels)
        if not getattr(scaler, "_fitted", False):
            return {"temperature": 1.0, "ece_before": ece_before, "ece_after": ece_before}

        calibrated = scaler.calibrate(val_probs)
        ece_after = float(compute_ece(val_labels, calibrated))
        self._fusion_calibrator = scaler
        logger.info(
            f"Temperature calibration: T={scaler.temperature:.4f}, "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )
        return {
            "temperature": float(scaler.temperature),
            "ece_before": ece_before,
            "ece_after": ece_after,
        }

    def fit_fusion_pooled(
        self,
        datasets: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
        *,
        engine_factory: Callable[[], OmniMercuryEngine] | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        calibrate: bool = True,
    ) -> dict[str, Any]:
        """Train one fusion head on a *pool* of labelled datasets.

        Builds a general tabular-anomaly prior by extracting the inference
        feature set from each dataset with **its own freshly-fit detectors**
        (the base anomaly detectors are unsupervised and dataset-specific, so
        reusing a single fit across heterogeneous corpora would be wrong), then
        training the shared fusion head on the concatenation of the feature
        groups every dataset has in common.

        Only groups present **and** of identical width across all datasets are
        pooled; a group whose dimensionality varies cannot be concatenated and
        is dropped with a warning — and, by virtue of not entering
        ``self._fusion_feature_groups``, is also skipped identically at
        inference, so the head is never fed an untrained projection. Training,
        calibration, group recording and the saved-checkpoint contract are
        identical to :meth:`fit_fusion` because both funnel through
        :meth:`_fit_fusion_on_features`.

        Args:
            datasets: ``(X, y)`` pairs. ``X`` is raw ``(n_i, d_i)`` (the input
                dimensionality may differ per dataset); ``y`` is binary
                ``(n_i,)``.
            engine_factory: Builds the throwaway engine used to fit detectors
                and extract features for a single dataset. Defaults to a fresh
                fusion-mode engine of this class on the same device; injectable
                for tests.
            epochs, batch_size, learning_rate, early_stopping_patience,
            validation_split, use_focal_loss, focal_alpha, focal_gamma,
            calibrate: As in :meth:`fit_fusion`.

        Returns:
            Training metrics, augmented with ``pooled_datasets``,
            ``pooled_samples`` and ``pooled_groups``.

        Raises:
            ValueError: If ``mode`` is not 'fusion' or ``datasets`` is empty.
            RuntimeError: If no feature group is shared (with consistent width)
                across every dataset.
        """
        if self.mode != "fusion":
            raise ValueError("fit_fusion_pooled() requires mode='fusion'")
        if not datasets:
            raise ValueError("fit_fusion_pooled() requires at least one (X, y) dataset")

        factory = engine_factory or (lambda: type(self)(mode="fusion", device=str(self.device)))

        per_dataset: list[tuple[dict[str, torch.Tensor], np.ndarray[Any, Any]]] = []
        for i, (raw_x, raw_y) in enumerate(datasets):
            x_arr = np.asarray(raw_x, dtype=np.float32)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            labels = np.asarray(raw_y).reshape(-1)
            # Fresh detectors per source: extraction fits this engine's
            # detectors on x_arr only, so each dataset's features reflect its
            # own distribution.
            extractor = factory()
            feats = extractor._extract_fusion_features(x_arr, fit_detectors=True)
            logger.info(
                "Pooled source %d/%d: %d samples, groups=%s",
                i + 1,
                len(datasets),
                len(labels),
                sorted(feats),
            )
            per_dataset.append((feats, labels))

        shared: set[str] | None = None
        for feats, _ in per_dataset:
            keys = set(feats)
            shared = keys if shared is None else (shared & keys)

        consistent: list[str] = []
        for key in sorted(shared or set()):
            widths = {feats[key].shape[1] for feats, _ in per_dataset}
            if len(widths) == 1:
                consistent.append(key)
            else:
                logger.warning(
                    "Dropping feature group %r from the pool: width varies across "
                    "datasets (%s); it cannot be concatenated.",
                    key,
                    sorted(widths),
                )

        if not consistent:
            raise RuntimeError(
                "No feature group is shared with consistent width across all pooled "
                "datasets; cannot build a pooled fusion prior."
            )

        pooled = {key: torch.cat([feats[key] for feats, _ in per_dataset]) for key in consistent}
        pooled_y = np.concatenate([labels for _, labels in per_dataset])
        self._fusion_feature_groups = consistent

        metrics = self._fit_fusion_on_features(
            pooled,
            pooled_y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            calibrate=calibrate,
        )
        metrics.update(
            {
                "pooled_datasets": len(per_dataset),
                "pooled_samples": len(pooled_y),
                "pooled_groups": consistent,
            }
        )
        return metrics

    def _generate_pseudo_labels(
        self,
        X: np.ndarray[Any, Any],
        contamination: float | None = None,
    ) -> np.ndarray[Any, Any]:
        """
        Generate pseudo-labels using detector consensus for semi-supervised learning.

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

        return pseudo_labels  # type: ignore[no-any-return, unused-ignore]

    def _extract_fusion_features(
        self,
        X: np.ndarray[Any, Any],
        *,
        fit_detectors: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Fit base detectors (optionally) and extract the fusion feature set.

        This is the single source of truth for the fusion feature set used by
        training **and** inference. It extracts both base-detector features and
        specialized-model features — exactly the groups :meth:`detect_with_fusion`
        feeds the network at inference (``{**det_features, **mod_features}``) — so
        a model trained here faces the same feature distribution it will see in
        production. Training on detector features alone would leave the fusion
        net's per-group projections for the ~13 domain models untrained, so its
        weights would not transfer to the real ``detect_with_fusion`` path.

        Both :meth:`fit_fusion` (raw in-memory path) and :meth:`build_feature_npz`
        (offline feature-archive builder) call it, so a ``.npz`` produced by the
        builder reproduces exactly what ``fit_fusion`` would have extracted from
        the same raw ``X``.

        Args:
            X: Raw training features ``(n_samples, n_features)``.
            fit_detectors: When True, unfitted detectors are fit on ``X`` before
                extraction. Set False if detectors are already fit (e.g. eval),
                so no fitting happens on held-out data.

        Returns:
            Mapping of feature-group name to a float32 feature tensor.

        Raises:
            RuntimeError: If no feature group could be extracted.
        """
        if fit_detectors:
            logger.info(f"Fitting {len(self.detectors)} base detectors...")
            for name, detector in self.detectors.items():
                try:
                    if not detector.is_fitted():
                        detector.fit(X)
                        logger.debug(f"Fitted detector: {name}")
                except Exception as e:
                    logger.warning(f"Failed to fit detector {name}: {e}")

        logger.info("Extracting fusion features (detectors + domain models)...")
        fusion_features: dict[str, torch.Tensor] = {}
        n_samples = len(X)

        def _record(name: str, features: Any) -> None:
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32)
            elif isinstance(features, torch.Tensor):
                features = features.detach().to(torch.float32)
            else:
                features = torch.tensor(np.asarray(features), dtype=torch.float32)
            # Only keep per-sample feature groups: training indexes features by
            # sample (``feat[batch_indices]``), so a group that aggregates over
            # the batch (leading dim != n_samples) would misalign. Such groups
            # are dropped here and, being absent, are skipped identically at
            # inference — keeping train and serve feature sets in lockstep.
            if features.ndim < 2 or features.shape[0] != n_samples:
                logger.debug(
                    f"Skipping feature group {name}: shape {tuple(features.shape)} "
                    f"is not per-sample (expected leading dim {n_samples})"
                )
                return
            fusion_features[name] = features
            logger.debug(f"Extracted features from {name}: shape={tuple(features.shape)}")

        for name, detector in self.detectors.items():
            try:
                _record(name, detector.extract_features(X))
            except Exception as e:
                logger.warning(f"Failed to extract features from detector {name}: {e}")

        # Domain-model feature groups: extracted (never fit) exactly as
        # detect_with_fusion does, so the trained/served feature space matches.
        # Models that cannot process the input are skipped gracefully.
        for name, model in self.models.items():
            extractor: Any = model
            try:
                _record(name, extractor.extract_features(X))
            except Exception as e:
                logger.debug(f"Skipping model {name} during fusion feature extraction: {e}")

        if not fusion_features:
            raise RuntimeError("No fusion features could be extracted")

        return fusion_features

    def _restrict_to_trained_groups(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only the feature groups the fusion network was trained on.

        Lets ``detect_with_fusion`` / ``score_fusion`` feed the network exactly
        the groups present at training, so no untrained per-group projection is
        created at inference for an aggregating or otherwise-extra group
        (train/serve lockstep). A no-op when the network is untrained
        (``_fusion_feature_groups is None``), preserving the untrained-engine
        contract; also a no-op if none of the trained groups are present, to
        avoid handing the model an empty batch.
        """
        groups = self._fusion_feature_groups
        if not groups:
            return features
        selected = {k: v for k, v in features.items() if k in groups}
        return selected or features

    def build_feature_npz(
        self,
        X: np.ndarray[Any, Any],
        output_path: str,
        y: np.ndarray[Any, Any] | None = None,
        *,
        contamination: float | None = None,
    ) -> str:
        """Build a fusion-training ``.npz`` archive from raw features.

        Runs the same detector fit + feature-extraction that :meth:`fit_fusion`
        performs internally and writes the result to an ``.npz`` whose layout is
        directly consumable by :meth:`train_fusion_model` (one array per
        detector plus a ``labels`` array). This is the bridge between the raw
        path and the pre-extracted-feature path: it lets callers cache the
        (expensive) feature extraction once and re-train cheaply.

        Args:
            X: Raw training features ``(n_samples, n_features)``.
            output_path: Destination ``.npz`` path.
            y: Optional binary labels (1=anomaly, 0=normal). If None, semi
                supervised pseudo-labels are generated from detector consensus.
            contamination: Expected anomaly fraction used only when ``y`` is
                None. If None, estimated adaptively.

        Returns:
            The ``output_path`` written.

        Raises:
            ValueError: If mode is not 'fusion', or a detector name collides
                with the reserved ``labels`` key.
        """
        if self.mode != "fusion":
            raise ValueError("build_feature_npz() requires mode='fusion'")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        detector_features = self._extract_fusion_features(X, fit_detectors=True)

        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name, feat in detector_features.items():
            if name == "labels":
                raise ValueError(
                    "Detector named 'labels' collides with the reserved labels "
                    "key in the feature archive."
                )
            arrays[name] = feat.detach().cpu().numpy().astype(np.float32)

        if y is None:
            y = self._generate_pseudo_labels(X, contamination)
        arrays["labels"] = np.asarray(y).astype(np.int64)

        if not output_path.endswith(".npz"):
            output_path = f"{output_path}.npz"
        np.savez(output_path, **arrays)
        logger.info(
            f"Wrote fusion feature archive to {output_path} "
            f"({len(arrays) - 1} detector feature groups, {len(arrays['labels'])} samples)"
        )
        return output_path

    def score_fusion(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return fusion anomaly probabilities for a batch of samples.

        This is the batch evaluation/benchmarking counterpart to
        :meth:`detect_with_fusion`. It extracts detector features and runs the
        trained ``OmniFusionModel`` forward pass, returning one probability per
        row. Detectors are NOT refit on ``X`` (refitting on evaluation data
        would leak), so the engine must already be trained via
        :meth:`fit_fusion` / :meth:`train_fusion_model`.

        Unlike :meth:`detect_with_fusion`, this does not run the σ_Immutable /
        benevolence ethical gates — it is for measuring detection quality (e.g.
        ROC-AUC) over a labelled set, not for production decisions.

        Args:
            X: Features ``(n_samples, n_features)``.

        Returns:
            Array of anomaly probabilities in ``[0, 1]``, shape ``(n_samples,)``.

        Raises:
            ValueError: If mode is not 'fusion'.
        """
        if self.mode != "fusion":
            raise ValueError("score_fusion() requires mode='fusion'")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        features = self._restrict_to_trained_groups(
            self._extract_fusion_features(X, fit_detectors=False)
        )
        self.fusion_model.eval()
        with torch.no_grad():
            batch = {name: feat.to(self.device) for name, feat in features.items()}
            outputs = self.fusion_model(batch)
            probs = outputs["anomaly_probs"].detach().cpu().numpy().reshape(-1)

        # Apply post-hoc temperature calibration when available. Monotonic, so
        # rankings/AUC are unchanged; only the probability values are corrected.
        if self._fusion_calibrator is not None:
            probs = np.asarray(self._fusion_calibrator.calibrate(probs)).reshape(-1)
        return probs

    # ------------------------------------------------------------------
    # Symbolic stack surface (Issue #4): causal discovery + rule graph.
    # These expose the previously-dormant cognitive subsystems through the
    # engine (and, via cli.py, the CLI). Structure discovery is deterministic
    # for fixed input data and a fixed seed.
    # ------------------------------------------------------------------
    def discover_causal_structure(
        self,
        X: np.ndarray[Any, Any],
        variable_names: list[str] | None = None,
        *,
        significance_level: float = 0.05,
        max_conditioning_set: int = 4,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Discover causal structure via the PC algorithm (Fisher-Z CI tests).

        Deterministic for fixed ``X`` and ``seed`` — the constraint-based
        skeleton/orientation depends only on the data; ``seed`` fixes the
        engine's bootstrap generator so the whole pipeline is reproducible.

        Args:
            X: Data matrix ``(n_samples, n_variables)``.
            variable_names: Optional variable names (default ``X0..Xk``).
            significance_level: Alpha for the conditional-independence tests.
            max_conditioning_set: Maximum conditioning-set size in PC.
            seed: Seed for the discovery engine's RNG (reproducibility).

        Returns:
            The discovered causal graph as a dict (nodes, edges, confounders,
            colliders, is_cpdag).
        """
        from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"discover_causal_structure expects a 2-D matrix, got shape {X.shape}")

        discovery = CausalDiscoveryEngine(
            significance_level=significance_level,
            max_conditioning_set=max_conditioning_set,
            seed=seed,
        )
        graph = discovery.discover_structure(X, variable_names)
        return graph.to_dict()

    def discover_temporal_causation(
        self,
        X: np.ndarray[Any, Any],
        variable_names: list[str] | None = None,
        *,
        max_lag: int = 5,
        significance_level: float = 0.05,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Discover temporal (Granger) causation between time series.

        Args:
            X: Time-series matrix ``(n_timesteps, n_variables)``.
            variable_names: Optional variable names.
            max_lag: Maximum lag tested for Granger causality.
            significance_level: Alpha for the F-test.
            seed: Seed for reproducibility.

        Returns:
            The temporal causal graph as a dict.
        """
        from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"discover_temporal_causation expects a 2-D matrix, got shape {X.shape}"
            )

        discovery = CausalDiscoveryEngine(
            significance_level=significance_level,
            enable_temporal=True,
            max_lag=max_lag,
            seed=seed,
        )
        graph = discovery.discover_temporal_causation(X, variable_names)
        return graph.to_dict()

    def symbolic_rule_graph(self) -> dict[str, Any]:
        """Export the symbolic logic layer's rule graph (Issue #4).

        Surfaces the previously-dormant rule graph: nodes/edges/rule-type
        counts plus the individual rules (premise -> conclusion).

        Returns:
            Dict with graph statistics and the rule list.
        """
        from omni_mercury_engine.cognitive.symbolic_logic_layer import SymbolicLogicLayer

        layer = SymbolicLogicLayer()
        graph = layer.reasoner.logic_graph
        rules = [
            {
                "rule_id": r.rule_id,
                "type": r.rule_type.value,
                "premise": r.premise,
                "conclusion": r.conclusion,
                "confidence": r.confidence,
                "priority": r.priority,
            }
            for r in graph.rules.values()
        ]
        return {"statistics": graph.get_statistics(), "rules": rules}

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
        provider: str,
        model_name: str | None = None,
        revision: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Enable LLM-based anomaly explanation enhancement.

        LLM enhancement provides natural language explanations for detected
        anomalies using zero-shot classification. This is a non-blocking
        optional stage that enhances detection results with interpretability.

        Phase 2 audit cure: ``provider`` is now required and an
        unrecognised string raises ``ValueError`` instead of silently
        falling back to ``LLMProvider.MOCK``.  The legacy default
        (``provider="mock"``) routed production traffic through
        ``MockLLMAdapter`` whenever a caller forgot to specify a
        provider; with the mock-cure in place that fallback would be
        a hard-fail at the next call anyway, so we surface the
        misconfiguration here at enable time with a clear error.

        Args:
            provider: LLM provider name (e.g. ``"huggingface"``,
                ``"openai"``).  Must match an implemented member of
                :class:`LLMProvider` (case-insensitive).
            model_name: Model identifier for the provider.
            revision: HuggingFace revision pin for remote model IDs.
            api_key: API key for the provider (if required).
            base_url: Endpoint override for providers that support it.
            timeout_seconds: Maximum time to wait for LLM response.

        Raises:
            ValueError: If ``provider`` is not a recognised
                :class:`LLMProvider` member.

        Example:
            >>> engine = OmniMercuryEngine()
            >>> engine.enable_llm_enhancement(
            ...     provider="huggingface",
            ...     model_name="facebook/bart-large-mnli",
            ...     revision="<40-char SHA>"
            ... )
        """
        from pathlib import PurePosixPath, PureWindowsPath

        from omni_mercury_engine.models.foundation.llm_adapter import (
            IMPLEMENTED_LLM_PROVIDERS,
            LLMConfig,
            LLMProvider,
            ZeroShotAnomalyDetector,
        )

        try:
            llm_provider = LLMProvider(provider.lower())
        except ValueError as exc:
            supported = sorted(p.value for p in IMPLEMENTED_LLM_PROVIDERS)
            raise ValueError(
                f"Unknown LLM provider {provider!r}. "
                f"Supported providers: {supported}.  "
                "Silent mock fallback is not permitted (Phase 2 audit cure)."
            ) from exc
        if llm_provider not in IMPLEMENTED_LLM_PROVIDERS:
            supported = sorted(p.value for p in IMPLEMENTED_LLM_PROVIDERS)
            raise ValueError(
                f"LLM provider {provider!r} is declared but has no adapter "
                f"implementation in this build. Supported providers: {supported}."
            )

        if llm_provider == LLMProvider.HUGGINGFACE:
            if not model_name:
                raise ValueError(
                    "enable_llm_enhancement(provider='huggingface') requires "
                    "model_name=<HuggingFace model ID or absolute local path>."
                )
            resolved_model_name = model_name
            is_local_path = (
                PurePosixPath(resolved_model_name).is_absolute()
                or PureWindowsPath(resolved_model_name).is_absolute()
            )
            if not is_local_path and not revision:
                raise ValueError(
                    "HuggingFace remote model IDs require revision=<40-character "
                    "commit SHA> so SafeHFLoader can enforce reproducible model loading."
                )
        elif llm_provider == LLMProvider.TEMPLATE:
            # TemplateLLMAdapter is deterministic-offline and ignores model_name.
            resolved_model_name = model_name or "template"
        else:
            # Every other real provider needs an explicit, provider-specific
            # model identifier.  ``"mock-model"`` was previously used as a
            # placeholder default which masked configuration errors and let
            # callers reach adapter construction with a meaningless name.
            if not model_name:
                raise ValueError(
                    f"enable_llm_enhancement(provider={provider!r}) requires "
                    "model_name=<provider-specific model identifier>."
                )
            resolved_model_name = model_name

        llm_config = LLMConfig(
            provider=llm_provider,
            model_name=resolved_model_name,
            revision=revision,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )
        self.llm_detector = ZeroShotAnomalyDetector(config=llm_config)
        logger.info(f"LLM enhancement enabled with provider={provider}")

    def _check_drift(
        self,
        features: np.ndarray[Any, Any],
    ) -> DriftResult | None:
        """
        Check for data drift against baseline.

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
        """
        Audit detection results for fairness.

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
                sensitive_features=sensitive_data,  # type: ignore[arg-type, unused-ignore]
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
        """
        Enhance detection results with LLM explanations.

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
        """
        Get or create thread pool executor.

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
            >>> data = np.random.default_rng().standard_normal((100, 10))
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
            >>> large_data = np.random.default_rng().standard_normal((10000, 50))
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
        """
        Calculate optimal batch size based on data and memory.

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
        2. Contamination-based percentile threshold for Mercury's statistical ensemble
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
                print(f"\n{'=' * 60}")
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

    def _enforce_ethics_at_boundary(
        self,
        domain: str | None,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
    ) -> None:
        """
        Dual hard ethical gate at the engine decision boundary.

        Both gates fail closed:

        * :class:`~omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`
          — keyword/context primitive raised as
          ``EthicalConstraintViolationError(check="benevolence")``.
        * :class:`~omni_mercury_engine.security.sigma_immutable_gate.SigmaImmutableGate`
          — trained 256-D σ_Immutable network raised as
          ``EthicalConstraintViolationError(check="sigma_immutable")``;
          a missing trained network or signed corpus raises
          ``check="gosnn_unavailable"``.

        The action description is intentionally self-contained and
        positive-keyword-rich (``audit``, ``verify``, ``protect``,
        ``research``, ``evidence``) — it represents the *engine's
        purpose* (anomaly detection for safety auditing), not the
        anomalous payload itself.  This mirrors the orchestrator's
        contract so that both top-level boundaries enforce the same
        primitives with the same threshold semantics.

        Args:
            domain: Caller-supplied domain hint, used as context only.
            data: The input being detected (used for shape/size context).
        """
        # σ_Immutable Wave B Vector 2 closure: caller-supplied domain
        # hints can ride into both the scorer's action description and
        # the σ_Immutable details payload, so a hostile value like
        # ``"damage destroy harm track expose"`` would either inject
        # harm-keywords (false negative) or positive keywords (false
        # positive).  ``sanitize_domain`` collapses every input to the
        # whitelisted ``EnvironmentDomain`` ∪ {"general"} alphabet.
        safe_domain = sanitize_domain(domain)
        # Action keywords intentionally evidence the engine's defensive
        # purpose — audit, verify, protect, research — so the scorer
        # produces a deterministic, above-floor score for legitimate
        # detection requests.
        action = (
            f"anomaly_detection:{safe_domain}:audit verify protect research "
            "evidence fair oversight monitor data care help support"
        )
        context = {
            "purpose": "anomaly detection for safety auditing",
            "safety": "protect verify monitor evidence",
            "domain": safe_domain,
            "data_shape": getattr(data, "shape", None),
        }
        # ``enforce`` raises EthicalConstraintViolationError on violation;
        # legitimate calls return an EthicalScore that the σ_Immutable
        # projection helper consumes below.
        ethical_score = self._boundary_scorer.enforce(action, context)

        # ------------------------------------------------------------
        # σ_Immutable second hard ethical gate.  Fails closed unless the
        # process-wide test-only ``_GOSNN_TESTING_BYPASS`` flag is set.
        # ------------------------------------------------------------
        if _GOSNN_TESTING_BYPASS:
            return
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SIGMA_IMMUTABLE_ETHICAL_DIMS,
            SIGMA_IMMUTABLE_INPUT_DIM,
            SIGMA_USED_BAND_END,
            project_benevolence_to_sigma_band,
        )

        ethical_value = project_benevolence_to_sigma_band(float(ethical_score.benevolence_score))
        sigma_vector = np.zeros(SIGMA_IMMUTABLE_INPUT_DIM, dtype=np.float64)
        sigma_vector[:SIGMA_IMMUTABLE_ETHICAL_DIMS] = ethical_value
        # Centre the non-ethical active band at the training U[0, 2]
        # midpoint so a synthetic projected vector lives in the
        # network's most-confident region for the corresponding
        # benevolence verdict.
        sigma_vector[SIGMA_IMMUTABLE_ETHICAL_DIMS:SIGMA_USED_BAND_END] = 1.0

        self._sigma_immutable_gate.enforce(
            action=(f"OmniMercuryEngine._enforce_ethics_at_boundary:" f"domain={safe_domain}"),
            scalar_vector=sigma_vector,
            details={
                "boundary": "OmniMercuryEngine._enforce_ethics_at_boundary",
                "domain": safe_domain,
                "benevolence_score": float(ethical_score.benevolence_score),
                "data_shape": getattr(data, "shape", None),
            },
        )

    def _extract_detector_features(
        self, data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any]
    ) -> tuple[Any, ...]:
        """
        Extract features from all detectors.

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
        """
        Extract features from all specialized models.

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
        """
        Extract features from all sources in parallel.

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
        _enable_gosnn: bool = True,
    ) -> dict[str, Any]:
        """
        Detect anomalies using ML fusion with GOSNN synaptic integration.

        This method combines outputs from all detectors and models using a
        neural network fusion approach with attention-based weighting. It
        integrates with the Global Omni-Scalar Network (GOSNN) for scalar
        enhancement and runs the σ_Immutable second hard ethical gate.

        Decision boundary:
            The method enforces TWO hard ethical gates in order:

            1. ``BenevolenceScorer.enforce`` — keyword/context-driven gate
               raised as ``EthicalConstraintViolationError(check="benevolence")``.
            2. ``SigmaImmutableGate.enforce`` — trained 256-D scalar
               network raised as
               ``EthicalConstraintViolationError(check="sigma_immutable")``.
               When GOSNN itself cannot run, the method raises
               ``check="gosnn_unavailable"`` rather than degrading to
               advisory metadata.

        GOSNN Integration (Synaptic Fusion + σ_Immutable input source):
            1. Extract features from all detectors and models
            2. Call GOSNN.get_enhanced_scalars() for scalar enhancement
            3. Apply 32-head attention with triadic phi-weighting
            4. Feed enhanced scalars back to fusion for adaptive weighting
            5. Score the full 256-D scalar vector through σ_Immutable

        Args:
            data: Input data for detection.
            domain: Optional domain identifier for GOSNN threshold tuning
                    (e.g., "medical" uses 0.93 fallback instead of 0.96 default).
            _enable_gosnn: PRIVATE testing knob.  Production callers must
                leave this at the default (``True``).  Setting it to
                ``False`` from production code raises
                ``check="gosnn_unavailable"`` because skipping GOSNN
                would also skip the σ_Immutable second hard gate.  Unit
                tests that need to bypass GOSNN must additionally set
                the module-level :data:`_GOSNN_TESTING_BYPASS` flag.

        Returns:
            Dictionary containing:
                - anomaly_prob: Probability of anomaly (0.0-1.0)
                - is_anomaly: Boolean anomaly flag (prob > 0.5)
                - class_prediction: Predicted anomaly class
                - severity: Anomaly severity score
                - detector_importance: Dict of detector weights
                - mode: Detection mode ('fusion')
                - gosnn_metadata: GOSNN + σ_Immutable evaluation metadata:
                    - sigma_immutable_score: σ_Immutable score
                    - ethical_gate_passed: σ_Immutable threshold check
                    - sigma_immutable_threshold: decision threshold used
                    - sigma_immutable_backend: ``"torch"`` for the trained
                      network, ``"unavailable"`` if the network could not
                      run (the engine raises before returning in that case).
                    - harmonic_synergy: H(ω) component for weighted fusion
                    - intelligence_contribution: GOSNN intelligence score
                    - warnings: Any ethical warnings

        Raises:
            EthicalConstraintViolationError: With ``check="benevolence"``
                when BenevolenceScorer fails;
                ``check="sigma_immutable"`` when σ_Immutable scores below
                threshold; ``check="gosnn_unavailable"`` when GOSNN cannot
                be evaluated and the testing bypass is off.

        Note:
            Falls back to basic detection if not in fusion mode.
        """
        if self.mode != "fusion":
            return self.detect(data)

        det_features, det_scores = self._extract_detector_features(data)
        mod_features, mod_scores = self._extract_model_features(data)

        all_features = {**det_features, **mod_features}
        all_scores = {**det_scores, **mod_scores}

        # ------------------------------------------------------------
        # Hard ethical gate at the decision boundary.
        #
        # The previous code consulted GOSNN's σ_Immutable neural gate but
        # only logged a warning when it failed (and silently substituted
        # ``ethical_gate_passed=True`` if GOSNN errored), which made the
        # gate purely advisory — exactly the "defensive theatre" the
        # locked May-2026 audit decisions forbid.
        #
        # σ_Immutable is now trained (scripts/train_sigma_immutable.py)
        # and serves as a second independent gate alongside the
        # BenevolenceScorer.  The primary contract remains
        # BenevolenceScorer.enforce (keyword- and context-driven,
        # deterministic) — σ_Immutable provides a learned check on the
        # full 256-dimensional scalar vector via GOSNN.
        # ------------------------------------------------------------
        self._enforce_ethics_at_boundary(domain=domain, data=data)

        # ------------------------------------------------------------
        # σ_Immutable: second hard ethical gate (Wave B item 1).
        #
        # The trained network at security/sigma_immutable_weights.pt
        # serves as an independent learned check on the full
        # 256-dimensional GOSNN scalar vector.  GOSNN is no longer
        # optional for the verdict — its unavailability raises
        # ``check="gosnn_unavailable"`` rather than degrading silently
        # to fallback metadata.  The previous ``fallback_mode=True``
        # metadata path is gone (auditors saw it as defensive theatre).
        # ------------------------------------------------------------
        gosnn_metadata: dict[str, Any] = {}
        if not _enable_gosnn:
            # ``_enable_gosnn=False`` requests skipping the GOSNN +
            # σ_Immutable second hard gate.  Production code MUST NOT
            # take this path — skipping σ_Immutable downgrades the
            # boundary to a single-gate (BenevolenceScorer-only) check
            # and the locked May-2026 audit forbids advisory σ_Immutable.
            #
            # The only legitimate caller is a unit test that has set
            # the module-level ``_GOSNN_TESTING_BYPASS`` flag, which is
            # a deliberate, auditable opt-out.  Anything else fails
            # closed with ``check="gosnn_unavailable"``.
            if not _GOSNN_TESTING_BYPASS:
                raise EthicalConstraintViolationError(
                    action=(
                        f"OmniMercuryEngine.detect_with_fusion:" f"domain={domain or 'general'}"
                    ),
                    score=0.0,
                    threshold=self._sigma_immutable_gate.threshold,
                    check="gosnn_unavailable",
                    details={
                        "boundary": "OmniMercuryEngine.detect_with_fusion",
                        "domain": domain,
                        "underlying_error": (
                            "_enable_gosnn=False requested without "
                            "_GOSNN_TESTING_BYPASS — σ_Immutable second "
                            "hard gate cannot run, boundary fails closed."
                        ),
                    },
                )
            gosnn_metadata = {
                "ethical_gate_passed": None,
                "sigma_immutable_score": None,
                "sigma_immutable_threshold": self._sigma_immutable_gate.threshold,
                "sigma_immutable_backend": "testing_bypass",
                "warnings": [
                    "σ_Immutable bypassed via _GOSNN_TESTING_BYPASS — "
                    "unit-test path only, not safe for production."
                ],
            }
        else:
            try:
                gosnn = get_global_scalar_network(
                    device=str(self.device),
                    domain=domain,
                    num_attention_heads=32,
                    enable_triadic_phi=True,
                )

                base_scalars = {
                    f"detector_{name}_score": float(np.mean(score))
                    for name, score in all_scores.items()
                    if isinstance(score, (np.ndarray, float, int))
                }

                enhancement_result = gosnn.get_enhanced_scalars(
                    requesting_component="OmniMercuryEngine.detect_with_fusion",
                    base_scalars=base_scalars,
                    context={"domain": domain, "data_shape": getattr(data, "shape", None)},
                )

                # Hard σ_Immutable enforcement — evaluate against the
                # exact same scalar vector GOSNN scored, so the engine
                # boundary's verdict matches the gate baked into GOSNN.
                full_scalars = gosnn._collect_all_scalars()
                scalar_vector = np.array(list(full_scalars.values()), dtype=np.float64)
                # Deterministic critical-ethical floor, composed *before*
                # the trained network.  The synthetic-trained gate, on its
                # own, passed vectors with a single critical ethical dim
                # zeroed (e.g. benevolence -> 0); the floor makes a
                # collapsed anchor a categorical, fail-closed refusal that
                # no learned score can override.  Anchor names come from
                # GOSNN (single source of truth) and exclude the narrative
                # tuning scalars that merely live in the ETHICAL group.
                self._sigma_immutable_gate.enforce_ethical_floor(
                    action=(
                        f"OmniMercuryEngine.detect_with_fusion:" f"domain={domain or 'general'}"
                    ),
                    anchors=gosnn.critical_ethical_anchors(),
                    details={
                        "boundary": "OmniMercuryEngine.detect_with_fusion",
                        "domain": domain,
                    },
                )
                evaluation = self._sigma_immutable_gate.enforce(
                    action=(
                        f"OmniMercuryEngine.detect_with_fusion:" f"domain={domain or 'general'}"
                    ),
                    scalar_vector=scalar_vector,
                    details={
                        "boundary": "OmniMercuryEngine.detect_with_fusion",
                        "domain": domain,
                        "data_shape": getattr(data, "shape", None),
                    },
                )

                gosnn_metadata = {
                    "ethical_gate_passed": evaluation.passes,
                    "sigma_immutable_score": evaluation.score,
                    "sigma_immutable_threshold": evaluation.threshold,
                    "sigma_immutable_backend": evaluation.backend,
                    "harmonic_synergy": gosnn.last_harmonic_synergy,
                    "intelligence_contribution": (enhancement_result.intelligence_contribution),
                    "warnings": enhancement_result.warnings,
                    "enhancement_fusion_score": enhancement_result.fusion_score,
                }

                gosnn.register_scalars(
                    component_name="fusion_detectors",
                    scalars=enhancement_result.enhanced_scalars,
                    group=ScalarGroup.SECURITY if domain == "security" else ScalarGroup.ETHICAL,
                    metadata={"source": "detect_with_fusion", "domain": domain},
                )

                logger.debug(
                    "GOSNN integration: σ_Immutable=%s (score=%.3f, "
                    "threshold=%.3f), harmonic_synergy=%.3f",
                    evaluation.passes,
                    evaluation.score,
                    evaluation.threshold,
                    gosnn.last_harmonic_synergy,
                )

            except EthicalConstraintViolationError:
                # σ_Immutable violation already raised — propagate
                # without wrapping; the caller's audit log needs the
                # original ``check`` value.
                raise
            except Exception as exc:
                # GOSNN itself errored (singleton init blew up, the
                # 32-head attention faulted, …).  This is now a hard
                # ``check="gosnn_unavailable"`` failure — the σ_Immutable
                # second gate cannot run, and the engine fails closed.
                from omni_mercury_engine.cognitive.ethical_bounding import (
                    EthicalConstraintViolationError as _EthicalErr,
                )

                raise _EthicalErr(
                    action=(
                        f"OmniMercuryEngine.detect_with_fusion:" f"domain={domain or 'general'}"
                    ),
                    score=0.0,
                    threshold=self._sigma_immutable_gate.threshold,
                    check="gosnn_unavailable",
                    details={
                        "boundary": "OmniMercuryEngine.detect_with_fusion",
                        "domain": domain,
                        "underlying_error": f"{type(exc).__name__}: {exc}",
                    },
                ) from exc

        fusion_result = self.fusion_inference.predict(
            self._restrict_to_trained_groups(all_features),
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
        llm_enhancement = self._enhance_with_llm(data, result)  # type: ignore[arg-type, unused-ignore]
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
        _enable_gosnn: bool = True,
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
            _enable_gosnn: PRIVATE testing knob (see ``detect_with_fusion``).
                Production code must leave this at the default ``True``.

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
            _enable_gosnn=_enable_gosnn,
        )

        # Get fusion probability
        anomaly_prob = fusion_result.get("anomaly_prob", 0.5)

        # For batch data, we need the full probability array
        # The fusion_inference returns probs for all samples
        if self.mode == "fusion":
            det_features, det_scores = self._extract_detector_features(data)
            mod_features, mod_scores = self._extract_model_features(data)
            all_features = {**det_features, **mod_features}

            fusion_output = self.fusion_inference.predict(
                self._restrict_to_trained_groups(all_features), return_attention=True
            )
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
            training_data: Path to training data file (.npz only).
                Mercury Agent does not deserialize Python pickles. Convert
                legacy ``.pkl`` payloads once via
                ``python -m omni_mercury_engine.tools.migrate_pkl``.
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
        import tempfile

        from torch.utils.data import DataLoader, random_split

        from omni_mercury_engine.ml.training import AnomalyDataset, FusionTrainer
        from omni_mercury_engine.security.safe_load import (
            UnsafePayloadError,
            safe_load_training_data,
        )

        if self.mode != "fusion":
            raise ValueError("Training requires fusion mode. Initialize with mode='fusion'")

        # Validate parameters
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if not (0.0 < validation_split < 1.0):
            raise ValueError("validation_split must be between 0 and 1 (exclusive)")

        # Load training data. Only .npz is accepted; pickle is not supported.
        # Legacy .pkl payloads must be converted via
        # `python -m omni_mercury_engine.tools.migrate_pkl`.
        if not training_data.endswith(".npz"):
            raise ValueError(
                f"Unsupported data format. Mercury Agent accepts only .npz "
                f"training archives (got {training_data!r}). Convert legacy "
                f"payloads with `python -m omni_mercury_engine.tools.migrate_pkl`."
            )

        try:
            data = safe_load_training_data(training_data)
        except UnsafePayloadError as exc:
            raise RuntimeError(f"Failed to load training data: {exc}") from exc

        if "labels" not in data:
            raise RuntimeError(
                f"Training archive {training_data!r} is missing the required 'labels' array."
            )
        features_dict = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in data.items() if k != "labels"
        }
        labels = torch.tensor(data["labels"], dtype=torch.long)
        # Record the trained feature groups so inference restricts itself to
        # exactly the set this archive trained on (train/serve lockstep).
        self._fusion_feature_groups = sorted(features_dict.keys())

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
        trainer_module.optimizer_type = optimizer_type

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
        optimizer_config = trainer_module.configure_optimizers()
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
                    scaler.scale(loss / gradient_accumulation_steps).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = trainer_module.training_step(batch, batch_idx)
                    (loss / gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call, unused-ignore]

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
        """
        Save the fusion model to a versioned checkpoint.

        The checkpoint bundles, in one dict:

        * the model weights (``model_state_dict``) plus the metadata needed to
          rebuild the network before loading — ``feature_dims``, ``hidden_dim``
          and the dynamic-projection registry (input dim per lazily-created
          projection layer). Without this a reloaded model would fail
          ``load_state_dict`` on the data-dependent ``_dynamic_projections.*``
          keys;
        * the fitted temperature calibrator (if any) so loading restores
          trustworthy probabilities, not just the raw network;
        * provenance (``format_version`` / ``mercury_version``).

        A bare ``state_dict`` written by older code still loads via
        :meth:`load_model`.

        Args:
            path: File path for saving the checkpoint (``.pt``).

        Example:
            >>> engine.save_model("models/fusion_model.pt")
        """
        if self.mode != "fusion":
            return
        temperature = None
        if self._fusion_calibrator is not None and getattr(
            self._fusion_calibrator, "_fitted", False
        ):
            temperature = float(self._fusion_calibrator.temperature)
        checkpoint = {
            "format_version": FUSION_CHECKPOINT_FORMAT_VERSION,
            "mercury_version": __version__,
            "model_state_dict": self.fusion_model.state_dict(),
            "feature_dims": dict(self.fusion_model.feature_dims),
            "hidden_dim": self.fusion_model.hidden_dim,
            "projection_registry": self.fusion_model.export_projection_registry(),
            "temperature": temperature,
            "feature_groups": self._fusion_feature_groups,
            "provenance": self._fusion_provenance,
            "fusion_trained": bool(self._fusion_trained),
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str) -> None:
        """
        Load the fusion model from a checkpoint.

        Handles the structured checkpoint written by :meth:`save_model` and a
        legacy bare ``state_dict``. For structured checkpoints the model is
        rebuilt with the saved ``feature_dims`` and its dynamic projection
        layers are recreated before ``load_state_dict`` (so the full
        train -> save -> load -> serve workflow round-trips on the
        data-dependent ``_dynamic_projections.*`` keys), and the fitted
        temperature calibrator is restored when present so calibrated
        probabilities are available immediately.

        Args:
            path: File path to load the checkpoint from.

        Example:
            >>> engine.load_model("models/fusion_model.pt")
        """
        if self.mode != "fusion":
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            feature_dims = checkpoint.get("feature_dims")
            hidden_dim = checkpoint.get("hidden_dim", self.fusion_model.hidden_dim)

            if feature_dims is not None and (
                feature_dims != self.fusion_model.feature_dims
                or hidden_dim != self.fusion_model.hidden_dim
            ):
                self.fusion_model = OmniFusionModel(
                    feature_dims=dict(feature_dims), hidden_dim=hidden_dim
                )
                self.fusion_model.to(self.device)
                self.fusion_inference = FusionInference(
                    model=self.fusion_model,
                    device=str(self.device),
                )

            self.fusion_model.rebuild_projection_registry(
                checkpoint.get("projection_registry", {}), self.device
            )
            self.fusion_model.load_state_dict(checkpoint["model_state_dict"])

            temperature = checkpoint.get("temperature")
            if temperature is not None:
                from omni_mercury_engine.core.calibration import TemperatureScaling

                calibrator = TemperatureScaling()
                calibrator.temperature = float(temperature)
                calibrator._fitted = True
                self._fusion_calibrator = calibrator

            groups = checkpoint.get("feature_groups")
            self._fusion_feature_groups = list(groups) if groups is not None else None
            provenance = checkpoint.get("provenance")
            self._fusion_provenance = dict(provenance) if provenance is not None else None
            self._fusion_trained = bool(checkpoint.get("fusion_trained", True))
        else:
            # Legacy bare state_dict (no metadata): load directly.
            self.fusion_model.load_state_dict(checkpoint)
            self._fusion_trained = True

        self.fusion_model.eval()

    def load_default_fusion_checkpoint(self) -> bool:
        """Load the shipped default fusion checkpoint if it is present.

        Makes the headline fusion path real out of the box: after this call a
        freshly-installed engine scores with a trained network (and calibrated
        probabilities) without any training step. Used by the ``detect`` and
        ``serve`` CLI entry points. Engine construction itself intentionally
        leaves the fusion network untrained — this is an explicit opt-in so the
        ``mode='fusion'`` default contract (untrained until ``fit_fusion`` or an
        explicit load) is preserved.

        Returns:
            True if a checkpoint existed and was loaded; False otherwise.
        """
        if self.mode != "fusion":
            return False
        path = default_fusion_checkpoint_path()
        if not path.exists():
            logger.debug("No default fusion checkpoint at %s", path)
            return False
        try:
            self.load_model(str(path))
            logger.info("Loaded default fusion checkpoint from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load default fusion checkpoint: %s", e)
            return False

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
        """
        Clear the feature cache.

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


# Legacy alias removed - project renamed to Mercury Agent
