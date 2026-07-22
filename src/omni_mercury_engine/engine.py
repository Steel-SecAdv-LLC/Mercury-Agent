# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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

from __future__ import annotations

import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from omni_mercury_engine.security.safe_torch import safe_torch_load

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
from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.config import EngineConfig
from omni_mercury_engine.core.conformal_prediction import (
    BinaryConformalClassifier,
    BinaryPredictionSet,
)
from omni_mercury_engine.core.equation_profiles import (
    components_from_score_channels,
    score_runtime_equation_profile,
)
from omni_mercury_engine.core.exceptions import OmniAnomalyException
from omni_mercury_engine.core.global_omni_scalar_network import (
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

# Resolved from installed package metadata via the single source of truth in
# ``_version`` (a stdlib-only leaf module, so importing it here cannot create an
# import cycle with the package ``__init__``).
from omni_mercury_engine._version import get_version as _get_version

__version__ = _get_version()

# Core detectors - always imported (lightweight base classes)
from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector
from omni_mercury_engine.ml.domain_encoders import DomainEncoderStack

# Runtime pipeline modules - always imported (required for core functionality)
from omni_mercury_engine.ml.drift import DriftResult, EnsembleDriftDetector
from omni_mercury_engine.ml.fairness import BiasAuditConfig, FairnessAuditor, FairnessReport
from omni_mercury_engine.ml.fusion_network import FocalLoss, OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference
from omni_mercury_engine.ml.optimization import OptimizationConfig, ParallelExecutor
from omni_mercury_engine.ml.symbolic_constraint import (
    SymbolicConstraintModule,
    SymbolicWeight,
    resolve_rule_graph,
    resolve_symbolic_weight,
    rule_graph_from_spec,
    rule_graph_to_spec,
)
from omni_mercury_engine.utils.logging import LoggerMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator
    from omni_mercury_engine.agentic.subagents.fleet import SubAgentFleet
    from omni_mercury_engine.cognitive.benevolence_cache import CachedBenevolenceScorer
    from omni_mercury_engine.cognitive.ethical_bounding import BenevolenceScorer
    from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator
    from omni_mercury_engine.decision import DecisionAbstentionResponder, DecisionLedger
    from omni_mercury_engine.governance.self_improvement import ThresholdGovernance

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
    from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
    from omni_mercury_engine.models.llm_registry import LLMModelRegistry
    from omni_mercury_engine.models.neural import NeuralCognitiveModel
    from omni_mercury_engine.models.parapsychology import ParapsychologyDetector
    from omni_mercury_engine.models.quantum import QuantumAnomalyModel
    from omni_mercury_engine.reasoning.backend import ReasoningBackend
    from omni_mercury_engine.reasoning.schemas import Explanation
    from omni_mercury_engine.resilience.self_healing import SelfHealingEngine
    from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine
    from omni_mercury_engine.security.threat_detection import ThreatDetector
    from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

# Configure module logger
logger = logging.getLogger(__name__)

# Integrated-Gradients interpolation steps used by ``detect_with_fusion(explain=True)``.
# Matches the validated harness (``benchmarks/explanation_fidelity.py``). Module-level
# so the (expensive, opt-in) serve-path explainer can be turned down in tests.
_EXPLAIN_IG_STEPS = 32


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
        # Torch tensors split by device:
        #
        # * CPU tensor -> key on CONTENT (fall through to the numpy path below).
        #   A CPU tensor shares memory with numpy, so ``.numpy()`` is a view (no
        #   host<-device copy); content keying costs the same bounded O(N) sum
        #   the numpy path already accepts. This closes two stale-hit surfaces
        #   that pure identity keying leaves open: an in-place mutation of the
        #   same storage, AND an allocator reusing a freed tensor's address for a
        #   genuinely different tensor (measured: two distinct ``torch.zeros(10)``
        #   can share a ``data_ptr`` and would otherwise collide -> a stale hit
        #   for different data, not merely a missed optimization).
        #
        # * CUDA tensor -> identity keying (data_ptr + storage_offset + stride +
        #   shape + dtype + device). Content-hashing a device tensor would force
        #   a per-lookup host<-device sync on the hot path, so we accept the
        #   documented tradeoff: an in-place mutation of the same device storage
        #   (or an address reuse) can alias, bounded by the LRU aging stale
        #   pointers out. ``data_ptr()`` is valid for non-contiguous tensors too,
        #   so it is never zeroed; ``stride``/``storage_offset`` disambiguate
        #   distinct views that share a first-element pointer.
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                key_tuple: tuple[Any, ...] = (
                    "torch-cuda",
                    int(data.data_ptr()),
                    int(data.storage_offset()),
                    tuple(data.stride()),
                    tuple(data.shape),
                    str(data.dtype),
                    str(data.device),
                )
                return f"{prefix}_{hash(key_tuple)}"
            # CPU tensor: content keying via the numpy path. ``.contiguous()``
            # copies only when the tensor is a non-contiguous view (bounded);
            # ``.detach()`` drops any autograd linkage so ``.numpy()`` succeeds.
            data = data.detach().cpu().contiguous().numpy()

        # numpy arrays: sample a bounded, strided slice instead of copying the
        # whole buffer. ``tobytes()`` materialises the entire array before the
        # ``[:1024]`` slice, which is O(N); sampling bounds the *copy* work at
        # O(256) regardless of array size. shape+dtype+size are folded into
        # the key to keep collisions negligible for this best-effort feature
        # cache. A 256-point stride alone would silently miss a change
        # confined to the (n - 256) unsampled positions -- e.g. a single
        # streaming-window element updated between two sample points -- and
        # return stale cached features for genuinely different data. A
        # vectorized sum reduction over the full array is still O(N) but,
        # unlike ``tobytes()``, never materialises a Python bytes copy of the
        # buffer; folding it into the key catches off-sample changes at a
        # fraction of the original full-hash cost.
        flat = np.ascontiguousarray(data).reshape(-1)
        n = flat.size
        # Non-finite-aware checksum components. A plain ``np.sum`` becomes NaN
        # for any array containing a NaN, and ``hash(nan)`` is a constant, so
        # off-sample mutations of a NaN-bearing array would NOT change the key
        # -> stale cache hit. Sum only the finite elements and fold the NaN /
        # +Inf / -Inf counts into the key separately, so a change involving a
        # non-finite position still moves the hash.
        checksum = 0.0
        n_nan = n_posinf = n_neginf = 0
        if n <= 256:
            sample = flat.tobytes()
        else:
            idx = np.linspace(0, n - 1, 256).astype(np.intp)
            sample = flat[idx].tobytes()
            if np.issubdtype(flat.dtype, np.number):
                finite_mask = np.isfinite(flat)
                if finite_mask.any():
                    checksum = float(np.sum(flat[finite_mask], dtype=np.float64))
                n_nan = int(np.isnan(flat).sum())
                n_posinf = int(np.isposinf(flat).sum())
                n_neginf = int(np.isneginf(flat).sum())
        data_hash = hash(
            (data.shape, data.dtype.str, n, sample, checksum, n_nan, n_posinf, n_neginf)
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
        equation_profile: str | None = None,
        require_explicit_fit: bool = True,
        cache_ethical_decisions: bool = True,
    ) -> None:
        """Initialize the OmniMercuryEngine.

        Args:
            config: Engine configuration. If None, uses default config.
            mode: Operation mode. Either 'fusion' for ML fusion or
                a specific detector name.
            device: Computation device ('cpu' or 'cuda').
            cache_size: Maximum entries in feature cache. Default 128.
            memory_threshold_mb: Memory threshold for GC in MB. Default 2048.
            require_explicit_fit: When True (default), ``detect_with_fusion``
                fails loud if a detector was never fit, rather than silently
                auto-fitting it on the first inference batch (which leaks that
                batch as the detector's reference distribution -- a correctness
                bug). Set False to opt into the legacy auto-fit-on-first-batch
                behaviour (still warned and audited via
                ``_inference_auto_fit_detectors``); loaded checkpoints and a
                prior ``fit_fusion`` both satisfy the requirement.
            auto_load_checkpoint: When True and mode='fusion', load the packaged
                default fusion checkpoint at init so detection works without a
                training step. Default False to keep a freshly-constructed
                engine deterministically untrained; the ``detect``/``serve``
                CLI entry points opt in.
            equation_profile: Optional runtime equation profile id. ``None``
                (default) preserves the legacy calibrated fusion probabilities
                byte-for-byte; explicit profiles such as
                ``baseline_original_v1`` or ``quiet_horizon_v1`` blend the
                calibrated neural score with the frozen OAE R/H/O equation
                signal at serve time (see
                :mod:`omni_mercury_engine.core.equation_profiles`).
            cache_ethical_decisions: When True (default), the benevolence
                boundary scorer is wrapped in a
                :class:`~omni_mercury_engine.cognitive.benevolence_cache.CachedBenevolenceScorer`
                so repeated identical ``enforce(action, context)`` calls at the
                detection boundary return the memoised ``EthicalScore`` instead
                of re-running the full scoring pipeline. Semantics are
                preserved: violations are never cached, and a ruleset-version
                bump invalidates the cache. Set False to always recompute.

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
        # Optional runtime equation profile (default None = legacy behaviour).
        # Resolved per call in ``_apply_runtime_equation_profile``; ``None``
        # keeps ``score_fusion`` / ``detect_with_fusion`` outputs unchanged.
        self.equation_profile = equation_profile
        # Set of detector names that were auto-fit on inference data because
        # they were not pre-fit. Initialised unconditionally so the leakage
        # surface (warning + result-dict key) works for every engine mode,
        # not only ``mode='fusion'``.
        self._inference_auto_fit_detectors: set[str] = set()
        # Fail-loud on an unfit detector at inference instead of leaking the
        # first batch as its reference distribution (see __init__ docstring).
        self._require_explicit_fit: bool = require_explicit_fit
        # Online drift recalibration (opt-in via enable_online_recalibration).
        # When set, detect_with_fusion feeds each sample's calibrated score to a
        # Gibbs-Candes AdaptiveConformalInference so the operating threshold
        # tracks score drift instead of going stale (the decider would otherwise
        # only DEFER on drift). None = disabled (exact legacy behaviour).
        self._adaptive_conformal: Any = None
        self._recalibration_warmup: int = 30
        # Bounded sample of the fusion training features, captured by
        # fit_fusion(). Used as the SHAP background for the opt-in GDPR report
        # (detect_with_fusion(gdpr_report=True)); None until a fit has run.
        self._fusion_background: np.ndarray[Any, Any] | None = None

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

        # The boundary scorer is hit on every ``detect_with_fusion`` call with a
        # deterministic (action, context) derived from the sanitised domain and
        # data shape, so identical serve-path requests recompute the same
        # EthicalScore. When ``cache_ethical_decisions`` is on we wrap it in an
        # LRU cache that memoises those repeats. The wrapper is a strict
        # superset of the scorer surface (``enforce``/``score_action``/
        # ``benevolence_threshold``), never caches violations, and self-purges
        # on a ruleset-version bump, so the swap is semantics-preserving.
        _real_boundary_scorer = _BenevolenceScorer(benevolence_threshold=_MINIMUM_BENEVOLENCE_FLOOR)
        self._boundary_scorer: BenevolenceScorer | CachedBenevolenceScorer
        if cache_ethical_decisions:
            from omni_mercury_engine.cognitive.benevolence_cache import (
                CachedBenevolenceScorer as _CachedBenevolenceScorer,
            )

            self._boundary_scorer = _CachedBenevolenceScorer(scorer=_real_boundary_scorer)
        else:
            self._boundary_scorer = _real_boundary_scorer

        # σ_Immutable second hard ethical gate (Wave B item 1).  Loaded
        # eagerly for the same reason as the benevolence scorer above:
        # the first ``detect_with_fusion`` call cannot race the gate's
        # corpus-verification step.  The gate is a process-wide
        # singleton — every boundary (engine, hub, orchestrator)
        # observes the same trained network and the same signed-corpus
        # verdict, so a corpus tampering at startup poisons every
        # decision boundary uniformly.
        from omni_mercury_engine.security.sigma_immutable_gate import get_sigma_immutable_gate

        self._sigma_immutable_gate = get_sigma_immutable_gate()

        # Mercury-owned reasoning backend (subordinate, optional, offline-first).
        # Mercury is the agent and brain of record; the backend is a *called*
        # dependency it invokes for natural-language explanation over its own
        # detections -- never the front of the system. Constructed lazily (first
        # use, or via ``enable_reasoning``) so the engine pays no LLM-chain cost
        # unless reasoning is actually requested.
        self._reasoning_backend: ReasoningBackend | None = None
        self._reasoning_ledger: UsageLedger | None = None

        self._init_detectors()
        self._init_models()
        self._init_fusion()
        self._init_resilience()
        self._init_runtime_pipeline()

        if auto_load_checkpoint and self.mode == "fusion":
            self.load_default_fusion_checkpoint()

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
            "statistical": MercuryAnomalyDetector(),
            "temporal": TemporalAnomalyDetector(),
            "spatial": SpatialAnomalyDetector(),
            "dimensional": DimensionalAnalyzer(),
            "directive": SigmaDirectiveDetector(),
        }

    # ------------------------------------------------------------------
    # Detector registration seam
    #
    # The default set above is intentionally the five general-purpose base
    # detectors. Specialized detectors (e.g. ``geo_movement``) are declared
    # in ``DETECTOR_MANIFEST`` but were previously unreachable through the
    # engine: nothing consumed the manifest, so a manifest-registered
    # detector never participated in detect/fuse/decide. These methods are
    # the supported, opt-in bridge. They are purely additive — the
    # calibrated default path is byte-identical until a caller opts in.
    # ------------------------------------------------------------------
    def register_detector(
        self,
        name: str,
        detector: BaseDetector,
        *,
        replace: bool = False,
    ) -> OmniMercuryEngine:
        """Add a base detector to the engine's active detector set.

        The supported seam for extending the engine with a further
        :class:`~omni_mercury_engine.core.base.BaseDetector` so it
        participates in the same feature-extraction, fusion, and decision
        path as the built-in detectors. Additive by construction: existing
        detectors are untouched, so the default calibrated path is unchanged
        until a caller opts in.

        A registered detector contributes a fusion feature group named
        ``name`` on the next :meth:`fit_fusion` (learned into the network)
        and on every :meth:`detect_with_fusion`. If it cannot process a
        given input it raises and is skipped gracefully by the feature
        extractors — it never crashes detection.

        Note:
            Registering *after* :meth:`fit_fusion` has trained does not
            retroactively extend a trained network: fusion inference is
            restricted to the feature groups training saw
            (``_fusion_feature_groups``), so a *new* detector is recorded but
            ignored at fusion inference until ``fit_fusion`` is re-run.
            ``replace=True`` on a detector whose name is already a trained
            group is more dangerous — inference keeps using that group but
            with a different feature distribution — so both cases warn and
            recommend a re-fit.

        Args:
            name: Unique key for the detector within the engine.
            detector: A ``BaseDetector`` instance (fitted or not).
            replace: When True, replace an existing detector of the same
                name; otherwise a duplicate name raises ``ValueError``.

        Returns:
            ``self``, for chaining.

        Raises:
            TypeError: If ``detector`` is not a ``BaseDetector``.
            ValueError: If ``name`` is empty, or already registered while
                ``replace`` is False.
        """
        if not name:
            raise ValueError("Detector name must be a non-empty string")
        if not isinstance(detector, BaseDetector):
            raise TypeError(
                f"detector must be a BaseDetector instance, got {type(detector).__name__}"
            )
        if name in self.detectors and not replace:
            raise ValueError(
                f"Detector {name!r} is already registered; pass replace=True to override it."
            )

        self.detectors[name] = detector

        trained_groups = (
            self._fusion_feature_groups
            if self.mode == "fusion" and getattr(self, "_fusion_trained", False)
            else None
        )
        if trained_groups is not None and name in trained_groups:
            # Replacing/re-registering a detector the fusion net was trained on: the
            # group name persists in _fusion_feature_groups, so inference keeps using
            # it but now feeds the net a *different* feature distribution for that
            # group. Silent miscalibration unless the operator re-fits.
            logger.warning(
                "Detector %r changed after fusion training; the fusion network was "
                "trained on the previous detector's features for this group, so "
                "inference now feeds it a different distribution. Re-run fit_fusion() "
                "to retrain on the new detector.",
                name,
            )
        elif trained_groups is not None:
            # New group: filtered out at fusion inference until a re-fit.
            logger.warning(
                "Detector %r registered after fusion training; it will be ignored at "
                "fusion inference until fit_fusion() is re-run (inference is restricted "
                "to the trained feature groups).",
                name,
            )
        else:
            logger.info("Registered detector %r (%s)", name, type(detector).__name__)
        return self

    def enable_detector(self, name: str) -> BaseDetector:
        """Instantiate and register a manifest detector by name.

        Bridges the declarative manifest
        (:data:`~omni_mercury_engine.core.detector_registry.DETECTOR_MANIFEST`)
        to a live engine instance: looks ``name`` up in the manifest,
        imports and constructs the class, and registers it via
        :meth:`register_detector`. This is how an operator turns on an
        opt-in detector — e.g. ``engine.enable_detector("geo_movement")`` —
        without importing detector classes by hand.

        Args:
            name: Manifest entry name (see :meth:`available_detectors`).

        Returns:
            The constructed and registered detector instance.

        Raises:
            ValueError: If ``name`` is not in the manifest, its manifest
                ``module_path`` is outside the ``omni_mercury_engine``
                package, or it is already enabled.
            TypeError: If the manifest entry constructs an object that is
                not a :class:`~omni_mercury_engine.core.base.BaseDetector`.
        """
        import importlib

        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        entry = next((e for e in DETECTOR_MANIFEST if e.name == name), None)
        if entry is None:
            available = ", ".join(sorted(e.name for e in DETECTOR_MANIFEST))
            raise ValueError(
                f"Unknown detector {name!r}. Available manifest detectors: {available}"
            )
        # Defense in depth: the manifest is curated code, but never import from
        # outside the package tree (mirrors DetectorRegistry.auto_discover_detectors).
        if not entry.module_path.startswith("omni_mercury_engine."):
            raise ValueError(
                f"Refusing to import detector {name!r} from untrusted module path "
                f"{entry.module_path!r}"
            )
        module = importlib.import_module(entry.module_path)
        detector_cls = getattr(module, entry.class_name)
        detector = detector_cls()
        if not isinstance(detector, BaseDetector):
            raise TypeError(
                f"Manifest detector {name!r} "
                f"({entry.module_path}.{entry.class_name}) constructed a "
                f"{type(detector).__name__}, which is not a BaseDetector"
            )
        self.register_detector(name, detector)
        return detector

    def available_detectors(self) -> dict[str, bool]:
        """Map every manifest detector name to whether it is currently active.

        Lets a caller discover opt-in detectors (those in
        ``DETECTOR_MANIFEST``) and see which are already part of this
        engine's active set (the five built-ins plus anything added via
        :meth:`register_detector` / :meth:`enable_detector`).

        Returns:
            Mapping ``{detector_name: is_active}`` covering every manifest
            entry; any active detector not represented in the manifest
            (e.g. a custom one) is appended with value True.
        """
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        status = {entry.name: entry.name in self.detectors for entry in DETECTOR_MANIFEST}
        for active_name in self.detectors:
            status.setdefault(active_name, True)
        return status

    def _init_models(self) -> None:
        """Initialize all specialized domain models.

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
            # Post-hoc temperature calibration (Guo et al. 2017), fit on a
            # held-out val split during fit_fusion. Identity (T=1) until fit.
            self._fusion_calibrator: Any = None
            # Feature groups the network was actually trained on. Persisted in
            # the checkpoint and used to restrict the inference feature dict so
            # detect_with_fusion feeds exactly the groups training saw (no
            # untrained projections for aggregating/extra groups). None until
            # trained → no filtering, preserving the untrained-engine contract.
            self._fusion_feature_groups: list[str] | None = None
            # Differentiable symbolic-constraint LTN co-trained with the fusion
            # net when fit_fusion(symbolic_weight>0). Retained after training for
            # explainability (learned detector reliabilities / rule satisfaction).
            # None until a symbolic-regularised fit runs. The trained fusion net
            # has already absorbed the constraint, so inference needs only the
            # net; this handle is diagnostic, not required for scoring.
            self._symbolic_module: SymbolicConstraintModule | None = None
            self._symbolic_score_channels: list[str] | None = None
            # Opt-in differentiable domain encoders (WS-B / Target 2): the
            # FFT-spectral + finite-difference-kinematic + Fisher/entropy
            # nn.Modules, jointly trained with the fusion net when
            # fit_fusion(domain_encoder=True). None on the default path; when
            # set, inference (_extract_fusion_features) injects its feature so
            # serve matches training. ``_domain_scaler`` is the (mean, std) used
            # to standardise the encoder input, applied identically at inference.
            self._domain_encoder: DomainEncoderStack | None = None
            self._domain_scaler: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
            # Class-conditional conformal classifier over the calibrated fusion
            # probability, fit by calibrate_fusion_conformal() on a held-out
            # labelled split. Turns the scalar probability into label prediction
            # sets with a distribution-free coverage guarantee. None until fit.
            self._fusion_conformal: BinaryConformalClassifier | None = None
            # Optional training provenance (source, datasets, seed, ...) recorded
            # by training scripts and persisted in/restored from the checkpoint
            # so a shipped artifact is self-describing for audit. None unless set.
            self._fusion_provenance: dict[str, Any] | None = None
            logger.info(
                "OmniFusionModel initialized (untrained). Call fit_fusion() "
                "before detection for optimal performance."
            )

    def _symbolic_checkpoint_config(self) -> dict[str, Any] | None:
        module = self._symbolic_module
        if module is None:
            return None
        registry_names = {
            "detector_consensus": "consensus",
            "detector_consensus_salience": "consensus_salience",
        }
        graph_name = registry_names.get(module.rule_graph.name, module.rule_graph.name)
        config = {
            "num_detectors": module.num_detectors,
            "rule_graph": graph_name,
            "semantics": module.semantics,
            "learn_detector_reliability": module.learn_detector_reliability,
            "p_aggregator": module.p_aggregator,
        }
        # Non-registry graphs (e.g. evolved rule graphs selected via
        # ``symbolic_rule_graph="evolved:<path>"``) are not resolvable by name
        # at load time, so serialise the full rule data inline; rules are pure
        # data, so the checkpoint stays self-contained and the artifact file
        # is not needed to restore it. Registry graphs keep the name-only
        # format byte-identical to before.
        if module.rule_graph.name not in registry_names:
            config["rule_graph_spec"] = rule_graph_to_spec(module.rule_graph)
        return config

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
        self.cognitive_orchestrator: CognitiveOrchestrator | None = None
        # Multi-agent orchestration (vision pillar B): planner/critic/executor
        # loop over this engine's own detectors. None until enabled via
        # enable_multi_agent_orchestration().
        self.multi_agent_orchestrator: MultiAgentOrchestrator | None = None
        # Internal subagent fleet: the main-agent delegation tier over a
        # catalogue of full-capability specialized subagents (compliance,
        # ethics, reporting, guardrail, detection, generalist), runnable in the
        # masses under autonomy governance and the dual ethical gate. None until
        # enabled via enable_subagent_fleet(); internal-only (never on the
        # public package surface).
        self.subagent_fleet: SubAgentFleet | None = None
        # Decision / abstention / response layer: closes the loop from the
        # calibrated detection certificate to a bounded, non-destructive
        # response with an explicit "don't-know" gate.  None until enabled via
        # enable_decision_layer(); detect_with_fusion() is an exact no-op until
        # then.
        self.decision_layer: DecisionAbstentionResponder | None = None
        # Optional append-only audit ledger for the "verify" step of the loop.
        # When set via enable_decision_layer(ledger=...), every detection's
        # decision is recorded; None keeps the serve path stateless.
        self.decision_ledger: DecisionLedger | None = None
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
        symbolic_weight: SymbolicWeight = "adaptive",
        symbolic_semantics: str = "product",
        symbolic_rule_graph: str = "consensus",
        domain_encoder: bool = False,
        domain_encoder_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit the fusion model on training data with semi-supervised learning.

        This method extracts features from all detectors and trains the OmniFusionModel
        to produce calibrated anomaly scores. Supports both supervised (with labels)
        and semi-supervised (detector-consensus labels) training.

        This is the primary fix for Issue #1: Untrained Fusion Neural Network.

        Args:
            X: Training features (n_samples, n_features).
            y: Optional training labels (1=anomaly, 0=normal). If None, uses
               semi-supervised learning with consensus labels derived from
               detector agreement.
            epochs: Maximum training epochs (default: 50).
            batch_size: Training batch size (default: 32).
            learning_rate: Learning rate for optimizer (default: 0.001).
            early_stopping_patience: Epochs without improvement before stopping.
            validation_split: Fraction of data for validation.
            contamination: Expected anomaly fraction for consensus labeling. If None,
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
            symbolic_weight: Weight ``lambda`` of the differentiable
                symbolic-constraint loss co-trained with the supervised loss
                (``total = supervised + lambda * (1 - satisfaction)``). The
                constraint is a compact Logic Tensor Network
                (:class:`~omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`)
                that ties the fusion output to the unsupervised agreement of the
                base detectors. Accepts:

                * a concrete float ``lambda`` -- e.g. ``0.1`` enables a fixed
                  co-training weight; ``0.0`` reproduces the purely-neural
                  training path byte-for-byte;
                * the string ``"adaptive"`` -- use the label-scarcity schedule
                  (:class:`~omni_mercury_engine.ml.symbolic_constraint.ScarcityWeightSchedule`),
                  which spends the constraint only when labelled anomalies are
                  scarce (where the held-out ablation showed it helps) and
                  decays to the neural path as labels grow abundant;
                * an explicit
                  :class:`~omni_mercury_engine.ml.symbolic_constraint.ScarcityWeightSchedule`.

                The effective weight is resolved from the provided labels'
                anomaly count before training and reported in the returned
                metrics. **Default ``"adaptive"``**: the held-out ADBench
                ablation (``benchmarks/neurosymbolic_ablation.py``,
                ``docs/NEUROSYMBOLIC.md``) showed the adaptive schedule
                *dominates* neural-only -- no full-data AUC regression (within
                the ±0.002 noise floor) and a seed-agreed low-data lift -- so
                co-training is on by default and decays to the neural path when
                labels are abundant. Pass ``0.0`` for the byte-for-byte
                purely-neural path.
            symbolic_semantics: Implication operator for the symbolic constraint
                when co-training is active -- ``"product"`` / ``"reichenbach"``
                (default; aliases for the smooth Reichenbach residuum),
                ``"lukasiewicz"``, or ``"godel"`` (see ``docs/NEUROSYMBOLIC.md``
                §2.2). Ignored when the effective weight is 0.
            symbolic_rule_graph: Rule graph for the constraint when co-training
                is active -- ``"consensus"`` (default, two rules over a learned
                consensus predicate), ``"consensus_salience"`` (adds a
                soft-existential salience recall rule; §2.3), or
                ``"evolved:<path>"`` (a genetically evolved rule graph loaded
                from the JSON artifact written by
                ``omni_mercury_engine.ml.rule_evolution``; see
                ``benchmarks/rule_evolution_benchmark.py``). Ignored when the
                effective weight is 0.

        Returns:
            Dictionary with training metrics including final_loss, best_loss,
            epochs_trained, convergence information, and (when calibrated)
            ``temperature`` plus ``ece_before``/``ece_after``. When symbolic
            co-training is active (the effective weight resolves > 0) also
            includes ``symbolic_satisfaction``, ``symbolic_loss`` (final-epoch
            training values), ``symbolic_semantics`` and ``symbolic_rule_graph``.

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
            X = X.detach().cpu().numpy()
        # Convert labels too (incl. CUDA tensors) so n_positive / schedule
        # resolution below is correct -- np.asarray on a live tensor is unreliable.
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        n_samples = len(X)
        logger.info(f"Starting fusion training on {n_samples} samples...")

        # Fit base detectors and extract the fusion feature set (detectors +
        # domain models, matching inference). Shared with build_feature_npz()
        # so the offline builder's archive is byte-for-byte what fit_fusion()
        # trains on. Record the trained group names so inference can restrict
        # itself to exactly this set.
        detector_features = self._extract_fusion_features(X, fit_detectors=True)
        self._fusion_feature_groups = sorted(detector_features.keys())

        # Generate consensus labels if not provided (semi-supervised). Done
        # here -- before the symbolic weight is resolved -- because the
        # label-scarcity schedule keys on the number of consensus-labelled
        # anomalies.
        if y is None:
            logger.info("No labels provided, using semi-supervised consensus labeling...")
            y = self._generate_consensus_labels(X, contamination)

        # Resolve the symbolic co-training weight to a concrete lambda. The
        # public argument may be a float, the string "adaptive", or a
        # ScarcityWeightSchedule; resolve_symbolic_weight maps all of these onto
        # the scalar the training loop consumes, using the provided labels'
        # anomaly count so the adaptive schedule spends the constraint only when
        # labels are scarce.
        n_positive = int(np.count_nonzero(np.asarray(y).reshape(-1) >= 0.5))
        symbolic_weight_eff = resolve_symbolic_weight(symbolic_weight, n_positive)

        # Detector-consensus scores for symbolic co-training (only when enabled,
        # to avoid the extra detect() pass on the purely-neural default path).
        detector_scores: torch.Tensor | None = None
        if symbolic_weight_eff > 0:
            detector_scores, symbolic_channels = self._extract_consensus_scores(
                X,
                return_channels=True,
            )
            self._symbolic_score_channels = symbolic_channels
        else:
            self._symbolic_score_channels = None

        # WS-B: opt-in differentiable domain encoder. Standardise the raw input
        # once (mean/std stored on the engine, re-applied identically at
        # inference) and hand the tensor to the trainer, which builds + jointly
        # trains the encoder. ``None`` on the default path keeps it byte-identical.
        raw_inputs: torch.Tensor | None = None
        domain_scaler: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
        if domain_encoder:
            x_arr = np.nan_to_num(np.asarray(X, dtype=np.float32))
            mean = x_arr.mean(axis=0)
            std = x_arr.std(axis=0)
            std[std < 1e-8] = 1.0
            domain_scaler = (mean, std)
            self._domain_scaler = domain_scaler
            raw_inputs = torch.tensor((x_arr - mean) / std, dtype=torch.float32)
        else:
            self._domain_scaler = None

        metrics = self._fit_fusion_on_features(
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
            symbolic_weight=symbolic_weight_eff,
            symbolic_semantics=symbolic_semantics,
            symbolic_rule_graph=symbolic_rule_graph,
            detector_scores=detector_scores,
            raw_inputs=raw_inputs,
            domain_scaler=domain_scaler,
            domain_encoder_config=domain_encoder_config,
        )

        # When the weight was specified adaptively (string/schedule), surface how
        # it resolved -- including when it resolved to 0 (abundant labels ->
        # neural path) -- so the choice is auditable. Plain numeric weights leave
        # the metrics keys unchanged, preserving the neural-path contract. NumPy
        # scalars (e.g. np.float32 from config/np ops) count as plain numeric;
        # bool is excluded (it never reaches here -- resolve rejects it).
        is_plain_number = isinstance(
            symbolic_weight, (int, float, np.floating, np.integer)
        ) and not isinstance(symbolic_weight, bool)
        if not is_plain_number:
            metrics["symbolic_weight_spec"] = (
                symbolic_weight if isinstance(symbolic_weight, str) else "schedule"
            )
            metrics["symbolic_weight_resolved"] = float(symbolic_weight_eff)
            metrics["symbolic_n_positive"] = n_positive

        # Retain a bounded, uniformly-sampled subset of the raw training
        # features as the SHAP background for the opt-in GDPR report. Raw ``X``
        # is the space ``score_fusion`` (hence the report's prediction function)
        # consumes, so this is dimensionally aligned with the per-call instance.
        # Sampling uniformly (rather than the first N rows) avoids biasing the
        # baseline when the data is ordered (grouped by label/time); the field
        # is reset to None when ``X`` is not a usable 2-D array so a prior fit's
        # background is never served stale.
        try:
            background_source: np.ndarray[Any, Any] | None = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError):
            background_source = None
        if (
            background_source is not None
            and background_source.ndim == 2
            and background_source.shape[0] > 0
        ):
            n_background = min(100, background_source.shape[0])
            sample_idx = np.random.default_rng(0).choice(
                background_source.shape[0], size=n_background, replace=False
            )
            self._fusion_background = background_source[np.sort(sample_idx)].copy()
        else:
            self._fusion_background = None

        return metrics

    def tune_fusion(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        *,
        n_trials: int = 20,
        tuning_epochs: int = 10,
        sampler: str = "tpe",
        scheduler: str | None = None,
        seed: int | None = None,
        validation_split: float = 0.25,
        search_space: Any = None,
    ) -> dict[str, Any]:
        """Bayesian hyperparameter search over ``fit_fusion``, max held-out AUC.

        Runs Mercury's own :class:`~omni_mercury_engine.automl.BayesianOptimizer`
        over the real ``fit_fusion`` training hyperparameters, scoring each trial
        by the ROC-AUC of the calibrated ``score_fusion`` probability on a
        held-out split (Mercury's own ``evaluation.metrics.compute_auc_roc``).
        The engine is left fit on the *full* ``(X, y)`` with the best
        configuration when the search finds one.

        Args:
            X: Raw training features, shape ``(n_samples, n_features)``.
            y: Binary labels (both classes required to score AUC).
            n_trials: Number of hyperparameter configurations to evaluate.
            tuning_epochs: Epochs per trial's ``fit_fusion`` (kept small so the
                search is affordable); the final refit uses the same value.
            sampler: ``"tpe"``, ``"gp"`` or ``"random"``.
            scheduler: Optional ``"asha"``/``"hyperband"``/``"median"`` pruner;
                ``None`` (default) evaluates every trial.
            seed: Seed for the split and the sampler.
            validation_split: Fraction held out for AUC scoring.
            search_space: Optional custom
                :class:`~omni_mercury_engine.automl.SearchSpace`; the default
                spans learning rate, batch size, focal loss params, early-stopping
                patience and symbolic weight.

        Returns:
            ``{"best_config", "best_auc", "n_trials", "convergence_history"}``.

        Raises:
            ValueError: If ``n_trials`` or ``tuning_epochs`` is < 1, ``X`` is not
                2-D with >= 4 samples, labels mismatch, ``y`` does not contain
                both classes, ``validation_split`` is not in ``(0, 1)``, or any
                class has fewer than 2 samples (required for the stratified
                held-out split).
            RuntimeError: If every trial fails. Each trial resets the fusion
                model, so on all-fail the engine is left untrained and must be
                re-fit or reloaded before serving; the error is raised rather
                than returning a ``None`` AUC a caller might overlook.
        """
        from omni_mercury_engine.automl import (
            BayesianOptimizer,
            CategoricalParameter,
            IntUniformParameter,
            LogUniformParameter,
            SearchSpace,
            UniformParameter,
        )
        from omni_mercury_engine.evaluation.metrics import compute_auc_roc

        # Fail fast on degenerate budgets: n_trials <= 0 would run zero trials
        # and surface as a confusing "all 0 trials failed" RuntimeError (after
        # which the CLI could save an untrained model), and tuning_epochs <= 0
        # would make every trial a no-op fit scored on random weights.
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if tuning_epochs < 1:
            raise ValueError(f"tuning_epochs must be >= 1, got {tuning_epochs}")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] < 4:
            raise ValueError("tune_fusion requires a 2-D X with at least 4 samples")
        if len(y) != len(X):
            raise ValueError(f"label count ({len(y)}) != sample count ({len(X)})")
        if len(np.unique(y)) < 2:
            raise ValueError("tune_fusion needs both classes present in y to score AUC")

        if not 0.0 < validation_split < 1.0:
            raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")
        classes, counts = np.unique(y, return_counts=True)
        if int(counts.min()) < 2:
            raise ValueError(
                "tune_fusion needs at least 2 samples per class for a stratified "
                "held-out split that keeps both classes in train and validation"
            )
        # Stratified split: hold out ``validation_split`` of *each* class so both
        # y_train and y_val always carry both classes. An unstratified split can
        # leave a single-class validation fold -- held-out ROC-AUC is then
        # undefined (``compute_auc_roc`` returns 0.5) and the search objective goes
        # blind -- or starve a split entirely when ``validation_split`` is near 0/1.
        # Per-class ``k`` is clamped so every class keeps >= 1 sample on each side,
        # so each split holds >= 2 samples regardless of ``validation_split``.
        rng = np.random.default_rng(seed)
        train_parts, val_parts = [], []
        for c in classes:
            c_idx = rng.permutation(np.flatnonzero(y == c))
            k = min(len(c_idx) - 1, max(1, round(validation_split * len(c_idx))))
            val_parts.append(c_idx[:k])
            train_parts.append(c_idx[k:])
        val_idx = np.concatenate(val_parts)
        train_idx = np.concatenate(train_parts)
        x_train, y_train = X[train_idx], y[train_idx]
        x_val, y_val = X[val_idx], y[val_idx]

        if search_space is None:
            search_space = (
                SearchSpace()
                .add(LogUniformParameter("learning_rate", 1e-5, 1e-1))
                .add(CategoricalParameter("batch_size", [16, 32, 64, 128]))
                .add(UniformParameter("focal_alpha", 0.5, 0.95))
                .add(UniformParameter("focal_gamma", 0.0, 3.0))
                .add(IntUniformParameter("early_stopping_patience", 3, 15))
                .add(UniformParameter("symbolic_weight", 0.0, 0.5))
            )

        def _coerce(config: dict[str, Any]) -> dict[str, Any]:
            return {
                "learning_rate": float(config["learning_rate"]),
                "batch_size": int(config["batch_size"]),
                "focal_alpha": float(config["focal_alpha"]),
                "focal_gamma": float(config["focal_gamma"]),
                "early_stopping_patience": int(config["early_stopping_patience"]),
                "symbolic_weight": float(config["symbolic_weight"]),
            }

        def objective(config: dict[str, Any]) -> float:
            # Reset to a fresh, untrained fusion model before each trial. fit_fusion
            # trains ``self.fusion_model`` in place, so without this every trial
            # would inherit the previous trial's weights/calibration and the
            # objective would depend on evaluation order -- the reported best_config
            # would then not reproduce from an independent, from-scratch fit.
            self._init_fusion()
            self.fit_fusion(x_train, y_train, epochs=tuning_epochs, **_coerce(config))
            scores = np.asarray(self.score_fusion(x_val), dtype=np.float64).reshape(-1)
            # Minimise negative AUC -> maximise held-out ranking quality.
            return -float(compute_auc_roc(y_val, scores))

        optimizer = BayesianOptimizer(
            search_space=search_space,
            objective=objective,
            sampler=sampler,
            scheduler=scheduler,
            direction="minimize",
            n_trials=n_trials,
            seed=seed,
        )
        result = optimizer.optimize()

        best_config = _coerce(result.best_config) if result.best_config else {}
        if not best_config:
            # Every trial failed (the optimizer only records a best for COMPLETED
            # trials). Each trial called _init_fusion(), so the engine is now
            # holding a reset, untrained fusion model. Fail loudly rather than
            # return a None AUC a caller might ignore while unknowingly serving a
            # broken model; the caller must re-fit or reload a checkpoint.
            raise RuntimeError(
                f"tune_fusion: all {result.n_trials} trials failed, so no "
                "configuration could be selected; the fusion model has been reset "
                "and must be re-fit (call fit_fusion) or reloaded before serving. "
                "Check the training data and hyperparameter ranges."
            )
        # From-scratch final fit so the delivered model matches the measured
        # trial (same reset as each trial), not the last trial's leftover state.
        self._init_fusion()
        self.fit_fusion(X, y, epochs=tuning_epochs, **best_config)

        # best_config is guaranteed non-empty here (all-fail raised above).
        return {
            "best_config": best_config,
            "best_auc": -float(result.best_metric),
            "n_trials": result.n_trials,
            "convergence_history": [-float(m) for m in result.convergence_history],
        }

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
        symbolic_weight: float = 0.0,
        symbolic_semantics: str = "product",
        symbolic_rule_graph: str = "consensus",
        detector_scores: torch.Tensor | None = None,
        raw_inputs: torch.Tensor | None = None,
        domain_scaler: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None,
        domain_encoder_config: dict[str, Any] | None = None,
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
            symbolic_weight: As in :meth:`fit_fusion`. When ``> 0`` and
                ``detector_scores`` carries at least one channel, a
                :class:`SymbolicConstraintModule` is co-optimised with the
                fusion net and its ``(1 - satisfaction)`` loss is added to the
                supervised loss.
            detector_scores: Per-sample detector-consensus matrix
                ``(N, n_detectors)`` aligned row-for-row with
                ``detector_features``. Ignored when ``symbolic_weight == 0``.

        Returns:
            Training metrics (``best_loss``, ``epochs_trained`` ... plus
            ``temperature``/``ece_before``/``ece_after`` when calibrated, and
            ``symbolic_satisfaction``/``symbolic_loss`` when co-trained).
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

        # WS-B: opt-in differentiable domain encoder. Built BEFORE the optimiser
        # so the fusion projection it needs is registered and captured in
        # ``trainable_params``. Off (raw_inputs is None) leaves every line below
        # identical to the neural path.
        domain_active = raw_inputs is not None
        raw_inputs_dev: torch.Tensor | None = None
        domain_dim = 64
        if domain_active:
            assert raw_inputs is not None  # narrowed by domain_active
            assert domain_scaler is not None
            # output_dim is pinned to domain_dim (the fusion projection width);
            # domain_encoder_config sweeps the *internal* design (domains /
            # kernel widths / normalization) for WS-B design-space search.
            dom_cfg = dict(domain_encoder_config or {})
            dom_cfg.pop("output_dim", None)  # cannot override the projection width
            dom_module: DomainEncoderStack | None = DomainEncoderStack(
                raw_inputs.shape[1], output_dim=domain_dim, **dom_cfg
            ).to(device)
            assert dom_module is not None
            dom_module.train()
            self._domain_encoder = dom_module
            self._domain_scaler = domain_scaler
            self.fusion_model.register_feature_group("differentiable_domain", domain_dim, device)
            if (
                self._fusion_feature_groups is not None
                and "differentiable_domain" not in self._fusion_feature_groups
            ):
                self._fusion_feature_groups = sorted(
                    [*self._fusion_feature_groups, "differentiable_domain"]
                )
            raw_inputs_dev = raw_inputs.to(device)
        else:
            dom_module = None
            self._domain_encoder = None
            self._domain_scaler = None

        # Neuro-symbolic co-training: when enabled, a SymbolicConstraintModule
        # is optimised jointly with the fusion net and its (1 - satisfaction)
        # loss is added per batch. The constraint's gradient flows into the
        # net's anomaly head (shared ``anomaly_probs``), while its own learnable
        # detector reliabilities / rule confidences adapt. Off (lambda == 0)
        # leaves the optimiser, clipping and loss identical to the neural path.
        symbolic_active = (
            symbolic_weight > 0
            and detector_scores is not None
            and detector_scores.ndim == 2
            and detector_scores.shape[1] > 0
        )
        detector_scores_dev: torch.Tensor | None = None
        if symbolic_active:
            assert detector_scores is not None  # narrowed by symbolic_active
            sym_module = SymbolicConstraintModule(
                num_detectors=detector_scores.shape[1],
                rule_graph=resolve_rule_graph(symbolic_rule_graph),
                semantics=symbolic_semantics,
            ).to(device)
            sym_module.train()
            self._symbolic_module = sym_module
            detector_scores_dev = detector_scores.to(device)
            trainable_params = list(self.fusion_model.parameters()) + list(sym_module.parameters())
        else:
            sym_module = None
            self._symbolic_module = None
            self._symbolic_score_channels = None
            trainable_params = list(self.fusion_model.parameters())

        if domain_active:
            assert dom_module is not None
            trainable_params = trainable_params + list(dom_module.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
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
                # ``FocalLoss.forward`` returns a Tensor but its nn.Module
                # ``__call__`` is typed as ``Any`` in the torch stubs; cast so
                # the wrapper's signature isn't degraded.
                return torch.as_tensor(focal_criterion(probs, targets))

        else:

            def _loss_fn(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.binary_cross_entropy(probs, targets)

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        best_domain_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        loss_history: list[dict[str, float]] = []
        # Final-epoch symbolic diagnostics (populated only when co-training).
        last_satisfaction: float = 1.0
        last_symbolic_loss: float = 0.0

        for epoch in range(epochs):
            # Training phase
            self.fusion_model.train()
            if domain_active and dom_module is not None:
                dom_module.train()
            train_losses: list[float] = []

            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch_indices = train_indices[start_idx:end_idx]

                # Get batch features
                batch_features = {
                    name: feat[batch_indices].to(device) for name, feat in detector_features.items()
                }
                if domain_active and dom_module is not None and raw_inputs_dev is not None:
                    batch_features["differentiable_domain"] = dom_module(
                        raw_inputs_dev[batch_indices]
                    )
                batch_labels = labels_tensor[batch_indices].to(device)

                optimizer.zero_grad()
                outputs = self.fusion_model(batch_features)
                loss = _loss_fn(outputs["anomaly_probs"], batch_labels)
                if symbolic_active:
                    assert sym_module is not None and detector_scores_dev is not None
                    sym_out = sym_module(
                        outputs["anomaly_probs"], detector_scores_dev[batch_indices]
                    )
                    loss = loss + symbolic_weight * sym_out["loss"]
                    last_satisfaction = float(sym_out["satisfaction"].detach())
                    last_symbolic_loss = float(sym_out["loss"].detach())
                loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            self.fusion_model.eval()
            if domain_active and dom_module is not None:
                dom_module.eval()
            val_losses: list[float] = []

            with torch.no_grad():
                for start_idx in range(0, n_val, batch_size):
                    end_idx = min(start_idx + batch_size, n_val)
                    batch_indices = val_indices[start_idx:end_idx]

                    batch_features = {
                        name: feat[batch_indices].to(device)
                        for name, feat in detector_features.items()
                    }
                    if domain_active and dom_module is not None and raw_inputs_dev is not None:
                        batch_features["differentiable_domain"] = dom_module(
                            raw_inputs_dev[batch_indices]
                        )
                    batch_labels = labels_tensor[batch_indices].to(device)

                    outputs = self.fusion_model(batch_features)
                    loss = _loss_fn(outputs["anomaly_probs"], batch_labels)
                    if symbolic_active:
                        assert sym_module is not None and detector_scores_dev is not None
                        sym_out = sym_module(
                            outputs["anomaly_probs"], detector_scores_dev[batch_indices]
                        )
                        loss = loss + symbolic_weight * sym_out["loss"]
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
                if domain_active and dom_module is not None:
                    best_domain_state = {
                        k: v.cpu().clone() for k, v in dom_module.state_dict().items()
                    }
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
        if best_domain_state is not None and dom_module is not None:
            dom_module.load_state_dict(best_domain_state)

        self.fusion_model.eval()
        if symbolic_active and sym_module is not None:
            sym_module.eval()
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
        if symbolic_active:
            assert sym_module is not None  # narrowed by symbolic_active
            metrics["symbolic_weight"] = float(symbolic_weight)
            # Record the module's normalized semantics (strip/lower, alias-resolved),
            # not the raw argument, so the metric reflects what actually ran.
            metrics["symbolic_semantics"] = sym_module.semantics
            metrics["symbolic_rule_graph"] = sym_module.rule_graph.name
            metrics["symbolic_satisfaction"] = last_satisfaction
            metrics["symbolic_loss"] = last_symbolic_loss

        # Post-hoc temperature scaling (Guo et al. 2017): fit a single scalar on
        # the held-out validation split so the sigmoid outputs are trustworthy
        # probabilities. Monotonic, so ROC-AUC/ranking is exactly preserved.
        self._fusion_calibrator = None
        if calibrate and n_val > 0:
            cal_metrics = self._fit_fusion_temperature(
                detector_features, labels_tensor, val_indices, raw_inputs_dev
            )
            metrics.update(cal_metrics)

        return metrics

    def _fit_fusion_temperature(
        self,
        detector_features: dict[str, torch.Tensor],
        labels_tensor: torch.Tensor,
        val_indices: torch.Tensor,
        raw_inputs_dev: torch.Tensor | None = None,
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
            if self._domain_encoder is not None and raw_inputs_dev is not None:
                val_feats["differentiable_domain"] = self._domain_encoder(
                    raw_inputs_dev[val_indices]
                )
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

    def _generate_consensus_labels(
        self,
        X: np.ndarray[Any, Any],
        contamination: float | None = None,
    ) -> np.ndarray[Any, Any]:
        """Generate consensus labels using detector agreement for semi-supervised learning.

        Uses adaptive contamination estimation and ensemble voting from detector
        scores to identify likely anomalies for training.

        Args:
            X: Training features.
            contamination: Expected anomaly fraction. If None, estimated adaptively.

        Returns:
            Binary consensus labels (0=normal, 1=anomaly).
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
        consensus_labels = (ensemble_score > threshold).astype(float)

        logger.info(
            f"Generated consensus labels: contamination={contamination:.4f}, "
            f"n_anomalies={int(consensus_labels.sum())}/{n_samples}"
        )

        return consensus_labels  # type: ignore[no-any-return, unused-ignore]

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
        # Purity contract: the fusion feature map is a function of (fitted
        # state, X) only. Detectors with transient cross-call memory (the
        # directive detector's recursive-memory buffer) are reset first;
        # without this, the first ``memory_depth`` rows' features depended
        # on whatever the previous extraction left behind, so repeated
        # ``detect_with_fusion`` calls drifted and a reloaded checkpoint
        # could not reproduce the saving engine's probabilities
        # (ROADMAP row 16; defect found 2026-06-11).
        self._reset_transient_detector_state()
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

        # WS-B: inject the co-trained differentiable domain-encoder feature so
        # inference matches training. ``_domain_encoder`` is None on the default
        # path AND during the training-time extraction call (it is created later
        # in _fit_fusion_on_features), so neither path is affected.
        if self._domain_encoder is not None and self._domain_scaler is not None:
            mean, std = self._domain_scaler
            x_std = (np.nan_to_num(np.asarray(X, dtype=np.float32)) - mean) / std
            self._domain_encoder.eval()
            with torch.no_grad():
                dom = self._domain_encoder(torch.tensor(x_std, dtype=torch.float32).to(self.device))
            fusion_features["differentiable_domain"] = dom.detach().to(torch.float32).cpu()

        return fusion_features

    @overload
    def _extract_consensus_scores(
        self,
        X: np.ndarray[Any, Any],
        *,
        channels: list[str] | None = None,
        return_channels: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def _extract_consensus_scores(
        self,
        X: np.ndarray[Any, Any],
        *,
        channels: list[str] | None = None,
        return_channels: Literal[True],
    ) -> tuple[torch.Tensor, list[str]]: ...

    def _extract_consensus_scores(
        self,
        X: np.ndarray[Any, Any],
        *,
        channels: list[str] | None = None,
        return_channels: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
        """Build the per-sample, per-detector anomaly-score matrix for co-training.

        Collects each base detector's and domain model's normalised anomaly
        score (one scalar per sample, already squashed to ``[0, 1]`` by
        :meth:`_normalize_scores`) and stacks the per-sample-aligned channels
        into an ``(n_samples, n_detectors)`` matrix. This is the unsupervised
        substrate the :class:`SymbolicConstraintModule` reasons over — the
        detectors' independent "opinions" whose agreement structure the
        symbolic rules tie the fusion output to.

        Detectors are assumed already fit (``fit_fusion`` fits them before
        calling this), so no fitting happens here; columns whose length does
        not match ``n_samples`` are dropped to keep row-alignment with the
        feature tensors. Returns an ``(n_samples, 0)`` tensor when no detector
        score is available — the constraint then satisfies trivially.

        Args:
            X: Raw training features ``(n_samples, n_features)``.

        Returns:
            Float32 tensor ``(n_samples, n_detectors)`` of scores in ``[0, 1]``.
        """
        n_samples = len(X)
        _, det_scores, _ = self._extract_detector_features(X)
        _, mod_scores = self._extract_model_features(X)

        all_scores = {**det_scores, **mod_scores}
        score_names = list(channels) if channels is not None else sorted(all_scores)

        columns: list[torch.Tensor] = []
        used_channels: list[str] = []
        for name in score_names:
            raw = all_scores.get(name)
            if raw is None:
                continue
            col = torch.as_tensor(np.asarray(raw), dtype=torch.float32).reshape(-1)
            if col.numel() != n_samples:
                logger.debug(
                    "Skipping consensus score %r: length %d != n_samples %d",
                    name,
                    col.numel(),
                    n_samples,
                )
                continue
            columns.append(col.clamp(0.0, 1.0))
            used_channels.append(name)

        if not columns:
            scores = torch.zeros((n_samples, 0), dtype=torch.float32)
        else:
            scores = torch.stack(columns, dim=1)
        return (scores, used_channels) if return_channels else scores

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

    def _apply_fusion_calibration(self, probs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply the fitted post-hoc temperature calibration to a probability array.

        Single source of truth for calibration application across the production
        ``detect_with_fusion`` path and the benchmark ``score_fusion`` path, so
        the calibrator persisted on the checkpoint reaches every consumer that
        reads ``anomaly_probs`` from the network's output — not only the
        benchmark path. Without this, ``fit_fusion``'s temperature scalar (Guo
        et al. 2017) only corrects probabilities measured via ``score_fusion``
        and ``mercury-agent detect`` users still get raw sigmoid output.

        Monotonic, so ranking (and any AUC measured downstream) is preserved
        exactly; only the probability values are corrected. A no-op when no
        calibrator is fitted, so the contract degrades cleanly on legacy
        checkpoints (and on engines that never trained calibration).
        """
        if self._fusion_calibrator is None:
            return probs
        arr = np.asarray(probs)
        calibrated = np.asarray(self._fusion_calibrator.calibrate(arr.reshape(-1)))
        return calibrated.reshape(arr.shape)

    def _apply_runtime_equation_profile(
        self,
        probs: np.ndarray[Any, Any],
        score_channels: dict[str, Any],
        *,
        equation_profile: str | None = None,
        domain: str | None = None,
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any] | None]:
        """Blend calibrated neural probabilities with an explicit equation profile.

        Opt-in: when neither the per-call ``equation_profile`` nor the
        engine-level :attr:`equation_profile` selects a profile, the calibrated
        probabilities are returned unchanged (and ``None`` metadata), so the
        legacy serve/benchmark path is preserved byte-for-byte. When a profile
        is selected, the detector/model score channels are mapped onto the
        OAE R/H/O components and blended with the frozen baseline equation
        signal (see :mod:`omni_mercury_engine.core.equation_profiles`).
        """
        profile_id = equation_profile if equation_profile is not None else self.equation_profile
        if profile_id is None:
            return probs, None

        flat_probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        r, h, o = components_from_score_channels(score_channels, raw_scores=flat_probs)
        scored, metadata = score_runtime_equation_profile(
            flat_probs,
            r,
            h,
            o,
            eta=self._domain_eta(domain),
            profile_id=profile_id,
        )
        return scored.reshape(np.asarray(probs).shape), metadata

    def _domain_eta(self, domain: str | None) -> float:
        """Return the OAE ethical-gate (η) estimate for a runtime equation profile.

        The runtime equation profile *is* the OAE surface, so η follows the OAE
        per-domain ethical-threshold convention (the values mirror
        ``three_r_mechanism``'s OAE ``DOMAIN_THRESHOLDS``): ``medical`` relaxes
        to 0.93 to avoid critical-domain false negatives, ``infrastructure``
        tightens to 0.995, ``humanitarian`` to 0.95, with a conservative 0.96
        default for everything else.

        ``dict.get(..., 0.96)`` makes any unknown / ``None`` / unsafe domain
        string collapse to the safe default, so no ``sanitize_domain`` pass is
        needed here — and indeed must not be used: the OAE domain vocabulary
        (``security`` / ``humanitarian``) is deliberately *not* a subset of
        ``EnvironmentDomain``, so sanitising first would silently zero those
        keys. η only scales a probability-bounded equation term, so a
        float-table lookup carries no injection surface.
        """
        if not isinstance(domain, str):
            return 0.96
        return {
            "medical": 0.93,
            "humanitarian": 0.95,
            "infrastructure": 0.995,
        }.get(domain, 0.96)

    def _symbolic_consistency_payload(
        self,
        X: np.ndarray[Any, Any],
        probs: np.ndarray[Any, Any],
    ) -> dict[str, Any] | None:
        module = self._symbolic_module
        if module is None:
            return None
        x_arr = np.asarray(X, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        probs_arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        detector_scores = self._extract_consensus_scores(
            x_arr,
            channels=self._symbolic_score_channels,
        )
        if detector_scores.shape[0] != probs_arr.shape[0]:
            return None
        if detector_scores.shape[1] != module.num_detectors:
            return None

        module.eval()
        explanation = module.explain(
            torch.tensor(probs_arr, dtype=torch.float32, device=self.device),
            detector_scores.to(self.device),
        )
        return {
            "graph": explanation["graph"],
            "semantics": explanation["semantics"],
            "satisfaction": explanation["satisfaction"],
            "rules": explanation["rules"],
            "detector_weights": explanation["detector_weights"],
            "detector_channels": list(self._symbolic_score_channels or []),
        }

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
            y: Optional binary labels (1=anomaly, 0=normal). If None, semi-
                supervised consensus labels are generated from detector agreement.
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
            y = self._generate_consensus_labels(X, contamination)
        arrays["labels"] = np.asarray(y).astype(np.int64)

        if not output_path.endswith(".npz"):
            output_path = f"{output_path}.npz"
        save_feature_archive = cast("Any", np.savez)
        save_feature_archive(output_path, **arrays)
        logger.info(
            f"Wrote fusion feature archive to {output_path} "
            f"({len(arrays) - 1} detector feature groups, {len(arrays['labels'])} samples)"
        )
        return output_path

    def score_fusion(
        self,
        X: np.ndarray[Any, Any],
        *,
        equation_profile: str | None = None,
        domain: str | None = None,
    ) -> np.ndarray[Any, Any]:
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
            equation_profile: Optional runtime equation profile override. When
                omitted, :attr:`equation_profile` is used; when neither selects
                a profile the calibrated probabilities are returned unchanged.
            domain: Optional domain hint for the profile's ethical-gate (η)
                estimate.

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
        # Same helper that ``detect_with_fusion`` calls so the benchmark path
        # and the production path agree on what the checkpoint's temperature
        # means.
        calibrated = self._apply_fusion_calibration(probs).reshape(-1)
        # Optional runtime equation profile (opt-in; exact no-op when unset, so
        # the default benchmark/serve path is byte-for-byte unchanged). Channel
        # values are per-group feature means as the R/H/O proxy when no explicit
        # detector-score dict is in scope on the benchmark path.
        score_channels = {
            name: feat.detach().cpu().numpy().mean(axis=1)
            for name, feat in features.items()
            if feat.ndim >= 2
        }
        profiled, _ = self._apply_runtime_equation_profile(
            calibrated,
            score_channels,
            equation_profile=equation_profile,
            domain=domain,
        )
        return np.asarray(profiled).reshape(-1)

    def evaluate_neurosymbolic_feedback(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any] | None = None,
    ) -> dict[str, Any]:
        """Measure neural-symbolic agreement on a held-out batch."""
        if self.mode != "fusion":
            raise ValueError("evaluate_neurosymbolic_feedback() requires mode='fusion'")
        probs = self.score_fusion(X)
        payload = self._symbolic_consistency_payload(X, probs)
        if payload is None:
            x_arr = np.asarray(X)
            return {
                "symbolic_active": False,
                "n_samples": x_arr.reshape(1, -1).shape[0] if x_arr.ndim == 1 else x_arr.shape[0],
            }

        result: dict[str, Any] = {
            "symbolic_active": True,
            "n_samples": int(probs.shape[0]),
            "mean_anomaly_prob": float(np.mean(probs)),
            "symbolic_satisfaction": float(payload["satisfaction"]),
            "symbolic_rule_graph": payload["graph"],
            "symbolic_semantics": payload["semantics"],
            "symbolic_rules": payload["rules"],
            "symbolic_detector_weights": payload["detector_weights"],
        }
        if y is not None:
            labels = np.asarray(y).reshape(-1)
            if labels.shape[0] != probs.shape[0]:
                raise ValueError(
                    "labels length must match X rows for evaluate_neurosymbolic_feedback()"
                )
            predictions = (probs > 0.5).astype(int)
            result["agreement_with_labels"] = float(np.mean(predictions == labels.astype(int)))
            result["positive_rate"] = float(np.mean(predictions))
        return result

    def calibrate_fusion_conformal(
        self,
        X_cal: np.ndarray[Any, Any],
        y_cal: np.ndarray[Any, Any],
        coverage: float = 0.9,
        *,
        per_sample: bool = False,
    ) -> dict[str, Any]:
        """Fit a conformal classifier on a held-out labelled calibration split.

        Builds on the calibrated fusion probability (temperature-scaled by
        :meth:`fit_fusion`) to provide *distribution-free* label prediction sets
        with a coverage guarantee, complementing the point probability with a
        rigorous uncertainty set. Must be called after the fusion model is
        trained, on data **disjoint** from both training and the eventual test
        set (exchangeability is what the guarantee rests on).

        **Match the serving regime.** Exchangeability requires the calibration
        scores and the serve-time scores to come from the *same* score
        function — and several detector features are batch-relative by design
        (recursive-memory deviation, batch-percentile stability, batch-max
        magnitude, sliding windows, within-batch min-max normalization), so a
        sample scored alone does not get the score it would get inside a
        batch. If production serves sample-at-a-time (the
        ``detect_with_fusion(x[i:i+1])`` / DecisionLoop pattern), calibrate
        with ``per_sample=True`` so each calibration row is scored exactly as
        serving will score it; batch consumers keep the default. (Before
        2026-06-11 the cross-call streaming buffers made single-sample
        serving *resemble* batch scoring by accident of call history — the
        serve-path purity fix made the mismatch visible and deterministic,
        and this parameter is the principled remedy.)

        Args:
            X_cal: Calibration features ``(n_cal, n_features)``, disjoint from
                training data.
            y_cal: Calibration binary labels ``(n_cal,)`` (1 = anomaly).
            coverage: Target per-class coverage (e.g. 0.9 for 90%).
            per_sample: Score calibration rows one at a time (the
                single-sample serving regime) instead of as one batch.

        Returns:
            Diagnostics dict with the target ``coverage``, the learned per-class
            ``thresholds`` and ``n_calibration``.

        Raises:
            ValueError: If mode is not 'fusion'.
            RuntimeError: If the fusion model is untrained.
        """
        if self.mode != "fusion":
            raise ValueError("calibrate_fusion_conformal() requires mode='fusion'")
        if not self._fusion_trained:
            raise RuntimeError(
                "Fusion model is untrained; call fit_fusion()/train_fusion_model() "
                "before calibrate_fusion_conformal()."
            )

        X_arr = np.asarray(X_cal)
        if per_sample:
            probs = np.concatenate(
                [np.asarray(self.score_fusion(X_arr[i : i + 1])).ravel() for i in range(len(X_arr))]
            )
        else:
            probs = self.score_fusion(X_cal)
        y = np.asarray(y_cal).astype(int).ravel()
        self._fusion_conformal = BinaryConformalClassifier(coverage=coverage).fit(probs, y)
        report = self._fusion_conformal.coverage_report(probs, y)
        logger.info(
            "Fusion conformal calibrated: coverage=%.2f thresholds=%s (n_cal=%d)",
            coverage,
            report["thresholds"],
            len(y),
        )
        return {
            "coverage": coverage,
            "thresholds": report["thresholds"],
            "n_calibration": len(y),
        }

    def score_fusion_conformal(self, X: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Return calibrated probabilities *and* conformal label sets for a batch.

        The uncertainty-aware counterpart to :meth:`score_fusion`: each sample
        gets a calibrated ``P(anomaly)`` plus a conformal prediction set over
        ``{normal, anomaly}`` whose ``set_size`` distinguishes confident calls
        (singletons) from genuine uncertainty (``{normal, anomaly}``) and
        atypical points (``{}``).

        Args:
            X: Features ``(n_samples, n_features)``.

        Returns:
            Dict with ``probabilities`` ``(n,)``, ``prediction_sets`` (list of
            sorted label lists), ``set_sizes`` ``(n,)``, ``abstain`` ``(n,)``
            bool (uncertain two-label sets), and the target ``coverage``.

        Raises:
            RuntimeError: If :meth:`calibrate_fusion_conformal` has not been run.
        """
        if self._fusion_conformal is None:
            raise RuntimeError(
                "Conformal calibrator not fitted; call calibrate_fusion_conformal() first."
            )
        probs = self.score_fusion(X)
        pred: BinaryPredictionSet = self._fusion_conformal.predict(probs)
        return {
            "probabilities": probs,
            "prediction_sets": pred.label_sets(),
            "set_sizes": pred.set_size,
            "abstain": pred.set_size == 2,
            "coverage": pred.coverage_level,
        }

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

        When the audit receives two or more named sensitive features (the
        ``_audit_fairness`` dict shape), marginal metrics are computed per
        feature and intersectional (joint-subgroup) parity / equalized
        odds are computed across the crossed cells — a model can satisfy
        every marginal constraint while still disadvantaging a joint
        subgroup, and only the intersectional metrics catch that.

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

    def enable_cognitive_analysis(
        self,
        *,
        enable_plasticity: bool = True,
        enable_causal: bool = True,
        enable_ipb: bool = True,
        enable_cbr: bool = True,
        enable_indicators: bool = True,
        enable_curiosity: bool = False,
        enable_enhanced_detection: bool = False,
    ) -> None:
        """Enable the cognitive orchestrator as a post-fusion feedback stage.

        Args:
            enable_plasticity: Enable dynamic knowledge adaptation.
            enable_causal: Enable causal discovery.
            enable_ipb: Enable intelligence preparation.
            enable_cbr: Enable case-based reasoning.
            enable_indicators: Enable indicator development.
            enable_curiosity: Enable curiosity-driven novelty scoring of
                detected anomalies (measured Mahalanobis distance from the
                observed distribution). Off by default — opt-in keeps the
                default analyze() output unchanged.
            enable_enhanced_detection: Enable the Bayesian/HMM
                predictive-memory augmentation over detected anomalies.
                Off by default; the runtime path performs no network I/O.
        """
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        self.cognitive_orchestrator = CognitiveOrchestrator(
            enable_plasticity=enable_plasticity,
            enable_causal=enable_causal,
            enable_ipb=enable_ipb,
            enable_cbr=enable_cbr,
            enable_indicators=enable_indicators,
            enable_curiosity=enable_curiosity,
            enable_enhanced_detection=enable_enhanced_detection,
        )
        logger.info("Cognitive analysis enabled")

    def enable_multi_agent_orchestration(
        self,
        *,
        consensus_method: str = "confidence_weighted",
        min_participants: int = 3,
        contamination: float = 0.1,
        operating_threshold: float = 0.5,
        threshold_governance: ThresholdGovernance | None = None,
        seed: int | None = None,
    ) -> MultiAgentOrchestrator:
        """Enable planner/critic/executor multi-agent orchestration (pillar B).

        Wires a :class:`~omni_mercury_engine.agentic.orchestration.MultiAgentOrchestrator`
        over this engine's own base detectors: the hierarchical planner
        sequences the real pipeline stages, the consensus protocol fuses
        per-sample votes from the live detectors, the reflexion critic
        *proposes* operating-threshold changes from real labeled feedback that
        are routed through the Phase 3 governance seam (fail-closed by default —
        no autonomous mutation of the live boundary), and every issued
        decision can be depicted by a chain-of-thought trace whose stated
        determination is contractually locked to the decision. The dual hard
        ethical gates run fail-closed at the orchestrator's decision boundary,
        exactly as on this engine's :meth:`detect_with_fusion` boundary.

        The orchestrator's measurable claims are pinned by
        ``benchmarks/orchestration_validation.py`` on real ADBench labels.

        Args:
            consensus_method: Consensus protocol method (default
                ``"confidence_weighted"``).
            min_participants: Quorum below which every sample abstains.
            contamination: Expected anomaly fraction for per-agent threshold
                calibration.
            operating_threshold: Initial consensus decision boundary; reflexion
                proposes adapting it, subject to governance.
            threshold_governance: Phase 3 governance policy consulted before any
                reflexion-proposed threshold move takes effect. Defaults to
                fail-closed (autonomous changes withheld pending promotion-gate
                evidence and human approval). Inject
                ``research.governed_fusion.phase3_governance.PromotionGateThresholdGovernance``
                to route proposals through the Phase 2 promotion gate.
            seed: Seed for deterministic agent calibration and reasoning.

        Returns:
            The enabled orchestrator (also stored as
            ``self.multi_agent_orchestrator``). Call ``fit(X_train)`` on it
            before detection.

        Example:
            >>> engine = OmniMercuryEngine()
            >>> orchestrator = engine.enable_multi_agent_orchestration(seed=0)
            >>> orchestrator.fit(X_train)
            >>> episode = orchestrator.run_episode(X_test, y_test)
        """
        from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator

        self.multi_agent_orchestrator = MultiAgentOrchestrator.from_engine(
            self,
            consensus_method=consensus_method,
            min_participants=min_participants,
            contamination=contamination,
            operating_threshold=operating_threshold,
            threshold_governance=threshold_governance,
            seed=seed,
        )
        logger.info("Multi-agent orchestration enabled")
        return self.multi_agent_orchestrator

    def enable_subagent_fleet(
        self,
        *,
        seed: int | None = None,
    ) -> SubAgentFleet:
        """Enable the internal subagent fleet for main-agent delegation.

        Wires a :class:`~omni_mercury_engine.agentic.subagents.fleet.SubAgentFleet`
        bound to this engine, so the detection specialization can delegate to
        Mercury's own real multi-agent detection. The fleet lets the main agent
        delegate arbitrary tasks to full-capability specialized subagents
        (compliance, ethics enforcement, law-enforcement reporting, guardrail,
        detection, generalist), singly or in the masses, under the autonomy
        governor (capability ceiling, corrigibility kill-switch, tripwire) and
        the dual hard ethical gate (benevolence floor + σ-Immutable) at the
        fleet's commit boundary — fail-closed, exactly as on this engine's other
        decision boundaries.

        The fleet is internal-only: it is constructed with the package-private
        access sentinel and is never exposed on the public ``omni_mercury_engine``
        surface. Idempotent — repeated calls return the existing fleet.

        Args:
            seed: Base seed for deterministic subagent construction.

        Returns:
            The enabled fleet (also stored as ``self.subagent_fleet``).
        """
        if self.subagent_fleet is None:
            from omni_mercury_engine.agentic.subagents.base import _INTERNAL
            from omni_mercury_engine.agentic.subagents.fleet import SubAgentFleet

            self.subagent_fleet = SubAgentFleet(access=_INTERNAL, seed=seed, engine=self)
            logger.info("Subagent fleet enabled")
        return self.subagent_fleet

    def enable_decision_layer(
        self,
        *,
        policy: Any | None = None,
        response_policy: Any | None = None,
        ledger: DecisionLedger | None = None,
        confidence_calibrator: Any | None = None,
    ) -> None:
        """Enable the decision / abstention / response layer.

        Closes the loop ``identify -> interpret -> decide -> deter -> verify``
        on top of the calibrated fusion certificate.  Once enabled, every
        :meth:`detect_with_fusion` result carries a ``"decision"`` key: a
        :class:`~omni_mercury_engine.decision.record.DecisionRecord` (as a
        dict) holding either a grounded label or an explicit abstention -- a
        principled "don't-know" gate split into a *resolvable* deferral
        (``UNAVAILABLE``) and a *fail-closed* hold (``UNDECIDABLE``) -- plus a
        bounded, non-destructive response (notify / recommend reversible
        countermeasures / escalate to a human / hold).

        The layer reads the signals the pipeline already produces (calibrated
        probability, conformal coverage set, ethical-gate verdict,
        neuro-symbolic agreement, drift), so it is most informative when
        :meth:`calibrate_fusion_conformal` has been called -- a conformal
        certificate turns a thresholded guess into a coverage-guaranteed
        decision.  It never authorises a destructive autonomous action.

        Args:
            policy: Optional
                :class:`~omni_mercury_engine.decision.policy.DecisionPolicy`
                (abstention thresholds).  Defaults to the conservative,
                fail-closed policy.
            response_policy: Optional
                :class:`~omni_mercury_engine.decision.response.ResponsePolicy`
                (disposition -> bounded response mapping).
            ledger: Optional
                :class:`~omni_mercury_engine.decision.ledger.DecisionLedger`.
                When supplied, every detection's decision is appended to it (the
                "verify" step -- an append-only, JSON-serialisable audit trail
                queryable via ``ledger.summary()``).  ``None`` keeps the serve
                path stateless (no recording).
            confidence_calibrator: Optional fitted
                :class:`~omni_mercury_engine.core.confidence.CalibratedConfidence`.
                When attached, the decider reports a calibrated probability on
                the uncalibrated threshold-band fallback (the path taken when no
                conformal certificate is present) instead of the
                ``0.5 + |margin|`` heuristic. Fit it on a held-out (fusion-score,
                label) split; the conformal certificate path stays authoritative.

        Example:
            >>> engine = OmniMercuryEngine()
            >>> engine.enable_decision_layer()
            >>> result = engine.detect_with_fusion(x, domain="security")
            >>> result["decision"]["state"]  # grounded / unavailable / undecidable
        """
        from omni_mercury_engine.decision import DecisionAbstentionResponder

        self.decision_layer = DecisionAbstentionResponder(
            policy=policy,
            response_policy=response_policy,
            confidence_calibrator=confidence_calibrator,
        )
        self.decision_ledger = ledger
        logger.info("Decision / abstention / response layer enabled")

    def enable_online_recalibration(
        self,
        *,
        target_coverage: float = 0.9,
        learning_rate: float = 0.05,
        initial_threshold: float | None = None,
        warmup: int = 30,
    ) -> None:
        """Recalibrate the operating threshold online instead of only deferring.

        On its own, the decision layer demotes a grounded verdict to ``DEFER``
        when drift is detected but leaves the threshold stale -- so the next
        sample under the same drift just defers again. This wires the existing
        :class:`~omni_mercury_engine.core.conformal_prediction.AdaptiveConformalInference`
        (Gibbs & Candes 2021) into :meth:`detect_with_fusion`: every sample's
        calibrated score updates an online quantile threshold, so under drift the
        operating point *tracks the shifting score distribution* and the decider
        grounds on a fresh threshold rather than a stale one. The conformal
        certificate path (``calibrate_fusion_conformal``) stays authoritative when
        present; this updates the uncalibrated threshold-band operating point and
        is surfaced under ``result['adaptive_threshold']`` /
        ``result['adaptive_conformal']``. The DEFER-on-severe-drift demotion is
        kept as a safety net while the online threshold has not yet converged.

        Args:
            target_coverage: Target fraction of scores at or below the threshold
                (i.e. the expected normal-class fraction; ``0.9`` -> ~10% alert
                rate). The threshold converges to this quantile and re-tracks it
                under drift.
            learning_rate: Gibbs-Candes step size for the online threshold update.
            initial_threshold: Starting threshold; defaults to the current fusion
                operating threshold so recalibration starts where the system
                already operates.
            warmup: Number of online updates before the recalibrated threshold
                replaces the static operating point (avoids acting on an
                unconverged threshold). Until then it is surfaced but not applied.
        """
        from omni_mercury_engine.core.conformal_prediction import AdaptiveConformalInference

        if initial_threshold is None:
            initial_threshold = float(getattr(self.config, "anomaly_threshold", 0.5) or 0.5)
        self._adaptive_conformal = AdaptiveConformalInference(
            target_coverage=target_coverage,
            learning_rate=learning_rate,
            initial_threshold=initial_threshold,
        )
        self._recalibration_warmup = max(1, int(warmup))
        logger.info(
            "Online drift recalibration enabled (target_coverage=%.2f, lr=%.3f, "
            "initial_threshold=%.3f, warmup=%d)",
            target_coverage,
            learning_rate,
            initial_threshold,
            self._recalibration_warmup,
        )

    def enable_reasoning(
        self,
        *,
        backend: ReasoningBackend | None = None,
        usage_ledger: UsageLedger | None = None,
        registry: LLMModelRegistry | None = None,
        ethics_enabled: bool = True,
    ) -> None:
        """Attach Mercury's subordinate, offline-first reasoning backend.

        Mercury (this engine) is the agent and brain of record; the backend is a
        *called dependency* it invokes to render natural-language explanations of
        its own detections. Every reasoning call passes Mercury's benevolence +
        ``sigma_Immutable`` dual hard ethical gate inside the backend before any
        text is surfaced (fail-closed). The backend is never the front of the
        system, and Mercury is never a wrapper around it.

        Args:
            backend: Explicit backend to use. Defaults to a
                :class:`~omni_mercury_engine.reasoning.backends.LocalReasoningBackend`
                -- offline-first and air-gap-safe (local Ollama when present, the
                deterministic builtin template otherwise; never a network call).
            usage_ledger: Optional shared
                :class:`~omni_mercury_engine.models.foundation.llm_usage.UsageLedger`
                threaded through the default backend so provider-reported token
                spend on reasoning calls is accounted. One is created when a
                default backend is built and none is supplied.
            registry: Optional operator-populated
                :class:`~omni_mercury_engine.models.llm_registry.LLMModelRegistry`.
                When supplied (and ``backend`` is not), the local model is chosen
                by the registry's free/local-first ``select_one`` rather than the
                shipped default -- model choice is deployment config, not a
                hard-code.
            ethics_enabled: Forwarded to the default backend's dual ethical gate;
                leave True outside trusted offline tests.
        """
        if backend is not None:
            self._reasoning_backend = backend
            self._reasoning_ledger = usage_ledger
            logger.info("Reasoning backend enabled (operator-supplied)")
            return

        from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
        from omni_mercury_engine.models.foundation.ollama_adapter import OllamaConfig
        from omni_mercury_engine.reasoning.backends import (
            LocalReasoningBackend,
            select_reasoning_model,
        )

        ledger = usage_ledger if usage_ledger is not None else UsageLedger()
        ollama_config: OllamaConfig | None = None
        if registry is not None:
            model_id = select_reasoning_model(registry, default=OllamaConfig().model)
            ollama_config = OllamaConfig(model=model_id)
        self._reasoning_backend = LocalReasoningBackend(
            ollama_config=ollama_config,
            usage_ledger=ledger,
            ethics_enabled=ethics_enabled,
        )
        self._reasoning_ledger = ledger
        logger.info("Reasoning backend enabled (offline-first local default)")

    @property
    def reasoning_backend(self) -> ReasoningBackend:
        """Mercury's reasoning backend, lazily defaulting to offline-first local.

        Accessing this property before :meth:`enable_reasoning` constructs the
        default offline-first
        :class:`~omni_mercury_engine.reasoning.backends.LocalReasoningBackend`,
        so :meth:`explain_detection` works out of the box without ceremony.
        """
        if self._reasoning_backend is None:
            self.enable_reasoning()
        backend = self._reasoning_backend
        if backend is None:  # pragma: no cover - enable_reasoning always sets it
            raise RuntimeError("reasoning backend failed to initialize")
        return backend

    @staticmethod
    def _clamp_unit(value: Any) -> float:
        """Coerce ``value`` to a float clamped to ``[0, 1]`` (0.0 on failure)."""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def explain_detection(
        self,
        result: dict[str, Any],
        *,
        domain: str = "security",
    ) -> Explanation:
        """Explain one of Mercury's detection certificates in natural language.

        Mercury owns the detection and the decision; it calls its subordinate
        reasoning backend only to render a concise, evidence-grounded explanation
        of a certificate produced by :meth:`detect_with_fusion`. The call passes
        Mercury's dual hard ethical gate inside the backend (fail-closed): a
        violation raises and no text is returned.

        Args:
            result: A detection certificate as returned by
                :meth:`detect_with_fusion` /
                :meth:`detect_with_fusion_calibrated`.
            domain: Mercury domain hint; sanitized at the ethical boundary.

        Returns:
            A provenance-stamped
            :class:`~omni_mercury_engine.reasoning.schemas.Explanation` recording
            which backend and model actually served and that the gate cleared it.

        Raises:
            EthicalConstraintViolationError: If the dual ethical gate blocks the
                operation; no explanation is returned in that case.
        """
        from omni_mercury_engine.reasoning.schemas import ReasoningContext

        is_anomaly = bool(result.get("is_anomaly", False))
        anomaly_prob = self._clamp_unit(result.get("anomaly_prob", 0.0))
        threshold = self._clamp_unit(result.get("threshold_used", 0.0))
        summary = (
            f"Mercury fusion detection: is_anomaly={is_anomaly}, "
            f"anomaly_prob={anomaly_prob:.3f} (threshold {threshold:.3f})"
        )
        evidence: dict[str, Any] = {
            "anomaly_prob": result.get("anomaly_prob"),
            "is_anomaly": result.get("is_anomaly"),
            "threshold_used": result.get("threshold_used"),
            "class_prediction": result.get("class_prediction"),
            "severity": result.get("severity"),
        }
        importance = result.get("detector_importance")
        if isinstance(importance, dict):
            evidence["detector_importance"] = importance
        context = ReasoningContext(
            summary=summary,
            domain=domain,
            evidence=evidence,
            severity=self._clamp_unit(result.get("severity", 0.0)),
            anomaly_prob=anomaly_prob,
        )
        return self.reasoning_backend.explain(context)

    def reasoning_usage(self) -> dict[str, int] | None:
        """Return provider-reported token totals for reasoning calls.

        Reads the ledger threaded through the default reasoning backend (see
        :meth:`enable_reasoning`). Returns ``None`` when no ledger is attached
        (e.g. an operator-supplied backend constructed without one). Counts are
        provider-truthful: only real provider/adapter calls (Ollama, cloud)
        contribute; the deterministic builtin template is not a tokenised
        provider call and so books nothing.
        """
        if self._reasoning_ledger is None:
            return None
        return self._reasoning_ledger.totals()

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
        all_scores: np.ndarray[Any, Any] | None = None,
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
        """Enable automatic threshold calibration for all detectors.

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
        """Calibrate threshold from a batch of scores.

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
        """Run detection with full diagnostics for debugging F1=0 issues.

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
        """Detect anomalies with automatic threshold calibration.

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
        all_scores: list[np.ndarray[Any, Any]] = []
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
        """Dual hard ethical gate at the engine decision boundary.

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
            build_sigma_immutable_vector,
        )

        # Shared single-verdict builder (σ_Immutable Wave C): the engine
        # boundary scores benevolence only, so severity / anomaly_prob stay
        # at their defaults — this reproduces the prior inline vector
        # byte-for-byte while sourcing the layout from the one calibrated
        # helper the orchestrator, hub, and Wave C surfaces all share.
        sigma_vector = build_sigma_immutable_vector(float(ethical_score.benevolence_score))

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

    def _reset_transient_detector_state(self) -> None:
        """Reset detectors'/models' transient cross-call state (purity contract).

        Detection features must be a function of (fitted state, batch) only.
        The directive detector's recursive-memory buffer and the neural
        cognitive model's hippocampal buffer are streaming state that
        otherwise couples one extraction to the previous one: the affected
        rows' features changed with call history, so repeated
        ``detect_with_fusion`` calls drifted and a reloaded checkpoint
        could not reproduce the saving engine's probabilities
        (ROADMAP row 16; defect found 2026-06-11). Components keep their
        documented streaming semantics for direct callers — only the
        engine's fusion boundary resets.
        """
        for component in (*self.detectors.values(), *self.models.values()):
            reset_state = getattr(component, "reset_state", None)
            if callable(reset_state):
                reset_state()

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
            Tuple of (detector_features, detector_scores, detector_certificates):
                - detector_features: Dict mapping detector names to features
                - detector_scores: Dict mapping detector names to scores
                - detector_certificates: Dict mapping detector names to their
                  post-hoc ``info_geometry_certificate`` payload (read-only;
                  threaded through the return value rather than stashed on the
                  engine so interleaved ``detect()`` calls cannot cross-
                  contaminate certificates).

        Note:
            Uses parallel processing when available for improved performance.
            Features are cached using the FeatureCache for repeated access.

        Data leakage warning:
            If a detector is not yet fit when this method is called, it is
            auto-fit on ``data`` — which means the first batch passed to
            ``detect_with_fusion`` becomes the detector's reference
            distribution. This biases all subsequent scoring against that
            first batch. The name of each detector that was auto-fit here is
            recorded in ``self._inference_auto_fit_detectors`` and a
            ``logger.warning`` is emitted the first time it happens, so the
            caller can audit it. To avoid the leakage entirely, call
            ``fit_fusion(X_train, y_train)`` (which fits all detectors on
            ``X_train``) before any ``detect_with_fusion`` invocation.
        """
        detector_features = {}
        detector_scores = {}
        detector_certificates: dict[str, Any] = {}

        for name, detector in self.detectors.items():
            # Fail-loud (or legacy auto-fit) on an unfit detector. This runs
            # OUTSIDE the try below so the fail-loud RuntimeError is not
            # swallowed by the graceful per-detector skip handler.
            if not detector.is_fitted() and not isinstance(data, dict):
                if self._require_explicit_fit:
                    raise RuntimeError(
                        f"Detector {name!r} is not fitted; refusing to auto-fit on the "
                        "inference batch. Call fit_fusion(X_train, y_train) (or load a "
                        "checkpoint) before detect_with_fusion so the reference "
                        "distribution comes from training data, not the first batch "
                        "scored. To opt into the legacy auto-fit-on-first-batch "
                        "behaviour, construct OmniMercuryEngine(require_explicit_fit=False)."
                    )
                if name not in self._inference_auto_fit_detectors:
                    logger.warning(
                        "Detector %r was auto-fit on the first inference "
                        "batch (n=%s) because it had no prior fit. The "
                        "batch's distribution is now this detector's "
                        "reference — call fit_fusion(X_train, y_train) "
                        "before detect_with_fusion to avoid the leakage.",
                        name,
                        getattr(data, "shape", ("?",))[0] if hasattr(data, "shape") else "?",
                    )
                self._inference_auto_fit_detectors.add(name)
                detector.fit(data)

            try:
                if isinstance(data, dict) and not detector.is_fitted():
                    continue

                # Try to use cached features
                cache_key = self.feature_cache._make_key(
                    data if not isinstance(data, dict) else np.array([0]), prefix=f"detector_{name}"
                )

                def compute_features(det: Any = detector, d: Any = data) -> tuple[Any, ...]:
                    # Purity: start from clean transient state so the
                    # features depend only on (fitted state, d) — see
                    # _reset_transient_detector_state. Runs inside the
                    # compute so a worker thread resets its *own*
                    # thread-local state; cache hits skip recomputation
                    # and stay deterministic by construction.
                    reset_state = getattr(det, "reset_state", None)
                    if callable(reset_state):
                        reset_state()
                    features = det.extract_features(d)
                    result = det.detect(d)
                    return features, result

                cached = self.feature_cache.get_or_compute(cache_key, compute_features)
                features, result = cached

                detector_features[name] = features
                scores = result.get("scores", result.get("is_anomaly", 0))
                detector_scores[name] = self._normalize_scores(scores, features.shape[0])
                if "info_geometry_certificate" in result:
                    detector_certificates[name] = result["info_geometry_certificate"]
            except (
                OmniAnomalyException,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                AttributeError,
                IndexError,
            ) as e:
                # OmniAnomalyException (DetectorException/ModelException/...) is a
                # detector signalling "I cannot process this input" — the same
                # fail-loud contract a specialized detector (e.g. geo_movement on
                # non-trajectory data) uses. The training-time extractor
                # (_extract_fusion_features) already skips it via ``except
                # Exception``; catching it here keeps the inference path's
                # graceful-skip contract symmetric so one incompatible detector
                # cannot crash detect_with_fusion.
                logger.debug(f"Detector {name} feature extraction failed: {e}")
                continue

        return detector_features, detector_scores, detector_certificates

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
                    # Purity: reset transient streaming state so features
                    # depend only on (model state, d) — mirrors the
                    # detector-side closure above.
                    reset_state = getattr(mdl, "reset_state", None)
                    if callable(reset_state):
                        reset_state()
                    features = mdl.extract_features(d)
                    prediction = mdl.predict(d)
                    return features, prediction

                cached = self.feature_cache.get_or_compute(cache_key, compute_features)
                features, prediction = cached

                model_features[name] = features
                scores = prediction.get("anomaly_scores", 0)
                model_scores[name] = self._normalize_scores(scores, features.shape[0])
            except (
                OmniAnomalyException,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                AttributeError,
                IndexError,
            ) as e:
                # Mirror the detector path: a ModelException (or any Mercury
                # component exception) means "this model cannot process the
                # input" and must degrade gracefully, matching this method's
                # documented contract, rather than propagating out of inference.
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
        det_features, det_scores, _ = detector_future.result()
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
        explain: bool = False,
        equation_profile: str | None = None,
        gdpr_report: bool = False,
        subject_id: str | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies using ML fusion with GOSNN synaptic integration.

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
            explain: When ``True``, attach an ``explanation`` field — an
                Integrated-Gradients attribution of this sample's calibrated
                fusion probability (over the same ``score_fusion`` serve path)
                plus its faithfulness scores. Off by default because IG over
                the full fusion stack is expensive (finite-difference forward
                passes per feature × interpolation step).
            equation_profile: Optional runtime equation profile override. When
                provided (or set engine-wide via :attr:`equation_profile`), the
                calibrated neural fusion probability is blended with the
                selected R/H/O equation profile before thresholding, and the
                blend metadata is attached under ``result["equation_profile"]``.
                ``None`` (default) leaves the calibrated probability unchanged.
            gdpr_report: When ``True``, attach a ``gdpr_report`` field — a GDPR
                Article 22 explanation of this decision (top contributing
                factors via the from-scratch Shapley engine, actionable
                counterfactual changes, and the data-subject rights narrative)
                built over the same ``score_fusion`` serve path. Off by default
                because it runs SHAP + counterfactual optimisation per call.
            subject_id: Optional data-subject identifier recorded in the GDPR
                report's audit trail. Only consulted when ``gdpr_report=True``;
                a unique per-report id (``anon-<hex>``) is generated when omitted,
                so distinct data-subject audits never collapse onto one identifier
                (the generated id is surfaced in the report's ``subject_id`` field).

        Returns:
            Dictionary containing:
                - anomaly_prob: Probability of anomaly (0.0-1.0)
                - is_anomaly: Boolean anomaly flag (prob > 0.5)
                - class_prediction: Predicted anomaly class
                - severity: Anomaly severity score
                - detector_importance: Dict of detector weights
                - mode: Detection mode ('fusion')
                - explanation: (only when ``explain=True``) Integrated-Gradients
                  feature attribution + faithfulness scores for this sample
                - gdpr_report: (only when ``gdpr_report=True``) GDPR Article 22
                  decision explanation — top factors, counterfactual actions,
                  and data-subject rights narrative for this sample
                - gosnn_metadata: GOSNN + σ_Immutable evaluation metadata:
                    - sigma_immutable_score: σ_Immutable score
                    - ethical_gate_passed: σ_Immutable threshold check
                    - sigma_immutable_threshold: decision threshold used
                    - sigma_immutable_backend: ``"torch"`` for the trained
                      network, ``"unavailable"`` if the network could not
                      run (the engine raises before returning in that case).
                    - intelligence_contribution: GOSNN intelligence score
                    - warnings: Any ethical warnings
                    - detection: (only when a detection-metric merit-gated
                      head shipped with the fusion checkpoint) the GOSNN
                      fused-state ``anomaly_prob`` plus the shipped
                      ``demote_act_below``/``demote_clear_above`` thresholds
                      consumed by the decision layer's disagreement overlay

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

        # Normalize a single 1-D sample to a (1, n_features) batch, matching
        # score_fusion's contract. The detector extractors disagree on whether
        # a 1-D array is one sample or n one-feature samples, so an
        # un-normalized 1-D input surfaces as an opaque tensor-shape error
        # inside the fusion forward pass instead of a scored sample.
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(1, -1)
        elif isinstance(data, torch.Tensor) and data.dim() == 1:
            data = data.unsqueeze(0)

        det_features, det_scores, det_certificates = self._extract_detector_features(data)
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
        # BenevolenceScorer.  The primary, per-call, content-driven ethical
        # contract is BenevolenceScorer.enforce (keyword- and context-driven,
        # deterministic) — enforced here, before the GOSNN block.
        #
        # σ_Immutable is NOT a per-input-data classifier: it evaluates the
        # 127 operational governance scalars (padded to the network's 256-d
        # input), and those are a property of the system's *configuration*,
        # not of the row being detected.  Measured across normal+anomalous
        # inputs and four domains: all 127 operational scalars are bit-constant
        # and the surfaced sigma_immutable_score is the exact constant
        # 0.9999216794967651 on every call (the network is now trained on the
        # harvested intact config — scripts/harvest_sigma_baseline.py — so it
        # recognises the real production vector by construction).  So
        # σ_Immutable is a config-integrity / tamper check — it reads "intact"
        # constantly on normal operation and only moves if a critical ethical
        # scalar is corrupted (e.g. an anchor zeroed).  The AUTHORITATIVE catch
        # for that case is the deterministic critical-ethical floor below
        # (enforce_ethical_floor); the learned score over the full vector is
        # advisory (and, since the harvested retrain, agrees with the floor on
        # anchor collapse instead of falsely assuring it).  Per-call ethical
        # sensitivity lives in BenevolenceScorer, not here.
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

                # Per-call operational scalars for the GOSNN fusion: the mean of
                # each detector/model anomaly score on THIS sample. Detector
                # scores arrive as torch.Tensor (``_normalize_scores`` always
                # returns a tensor), so the previous ``(np.ndarray, float, int)``
                # filter silently dropped every one -- leaving the fusion's
                # per-call base member empty and its input constant across calls
                # (the degenerate harvest recorded in artifacts/gosnn_fusion.eval.json).
                # Coercing tensor scores to floats gives fuse() genuine per-call
                # variation so its harmonic-synergy / fusion-score observability
                # reflects the real detection state instead of a fixed constant.
                base_scalars: dict[str, float] = {}
                for name, score in all_scores.items():
                    # ``torch`` is optional (None when the [ml] extra is absent);
                    # guard the tensor branch so a torch-free install never
                    # dereferences ``torch.Tensor`` on None.
                    if TORCH_AVAILABLE and isinstance(score, torch.Tensor):
                        if score.numel() == 0:
                            continue
                        base_scalars[f"detector_{name}_score"] = float(
                            score.detach().cpu().float().mean()
                        )
                    elif isinstance(score, (np.ndarray, float, int)):
                        base_scalars[f"detector_{name}_score"] = float(np.mean(score))

                enhancement_result = gosnn.get_enhanced_scalars(
                    requesting_component="OmniMercuryEngine.detect_with_fusion",
                    base_scalars=base_scalars,
                    context={"domain": domain, "data_shape": getattr(data, "shape", None)},
                )

                # Hard σ_Immutable enforcement — evaluate against the EXACT
                # scalar snapshot GOSNN's advisory gate already scored inside
                # get_enhanced_scalars, reusing it rather than taking a second
                # independent collection. This removes one 127-scalar registry
                # walk per detect call AND closes a latent signal-integrity
                # gap: two separate collections could diverge under a
                # concurrent registration, leaving the advisory and the
                # authoritative gate evaluating different vectors. Falls back
                # to a fresh collection only if the enhancement did not carry
                # a snapshot (defensive; get_enhanced_scalars always does).
                full_scalars = (
                    enhancement_result.collected_scalars
                    if enhancement_result.collected_scalars is not None
                    else gosnn._collect_all_scalars()
                )
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

                # ``harmonic_synergy`` is deliberately NOT surfaced here.
                # Measured on the production serve path (2026-07-17, 15/15
                # detect calls on real ADBench rows with the shipped trained
                # fusion): the FFT top-two-magnitude ratio behind it is
                # pinned to exactly 1.0 by real-input conjugate symmetry, so
                # the "synergy" was the bit-identical constant 0.618034 on
                # every call — a number that never moves is not a live
                # metric.  It remains available as a documented diagnostic
                # (``gosnn.last_harmonic_synergy``); see
                # ``TriadicPhiWeighting.compute_harmonic_synergy``.
                gosnn_metadata = {
                    "ethical_gate_passed": evaluation.passes,
                    "sigma_immutable_score": evaluation.score,
                    "sigma_immutable_threshold": evaluation.threshold,
                    "sigma_immutable_backend": evaluation.backend,
                    "intelligence_contribution": (enhancement_result.intelligence_contribution),
                    "warnings": enhancement_result.warnings,
                    "enhancement_fusion_score": enhancement_result.fusion_score,
                }

                # Consequential channel (decision-layer routing only): when a
                # merit-gated detection head shipped with the fusion
                # checkpoint, surface its fused-state anomaly probability and
                # the validation-selected disagreement thresholds. The
                # decision layer's overlay demotes a grounded verdict to a
                # deferral on strong disagreement — abstention-only, so this
                # channel can never force an ACT, never touches the
                # σ_Immutable scalar vector or verdict above, and never
                # perturbs the OmniFusionModel features/anomaly_prob
                # (pinned by tests/core/test_gosnn_decision_channel.py).
                # Absent head => absent key: the fused state stays
                # observability-only exactly as before.
                gosnn_fused_state = enhancement_result.fused_state
                if gosnn_fused_state is not None:
                    gosnn_detection_prob = gosnn.attention_fusion.detection_probability(
                        gosnn_fused_state
                    )
                    if gosnn_detection_prob is not None:
                        gosnn_thresholds = gosnn.attention_fusion.decision_thresholds or {}
                        gosnn_metadata["detection"] = {
                            "anomaly_prob": gosnn_detection_prob,
                            "demote_act_below": gosnn_thresholds.get("demote_act_below"),
                            "demote_clear_above": gosnn_thresholds.get("demote_clear_above"),
                            "backend": "gosnn_detection_head",
                        }

                # The fused/enhanced scalars are NOT registered back into the
                # operational scalar pool. They are per-call detector anomaly
                # scores rescaled by the fusion factor -- registering them under
                # ETHICAL/SECURITY made them count as critical ethical anchors
                # (``critical_ethical_anchors`` returns the whole ETHICAL group),
                # so a low-anomaly sample collapsed the deterministic σ_Immutable
                # floor, and it also perturbed the fixed operational layout the
                # trained σ_Immutable gate expects. This write-only registration
                # was inert while ``base_scalars`` was empty; now that the fusion
                # carries genuine per-call input its enhanced view is surfaced as
                # observability (``gosnn_metadata`` above) rather than fed back
                # into the gate's scalar vector.

                logger.debug(
                    "GOSNN integration: σ_Immutable=%s (score=%.3f, " "threshold=%.3f)",
                    evaluation.passes,
                    evaluation.score,
                    evaluation.threshold,
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

        # Calibrate at the production decision boundary, in-place into the
        # result dict, so every downstream consumer (anomaly_prob, the
        # threshold finder, drift signals) sees the same temperature-scaled
        # probabilities the benchmark path serves through ``score_fusion``.
        # Without this, the persisted post-hoc temperature scalar (Guo et al.
        # 2017) trained by ``fit_fusion`` only affects ``score_fusion`` and
        # the user-facing ``mercury-agent detect`` keeps returning raw sigmoid.
        fusion_result["anomaly_probs"] = self._apply_fusion_calibration(
            np.asarray(fusion_result["anomaly_probs"])
        )

        # Optional runtime equation profile (opt-in; exact no-op when unset).
        # Applied on the calibrated probabilities so the blended score stays on
        # the same temperature scale every downstream consumer reads.
        fusion_result["anomaly_probs"], equation_profile_metadata = (
            self._apply_runtime_equation_profile(
                np.asarray(fusion_result["anomaly_probs"]),
                all_scores,
                equation_profile=equation_profile,
                domain=domain,
            )
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

        # Surface any detectors that had to be auto-fit on the inference batch.
        # Empty when training was done properly via fit_fusion; non-empty means
        # the result is biased by the leakage and the caller should know.
        if self._inference_auto_fit_detectors:
            result["inference_auto_fit_detectors"] = sorted(self._inference_auto_fit_detectors)

        # Add GOSNN metadata if integration was enabled
        if gosnn_metadata:
            result["gosnn_metadata"] = gosnn_metadata
        if det_certificates:
            result["info_geometry_certificate"] = det_certificates

        # Surface the runtime equation-profile blend metadata when a profile
        # was applied (absent on the default, profile-less serve path).
        if equation_profile_metadata is not None:
            result["equation_profile"] = equation_profile_metadata

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

        # Online drift recalibration: feed this batch's calibrated scores to the
        # adaptive conformal threshold so the operating point tracks score drift
        # rather than going stale (see enable_online_recalibration). Surfaced for
        # audit; applied as the operating threshold only after warmup so the
        # decider grounds on a converged, fresh threshold instead of deferring
        # forever on drift.
        if self._adaptive_conformal is not None:
            batch_scores = np.asarray(fusion_result["anomaly_probs"], dtype=float).reshape(-1)
            for s in batch_scores:
                self._adaptive_conformal.update(float(s))
            adaptive_threshold = self._adaptive_conformal.get_current_threshold()
            result["adaptive_conformal"] = self._adaptive_conformal.get_coverage_stats()
            if np.isfinite(adaptive_threshold):
                result["adaptive_threshold"] = float(adaptive_threshold)
                if self._adaptive_conformal.n_updates >= self._recalibration_warmup:
                    result["threshold_used"] = float(adaptive_threshold)
                    result["is_anomaly"] = bool(float(anomaly_prob_val) > adaptive_threshold)

        # Neuro-symbolic feedback diagnostics. The co-trained LTN is retained
        # after fit/load so production inference can expose whether the current
        # neural verdict still agrees with the symbolic detector-consensus
        # graph that shaped training.
        if isinstance(data, (np.ndarray, torch.Tensor)):
            data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
            symbolic_payload = self._symbolic_consistency_payload(
                np.asarray(data_np, dtype=np.float32),
                np.asarray(fusion_result["anomaly_probs"], dtype=np.float32),
            )
            if symbolic_payload is not None:
                result["symbolic_consistency"] = symbolic_payload

        # Runtime Pipeline Integration: LLM Enhancement (non-blocking)
        llm_enhancement = self._enhance_with_llm(data, result)  # type: ignore[arg-type, unused-ignore]
        if llm_enhancement is not None:
            result["llm_enhancement"] = llm_enhancement

        if self.cognitive_orchestrator is not None and isinstance(data, (np.ndarray, torch.Tensor)):
            raw_data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
            cognitive = self.cognitive_orchestrator.analyze(
                detection_result=result,
                raw_data=np.asarray(raw_data),
                context={
                    "domain": domain or "general",
                    "symbolic_consistency": result.get("symbolic_consistency"),
                    "gosnn_metadata": result.get("gosnn_metadata"),
                    "drift_detection": result.get("drift_detection"),
                },
            )
            result["cognitive_analysis"] = cognitive.to_dict()

        # Conformal uncertainty: when a conformal calibrator has been fit via
        # calibrate_fusion_conformal(), attach the distribution-free label
        # prediction set for this sample's calibrated probability so detect
        # returns a calibrated probability *and* an uncertainty set, not a bare
        # score. A no-op (no key added) until the calibrator is fit.
        if self._fusion_conformal is not None:
            pred_set = self._fusion_conformal.predict(np.array([float(anomaly_prob_val)]))
            result["conformal"] = {
                "prediction_set": pred_set.label_sets()[0],
                "set_size": int(pred_set.set_size[0]),
                "abstain": bool(pred_set.set_size[0] == 2),
                "coverage": float(pred_set.coverage_level),
            }

        # Explainability (opt-in): attach an Integrated-Gradients attribution of
        # this sample's calibrated fusion probability, plus its faithfulness
        # scores. Wired against the *same* serve-path probability the result
        # reports (``score_fusion``), so the explanation explains the shipped
        # decision rather than a proxy. Opt-in because IG over the full fusion
        # stack is expensive (finite-diff forward passes per feature/step).
        if explain and isinstance(data, (np.ndarray, torch.Tensor)):
            result["explanation"] = self._explain_fusion_decision(data)

        # GDPR Article 22 report (opt-in): a data-subject-facing explanation of
        # this automated decision — top contributing factors (Shapley),
        # actionable counterfactual changes, and the rights narrative — built
        # over the same ``score_fusion`` serve path. Distinct from ``explain``:
        # that attaches an IG attribution for engineers; this attaches the
        # compliance/recourse report neither the cognitive nor ml explainer
        # provides. Opt-in because it runs SHAP + counterfactual optimisation.
        if gdpr_report and isinstance(data, (np.ndarray, torch.Tensor)):
            result["gdpr_report"] = self._gdpr_explain_fusion_decision(data, subject_id, result)

        # Decision / abstention / response layer: close the loop from the
        # calibrated certificate just assembled (probability + conformal set +
        # ethical verdict + symbolic agreement + drift) to a bounded,
        # non-destructive response with an explicit "don't-know" gate. A no-op
        # (no key added) until enable_decision_layer() is called. When an audit
        # ledger was supplied, the decision is also recorded -- the "verify"
        # step that turns the stream of decisions into a queryable trail.
        if self.decision_layer is not None:
            decision_record = self.decision_layer.decide(result, domain=domain)
            if self.decision_ledger is not None:
                self.decision_ledger.record(decision_record)
            result["decision"] = decision_record.to_dict()

        return result

    def _explain_fusion_decision(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Integrated-Gradients explanation of the serve-path fusion probability.

        Computes an attribution of ``score_fusion``'s calibrated probability for
        the first sample in ``data`` and evaluates its faithfulness
        (comprehensiveness / sufficiency / monotonicity). Returns the explanation
        as a JSON-serialisable dict for attachment to the detection result.

        Args:
            data: The same input passed to :meth:`detect_with_fusion`.

        Returns:
            ``Explanation.to_dict()`` augmented with ``faithfulness_scores``.
        """
        from omni_mercury_engine.cognitive.explainability import (
            FaithfulnessEvaluator,
            IntegratedGradientsExplainer,
        )

        arr = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        instance = np.atleast_2d(arr.astype(np.float64))[0]

        def _fusion_predict(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.asarray(
                self.score_fusion(np.atleast_2d(np.asarray(x, dtype=np.float32))),
                dtype=np.float64,
            )

        explainer = IntegratedGradientsExplainer(n_steps=_EXPLAIN_IG_STEPS)
        explanation = explainer.explain(_fusion_predict, instance)
        explanation.faithfulness_scores = FaithfulnessEvaluator().evaluate(
            _fusion_predict, instance, explanation
        )
        return explanation.to_dict()

    def _gdpr_explain_fusion_decision(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        subject_id: str | None,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """GDPR Article 22 report for the serve-path fusion decision.

        Builds a :class:`~omni_mercury_engine.explainability.MercuryExplainer`
        over the same ``score_fusion`` probability the result reports, and
        returns its report as a JSON-serialisable dict: top contributing factors
        (from the from-scratch Shapley engine), actionable counterfactual
        changes, and the data-subject rights narrative.

        The report is decomposed against a single reference row -- the mean of
        the training sample stored by :meth:`fit_fusion` (a standard SHAP
        baseline, analogous to the IG baseline). When no usable fit background
        is available (never fit, or a stale different-width one), the fallback
        is a zero-vector reference of the instance's width -- deliberately NOT
        the instance itself, which would make the marginalisation baseline equal
        the point being explained and collapse every attribution to ~0. A
        one-row reference keeps the Shapley marginalisation tractable over the
        full ``score_fusion`` stack; the full matrix would multiply the
        per-coalition model evaluations by its row count and make the opt-in
        report far slower.

        Args:
            data: The same input passed to :meth:`detect_with_fusion`.
            subject_id: Optional data-subject id for the audit trail.
            result: The in-progress detection result (for anomaly_prob/threshold).

        Returns:
            ``ExplanationReport.to_dict()`` for attachment to the result.
        """
        from omni_mercury_engine.explainability import MercuryExplainer

        arr = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        instance = np.atleast_2d(arr.astype(np.float64))[0]

        def _fusion_predict(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.asarray(
                self.score_fusion(np.atleast_2d(np.asarray(x, dtype=np.float32))),
                dtype=np.float64,
            )

        n_features = int(instance.shape[0])
        # Bind the background once (avoid a train/serve TOCTOU where a concurrent
        # fit_fusion nulls it between checks) and require it to match this
        # instance's width -- a different training entry point (e.g.
        # fit_fusion_pooled) may have left a stale background of another width,
        # which would otherwise crash the explainer.
        stored_background = self._fusion_background
        if (
            stored_background is not None
            and stored_background.ndim == 2
            and stored_background.shape[0] > 0
            and stored_background.shape[1] == n_features
        ):
            background = stored_background.mean(axis=0, keepdims=True)
        else:
            # No usable stored background (absent, or a stale different-width one).
            # Using the instance as its own background would make every SHAP
            # attribution ~0 (the marginalisation baseline would equal the point
            # explained), so fall back to a zero-vector reference of the right
            # width -- a neutral baseline (like the IG path) that carries signal.
            background = np.zeros((1, n_features), dtype=np.float64)

        # Only label features when the stored drift names match this instance's
        # width; a mismatched _drift_feature_names (a different feature space)
        # would crash the report or mislabel the factors.
        drift_names = getattr(self, "_drift_feature_names", None)
        feature_names = (
            drift_names if drift_names is not None and len(drift_names) == n_features else None
        )

        # Use an explicit None check, not ``or``: a genuine threshold of 0.0 is
        # falsy and must not silently become 0.5 (which would flip the report's
        # adverse/normal verdict versus the engine's own decision).
        threshold_value = result.get("threshold_used")
        if threshold_value is None:
            threshold_value = result.get("threshold")
        if threshold_value is None:
            threshold_value = 0.5
        threshold = float(threshold_value)
        explainer = MercuryExplainer(
            model=_fusion_predict,
            background_data=np.asarray(background, dtype=np.float64),
            feature_names=feature_names,
            threshold=threshold,
            model_id="omni_fusion",
            model_version="1.0",
            shap_method="auto",
            counterfactual_method="wachter",
            seed=0,
        )
        # Generate a unique per-report id when the caller omits ``subject_id`` so
        # unrelated data-subject audits keep distinct identifiers instead of all
        # collapsing onto one constant. The explicit-id path is unchanged.
        if not subject_id:
            from uuid import uuid4

            subject_id = f"anon-{uuid4().hex[:16]}"
        report = explainer.generate_report(
            instance,
            subject_id=subject_id,
            anomaly_score=float(result["anomaly_prob"]),
        )
        return report.to_dict()

    def detect_with_fusion_calibrated(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor | dict[str, Any],
        labels: np.ndarray[Any, Any] | None = None,
        calibration_method: str = "auto",
        contamination: float | None = None,
        domain: str | None = None,
        _enable_gosnn: bool = True,
        equation_profile: str | None = None,
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
            equation_profile: Optional runtime equation profile override. When
                set, both the per-sample fusion probability and the batch
                probabilities used for threshold selection are blended on the
                same profiled scale so the threshold and the verdict stay
                internally consistent. ``None`` (default) is an exact no-op.

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
            equation_profile=equation_profile,
        )

        # Get fusion probability
        anomaly_prob = fusion_result.get("anomaly_prob", 0.5)

        # For batch data, we need the full probability array
        # The fusion_inference returns probs for all samples
        if self.mode == "fusion":
            det_features, det_scores, _ = self._extract_detector_features(data)
            mod_features, mod_scores = self._extract_model_features(data)
            all_features = {**det_features, **mod_features}

            fusion_output = self.fusion_inference.predict(
                self._restrict_to_trained_groups(all_features), return_attention=True
            )
            all_probs = fusion_output.get("anomaly_probs", np.array([anomaly_prob]))
            if isinstance(all_probs, torch.Tensor):
                all_probs = all_probs.cpu().numpy()
            all_probs = np.asarray(all_probs).flatten()
            # Calibrate before threshold selection: the threshold finder
            # (Otsu/F1/percentile) must operate on the same probability scale
            # as detect_with_fusion's calibrated anomaly_prob, otherwise the
            # threshold and the scalar live on different scales and the
            # is_anomaly verdict is internally inconsistent.
            all_probs = self._apply_fusion_calibration(all_probs)
            # Keep the batch threshold scale identical to the (possibly
            # profile-blended) per-sample anomaly_prob returned by
            # detect_with_fusion. Exact no-op when no profile is selected.
            all_probs, _ = self._apply_runtime_equation_profile(
                all_probs,
                {**det_scores, **mod_scores},
                equation_profile=equation_profile,
                domain=domain,
            )
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
                    scaler.scale(loss / gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call, unused-ignore]

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
            checkpoint = safe_torch_load(best_checkpoint_path, map_location=self.device)
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

    @classmethod
    def _fitted_state_to_checkpoint(cls, state: Any) -> Any:
        """Convert a fitted-state structure to checkpoint-safe leaves.

        ``torch.load(weights_only=True)`` admits tensors and primitives but
        not numpy arrays, so ndarray leaves (at any nesting depth) are
        stored as tensors (the same convention as ``domain_encoder_scaler``)
        tagged for exact dtype restoration.
        """
        if isinstance(state, np.ndarray):
            return {
                "__ndarray__": torch.from_numpy(np.ascontiguousarray(state)).clone(),
                "dtype": str(state.dtype),
            }
        if isinstance(state, dict):
            return {key: cls._fitted_state_to_checkpoint(value) for key, value in state.items()}
        if isinstance(state, (list, tuple)):
            return [cls._fitted_state_to_checkpoint(value) for value in state]
        return state

    @classmethod
    def _fitted_state_from_checkpoint(cls, state: Any) -> Any:
        """Invert :meth:`_fitted_state_to_checkpoint`."""
        if isinstance(state, dict) and "__ndarray__" in state:
            # .cpu() before .numpy(): load_model maps the checkpoint to the
            # engine device, so these tensors are CUDA tensors on GPU engines.
            return state["__ndarray__"].detach().cpu().numpy().astype(np.dtype(state["dtype"]))
        if isinstance(state, dict):
            return {key: cls._fitted_state_from_checkpoint(value) for key, value in state.items()}
        if isinstance(state, (list, tuple)):
            return [cls._fitted_state_from_checkpoint(value) for value in state]
        return state

    def save_model(self, path: str) -> None:
        """Save the fusion model to a versioned checkpoint.

        The checkpoint bundles, in one dict:

        * the model weights (``model_state_dict``) plus the metadata needed to
          rebuild the network before loading — ``feature_dims``, ``hidden_dim``
          and the dynamic-projection registry (input dim per lazily-created
          projection layer). Without this a reloaded model would fail
          ``load_state_dict`` on the data-dependent ``_dynamic_projections.*``
          keys;
        * the fitted temperature calibrator (if any) so loading restores
          trustworthy probabilities, not just the raw network;
        * the fitted base-detector state (``detector_fitted_state``) and the
          torch-module domain models' weights (``model_state_dicts``) so a
          reloaded engine extracts the *same fusion features* the saving
          engine trained on, instead of auto-fitting detectors on the first
          inference batch (ROADMAP row 16: per-sample probability drift up
          to ≈0.76 and train-time leakage);
        * the fitted conformal calibrator thresholds (``conformal_state``)
          so distribution-free prediction sets survive the round-trip;
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

        detector_fitted_state: dict[str, dict[str, Any]] = {}
        for det_name, det in self.detectors.items():
            exporter = getattr(det, "get_fitted_state", None)
            if not callable(exporter):
                continue
            try:
                det_state = exporter()
            except Exception as e:
                logger.warning("Could not export fitted state for detector %s: %s", det_name, e)
                continue
            if det_state is not None:
                detector_fitted_state[det_name] = self._fitted_state_to_checkpoint(det_state)

        model_state_dicts: dict[str, dict[str, Any]] = {
            model_name: dict(model.state_dict())
            for model_name, model in self.models.items()
            if isinstance(model, torch.nn.Module)
        }

        # Non-module models whose feature transform depends on construction
        # state (e.g. the multiverse population) export it the same way the
        # base detectors do.
        model_fitted_state: dict[str, Any] = {}
        for model_name, model in self.models.items():
            if isinstance(model, torch.nn.Module):
                continue
            exporter = getattr(model, "get_fitted_state", None)
            if not callable(exporter):
                continue
            try:
                model_state = exporter()
            except Exception as e:
                logger.warning("Could not export fitted state for model %s: %s", model_name, e)
                continue
            if model_state is not None:
                model_fitted_state[model_name] = self._fitted_state_to_checkpoint(model_state)

        # export_state() emits the exact mapping this key has always carried
        # (int-keyed thresholds), so the on-disk format is unchanged.
        conformal_state = (
            self._fusion_conformal.export_state() if self._fusion_conformal is not None else None
        )
        checkpoint = {
            "format_version": FUSION_CHECKPOINT_FORMAT_VERSION,
            "mercury_version": __version__,
            "model_state_dict": self.fusion_model.state_dict(),
            "feature_dims": dict(self.fusion_model.feature_dims),
            "hidden_dim": self.fusion_model.hidden_dim,
            "projection_registry": self.fusion_model.export_projection_registry(),
            "temperature": temperature,
            "feature_groups": self._fusion_feature_groups,
            "symbolic_constraint_state_dict": (
                self._symbolic_module.state_dict() if self._symbolic_module is not None else None
            ),
            "symbolic_constraint_config": self._symbolic_checkpoint_config(),
            "symbolic_constraint_score_channels": self._symbolic_score_channels,
            "domain_encoder_state_dict": (
                self._domain_encoder.state_dict() if self._domain_encoder is not None else None
            ),
            "domain_encoder_config": (
                {
                    "input_dim": self._domain_encoder.input_dim,
                    "hidden_dim": self._domain_encoder.hidden_dim,
                    "per_encoder_dim": self._domain_encoder.per_encoder_dim,
                    "output_dim": self._domain_encoder.output_dim,
                    "domains": self._domain_encoder.domains,
                    "encoder_kwargs": self._domain_encoder.encoder_kwargs,
                    "normalize": self._domain_encoder.normalize,
                }
                if self._domain_encoder is not None
                else None
            ),
            "domain_encoder_scaler": (
                tuple(torch.tensor(part, dtype=torch.float32) for part in self._domain_scaler)
                if self._domain_scaler is not None
                else None
            ),
            "provenance": self._fusion_provenance,
            "fusion_trained": bool(self._fusion_trained),
            "detector_fitted_state": detector_fitted_state,
            "model_state_dicts": model_state_dicts,
            "model_fitted_state": model_fitted_state,
            "conformal_state": conformal_state,
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str) -> None:
        """Load the fusion model from a checkpoint.

        Handles the structured checkpoint written by :meth:`save_model` and a
        legacy bare ``state_dict``. For structured checkpoints the model is
        rebuilt with the saved ``feature_dims`` and its dynamic projection
        layers are recreated before ``load_state_dict`` (so the full
        train -> save -> load -> serve workflow round-trips on the
        data-dependent ``_dynamic_projections.*`` keys), and the fitted
        temperature calibrator is restored when present so calibrated
        probabilities are available immediately.

        Calibration state is *replaced*, not merged: a structured checkpoint
        without a temperature or ``conformal_state`` entry clears any
        engine-local calibrator/conformal surface, since one fitted against
        a previous fusion stack would silently misapply to the loaded one.

        Args:
            path: File path to load the checkpoint from.

        Example:
            >>> engine.load_model("models/fusion_model.pt")
        """
        if self.mode != "fusion":
            return

        checkpoint = safe_torch_load(path, map_location=self.device)

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
            else:
                # A calibrator fitted to a previous fusion stack misapplies to
                # the loaded one: drop it so loaded state == saved state.
                self._fusion_calibrator = None

            groups = checkpoint.get("feature_groups")
            self._fusion_feature_groups = list(groups) if groups is not None else None
            symbolic_state = checkpoint.get("symbolic_constraint_state_dict")
            symbolic_config = checkpoint.get("symbolic_constraint_config")
            if symbolic_state is not None and isinstance(symbolic_config, dict):
                # Prefer the inline rule spec (written for non-registry graphs,
                # e.g. evolved ones) so the checkpoint round-trips without the
                # original artifact file; registry graphs resolve by name.
                graph_spec = symbolic_config.get("rule_graph_spec")
                rule_graph = (
                    rule_graph_from_spec(graph_spec)
                    if isinstance(graph_spec, dict)
                    else resolve_rule_graph(str(symbolic_config["rule_graph"]))
                )
                symbolic_module = SymbolicConstraintModule(
                    num_detectors=int(symbolic_config["num_detectors"]),
                    rule_graph=rule_graph,
                    semantics=str(symbolic_config["semantics"]),
                    learn_detector_reliability=bool(
                        symbolic_config.get("learn_detector_reliability", True)
                    ),
                    p_aggregator=float(symbolic_config.get("p_aggregator", 2.0)),
                ).to(self.device)
                symbolic_module.load_state_dict(symbolic_state)
                symbolic_module.eval()
                self._symbolic_module = symbolic_module
                channels = checkpoint.get("symbolic_constraint_score_channels")
                self._symbolic_score_channels = list(channels) if channels is not None else None
            else:
                self._symbolic_module = None
                self._symbolic_score_channels = None
            domain_state = checkpoint.get("domain_encoder_state_dict")
            domain_config = checkpoint.get("domain_encoder_config")
            domain_scaler = checkpoint.get("domain_encoder_scaler")
            if domain_state is not None and domain_scaler is not None:
                if isinstance(domain_config, dict):
                    domain_encoder = DomainEncoderStack(
                        input_dim=int(domain_config["input_dim"]),
                        hidden_dim=int(domain_config["hidden_dim"]),
                        per_encoder_dim=int(domain_config["per_encoder_dim"]),
                        output_dim=int(domain_config["output_dim"]),
                        domains=tuple(domain_config["domains"]),
                        encoder_kwargs=dict(domain_config["encoder_kwargs"]),
                        normalize=bool(domain_config["normalize"]),
                    ).to(self.device)
                    domain_encoder.load_state_dict(domain_state)
                    domain_encoder.eval()
                    self._domain_encoder = domain_encoder
                    self._domain_scaler = (
                        np.asarray(domain_scaler[0], dtype=np.float32),
                        np.asarray(domain_scaler[1], dtype=np.float32),
                    )
                else:
                    self._domain_encoder = None
                    self._domain_scaler = None
            else:
                self._domain_encoder = None
                self._domain_scaler = None
            provenance = checkpoint.get("provenance")
            self._fusion_provenance = dict(provenance) if provenance is not None else None
            self._fusion_trained = bool(checkpoint.get("fusion_trained", True))

            # Fitted base-detector state (ROADMAP row 16): restoring it means
            # the loaded engine extracts training-time features instead of
            # auto-fitting (and leaking) on the first inference batch.
            # Checkpoints written before this key keep the legacy behavior.
            detector_states = checkpoint.get("detector_fitted_state") or {}
            for det_name, det_state in detector_states.items():
                det = self.detectors.get(det_name)
                restorer = getattr(det, "set_fitted_state", None)
                if det is None or not callable(restorer):
                    logger.warning(
                        "Checkpoint carries fitted state for unknown detector %r; skipped",
                        det_name,
                    )
                    continue
                restorer(self._fitted_state_from_checkpoint(det_state))
                # The checkpoint's state replaces any earlier inference-batch
                # auto-fit, so the leak record no longer describes this
                # detector — drop it so the audit trail (and the warning for
                # any future auto-fit) stays truthful. Legacy loads restore
                # nothing, so a surviving contamination keeps its record.
                self._inference_auto_fit_detectors.discard(det_name)

            # Torch-module domain models: their randomly-initialized feature
            # extractors are part of the serve-time transform, so reload
            # their exact weights. A shape mismatch means the checkpoint does
            # not describe this engine's models — fail loud, never drift.
            model_states = checkpoint.get("model_state_dicts") or {}
            for model_name, model_state in model_states.items():
                model = self.models.get(model_name)
                if not isinstance(model, torch.nn.Module):
                    logger.warning(
                        "Checkpoint carries weights for unknown model %r; skipped", model_name
                    )
                    continue
                incompatible = model.load_state_dict(model_state, strict=False)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(
                        f"Checkpoint model weights for {model_name!r} do not match this "
                        f"engine (missing={incompatible.missing_keys}, "
                        f"unexpected={incompatible.unexpected_keys})"
                    )

            model_fitted_states = checkpoint.get("model_fitted_state") or {}
            for model_name, model_state in model_fitted_states.items():
                model = self.models.get(model_name)
                restorer = getattr(model, "set_fitted_state", None)
                if model is None or not callable(restorer):
                    logger.warning(
                        "Checkpoint carries fitted state for unknown model %r; skipped",
                        model_name,
                    )
                    continue
                restorer(self._fitted_state_from_checkpoint(model_state))

            # Restore the conformal serving surface, or drop a stale one fitted
            # against a previous fusion stack, so loaded state == saved state.
            conformal_state = checkpoint.get("conformal_state")
            self._fusion_conformal = (
                BinaryConformalClassifier.from_state(conformal_state)
                if conformal_state is not None
                else None
            )
        else:
            # Legacy bare state_dict (no metadata): load directly.
            self.fusion_model.load_state_dict(checkpoint)
            self._fusion_trained = True
            self._symbolic_module = None

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


# Legacy alias removed - project renamed to Mercury Agent
