# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""DEPRECATED: This module uses Mercury's mercury_ml (PCA) for anomaly detection. Mercury's production detector is MercuryAnomalyDetector in detectors/statistical.py. This module is retained for reference only.

Original: Sigma Directive detector implementing PCP, GSIS, RMD, and EOA protocols.

Enhanced with quantum pattern containment and nano-scale detection for
critical applications in medical diagnostics, geological monitoring,
and search-and-rescue systems.

Thread Safety:
    This detector uses thread-local storage for mutable state (memory_buffer)
    to ensure safe concurrent access in multi-threaded environments.

Memory Management:
    The memory buffer is bounded by memory_depth and can be explicitly
    cleared via clear_memory() or reset_state() methods.
"""

from __future__ import annotations

import warnings

warnings.warn(
    f"{__name__} is deprecated. Use MercuryAnomalyDetector.",
    DeprecationWarning,
    stacklevel=2,
)
import hashlib
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from scipy.fft import fft

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException

# Optional numba lane for the O(n^2*d) GSIS distance computation
# (``[performance]`` extra, mirrors ``detectors/spatial.py``). The kernel is
# an exact compiled form of the numpy reference: the inner sum of squared
# differences replicates numpy's pairwise-summation order (sequential below
# 8 elements, the 8-accumulator block up to 128, recursive halving above),
# so the produced distances — and therefore the percentile/count and the
# final scores — are bit-identical to the chunked-broadcast path. Pinned by
# ``tests/test_native_acceleration.py`` across seeds, shapes and
# duplicate-heavy (tie-boundary) data; measured 2.9x at 1.1k rows and 5.5x
# at 4k rows (2026-06-11).
try:
    from numba import njit, prange

    @njit(cache=True)
    def _gsis_pairwise_sum_sq(row_a, row_b, n):  # type: ignore[no-untyped-def]
        """Sum of squared differences in numpy's pairwise-summation order.

        Covers numpy's sequential (n < 8) and 8-accumulator (n <= 128)
        regimes; the recursive-halving regime (n > 128) is deliberately not
        implemented — self-recursion inside a numba parallel region is
        fragile (observed segfault), so wider rows take the numpy lane via
        the dtype/width gate in ``_gravitational_stability_check``.
        """
        if n < 8:
            res = 0.0
            for i in range(n):
                d = row_a[i] - row_b[i]
                res += d * d
            return res
        r = np.empty(8, dtype=np.float64)
        for j in range(8):
            d = row_a[j] - row_b[j]
            r[j] = d * d
        i = 8
        limit = n - (n % 8)
        while i < limit:
            for j in range(8):
                d = row_a[i + j] - row_b[i + j]
                r[j] += d * d
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            d = row_a[i] - row_b[i]
            res += d * d
            i += 1
        return res

    @njit(cache=True, parallel=True)
    def _gsis_distance_block(data, start, stop):  # type: ignore[no-untyped-def]
        """Distances from rows ``start:stop`` to every row (one chunk)."""
        n, d = data.shape
        out = np.empty((stop - start, n), dtype=np.float64)
        for i in prange(stop - start):
            for j in range(n):
                out[i, j] = np.sqrt(_gsis_pairwise_sum_sq(data[start + i], data[j], d))
        return out

    GSIS_NUMBA_AVAILABLE = True
except ImportError:
    GSIS_NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class DirectiveWeights:
    """Configurable weights for Sigma Directive score combination.

    All weights must sum to 1.0 within each category for proper normalization.
    These weights control the relative importance of each detection protocol.

    Attributes:
        pcp_weight: Weight for Pattern Convergence Protocol scores.
        gsis_weight: Weight for Gravitational Stability Integrity System scores.
        rmd_weight: Weight for Recursive Memory Dynamics scores.
        eoa_weight: Weight for Ethical Oversight Amplifier scores.
        quantum_blend: Blend factor for quantum enhancement (0=disable, 1=full).
        nano_blend: Blend factor for nano-scale detection (0=disable, 1=full).
        harmonic_blend: Blend factor for harmonic analysis (0=disable, 1=full).

    Example:
        >>> weights = DirectiveWeights(pcp_weight=0.4, gsis_weight=0.3)
        >>> detector = SigmaDirectiveDetector({"weights": weights})
    """

    pcp_weight: float = 0.3
    gsis_weight: float = 0.3
    rmd_weight: float = 0.2
    eoa_weight: float = 0.2
    quantum_blend: float = 0.2
    nano_blend: float = 0.15
    harmonic_blend: float = 0.1

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0 for base protocols."""
        base_sum = self.pcp_weight + self.gsis_weight + self.rmd_weight + self.eoa_weight
        if abs(base_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Base protocol weights must sum to 1.0, got {base_sum:.4f}. "
                f"Weights: PCP={self.pcp_weight}, GSIS={self.gsis_weight}, "
                f"RMD={self.rmd_weight}, EOA={self.eoa_weight}"
            )
        for name, val in [
            ("quantum_blend", self.quantum_blend),
            ("nano_blend", self.nano_blend),
            ("harmonic_blend", self.harmonic_blend),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass
class _ThreadLocalState:
    """Thread-local state for SigmaDirectiveDetector.

    Ensures thread safety by isolating mutable state per thread. Each thread gets its own memory
    buffer, preventing race conditions during concurrent detect() calls.
    """

    memory_buffer: deque[NDArray[np.float64]] = field(default_factory=deque)


class SigmaDirectiveDetector(BaseDetector):
    """Sigma Directive protocols for anomaly detection.

    Implements multiple complementary detection protocols for robust
    anomaly identification in critical systems:

    - **PCP** (Pattern Convergence Protocol): Detects deviation from baseline patterns
    - **GSIS** (Gravitational Stability Integrity System): Analyzes local density stability
    - **RMD** (Recursive Memory Dynamics): Tracks temporal memory-based anomalies
    - **EOA** (Ethical Oversight Amplifier): Amplifies ethically significant anomalies

    Optional enhancements:
    - Quantum Pattern Containment: Phase coherence and entanglement analysis
    - Nano-Scale Detection: Bit-level and micro-pattern anomalies
    - Harmonic Analysis: FFT-based frequency domain anomalies

    Thread Safety:
        This detector is thread-safe. Each thread maintains isolated state
        via thread-local storage, allowing safe concurrent detect() calls.

    Memory Management:
        Memory buffer is bounded by `memory_depth` configuration.
        Use `clear_memory()` to explicitly reset state between independent
        detection sessions.

    Attributes:
        convergence_threshold: Threshold for pattern convergence (default: 0.01).
        stability_factor: Multiplier for stability scores (default: 1.0).
        memory_depth: Maximum samples retained in memory buffer (default: 5).
        weights: DirectiveWeights instance for configurable score combination.

    Example:
        >>> detector = SigmaDirectiveDetector({
        ...     "convergence_threshold": 0.01,
        ...     "memory_depth": 10,
        ...     "weights": DirectiveWeights(pcp_weight=0.4, gsis_weight=0.3,
        ...                                  rmd_weight=0.2, eoa_weight=0.1),
        ... })
        >>> detector.fit(training_data)
        >>> result = detector.detect(test_data)
        >>> anomalies = result["is_anomaly"]
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize SigmaDirectiveDetector with configuration.

        Args:
            config: Configuration dictionary with optional keys:
                - convergence_threshold: PCP sensitivity (default: 0.01)
                - stability_factor: GSIS scaling factor (default: 1.0)
                - memory_depth: RMD buffer size (default: 5)
                - use_quantum_enhanced: Enable quantum protocols (default: True)
                - use_nano_detection: Enable nano-scale detection (default: True)
                - use_harmonic_detection: Enable harmonic analysis (default: True)
                - weights: DirectiveWeights instance or dict for score combination
        """
        super().__init__(config)
        self.convergence_threshold: float = self.config.get("convergence_threshold", 0.01)
        self.stability_factor: float = self.config.get("stability_factor", 1.0)
        self.memory_depth: int = self.config.get("memory_depth", 5)
        self.use_quantum_enhanced: bool = self.config.get("use_quantum_enhanced", True)
        self.use_nano_detection: bool = self.config.get("use_nano_detection", True)
        self.use_harmonic_detection: bool = self.config.get("use_harmonic_detection", True)

        # Configurable weights (addresses hardcoded magic numbers issue)
        weights_config = self.config.get("weights", None)
        if weights_config is None:
            self.weights = DirectiveWeights()
        elif isinstance(weights_config, DirectiveWeights):
            self.weights = weights_config
        elif isinstance(weights_config, dict):
            self.weights = DirectiveWeights(**weights_config)
        else:
            raise ValueError(
                f"weights must be DirectiveWeights or dict, got {type(weights_config)}"
            )

        self.baseline_pattern: NDArray[np.float64] | None = None

        # Thread-safe memory management using thread-local storage
        # Fixes: Thread Safety issue with memory_buffer mutation
        self._thread_local = threading.local()

        # Memory buffer capacity validation
        if self.memory_depth < 1:
            raise ValueError(f"memory_depth must be >= 1, got {self.memory_depth}")

    def _get_thread_state(self) -> _ThreadLocalState:
        """Get or create thread-local state.

        Returns:
            Thread-local state instance with isolated memory buffer.
        """
        if not hasattr(self._thread_local, "state"):
            self._thread_local.state = _ThreadLocalState(
                memory_buffer=deque(maxlen=self.memory_depth)
            )
        state: _ThreadLocalState = self._thread_local.state
        return state

    def clear_memory(self) -> None:
        """Clear the memory buffer for the current thread.

        Call this method between independent detection sessions to prevent memory from one session
        affecting another. This is automatically handled per-thread, but explicit clearing may be
        desired for deterministic behavior in single-threaded scenarios.
        """
        state = self._get_thread_state()
        state.memory_buffer.clear()

    def reset_state(self) -> None:
        """Reset all mutable state including memory buffer.

        This provides a full reset equivalent to creating a new detector instance while preserving
        fitted parameters (baseline_pattern).
        """
        self.clear_memory()
        # Reset calibration state from base class
        self._calibrated_threshold = None
        self._last_diagnostics = None

    def fit(self, data: NDArray[np.float64] | torch.Tensor) -> SigmaDirectiveDetector:
        """Fit Sigma protocols to normal/baseline patterns.

        Computes the baseline pattern from training data that represents
        "normal" behavior. All subsequent detection is relative to this baseline.

        Args:
            data: Training data array of shape (n_samples, n_features) or tensor.
                Should contain representative normal/non-anomalous samples.

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty or contains only NaN/Inf values.

        Example:
            >>> detector = SigmaDirectiveDetector()
            >>> detector.fit(normal_training_data)
            >>> result = detector.detect(test_data)
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        assert isinstance(data, np.ndarray)

        # Validate data
        if data.size == 0:
            raise DetectorException(
                "Cannot fit SigmaDirectiveDetector with empty data. "
                "Provide representative normal samples for baseline computation."
            )

        # Handle NaN/Inf values
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        finite_mask = np.isfinite(data).all(axis=1)
        if not np.any(finite_mask):
            raise DetectorException(
                "Cannot fit SigmaDirectiveDetector: all data values are NaN or Inf."
            )
        if not np.all(finite_mask):
            data = data[finite_mask]

        self.baseline_pattern = np.mean(data, axis=0).astype(np.float64)

        # Clear memory buffer on new fit (fresh start)
        self.clear_memory()

        self._is_fitted = True
        return self

    def detect(self, data: NDArray[np.float64] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using Sigma Directive protocols.

        Executes multiple complementary detection protocols and combines their
        scores using configurable weights. This multi-protocol approach provides
        robust anomaly detection for critical applications.

        Thread Safety:
            This method is thread-safe. Each thread maintains isolated memory
            state, allowing concurrent detection calls without data corruption.

        Auto-Calibration:
            When auto_calibrate=True (via enable_auto_calibration()), the
            threshold is automatically calibrated based on the score
            distribution, solving the F1=0 problem where good ROC-AUC
            is achieved but fixed threshold produces no positive predictions.

        Args:
            data: Input data array of shape (n_samples, n_features) or tensor.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Combined anomaly scores in [0, 1] range
                - pcp_scores: Pattern Convergence Protocol scores
                - gsis_scores: Gravitational Stability scores
                - rmd_scores: Recursive Memory Dynamics scores
                - eoa_scores: Ethical Oversight Amplifier scores
                - quantum_scores: Quantum enhancement scores (dict, if enabled)
                - nano_scores: Nano-scale detection scores (dict, if enabled)
                - harmonic_score: Harmonic analysis score (float, if enabled)
                - detector_type: "directive"
                - threshold: Effective threshold used (may be calibrated)
                - calibration_diagnostics: CalibrationDiagnostics if auto-calibrated

        Raises:
            DetectorException: If detector has not been fitted.

        Example:
            >>> detector = SigmaDirectiveDetector()
            >>> detector.fit(train_data).enable_auto_calibration(contamination=0.05)
            >>> result = detector.detect(test_data)
            >>> print(f"Found {result['is_anomaly'].sum()} anomalies")
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        assert isinstance(data, np.ndarray)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Execute core detection protocols
        pcp_scores = self._pattern_convergence_protocol(data)
        gsis_scores = self._gravitational_stability_check(data)
        rmd_scores = self._recursive_memory_dynamics(data)
        eoa_scores = self._ethical_oversight_amplifier(data)

        # Execute optional enhancements
        quantum_scores: dict[str, float] = {}
        if self.use_quantum_enhanced:
            quantum_scores = self._quantum_pattern_containment(data)

        nano_scores: dict[str, float] = {}
        if self.use_nano_detection:
            nano_scores = self._nano_scale_detection(data)

        harmonic_score: float = 0.0
        if self.use_harmonic_detection:
            harmonic_score = self._harmonic_anomaly_detection(data)

        # Combine base protocol scores using configurable weights
        # (Fixes hardcoded magic numbers issue)
        combined_scores = (
            pcp_scores * self.weights.pcp_weight
            + gsis_scores * self.weights.gsis_weight
            + rmd_scores * self.weights.rmd_weight
            + eoa_scores * self.weights.eoa_weight
        )

        # Blend in optional enhancement scores
        if quantum_scores and self.weights.quantum_blend > 0:
            quantum_avg = np.mean(list(quantum_scores.values()))
            blend = self.weights.quantum_blend
            combined_scores = combined_scores * (1 - blend) + quantum_avg * blend

        if nano_scores and self.weights.nano_blend > 0:
            nano_avg = np.mean(list(nano_scores.values()))
            blend = self.weights.nano_blend
            combined_scores = combined_scores * (1 - blend) + nano_avg * blend

        if harmonic_score > 0 and self.weights.harmonic_blend > 0:
            blend = self.weights.harmonic_blend
            combined_scores = combined_scores * (1 - blend) + harmonic_score * blend

        # Ensure scores are finite and in valid range
        if np.any(~np.isfinite(combined_scores)):
            combined_scores = np.nan_to_num(combined_scores, nan=0.5, posinf=1.0, neginf=0.0)
        combined_scores = np.clip(combined_scores, 0.0, 1.0)

        # Auto-calibration: compute optimal threshold from score distribution
        effective_threshold = self.threshold
        calibration_diagnostics = None

        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = combined_scores > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "pcp_scores": pcp_scores,
            "gsis_scores": gsis_scores,
            "rmd_scores": rmd_scores,
            "eoa_scores": eoa_scores,
            "quantum_scores": quantum_scores,
            "nano_scores": nano_scores,
            "harmonic_score": harmonic_score,
            "detector_type": "directive",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract Sigma protocol features for ML fusion."""
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        assert isinstance(data, np.ndarray)

        if not self._is_fitted:
            self.fit(data)

        pcp_scores = self._pattern_convergence_protocol(data)
        gsis_scores = self._gravitational_stability_check(data)
        rmd_scores = self._recursive_memory_dynamics(data)
        eoa_scores = self._ethical_oversight_amplifier(data)

        features = np.column_stack(
            [
                pcp_scores,
                gsis_scores,
                rmd_scores,
                eoa_scores,
                np.mean(data, axis=1),
                np.std(data, axis=1),
            ]
        )

        if features.shape[1] < 20:
            padding = np.zeros((features.shape[0], 20 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

    def _pattern_convergence_protocol(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """PCP: Detect pattern convergence anomalies.

        Returns continuous scores without hard clipping to preserve ranking information for
        downstream fusion models.

        Fix for Issue #7: No Score Continuity. Previously used np.minimum(..., 1.0) which capped
        scores, losing differentiation between extreme anomalies. Now uses soft normalization.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        convergence_diffs = np.linalg.norm(data - self.baseline_pattern, axis=1)

        normalized_diffs = convergence_diffs / (np.linalg.norm(self.baseline_pattern) + 1e-6)  # type: ignore[arg-type, operator, unused-ignore]

        # Soft normalization: x / (threshold + x) approaches 1 asymptotically
        # Preserves ordering while keeping scores in [0, 1) range
        scores = normalized_diffs / (self.convergence_threshold + normalized_diffs)

        return scores

    def _gravitational_stability_check(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """GSIS: Check gravitational stability (data distribution stability).

        Vectorized (2026-06-11): the per-row Python loop (one full distance
        pass plus one percentile call per sample) is replaced by chunked
        broadcasting — pairwise norms and the per-row 20th percentile are
        computed a block of rows at a time, bounding peak memory while
        removing ~2n small numpy calls. The norm uses the same ufunc
        reduction as the former ``np.linalg.norm(data - data[i], axis=1)``,
        so scores are bit-identical (pinned by
        ``tests/test_native_acceleration.py``).

        Numba lane (2026-06-11, ``[performance]`` extra): for float64 input
        up to 128 features the chunk distances come from a parallel JIT
        kernel that replicates numpy's pairwise-summation order exactly —
        still bit-identical (same parity test), ~3x at 1k rows and ~5.5x at
        4k rows, without the (chunk, n, d) broadcast temporaries. Other
        dtypes, wider rows (numpy's recursive-halving summation regime) and
        numba-less installs keep the numpy path.
        """
        n = len(data)
        if n < 2:
            return np.zeros(n)

        # The numba lane computes the same distances in the same summation
        # order without the (chunk, n, d) broadcast temporaries (bit-equal;
        # see the kernel docstring). float64 only: other dtypes follow the
        # numpy path so their (dtype-specific) arithmetic is untouched.
        use_numba = (
            GSIS_NUMBA_AVAILABLE
            and data.dtype == np.float64
            and data.ndim == 2
            and data.shape[1] <= 128  # numpy enters recursive pairwise halving above
        )
        data_contiguous = np.ascontiguousarray(data) if use_numba else data

        scores = np.zeros(n)
        chunk = max(1, min(128, n))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            if use_numba:
                distances = _gsis_distance_block(data_contiguous, start, stop)
            else:
                block = data[start:stop]
                distances = np.linalg.norm(block[:, None, :] - data[None, :, :], axis=2)
            thresholds = np.percentile(distances, 20, axis=1)
            local_density = np.sum(distances < thresholds[:, None], axis=1)
            scores[start:stop] = 1.0 - local_density / n

        return scores * self.stability_factor

    def _recursive_memory_dynamics(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """RMD: Detect anomalies using recursive memory dynamics.

        Tracks a sliding window of recent samples and detects anomalies based
        on deviation from the memory mean. This captures temporal patterns
        that other protocols may miss.

        Thread Safety:
            Uses thread-local storage for memory buffer, ensuring safe
            concurrent execution across threads.

        Args:
            data: Input data array of shape (n_samples, n_features).

        Returns:
            Continuous anomaly scores in [0, 1) range, where higher values
            indicate greater deviation from recent memory.

        Note:
            Scores use soft normalization (deviation / (1 + deviation)) to
            preserve ranking information for downstream fusion models.
        """
        scores = np.zeros(len(data), dtype=np.float64)

        # Get thread-local memory buffer (fixes thread safety issue)
        state = self._get_thread_state()
        memory_buffer = state.memory_buffer

        for i, sample in enumerate(data):
            # deque with maxlen automatically handles bounded size
            memory_buffer.append(sample.astype(np.float64))

            if len(memory_buffer) > 1:
                memory_array = np.array(memory_buffer)
                memory_mean = np.mean(memory_array, axis=0)
                deviation = np.linalg.norm(sample - memory_mean)
                # Soft normalization: deviation / (1 + deviation) for [0, 1) range
                scores[i] = deviation / (1.0 + deviation)

        return scores

    def _ethical_oversight_amplifier(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """EOA: Amplify detection of ethically significant anomalies."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        magnitude = np.linalg.norm(data, axis=1)
        magnitude_norm = magnitude / (np.max(magnitude) + 1e-6)

        return magnitude_norm

    def _quantum_pattern_containment(self, data: np.ndarray[Any, Any]) -> dict[str, float]:
        """Quantum Pattern Containment Protocol (QPCP)."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        normalized = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-10)

        superposition_state = np.sum(normalized, axis=0) / len(normalized)

        coherence = np.abs(superposition_state)
        entanglement = np.std(normalized, axis=0)

        pattern_scores = {
            "coherence": float(np.mean(coherence)),
            "entanglement": float(np.mean(entanglement)),
            "superposition_strength": float(np.linalg.norm(superposition_state)),
        }

        return pattern_scores

    def _nano_scale_detection(self, data: np.ndarray[Any, Any]) -> dict[str, float]:
        """Nano-Scale Detection & Response System (NDRS) Enhanced N term with dimensional.

        downsampling for micro-anomaly detection.
        """
        if data.ndim == 1:
            data = data.reshape(-1)

        data_bytes = data.tobytes()

        molecular_hash = self._molecular_hash_function(data_bytes)

        checksum = self._quantum_dot_checksum(data_bytes)

        bit_anomalies = self._detect_bit_anomalies(data_bytes)

        micro_anomalies = self._detect_micro_anomalies(data)

        dimensional_micro = self._dimensional_downsampling_detection(data)

        scores = {
            "molecular_hash_entropy": float(molecular_hash),
            "quantum_checksum": float(checksum),
            "bit_anomaly_rate": float(bit_anomalies),
            "micro_anomaly_score": float(micro_anomalies),
        }
        # Include the dimensional sub-score only when it computed successfully.
        # On failure it is None and is omitted, so the nano blend averages the
        # components that actually produced a score instead of being dragged
        # down by a spurious 0.0.
        if dimensional_micro is not None:
            scores["dimensional_micro_score"] = float(dimensional_micro)
        return scores

    def _harmonic_anomaly_detection(self, data: np.ndarray[Any, Any]) -> float:
        """Harmonic anomaly detection using FFT."""
        signal = data if data.ndim == 1 else data.flatten()

        if len(signal) < 8:
            return 0.0

        fft_result = fft(signal)
        power_spectrum = np.abs(fft_result) ** 2

        frequencies = np.fft.fftfreq(len(signal))

        fundamental_freq = frequencies[1] if len(frequencies) > 1 else 0.0

        harmonic_powers = []
        for n in range(1, min(8, len(signal) // 2)):
            harmonic_idx = int(n * fundamental_freq * len(signal))
            if 0 <= harmonic_idx < len(power_spectrum):
                harmonic_powers.append(power_spectrum[harmonic_idx])

        if not harmonic_powers:
            return 0.0

        total_power = np.sum(power_spectrum)
        harmonic_power = np.sum(harmonic_powers)

        harmonic_ratio = harmonic_power / (total_power + 1e-10)

        anomaly_score = 1.0 - min(harmonic_ratio * 2.0, 1.0)

        return float(anomaly_score)

    @staticmethod
    def _molecular_hash_function(data: bytes) -> float:
        """Molecular-level hash function for nano-scale integrity."""
        hash_obj = hashlib.sha3_256(data)
        hash_bytes = hash_obj.digest()

        byte_values = np.frombuffer(hash_bytes, dtype=np.uint8)

        _, counts = np.unique(byte_values, return_counts=True)
        probabilities = counts / len(byte_values)

        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        normalized_entropy = entropy / 8.0

        return float(normalized_entropy)

    @staticmethod
    def _quantum_dot_checksum(data: bytes) -> float:
        """Quantum dot-inspired checksum."""
        current = data
        checksum = 0.0

        for _i in range(4):
            hash_obj = hashlib.sha3_256(current)
            current = hash_obj.digest()

            byte_sum = sum(current)
            checksum += byte_sum / (256.0 * len(current))

        return checksum / 4.0

    @staticmethod
    def _detect_bit_anomalies(data: bytes) -> float:
        """Detect bit-level anomalies."""
        if len(data) < 2:
            return 0.0

        byte_values = np.frombuffer(data, dtype=np.uint8)

        transitions = np.abs(np.diff(byte_values.astype(int)))

        anomalous_transitions = np.sum(transitions > 200)

        anomaly_rate = anomalous_transitions / len(transitions)

        return float(anomaly_rate)

    def _detect_micro_anomalies(self, data: np.ndarray[Any, Any]) -> float:
        """N Term Enhancement: Detect micro-anomalies at sub-feature level.

        Vectorized (2026-06-11): the former element-by-element loop issued
        one tiny ``np.var`` call per sliding window (~24k calls on a
        1.1k x 21 batch); a strided window view computes all window
        variances in one reduction. Equivalence-pinned by
        ``tests/test_native_acceleration.py``.
        """
        if data.size < 4:
            return 0.0

        data_flat = data.flatten()
        window_size = min(4, len(data_flat) // 2)

        windows = sliding_window_view(data_flat, window_size)
        if windows.shape[0] == 0:
            return 0.0

        variance_array = np.var(windows, axis=1)
        variance_changes = np.abs(np.diff(variance_array))

        micro_score = np.mean(variance_changes) / (np.std(variance_array) + 1e-10)

        return float(min(micro_score, 1.0))

    def _dimensional_downsampling_detection(self, data: np.ndarray[Any, Any]) -> float | None:
        """N Term Enhancement: dimensional downsampling for micro-anomaly detection.

        Downsamples to low dimensions to detect subtle micro-patterns.
        """
        data_2d = data.reshape(-1, 1) if data.ndim == 1 else data

        if data_2d.shape[1] < 2:
            return 0.0

        target_dim = max(1, min(3, data_2d.shape[1] // 2))

        try:
            from omni_mercury_engine.ml.mercury_ml import PCA

            pca = PCA(n_components=target_dim)
            downsampled = pca.fit_transform(data_2d)

            reconstructed = pca.inverse_transform(downsampled)

            micro_residuals = np.abs(data_2d - reconstructed)

            residual_threshold = np.percentile(micro_residuals.flatten(), 95)

            micro_anomaly_pixels = np.sum(micro_residuals > residual_threshold)
            total_pixels = micro_residuals.size

            micro_anomaly_rate = micro_anomaly_pixels / total_pixels

            local_concentrations = []
            for i in range(min(5, data_2d.shape[0])):
                row_residuals = micro_residuals[i, :]
                concentration = np.max(row_residuals) / (np.mean(row_residuals) + 1e-10)
                local_concentrations.append(concentration)

            concentration_score = np.mean(local_concentrations) if local_concentrations else 0.0

            final_score = micro_anomaly_rate * 0.6 + min(concentration_score / 10.0, 1.0) * 0.4

            return float(min(final_score, 1.0))

        except Exception as e:
            # Return None (not 0.0) so a failed sub-score is EXCLUDED from the
            # nano blend rather than averaged in as a spurious 0.0 that would
            # systematically depress the fused anomaly score (missed
            # detections). Warn — a persistent failure must be visible, not a
            # silent DEBUG line.
            logger.warning("Dimensional-downsampling nano detection failed; excluding: %s", e)
            return None

    def get_fitted_state(self) -> dict[str, Any] | None:
        """Export the fitted state for checkpoint round-tripping.

        The transient recursive-memory buffer is deliberately *not*
        exported: it is per-thread streaming state, and the engine's
        fusion-feature boundary resets it before every extraction.

        Returns:
            Mapping with the baseline pattern, or ``None`` when unfitted.
        """
        if not self._is_fitted or self.baseline_pattern is None:
            return None
        return {"baseline_pattern": np.asarray(self.baseline_pattern, dtype=np.float64)}

    def set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`get_fitted_state`."""
        self.baseline_pattern = np.asarray(state["baseline_pattern"], dtype=np.float64)
        self.clear_memory()
        self._is_fitted = True
