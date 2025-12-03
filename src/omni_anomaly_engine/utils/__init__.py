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
Utilities subpackage
Enhanced with Black Hole Engine compression and gravitational lensing utilities
"""

import zlib
from typing import Any, Union

import numpy as np

# Make torch optional to support environments without ML dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from omni_anomaly_engine.utils.comm import (
    AsyncMessageQueue,
    Message,
    MessagePriority,
    SimplePubSub,
)
from omni_anomaly_engine.utils.constants import (
    MPMATH_AVAILABLE,
    SYMPY_AVAILABLE,
    MathConstant,
    MathematicalConstants,
    Precision,
    get_constant,
    validate_all_constants_with_sympy,
    validate_constant,
)
from omni_anomaly_engine.utils.logging import (
    ColoredFormatter,
    PerformanceLogger,
    StructuredFormatter,
    configure_logging,
    correlation_context,
    get_correlation_id,
    get_logger,
    log_function_call,
    set_correlation_id,
)
from omni_anomaly_engine.utils.resilience import (
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    GracefulShutdown,
    HealthChecker,
    HealthStatus,
    ShutdownInProgressError,
    retry,
    timeout,
)
from omni_anomaly_engine.utils.rng import (
    DeterministicRNG,
    RNGContext,
    RNGRegistry,
    RNGState,
    ThreadSafeRNGManager,
    get_global_rng,
    get_rng_registry,
    get_thread_local_rng,
    reset_global_rng,
    set_global_seed,
)


def normalize_data(
    data: Union[np.ndarray, "torch.Tensor"],
    method: str = "standard",
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Normalize data using specified method.

    Args:
        data: Input data (numpy array or torch tensor if torch is available)
        method: Normalization method ('standard', 'minmax', 'robust')

    Returns:
        Normalized data
    """
    is_torch = TORCH_AVAILABLE and torch is not None and isinstance(data, torch.Tensor)

    if is_torch:
        data_np = data.cpu().numpy()
    else:
        data_np = data

    if method == "standard":
        mean = np.mean(data_np, axis=0)
        std = np.std(data_np, axis=0) + 1e-6
        normalized = (data_np - mean) / std

    elif method == "minmax":
        min_val = np.min(data_np, axis=0)
        max_val = np.max(data_np, axis=0)
        normalized = (data_np - min_val) / (max_val - min_val + 1e-6)

    elif method == "robust":
        median = np.median(data_np, axis=0)
        iqr = np.percentile(data_np, 75, axis=0) - np.percentile(data_np, 25, axis=0)
        normalized = (data_np - median) / (iqr + 1e-6)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if is_torch:
        return torch.tensor(normalized, dtype=data.dtype, device=data.device)

    return normalized


def compute_complexity(func_code: str) -> int:
    """
    Compute cyclomatic complexity of a function.

    Args:
        func_code: Function source code

    Returns:
        Cyclomatic complexity score
    """
    complexity = 1

    decision_points = [
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "and ",
        "or ",
        "try:",
        "except",
        "with ",
    ]

    for keyword in decision_points:
        complexity += func_code.count(keyword)

    return complexity


def compress_information(
    data: np.ndarray,
    compression_level: int = 9,
) -> tuple[bytes, dict[str, Any]]:
    """
    Extreme data compression inspired by black hole information density
    """
    original_shape = data.shape
    original_dtype = str(data.dtype)
    original_size = data.nbytes

    data_bytes = data.tobytes()

    compressed = zlib.compress(data_bytes, level=compression_level)

    metadata = {
        "shape": original_shape,
        "dtype": original_dtype,
        "original_size": original_size,
        "compressed_size": len(compressed),
        "compression_ratio": original_size / len(compressed),
    }

    return compressed, metadata


def decompress_information(
    compressed: bytes,
    metadata: dict[str, Any],
) -> np.ndarray:
    """
    Decompress data compressed by compress_information
    """
    decompressed = zlib.decompress(compressed)

    dtype = np.dtype(metadata["dtype"])
    shape = tuple(metadata["shape"])

    data = np.frombuffer(decompressed, dtype=dtype).reshape(shape)

    return data


def gravitational_lensing(
    signal: np.ndarray,
    amplification_factor: float = 3.0,
) -> np.ndarray:
    """
    Amplify weak signals using gravitational lensing analogy
    """
    signal_strength = np.abs(signal)
    median_strength = np.median(signal_strength)

    weak_signal_mask = signal_strength < median_strength

    amplified = signal.copy()
    amplified[weak_signal_mask] *= amplification_factor

    return amplified


def detect_singularity(
    data: np.ndarray,
    threshold_percentile: float = 99.0,
) -> dict[str, Any]:
    """
    Detect singularity points (critical decision points) in data
    """
    data_flat = data.flatten()

    threshold = np.percentile(np.abs(data_flat), threshold_percentile)

    singularity_mask = np.abs(data_flat) >= threshold
    singularity_indices = np.where(singularity_mask)[0]
    singularity_values = data_flat[singularity_indices]

    if len(singularity_indices) > 0:
        singularity_strength = np.mean(np.abs(singularity_values)) / (
            np.mean(np.abs(data_flat)) + 1e-10
        )
    else:
        singularity_strength = 0.0

    return {
        "singularity_detected": len(singularity_indices) > 0,
        "singularity_count": len(singularity_indices),
        "singularity_indices": singularity_indices,
        "singularity_values": singularity_values,
        "singularity_strength": singularity_strength,
    }


def compute_time_dilation(
    priority_scores: np.ndarray,
    mass_factor: float = 1.0,
) -> np.ndarray:
    """
    Compute time dilation factor for priority weighting
    """
    normalized_scores = priority_scores / (np.max(priority_scores) + 1e-10)

    schwarzschild_radius = 2.0 * mass_factor

    r_values = 1.0 + (1.0 - normalized_scores) * schwarzschild_radius

    ratio = schwarzschild_radius / r_values
    ratio = np.clip(ratio, 0.0, 0.99)

    time_dilation = 1.0 / np.sqrt(1.0 - ratio)

    time_dilation = np.clip(time_dilation, 1.0, 10.0)

    return time_dilation


__all__ = [
    # Communication utilities
    "Message",
    "MessagePriority",
    "AsyncMessageQueue",
    "SimplePubSub",
    # RNG utilities
    "DeterministicRNG",
    "RNGState",
    "RNGRegistry",
    "RNGContext",
    "ThreadSafeRNGManager",
    "get_global_rng",
    "set_global_seed",
    "reset_global_rng",
    "get_thread_local_rng",
    "get_rng_registry",
    # Constants
    "MathematicalConstants",
    "MathConstant",
    "Precision",
    "get_constant",
    "validate_constant",
    "validate_all_constants_with_sympy",
    "SYMPY_AVAILABLE",
    "MPMATH_AVAILABLE",
    # Logging utilities
    "get_logger",
    "configure_logging",
    "correlation_context",
    "get_correlation_id",
    "set_correlation_id",
    "StructuredFormatter",
    "ColoredFormatter",
    "PerformanceLogger",
    "log_function_call",
    # Resilience utilities
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "retry",
    "GracefulShutdown",
    "ShutdownInProgressError",
    "Bulkhead",
    "BulkheadFullError",
    "HealthChecker",
    "HealthStatus",
    "timeout",
    # Data utilities
    "normalize_data",
    "compute_complexity",
    "compress_information",
    "decompress_information",
    "gravitational_lensing",
    "detect_singularity",
    "compute_time_dilation",
]
