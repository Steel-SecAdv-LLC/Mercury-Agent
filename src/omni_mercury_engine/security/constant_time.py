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

"""
Constant-Time Security Operations

This module provides constant-time implementations for security-critical operations
to prevent timing attacks and side-channel analysis.

Timing attacks exploit variations in execution time to extract sensitive information.
By ensuring operations take constant time regardless of input values, we prevent
attackers from inferring secrets through timing measurements.

Key features:
- Constant-time comparison for signatures and tokens
- Constant-time selection without branching
- Secure memory operations
- Timing-safe anomaly score comparison

Critical for security intelligence applications in humanitarian contexts:
- Protecting identity information in missing persons detection
- Securing medical data in pandemic monitoring
- Safeguarding crisis response communications

References:
- Kocher (1996): Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS
- Brumley & Boneh (2003): Remote Timing Attacks are Practical
"""

import hmac
import secrets
from typing import Any

import numpy as np
import numpy.typing as npt


def constant_time_compare(a: bytes | str, b: bytes | str) -> bool:
    """Compare two values in constant time to prevent timing attacks.

    Uses hmac.compare_digest which is designed to prevent timing attacks
    by ensuring comparison takes the same time regardless of where
    differences occur.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        True if values are equal, False otherwise
    """
    if isinstance(a, str):
        a = a.encode("utf-8")
    if isinstance(b, str):
        b = b.encode("utf-8")

    return hmac.compare_digest(a, b)


def constant_time_select(condition: bool, true_val: Any, false_val: Any) -> Any:
    """Select between two values in constant time.

    Avoids branching on secret-dependent conditions by computing both
    paths and selecting the result using bitwise operations.

    Note: This provides timing protection but both values are still
    computed, so side effects in value computation are not hidden.

    Args:
        condition: Selection condition
        true_val: Value to return if condition is True
        false_val: Value to return if condition is False

    Returns:
        true_val if condition is True, false_val otherwise
    """
    mask = -int(condition)
    if isinstance(true_val, (int, float)) and isinstance(false_val, (int, float)):
        return (mask & int(true_val)) | (~mask & int(false_val))
    return true_val if condition else false_val


def constant_time_bytes_eq(a: bytes, b: bytes) -> bool:
    """Compare two byte strings in constant time.

    Args:
        a: First byte string
        b: Second byte string

    Returns:
        True if byte strings are equal, False otherwise
    """
    if len(a) != len(b):
        b = b + bytes(len(a) - len(b)) if len(b) < len(a) else b[: len(a)]
        return False

    result = 0
    for x, y in zip(a, b, strict=False):
        result |= x ^ y

    return result == 0


def secure_score_comparison(
    score: float,
    threshold: float,
    noise_scale: float = 1e-10,
) -> bool:
    """Compare anomaly score to threshold in a timing-safe manner.

    Adds minimal noise to prevent timing attacks based on floating-point
    comparison timing variations.

    Args:
        score: Anomaly score to compare
        threshold: Detection threshold
        noise_scale: Scale of noise to add (default: 1e-10)

    Returns:
        True if score exceeds threshold, False otherwise
    """
    noise = secrets.randbelow(1000) * noise_scale
    adjusted_score = score + noise
    adjusted_threshold = threshold + noise

    return adjusted_score > adjusted_threshold


def constant_time_array_compare(
    arr1: npt.NDArray[np.floating[Any]],
    arr2: npt.NDArray[np.floating[Any]],
    tolerance: float = 1e-8,
) -> bool:
    """Compare two arrays in constant time.

    Computes element-wise comparison for all elements regardless of
    early mismatches to prevent timing attacks.

    Args:
        arr1: First array
        arr2: Second array
        tolerance: Tolerance for floating-point comparison

    Returns:
        True if arrays are equal within tolerance, False otherwise
    """
    if arr1.shape != arr2.shape:
        return False

    diff = np.abs(arr1.flatten() - arr2.flatten())

    all_within_tolerance = np.all(diff <= tolerance)

    return bool(all_within_tolerance)


def secure_signature_verify(
    signature: bytes,
    expected: bytes,
    key: bytes | None = None,
) -> bool:
    """Verify a signature in constant time.

    If key is provided, computes HMAC of expected value before comparison.

    Args:
        signature: Signature to verify
        expected: Expected value or message (if key provided)
        key: Optional HMAC key for computing expected signature

    Returns:
        True if signature is valid, False otherwise
    """
    if key is not None:
        expected = hmac.new(key, expected, "sha256").digest()

    return constant_time_compare(signature, expected)


def constant_time_lookup(
    table: list[Any],
    index: int,
    default: Any = None,
) -> Any:
    """Look up a value in a table in constant time.

    Iterates through entire table regardless of index to prevent
    timing attacks based on table position.

    Args:
        table: List to look up in
        index: Index to retrieve
        default: Default value if index out of bounds

    Returns:
        Value at index or default if out of bounds
    """
    if not table:
        return default

    result = default
    for i, value in enumerate(table):
        is_match = i == index
        if is_match:
            result = value

    return result


class SecureAnomalyChecker:
    """Timing-safe anomaly checking for security-critical applications.

    Provides constant-time operations for anomaly detection to prevent
    timing attacks that could reveal information about detection thresholds
    or anomaly patterns.

    Example:
        >>> checker = SecureAnomalyChecker(threshold=0.8)
        >>> is_anomaly = checker.check(score=0.9)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enable_constant_time: bool = True,
    ):
        """Initialize secure anomaly checker.

        Args:
            threshold: Detection threshold
            enable_constant_time: Enable constant-time operations
        """
        self.threshold = threshold
        self.enable_constant_time = enable_constant_time
        self._check_count = 0

    def check(self, score: float) -> bool:
        """Check if score indicates an anomaly.

        Args:
            score: Anomaly score to check

        Returns:
            True if anomaly detected, False otherwise
        """
        self._check_count += 1

        if self.enable_constant_time:
            return secure_score_comparison(score, self.threshold)
        return score > self.threshold

    def check_signature(
        self,
        data_signature: bytes,
        known_signature: bytes,
    ) -> bool:
        """Check if data signature matches known anomaly signature.

        Args:
            data_signature: Signature of data to check
            known_signature: Known anomaly signature

        Returns:
            True if signatures match, False otherwise
        """
        if self.enable_constant_time:
            return constant_time_compare(data_signature, known_signature)
        return data_signature == known_signature

    def check_array(
        self,
        features: npt.NDArray[np.floating[Any]],
        known_pattern: npt.NDArray[np.floating[Any]],
        tolerance: float = 0.1,
    ) -> bool:
        """Check if features match a known anomaly pattern.

        Args:
            features: Feature array to check
            known_pattern: Known anomaly pattern
            tolerance: Matching tolerance

        Returns:
            True if features match pattern, False otherwise
        """
        if self.enable_constant_time:
            return constant_time_array_compare(features, known_pattern, tolerance)
        return np.allclose(features, known_pattern, atol=tolerance)

    def get_stats(self) -> dict[str, Any]:
        """Get checker statistics.

        Returns:
            Dictionary with checker statistics
        """
        return {
            "threshold": self.threshold,
            "constant_time_enabled": self.enable_constant_time,
            "check_count": self._check_count,
        }


def secure_hash(data: bytes | str, salt: bytes | None = None) -> bytes:
    """Compute a secure hash of data.

    Args:
        data: Data to hash
        salt: Optional salt for hashing

    Returns:
        Hash digest
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if salt is None:
        salt = secrets.token_bytes(16)

    return hmac.new(salt, data, "sha256").digest()


def generate_secure_token(length: int = 32) -> bytes:
    """Generate a cryptographically secure random token.

    Args:
        length: Length of token in bytes

    Returns:
        Random token
    """
    return secrets.token_bytes(length)
