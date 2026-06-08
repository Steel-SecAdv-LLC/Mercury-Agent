# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for constant-time security operations."""

import numpy as np

from omni_mercury_engine.security.constant_time import (
    SecureAnomalyChecker,
    constant_time_array_compare,
    constant_time_bytes_eq,
    constant_time_compare,
    constant_time_lookup,
    constant_time_select,
    generate_secure_token,
    secure_hash,
    secure_score_comparison,
    secure_signature_verify,
)


class TestConstantTimeCompare:
    """Tests for constant_time_compare function."""

    def test_equal_strings(self) -> None:
        """Test comparison of equal strings."""
        assert constant_time_compare("hello", "hello") is True

    def test_unequal_strings(self) -> None:
        """Test comparison of unequal strings."""
        assert constant_time_compare("hello", "world") is False

    def test_equal_bytes(self) -> None:
        """Test comparison of equal bytes."""
        assert constant_time_compare(b"hello", b"hello") is True

    def test_unequal_bytes(self) -> None:
        """Test comparison of unequal bytes."""
        assert constant_time_compare(b"hello", b"world") is False

    def test_mixed_string_bytes(self) -> None:
        """Test comparison of string and bytes."""
        assert constant_time_compare("hello", b"hello") is True

    def test_empty_values(self) -> None:
        """Test comparison of empty values."""
        assert constant_time_compare("", "") is True
        assert constant_time_compare(b"", b"") is True

    def test_different_lengths(self) -> None:
        """Test comparison of different length values."""
        assert constant_time_compare("hello", "hello world") is False


class TestConstantTimeSelect:
    """Tests for constant_time_select function."""

    def test_select_true_int(self) -> None:
        """Test selection with True condition for integers."""
        result = constant_time_select(True, 10, 20)
        assert result == 10

    def test_select_false_int(self) -> None:
        """Test selection with False condition for integers."""
        result = constant_time_select(False, 10, 20)
        assert result == 20

    def test_select_true_float(self) -> None:
        """Test selection with True condition for floats."""
        result = constant_time_select(True, 1.5, 2.5)
        assert result == 1

    def test_select_false_float(self) -> None:
        """Test selection with False condition for floats."""
        result = constant_time_select(False, 1.5, 2.5)
        assert result == 2

    def test_select_true_string(self) -> None:
        """Test selection with True condition for strings."""
        result = constant_time_select(True, "yes", "no")
        assert result == "yes"

    def test_select_false_string(self) -> None:
        """Test selection with False condition for strings."""
        result = constant_time_select(False, "yes", "no")
        assert result == "no"


class TestConstantTimeBytesEq:
    """Tests for constant_time_bytes_eq function."""

    def test_equal_bytes(self) -> None:
        """Test comparison of equal byte strings."""
        assert constant_time_bytes_eq(b"hello", b"hello") is True

    def test_unequal_bytes(self) -> None:
        """Test comparison of unequal byte strings."""
        assert constant_time_bytes_eq(b"hello", b"world") is False

    def test_different_lengths(self) -> None:
        """Test comparison of different length byte strings."""
        assert constant_time_bytes_eq(b"hello", b"hi") is False

    def test_empty_bytes(self) -> None:
        """Test comparison of empty byte strings."""
        assert constant_time_bytes_eq(b"", b"") is True


class TestSecureScoreComparison:
    """Tests for secure_score_comparison function."""

    def test_score_above_threshold(self) -> None:
        """Test score clearly above threshold."""
        assert secure_score_comparison(0.9, 0.5) is True

    def test_score_below_threshold(self) -> None:
        """Test score clearly below threshold."""
        assert secure_score_comparison(0.3, 0.5) is False

    def test_score_at_threshold(self) -> None:
        """Test score at threshold (edge case)."""
        result = secure_score_comparison(0.5, 0.5)
        assert isinstance(result, bool)

    def test_custom_noise_scale(self) -> None:
        """Test with custom noise scale."""
        result = secure_score_comparison(0.9, 0.5, noise_scale=1e-12)
        assert result is True


class TestConstantTimeArrayCompare:
    """Tests for constant_time_array_compare function."""

    def test_equal_arrays(self) -> None:
        """Test comparison of equal arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        assert constant_time_array_compare(arr1, arr2) is True

    def test_unequal_arrays(self) -> None:
        """Test comparison of unequal arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        assert constant_time_array_compare(arr1, arr2) is False

    def test_different_shapes(self) -> None:
        """Test comparison of arrays with different shapes."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0])
        assert constant_time_array_compare(arr1, arr2) is False

    def test_within_tolerance(self) -> None:
        """Test comparison within tolerance."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0 + 1e-9, 2.0, 3.0])
        assert constant_time_array_compare(arr1, arr2, tolerance=1e-8) is True

    def test_outside_tolerance(self) -> None:
        """Test comparison outside tolerance."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.1, 2.0, 3.0])
        assert constant_time_array_compare(arr1, arr2, tolerance=1e-8) is False


class TestSecureSignatureVerify:
    """Tests for secure_signature_verify function."""

    def test_matching_signatures(self) -> None:
        """Test verification of matching signatures."""
        sig = b"test_signature"
        assert secure_signature_verify(sig, sig) is True

    def test_non_matching_signatures(self) -> None:
        """Test verification of non-matching signatures."""
        sig1 = b"signature1"
        sig2 = b"signature2"
        assert secure_signature_verify(sig1, sig2) is False

    def test_with_hmac_key(self) -> None:
        """Test verification with HMAC key."""
        import hmac

        key = b"secret_key"
        message = b"test_message"
        expected_sig = hmac.new(key, message, "sha256").digest()
        assert secure_signature_verify(expected_sig, message, key) is True


class TestConstantTimeLookup:
    """Tests for constant_time_lookup function."""

    def test_valid_index(self) -> None:
        """Test lookup with valid index."""
        table = ["a", "b", "c", "d"]
        assert constant_time_lookup(table, 2) == "c"

    def test_first_index(self) -> None:
        """Test lookup with first index."""
        table = ["a", "b", "c", "d"]
        assert constant_time_lookup(table, 0) == "a"

    def test_last_index(self) -> None:
        """Test lookup with last index."""
        table = ["a", "b", "c", "d"]
        assert constant_time_lookup(table, 3) == "d"

    def test_out_of_bounds(self) -> None:
        """Test lookup with out of bounds index."""
        table = ["a", "b", "c"]
        assert constant_time_lookup(table, 10, default="default") == "default"

    def test_empty_table(self) -> None:
        """Test lookup with empty table."""
        assert constant_time_lookup([], 0, default="default") == "default"

    def test_negative_index(self) -> None:
        """Test lookup with negative index."""
        table = ["a", "b", "c"]
        assert constant_time_lookup(table, -1, default="default") == "default"


class TestSecureAnomalyChecker:
    """Tests for SecureAnomalyChecker class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        checker = SecureAnomalyChecker()
        assert checker.threshold == 0.5
        assert checker.enable_constant_time is True

    def test_init_custom_threshold(self) -> None:
        """Test initialization with custom threshold."""
        checker = SecureAnomalyChecker(threshold=0.8)
        assert checker.threshold == 0.8

    def test_check_anomaly_detected(self) -> None:
        """Test anomaly detection when score exceeds threshold."""
        checker = SecureAnomalyChecker(threshold=0.5)
        assert checker.check(0.9) is True

    def test_check_no_anomaly(self) -> None:
        """Test no anomaly when score below threshold."""
        checker = SecureAnomalyChecker(threshold=0.5)
        assert checker.check(0.3) is False

    def test_check_without_constant_time(self) -> None:
        """Test check without constant time enabled."""
        checker = SecureAnomalyChecker(threshold=0.5, enable_constant_time=False)
        assert checker.check(0.9) is True
        assert checker.check(0.3) is False

    def test_check_signature_match(self) -> None:
        """Test signature matching."""
        checker = SecureAnomalyChecker()
        sig = b"test_signature"
        assert checker.check_signature(sig, sig) is True

    def test_check_signature_no_match(self) -> None:
        """Test signature non-matching."""
        checker = SecureAnomalyChecker()
        assert checker.check_signature(b"sig1", b"sig2") is False

    def test_check_array_match(self) -> None:
        """Test array pattern matching."""
        checker = SecureAnomalyChecker()
        features = np.array([1.0, 2.0, 3.0])
        pattern = np.array([1.0, 2.0, 3.0])
        assert checker.check_array(features, pattern) is True

    def test_check_array_no_match(self) -> None:
        """Test array pattern non-matching."""
        checker = SecureAnomalyChecker()
        features = np.array([1.0, 2.0, 3.0])
        pattern = np.array([4.0, 5.0, 6.0])
        assert checker.check_array(features, pattern) is False

    def test_check_array_without_constant_time(self) -> None:
        """Test array check without constant time."""
        checker = SecureAnomalyChecker(enable_constant_time=False)
        features = np.array([1.0, 2.0, 3.0])
        pattern = np.array([1.0, 2.0, 3.0])
        assert checker.check_array(features, pattern) is True

    def test_get_stats(self) -> None:
        """Test getting checker statistics."""
        checker = SecureAnomalyChecker(threshold=0.7)
        checker.check(0.8)
        checker.check(0.5)
        stats = checker.get_stats()
        assert stats["threshold"] == 0.7
        assert stats["constant_time_enabled"] is True
        assert stats["check_count"] == 2


class TestSecureHash:
    """Tests for secure_hash function."""

    def test_hash_string(self) -> None:
        """Test hashing a string."""
        result = secure_hash("test_data")
        assert isinstance(result, bytes)
        assert len(result) == 32

    def test_hash_bytes(self) -> None:
        """Test hashing bytes."""
        result = secure_hash(b"test_data")
        assert isinstance(result, bytes)
        assert len(result) == 32

    def test_hash_with_salt(self) -> None:
        """Test hashing with custom salt."""
        salt = b"custom_salt_1234"
        result = secure_hash("test_data", salt=salt)
        assert isinstance(result, bytes)
        assert len(result) == 32

    def test_different_salts_different_hashes(self) -> None:
        """Test that different salts produce different hashes."""
        salt1 = b"salt1___________"
        salt2 = b"salt2___________"
        hash1 = secure_hash("test_data", salt=salt1)
        hash2 = secure_hash("test_data", salt=salt2)
        assert hash1 != hash2


class TestGenerateSecureToken:
    """Tests for generate_secure_token function."""

    def test_default_length(self) -> None:
        """Test generating token with default length."""
        token = generate_secure_token()
        assert isinstance(token, bytes)
        assert len(token) == 32

    def test_custom_length(self) -> None:
        """Test generating token with custom length."""
        token = generate_secure_token(length=64)
        assert len(token) == 64

    def test_tokens_are_unique(self) -> None:
        """Test that generated tokens are unique."""
        tokens = [generate_secure_token() for _ in range(100)]
        assert len(set(tokens)) == 100
