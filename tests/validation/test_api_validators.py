# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the API input-validation surface (``api_validators``).

Covers:
- ``ValidationError``/``ValidationResult`` serialization contracts, including
  the 100-character value truncation in API error payloads
- ``ValidationConfig`` defaults that define the public size/value limits
- ``InputSanitizer``: HTML-entity escaping, truncate-then-escape ordering,
  null-byte and control-character stripping, non-string coercion, and
  recursive dict/list sanitization with depth capping
- ``DataArrayValidator.validate_univariate`` / ``validate_multivariate``:
  happy paths plus every triggerable rejection — non-numeric and ragged
  input, wrong dimensionality, exact boundaries around the configured size
  limits, NaN/Inf ratio thresholds (error vs. warning side), value-range
  limits, and zero-variance warnings
- ``ParameterValidator``: sensitivity range/type checks (including NaN),
  feature-name repair and truncation, adversarial injection-shaped names
  (XSS, SQL, template, JNDI payloads), duplicate detection including
  post-sanitization collisions, and domain normalization
- ``APIRequestValidator`` and the module-level convenience functions:
  error/warning aggregation, feature-count constraint, config threading,
  and tolerance of unknown kwargs
- Every reachable ``ValidationErrorType``.  ``MISSING_REQUIRED`` is defined
  but no code path in the module constructs it, so it cannot be triggered
  legitimately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from omni_mercury_engine.validation.api_validators import (
    APIRequestValidator,
    DataArrayValidator,
    InputSanitizer,
    ParameterValidator,
    ValidationConfig,
    ValidationError,
    ValidationErrorType,
    ValidationResult,
    sanitize_input,
    validate_multivariate,
    validate_univariate,
)

NAN = float("nan")
INF = float("inf")


def _error_types(result: ValidationResult) -> list[ValidationErrorType]:
    """Extract the error types from a validation result, in order."""
    return [e.error_type for e in result.errors]


# =============================================================================
# ValidationError / ValidationResult serialization
# =============================================================================


class TestValidationErrorSerialization:
    """``ValidationError.to_dict`` — the API-facing error payload."""

    def test_to_dict_carries_all_fields(self) -> None:
        err = ValidationError(
            error_type=ValidationErrorType.INVALID_RANGE,
            field="sensitivity",
            message="out of range",
            value=1.5,
            constraint="[0.0, 1.0]",
        )
        assert err.to_dict() == {
            "error_type": "invalid_range",
            "field": "sensitivity",
            "message": "out of range",
            "value": "1.5",
            "constraint": "[0.0, 1.0]",
        }

    def test_to_dict_truncates_value_to_100_chars(self) -> None:
        err = ValidationError(
            error_type=ValidationErrorType.INVALID_VALUE,
            field="data",
            message="oversized",
            value="x" * 500,
        )
        payload = err.to_dict()
        assert payload["value"] == "x" * 100

    def test_to_dict_none_value_stays_none(self) -> None:
        err = ValidationError(
            error_type=ValidationErrorType.INVALID_TYPE,
            field="data",
            message="bad type",
        )
        assert err.to_dict()["value"] is None
        assert err.to_dict()["constraint"] is None


class TestValidationResultSerialization:
    """``ValidationResult`` defaults and ``to_dict`` shape."""

    def test_defaults_are_empty(self) -> None:
        result = ValidationResult(is_valid=True)
        assert result.errors == []
        assert result.warnings == []
        assert result.sanitized_data is None

    def test_to_dict_serializes_nested_errors(self) -> None:
        err = ValidationError(
            error_type=ValidationErrorType.DUPLICATE_VALUE,
            field="features[1]",
            message="dup",
            value="a",
        )
        result = ValidationResult(is_valid=False, errors=[err], warnings=["heads up"])
        payload = result.to_dict()
        assert payload["is_valid"] is False
        assert payload["warnings"] == ["heads up"]
        assert payload["errors"] == [err.to_dict()]
        # sanitized_data is intentionally NOT part of the API payload
        assert "sanitized_data" not in payload


class TestValidationConfigDefaults:
    """The documented default limits are the public contract of the module."""

    def test_default_limits(self) -> None:
        config = ValidationConfig()
        assert config.max_data_points == 100000
        assert config.max_features == 1000
        assert config.max_string_length == 256
        assert config.max_array_depth == 3
        assert config.min_value == -1e15
        assert config.max_value == 1e15
        assert config.max_nan_ratio == 0.1
        assert config.max_inf_ratio == 0.01
        assert config.enable_domain_validation is True
        assert config.strict_mode is False

    def test_forbidden_patterns_cover_known_injection_families(self) -> None:
        config = ValidationConfig()
        assert r"<script" in config.forbidden_patterns
        assert r"javascript:" in config.forbidden_patterns
        assert r"union.*select" in config.forbidden_patterns
        assert r"\$\{" in config.forbidden_patterns
        assert r"\{\{" in config.forbidden_patterns


# =============================================================================
# InputSanitizer
# =============================================================================


class TestSanitizeString:
    """``InputSanitizer.sanitize_string`` — escaping, truncation, stripping."""

    @pytest.mark.parametrize(
        ("raw", "escaped"),
        [
            ("&", "&amp;"),
            ("<", "&lt;"),
            (">", "&gt;"),
            ('"', "&quot;"),
            ("'", "&#x27;"),
        ],
    )
    def test_escapes_each_html_entity(self, raw: str, escaped: str) -> None:
        assert InputSanitizer.sanitize_string(raw) == escaped

    def test_xss_payload_fully_neutralized(self) -> None:
        out = InputSanitizer.sanitize_string("<script>alert('xss')</script>")
        assert "<" not in out
        assert ">" not in out
        assert "'" not in out
        assert out == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"

    def test_ampersand_escaped_first_no_double_escaping(self) -> None:
        # "&" must be escaped before "<" so the inserted "&lt;" is not re-escaped.
        assert InputSanitizer.sanitize_string("&<") == "&amp;&lt;"

    def test_truncates_input_before_escaping(self) -> None:
        # Truncation applies to the RAW input; entity expansion can therefore
        # legitimately exceed max_length in the output.
        out = InputSanitizer.sanitize_string("<" * 10, max_length=5)
        assert out == "&lt;" * 5

    def test_removes_null_bytes(self) -> None:
        assert InputSanitizer.sanitize_string("a\x00b\x00c") == "abc"

    def test_strips_control_chars_keeps_newline_and_tab(self) -> None:
        assert InputSanitizer.sanitize_string("a\x01b\rc\nd\te") == "abc\nd\te"

    def test_non_string_input_coerced_via_str(self) -> None:
        # Signature says str, but the sanitizer defends against non-strings.
        assert InputSanitizer.sanitize_string(cast("str", 123)) == "123"

    def test_empty_string_passthrough(self) -> None:
        assert InputSanitizer.sanitize_string("") == ""

    def test_benign_string_unchanged(self) -> None:
        assert InputSanitizer.sanitize_string("sensor_temp.1-a") == "sensor_temp.1-a"


class TestSanitizeDictAndList:
    """Recursive sanitization with depth capping."""

    def test_values_and_keys_sanitized(self) -> None:
        out = InputSanitizer.sanitize_dict({"<key>": "<val>"})
        assert out == {"&lt;key&gt;": "&lt;val&gt;"}

    def test_non_string_scalars_pass_through(self) -> None:
        data: dict[str, Any] = {"i": 1, "f": 2.5, "n": None, "b": True}
        assert InputSanitizer.sanitize_dict(data) == data

    def test_nested_dict_and_list_sanitized(self) -> None:
        data: dict[str, Any] = {"outer": {"inner": "<x>"}, "items": ["<y>", 7]}
        out = InputSanitizer.sanitize_dict(data)
        assert out == {"outer": {"inner": "&lt;x&gt;"}, "items": ["&lt;y&gt;", 7]}

    def test_dict_nested_beyond_max_depth_is_emptied(self) -> None:
        data: dict[str, Any] = {"l1": {"l2": {"l3": {"l4": "dropped"}}}}
        out = InputSanitizer.sanitize_dict(data, max_depth=3)
        assert out == {"l1": {"l2": {"l3": {}}}}

    def test_dict_with_zero_depth_returns_empty(self) -> None:
        assert InputSanitizer.sanitize_dict({"a": 1}, max_depth=0) == {}

    def test_list_strings_escaped_and_nested_dicts_recursed(self) -> None:
        out = InputSanitizer.sanitize_list(["<a>", {"k": "<b>"}, 3])
        assert out == ["&lt;a&gt;", {"k": "&lt;b&gt;"}, 3]

    def test_list_nested_beyond_max_depth_is_emptied(self) -> None:
        data: list[Any] = [[[["dropped"]]]]
        out = InputSanitizer.sanitize_list(data, max_depth=3)
        assert out == [[[[]]]]

    def test_list_with_zero_depth_returns_empty(self) -> None:
        assert InputSanitizer.sanitize_list(["a"], max_depth=0) == []

    def test_sanitize_input_convenience_wraps_sanitize_dict(self) -> None:
        out = sanitize_input({"q": "1; DROP TABLE users --", "d": {"x": "<i>"}})
        assert out == {"q": "1; DROP TABLE users --", "d": {"x": "&lt;i&gt;"}}


# =============================================================================
# DataArrayValidator — univariate
# =============================================================================


class TestValidateUnivariate:
    """1-D data validation: conversion, shape, size, NaN/Inf, range."""

    def test_valid_list_passes_and_is_converted_to_float64(self) -> None:
        result = DataArrayValidator().validate_univariate([1, 2, 3, 4])
        assert result.is_valid
        assert result.errors == []
        assert isinstance(result.sanitized_data, np.ndarray)
        assert result.sanitized_data.dtype == np.float64
        assert result.sanitized_data.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_valid_numpy_array_passes(self) -> None:
        arr = np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
        result = DataArrayValidator().validate_univariate(arr)
        assert result.is_valid
        assert result.warnings == []

    def test_non_numeric_strings_rejected_as_invalid_type(self) -> None:
        data = cast("list[float]", ["a", "b", "c"])
        result = DataArrayValidator().validate_univariate(data)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_TYPE]
        assert result.sanitized_data is None

    def test_ragged_nested_list_rejected_as_invalid_type(self) -> None:
        data = cast("list[float]", [[1.0, 2.0], [3.0]])
        result = DataArrayValidator().validate_univariate(data)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_TYPE]

    def test_2d_array_rejected_as_invalid_format(self) -> None:
        result = DataArrayValidator().validate_univariate(np.zeros((4, 2)))
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_FORMAT]
        assert "2D" in result.errors[0].message
        assert result.errors[0].value == (4, 2)

    def test_scalar_rejected_as_invalid_format(self) -> None:
        scalar = cast("NDArray[np.float64]", np.float64(5.0))
        result = DataArrayValidator().validate_univariate(scalar)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_FORMAT]
        assert "0D" in result.errors[0].message

    def test_length_at_max_data_points_allowed(self) -> None:
        validator = DataArrayValidator(ValidationConfig(max_data_points=10))
        result = validator.validate_univariate([float(i) for i in range(10)])
        assert result.is_valid

    def test_length_above_max_data_points_rejected(self) -> None:
        validator = DataArrayValidator(ValidationConfig(max_data_points=10))
        result = validator.validate_univariate([float(i) for i in range(11)])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.SIZE_LIMIT_EXCEEDED]
        assert result.errors[0].value == 11
        assert result.errors[0].constraint == 10

    def test_minimum_three_points_boundary(self) -> None:
        validator = DataArrayValidator()
        assert validator.validate_univariate([1.0, 2.0, 3.0]).is_valid
        result = validator.validate_univariate([1.0, 2.0])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.SIZE_LIMIT_EXCEEDED]
        assert result.errors[0].constraint == 3

    def test_empty_input_rejected_without_crash(self) -> None:
        result = DataArrayValidator().validate_univariate([])
        assert not result.is_valid
        assert ValidationErrorType.SIZE_LIMIT_EXCEEDED in _error_types(result)

    def test_nan_ratio_above_limit_rejected(self) -> None:
        # 2 NaN in 10 points = 20% > default 10% limit
        data = [float(i) for i in range(8)] + [NAN, NAN]
        result = DataArrayValidator().validate_univariate(data)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_VALUE]
        assert result.errors[0].constraint == 0.1

    def test_nan_ratio_at_limit_is_warning_only(self) -> None:
        # 1 NaN in 10 points: ratio 1/(10+1e-10) is just under the 10% limit
        data = [float(i) for i in range(9)] + [NAN]
        result = DataArrayValidator().validate_univariate(data)
        assert result.is_valid
        assert any("1 NaN" in w for w in result.warnings)

    def test_inf_ratio_above_limit_rejected(self) -> None:
        # 1 Inf in 10 points = 10% > default 1% limit
        data = [float(i) for i in range(9)] + [INF]
        result = DataArrayValidator().validate_univariate(data)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_VALUE]
        assert "Inf ratio" in result.errors[0].message

    def test_inf_ratio_at_limit_is_warning_only(self) -> None:
        # 1 Inf in 100 points: ratio just under the 1% limit
        data = [float(i) for i in range(99)] + [INF]
        result = DataArrayValidator().validate_univariate(data)
        assert result.is_valid
        assert any("1 Inf" in w for w in result.warnings)

    def test_value_above_max_rejected(self) -> None:
        result = DataArrayValidator().validate_univariate([0.0, 1.0, 2e15])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_RANGE]
        assert result.errors[0].constraint == 1e15

    def test_value_below_min_rejected(self) -> None:
        result = DataArrayValidator().validate_univariate([-2e15, 0.0, 1.0])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_RANGE]
        assert result.errors[0].constraint == -1e15

    def test_both_range_violations_reported_together(self) -> None:
        result = DataArrayValidator().validate_univariate([-2e15, 0.0, 2e15])
        assert _error_types(result) == [
            ValidationErrorType.INVALID_RANGE,
            ValidationErrorType.INVALID_RANGE,
        ]

    def test_constant_data_warns_zero_variance(self) -> None:
        result = DataArrayValidator().validate_univariate([5.0, 5.0, 5.0, 5.0])
        assert result.is_valid
        assert any("zero variance" in w for w in result.warnings)

    def test_all_nan_data_rejected_without_range_errors(self) -> None:
        result = DataArrayValidator().validate_univariate([NAN, NAN, NAN])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_VALUE]

    def test_custom_nan_ratio_config_honored(self) -> None:
        validator = DataArrayValidator(ValidationConfig(max_nan_ratio=0.5))
        data = [1.0, 2.0, 3.0, NAN, NAN]  # 40% NaN, under the custom 50%
        result = validator.validate_univariate(data)
        assert result.is_valid


# =============================================================================
# DataArrayValidator — multivariate
# =============================================================================


class TestValidateMultivariate:
    """2-D data validation: shape, per-feature NaN, size/feature limits."""

    def test_valid_2d_data_passes(self) -> None:
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        result = DataArrayValidator().validate_multivariate(data)
        assert result.is_valid
        assert result.warnings == []
        assert result.sanitized_data.shape == (4, 2)

    def test_1d_input_rejected_as_invalid_format(self) -> None:
        result = DataArrayValidator().validate_multivariate(np.arange(5.0))
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_FORMAT]
        assert "got 1D" in result.errors[0].message

    def test_3d_input_rejected_as_invalid_format(self) -> None:
        result = DataArrayValidator().validate_multivariate(np.zeros((3, 3, 3)))
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_FORMAT]

    def test_non_numeric_rejected_as_invalid_type(self) -> None:
        data = cast("list[list[float]]", [["a", "b"], ["c", "d"], ["e", "f"]])
        result = DataArrayValidator().validate_multivariate(data)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_TYPE]

    def test_sample_count_boundary(self) -> None:
        validator = DataArrayValidator(ValidationConfig(max_data_points=5))
        ok = np.arange(10.0).reshape(5, 2)
        assert validator.validate_multivariate(ok).is_valid
        over = np.arange(12.0).reshape(6, 2)
        result = validator.validate_multivariate(over)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.SIZE_LIMIT_EXCEEDED]
        assert result.errors[0].constraint == 5

    def test_feature_count_boundary(self) -> None:
        validator = DataArrayValidator(ValidationConfig(max_features=3))
        ok = np.arange(9.0).reshape(3, 3)
        assert validator.validate_multivariate(ok).is_valid
        over = np.arange(12.0).reshape(3, 4)
        result = validator.validate_multivariate(over)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.SIZE_LIMIT_EXCEEDED]
        assert "Feature count" in result.errors[0].message

    def test_fewer_than_three_samples_rejected(self) -> None:
        result = DataArrayValidator().validate_multivariate([[1.0, 2.0], [3.0, 4.0]])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.SIZE_LIMIT_EXCEEDED]
        assert result.errors[0].constraint == 3

    def test_nan_column_flagged_per_feature(self) -> None:
        arr = np.arange(15.0).reshape(5, 3)
        arr[:, 1] = np.nan
        result = DataArrayValidator().validate_multivariate(arr)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_VALUE]
        assert result.errors[0].field == "data[feature_1]"

    def test_inf_ratio_above_limit_rejected(self) -> None:
        arr = np.arange(6.0).reshape(3, 2)
        arr[0, 0] = np.inf  # 1/6 ≈ 17% > 1% limit
        result = DataArrayValidator().validate_multivariate(arr)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_VALUE]
        assert "Inf ratio" in result.errors[0].message

    def test_inf_ratio_under_limit_accepted_silently(self) -> None:
        # Unlike the univariate path, multivariate emits NO warning for a
        # sub-threshold Inf count — the data simply passes.
        arr = np.arange(200.0).reshape(100, 2)
        arr[0, 0] = np.inf  # 1/200 = 0.5% < 1% limit
        result = DataArrayValidator().validate_multivariate(arr)
        assert result.is_valid
        assert result.errors == []

    def test_all_nan_data_rejected_without_range_errors(self) -> None:
        arr = np.full((3, 2), np.nan)
        result = DataArrayValidator().validate_multivariate(arr)
        assert not result.is_valid
        # One per-feature NaN error per column; no range/variance crashes
        # even though there are zero finite values.
        assert _error_types(result) == [
            ValidationErrorType.INVALID_VALUE,
            ValidationErrorType.INVALID_VALUE,
        ]
        assert result.errors[0].field == "data[feature_0]"
        assert result.errors[1].field == "data[feature_1]"

    def test_range_violations_reported(self) -> None:
        data = [[-2e15, 2e15], [0.0, 0.0], [1.0, 1.0]]
        result = DataArrayValidator().validate_multivariate(data)
        assert not result.is_valid
        assert _error_types(result) == [
            ValidationErrorType.INVALID_RANGE,
            ValidationErrorType.INVALID_RANGE,
        ]

    def test_constant_feature_warns_with_index(self) -> None:
        data = [[7.0, 1.0], [7.0, 2.0], [7.0, 3.0], [7.0, 4.0]]
        result = DataArrayValidator().validate_multivariate(data)
        assert result.is_valid
        assert any("zero variance: [0]" in w for w in result.warnings)


# =============================================================================
# ParameterValidator — sensitivity
# =============================================================================


class TestValidateSensitivity:
    """Sensitivity: [0, 1] range check with a 0.5 default."""

    def test_none_defaults_to_half(self) -> None:
        result = ParameterValidator().validate_sensitivity(None)
        assert result.is_valid
        assert result.sanitized_data == 0.5

    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 1.0])
    def test_in_range_values_accepted(self, value: float) -> None:
        result = ParameterValidator().validate_sensitivity(value)
        assert result.is_valid
        assert result.sanitized_data == value

    def test_int_accepted_and_coerced_to_float(self) -> None:
        result = ParameterValidator().validate_sensitivity(1)
        assert result.is_valid
        assert result.sanitized_data == 1.0
        assert isinstance(result.sanitized_data, float)

    @pytest.mark.parametrize("value", [-0.1, 1.5, 100.0])
    def test_out_of_range_rejected(self, value: float) -> None:
        result = ParameterValidator().validate_sensitivity(value)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_RANGE]
        assert result.errors[0].constraint == "[0.0, 1.0]"
        # sanitized_data passes the raw number through; callers must gate on
        # is_valid before consuming it.
        assert result.sanitized_data == value

    def test_nan_rejected_as_invalid_range(self) -> None:
        result = ParameterValidator().validate_sensitivity(NAN)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_RANGE]

    def test_non_numeric_rejected_with_default_fallback(self) -> None:
        result = ParameterValidator().validate_sensitivity(cast("float", "high"))
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_TYPE]
        assert result.sanitized_data == 0.5

    def test_bool_accepted_as_int_subclass(self) -> None:
        # bool is a subclass of int in Python, so True validates as 1.0.
        # Documented behavior, not an endorsement.
        result = ParameterValidator().validate_sensitivity(True)
        assert result.is_valid
        assert result.sanitized_data == 1.0


# =============================================================================
# ParameterValidator — feature names
# =============================================================================


class TestValidateFeatureNames:
    """Feature names: format repair, injection detection, duplicates."""

    def test_none_passthrough(self) -> None:
        result = ParameterValidator().validate_feature_names(None)
        assert result.is_valid
        assert result.sanitized_data is None

    def test_valid_names_unchanged(self) -> None:
        names = ["temp", "pressure_1", "sensor.reading-2", "_hidden"]
        result = ParameterValidator().validate_feature_names(names)
        assert result.is_valid
        assert result.sanitized_data == names
        assert result.warnings == []

    def test_empty_list_valid(self) -> None:
        result = ParameterValidator().validate_feature_names([])
        assert result.is_valid
        assert result.sanitized_data == []

    def test_non_string_name_rejected_and_excluded(self) -> None:
        names = cast("list[str]", ["ok", 42])
        result = ParameterValidator().validate_feature_names(names)
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_TYPE]
        assert result.errors[0].field == "features[1]"
        assert result.sanitized_data == ["ok"]

    def test_leading_digit_name_replaced_with_placeholder(self) -> None:
        result = ParameterValidator().validate_feature_names(["123bad"])
        assert result.is_valid
        assert result.sanitized_data == ["feature_0"]
        assert any("sanitized" in w for w in result.warnings)

    def test_illegal_characters_replaced_with_underscores(self) -> None:
        result = ParameterValidator().validate_feature_names(["my name!"])
        assert result.is_valid
        assert result.sanitized_data == ["my_name_"]
        assert any("sanitized" in w for w in result.warnings)

    def test_oversized_plain_name_truncated_to_limit(self) -> None:
        result = ParameterValidator().validate_feature_names(["a" * 300])
        assert result.is_valid
        assert result.sanitized_data == ["a" * 256]

    def test_entity_expansion_past_limit_rejected(self) -> None:
        # HTML escaping expands "&" fivefold AFTER truncation, so a name of
        # 256 ampersands sanitizes to >256 chars and trips the size check.
        result = ParameterValidator().validate_feature_names(["a" + "&" * 300])
        assert not result.is_valid
        assert ValidationErrorType.SIZE_LIMIT_EXCEEDED in _error_types(result)

    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)",
            "javascript:alert(1)",
            "x--",
            "x; drop table users",
            "1 union select password",
            "${jndi:ldap://evil}",
            "{{7*7}}",
        ],
        ids=[
            "xss_script",
            "xss_javascript",
            "sql_comment",
            "sql_drop",
            "sql_union",
            "template_dollar",
            "template_braces",
        ],
    )
    def test_injection_shaped_names_detected(self, payload: str) -> None:
        result = ParameterValidator().validate_feature_names([payload])
        assert not result.is_valid
        assert ValidationErrorType.INJECTION_DETECTED in _error_types(result)

    def test_injection_detection_is_case_insensitive(self) -> None:
        result = ParameterValidator().validate_feature_names(["x UNION ALL SELECT y"])
        assert not result.is_valid
        assert ValidationErrorType.INJECTION_DETECTED in _error_types(result)

    def test_literal_duplicates_rejected(self) -> None:
        result = ParameterValidator().validate_feature_names(["temp", "temp"])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.DUPLICATE_VALUE]
        assert result.errors[0].field == "features[1]"

    def test_post_sanitization_collision_rejected(self) -> None:
        # "a!" and "a?" both sanitize to "a_" — the collision must be caught
        # on the SANITIZED names, or two distinct inputs would alias silently.
        result = ParameterValidator().validate_feature_names(["a!", "a?"])
        assert not result.is_valid
        assert ValidationErrorType.DUPLICATE_VALUE in _error_types(result)
        assert result.sanitized_data == ["a_", "a_"]


# =============================================================================
# ParameterValidator — domain
# =============================================================================


class TestValidateDomain:
    """Domain: allowlist with normalization and a 'general' fallback."""

    def test_none_defaults_to_general(self) -> None:
        result = ParameterValidator().validate_domain(None)
        assert result.is_valid
        assert result.sanitized_data == "general"

    @pytest.mark.parametrize(
        "domain",
        ["medical", "financial", "infrastructure", "security", "humanitarian", "general"],
    )
    def test_each_allowed_domain_accepted(self, domain: str) -> None:
        result = ParameterValidator().validate_domain(domain)
        assert result.is_valid
        assert result.sanitized_data == domain

    def test_case_and_whitespace_normalized(self) -> None:
        result = ParameterValidator().validate_domain("  SECURITY  ")
        assert result.is_valid
        assert result.sanitized_data == "security"

    def test_unknown_domain_rejected_with_general_fallback(self) -> None:
        result = ParameterValidator().validate_domain("quantum")
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_VALUE]
        assert "medical" in result.errors[0].constraint
        assert result.sanitized_data == "general"

    def test_injection_shaped_domain_rejected(self) -> None:
        result = ParameterValidator().validate_domain("<script>medical")
        assert not result.is_valid
        assert result.sanitized_data == "general"


# =============================================================================
# APIRequestValidator — combined request validation
# =============================================================================


class TestUnivariateRequest:
    """End-to-end univariate request: data + sensitivity aggregation."""

    def test_valid_request_returns_sanitized_bundle(self) -> None:
        result = APIRequestValidator().validate_univariate_request(
            [1.0, 2.0, 3.0, 4.0], sensitivity=0.7
        )
        assert result.is_valid
        assert result.sanitized_data["sensitivity"] == 0.7
        assert result.sanitized_data["data"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_missing_sensitivity_defaults_to_half(self) -> None:
        result = APIRequestValidator().validate_univariate_request([1.0, 2.0, 3.0])
        assert result.is_valid
        assert result.sanitized_data["sensitivity"] == 0.5

    def test_data_and_sensitivity_errors_aggregated(self) -> None:
        result = APIRequestValidator().validate_univariate_request([1.0, 2.0], sensitivity=2.0)
        assert not result.is_valid
        assert sorted(e.error_type.value for e in result.errors) == [
            "invalid_range",
            "size_limit_exceeded",
        ]

    def test_data_warnings_propagated(self) -> None:
        data = [float(i) for i in range(9)] + [NAN]
        result = APIRequestValidator().validate_univariate_request(data)
        assert result.is_valid
        assert any("NaN" in w for w in result.warnings)

    def test_unknown_kwargs_tolerated(self) -> None:
        result = APIRequestValidator().validate_univariate_request(
            [1.0, 2.0, 3.0], sensitivity=0.5, unknown_option="ignored"
        )
        assert result.is_valid

    def test_custom_config_threaded_to_sub_validators(self) -> None:
        validator = APIRequestValidator(ValidationConfig(max_data_points=5))
        result = validator.validate_univariate_request([float(i) for i in range(6)])
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.SIZE_LIMIT_EXCEEDED]


class TestMultivariateRequest:
    """End-to-end multivariate request: data + features + sensitivity."""

    @staticmethod
    def _data(n_samples: int = 4, n_features: int = 2) -> list[list[float]]:
        rng = np.random.default_rng(42)
        return cast("list[list[float]]", rng.normal(size=(n_samples, n_features)).tolist())

    def test_valid_request_with_matching_features(self) -> None:
        result = APIRequestValidator().validate_multivariate_request(
            self._data(), features=["a", "b"], sensitivity=0.3
        )
        assert result.is_valid
        assert result.sanitized_data["features"] == ["a", "b"]
        assert result.sanitized_data["sensitivity"] == 0.3
        assert result.sanitized_data["data"].shape == (4, 2)

    def test_feature_count_mismatch_is_constraint_violation(self) -> None:
        result = APIRequestValidator().validate_multivariate_request(
            self._data(n_features=2), features=["a", "b", "c"]
        )
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.CONSTRAINT_VIOLATION]
        assert result.errors[0].value == 3
        assert result.errors[0].constraint == 2

    def test_empty_feature_list_mismatches_nonempty_data(self) -> None:
        result = APIRequestValidator().validate_multivariate_request(self._data(), features=[])
        assert not result.is_valid
        assert ValidationErrorType.CONSTRAINT_VIOLATION in _error_types(result)

    def test_none_features_skip_mismatch_check(self) -> None:
        result = APIRequestValidator().validate_multivariate_request(self._data())
        assert result.is_valid
        assert result.sanitized_data["features"] is None

    def test_1d_data_with_features_fails_format_only(self) -> None:
        # When the data fails shape validation there is no sanitized array, so
        # the feature-count check must be skipped rather than crash.
        result = APIRequestValidator().validate_multivariate_request(
            cast("list[list[float]]", [1.0, 2.0, 3.0]), features=["a"]
        )
        assert not result.is_valid
        assert _error_types(result) == [ValidationErrorType.INVALID_FORMAT]

    def test_errors_from_all_three_validators_aggregated(self) -> None:
        result = APIRequestValidator().validate_multivariate_request(
            self._data(n_samples=2),  # too few samples
            features=["<script>bad", "b"],  # injection
            sensitivity=cast("float", "max"),  # wrong type
        )
        assert not result.is_valid
        types = set(_error_types(result))
        assert ValidationErrorType.SIZE_LIMIT_EXCEEDED in types
        assert ValidationErrorType.INJECTION_DETECTED in types
        assert ValidationErrorType.INVALID_TYPE in types


# =============================================================================
# Module-level convenience functions and exports
# =============================================================================


class TestConvenienceFunctions:
    """``validate_univariate`` / ``validate_multivariate`` free functions."""

    def test_validate_univariate_happy_path(self) -> None:
        result = validate_univariate([1.0, 2.0, 3.0], sensitivity=0.9)
        assert result.is_valid
        assert result.sanitized_data["sensitivity"] == 0.9

    def test_validate_univariate_rejects_bad_data(self) -> None:
        result = validate_univariate([1.0, 2.0])
        assert not result.is_valid

    def test_validate_multivariate_happy_path(self) -> None:
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = validate_multivariate(data, features=["x", "y"])
        assert result.is_valid

    def test_validate_multivariate_detects_mismatch(self) -> None:
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = validate_multivariate(data, features=["only_one"])
        assert not result.is_valid
        assert ValidationErrorType.CONSTRAINT_VIOLATION in _error_types(result)


def test_all_exports_resolve() -> None:
    """Every name in ``__all__`` must be importable from the module."""
    import omni_mercury_engine.validation.api_validators as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"__all__ lists missing attribute {name}"
