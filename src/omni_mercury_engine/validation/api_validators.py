# Copyright (C) 2025 Steel Security Advisors LLC
"""API Input Validation Module."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ValidationErrorType(Enum):
    """Types of validation errors."""

    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"
    INVALID_VALUE = "invalid_value"
    INVALID_TYPE = "invalid_type"
    INVALID_RANGE = "invalid_range"
    INVALID_FORMAT = "invalid_format"
    INJECTION_DETECTED = "injection_detected"
    MISSING_REQUIRED = "missing_required"
    DUPLICATE_VALUE = "duplicate_value"
    CONSTRAINT_VIOLATION = "constraint_violation"


@dataclass
class ValidationError:
    """Structured validation error."""

    error_type: ValidationErrorType
    field: str
    message: str
    value: Any = None
    constraint: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "error_type": self.error_type.value,
            "field": self.field,
            "message": self.message,
            "value": str(self.value)[:100] if self.value is not None else None,
            "constraint": self.constraint,
        }


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_data: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
        }


@dataclass
class ValidationConfig:
    """Configuration for input validation."""

    # Size limits
    max_data_points: int = 100000
    max_features: int = 1000
    max_string_length: int = 256
    max_array_depth: int = 3

    # Value limits
    min_value: float = -1e15
    max_value: float = 1e15
    max_nan_ratio: float = 0.1
    max_inf_ratio: float = 0.01

    # Format constraints
    allowed_feature_name_pattern: str = r"^[a-zA-Z_][a-zA-Z0-9_\-\.]*$"
    forbidden_patterns: list[str] = field(
        default_factory=lambda: [
            r"<script",  # XSS prevention
            r"javascript:",  # XSS prevention
            r"--",  # SQL injection
            r";.*drop",  # SQL injection
            r"union.*select",  # SQL injection
            r"\$\{",  # Template injection
            r"\{\{",  # Template injection
        ]
    )

    # Domain-specific
    enable_domain_validation: bool = True
    strict_mode: bool = False


class InputSanitizer:
    """Sanitizes input data to prevent injection attacks."""

    # HTML entities to escape
    HTML_ENTITIES = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
    }

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 256) -> str:
        """Sanitize a string value.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)

        # Truncate to max length
        value = value[:max_length]

        # Escape HTML entities
        for char, entity in cls.HTML_ENTITIES.items():
            value = value.replace(char, entity)

        # Remove null bytes
        value = value.replace("\x00", "")

        # Remove control characters (except newline and tab)
        value = "".join(c for c in value if c.isprintable() or c in "\n\t")

        return value

    @classmethod
    def sanitize_dict(cls, data: dict[str, Any], max_depth: int = 3) -> dict[str, Any]:
        """Recursively sanitize a dictionary.

        Args:
            data: Dictionary to sanitize
            max_depth: Maximum recursion depth

        Returns:
            Sanitized dictionary
        """
        if max_depth <= 0:
            return {}

        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            # Sanitize key
            safe_key = cls.sanitize_string(key)

            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[safe_key] = cls.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[safe_key] = cls.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                sanitized[safe_key] = cls.sanitize_list(value, max_depth - 1)
            else:
                sanitized[safe_key] = value

        return sanitized

    @classmethod
    def sanitize_list(cls, data: list[Any], max_depth: int = 3) -> list[Any]:
        """Recursively sanitize a list.

        Args:
            data: List to sanitize
            max_depth: Maximum recursion depth

        Returns:
            Sanitized list
        """
        if max_depth <= 0:
            return []

        sanitized: list[Any] = []
        for item in data:
            if isinstance(item, str):
                sanitized.append(cls.sanitize_string(item))
            elif isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, max_depth - 1))
            elif isinstance(item, list):
                sanitized.append(cls.sanitize_list(item, max_depth - 1))
            else:
                sanitized.append(item)

        return sanitized


class DataArrayValidator:
    """Validates numerical data arrays."""

    def __init__(self, config: ValidationConfig | None = None):
        """Initialize data array validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

    def validate_univariate(self, data: list[float] | NDArray[np.float64]) -> ValidationResult:
        """Validate univariate time series data.

        Args:
            data: 1D numerical data

        Returns:
            Validation result
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        try:
            arr = np.asarray(data, dtype=np.float64)
        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_TYPE,
                        field="data",
                        message=f"Cannot convert data to numerical array: {e}",
                    )
                ],
            )

        # Check dimensionality
        if arr.ndim != 1:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    field="data",
                    message=f"Expected 1D array, got {arr.ndim}D",
                    value=arr.shape,
                )
            )
            return ValidationResult(is_valid=False, errors=errors)

        # Check size limit
        if len(arr) > self.config.max_data_points:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.SIZE_LIMIT_EXCEEDED,
                    field="data",
                    message=f"Data length {len(arr)} exceeds maximum {self.config.max_data_points}",
                    value=len(arr),
                    constraint=self.config.max_data_points,
                )
            )

        # Check minimum size
        if len(arr) < 3:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.SIZE_LIMIT_EXCEEDED,
                    field="data",
                    message="Data must contain at least 3 points for statistical analysis",
                    value=len(arr),
                    constraint=3,
                )
            )

        # Check for NaN values
        nan_count = np.sum(np.isnan(arr))
        nan_ratio = nan_count / (len(arr) + 1e-10)
        if nan_ratio > self.config.max_nan_ratio:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_VALUE,
                    field="data",
                    message=f"NaN ratio {nan_ratio:.2%} exceeds maximum {self.config.max_nan_ratio:.2%}",
                    value=nan_ratio,
                    constraint=self.config.max_nan_ratio,
                )
            )
        elif nan_count > 0:
            warnings.append(f"Data contains {nan_count} NaN values ({nan_ratio:.2%})")

        # Check for Inf values
        inf_count = np.sum(np.isinf(arr))
        inf_ratio = inf_count / (len(arr) + 1e-10)
        if inf_ratio > self.config.max_inf_ratio:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_VALUE,
                    field="data",
                    message=f"Inf ratio {inf_ratio:.2%} exceeds maximum {self.config.max_inf_ratio:.2%}",
                    value=inf_ratio,
                    constraint=self.config.max_inf_ratio,
                )
            )
        elif inf_count > 0:
            warnings.append(f"Data contains {inf_count} Inf values ({inf_ratio:.2%})")

        # Check value range
        finite_arr = arr[np.isfinite(arr)]
        if len(finite_arr) > 0:
            min_val = np.min(finite_arr)
            max_val = np.max(finite_arr)

            if min_val < self.config.min_value:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_RANGE,
                        field="data",
                        message=f"Minimum value {min_val} below allowed {self.config.min_value}",
                        value=min_val,
                        constraint=self.config.min_value,
                    )
                )

            if max_val > self.config.max_value:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_RANGE,
                        field="data",
                        message=f"Maximum value {max_val} exceeds allowed {self.config.max_value}",
                        value=max_val,
                        constraint=self.config.max_value,
                    )
                )

        # Check for constant data
        if len(finite_arr) > 0 and np.std(finite_arr) == 0:
            warnings.append("Data has zero variance (all values identical)")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=arr,
        )

    def validate_multivariate(
        self, data: list[list[float]] | NDArray[np.float64]
    ) -> ValidationResult:
        """Validate multivariate time series data.

        Args:
            data: 2D numerical data (samples x features)

        Returns:
            Validation result
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        try:
            arr = np.asarray(data, dtype=np.float64)
        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_TYPE,
                        field="data",
                        message=f"Cannot convert data to numerical array: {e}",
                    )
                ],
            )

        # Check dimensionality
        if arr.ndim != 2:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    field="data",
                    message=f"Expected 2D array, got {arr.ndim}D",
                    value=arr.shape,
                )
            )
            return ValidationResult(is_valid=False, errors=errors)

        n_samples, n_features = arr.shape

        # Check size limits
        if n_samples > self.config.max_data_points:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.SIZE_LIMIT_EXCEEDED,
                    field="data",
                    message=f"Sample count {n_samples} exceeds maximum {self.config.max_data_points}",
                    value=n_samples,
                    constraint=self.config.max_data_points,
                )
            )

        if n_features > self.config.max_features:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.SIZE_LIMIT_EXCEEDED,
                    field="data",
                    message=f"Feature count {n_features} exceeds maximum {self.config.max_features}",
                    value=n_features,
                    constraint=self.config.max_features,
                )
            )

        # Check minimum size
        if n_samples < 3:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.SIZE_LIMIT_EXCEEDED,
                    field="data",
                    message="Data must contain at least 3 samples",
                    value=n_samples,
                    constraint=3,
                )
            )

        # Check for NaN values per feature
        for i in range(n_features):
            nan_count = np.sum(np.isnan(arr[:, i]))
            nan_ratio = nan_count / (n_samples + 1e-10)
            if nan_ratio > self.config.max_nan_ratio:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_VALUE,
                        field=f"data[feature_{i}]",
                        message=f"Feature {i} NaN ratio {nan_ratio:.2%} exceeds maximum",
                        value=nan_ratio,
                    )
                )

        # Check for Inf values
        inf_count = np.sum(np.isinf(arr))
        if inf_count > 0:
            inf_ratio = inf_count / arr.size
            if inf_ratio > self.config.max_inf_ratio:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_VALUE,
                        field="data",
                        message=f"Inf ratio {inf_ratio:.2%} exceeds maximum",
                        value=inf_ratio,
                    )
                )

        # Check value range
        finite_arr = arr[np.isfinite(arr)]
        if len(finite_arr) > 0:
            if np.min(finite_arr) < self.config.min_value:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_RANGE,
                        field="data",
                        message=f"Values below minimum {self.config.min_value}",
                    )
                )
            if np.max(finite_arr) > self.config.max_value:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_RANGE,
                        field="data",
                        message=f"Values exceed maximum {self.config.max_value}",
                    )
                )

        # Check for constant features
        constant_features = []
        for i in range(n_features):
            col_finite = arr[:, i][np.isfinite(arr[:, i])]
            if len(col_finite) > 0 and np.std(col_finite) == 0:
                constant_features.append(i)

        if constant_features:
            warnings.append(f"Features with zero variance: {constant_features}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=arr,
        )


class ParameterValidator:
    """Validates API parameters."""

    def __init__(self, config: ValidationConfig | None = None):
        """Initialize parameter validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self._feature_name_pattern = re.compile(self.config.allowed_feature_name_pattern)
        self._forbidden_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.forbidden_patterns
        ]

    def validate_sensitivity(self, value: float | None) -> ValidationResult:
        """Validate sensitivity parameter."""
        if value is None:
            return ValidationResult(is_valid=True, sanitized_data=0.5)

        errors = []
        if not isinstance(value, (int, float)):
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_TYPE,
                    field="sensitivity",
                    message="Sensitivity must be a number",
                    value=type(value).__name__,
                )
            )
        elif not 0.0 <= value <= 1.0:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_RANGE,
                    field="sensitivity",
                    message="Sensitivity must be between 0.0 and 1.0",
                    value=value,
                    constraint="[0.0, 1.0]",
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=float(value) if isinstance(value, (int, float)) else 0.5,
        )

    def validate_feature_names(self, names: list[str] | None) -> ValidationResult:
        """Validate feature name list."""
        if names is None:
            return ValidationResult(is_valid=True, sanitized_data=None)

        errors = []
        warnings = []
        sanitized = []

        seen_names: set[str] = set()

        for i, name in enumerate(names):
            # Check type
            if not isinstance(name, str):
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_TYPE,
                        field=f"features[{i}]",
                        message="Feature name must be a string",
                        value=type(name).__name__,
                    )
                )
                continue

            # Sanitize
            safe_name = InputSanitizer.sanitize_string(name, self.config.max_string_length)

            # Check length
            if len(safe_name) > self.config.max_string_length:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.SIZE_LIMIT_EXCEEDED,
                        field=f"features[{i}]",
                        message=f"Feature name too long (max {self.config.max_string_length})",
                        value=len(name),
                    )
                )

            # Check format
            if not self._feature_name_pattern.match(safe_name):
                # Try to make it valid
                safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", safe_name)
                if not safe_name or not safe_name[0].isalpha():
                    safe_name = f"feature_{i}"
                warnings.append(f"Feature name '{name}' sanitized to '{safe_name}'")

            # Check for forbidden patterns
            for pattern in self._forbidden_patterns:
                if pattern.search(name):
                    errors.append(
                        ValidationError(
                            error_type=ValidationErrorType.INJECTION_DETECTED,
                            field=f"features[{i}]",
                            message="Feature name contains forbidden pattern",
                        )
                    )
                    break

            # Check for duplicates
            if safe_name in seen_names:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.DUPLICATE_VALUE,
                        field=f"features[{i}]",
                        message=f"Duplicate feature name: {safe_name}",
                        value=safe_name,
                    )
                )
            seen_names.add(safe_name)

            sanitized.append(safe_name)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized,
        )

    def validate_domain(self, domain: str | None) -> ValidationResult:
        """Validate domain parameter."""
        if domain is None:
            return ValidationResult(is_valid=True, sanitized_data="general")

        errors = []
        valid_domains = [
            "medical",
            "financial",
            "infrastructure",
            "security",
            "humanitarian",
            "general",
        ]

        # Sanitize
        safe_domain = InputSanitizer.sanitize_string(domain.lower().strip(), 50)

        if safe_domain not in valid_domains:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_VALUE,
                    field="domain",
                    message=f"Invalid domain. Valid options: {valid_domains}",
                    value=safe_domain,
                    constraint=valid_domains,
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=safe_domain if safe_domain in valid_domains else "general",
        )


class APIRequestValidator:
    """Unified API request validator."""

    def __init__(self, config: ValidationConfig | None = None):
        """Initialize API request validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.data_validator = DataArrayValidator(config)
        self.param_validator = ParameterValidator(config)

    def validate_univariate_request(
        self,
        data: list[float],
        sensitivity: float | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate univariate detection request.

        Args:
            data: Time series data
            sensitivity: Detection sensitivity
            **kwargs: Additional parameters

        Returns:
            Combined validation result
        """
        all_errors: list[ValidationError] = []
        all_warnings: list[str] = []

        # Validate data
        data_result = self.data_validator.validate_univariate(data)
        all_errors.extend(data_result.errors)
        all_warnings.extend(data_result.warnings)

        # Validate sensitivity
        sens_result = self.param_validator.validate_sensitivity(sensitivity)
        all_errors.extend(sens_result.errors)
        all_warnings.extend(sens_result.warnings)

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_data={
                "data": data_result.sanitized_data,
                "sensitivity": sens_result.sanitized_data,
            },
        )

    def validate_multivariate_request(
        self,
        data: list[list[float]],
        features: list[str] | None = None,
        sensitivity: float | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate multivariate detection request.

        Args:
            data: Multi-dimensional time series data
            features: Feature names
            sensitivity: Detection sensitivity
            **kwargs: Additional parameters

        Returns:
            Combined validation result
        """
        all_errors: list[ValidationError] = []
        all_warnings: list[str] = []

        # Validate data
        data_result = self.data_validator.validate_multivariate(data)
        all_errors.extend(data_result.errors)
        all_warnings.extend(data_result.warnings)

        # Validate feature names
        features_result = self.param_validator.validate_feature_names(features)
        all_errors.extend(features_result.errors)
        all_warnings.extend(features_result.warnings)

        # Check feature count matches data
        if data_result.sanitized_data is not None and features_result.sanitized_data is not None:
            n_features = data_result.sanitized_data.shape[1]
            n_names = len(features_result.sanitized_data)
            if n_features != n_names:
                all_errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.CONSTRAINT_VIOLATION,
                        field="features",
                        message=f"Feature count mismatch: data has {n_features}, names has {n_names}",
                        value=n_names,
                        constraint=n_features,
                    )
                )

        # Validate sensitivity
        sens_result = self.param_validator.validate_sensitivity(sensitivity)
        all_errors.extend(sens_result.errors)
        all_warnings.extend(sens_result.warnings)

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_data={
                "data": data_result.sanitized_data,
                "features": features_result.sanitized_data,
                "sensitivity": sens_result.sanitized_data,
            },
        )


# Convenience functions
def validate_univariate(data: list[float], **kwargs: Any) -> ValidationResult:
    """Validate univariate detection request."""
    validator = APIRequestValidator()
    return validator.validate_univariate_request(data, **kwargs)


def validate_multivariate(data: list[list[float]], **kwargs: Any) -> ValidationResult:
    """Validate multivariate detection request."""
    validator = APIRequestValidator()
    return validator.validate_multivariate_request(data, **kwargs)


def sanitize_input(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize input dictionary."""
    return InputSanitizer.sanitize_dict(data)


# Exports
__all__ = [
    "APIRequestValidator",
    "DataArrayValidator",
    "InputSanitizer",
    "ParameterValidator",
    "ValidationConfig",
    "ValidationError",
    "ValidationErrorType",
    "ValidationResult",
    "sanitize_input",
    "validate_multivariate",
    "validate_univariate",
]
