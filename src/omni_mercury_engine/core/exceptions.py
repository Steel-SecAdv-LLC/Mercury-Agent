# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Custom exceptions for Mercury Agent."""

from __future__ import annotations


class OmniAnomalyException(Exception):
    """Base exception for all Mercury Agent errors."""

    pass


class DetectorException(OmniAnomalyException):
    """Exception raised by detector modules."""

    pass


class ModelException(OmniAnomalyException):
    """Exception raised by model modules."""

    pass


class FusionException(OmniAnomalyException):
    """Exception raised by ML fusion components."""

    pass


class ConfigException(OmniAnomalyException):
    """Exception raised for configuration errors."""

    pass


class DataException(OmniAnomalyException):
    """Exception raised for data processing errors."""

    pass


class SecurityException(OmniAnomalyException):
    """Exception raised for security-related errors."""

    pass


class InputModalityError(ValueError):
    """A component declined input of a modality it does not process.

    Raised when, for example, a 1-D signal detector receives a 2-D feature
    matrix or a mapping-only model receives an array. This is a correct
    refusal, not a fault: the engine skips the component and the metrics
    layer counts it as *declined* rather than as an inference error, so an
    error-rate alert reflects real failures only.

    Subclasses ``ValueError`` and nothing else, so every existing
    ``except ValueError`` handler and test keeps its behaviour and no
    ``except OmniAnomalyException`` handler starts catching it.
    """
