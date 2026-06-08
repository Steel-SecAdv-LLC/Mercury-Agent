# Copyright (C) 2025 Steel Security Advisors LLC
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
