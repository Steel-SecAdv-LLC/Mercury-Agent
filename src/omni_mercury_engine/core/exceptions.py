"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Custom exceptions for Mercury Agent ♱
"""


class OmniAnomalyException(Exception):
    """Base exception for all Mercury Agent ♱ errors"""

    pass


class DetectorException(OmniAnomalyException):
    """Exception raised by detector modules"""

    pass


class ModelException(OmniAnomalyException):
    """Exception raised by model modules"""

    pass


class FusionException(OmniAnomalyException):
    """Exception raised by ML fusion components"""

    pass


class ConfigException(OmniAnomalyException):
    """Exception raised for configuration errors"""

    pass


class DataException(OmniAnomalyException):
    """Exception raised for data processing errors"""

    pass


class SecurityException(OmniAnomalyException):
    """Exception raised for security-related errors"""

    pass
