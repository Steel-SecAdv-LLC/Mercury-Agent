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
from __future__ import annotations

"""
Tests for exception classes
"""

import pytest

from omni_anomaly_engine.core.exceptions import (
    ConfigException,
    DataException,
    DetectorException,
    FusionException,
    ModelException,
    OmniAnomalyException,
    SecurityException,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_omni_anomaly_exception(self):
        """Test base exception."""
        with pytest.raises(OmniAnomalyException):
            raise OmniAnomalyException("Test error")

    def test_detector_exception(self):
        """Test detector exception."""
        with pytest.raises(DetectorException):
            raise DetectorException("Detector error")

        with pytest.raises(OmniAnomalyException):
            raise DetectorException("Also a base exception")

    def test_model_exception(self):
        """Test model exception."""
        with pytest.raises(ModelException):
            raise ModelException("Model error")

        with pytest.raises(OmniAnomalyException):
            raise ModelException("Also a base exception")

    def test_fusion_exception(self):
        """Test fusion exception."""
        with pytest.raises(FusionException):
            raise FusionException("Fusion error")

        with pytest.raises(OmniAnomalyException):
            raise FusionException("Also a base exception")

    def test_config_exception(self):
        """Test config exception."""
        with pytest.raises(ConfigException):
            raise ConfigException("Config error")

        with pytest.raises(OmniAnomalyException):
            raise ConfigException("Also a base exception")

    def test_data_exception(self):
        """Test data exception."""
        with pytest.raises(DataException):
            raise DataException("Data error")

        with pytest.raises(OmniAnomalyException):
            raise DataException("Also a base exception")

    def test_security_exception(self):
        """Test security exception."""
        with pytest.raises(SecurityException):
            raise SecurityException("Security error")

        with pytest.raises(OmniAnomalyException):
            raise SecurityException("Also a base exception")

    def test_exception_messages(self):
        """Test that exception messages are preserved."""
        msg = "Custom error message"

        try:
            raise OmniAnomalyException(msg)
        except OmniAnomalyException as e:
            assert str(e) == msg

        try:
            raise DetectorException(msg)
        except DetectorException as e:
            assert str(e) == msg

    def test_exception_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(DetectorException, OmniAnomalyException)
        assert issubclass(ModelException, OmniAnomalyException)
        assert issubclass(FusionException, OmniAnomalyException)
        assert issubclass(ConfigException, OmniAnomalyException)
        assert issubclass(DataException, OmniAnomalyException)
        assert issubclass(SecurityException, OmniAnomalyException)
        assert issubclass(OmniAnomalyException, Exception)
