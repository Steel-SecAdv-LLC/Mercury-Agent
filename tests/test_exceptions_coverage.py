# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for exception classes."""

from __future__ import annotations

import pytest

from omni_mercury_engine.core.exceptions import (
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

    def test_omni_anomaly_exception(self) -> None:
        """Test base exception."""
        with pytest.raises(OmniAnomalyException):
            raise OmniAnomalyException("Test error")

    def test_detector_exception(self) -> None:
        """Test detector exception."""
        with pytest.raises(DetectorException):
            raise DetectorException("Detector error")

    def test_detector_exception_inheritance(self) -> None:
        """Test detector exception inherits from base."""
        with pytest.raises(OmniAnomalyException):
            raise DetectorException("Also a base exception")

    def test_model_exception(self) -> None:
        """Test model exception."""
        with pytest.raises(ModelException):
            raise ModelException("Model error")

    def test_model_exception_inheritance(self) -> None:
        """Test model exception inherits from base."""
        with pytest.raises(OmniAnomalyException):
            raise ModelException("Also a base exception")

    def test_fusion_exception(self) -> None:
        """Test fusion exception."""
        with pytest.raises(FusionException):
            raise FusionException("Fusion error")

    def test_fusion_exception_inheritance(self) -> None:
        """Test fusion exception inherits from base."""
        with pytest.raises(OmniAnomalyException):
            raise FusionException("Also a base exception")

    def test_config_exception(self) -> None:
        """Test config exception."""
        with pytest.raises(ConfigException):
            raise ConfigException("Config error")

    def test_config_exception_inheritance(self) -> None:
        """Test config exception inherits from base."""
        with pytest.raises(OmniAnomalyException):
            raise ConfigException("Also a base exception")

    def test_data_exception(self) -> None:
        """Test data exception."""
        with pytest.raises(DataException):
            raise DataException("Data error")

    def test_data_exception_inheritance(self) -> None:
        """Test data exception inherits from base."""
        with pytest.raises(OmniAnomalyException):
            raise DataException("Also a base exception")

    def test_security_exception(self) -> None:
        """Test security exception."""
        with pytest.raises(SecurityException):
            raise SecurityException("Security error")

    def test_security_exception_inheritance(self) -> None:
        """Test security exception inherits from base."""
        with pytest.raises(OmniAnomalyException):
            raise SecurityException("Also a base exception")

    def test_exception_messages(self) -> None:
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

    def test_exception_inheritance(self) -> None:
        """Test exception inheritance."""
        assert issubclass(DetectorException, OmniAnomalyException)
        assert issubclass(ModelException, OmniAnomalyException)
        assert issubclass(FusionException, OmniAnomalyException)
        assert issubclass(ConfigException, OmniAnomalyException)
        assert issubclass(DataException, OmniAnomalyException)
        assert issubclass(SecurityException, OmniAnomalyException)
        assert issubclass(OmniAnomalyException, Exception)
