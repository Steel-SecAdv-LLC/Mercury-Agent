# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test humanitarian detectors."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.infrastructure.humanitarian import (
    AgriFoodSecurityDetector,
    ClimateResilienceDetector,
    EconomicResilienceDetector,
    EducationEquityDetector,
    NeuroscienceDetector,
)


class TestAgriFoodSecurityDetector:
    def test_instantiation(self) -> None:
        detector = AgriFoodSecurityDetector()
        assert detector is not None

    def test_has_detect_method(self) -> None:
        detector = AgriFoodSecurityDetector()
        assert callable(getattr(detector, "detect", None))

    def test_detect_returns_result(self) -> None:
        detector = AgriFoodSecurityDetector()
        data = np.random.default_rng(42).random((10, 5))
        result = detector.detect(data)
        assert result is not None


class TestClimateResilienceDetector:
    def test_instantiation(self) -> None:
        detector = ClimateResilienceDetector()
        assert detector is not None

    def test_has_detect_method(self) -> None:
        detector = ClimateResilienceDetector()
        assert callable(getattr(detector, "detect", None))

    def test_detect_returns_result(self) -> None:
        detector = ClimateResilienceDetector()
        data = np.random.default_rng(42).random((10, 5))
        result = detector.detect(data)
        assert result is not None


class TestEconomicResilienceDetector:
    def test_instantiation(self) -> None:
        detector = EconomicResilienceDetector()
        assert detector is not None

    def test_has_detect_method(self) -> None:
        detector = EconomicResilienceDetector()
        assert callable(getattr(detector, "detect", None))

    def test_detect_returns_result(self) -> None:
        detector = EconomicResilienceDetector()
        data = np.random.default_rng(42).random((10, 5))
        result = detector.detect(data)
        assert result is not None


class TestEducationEquityDetector:
    def test_instantiation(self) -> None:
        detector = EducationEquityDetector()
        assert detector is not None

    def test_has_detect_method(self) -> None:
        detector = EducationEquityDetector()
        assert callable(getattr(detector, "detect", None))

    def test_detect_returns_result(self) -> None:
        detector = EducationEquityDetector()
        data = np.random.default_rng(42).random((10, 5))
        result = detector.detect(data)
        assert result is not None


class TestNeuroscienceDetector:
    def test_instantiation(self) -> None:
        detector = NeuroscienceDetector()
        assert detector is not None

    def test_has_detect_method(self) -> None:
        detector = NeuroscienceDetector()
        assert callable(getattr(detector, "detect", None))

    def test_detect_returns_result(self) -> None:
        detector = NeuroscienceDetector()
        data = np.random.default_rng(42).random((10, 5))
        result = detector.detect(data)
        assert result is not None
