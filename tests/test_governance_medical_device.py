"""Tests for the ISO 14971 medical-device risk governance scalar."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # governance.contract -> GOSNN imports numpy.

from omni_mercury_engine.governance import medical_device
from omni_mercury_engine.governance.contract import ScalarStatus


def test_risk_index_computes_and_tiers() -> None:
    """severity x probability yields the normalized risk index and acceptability tier."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 4, "probability": 4})
    assert scalar.status is ScalarStatus.AVAILABLE
    assert scalar.value == pytest.approx(16 / 25)
    assert scalar.provenance["tier"] == "unacceptable"


def test_low_risk_is_acceptable() -> None:
    """A low severity/probability combination lands in the acceptable band."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 2, "probability": 2})
    assert scalar.value == pytest.approx(4 / 25)
    assert scalar.provenance["tier"] == "acceptable"


def test_abstains_when_coordinate_missing() -> None:
    """A missing coordinate abstains: a risk index needs both severity and probability."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 3})
    assert scalar.status is ScalarStatus.UNAVAILABLE
    assert scalar.value is None


def test_abstains_on_out_of_range() -> None:
    """Out-of-range levels (not 1-5) abstain rather than clamp silently."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 9, "probability": 3})
    assert scalar.status is ScalarStatus.UNAVAILABLE
