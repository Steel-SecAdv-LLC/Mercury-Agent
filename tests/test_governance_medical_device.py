"""Tests for the ISO 14971 medical-device risk governance scalar (three-state)."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # governance.contract -> GOSNN imports numpy.

from omni_mercury_engine.governance import medical_device
from omni_mercury_engine.governance.contract import ThreeState


def test_iso14971_worked_example_and_tier() -> None:
    """severity 4 x probability 4 = 16/25, in the 'unacceptable' band."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 4, "probability": 4})
    assert scalar.state is ThreeState.GROUNDED
    assert scalar.value == pytest.approx(16 / 25)
    assert scalar.provenance["risk_index"] == 16
    assert scalar.provenance["tier"] == "unacceptable"


def test_iso14971_acceptable_band() -> None:
    """severity 1 x probability 2 = 2/25, in the 'acceptable' band."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 1, "probability": 2})
    assert scalar.value == pytest.approx(2 / 25)
    assert scalar.provenance["tier"] == "acceptable"


def test_iso14971_abstains_on_missing_coordinate() -> None:
    """An absent coordinate abstains UNAVAILABLE, recording which is missing."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 4})
    assert scalar.state is ThreeState.UNAVAILABLE
    assert scalar.value is None
    assert scalar.missing_inputs == ("probability",)


def test_iso14971_abstains_on_out_of_range() -> None:
    """A coordinate outside 1-5 abstains rather than being clamped to a fabricated value."""
    scalar = medical_device.iso14971_risk_scalar({"severity": 9, "probability": 3})
    assert scalar.state is ThreeState.UNAVAILABLE
    assert scalar.missing_inputs == ("severity",)


def test_not_for_device_clearance_boundary_is_documented() -> None:
    """The module must carry the not-for-device-clearance boundary in its source."""
    import inspect

    assert "NOT FOR DEVICE CLEARANCE" in inspect.getsource(medical_device).upper()
