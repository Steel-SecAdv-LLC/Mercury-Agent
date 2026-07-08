# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Oracle checkpoint-restore failures must be surfaced, not swallowed (F9).

Regression: MercuryAnomalyDetector._restore_oracle_from_ref_stats caught every
exception at DEBUG and left the detector marked fitted but scoring WITHOUT its
Oracle component — silently diverging from the checkpointed model. It now logs
at WARNING and records restore_failed in the Oracle metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

if TYPE_CHECKING:
    import pytest


def test_restore_failure_is_recorded_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = MercuryAnomalyDetector()

    # Force Oracle reconstruction to fail even though stats WERE present.
    import omni_mercury_engine.detectors.spectral_domain_frequency as sdf

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic oracle restore failure")

    monkeypatch.setattr(sdf, "SpectralDomainFrequency", _boom)

    detector._restore_oracle_from_ref_stats({"domain": "environmental"})

    assert detector._oracle_detector is None
    assert detector._oracle_metadata.get("restore_failed") is True
    assert detector._oracle_metadata.get("active") is False
    assert "synthetic oracle restore failure" in detector._oracle_metadata.get("error", "")


def test_no_oracle_stats_is_not_a_failure() -> None:
    """The genuine 'no Oracle in checkpoint' case stays a clean inactive state."""
    detector = MercuryAnomalyDetector()
    detector._restore_oracle_from_ref_stats(None)

    assert detector._oracle_detector is None
    assert detector._oracle_metadata.get("active") is False
    assert "restore_failed" not in detector._oracle_metadata
