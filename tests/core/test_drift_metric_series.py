# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Drift counter series exist at 0 from import, before the first increment.

A series created on its first ``.inc()`` gives ``increase()`` a single sample,
so the first drift a pod detected read as 0 and ``OmniMercuryDataDrift`` never
fired for it.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.core import metrics
from omni_mercury_engine.ml.drift import DriftSeverity

prometheus = pytest.importorskip("prometheus_client")


def test_severity_labels_cover_every_drift_severity() -> None:
    assert set(metrics.DRIFT_SEVERITIES) == {s.name for s in DriftSeverity} - {"NONE"}


@pytest.mark.parametrize("severity", metrics.DRIFT_SEVERITIES)
def test_every_severity_series_is_pre_created(severity: str) -> None:
    value = prometheus.REGISTRY.get_sample_value(
        "omni_drift_detected_total", {"severity": severity}
    )
    assert value is not None


@pytest.mark.parametrize("status", metrics.DRIFT_CHECK_STATUSES)
def test_every_status_series_is_pre_created(status: str) -> None:
    value = prometheus.REGISTRY.get_sample_value("omni_drift_checks_total", {"status": status})
    assert value is not None
