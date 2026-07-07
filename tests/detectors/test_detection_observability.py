# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Observability tests: the non-finite correction guard is metered and logged.

Asserts that every guard that rescues a non-finite value increments the
``omni_detector_nonfinite_corrected{detector,policy,field}`` Prometheus counter
and emits a structured log carrying the detector, field, value-type counts, and
remediation -- for the standalone guards and end-to-end through the tier
detectors (SPOT / digital-twin / BOCPD / RCA).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("prometheus_client")

from omni_mercury_engine.core.metrics import (
    DETECTOR_NONFINITE_CORRECTED,
    is_prometheus_available,
)
from omni_mercury_engine.detectors.detection_config import (
    apply_nan_policy,
    guard_finite_scalar,
)


def _counter_value(detector: str, policy: str, field: str) -> float:
    """Read the current value of the nonfinite-corrected counter for a label set."""
    for metric in DETECTOR_NONFINITE_CORRECTED.collect():
        for sample in metric.samples:
            if (
                sample.name.endswith("_total")
                and sample.labels.get("detector") == detector
                and sample.labels.get("policy") == policy
                and sample.labels.get("field") == field
            ):
                return float(sample.value)
    return 0.0


@pytest.mark.skipif(not is_prometheus_available(), reason="prometheus_client not installed")
class TestGuardMetrics:
    def test_apply_nan_policy_increments_counter(self) -> None:
        before = _counter_value("metric_det", "neutral", "input")
        apply_nan_policy(
            np.array([1.0, np.nan, np.inf]),
            policy="neutral",
            detector="metric_det",
            field="input",
            max_magnitude=1e6,
        )
        after = _counter_value("metric_det", "neutral", "input")
        # two corrected values (one NaN, one Inf)
        assert after - before == pytest.approx(2.0)

    def test_clean_input_does_not_increment(self) -> None:
        before = _counter_value("metric_det", "neutral", "input")
        apply_nan_policy(
            np.array([1.0, 2.0, 3.0]), policy="neutral", detector="metric_det", field="input"
        )
        assert _counter_value("metric_det", "neutral", "input") == before

    def test_scalar_guard_increments_counter(self) -> None:
        before = _counter_value("metric_det", "neutral", "z_q")
        guard_finite_scalar(np.nan, detector="metric_det", field="z_q", max_magnitude=1e6)
        assert _counter_value("metric_det", "neutral", "z_q") - before == pytest.approx(1.0)

    def test_policy_label_distinguishes(self) -> None:
        b_impute = _counter_value("pol_det", "impute", "input")
        apply_nan_policy(np.array([np.nan]), policy="impute", detector="pol_det", field="input")
        assert _counter_value("pol_det", "impute", "input") - b_impute == pytest.approx(1.0)

    def test_spot_metadata_guard_metered(self) -> None:
        """A SPOT detector whose fitted gamma is non-finite meters the z_q/gamma guard."""
        from omni_mercury_engine.detectors.spot_evt import SPOTDetector

        det = SPOTDetector().fit(np.random.default_rng(0).normal(size=400))
        det._gamma = float("nan")  # force a non-finite metadata field
        before = _counter_value(det.name, "neutral", "gamma")
        out = det.detect(np.random.default_rng(1).normal(size=100))
        assert np.isfinite(out["metadata"]["gamma"])  # guarded to a finite value
        assert _counter_value(det.name, "neutral", "gamma") - before >= 1.0


@pytest.mark.skipif(not is_prometheus_available(), reason="prometheus_client not installed")
class TestGuardStructuredLogs:
    def test_apply_nan_policy_logs_structured(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(
            logging.WARNING, logger="omni_mercury_engine.detectors.detection_config"
        ):
            apply_nan_policy(
                np.array([1.0, np.nan, np.inf, -np.inf]),
                policy="neutral",
                detector="log_det",
                field="scores",
                max_magnitude=1e6,
            )
        records = [r for r in caplog.records if getattr(r, "detector", None) == "log_det"]
        assert records, "expected a structured warning for the correction"
        rec = records[-1]
        assert rec.field == "scores"
        assert rec.policy == "neutral"
        assert rec.n_corrected == 3
        assert rec.n_nan == 1
        assert rec.n_inf == 2
        assert "remediation" in rec.__dict__

    def test_scalar_guard_logs_structured(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(
            logging.WARNING, logger="omni_mercury_engine.detectors.detection_config"
        ):
            guard_finite_scalar(float("inf"), detector="log_det", field="z_q", max_magnitude=1e6)
        records = [r for r in caplog.records if getattr(r, "detector", None) == "log_det"]
        assert records
        assert records[-1].field == "z_q"
        assert records[-1].n_inf == 1


@pytest.mark.skipif(not is_prometheus_available(), reason="prometheus_client not installed")
class TestDetectorsMeterNonFiniteInput:
    """End-to-end: feeding non-finite input to a tier detector meters the guard."""

    @pytest.mark.parametrize("policy_field", ["input"])
    def test_digital_twin_meters_nonfinite_input(self, policy_field: str) -> None:
        from omni_mercury_engine.detectors.digital_twin import DigitalTwinResidualDetector

        det = DigitalTwinResidualDetector()
        det.fit(np.random.default_rng(0).normal(size=300))
        before = _counter_value(det.name, "neutral", "input")
        data = np.random.default_rng(1).normal(size=100)
        data[10] = np.nan
        data[20] = np.inf
        out = det.detect(data)
        assert np.all(np.isfinite(out["scores"]))
        assert _counter_value(det.name, "neutral", "input") - before >= 2.0

    def test_bocpd_meters_nonfinite_input(self) -> None:
        from omni_mercury_engine.detectors.bocpd import BOCPDDetector

        det = BOCPDDetector().fit(np.random.default_rng(0).normal(size=200))
        before = _counter_value(det.name, "neutral", "input")
        data = np.random.default_rng(1).normal(size=80)
        data[5] = np.nan
        det.detect(data)
        assert _counter_value(det.name, "neutral", "input") - before >= 1.0

    def test_rca_meters_nonfinite_input(self) -> None:
        from omni_mercury_engine.detectors.rca import RootCauseGraphDetector

        det = RootCauseGraphDetector().fit(np.random.default_rng(0).normal(size=(100, 4)))
        before = _counter_value(det.name, "neutral", "input")
        rows = np.random.default_rng(1).normal(size=(20, 4))
        rows[3, 1] = np.inf
        det.detect(rows)
        assert _counter_value(det.name, "neutral", "input") - before >= 1.0
