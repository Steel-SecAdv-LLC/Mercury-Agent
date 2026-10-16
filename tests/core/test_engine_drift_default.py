# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Drift detection is on by default and never adopts an inference batch.

Before this, ``enable_drift_detection`` had no caller, so the served engines
never tested for drift (a batch shifted by several cluster widths came back
"grounded / clear"); when enabled, the first inference batch became its own
baseline. The reference is now the training rows the model calls normal, and
only the rows the model calls normal are tested, so the two sides are filtered
by the same rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.config import EngineConfig
from omni_mercury_engine.decision import DecisionAbstentionResponder, ThreeState
from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.ml.drift import (
    DriftIncomparableError,
    DriftSeverity,
    EnsembleDriftDetector,
    KolmogorovSmirnovDriftDetector,
)
from scripts.train_default_fusion import N_FEATURES, SEED

# The shipped checkpoint's training distribution: four Gaussian clusters
# (scripts/train_default_fusion.build_dataset draws the centres first).
_CENTERS = np.random.default_rng(SEED).normal(0.0, 4.0, size=(4, N_FEATURES))


def _normals(n: int, seed: int) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    rows = _CENTERS[rng.integers(0, 4, size=n)] + rng.normal(0.0, 1.0, size=(n, N_FEATURES))
    return rows.astype(np.float32)


@pytest.fixture(scope="module")
def shipped() -> OmniMercuryEngine:
    engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
    assert engine.load_default_fusion_checkpoint()
    engine.enable_decision_layer()
    return engine


def test_drift_detection_is_on_by_default_and_can_be_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert OmniMercuryEngine(mode="fusion").drift_detector is not None
    assert (
        OmniMercuryEngine(mode="fusion", config=EngineConfig(drift_detection=False)).drift_detector
        is None
    )
    monkeypatch.setenv("MERCURY_DRIFT_DETECTION", "0")
    assert OmniMercuryEngine(mode="fusion").drift_detector is None


def test_a_shifted_batch_is_detected_and_deferred(shipped: OmniMercuryEngine) -> None:
    result = shipped.detect_with_fusion(_normals(50, seed=11) + 2.0)
    drift = result["drift_detection"]
    assert drift["status"] == "tested" and drift["is_drift"] is True
    assert drift["reference_source"] == "training_rows:model_normals"
    assert result["decision"]["disposition"] == "defer"


def test_in_distribution_batches_are_not_flagged(shipped: OmniMercuryEngine) -> None:
    for seed, n in ((21, 50), (22, 200), (23, 500)):
        drift = shipped.detect_with_fusion(_normals(n, seed))["drift_detection"]
        assert drift["status"] == "tested"
        assert drift["is_drift"] is False and drift["severity"] == "NONE"


def test_too_few_normal_rows_are_not_tested(shipped: OmniMercuryEngine) -> None:
    drift = shipped.detect_with_fusion(_normals(1, seed=31))["drift_detection"]
    assert drift["status"] == "insufficient_rows" and drift["is_drift"] is None


def test_a_different_width_is_incomparable_not_drift(shipped: OmniMercuryEngine) -> None:
    for rows in (40, 5):  # a small batch is still reported as incomparable, not too short
        payload = shipped._drift_payload(np.zeros((rows, 10)), None)
        assert payload is not None
        assert payload["status"] == "incomparable" and payload["is_drift"] is None


def test_an_inference_batch_never_becomes_the_reference() -> None:
    engine = OmniMercuryEngine(mode="fusion")  # nothing fitted: no training rows
    batch = _normals(60, seed=41)
    for _ in range(2):
        payload = engine._drift_payload(batch, None)
        assert payload is not None and payload["status"] == "no_baseline"
    assert engine._drift_reference is None


def test_loading_a_model_invalidates_the_reference(shipped: OmniMercuryEngine) -> None:
    assert shipped._ensure_drift_reference()
    shipped.load_default_fusion_checkpoint()
    assert shipped._drift_reference is None
    assert shipped._ensure_drift_reference()  # re-derived from the reloaded training rows


def test_drift_checks_are_counted() -> None:
    prometheus = pytest.importorskip("prometheus_client")
    engine = OmniMercuryEngine(mode="fusion")
    before = prometheus.REGISTRY.get_sample_value(
        "omni_drift_checks_total", {"status": "no_baseline"}
    )
    engine._drift_payload(_normals(30, seed=51), None)
    after = prometheus.REGISTRY.get_sample_value(
        "omni_drift_checks_total", {"status": "no_baseline"}
    )
    assert after == (before or 0.0) + 1.0


# --------------------------------------------------------------------------- #
# The drift module and the decider.
# --------------------------------------------------------------------------- #
def test_width_mismatch_raises_instead_of_comparing_against_column_zero() -> None:
    detector = KolmogorovSmirnovDriftDetector().fit(np.random.default_rng(0).normal(size=(100, 4)))
    with pytest.raises(DriftIncomparableError):
        detector.detect(np.random.default_rng(1).normal(size=(50, 6)))


def test_ensemble_reports_no_severity_without_drift() -> None:
    rng = np.random.default_rng(0)
    detector = EnsembleDriftDetector().fit(rng.normal(size=(600, 5)))
    for _ in range(200):
        found = detector.detect(rng.normal(size=(50, 5)))
        if not found.is_drift:
            assert found.severity is DriftSeverity.NONE


def test_decider_does_not_defer_on_severity_without_detected_drift() -> None:
    def state(drift: dict[str, Any]) -> ThreeState:
        result = {
            "anomaly_prob": 0.92,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.9,
            "conformal": {"prediction_set": [1], "set_size": 1, "abstain": False, "coverage": 0.9},
            "drift_detection": drift,
        }
        return DecisionAbstentionResponder().decide(result).state

    assert state({"is_drift": False, "severity": "HIGH"}) is ThreeState.GROUNDED
    assert state({"is_drift": True, "severity": "HIGH"}) is ThreeState.UNAVAILABLE
