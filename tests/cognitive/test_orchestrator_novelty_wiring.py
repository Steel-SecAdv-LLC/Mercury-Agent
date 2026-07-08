# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""CuriosityEngine and EnhancedAnomalyDetector are invoked on the analyze() path.

Both were runtime-orphaned. They are now optional CognitiveOrchestrator
components (opt-in) that run inside ``analyze()`` when an anomaly is detected.
These pin that they are actually invoked -- their counters advance and their
output lands on the result -- not merely importable, and that they stay off by
default.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

_ANOMALY = {"is_anomaly": True, "anomaly_prob": 0.9, "severity": 0.8}


def test_components_off_by_default() -> None:
    orch = CognitiveOrchestrator()
    assert orch.curiosity is None
    assert orch.enhanced_detector is None


def test_default_analyze_payload_schema_is_unchanged() -> None:
    """With both opt-in components off, to_dict() must not carry the new keys.

    Emitting them unconditionally would change the default analyze() JSON
    schema for strict consumers even though nothing was opted into.
    """
    orch = CognitiveOrchestrator()
    result = orch.analyze(_ANOMALY, raw_data=np.zeros((8, 4)), context={"domain": "cyber"})
    payload = result.to_dict()
    assert "novelty_score" not in payload
    assert "is_novel" not in payload
    assert "predictive_forecast" not in payload


def test_enabled_components_add_their_keys() -> None:
    """Opting in surfaces the keys — presence tracks the component running."""
    orch = CognitiveOrchestrator(enable_curiosity=True, enable_enhanced_detection=True)
    result = orch.analyze(_ANOMALY, raw_data=np.zeros((8, 4)), context={"domain": "cyber"})
    payload = result.to_dict()
    assert "novelty_score" in payload
    assert "is_novel" in payload
    assert "predictive_forecast" in payload


def test_curiosity_invoked_in_analyze() -> None:
    orch = CognitiveOrchestrator(enable_curiosity=True)
    assert orch.curiosity is not None

    result = orch.analyze(_ANOMALY, raw_data=np.zeros((16, 4)), context={"domain": "cyber"})
    # The novelty field is populated and the engine actually ran an exploration.
    assert "novelty_score" in result.to_dict()
    assert orch.curiosity.get_statistics()["explorations_performed"] >= 1
    assert orch.get_statistics()["curiosity"]["explorations_performed"] >= 1


def test_curiosity_flags_out_of_distribution_observations() -> None:
    orch = CognitiveOrchestrator(enable_curiosity=True)
    rng = np.random.default_rng(0)
    # Establish a baseline of ordinary observations...
    for _ in range(30):
        orch.analyze(_ANOMALY, raw_data=rng.normal(0.0, 1.0, (8, 4)), context={"domain": "cyber"})
    # ...then a wildly out-of-distribution one should score as novel.
    outlier = orch.analyze(_ANOMALY, raw_data=np.full((8, 4), 25.0), context={"domain": "cyber"})
    assert outlier.is_novel is True
    assert outlier.novelty_score is not None  # curiosity ran -> field populated
    assert outlier.novelty_score > 0.7


def test_enhanced_detector_invoked_in_analyze() -> None:
    orch = CognitiveOrchestrator(enable_enhanced_detection=True)
    assert orch.enhanced_detector is not None

    result = orch.analyze(_ANOMALY, raw_data=np.zeros((16, 4)), context={"domain": "cyber"})
    forecast = result.to_dict()["predictive_forecast"]
    assert forecast  # non-empty
    assert "probability" in forecast
    assert orch.enhanced_detector.get_statistics()["predictions_made"] >= 1
    assert "enhanced_detection" in orch.get_statistics()


def test_enhanced_detector_does_no_network_io() -> None:
    # use_simulated_sources=False and include_external=False -> no external fetch.
    orch = CognitiveOrchestrator(enable_enhanced_detection=True)
    assert orch.enhanced_detector is not None
    assert orch.enhanced_detector.use_simulated_sources is False


def test_predictor_observes_non_anomalies_too() -> None:
    """The Bayesian predictor must see failures, not only anomalies.

    Updating only on anomalies feeds a success-only stream and drives the
    Beta-Bernoulli forecast toward 1.0. After many normal (non-anomaly)
    analyses, a single anomaly's forecast probability must stay well below 1.0.
    """
    orch = CognitiveOrchestrator(enable_enhanced_detection=True)
    normal = {"is_anomaly": False, "anomaly_prob": 0.1, "severity": 0.1}
    for _ in range(20):
        orch.analyze(normal, raw_data=np.zeros((8, 4)), context={"domain": "cyber"})

    result = orch.analyze(_ANOMALY, raw_data=np.zeros((8, 4)), context={"domain": "cyber"})
    forecast = result.to_dict()["predictive_forecast"]
    assert forecast  # anomaly -> forecast surfaced
    # The surfaced probability is the CALIBRATED Bayesian estimate: after ~20
    # normals and 1 anomaly the base rate is low, so it tracks near it -- not the
    # HMM-saturated blend (which is kept separately as blended_score).
    assert forecast["probability"] < 0.4
    assert "blended_score" in forecast
    assert 0.0 <= forecast["probability"] <= 1.0
