# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Golden-dataset regression for the Cognitive Orchestrator's served payload.

The orchestrator runs on the served flagship by default, and its payload
feeds the DecisionRecord's cognitive summary (reliability, the uncertainties
with their measured/placeholder flags, novelty, causal factors, the
reasoning chain the hypothesis is read from).  A silent change to that
payload — a dropped key, a renamed flag, a placeholder presented as a
measurement, non-determinism — would degrade every served decision's
transparency without failing any existing test.  This pins the contract on a
fixed input: the exact key set, byte-stable determinism, the honest
measured/unmeasured flags, and the bounded discovery signals.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the orchestrator's uncertainty stack needs [ml]")

from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

#: The committed payload schema: exactly these keys, for the default
#: served configuration with curiosity on.  A new key is an intentional
#: schema change that updates this set in the same commit; a dropped key is
#: a regression.
_GOLDEN_KEYS = {
    "aleatoric_measured",
    "aleatoric_uncertainty",
    "analysis_time_ms",
    "anomaly_detected",
    "anomaly_score",
    "benevolence_score",
    "causal_factors",
    "confidence",
    "confidence_calibrated",
    "epistemic_measured",
    "epistemic_uncertainty",
    "ethical_permissible",
    "feedback_signals",
    "is_novel",
    "is_reliable",
    "knowledge_updates",
    "novelty_distance",
    "novelty_score",
    "novelty_threshold",
    "reasoning_chain",
    "recommendations",
    "severity",
    "similar_cases",
    "symbolic_consistency",
    "warnings",
}


def _analyze() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orchestrator = CognitiveOrchestrator(enable_curiosity=True)
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(40, 4))
        raw[20] += 15.0  # one clear outlier row
        detection = {"is_anomaly": True, "anomaly_prob": 0.93, "severity": 0.8}
        return orchestrator.analyze(
            detection_result=detection,
            raw_data=raw,
            context={"domain": "general"},
        ).to_dict()


def _stable(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload minus the one non-deterministic field (wall-clock timing)."""
    return {k: v for k, v in payload.items() if k != "analysis_time_ms"}


class TestOrchestratorGolden:
    def test_payload_schema_is_exactly_the_golden_key_set(self) -> None:
        payload = _analyze()
        assert set(payload) == _GOLDEN_KEYS, (
            "orchestrator payload schema changed; update _GOLDEN_KEYS in the "
            "same commit if intentional. "
            f"added={set(payload) - _GOLDEN_KEYS} dropped={_GOLDEN_KEYS - set(payload)}"
        )

    def test_same_input_is_byte_stable(self) -> None:
        assert _stable(_analyze()) == _stable(_analyze())

    def test_transparency_flags_are_present_and_boolean(self) -> None:
        payload = _analyze()
        for flag in ("epistemic_measured", "aleatoric_measured", "confidence_calibrated"):
            assert isinstance(payload[flag], bool)

    def test_single_point_uncertainty_is_declared_unmeasured(self) -> None:
        """A single detection point cannot measure epistemic/aleatoric spread;
        the flags must say so rather than present the placeholder as fact."""
        payload = _analyze()
        assert payload["epistemic_measured"] is False
        assert payload["aleatoric_measured"] is False

    def test_discovery_signals_are_bounded(self) -> None:
        payload = _analyze()
        assert isinstance(payload["causal_factors"], list)
        assert len(payload["causal_factors"]) <= 5
        # novelty_score is present (curiosity on) and in range or the neutral
        # warm-up placeholder; never NaN/inf on the served payload.
        score = payload["novelty_score"]
        assert score is None or (isinstance(score, float) and 0.0 <= score < 1.0)

    def test_clean_run_omits_novelty(self) -> None:
        """Curiosity scores only flagged anomalies; a clean run carries none."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orchestrator = CognitiveOrchestrator(enable_curiosity=True)
            payload = orchestrator.analyze(
                detection_result={"is_anomaly": False, "anomaly_prob": 0.05, "severity": 0.0},
                raw_data=np.zeros((10, 4)),
                context={"domain": "general"},
            ).to_dict()
        assert "novelty_score" not in payload
