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

It also pins *which* fields may be reported unmeasured and which of them are
unmeasured on the golden input (``TestUnmeasuredFieldsArePinned``), so no
change can start presenting a placeholder as a measurement without editing
this file.  Writing that pin found one path already doing it: an analysis
given no ``raw_data`` skipped the uncertainty step, kept the result's
default flags, and reported both uncertainties as ``0.0`` *and measured*.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the orchestrator's uncertainty stack needs [ml]")

from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator
from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence

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


#: The fields that may be reported WITHOUT having been measured, each with
#: the flag that says so (``False`` = a placeholder, or an uncalibrated
#: prior).  Exactly these: every other number in the payload is either
#: copied from the detection or computed outright.  A field that gains a
#: placeholder path must be added here with its flag in the same commit; a
#: flag that is dropped or renamed fails the exact-set test below.
_MEASUREMENT_FLAGS = {
    "epistemic_uncertainty": "epistemic_measured",
    "aleatoric_uncertainty": "aleatoric_measured",
    "confidence": "confidence_calibrated",
}

#: Which of them are unmeasured on the golden input: one detection point, no
#: ensemble, no model, no fitted calibrator -- so all three.  Removing a
#: name from this set is the claim "this is now measured here"; it needs the
#: measurement (an ensemble or MC spread, a heteroscedastic residual, a
#: fitted calibrator) landing in the same commit, not a flipped default.
_GOLDEN_UNMEASURED = {"epistemic_uncertainty", "aleatoric_uncertainty", "confidence"}


def _unmeasured(payload: dict[str, Any]) -> set[str]:
    """The fields ``payload`` itself declares unmeasured."""
    return {field for field, flag in _MEASUREMENT_FLAGS.items() if payload[flag] is not True}


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


class TestUnmeasuredFieldsArePinned:
    def test_the_flags_in_the_payload_are_exactly_the_pinned_ones(self) -> None:
        """A new, dropped or renamed transparency flag changes this set."""
        payload = _analyze()
        flags = {key for key in payload if key.endswith(("_measured", "_calibrated"))}
        assert flags == set(_MEASUREMENT_FLAGS.values())
        assert set(_MEASUREMENT_FLAGS) <= set(payload)  # each flagged value is present

    def test_the_golden_input_is_unmeasured_in_exactly_these_fields(self) -> None:
        assert _unmeasured(_analyze()) == _GOLDEN_UNMEASURED, (
            "a field changed between measured and unmeasured on the golden input; "
            "update _GOLDEN_UNMEASURED in the same commit, with the measurement"
        )

    def test_an_analysis_without_raw_data_measures_nothing(self) -> None:
        """Fails on the prior code: no estimate ran, yet both read 0.0 *and measured*."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = (
                CognitiveOrchestrator(enable_curiosity=True)
                .analyze(
                    detection_result={"is_anomaly": True, "anomaly_prob": 0.93, "severity": 0.8},
                    raw_data=None,
                    context={"domain": "general"},
                )
                .to_dict()
            )
        assert _unmeasured(payload) == set(_MEASUREMENT_FLAGS)

    @pytest.mark.parametrize("flag", sorted(_MEASUREMENT_FLAGS.values()))
    def test_the_decision_layer_reads_a_missing_or_mistyped_flag_as_unknown(
        self, flag: str
    ) -> None:
        """Only a literal ``True`` is a measurement; nothing else is promoted to one."""
        payload = _analyze()
        carried = Evidence.from_detection({"cognitive_analysis": payload}).cognitive
        assert carried is not None
        assert carried[flag] is payload[flag]  # carried through unchanged
        for not_a_flag in ("absent", 1, "true", 0.99, None):
            altered = dict(payload)
            if not_a_flag == "absent":
                del altered[flag]
            else:
                altered[flag] = not_a_flag
            summary = Evidence.from_detection({"cognitive_analysis": altered}).cognitive
            assert summary is not None
            assert summary[flag] is None

    def test_the_golden_critique_is_not_counted_as_measured(self) -> None:
        """With every field unmeasured, the critique rests on no measurement."""
        carried = Evidence.from_detection({"cognitive_analysis": _analyze()}).cognitive
        assert carried is not None
        assert DecisionAbstentionResponder._critique_measured(carried) is False
