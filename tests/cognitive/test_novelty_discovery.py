# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Curiosity novelty: measured, read-only scorable, and on the record.

Before this, the CuriosityEngine was opt-in and advisory: no number said its
score separates unseen structure from seen (so nothing could justify serving
it), scoring an observation always mutated the running estimate (so a
measurement drifted toward what it measured), and the decision record's
cognitive summary carried no novelty at all.  These tests pin the read-only
scorer, the measured artifact and its floor, the record surfacing, and the
recorded adoption.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.cognitive.cognitive_evolution_engine import CuriosityEngine

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = _ROOT / "artifacts" / "novelty_discovery.json"


class TestReadOnlyScorer:
    def test_novelty_does_not_move_the_estimate(self) -> None:
        engine = CuriosityEngine()
        rng = np.random.default_rng(0)
        for row in rng.normal(size=(50, 4)):
            engine.explore("fit", row)
        seen = engine.observations_seen
        far = np.full(4, 25.0)
        first = engine.novelty("held", far)
        second = engine.novelty("held", far)
        assert engine.observations_seen == seen
        assert first.novelty_score == pytest.approx(second.novelty_score)
        assert first.measured is True
        assert first.is_novel is True

    def test_explore_does_move_it(self) -> None:
        engine = CuriosityEngine()
        rng = np.random.default_rng(0)
        for row in rng.normal(size=(50, 4)):
            engine.explore("fit", row)
        far = np.full(4, 25.0)
        first = engine.explore("held", far).novelty_score
        # The estimate absorbed the far point, so the same point is less novel.
        second = engine.explore("held", far).novelty_score
        assert second < first

    def test_dimension_mismatch_is_unmeasured_not_a_reset(self) -> None:
        engine = CuriosityEngine()
        for row in np.random.default_rng(1).normal(size=(10, 4)):
            engine.explore("fit", row)
        seen = engine.observations_seen
        result = engine.novelty("held", np.zeros(7))
        assert result.measured is False
        assert engine.observations_seen == seen  # read-only even on mismatch

    def test_warmup_is_declared_unmeasured(self) -> None:
        engine = CuriosityEngine()
        result = engine.novelty("held", np.zeros(3))
        assert result.measured is False
        assert result.is_novel is False


class TestMeasuredArtifact:
    @pytest.fixture(scope="class")
    def artifact(self) -> dict:
        assert _ARTIFACT.is_file(), "run research/cognitive/harness_novelty_discovery.py"
        return json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_fifteen_evaluations_two_protocols(self, artifact: dict) -> None:
        assert artifact["protocol"]["n_evaluations"] == 15
        for values in artifact["per_dataset"].values():
            assert set(values) == {"read_only_auroc", "absorbing_auroc"}

    def test_the_signal_clears_a_real_floor(self, artifact: dict) -> None:
        """The adoption rested on ~0.93 mean; 0.85 is the never-regress floor."""
        assert artifact["mean"]["read_only_auroc"] >= 0.85

    def test_ledger_records_the_adoption(self, artifact: dict) -> None:
        from omni_mercury_engine.governance.improvement_ledger import published

        data = published()
        assert data is not None
        entry = next(
            d for d in data["decisions"] if d["component"] == "cognitive.curiosity_novelty"
        )
        assert entry["verdict"] == "adopted"
        assert entry["impact"]["read_only_mean_auroc"] == pytest.approx(
            artifact["mean"]["read_only_auroc"], abs=1e-4
        )


class TestRecordSurfacing:
    def test_cognitive_summary_carries_novelty_and_causal_factors(self) -> None:
        from omni_mercury_engine.decision.evidence import _cognitive_summary

        summary = _cognitive_summary(
            {
                "is_reliable": True,
                "confidence": 0.8,
                "novelty_score": 0.91,
                "is_novel": True,
                "causal_factors": ["pressure drop precedes the spike", "x" * 500],
            }
        )
        assert summary is not None
        assert summary["novelty_score"] == pytest.approx(0.91)
        assert summary["is_novel"] is True
        assert len(summary["causal_factors"]) == 2
        assert len(summary["causal_factors"][1]) <= 160  # bounded

    def test_absent_component_stays_absent(self) -> None:
        from omni_mercury_engine.decision.evidence import _cognitive_summary

        summary = _cognitive_summary({"is_reliable": True})
        assert summary is not None
        assert summary["novelty_score"] is None
        assert summary["is_novel"] is None
        assert summary["causal_factors"] == []


class TestHypothesisOnTheRecord:
    def test_chain_conclusion_reaches_the_summary(self) -> None:
        from omni_mercury_engine.decision.evidence import _cognitive_summary

        summary = _cognitive_summary(
            {
                "is_reliable": True,
                "reasoning_chain": [
                    {"step": 0, "rule": "modus_ponens", "conclusion": "a", "confidence": 0.9},
                    {
                        "step": 1,
                        "rule": "abduction",
                        "conclusion": "sensor drift best explains the deviation",
                        "confidence": 0.74,
                    },
                ],
            }
        )
        assert summary is not None
        hypothesis = summary["hypothesis"]
        assert hypothesis["conclusion"].startswith("sensor drift")
        assert hypothesis["confidence"] == pytest.approx(0.74)
        assert hypothesis["rule"] == "abduction"
        assert hypothesis["chain_depth"] == 2

    def test_no_chain_no_fabricated_hypothesis(self) -> None:
        from omni_mercury_engine.decision.evidence import _cognitive_summary

        summary = _cognitive_summary({"is_reliable": True, "reasoning_chain": []})
        assert summary is not None
        assert summary["hypothesis"] is None
