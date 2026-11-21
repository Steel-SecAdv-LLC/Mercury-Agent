# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The console's decision panel shows every field the decision record carries.

A ``DecisionRecord`` is the operator's audit surface, and the console is where
most operators read one.  These tests render a maximal record — every field of
``decision/record.py`` set, every signal of ``decision/evidence.py`` populated —
through the generated summariser (``mass.js``) and pin three things:

* **completeness against the code**: every field name ``DecisionRecord.to_dict``
  emits, and every signal ``Evidence.to_dict`` emits, surfaces in some row of
  the panel (so a field added to the record fails here until the console shows
  it);
* the panel is organised as the three answers an operator needs before acting
  on a score — why this row, how sure, the record — and the decision-trail
  summary (``GET /detect/flagship/decisions``) renders its rates, churn and
  admission counters;
* nothing renders as ``[object Object]``, ``undefined`` or ``NaN``.

Browser-free: Node executes the real generated module, exactly as
``tests/api/test_console_discovery_rendering.py`` does.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CONSOLE = _REPO / "src" / "omni_mercury_engine" / "api" / "frontend_assets" / "static" / "console"
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="node is not installed")

#: Record fields whose value the panel need not echo verbatim (structured
#: containers whose CONTENTS are shown instead).
_CONTAINERS = {"signals", "response", "reasons", "caveats"}

#: Signal keys rendered under another name; each maps to text its row must carry.
_SIGNAL_ROW_TEXT = {
    "anomaly_prob": "P(anomaly)",
    "is_anomaly": "flagged",
    "threshold": "threshold",
    "severity": "severity",
    "calibrated": "calibrated",
    "conformal_set_size": "conformal set",
    "conformal_labels": "conformal set",
    "coverage": "coverage",
    "conformal_regime": "certificate",
    "conformal_rows": "batch of",
    "ethical_gate_passed": "gate passed",
    "ethical_score": "score",
    "ethical_threshold": "threshold",
    "ethics_report": "check",
    "symbolic_satisfaction": "symbolic agreement",
    "drift_detected": "drift",
    "drift_severity": "drift",
    "gosnn_anomaly_prob": "gosnn",
    "gosnn_demote_act_below": "demoted below",
    "gosnn_demote_clear_above": "negative above",
    "skipped_detectors": "skipped detectors",
    "skipped_detail": "skipped detectors",
    "detector_uncertainties": "detector uncertainty",
    "out_of_support_rows": "out of support",
    "model_provenance": "model",
    "contributions": "contributions",
    "cognitive": "critique",
    "domain": "domain",
    "policy": "policy",
}


def _maximal_record() -> dict[str, Any]:
    """A decision record with every field and signal populated, as served."""
    artifact = {
        "schema": "mercury.discovery/v1",
        "kind": "causal",
        "procedure": "granger_f_bonferroni_holm/v2",
        "claim": {"assessed": True, "edges": [], "pairs_tested": 20},
        "inputs": {"alpha": 0.05},
        "challenge": {"falsifiable_by": "replay"},
        "replayable": "from_record",
        "digest": "ab" * 32,
    }
    return {
        "state": "unavailable",
        "disposition": "defer",
        "decision_label": 1,
        "abstained": True,
        "anomaly_prob": 0.97,
        "threshold": 0.5,
        "decision_confidence": None,
        "coverage": 0.9,
        "calibrated": True,
        "severity": 0.41,
        "response": {
            "action": "request_input",
            "urgency": "elevated",
            "requires_human": True,
            "notify": True,
            "fail_closed": True,
            "countermeasures": ["Route to a qualified human."],
            "rationale": "Calibration may not hold under drift.",
        },
        "reasons": ["conformal singleton {1} at 90% coverage (calibrated decision)"],
        "caveats": ["the certificate is per-sample and this verdict is the batch-max row"],
        "signals": {
            "anomaly_prob": 0.97,
            "is_anomaly": True,
            "threshold": 0.5,
            "severity": 0.41,
            "calibrated": True,
            "conformal_set_size": 1,
            "conformal_labels": [1],
            "coverage": 0.9,
            "conformal_regime": "per_sample",
            "conformal_rows": 80,
            "ethical_gate_passed": True,
            "ethical_score": 0.999,
            "ethical_threshold": 0.93,
            "ethics_report": {"check": "harm_uplift", "ruleset_version": "2026.3", "details": "x"},
            "symbolic_satisfaction": 0.8,
            "drift_detected": True,
            "drift_severity": "CRITICAL",
            "gosnn_anomaly_prob": 0.9,
            "gosnn_demote_act_below": 0.2,
            "gosnn_demote_clear_above": 0.8,
            "skipped_detectors": ["temporal"],
            "skipped_detail": {"temporal": {"reason": "timeout", "trained_group": True}},
            "detector_uncertainties": {"statistical": 0.5, "temporal": 0.8},
            "out_of_support_rows": [7, 9],
            "model_provenance": {
                "checkpoint": "default_fusion.pt",
                "checkpoint_sha256": "5a" * 32,
                "temperature": 0.1,
                "conformal": {
                    "coverage_target": 0.9,
                    "coverage_achieved": 0.9,
                    "n_calibration": 251,
                },
                "trained_groups": ["statistical", "temporal"],
            },
            "contributions": {
                "detector_importance": {"neural": 0.3, "statistical": 0.2},
                "integrated_gradients": None,
                "shapley": None,
            },
            "cognitive": {
                "is_reliable": True,
                "confidence": 0.88,
                "confidence_calibrated": False,
                "calibration_evidence": None,
                "epistemic_uncertainty": 0.1,
                "epistemic_measured": False,
                "epistemic_evidence": None,
                "aleatoric_uncertainty": 0.1,
                "aleatoric_measured": False,
                "aleatoric_evidence": None,
                "ethical_permissible": True,
                "novelty_score": 0.8,
                "is_novel": True,
                "novelty_measured": True,
                "novelty_evidence": {"method": "conformal_rank", "n": 24, "digest": "ab" * 32},
                "novelty_challenge": {
                    "standardized_distance": 3.2,
                    "p_value": 0.02,
                    "alpha": 0.05,
                    "falsifiable_by": "replay",
                },
                "causal_factors": ["X0 → X1 (lag=1, partial R²=0.25, adjusted p=0.00063)"],
                "hypothesis": {
                    "conclusion": "X0 leads X1 at lag 1.",
                    "confidence": None,
                    "rule": "flagged_and_lagged_dependence",
                    "chain_depth": 1,
                    "falsifier": {
                        "weakest_step": "The detection flagged this input.",
                        "rule": "anomaly_flagged",
                        "confidence": None,
                        "margin": 1.94,
                        "withdrawn_if": "anomaly_flagged: statistic < 0.5",
                    },
                },
                "discovery": {"causal": artifact},
                "similar_cases": ["case-7"],
                "recommendations": ["inspect the X0 feed"],
            },
            "domain": "network_security",
            "policy": {
                "indecision_margin": 0.05,
                "symbolic_agreement_floor": 0.5,
                "defer_on_drift": True,
                "drift_defer_severities": ["CRITICAL", "HIGH", "MEDIUM"],
                "defer_on_gosnn_disagreement": True,
                "require_calibrated_for_act": False,
                "fail_closed_on_atypical": True,
                "fail_closed_on_ethical_block": True,
                "defer_on_unreliable_critique": True,
                "critique_support_confidence_floor": 0.75,
            },
        },
        "domain": "network_security",
    }


_HARNESS = """
'use strict';
global.window = { matchMedia: () => ({ matches: false }) };
global.document = { documentElement: {} };
const CONSOLE = process.argv[2];
require(CONSOLE + "/mass.js");
const M = window.MercuryCLIMass;
const record = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const out = {};
out.panel = M.summarize({
  anomaly_prob: 0.97, is_anomaly: true, threshold_used: 0.5, top_row: 12,
  anomaly_probs: Array.from({ length: 80 }, (_, i) => (i === 12 ? 0.97 : 0.02)),
  is_anomaly_rows: Array.from({ length: 80 }, (_, i) => i === 12),
  decision: record,
}).kv;
out.trail = M.summarize({
  total: 5, by_state: { grounded: 3, undecidable: 2 }, by_disposition: { act: 3, hold: 2 },
  by_response_action: { alert: 3, hold: 2 }, abstention_rate: 0.4, calibrated_rate: 1.0,
  churn: 0.25, recorded: true,
  admission: { max_queue: 8, admitted_total: 5, rejected_total: 1, rejection_rate: 1 / 6,
    in_flight: 0, peak_in_flight: 2 },
});
out.empty_trail = M.summarize({ total: 0, recorded: false,
  admission: { max_queue: 8, admitted_total: 0, rejected_total: 0, rejection_rate: 0,
    in_flight: 0, peak_in_flight: 0 } }).kv;
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    assert _NODE is not None
    folder = tmp_path_factory.mktemp("decision-panel")
    (folder / "harness.js").write_text(_HARNESS, encoding="utf-8")
    (folder / "record.json").write_text(json.dumps(_maximal_record()), encoding="utf-8")
    done = subprocess.run(
        [_NODE, str(folder / "harness.js"), str(_CONSOLE), str(folder / "record.json")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    data: dict[str, Any] = json.loads(done.stdout)
    return data


def _text(rows: list[list[str]]) -> str:
    return "\n".join(f"{key} {value}" for key, value in rows)


class TestEveryRecordFieldSurfaces:
    def test_every_record_field_has_a_row(self, rendered: dict[str, Any]) -> None:
        """Pinned against decision/record.py: a new field fails until shown."""
        from omni_mercury_engine.decision.record import DecisionRecord

        record = DecisionRecord.from_dict(_maximal_record())
        panel = _text(rendered["panel"]).lower()
        checks = {
            "state": "unavailable",
            "disposition": "defer",
            "decision_label": "label 1",
            "abstained": "abstained true",
            "anomaly_prob": "p(anomaly) 0.970",
            "threshold": "vs threshold 0.500",
            "decision_confidence": "none stated: the decision abstained",
            "coverage": "coverage 90.0%",
            "calibrated": "calibrated true",
            "severity": "severity 0.410",
            "response": "request_input · needs a human · urgency elevated",
            "reasons": "conformal singleton {1}",
            "caveats": "batch-max row",
            "signals": "model default_fusion.pt",
            "domain": "network_security",
        }
        emitted = set(record.to_dict())
        assert emitted == set(checks), "a record field appeared or went; re-pin this test"
        for field, needle in checks.items():
            assert needle in panel, f"the record's {field!r} does not surface"

    def test_every_signal_surfaces(self, rendered: dict[str, Any]) -> None:
        """Pinned against decision/evidence.py: a new signal fails until shown."""
        from omni_mercury_engine.decision.evidence import Evidence

        # The decider attaches its policy beside the evidence signals (decider.py).
        signals = set(Evidence.from_detection({"anomaly_prob": 0.5}).to_dict()) | {"policy"}
        assert signals == set(_SIGNAL_ROW_TEXT), "a signal appeared or went; re-pin this test"
        panel = _text(rendered["panel"]).lower()
        for signal, needle in _SIGNAL_ROW_TEXT.items():
            assert needle.lower() in panel, f"signal {signal!r} does not surface"

    def test_the_three_answers_structure_the_panel(self, rendered: dict[str, Any]) -> None:
        keys = [key for key, _value in rendered["panel"]]
        why = keys.index("1 · WHY THIS ROW")
        sure = keys.index("2 · HOW SURE")
        record = keys.index("3 · THE RECORD")
        assert why < keys.index("· contributions") < sure
        assert why < keys.index("· causal factors") < sure
        assert sure < keys.index("· conformal set") < record
        assert sure < keys.index("· critique") < record
        assert record < keys.index("· model")
        assert record < keys.index("· replay")

    def test_specific_renderings_say_what_the_numbers_mean(self, rendered: dict[str, Any]) -> None:
        rows = dict(rendered["panel"])
        assert rows["· row"].startswith("row 12 of 80 carries the verdict")
        assert "1 row(s) flagged" in rows["· row"]
        assert "singleton: one class at the target coverage" in rows["· conformal set"]
        assert "calibrated on 251 held-out points" in rows["· conformal set"]
        assert rows["· certificate"] == "per-sample · batch of 80 (coverage is not batch-level)"
        assert rows["· drift"].startswith("DETECTED · CRITICAL")
        assert "temporal (timeout)" in rows["· skipped detectors"]
        assert "partial evidence" in rows["· skipped detectors"]
        assert rows["· out of support"].startswith("2 row(s) outside the training envelope")
        assert "NOVEL" in rows["· novelty"] and "p=0.0200 vs level 0.05" in rows["· novelty"]
        assert "ranked among 24 reference distances" in rows["· novelty"]
        assert "epistemic unmeasured" in rows["· critique"]
        assert "uncalibrated" in rows["· critique"]
        assert "gate passed" in rows["· ethics"] and "0.999 ≥ threshold 0.930" in rows["· ethics"]
        assert "defers on drift at CRITICAL/HIGH/MEDIUM" in rows["· policy gates"]
        assert "CHALLENGE THIS DECISION" in rows["· replay"]
        assert "mercury-agent replay" in rows["· replay"]

    def test_nothing_renders_as_an_object_placeholder(self, rendered: dict[str, Any]) -> None:
        for name in ("panel", "empty_trail"):
            text = _text(rendered[name])
            for bad in ("[object Object]", "undefined", "NaN"):
                assert bad not in text, (name, bad)


class TestTheTrailSummaryRenders:
    def test_rates_churn_and_admission(self, rendered: dict[str, Any]) -> None:
        rows = dict(rendered["trail"]["kv"])
        assert rendered["trail"]["headline"] == "5 decision(s) in the trail"
        assert rows["· by state"] == "grounded 3 · undecidable 2"
        assert rows["· abstention rate"] == "40.0%"
        assert rows["· calibrated rate"] == "100.0%"
        assert rows["· churn"] == "25.0% of consecutive decisions changed disposition"
        assert rows["· admission"].startswith("5 admitted · 1 refused at the queue bound of 8")

    def test_an_empty_trail_says_so(self, rendered: dict[str, Any]) -> None:
        rows = dict(rendered["empty_trail"])
        assert rows["decisions"].startswith("none recorded")
