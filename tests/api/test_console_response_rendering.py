# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The console renders the flagship decision surface, not placeholders for it.

Five behaviors, each driven through the real shipped modules (``mass.js``,
``viz.js``, ``api.js``) in Node with the same window stub the chart-CSP test
uses -- no browser, no network, fully deterministic:

* a ``decision`` record renders whole: state, disposition, abstention,
  calibration, coverage as a percentage, up to five reasons and caveats,
  each clipped at ~100 characters (it used to collapse to one word);
* an object ``explanation`` (the flagship Integrated-Gradients dict) renders
  method, top attributions and faithfulness, never ``[object Object]``, and a
  ``gdpr_report`` explanation block renders the same way;
* the contribution chart reads ``detector_importance`` (the flagship
  response's map), after ``detector_contributions``;
* ``is_stable`` is shown only when ``stability_evaluated`` is not false; a
  short series says "not evaluated (series too short)" instead of UNSTABLE,
  and a response without the flag (an older server) keeps today's reading;
* a 403 maps to kind ``forbidden`` (payload kept), 401 alone stays ``auth``.

``tests/api/test_console_decision_forbidden_browser.py`` drives the same
behaviors through ``logic.js`` in the served page.
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

#: A reason over 100 characters, to see the clip (99 chars + ellipsis).
_LONG_REASON = (
    "calibrated probability 0.910 cleared the threshold 0.500 with a singleton "
    "conformal set backing the call at the stated coverage level"
)

_HARNESS = (
    """
'use strict';
global.window = { matchMedia: () => ({ matches: false }) };
global.document = { documentElement: {} };
global.getComputedStyle = () => ({ getPropertyValue: () => "" });
const CONSOLE = process.argv[2];
require(CONSOLE + "/mass.js");
require(CONSOLE + "/viz.js");
const M = window.MercuryCLIMass, V = window.MercuryCLIViz;
const out = {};

// 1. The decision record, as decision/record.py serializes it.
const decision = {
  state: "GROUNDED", disposition: "act", decision_label: 1, abstained: false,
  anomaly_prob: 0.91, threshold: 0.5, decision_confidence: 0.9, coverage: 0.9,
  calibrated: true, severity: 0.8,
  reasons: ["""
    + json.dumps(_LONG_REASON)
    + """, "r2", "r3", "r4", "r5", "r6"],
  caveats: ["ethical gate not run on this payload"],
};
out.decision = M.summarize({ anomaly_prob: 0.91, is_anomaly: true, threshold_used: 0.5, decision: decision }).kv;

// 2. The flagship explain: true object (Explanation.to_dict() + faithfulness_scores).
const explanation = {
  explanation_type: "integrated_gradients",
  feature_importances: [
    { feature: "f3", importance: 0.42, direction: "up", confidence: 0.9 },
    { feature: "f1", importance: -0.17 }, { feature: "f0", importance: 0.05 },
    { feature: "f2", importance: 0.04 }, { feature: "f4", importance: 0.01 },
    { feature: "f5", importance: 0.001 }],
  human_readable: "f3 drove the probability up",
  faithfulness_scores: { comprehensiveness: 0.81, sufficiency: 0.12 },
};
out.explanation = M.summarize({ is_anomaly: true, explanation: explanation }).kv;

// 3. The gdpr_report's explanation block (ExplanationReport.to_dict()["explanation"]).
out.gdpr = M.summarize({ is_anomaly: false, gdpr_report: { explanation: {
  logic: "shapley over score_fusion", feature_contributions: { f0: 0.1, f1: -0.3 },
  top_factors: [{ feature: "f1", contribution: -0.3, description: "d" }] } } }).kv;

// 4. Stability gate: evaluated false / true / absent (an older server).
out.stabShort = M.summarize({ fusion_score: 0.41, is_stable: false, stability_evaluated: false });
out.stabEval = M.summarize({ fusion_score: 0.41, is_stable: false, stability_evaluated: true });
out.stabLegacy = M.summarize({ fusion_score: 0.41, is_stable: false });

// 5. Charts: detector_importance, and the stability wording in the 3R charts.
out.contribImportance = V.render("contrib", { detector_importance: { statistical: 0.52, temporal: 0.31, dimensional: 0.17 } });
out.contribPriority = V.render("contrib", { detector_contributions: { a: 0.6 }, detector_importance: { b: 0.4 } });
const threeR = { fusion_score: 0.4, recursion_score: 0.3, resonance_score: 0.2,
  optimization_score: 0.1, lyapunov_bound: 0.9, ethical_scaling: 0.95, is_stable: false };
out.threeRShort = V.render("threeR", Object.assign({ stability_evaluated: false }, threeR));
out.threeRLegacy = V.render("threeR", Object.assign({}, threeR));
out.fusionnetShort = V.render("fusionnet", { fusion_score: 0.4, weights: { recursion: 1, resonance: 1 },
  recursion_score: 0.3, resonance_score: 0.2, lyapunov_bound: 0.9, is_stable: false, stability_evaluated: false });

// 6. api.js maps 403 to forbidden (data kept) and 401 to auth.
const responses = {};
global.fetch = async (url) => { const r = responses[String(url)]; return {
  ok: r.status < 400, status: r.status, headers: { get: () => null },
  text: async () => JSON.stringify(r.body) }; };
require(CONSOLE + "/api.js");
(async () => {
  const A = window.MercuryCLIAPI;
  A.set({ baseUrl: "http://engine.test" });
  responses["http://engine.test/api/v1/detect/fusion"] = { status: 403, body: {
    detail: "Request refused by the harm-uplift decision gate (check=harm_uplift).",
    ethics: { check: "harm_uplift", score: 0.97, threshold: 0.5, ruleset_version: "2026.3" } } };
  out.forbidden = await A.post("/api/v1/detect/fusion", { data: [[1]] });
  responses["http://engine.test/api/v1/detect/fusion"] = { status: 401, body: { detail: "API key required" } };
  out.auth = await A.post("/api/v1/detect/fusion", { data: [[1]] });
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e && e.stack || e); process.exit(1); });
"""
)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the harness once over the shipped modules; every test reads it."""
    path = tmp_path_factory.mktemp("console-render") / "harness.js"
    path.write_text(_HARNESS, encoding="utf-8")
    run = subprocess.run(
        [str(_NODE), str(path), str(_CONSOLE)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    parsed: dict[str, Any] = json.loads(run.stdout.strip().splitlines()[-1])
    return parsed


def _kv(pairs: list[list[str]]) -> list[tuple[str, str]]:
    return [(k, v) for k, v in pairs]


def test_decision_record_renders_state_abstention_calibration_coverage_reasons_caveats(
    rendered: dict[str, Any],
) -> None:
    """The decision block keeps the one-line disposition and adds the detail rows."""
    kv = _kv(rendered["decision"])
    assert ("decision", "act") in kv  # the existing single-line disposition survives
    assert ("· state", "GROUNDED") in kv
    assert ("· abstained", "false") in kv
    assert ("· calibrated", "true") in kv
    assert ("· coverage", "90.0%") in kv
    reasons = [v for k, v in kv if k.startswith("· reason")]
    assert len(reasons) == 5, kv  # six served, capped at five
    assert reasons[0] == _LONG_REASON[:99] + "…"  # clipped at ~100 characters
    assert len(reasons[0]) == 100
    assert ("· caveat 1", "ethical gate not run on this payload") in kv


def test_object_explanation_renders_method_top_attributions_and_faithfulness(
    rendered: dict[str, Any],
) -> None:
    """The Integrated-Gradients dict renders substance, never [object Object]."""
    assert "[object Object]" not in json.dumps(rendered["explanation"])
    value = dict(_kv(rendered["explanation"]))["explanation"]
    assert "integrated_gradients" in value
    assert "f3: 0.420" in value and "f1: -0.170" in value
    assert "f5" not in value  # top five by |importance| only
    assert "faithfulness" in value and "comprehensiveness 0.810" in value


def test_gdpr_report_explanation_block_renders(rendered: dict[str, Any]) -> None:
    """The report's Shapley-style block gets a row of its own."""
    value = dict(_kv(rendered["gdpr"])).get("gdpr report", "")
    assert "f1: -0.300" in value, rendered["gdpr"]


def test_stability_is_gated_on_stability_evaluated(rendered: dict[str, Any]) -> None:
    """A short series is 'not evaluated', not UNSTABLE; older servers keep today's reading."""
    short = rendered["stabShort"]
    assert "UNSTABLE" not in short["headline"]
    assert "stability not evaluated" in short["headline"]
    assert ["is_stable", "not evaluated (series too short)"] in short["kv"]
    assert "UNSTABLE" in rendered["stabEval"]["headline"]
    assert "UNSTABLE" in rendered["stabLegacy"]["headline"]  # flag absent: evaluated


def test_detector_importance_reaches_the_contribution_chart(rendered: dict[str, Any]) -> None:
    """The flagship response's map is charted; contributions still win when both exist."""
    chart = rendered["contribImportance"]
    assert isinstance(chart, str) and "DETECTOR IMPORTANCE" in chart and "statistical" in chart
    assert "DETECTOR CONTRIBUTION TO FUSED SCORE" in rendered["contribPriority"]


def test_3r_charts_use_the_same_stability_gate(rendered: dict[str, Any]) -> None:
    """The 3R mechanism and fusion-network charts say 'not evaluated' for a short series."""
    assert "stability not evaluated" in rendered["threeRShort"]
    assert "UNSTABLE" not in rendered["threeRShort"]
    assert "UNSTABLE" in rendered["threeRLegacy"]
    # The fusion-network chart clips its title at 110 characters (it always
    # has), so the full wording is not assertable; the property that matters
    # is that a short series claims neither UNSTABLE nor stable.
    assert "UNSTABLE" not in rendered["fusionnetShort"]
    assert "· stable" not in rendered["fusionnetShort"]


def test_403_is_kind_forbidden_with_the_payload_kept_and_401_stays_auth(
    rendered: dict[str, Any],
) -> None:
    """An ethical-gate 403 is not an authentication failure; its payload survives."""
    forbidden = rendered["forbidden"]
    assert forbidden["kind"] == "forbidden", forbidden
    assert forbidden["status"] == 403
    assert forbidden["data"]["ethics"]["check"] == "harm_uplift"
    assert "harm-uplift" in forbidden["error"]
    assert rendered["auth"]["kind"] == "auth"
    assert rendered["auth"]["status"] == 401
