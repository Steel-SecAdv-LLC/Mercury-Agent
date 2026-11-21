# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The console shows a finding's challenge path, not only that a finding exists.

A decision record carries its discovery artifacts as JSON.  A professional
reading the console should not have to open that JSON to learn what a novelty
verdict rests on, whether the causal test could be read at all, which premise
a hypothesis would fall with, and how to check any of it.  These tests render
real records -- built by the orchestrator and the decision layer, not typed in
-- through the shipped ``mass.js`` in Node, and read the rows the panel shows.

Before this, the panel printed a novelty score with a "cut" it no longer has,
and nothing about replay.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="the orchestrator's uncertainty stack needs [ml]")

from omni_mercury_engine import discovery as D
from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator
from omni_mercury_engine.decision.decider import DecisionAbstentionResponder

_REPO = Path(__file__).resolve().parents[2]
_CONSOLE = _REPO / "src" / "omni_mercury_engine" / "api" / "frontend_assets" / "static" / "console"
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="node is not installed")

_DETECTION = {"is_anomaly": True, "anomaly_prob": 0.93, "severity": 0.8, "threshold_used": 0.6}

_HARNESS = """
'use strict';
global.window = { matchMedia: () => ({ matches: false }) };
global.document = { documentElement: {} };
global.getComputedStyle = () => ({ getPropertyValue: () => "" });
require(process.argv[2] + "/mass.js");
const M = window.MercuryCLIMass;
const cases = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const out = {};
for (const name of Object.keys(cases)) out[name] = M.summarize(cases[name]).kv;
process.stdout.write(JSON.stringify(out));
"""


def _record(*, disclose: bool, shifted_level: bool = False) -> dict[str, Any]:
    """A flagged, novel window with a planted driver, as the served path records it."""
    rng = np.random.default_rng(0)
    context = {"domain": "general", "disclose_reference": disclose}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orchestrator = CognitiveOrchestrator(enable_curiosity=True)
        for _ in range(30):
            orchestrator.analyze(
                detection_result=_DETECTION, raw_data=rng.normal(size=(200, 4)), context=context
            )
        window = rng.normal(size=(200, 4)) + 3.0
        window[1:, 1] += 0.6 * window[:-1, 0]
        if shifted_level:
            window[60:110] += 3.0
        payload = orchestrator.analyze(
            detection_result=_DETECTION, raw_data=window, context=context
        ).to_dict()
    record = DecisionAbstentionResponder().decide({**_DETECTION, "cognitive_analysis": payload})
    wire: dict[str, Any] = json.loads(json.dumps(record.to_dict()))
    return wire


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, str]]:
    full = _record(disclose=True)
    cases = {
        "full": {**_DETECTION, "decision": full},
        "withheld": {**_DETECTION, "decision": _record(disclose=False)},
        "unstable": {**_DETECTION, "decision": _record(disclose=True, shifted_level=True)},
        "replayed": {
            "verdict": D.HOLDS,
            "artifacts": 3,
            "replay": D.replay_record(full),
        },
    }
    folder = tmp_path_factory.mktemp("console")
    (folder / "harness.js").write_text(_HARNESS)
    (folder / "cases.json").write_text(json.dumps(cases))
    assert _NODE is not None
    done = subprocess.run(
        [_NODE, str(folder / "harness.js"), str(_CONSOLE), str(folder / "cases.json")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return {name: dict(rows) for name, rows in json.loads(done.stdout).items()}


def test_the_novelty_verdict_shows_what_it_rests_on(rendered: dict[str, dict[str, str]]) -> None:
    row = rendered["full"]["· novelty"]
    assert "NOVEL" in row
    assert "p=" in row and "vs level 0.05" in row
    assert "ranked among" in row and "reference distances" in row
    assert "cut" not in row  # the verdict is a rank test now; there is no score cut to show


def test_the_causal_test_says_how_much_it_tested(rendered: dict[str, dict[str, str]]) -> None:
    assert "X0 → X1" in rendered["full"]["· causal factors"]
    assert "partial R²" in rendered["full"]["· causal factors"]
    row = rendered["full"]["· causal test"]
    assert row.startswith("12 pairs tested, corrected for lags and pairs")
    assert row.endswith("1 edge(s)")


def test_a_window_that_could_not_be_read_says_so(rendered: dict[str, dict[str, str]]) -> None:
    row = rendered["unstable"]["· causal test"]
    assert row.startswith("not assessed:")
    assert "· causal factors" not in rendered["unstable"]  # and no factor is shown


def test_the_hypothesis_shows_its_weakest_premise_and_when_it_falls(
    rendered: dict[str, dict[str, str]],
) -> None:
    assert "X0 leads X1 at lag 1" in rendered["full"]["· hypothesis"]
    assert "conf" not in rendered["full"]["· hypothesis"]  # none was measured, none is shown
    challenge = rendered["full"]["· challenge it"]
    assert challenge.startswith("weakest premise:")
    assert "margin x" in challenge
    assert "withdrawn if observation_novel: statistic > 0.05" in challenge


def test_the_replay_path_is_on_the_panel(rendered: dict[str, dict[str, str]]) -> None:
    row = rendered["full"]["· replay"]
    for kind in ("novelty", "causal", "hypothesis"):
        assert f"{kind} (from this record)" in row
    assert "POST /api/v1/detect/flagship/replay" in row
    assert "mercury-agent replay" in row


def test_an_artifact_that_cannot_be_replayed_says_why(rendered: dict[str, dict[str, str]]) -> None:
    assert "novelty (cannot)" in rendered["withheld"]["· replay"]
    assert "not disclosed" in rendered["withheld"]["· novelty not replayable"]
    assert "· novelty not replayable" not in rendered["full"]


def test_a_replay_renders_its_verdict_and_each_artifact(
    rendered: dict[str, dict[str, str]],
) -> None:
    rows = rendered["replayed"]
    assert rows["replay"] == "HOLDS · 3 artifact(s)"
    for kind in ("novelty", "causal", "hypothesis"):
        assert rows[f"· {kind}"].startswith("HOLDS · ")
