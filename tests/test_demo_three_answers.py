# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke check: the demonstration still gives the three answers on one real decision.

``mercury-agent demo`` is the first thing the README shows a newcomer.  It
scores one committed window with the bundled flagship model and states, in
plain language beside the complete decision record: why this row, how sure,
and the record that can be reconstructed and contested; then it challenges
the decision and ends with the command that re-runs the challenge.

These tests run it for real (no stand-in engine) and fail if any of the three
answers disappears, loses its substance, or stops being read from the record:

* each heading is there with the facts that make it an answer (the planted
  driver recovered, the conformal set and what calibrated it, the model
  digest and the ethical gate's outcome);
* the complete decision record is printed and is exactly the record kept;
* the challenge holds, a tampered copy fails, and the exit status says so;
* the last line is a command, and running it re-runs the challenge;
* the committed window is byte-for-byte what its recipe builds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="the flagship engine needs [ml]")

from click.testing import CliRunner

from omni_mercury_engine import demo
from omni_mercury_engine.cli import main as cli

if TYPE_CHECKING:
    from types import ModuleType

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "auditable_decision.py"
_RECORD_HEADER = "THE COMPLETE DECISION RECORD (every answer above was read from this)"
_CHALLENGE_HEADER = "CHALLENGE THIS DECISION (its recorded findings, replayed against the data)"


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One real run of the command: the bundled model, the committed window."""
    kept = tmp_path_factory.mktemp("mercury-demo")
    outcome = CliRunner().invoke(cli, ["demo", "--out", str(kept)])
    return {"kept": kept, "exit_code": outcome.exit_code, "text": outcome.output}


def _section(text: str, heading: str, following: str) -> str:
    start = text.index(heading)
    return text[start : text.index(following, start)]


class TestTheThreeAnswersAreThere:
    def test_it_succeeds_in_well_under_ten_minutes(self, ran: dict[str, Any]) -> None:
        assert ran["exit_code"] == 0, ran["text"][-2000:]

    def test_every_heading_is_present_in_order(self, ran: dict[str, Any]) -> None:
        positions = [ran["text"].index(heading) for heading in demo.HEADINGS]
        assert positions == sorted(positions)
        assert demo.HEADINGS == ("1. WHY THIS ROW", "2. HOW SURE", "3. THE RECORD")

    def test_why_this_row_recovers_what_was_planted(self, ran: dict[str, Any]) -> None:
        why = " ".join(_section(ran["text"], demo.HEADINGS[0], demo.HEADINGS[1]).split())
        assert f"flagged: {demo.PLANTED['anomaly_rows']}" in why
        assert "weighed these feature groups most" in why
        assert "X0 → X1 (lag=1" in why  # the planted driver, recovered
        assert "Within this window X0 leads X1 at lag 1" in why
        assert "weakest premise" in why and "withdrawn if" in why
        assert "decision.signals.cognitive.hypothesis.falsifier.withdrawn_if" in why

    def test_how_sure_is_a_certificate_with_its_coverage(self, ran: dict[str, Any]) -> None:
        sure = " ".join(_section(ran["text"], demo.HEADINGS[1], demo.HEADINGS[2]).split())
        assert "against a threshold of 0.500" in sure
        assert "conformal prediction set is {1}" in sure
        assert "90% coverage" in sure and "held-out points" in sure
        # One regime of four: the coverage is not claimed where the calibration
        # does not describe the data, and the decision says so by abstaining.
        assert "drift: " in sure and "coverage is not claimed for this window" in sure
        assert "Mercury therefore abstains" in sure
        assert "decision.signals.model_provenance.conformal" in sure

    def test_the_record_names_model_gate_and_what_replays(self, ran: dict[str, Any]) -> None:
        record = " ".join(_section(ran["text"], demo.HEADINGS[2], _RECORD_HEADER).split())
        assert "The decision is 'defer'" in record and "needs a human" in record
        assert "default_fusion.pt, sha256 " in record
        assert "ethical gate's outcome (passed: score " in record
        assert "replay from the record alone (causal, hypothesis)" in record
        assert "decision.signals.model_provenance.checkpoint_sha256" in record


class TestTheRecordAndTheChallenge:
    def test_the_complete_record_is_printed_and_is_the_record_kept(
        self, ran: dict[str, Any]
    ) -> None:
        printed = _section(ran["text"], _RECORD_HEADER, _CHALLENGE_HEADER)
        record = json.loads(printed[len(_RECORD_HEADER) :])
        kept = json.loads((ran["kept"] / "decision.json").read_text(encoding="utf-8"))
        assert record == kept["decision"]
        assert {"state", "disposition", "signals", "response", "reasons", "caveats"} <= set(record)

    def test_the_findings_hold_and_a_tampered_copy_fails(self, ran: dict[str, Any]) -> None:
        tail = ran["text"][ran["text"].index(_CHALLENGE_HEADER) :]
        assert "-> the decision's findings HOLD" in tail
        assert "-> the tampered copy FAILS" in tail
        assert tail.count("HOLDS") == 2 and tail.count("FAILS   ") == 2

    def test_it_ends_with_a_command_that_reruns_the_challenge(self, ran: dict[str, Any]) -> None:
        last = ran["text"].rstrip("\n").splitlines()[-1].strip()
        assert last == demo.replay_command(ran["kept"])
        words = last.split()
        assert words[:2] == ["mercury-agent", "replay"]
        replayed = CliRunner().invoke(cli, words[1:])
        assert replayed.exit_code == 0, replayed.output
        assert replayed.output.count("HOLDS") == 2

    def test_the_json_form_carries_the_same_parts(
        self, ran: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        kept = json.loads((ran["kept"] / "decision.json").read_text(encoding="utf-8"))
        monkeypatch.setattr(demo, "decide", lambda window: kept)  # the run above, not a second one
        status, text = demo.run(tmp_path, as_json=True)
        payload = json.loads(text)
        assert status == 0
        assert set(payload) == {"answers", "decision", "challenge", "replay_command"}
        assert set(payload["answers"]) == {"why_this_row", "how_sure", "the_record"}
        for answer in payload["answers"].values():
            assert answer["plain"] and answer["read_from"]


class TestItFailsWhenAnAnswerGoes:
    """The exit status and the wording follow the record; neither is scripted."""

    def test_a_run_whose_challenge_does_not_hold_exits_one(
        self, ran: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        kept = json.loads((ran["kept"] / "decision.json").read_text(encoding="utf-8"))
        monkeypatch.setattr(demo, "decide", lambda window: kept)
        broken = {"replay": [], "holds": False, "tampered_replay": [], "tampered_fails": True}
        monkeypatch.setattr(demo, "challenge", lambda result, window: broken)
        status, text = demo.run(tmp_path)
        assert status == 1
        assert "DO NOT HOLD" in text

    def test_a_record_without_findings_says_so_instead_of_inventing_them(
        self, ran: dict[str, Any]
    ) -> None:
        kept = json.loads((ran["kept"] / "decision.json").read_text(encoding="utf-8"))
        kept["decision"]["signals"]["cognitive"] = None
        kept["decision"]["signals"]["drift_detected"] = False
        answers = demo.three_answers(kept)
        assert "No lead-lag relation between columns survived correction" in (
            answers["why_this_row"]["plain"]
        )
        assert "No hypothesis was derived" in answers["why_this_row"]["plain"]
        assert "No drift against the model's reference was detected" in answers["how_sure"]["plain"]
        assert "0 finding(s) that replay" in answers["the_record"]["plain"]


@pytest.fixture(scope="module")
def example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auditable_decision", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_window_is_the_recipe(example: ModuleType) -> None:
    """The CSV shipped in the package is byte-for-byte what its recipe builds."""
    committed = demo.load_window()
    assert committed.shape == (demo.ROWS, demo.COLUMNS) == (80, 16)
    np.testing.assert_array_equal(committed, example.build_window())
