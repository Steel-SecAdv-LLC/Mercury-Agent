# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The end-to-end example delivers the three answers on one real decision.

``examples/auditable_decision.py`` is the walk the README promises: one
committed window through the shipped flagship model in the served posture,
the three answers (why this row, how sure, the record) read out of the
decision record, and a challenge the reader can re-run.  These tests run it
for real and pin:

* the committed window is exactly what the script's recipe builds from the
  shipped checkpoint's training data (so the example's claims about its own
  input stay true);
* the three answers come back with the planted structure found — the four
  planted anomaly rows flagged, the planted X0-leads-X1 driver recovered,
  a singleton conformal certificate, a provenance-complete record;
* the challenge holds on the honest record and FAILS on the tampered copy,
  and the script's exit status says so.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="the flagship engine needs [ml]")

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "auditable_decision.py"


@pytest.fixture(scope="module")
def example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auditable_decision", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def outcome(example: ModuleType) -> dict[str, Any]:
    """One real run: the shipped checkpoint, the committed window."""
    window = np.loadtxt(example.WINDOW, delimiter=",")
    result = example.decide(window)
    return {
        "window": window,
        "result": result,
        "answers": example.three_answers(result),
        "challenge": example.challenge(result, window),
    }


def test_the_committed_window_is_the_recipe(example: ModuleType) -> None:
    """The CSV in the repository is byte-for-byte what build_window() makes."""
    committed = np.loadtxt(example.WINDOW, delimiter=",")
    assert committed.shape == (80, 16)
    np.testing.assert_array_equal(committed, example.build_window())


class TestTheThreeAnswers:
    def test_why_this_row_names_the_planted_structure(self, outcome: dict[str, Any]) -> None:
        why = outcome["answers"]["why_this_row"]
        assert why["rows_flagged"] == [40, 41, 42, 43], why
        assert why["top_row"] in range(40, 44)
        assert len(why["detector_importance"]) == 5
        assert any(factor.startswith("X0 → X1 (lag=1") for factor in why["causal_factors"])
        assert "X0 leads X1 at lag 1" in (why["hypothesis"] or "")
        assert why["withdrawn_if"]

    def test_how_sure_is_a_certificate_not_a_score(self, outcome: dict[str, Any]) -> None:
        sure = outcome["answers"]["how_sure"]
        assert sure["anomaly_prob"] >= sure["threshold"]
        assert sure["conformal_set"] == [1]
        assert sure["coverage_target"] == 0.9
        assert sure["n_calibration"] >= 100
        assert sure["calibrated"] is True
        # One regime of the four the model was calibrated on: drift is seen,
        # and the decision honestly defers instead of claiming the coverage.
        assert sure["drift"] is not None
        assert (sure["state"], sure["abstained"]) == ("unavailable", True)

    def test_the_record_is_provenance_complete(self, outcome: dict[str, Any]) -> None:
        record = outcome["answers"]["the_record"]
        assert record["disposition"] == "defer"
        assert record["requires_human"] is True
        assert record["checkpoint"] == "default_fusion.pt"
        assert len(record["checkpoint_sha256"]) == 64
        assert record["ethical_gate_passed"] is True
        assert record["artifacts"] == ["causal", "hypothesis"]
        assert record["reasons"]


class TestTheChallenge:
    def test_the_decision_holds_and_a_tampered_copy_fails(self, outcome: dict[str, Any]) -> None:
        challenge = outcome["challenge"]
        assert challenge["holds"] is True
        assert [report["status"] for report in challenge["replay"]] == ["HOLDS", "HOLDS"]
        assert challenge["tampered_fails"] is True
        statuses = {report["kind"]: report["status"] for report in challenge["tampered_replay"]}
        assert statuses == {"causal": "FAILS", "hypothesis": "FAILS"}

    def test_the_exit_status_reports_the_outcome(
        self,
        example: ModuleType,
        outcome: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """main() is the operator surface: 0 exactly when the contract held."""
        monkeypatch.setattr(example, "decide", lambda window: outcome["result"])
        assert example.main(["--out", str(tmp_path)]) == 0
        printed = capsys.readouterr().out
        for heading in ("1. WHY THIS ROW", "2. HOW SURE", "3. THE RECORD", "CHALLENGE"):
            assert heading in printed
        assert "mercury-agent replay" in printed
        assert (tmp_path / "decision.json").is_file()
        assert (tmp_path / "window.csv").is_file()

        # A run whose challenge does not hold exits 1.
        broken = {
            "replay": [],
            "holds": False,
            "tampered_replay": [],
            "tampered_fails": True,
        }
        monkeypatch.setattr(example, "challenge", lambda result, window: broken)
        assert example.main(["--json"]) == 1
