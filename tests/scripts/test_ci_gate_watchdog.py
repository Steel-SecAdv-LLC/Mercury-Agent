# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the CI gate watchdog analyzer.

The analyzer is the pure decision core: given a workflow's recent run history
it decides whether the gate is failing to complete cleanly. All cases are
deterministic (``now`` is injected).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOD = REPO_ROOT / "scripts" / "ci_gate_watchdog.py"
_spec = importlib.util.spec_from_file_location("ci_gate_watchdog", _MOD)
assert _spec is not None and _spec.loader is not None
wd = importlib.util.module_from_spec(_spec)
# Register before exec: the module uses ``from __future__ import annotations``
# with ``@dataclass``, and dataclass field resolution looks the module up in
# ``sys.modules`` by name (``cls.__module__``). Without this the exec raises
# AttributeError on the first @dataclass.
sys.modules[_spec.name] = wd
_spec.loader.exec_module(wd)

NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)


def _run(
    conclusion: str | None,
    status: str = "completed",
    minutes_ago: int = 5,
    head_branch: str | None = None,
) -> dict[str, Any]:
    started = NOW.timestamp() - minutes_ago * 60
    run: dict[str, Any] = {
        "status": status,
        "conclusion": conclusion,
        "createdAt": datetime.fromtimestamp(started, tz=UTC).isoformat().replace("+00:00", "Z"),
        "event": "pull_request",
    }
    if head_branch is not None:
        run["headBranch"] = head_branch
    return run


class TestAnalyze:
    def test_all_success_no_alert(self) -> None:
        runs = [_run("success") for _ in range(10)]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.alert is False
        assert v.unhealthy_count == 0
        assert v.body == ""

    def test_failures_are_not_watchdog_concerns(self) -> None:
        # A red gate is the gate WORKING — never a watchdog alert.
        runs = [_run("failure") for _ in range(10)]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.alert is False
        assert v.unhealthy_count == 0

    def test_repeated_cancellation_alerts(self) -> None:
        runs = [_run("cancelled") for _ in range(5)] + [_run("success") for _ in range(3)]
        v = wd.analyze("Sigma-Immutable Mutation Gate", runs, now=NOW, threshold=3)
        assert v.alert is True
        assert v.unhealthy_count == 5
        assert "did not reach a clean verdict" in v.reasons[0]
        assert "Sigma-Immutable Mutation Gate" in v.title
        assert "cancel-in-progress" in v.body

    def test_below_threshold_no_alert(self) -> None:
        runs = [_run("cancelled"), _run("timed_out")] + [_run("success") for _ in range(8)]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.alert is False
        assert v.unhealthy_count == 2

    def test_timed_out_and_startup_failure_count(self) -> None:
        runs = [_run("timed_out"), _run("startup_failure"), _run("timed_out")]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.alert is True
        assert v.unhealthy_count == 3

    def test_stuck_in_progress_counts_as_unhealthy(self) -> None:
        # Three runs still in_progress, started 200 min ago (> 180 stuck floor).
        runs = [_run(None, status="in_progress", minutes_ago=200) for _ in range(3)]
        v = wd.analyze("W", runs, now=NOW, threshold=3, stuck_minutes=180)
        assert v.alert is True
        assert v.unhealthy_count == 3
        assert "still running past" in v.reasons[0]

    def test_recent_in_progress_not_stuck(self) -> None:
        runs = [_run(None, status="in_progress", minutes_ago=10) for _ in range(5)]
        v = wd.analyze("W", runs, now=NOW, threshold=3, stuck_minutes=180)
        assert v.alert is False
        assert v.unhealthy_count == 0

    def test_window_bounds_considered_runs(self) -> None:
        # 10 cancelled but window=5 -> only 5 considered (still >= threshold).
        runs = [_run("cancelled") for _ in range(10)]
        v = wd.analyze("W", runs, now=NOW, threshold=3, window=5)
        assert v.considered == 5
        assert v.unhealthy_count == 5

    def test_empty_history_no_alert(self) -> None:
        v = wd.analyze("W", [], now=NOW, threshold=3)
        assert v.alert is False
        assert v.considered == 0

    def test_superseded_cancellations_are_excused(self) -> None:
        # Every cancellation is on a branch that ALSO reached a clean verdict
        # in the window — the normal busy-PR cancel-in-progress shape. No alarm.
        runs = [
            _run("cancelled", head_branch="pr-1"),
            _run("success", head_branch="pr-1"),
            _run("cancelled", head_branch="pr-2"),
            _run("failure", head_branch="pr-2"),
            _run("cancelled", head_branch="pr-3"),
            _run("success", head_branch="pr-3"),
        ]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.alert is False
        assert v.unhealthy_count == 0

    def test_cancellation_without_later_verdict_still_alerts(self) -> None:
        # A branch cancelled repeatedly that NEVER reached a verdict is the
        # real #348 pathology — still an alarm even with headBranch present.
        runs = [_run("cancelled", head_branch="stuck-pr") for _ in range(4)]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.alert is True
        assert v.unhealthy_count == 4

    def test_missing_head_branch_keeps_strict_behavior(self) -> None:
        # Older feeds without headBranch fall back to counting every cancel.
        runs = [_run("cancelled") for _ in range(4)]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.alert is True
        assert v.unhealthy_count == 4

    def test_mixed_superseded_and_real_only_counts_real(self) -> None:
        # pr-1 cancels are superseded (benign); pr-2 never gets a verdict.
        runs = [
            _run("cancelled", head_branch="pr-1"),
            _run("success", head_branch="pr-1"),
            _run("cancelled", head_branch="pr-2"),
            _run("cancelled", head_branch="pr-2"),
            _run("cancelled", head_branch="pr-2"),
        ]
        v = wd.analyze("W", runs, now=NOW, threshold=3)
        assert v.unhealthy_count == 3
        assert v.alert is True
        assert "superseded" in v.reasons[0]


class TestMainCli:
    def test_main_reads_file_and_emits_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        runs = [_run("cancelled") for _ in range(4)]
        p = tmp_path / "runs.json"
        p.write_text(json.dumps(runs), encoding="utf-8")
        rc = wd.main(["--workflow", "W", "--runs", str(p), "--now", NOW.isoformat()])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["alert"] is True
        assert out["unhealthy_count"] == 4

    def test_main_bad_json_is_soft(self, tmp_path: Path) -> None:
        p = tmp_path / "runs.json"
        p.write_text("not json", encoding="utf-8")
        rc = wd.main(["--workflow", "W", "--runs", str(p)])
        assert rc == 0  # analyzer never hard-fails the watchdog workflow
