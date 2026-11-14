# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The benchmark PR's checks are dispatched against the commit that stays its head.

``benchmark.yml`` keeps one rolling pull request (``ci/benchmark-results``)
and moves its branch twice per run: once with the mercury results and the
README block, once more with the seven-axis results. Commits made with the
default ``GITHUB_TOKEN`` start no ``push``/``pull_request`` workflows, so the
job dispatches ``ci.yml`` against the branch by hand.

That dispatch sat between the two branch updates. CI ran on the first commit;
the second update then replaced the head with a commit that had no check runs,
and branch protection, which evaluates the head, kept the pull request blocked
on pending checks through every refresh. These tests pin the dispatch after
the last step that moves the branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "benchmark.yml"


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    job = next(j for j in workflow["jobs"].values() if j.get("name") == "Run Mercury Benchmarks")
    steps: list[dict[str, Any]] = job["steps"]
    return steps


def _index(steps: list[dict[str, Any]], predicate: str) -> list[int]:
    return [i for i, step in enumerate(steps) if predicate in str(step.get("run") or "")]


def test_the_dispatch_follows_every_step_that_moves_the_branch() -> None:
    steps = _steps()
    persists = _index(steps, "scripts/persist_benchmark_to_pr.py")
    dispatches = _index(steps, "gh workflow run ci.yml --ref ci/benchmark-results")
    assert len(persists) == 2, "expected the mercury persist and the seven-axis persist"
    assert len(dispatches) == 1
    assert dispatches[0] > max(persists)


def test_nothing_moves_the_branch_after_the_dispatch() -> None:
    steps = _steps()
    (dispatch,) = _index(steps, "gh workflow run ci.yml --ref ci/benchmark-results")
    for step in steps[dispatch + 1 :]:
        run = str(step.get("run") or "")
        assert "persist_benchmark_to_pr" not in run and "git push" not in run, step.get("name")


def test_the_dispatch_runs_when_either_persist_step_moved_the_branch() -> None:
    steps = _steps()
    ids = {step["id"] for step in steps if "persist_benchmark_to_pr" in str(step.get("run") or "")}
    assert ids == {"persist_mercury", "persist_seven_axis"}
    (dispatch,) = _index(steps, "gh workflow run ci.yml --ref ci/benchmark-results")
    condition = " ".join(str(steps[dispatch]["if"]).split())
    # always(): a failed seven-axis runner must not cancel the dispatch for the
    # mercury commit, which is then the head.
    assert condition.startswith("always() && (")
    assert "steps.persist_mercury.outcome == 'success'" in condition
    assert "steps.persist_seven_axis.outcome == 'success'" in condition
    assert "github.ref == 'refs/heads/main'" in condition


def test_the_workflow_may_dispatch_and_ci_accepts_it() -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["permissions"]["actions"] == "write"
    ci = yaml.safe_load((_WORKFLOW.parent / "ci.yml").read_text(encoding="utf-8"))
    triggers = ci.get("on", ci.get(True))
    assert "workflow_dispatch" in triggers
