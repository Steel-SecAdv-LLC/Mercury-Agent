# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pillar mutation gate refuses to score kills on a red baseline.

``scripts/mutate_pillar_controls.py`` infers "the mutation was caught" from
pytest exiting 1. That inference is only valid if the selection passed *before*
the mutation was applied: on an already-red selection every mutation exits 1,
every one is scored ``KILLED``, and the gate prints "Every disabled control was
caught. No vacuous pillar coverage found." while having observed nothing.

The gate had no such check. CI runs the pillar suite in the step immediately
before it, so step ordering protected it there, but that guarantee is external
and implicit: it does not hold for the standalone invocation the module's
``Usage`` block documents, it does not cover an individual ``selection`` that
fails or collects nothing on its own, and it vanishes silently if the step order
changes or the step gains ``continue-on-error``.

These tests pin the precondition as a local property of the script: a red
baseline makes it refuse, and the refusal happens **before any mutation runs**
-- a gate that mutated first and then complained would still have paid for the
mutation runs and could still have printed a kill count.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mutate_pillar_controls.py"


@pytest.fixture(scope="module")
def gate() -> Any:
    """Import the gate script as a module without executing its ``__main__``."""
    spec = importlib.util.spec_from_file_location("mercury_mutate_pillar_gate", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_declares_a_baseline_runner(gate: Any) -> None:
    """The precondition exists as callable code, not only as prose."""
    assert callable(gate.run_baseline)
    assert callable(gate.baseline_selections)


def test_baseline_selections_are_the_distinct_mutation_selections(gate: Any) -> None:
    """Each selection is checked once, not once per mutation that uses it."""
    selections = gate.baseline_selections()
    assert set(selections) == {m.selection for m in gate.MUTATIONS}
    assert len(selections) == len(set(selections)), "a selection would be run twice"
    # Cheap by construction: far fewer baseline runs than mutations.
    assert len(selections) < len(gate.MUTATIONS)


def test_a_red_baseline_refuses_before_any_mutation_runs(
    gate: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The decisive property: red baseline -> exit 1 and zero mutations run.

    Without this, a red selection scored 7/7 KILLED and the gate reported
    "No vacuous pillar coverage found."
    """
    mutations_run: list[str] = []

    def _never(mutation: Any) -> str:
        mutations_run.append(mutation.name)
        return "KILLED"

    monkeypatch.setattr(gate, "run_baseline", lambda _sel: (1, "1 failed, 2 passed"))
    monkeypatch.setattr(gate, "run_mutation", _never)
    monkeypatch.setattr(sys, "argv", ["mutate_pillar_controls.py"])

    exit_code = gate.main()
    out = capsys.readouterr().out

    assert exit_code == 1, "a red baseline must fail the gate"
    assert mutations_run == [], (
        f"mutations ran on a red baseline: {mutations_run} -- the refusal must "
        "come first, or the kill count is computed from meaningless runs"
    )
    assert "BASELINE NOT GREEN" in out
    assert (
        "No vacuous pillar coverage found" not in out
    ), "the gate claimed clean coverage on a red baseline"


def test_a_green_baseline_lets_the_mutations_run(
    gate: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The precondition must not block the normal path."""
    mutations_run: list[str] = []

    def _killed(mutation: Any) -> str:
        mutations_run.append(mutation.name)
        return "KILLED"

    monkeypatch.setattr(gate, "run_baseline", lambda _sel: (0, "42 passed"))
    monkeypatch.setattr(gate, "run_mutation", _killed)
    monkeypatch.setattr(sys, "argv", ["mutate_pillar_controls.py"])

    exit_code = gate.main()
    out = capsys.readouterr().out

    assert exit_code == 0
    assert len(mutations_run) == len(gate.MUTATIONS)
    assert "BASELINE NOT GREEN" not in out
    assert "Every disabled control was caught" in out


def test_a_survivor_still_fails_even_on_a_green_baseline(
    gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original contract is intact: a vacuous test is still a finding."""
    monkeypatch.setattr(gate, "run_baseline", lambda _sel: (0, "42 passed"))
    monkeypatch.setattr(gate, "run_mutation", lambda _m: "SURVIVED")
    monkeypatch.setattr(sys, "argv", ["mutate_pillar_controls.py"])
    assert gate.main() == 1


def test_an_errored_mutation_is_not_scored_as_a_kill(
    gate: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A collection/usage error never counts as observation."""
    monkeypatch.setattr(gate, "run_baseline", lambda _sel: (0, "42 passed"))
    monkeypatch.setattr(gate, "run_mutation", lambda _m: "ERROR")
    monkeypatch.setattr(sys, "argv", ["mutate_pillar_controls.py"])
    exit_code = gate.main()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "UNPROVEN" in out
    assert "Every disabled control was caught" not in out
