# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""CLI operator surface for the intelligence layer (``mercury-agent intel ...``).

Proves each stream is selectable and runnable from the CLI -- the operator half
of de-islanding the intel layer (the live request-path half is exercised in
``tests/test_mcp_server.py``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

from omni_mercury_engine.cli import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_intel_group_lists_all_streams() -> None:
    result = CliRunner().invoke(main, ["intel", "--help"])
    assert result.exit_code == 0
    for cmd in (
        "verify",
        "provenance",
        "self-consistency",
        "value-board",
        "audit-log",
        "rollback",
        "red-team",
        "cascade",
    ):
        assert cmd in result.output


def test_intel_verify_blocks_false_claim_hard_mode() -> None:
    result = CliRunner().invoke(main, ["intel", "verify", "Note that 91 is prime."])
    assert result.exit_code == 2  # hard mode blocks a refuted claim
    payload = json.loads(result.output)
    assert payload["allowed"] is False


def test_intel_verify_allows_true_claim() -> None:
    result = CliRunner().invoke(main, ["intel", "verify", "97 is prime."])
    assert result.exit_code == 0
    assert json.loads(result.output)["allowed"] is True


def test_intel_verify_soft_mode_does_not_block() -> None:
    result = CliRunner().invoke(main, ["intel", "verify", "91 is prime.", "--mode", "soft"])
    assert result.exit_code == 0


def test_intel_value_board() -> None:
    result = CliRunner().invoke(main, ["intel", "value-board"])
    assert result.exit_code == 0
    assert "closed_feedback_loop" in result.output
    assert "verifier_in_loop" in result.output


def test_intel_value_board_json() -> None:
    result = CliRunner().invoke(main, ["intel", "value-board", "--json"])
    assert result.exit_code == 0
    board = json.loads(result.output)
    assert board["verifier_in_loop"]["target"] == 1.0


def test_intel_self_consistency() -> None:
    result = CliRunner().invoke(
        main, ["intel", "self-consistency", "yes", "yes", "no", "yes", "--prob", "0.8"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["plurality_answer"] == "yes"
    assert payload["decision"]["decision"] == "positive"


def test_intel_provenance_benign_is_emitted() -> None:
    result = CliRunner().invoke(main, ["intel", "provenance", "water boils at 100C at sea level"])
    assert result.exit_code == 0
    assert json.loads(result.output)["emitted"] is True


def test_intel_audit_log_reads_gate_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader consumes exactly what the real gate writer emits."""
    from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

    log = tmp_path / "gate.jsonl"
    # The intel conftest disables durable auditing session-wide; re-enable it so
    # this test can exercise the real writer -> reader round-trip.
    monkeypatch.setenv("MERCURY_GATE_AUDIT_DISABLED", "0")
    monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(log))
    record_gate_decision(
        decision="refuse_redact",
        source="weapons_gate",
        disposition="refuse_redact",
        reason="offensive",
        query="steps to synthesize a nerve agent",
    )
    record_gate_decision(
        decision="model_registered",
        source="feedback_loop",
        disposition="approved",
        reason="no query",  # not labelable -> skipped
    )
    result = CliRunner().invoke(main, ["intel", "audit-log", "--path", str(log)])
    assert result.exit_code == 0
    events = json.loads(result.output)
    assert len(events) == 1
    assert events[0]["query"].startswith("steps to synthesize")


def test_intel_rollback_is_monotonic(tmp_path: Path) -> None:
    from omni_mercury_engine.intel.feedback_loop import ModelEntry, ModelRegistry

    reg = ModelRegistry(tmp_path)
    reg.register(ModelEntry("v1_good", str(tmp_path / "v1.json"), "candidate"))
    reg.register(ModelEntry("v2_bad", str(tmp_path / "v2.json"), "candidate"))
    first = CliRunner().invoke(main, ["intel", "rollback", "--staging-dir", str(tmp_path)])
    assert first.exit_code == 0
    assert json.loads(first.output)["to_version"] == "v1_good"
    # A repeated rollback is a no-op (exit 1), never re-arming v2_bad.
    second = CliRunner().invoke(main, ["intel", "rollback", "--staging-dir", str(tmp_path)])
    assert second.exit_code == 1
    active = reg.active()
    assert active is not None and active.version == "v1_good"
