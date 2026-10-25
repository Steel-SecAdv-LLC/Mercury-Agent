# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CLI groups and MCP tools reach the same gated service as the API routes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from click.testing import CliRunner

pytest.importorskip("torch")  # the decision gate needs the ML stack

from omni_mercury_engine.cli import main
from omni_mercury_engine.mcp_server import MercuryMCPServer

if TYPE_CHECKING:
    from pathlib import Path

POLICY = "MERCURY_HUMAN_SUBJECT_CAPABILITIES"
STATEMENTS = {
    "statements": [
        {
            "statement_id": "s1",
            "speaker": "w1",
            "made_at": 100,
            "claims": [{"subject": "w1", "predicate": "location", "value": "depot"}],
        }
    ]
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate.jsonl"))
    monkeypatch.setenv("MERCURY_ADJUDICATION_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv(POLICY, raising=False)
    monkeypatch.delenv("MERCURY_ADJUDICATION_SIGNING_KEY", raising=False)
    monkeypatch.delenv("MERCURY_ADJUDICATION_PUBLIC_KEY", raising=False)


def _write(tmp_path: Path, name: str, payload: Any) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _vitals(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    fs = 20.0
    t = np.arange(0.0, 60.0, 1.0 / fs)
    chest = 0.004 * np.sin(2 * np.pi * 0.25 * t) + 0.0003 * np.sin(2 * np.pi * 1.2 * t)
    np.savez(tmp_path / "radar.npz", radar_displacement_m=chest)
    payload: dict[str, Any] = {
        "purpose": "Sleep-lab respiration study; volunteers signed informed consent forms",
        "lawful_basis": "consent",
        "fs": fs,
        "recorded_at": 5.0,
        "subject_pseudonym": "subj-02",
        "consent": {
            "subject_pseudonym": "subj-02",
            "obtained_at": 0.0,
            "scope": ["contactless_vital_signs"],
            "obtained_by": "nurse-3",
        },
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def test_cli_verify_statements_from_file_and_stdin(tmp_path: Path) -> None:
    runner = CliRunner()
    path = _write(tmp_path, "s.json", STATEMENTS)
    result = runner.invoke(main, ["assess", "verify-statements", path])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["gate"]["surface"] == "cli.assessment.verify_statements"
    piped = runner.invoke(main, ["assess", "verify-statements", "-"], input=json.dumps(STATEMENTS))
    assert piped.exit_code == 0


def test_cli_malformed_input_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    bad = _write(tmp_path, "bad.json", {"statement": []})  # misspelt key
    assert runner.invoke(main, ["assess", "verify-statements", bad]).exit_code == 2
    (tmp_path / "broken.json").write_text("{not json")
    assert runner.invoke(main, ["assess", "screen", str(tmp_path / "broken.json")]).exit_code == 2


def test_cli_vital_signs_refused_until_enabled_then_reads_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    request = _write(tmp_path, "v.json", _vitals(tmp_path))
    arrays = str(tmp_path / "radar.npz")
    refused = runner.invoke(main, ["assess", "vital-signs", request, "--arrays", arrays])
    assert refused.exit_code == 3
    assert json.loads(refused.stderr)["check"] == "disabled"
    monkeypatch.setenv(POLICY, "consent")
    result = runner.invoke(main, ["assess", "vital-signs", request, "--arrays", arrays])
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["result"]["respiration"]["rate_bpm"] == pytest.approx(15.0, abs=0.5)


def test_cli_covert_purpose_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POLICY, "consent")
    payload = _vitals(tmp_path, purpose="covertly track my flatmate's breathing at night")
    request = _write(tmp_path, "v.json", payload)
    result = CliRunner().invoke(
        main, ["assess", "vital-signs", request, "--arrays", str(tmp_path / "radar.npz")]
    )
    assert result.exit_code == 3
    assert json.loads(result.stderr)["check"] == "covert_sensing"


def test_cli_adjudication_round_trip(tmp_path: Path) -> None:
    runner = CliRunner()
    case = {
        "case_id": "case-cli-1",
        "intel_reports": {"open_source": {"confidence": 0.7, "threat_score": 0.5}},
        "report_source_ids": ["Reuters"],
        "latest_report_at": "2026-01-01T00:00:00+00:00",
    }
    outcome = {
        "case_id": "case-cli-1",
        "observed_at": "2026-01-02T00:00:00+00:00",
        "description": "Crossing closed for two days; no casualties reported.",
        "outcome_source_ids": ["UN OCHA"],
    }
    rating = {"case_id": "case-cli-1", "level": 2, "rationale": "localised disruption"}
    for args in (
        ["adjudication", "add-case", _write(tmp_path, "c.json", case)],
        ["adjudication", "add-outcome", _write(tmp_path, "o.json", outcome)],
        ["adjudication", "rate", _write(tmp_path, "r.json", rating), "--rater", "alice"],
    ):
        result = runner.invoke(main, args)
        assert result.exit_code == 0, (args, result.output)
    view = runner.invoke(main, ["adjudication", "rater-view", "case-cli-1", "--rater", "bob"])
    assert json.loads(view.stdout)["other_ratings"] is None
    verify = runner.invoke(main, ["adjudication", "verify"])
    assert verify.exit_code == 0 and json.loads(verify.stdout)["valid"] is True
    status = json.loads(runner.invoke(main, ["adjudication", "status"]).stdout)
    assert status["pending"]["awaiting_ratings"] == ["case-cli-1"]
    # Too few cases for a merit export: refused, exit 1, nothing written.
    out = tmp_path / "heldout.json"
    export = runner.invoke(
        main, ["adjudication", "export", "--split", "heldout", "--out", str(out)]
    )
    assert export.exit_code == 1 and not out.exists()


def test_cli_keygen_signs_ledger_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        MLDSAProvider()
    except (ImportError, RuntimeError):  # pragma: no cover - AMA is required in CI
        pytest.skip("ML-DSA-65 needs the AMA native backend")
    runner = CliRunner()
    key_file = tmp_path / "adjudication.key"
    created = runner.invoke(main, ["adjudication", "keygen", "--out", str(key_file)])
    assert created.exit_code == 0, created.output
    public_hex = json.loads(created.stdout)["public_key_hex"]
    assert runner.invoke(main, ["adjudication", "keygen", "--out", str(key_file)]).exit_code == 1
    monkeypatch.setenv("MERCURY_ADJUDICATION_SIGNING_KEY", str(key_file))
    monkeypatch.setenv("MERCURY_ADJUDICATION_PUBLIC_KEY", public_hex)
    case = {
        "case_id": "case-signed",
        "intel_reports": {"open_source": {"threat_score": 0.5}},
        "report_source_ids": ["Reuters"],
        "latest_report_at": "2026-01-01T00:00:00+00:00",
    }
    added = runner.invoke(main, ["adjudication", "add-case", _write(tmp_path, "c.json", case)])
    assert added.exit_code == 0, added.output
    assert json.loads(added.stdout)["signed"] is True
    assert json.loads(runner.invoke(main, ["adjudication", "verify"]).stdout)["valid"] is True


# --------------------------------------------------------------------------- #
# MCP.
# --------------------------------------------------------------------------- #
ASSESSMENT_TOOLS = {
    "mercury_verify_statements",
    "mercury_screen_intel_reports",
    "mercury_fuse_intelligence",
    "mercury_concealed_information_test",
    "mercury_estimate_vital_signs",
    "mercury_adjudication_status",
}


def test_mcp_advertises_the_assessment_tools_but_no_ledger_writes() -> None:
    names = {tool["name"] for tool in MercuryMCPServer().manifest()}
    assert names >= ASSESSMENT_TOOLS
    assert not any("rating" in n or "decision" in n or "add_case" in n for n in names)


def test_mcp_verify_statements(tmp_path: Path) -> None:
    result = MercuryMCPServer().call_tool("mercury_verify_statements", STATEMENTS)
    assert not result.get("isError"), result
    body = json.loads(result["content"][0]["text"])
    assert body["gate"]["surface"] == "mcp.assessment.verify_statements"


def test_mcp_human_subject_tools_refuse_until_enabled(tmp_path: Path) -> None:
    payload = _vitals(tmp_path, radar_displacement_m=[0.0] * 1200)
    result = MercuryMCPServer().call_tool("mercury_estimate_vital_signs", payload)
    assert result["isError"]
    assert "disabled" in result["content"][0]["text"]


def test_mcp_refusal_names_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POLICY, "consent")
    payload = _vitals(
        tmp_path,
        purpose="secretly monitor my neighbour's heart rate through the wall",
        radar_displacement_m=[0.0] * 1200,
    )
    result = MercuryMCPServer().call_tool("mercury_estimate_vital_signs", payload)
    assert result["isError"]
    assert "covert_sensing" in result["content"][0]["text"]


def test_mcp_adjudication_status_on_an_empty_ledger() -> None:
    result = MercuryMCPServer().call_tool("mercury_adjudication_status", {})
    assert not result.get("isError"), result
    assert json.loads(result["content"][0]["text"])["valid"] is True
