# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``mercury-agent review FILE``: one file in, the three answers and its DecisionRecord out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from click.testing import CliRunner

from omni_mercury_engine.cli import main as cli
from omni_mercury_engine.cognitive import decision_gate
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.decision import DecisionRecord
from omni_mercury_engine.review import answers
from tests.review import builders

SAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data" / "review"
SAMPLE = {
    "document": SAMPLES / "sample_document.txt",
    "image": SAMPLES / "sample_image.png",
    "audio": SAMPLES / "sample_recording.wav",
}


@pytest.mark.parametrize("kind", sorted(SAMPLE))
def test_one_file_gives_the_three_answers_beside_its_complete_record(kind: str) -> None:
    outcome = CliRunner().invoke(cli, ["review", str(SAMPLE[kind])])
    assert outcome.exit_code == 0, outcome.output
    text = outcome.output
    positions = [text.index(heading) for heading in answers.HEADINGS]
    assert positions == sorted(positions)
    assert positions[-1] < text.index("THE COMPLETE DECISION RECORD")
    assert '"disposition":' in text and '"signals":' in text
    assert f"({kind}, " in text.splitlines()[0]
    assert text.rstrip().splitlines()[-1].strip() == f"mercury-agent review {SAMPLE[kind]}"


def test_json_prints_the_whole_result(tmp_path: Path) -> None:
    kept = tmp_path / "kept.json"
    outcome = CliRunner().invoke(
        cli, ["review", str(SAMPLE["document"]), "--json", "-o", str(kept), "--kind", "document"]
    )
    assert outcome.exit_code == 0, outcome.output
    result = json.loads(outcome.output)
    assert json.loads(kept.read_text(encoding="utf-8")) == result
    assert DecisionRecord.from_dict(result["decision"]).calibrated is True
    assert result["review"]["kind"] == "document"
    assert tuple(result["review"]["answers"]) == answers.KEYS


def test_a_file_that_cannot_be_reviewed_exits_2_with_the_reason(tmp_path: Path) -> None:
    short = tmp_path / "short.txt"
    short.write_text(builders.prose(60), encoding="utf-8")
    outcome = CliRunner().invoke(cli, ["review", str(short)])
    assert outcome.exit_code == 2
    assert "at least 12" in outcome.output

    compressed = tmp_path / "voice.mp3"
    compressed.write_bytes(b"ID3\x04\x00" + b"\x00" * 64)
    outcome = CliRunner().invoke(cli, ["review", str(compressed)])
    assert outcome.exit_code == 2
    assert "ffmpeg" in outcome.output

    outcome = CliRunner().invoke(cli, ["review", str(SAMPLE["image"]), "--kind", "audio"])
    assert outcome.exit_code == 2

    outcome = CliRunner().invoke(cli, ["review", str(tmp_path / "not-there.pdf")])
    assert outcome.exit_code == 2
    assert "cannot read" in outcome.output and "Traceback" not in outcome.output


def test_an_ethical_refusal_exits_1_and_reviews_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in gate that refuses everything: the command must not print a record."""

    def refuses(*_args: Any, **_kwargs: Any) -> Any:
        raise EthicalConstraintViolationError(
            action="review", score=0.0, threshold=1.0, check="harm_uplift", details={}
        )

    monkeypatch.setattr(decision_gate, "enforce_decision_boundary", refuses)
    outcome = CliRunner().invoke(cli, ["review", str(SAMPLE["document"])])
    assert outcome.exit_code == 1
    assert "harm_uplift" in outcome.output
    assert "DECISION RECORD" not in outcome.output
