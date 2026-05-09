# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Tests for `benchmarks.seven_axis_runner`.

Pins:

- The runner emits exactly the seven named axes.
- Every axis score is in [0, 1].
- The runner is deterministic for a fixed seed.
- ``regenerate_docs`` is idempotent (running it twice yields the same file
  bytes), which is what lets CI `git diff --exit-code` after regeneration.
- The JSON payload round-trips through `json.loads`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from benchmarks.seven_axis_runner import (
    AXIS_FUNCTIONS,
    DEFAULT_SEED,
    SECTION_FOOTER,
    SECTION_HEADER,
    regenerate_docs,
    render_markdown_table,
    run_seven_axis,
)

EXPECTED_AXES = [
    "Generalization",
    "Scalability",
    "Data Efficiency",
    "Reasoning",
    "Robustness",
    "Transferability",
    "Interpretability",
]


def test_runner_emits_exactly_seven_named_axes() -> None:
    report = run_seven_axis(seed=DEFAULT_SEED)
    names = [a.name for a in report.axes]
    assert names == EXPECTED_AXES
    assert len(AXIS_FUNCTIONS) == 7


def test_every_axis_score_is_in_unit_interval() -> None:
    report = run_seven_axis(seed=DEFAULT_SEED)
    for axis in report.axes:
        assert 0.0 <= axis.score <= 1.0, f"Axis {axis.name!r} score {axis.score} out of [0, 1]"


def test_runner_is_deterministic_for_fixed_seed() -> None:
    """Non-timing axes are bytewise reproducible; the timing-based Scalability
    axis is allowed a small tolerance because it measures wall-clock."""
    a = run_seven_axis(seed=DEFAULT_SEED)
    b = run_seven_axis(seed=DEFAULT_SEED)

    a_by_name = {x.name: x.score for x in a.axes}
    b_by_name = {x.name: x.score for x in b.axes}

    timing_axes = {"Scalability"}

    for name, a_score in a_by_name.items():
        b_score = b_by_name[name]
        if name in timing_axes:
            assert abs(a_score - b_score) <= 0.10, (
                f"{name}: wall-clock-based axis varied by more than 0.10 "
                f"between runs ({a_score:.4f} vs {b_score:.4f})"
            )
        else:
            assert (
                a_score == b_score
            ), f"{name}: deterministic axis was not reproducible ({a_score} vs {b_score})"


def test_json_payload_roundtrips() -> None:
    report = run_seven_axis(seed=DEFAULT_SEED)
    payload = report.to_dict()
    text = json.dumps(payload)
    parsed = json.loads(text)
    assert parsed["seed"] == DEFAULT_SEED
    assert len(parsed["axes"]) == 7
    for entry in parsed["axes"]:
        assert {"name", "score", "higher_is_better", "raw", "notes"} <= entry.keys()


def test_render_markdown_table_contains_all_axes_and_section_markers() -> None:
    report = run_seven_axis(seed=DEFAULT_SEED)
    md = render_markdown_table(report)
    assert SECTION_HEADER in md
    assert SECTION_FOOTER in md
    for name in EXPECTED_AXES:
        assert f"| {name} |" in md, f"axis {name!r} missing from markdown table"


def test_regenerate_docs_is_idempotent(tmp_path: Path) -> None:
    """Two consecutive calls produce identical bytes — `git diff` cleanliness."""
    docs = tmp_path / "BENCHMARKS.md"
    docs.write_text("# Mercury Agent Benchmark Results\n\nplaceholder body\n", encoding="utf-8")
    report = run_seven_axis(seed=DEFAULT_SEED)
    regenerate_docs(report, docs_path=docs)
    first = docs.read_bytes()
    regenerate_docs(report, docs_path=docs)
    second = docs.read_bytes()
    assert first == second


def test_regenerate_docs_replaces_section_in_place(tmp_path: Path) -> None:
    """Existing Seven-Axis section is replaced, not duplicated."""
    docs = tmp_path / "BENCHMARKS.md"
    docs.write_text(
        "# Mercury Agent Benchmark Results\n\nplaceholder body\n\n"
        f"{SECTION_HEADER}\n\nstale content\n\n{SECTION_FOOTER}\n\n"
        "tail content\n",
        encoding="utf-8",
    )
    report = run_seven_axis(seed=DEFAULT_SEED)
    regenerate_docs(report, docs_path=docs)
    text = docs.read_text(encoding="utf-8")
    # Section header appears exactly once after regeneration.
    assert text.count(SECTION_HEADER) == 1
    assert text.count(SECTION_FOOTER) == 1
    assert "stale content" not in text
    assert "tail content" in text  # downstream content preserved


def test_regenerate_docs_appends_when_no_section_exists(tmp_path: Path) -> None:
    docs = tmp_path / "BENCHMARKS.md"
    docs.write_text("# Mercury Agent Benchmark Results\n\nbody\n", encoding="utf-8")
    report = run_seven_axis(seed=DEFAULT_SEED)
    regenerate_docs(report, docs_path=docs)
    text = docs.read_text(encoding="utf-8")
    assert SECTION_HEADER in text
    assert SECTION_FOOTER in text


def test_regenerate_docs_raises_when_path_missing(tmp_path: Path) -> None:
    report = run_seven_axis(seed=DEFAULT_SEED)
    with pytest.raises(FileNotFoundError):
        regenerate_docs(report, docs_path=tmp_path / "missing.md")
