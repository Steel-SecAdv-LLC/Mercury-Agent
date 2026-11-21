# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``docs/DISCOVERY.md`` states measured rates; they are the artifacts' rates.

The guide tells a professional how often each discovery instrument is wrong.
Those figures come from two committed harness artifacts.  If a harness is
re-run and its numbers move, the guide has to move with them: this test reads
the artifacts and looks for each published figure in the page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DOC = (_REPO / "docs" / "DISCOVERY.md").read_text(encoding="utf-8")


def _artifact(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((_REPO / "artifacts" / name).read_text(encoding="utf-8"))
    return loaded


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def test_the_novelty_rates_are_the_artifacts() -> None:
    novelty = _artifact("novelty_discovery.json")
    mean = novelty["mean"]
    worst = novelty["verdict"]["rank_normal_called_novel_worst_table"]
    current = (
        f"| Rank test at 0.05 (current) | {_pct(mean['rank_normal_called_novel'])} "
        f"(worst table {_pct(worst)}) | {_pct(mean['rank_anomalies_called_novel'])} |"
    )
    replaced = (
        f"| Score at or above 0.7 (replaced) | {_pct(mean['score_cut_normal_called_novel'])} "
        f"| {_pct(mean['score_cut_anomalies_called_novel'])} |"
    )
    assert current in _DOC
    assert replaced in _DOC


@pytest.mark.parametrize(
    ("label", "scenario"),
    [
        ("Independent noise, 200 rows", "noise_n200_d5"),
        ("Independent noise, 60 rows", "noise_n60_d4"),
        ("Autocorrelated (AR(1) 0.7), independent columns", "ar1_0.7_n200_d5"),
        ("Common shock (9 sd, 6 rows)", "common_shock_9sd_6rows_n200_d5"),
        ("Common shift (3 sd, 40 rows)", "common_shift_3sd_40rows_n200_d5"),
        ("Common trend", "common_trend_n200_d5"),
        ("Random walk", "random_walk_n200_d5"),
    ],
)
def test_each_false_edge_row_is_the_artifacts(label: str, scenario: str) -> None:
    row = _artifact("causal_discovery.json")["results"][scenario]
    expected = (
        f"| {label} | {_pct(row['prior']['false_edge'])} | "
        f"{_pct(row['shipped']['false_edge'])} | {_pct(row['shipped']['not_assessed'])} |"
    )
    assert expected in _DOC


@pytest.mark.parametrize(
    ("label", "scenario"),
    [
        ("One planted edge, 200 rows", "planted_n200_d4"),
        ("One planted edge with a common shock, 200 rows", "planted_with_shock_n200_d4"),
        ("One planted edge, 100 rows", "planted_n100_d4"),
    ],
)
def test_each_planted_edge_row_is_the_artifacts(label: str, scenario: str) -> None:
    row = _artifact("causal_discovery.json")["results"][scenario]["shipped"]
    expected = (
        f"| {label} | {_pct(row['true_edge'])} | {_pct(row['false_edge'])} | "
        f"{_pct(row['not_assessed'])} |"
    )
    assert expected in _DOC


def test_the_limit_is_stated_with_its_number() -> None:
    declined = _artifact("causal_discovery.json")["results"]["ar1_0.7_n200_d5"]["shipped"][
        "not_assessed"
    ]
    assert f"({_pct(declined)} of the AR(1) 0.7 windows above)" in _DOC
    assert "is the open problem" in _DOC
    assert "no probability of its own" in _DOC
