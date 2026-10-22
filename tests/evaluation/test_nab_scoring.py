# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``evaluation.nab_scoring`` equals NAB's own scorer.

``tests/fixtures/nab`` holds four real NAB files and the raw scores NAB's
``Sweeper`` (``nab/sweeper.py``) gives seeded alarm patterns on them under
all three profiles (recorded by ``scripts/record_nab_fixtures.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.evaluation.nab_scoring import (
    PROFILES,
    CostProfile,
    NABStream,
    load_nab_corpus,
    normalized_score,
    probationary_length,
    scaled_sigmoid,
    score_alarms,
    score_corpus,
    sweep_table,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "nab"
CATEGORIES = ("realAWSCloudwatch", "realAdExchange", "realKnownCause", "realTraffic")


@pytest.fixture(scope="module")
def corpus() -> list[NABStream]:
    return load_nab_corpus(FIXTURES, CATEGORIES)


def test_reads_the_fixture_corpus(corpus: list[NABStream]) -> None:
    names = [s.name for s in corpus]
    assert names == [
        "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
        "realAdExchange/exchange-3_cpc_results.csv",
        "realKnownCause/rogue_agent_key_hold.csv",
        "realTraffic/speed_7578.csv",
    ]
    traffic = corpus[3]
    assert traffic.values.size == 1127 and len(traffic.windows) == 4
    assert traffic.timestamps.dtype == np.dtype("datetime64[us]")


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_raw_scores_equal_nab_reference(corpus: list[NABStream], profile: str) -> None:
    reference = json.loads((FIXTURES / "reference_scores.json").read_text())
    checked = 0
    for stream in corpus:
        entry = reference[stream.name]
        table = sweep_table(stream.values.size, stream.windows, profile)
        for indices, expected in zip(
            entry["alarm_patterns"], entry["raw_scores"][profile], strict=True
        ):
            alarms = np.zeros(stream.values.size, dtype=bool)
            alarms[indices] = True
            assert score_alarms(alarms, table).raw == pytest.approx(expected, rel=1e-12, abs=1e-12)
            checked += 1
    assert checked == 4 * 6


def test_scaled_sigmoid_matches_nabs_documented_values() -> None:
    assert scaled_sigmoid(-1.0) == pytest.approx(0.98661, abs=1e-5)
    assert scaled_sigmoid(-0.5) == pytest.approx(0.84828, abs=1e-5)
    assert scaled_sigmoid(0.0) == 0.0
    assert scaled_sigmoid(1.0) == pytest.approx(-0.98661, abs=1e-5)
    assert scaled_sigmoid(3.0) == pytest.approx(2 / (1 + np.exp(15)) - 1)
    assert scaled_sigmoid(3.0001) == -1.0


def test_probationary_length() -> None:
    assert probationary_length(1000) == 150
    assert probationary_length(10_000) == 750.0
    assert probationary_length(4999) == 749


def test_null_scores_zero_and_perfect_scores_one_hundred(corpus: list[NABStream]) -> None:
    tables = {s.name: sweep_table(s.values.size, s.windows) for s in corpus}
    none = score_corpus({s.name: np.zeros(s.values.size, bool) for s in corpus}, tables)
    assert none["score"] == 0.0 and none["detected"] == 0
    assert none["missed"] == sum(t.n_windows for t in tables.values())
    perfect = {}
    for s in corpus:
        alarms = np.zeros(s.values.size, bool)
        for first, _ in s.windows:
            alarms[max(first, int(np.ceil(probationary_length(s.values.size))))] = True
        perfect[s.name] = alarms
    assert score_corpus(perfect, tables)["score"] == pytest.approx(100.0)
    assert normalized_score(-3.0, -4.0, 2.0) == pytest.approx(100 / 6)


def test_window_credit_and_false_alarm_costs() -> None:
    n = 2000  # probation: the first 300 points
    table = sweep_table(n, [(1000, 1099)], "standard")
    early, late = np.zeros(n, bool), np.zeros(n, bool)
    early[1000] = True
    late[1090] = True
    assert score_alarms(early, table).raw == pytest.approx(1.0)  # a perfect hit
    assert score_alarms(early, table).raw > score_alarms(late, table).raw > 0
    twice = early.copy()
    twice[1050] = True  # a second alarm in the same window adds nothing
    assert score_alarms(twice, table).raw == score_alarms(early, table).raw
    false_alarm = early.copy()
    false_alarm[1500] = True  # far past the window: the full fp weight
    assert score_alarms(false_alarm, table).raw == pytest.approx(1.0 - 0.11)
    probation = np.zeros(n, bool)
    probation[100] = True  # not scored
    result = score_alarms(probation, table)
    assert result.raw == -1.0 and result.false_alarms == 0 and result.missed == 1
    low_fn = sweep_table(n, [(1000, 1099)], CostProfile(tp=1.0, fp=0.11, fn=2.0))
    assert score_alarms(np.zeros(n, bool), low_fn).raw == -2.0


def test_invalid_input_is_refused(tmp_path: Path) -> None:
    for windows in ([(10, 20), (15, 30)], [(20, 30), (10, 15)], [(5, 5)], [(90, 120)]):
        with pytest.raises(ValueError):
            sweep_table(100, windows)
    with pytest.raises(ValueError):
        score_alarms(np.zeros(5, bool), sweep_table(100, [(50, 60)]))
    (tmp_path / "data" / "x").mkdir(parents=True)
    (tmp_path / "labels").mkdir()
    (tmp_path / "data" / "x" / "f.csv").write_text(
        "timestamp,value\n2020-01-01 00:00:00,1\n2020-01-01 00:05:00,2\n"
    )
    (tmp_path / "labels" / "combined_windows.json").write_text(
        json.dumps({"x/f.csv": [["2020-01-01 00:01:00.000000", "2020-01-01 00:05:00.000000"]]})
    )
    with pytest.raises(ValueError, match="not a timestamp"):
        load_nab_corpus(tmp_path, ["x"])
    with pytest.raises(FileNotFoundError):
        load_nab_corpus(tmp_path, ["missing"])
