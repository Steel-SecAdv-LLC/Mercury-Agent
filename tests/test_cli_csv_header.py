# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A CSV that starts with column names loads; a numeric first row is data.

``mercury-agent detect -i data.csv`` crashed with ``could not convert string
to float: 'a'`` on any CSV exported with a header row, the most common shape.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import _csv_header_rows, _load_data, _load_labels, main

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    ("first_line", "rows"),
    [
        ("a,b,c", 1),
        ('"x","y"', 1),
        ("label", 1),
        ("sensor 1,sensor 2", 1),
        ("1.0,2.0,3.0", 0),
        ("1e-3,-4,+5.5", 0),
        ("nan,inf,-inf", 0),
        ("7", 0),
    ],
)
def test_header_detection(tmp_path: Path, first_line: str, rows: int) -> None:
    path = tmp_path / "f.csv"
    path.write_text(first_line + "\n1,2,3\n", encoding="utf-8")
    assert _csv_header_rows(path) == rows


def test_data_with_header_equals_data_without(tmp_path: Path) -> None:
    body = "1.5,2.5,3.5\n4,5,6\n7,8,9\n"
    plain = _load_data(_write(tmp_path / "plain.csv", body))
    headed = _load_data(_write(tmp_path / "headed.csv", "a,b,c\n" + body))
    np.testing.assert_array_equal(headed, plain)
    assert headed.shape == (3, 3) and headed.dtype == np.float32


def test_a_single_row_after_the_header_stays_two_dimensional(tmp_path: Path) -> None:
    assert _load_data(_write(tmp_path / "one.csv", "a,b,c\n1,2,3\n")).shape == (1, 3)


def test_numeric_first_row_is_never_dropped(tmp_path: Path) -> None:
    data = _load_data(_write(tmp_path / "n.csv", "1e-3,2\n3,4\n"))
    np.testing.assert_allclose(data, np.array([[1e-3, 2], [3, 4]]))


def test_labels_with_header(tmp_path: Path) -> None:
    labels = _load_labels(_write(tmp_path / "y.csv", "label\n0\n1\n1\n"))
    np.testing.assert_array_equal(labels, [0, 1, 1])


def test_non_numeric_body_still_fails_clearly(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.csv", "a,b\n1,2\n3,oops\n")
    with pytest.raises(ValueError, match="oops"):
        _load_data(path)


def test_detect_runs_on_a_csv_with_a_header(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(120, 4))
    rows[7] += 9
    text = "a,b,c,d\n" + "\n".join(",".join(f"{v:.6f}" for v in row) for row in rows) + "\n"
    out = tmp_path / "out.json"
    result = CliRunner().invoke(
        main,
        ["detect", "-i", _write(tmp_path / "in.csv", text), "-d", "statistical", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))
