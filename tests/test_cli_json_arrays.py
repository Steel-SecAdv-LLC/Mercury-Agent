# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``detect -o`` keeps every value of an array result.

The CLI serialised with ``json.dump(..., default=str)``, which rendered each
ndarray as numpy's abbreviated repr (``"[0.12 0.13 ... 0.26]"``). The default
statistical detector's per-row scores, z-scores and flags were therefore
unusable in the saved file: six values survived out of every array.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
from click.testing import CliRunner

from omni_mercury_engine.cli import _json_default, main

if TYPE_CHECKING:
    from pathlib import Path


def test_arrays_and_scalars_keep_their_values() -> None:
    payload: dict[str, Any] = {
        "scores": np.arange(2000, dtype=np.float32) / 2000,
        "flags": np.zeros(2000, dtype=bool),
        "count": np.int64(7),
        "ratio": np.float32(0.5),
        "labels": {"b", "a"},
        "other": object,
    }
    loaded = json.loads(json.dumps(payload, default=_json_default))

    assert loaded["scores"] == payload["scores"].tolist()
    assert loaded["flags"] == [False] * 2000
    assert loaded["count"] == 7 and loaded["ratio"] == 0.5
    assert loaded["labels"] == ["a", "b"]
    assert isinstance(loaded["other"], str)


def test_detect_output_file_holds_one_score_per_row(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(size=(300, 4))
    data[:10] += 6.0
    source = tmp_path / "data.csv"
    np.savetxt(source, data, delimiter=",", header="a,b,c,d", comments="")
    out = tmp_path / "out.json"

    result = CliRunner().invoke(main, ["detect", "-i", str(source), "-o", str(out)])

    assert result.exit_code == 0, result.output
    statistical = json.loads(out.read_text(encoding="utf-8"))["detectors"]["statistical"]
    scores = statistical["scores"]
    assert isinstance(scores, list) and len(scores) == 300
    assert all(isinstance(value, float) for value in scores)
    assert np.mean(scores[:10]) > np.mean(scores[10:])
