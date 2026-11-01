# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``mercury-agent detect -d <unknown>`` is a usage error, not "no anomaly".

``OmniMercuryEngine.detect`` skips detector names it does not hold. The CLI
passed ``--detector`` straight through, so a typo printed
``{"detectors": {}, "is_anomaly": false}`` and exited 0: a clean bill of
health from a detector that never ran.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import main

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")

_CSV = "a,b\n1,2\n1.1,2.1\n0.9,1.9\n1,2\n9,9\n1,2\n1.05,2\n0.95,2\n"


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "d.csv"
    path.write_text(_CSV, encoding="utf-8")
    return path


def test_unknown_detector_is_refused(data_file: Path) -> None:
    result = CliRunner().invoke(main, ["detect", "-i", str(data_file), "-d", "statisical"])
    assert result.exit_code == 2, result.output
    assert "unknown detector 'statisical'" in result.output
    assert "statistical" in result.output  # the valid names are listed
    assert "is_anomaly" not in result.output


def test_known_detector_still_runs(data_file: Path) -> None:
    result = CliRunner().invoke(main, ["detect", "-i", str(data_file), "-d", "statistical"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert "statistical" in payload["detectors"]
