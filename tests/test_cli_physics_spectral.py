# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``mercury physics spectral``: a real baseline, real descriptors, honest output.

The command fitted on its input and scored the same input, so its
``anomaly_score`` was one constant (0.1337) for a clean tone and for one with
a noise burst, and ``spectral_entropy``, ``harmonic_distortion`` and
``vibration_signature`` were fixed placeholders (0.0, 0.0, "unknown") because
it read result keys the detector does not produce.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import main

if TYPE_CHECKING:
    from pathlib import Path

N = 2048


def _tone(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N) / 1000.0
    return np.sin(2 * np.pi * 100 * t) + 0.1 * rng.standard_normal(N)


def _faulted(rng: np.random.Generator) -> np.ndarray:
    x = _tone(rng)
    x[1000:1200] += 4.0 * rng.standard_normal(200)  # a burst of broadband noise
    return x


def _run(*args: str) -> dict[str, Any]:
    result = CliRunner().invoke(main, ["physics", "spectral", *args])
    assert result.exit_code == 0, result.output
    text = result.output
    parsed: dict[str, Any] = json.loads(text[text.index("{") :])
    return parsed


@pytest.fixture
def recordings(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    paths = {
        "healthy": tmp_path / "healthy.csv",
        "clean": tmp_path / "clean.csv",
        "faulted": tmp_path / "faulted.csv",
    }
    np.savetxt(paths["healthy"], np.stack([_tone(rng) for _ in range(10)]), delimiter=",")
    np.savetxt(paths["clean"], _tone(rng).reshape(1, -1), delimiter=",")
    np.savetxt(paths["faulted"], _faulted(rng).reshape(1, -1), delimiter=",")
    return paths


def test_a_reference_gives_a_real_verdict(recordings: dict[str, Path]) -> None:
    """Before: no such option, and the score was the same constant either way."""
    clean = _run("-i", str(recordings["clean"]), "--reference", str(recordings["healthy"]))
    faulted = _run("-i", str(recordings["faulted"]), "--reference", str(recordings["healthy"]))

    assert clean["baseline"] == faulted["baseline"] == "reference"
    assert "warning" not in clean
    assert clean["is_anomaly"] is False
    assert faulted["is_anomaly"] is True
    assert faulted["anomaly_score"] > clean["anomaly_score"] + 0.3


def test_without_a_reference_the_output_says_the_verdict_means_nothing(
    recordings: dict[str, Path],
) -> None:
    out = _run("-i", str(recordings["clean"]))
    assert out["baseline"] == "input_itself"
    assert "do not measure a fault" in out["warning"]
    # Released keys and types are kept.
    assert isinstance(out["is_anomaly"], bool)
    assert isinstance(out["anomaly_score"], float)


def test_the_descriptors_are_the_inputs_own_not_placeholders(
    recordings: dict[str, Path],
) -> None:
    clean = _run("-i", str(recordings["clean"]))
    faulted = _run("-i", str(recordings["faulted"]))

    for out in (clean, faulted):
        assert 0.0 < out["spectral_entropy"] < 1.0
        assert out["harmonic_distortion"] > 0.0
        assert out["vibration_signature"] != "unknown"
    # A burst of broadband noise flattens the spectrum and adds distortion.
    assert faulted["spectral_entropy"] > clean["spectral_entropy"] + 0.2
    assert faulted["harmonic_distortion"] > clean["harmonic_distortion"]
    assert clean["vibration_signature"] != faulted["vibration_signature"]


def test_an_input_too_short_for_a_baseline_is_an_error_not_a_score(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    np.savetxt(path, np.sin(np.arange(100) * 0.1).reshape(-1, 1), delimiter=",")
    result = CliRunner().invoke(main, ["physics", "spectral", "-i", str(path)])
    assert result.exit_code == 1
    assert "cannot estimate a reference spread" in result.output
