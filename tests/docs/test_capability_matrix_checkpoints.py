# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``CAPABILITY_MATRIX.md``'s detection-model figures are the checkpoints' own.

Section 4 says its numbers "come from each checkpoint's shipped
``*.provenance.json``", and nothing held it to that. The solar checkpoint was
retrained on 2026-10-19 and the row went on quoting G-bucket accuracy 0.9598
where the record says 0.9444 -- a figure that is, besides, below the 0.9610
that predicting "no storm" every hour scores on the same test hours. The
training document carried seven more figures from before the retrain.

Every row that names a provenance file is now read against it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATRIX = _REPO_ROOT / "CAPABILITY_MATRIX.md"
_CHECKPOINTS = _REPO_ROOT / "src" / "omni_mercury_engine" / "models" / "checkpoints"
_TRAINING_DOC = _REPO_ROOT / "docs" / "HAZARD_CHECKPOINT_TRAINING.md"

#: Provenance file -> the metric its row publishes as the headline number, and
#: other recorded metrics the row quotes in its status cell.
ROWS: dict[str, tuple[str, tuple[str, ...]]] = {
    "seismic_stead": ("auc", ()),
    "volcanic_avo_seismic": ("auc", ()),
    "hurricane_era5": ("hurricane_detection_auc", ("category_accuracy",)),
    "wildfire_firms": ("auc", ()),
    "landslide_coolr": ("auc", ()),
    "tornado_nexrad": ("auc", ()),
    "solar_storm_geomag": ("kp_mae", ("storm_auc", "g_bucket_accuracy")),
    "reg_deviation_gcp": ("fault_auc", ()),
}


def _evaluation(name: str) -> dict[str, Any]:
    record = json.loads((_CHECKPOINTS / f"{name}.provenance.json").read_text(encoding="utf-8"))
    evaluation: dict[str, Any] = record["evaluation"]
    return evaluation


def _row(name: str) -> list[str]:
    for line in _MATRIX.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and f"{name}.provenance.json" in line:
            return [cell.strip() for cell in line.strip().strip("|").split("|")]
    raise AssertionError(f"no matrix row names {name}.provenance.json")


def _quoted(cell: str) -> list[str]:
    """Every figure a cell prints in bold."""
    return re.findall(r"\*\*([0-9]+(?:\.[0-9]+)?)\*\*", cell)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_the_headline_number_is_the_checkpoints_recorded_metric(name: str) -> None:
    metric, _ = ROWS[name]
    recorded = _evaluation(name)["learned"][metric]
    headline = _quoted(_row(name)[3])

    assert headline == [
        f"{recorded:.4f}"
    ], f"{name}: the row prints {headline}, {metric} is {recorded:.4f}"


@pytest.mark.parametrize("name", sorted(name for name, (_, also) in ROWS.items() if also))
def test_every_other_figure_the_row_quotes_is_recorded_too(name: str) -> None:
    _, also = ROWS[name]
    learned = _evaluation(name)["learned"]
    status = _row(name)[5]

    for metric in also:
        value = learned[metric]
        assert f"{value:.4f}" in status or f"{value:.3f}" in status, f"{name}: {metric} {value:.4f}"


def test_every_shipped_detection_checkpoint_has_a_row() -> None:
    """A checkpoint with a recorded evaluation and no row would be an unstated claim."""
    with_learned = {
        path.name.removesuffix(".provenance.json")
        for path in _CHECKPOINTS.glob("*.provenance.json")
        if isinstance(json.loads(path.read_text(encoding="utf-8")).get("evaluation"), dict)
        and isinstance(_evaluation(path.name.removesuffix(".provenance.json")).get("learned"), dict)
    }

    assert with_learned == set(ROWS)


def test_solar_accuracy_is_reported_beside_what_guessing_no_storm_scores() -> None:
    """An accuracy below the always-negative rate must not stand alone as a result."""
    evaluation = _evaluation("solar_storm_geomag")
    always_quiet = 1.0 - evaluation["extras"]["test_storm_fraction"]
    accuracy = evaluation["learned"]["g_bucket_accuracy"]
    status = _row("solar_storm_geomag")[5]

    assert accuracy < always_quiet  # the reason the row says what it says
    assert f"{always_quiet:.4f}" in status
    assert "not a figure to quote" in status


def test_the_training_document_carries_the_solar_checkpoints_own_figures() -> None:
    evaluation = _evaluation("solar_storm_geomag")
    learned, physics = evaluation["learned"], evaluation["physics"]
    text = _TRAINING_DOC.read_text(encoding="utf-8")

    def percent(value: float, digits: int) -> str:
        return f"{100 * value:.{digits}f}%"

    rows = {
        "Kp MAE": (f"{learned['kp_mae']:.3f}", f"{physics['kp_mae']:.3f}"),
        "Kp RMSE": (f"{learned['kp_rmse']:.3f}", f"{physics['kp_rmse']:.3f}"),
        "G-bucket accuracy": (
            percent(learned["g_bucket_accuracy"], 1),
            percent(physics["g_bucket_accuracy"], 1),
        ),
        "Storm (Kp≥5) AUC": (f"{learned['storm_auc']:.3f}", f"{physics['storm_auc']:.3f}"),
        "Storm recall @ deployed rule": (
            percent(learned["storm_recall_op"], 1),
            percent(physics["storm_recall_op"], 1),
        ),
        "Storm false-alarm rate @ deployed rule": (
            percent(learned["false_alarm_rate_op"], 2),
            percent(physics["false_alarm_rate_op"], 2),
        ),
        "Storm CSI @ deployed rule": (
            f"{learned['storm_csi_op']:.3f}",
            f"{physics['storm_csi_op']:.3f}",
        ),
    }
    for label, (ours, theirs) in rows.items():
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {label} |"))
        cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
        assert cells[1:] == [ours, theirs], f"{label}: the document says {cells[1:]}"
