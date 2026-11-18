# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The storm forecast's skill is measured walk-forward, and said as measured.

``research/forecast_kp/measure.py`` fits the forecast on the Kp record and
scores it on years it had not seen when it forecast them. The numbers are in
``evidence.json`` beside it and the fitted model ships in the package. These
tests fit it again from the record and fail if the evidence, the shipped
model or the document have drifted from what the record gives.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from omni_mercury_engine.watch.forecast import FEATURES, load_model
from research.forecast_kp import measure

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE: dict[str, Any] = json.loads(measure.EVIDENCE.read_text(encoding="utf-8"))
DOC = " ".join((_REPO_ROOT / "docs" / "WATCH.md").read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def measured() -> tuple[dict[str, Any], dict[str, Any]]:
    return measure.measure()


def test_the_evidence_is_what_the_record_gives(
    measured: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    evidence, _ = measured

    assert evidence["record"] == EVIDENCE["record"]
    assert evidence["model"] == EVIDENCE["model"]
    for part in ("walk_forward", "after_the_fit"):
        assert _close(evidence[part], EVIDENCE[part]), part


def _close(fresh: Any, kept: Any) -> bool:
    """The same, with numbers allowed the last digit they were rounded to."""
    if isinstance(kept, dict):
        return set(fresh) == set(kept) and all(_close(fresh[key], kept[key]) for key in kept)
    if isinstance(kept, list):
        return len(fresh) == len(kept) and all(map(_close, fresh, kept))
    if isinstance(kept, float):
        return bool(abs(float(fresh) - kept) <= 2e-4)
    return bool(fresh == kept)


def test_the_model_that_ships_is_the_one_the_record_fits(
    measured: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, fitted = measured
    shipped = json.loads(measure.MODEL.read_text(encoding="utf-8"))

    assert shipped["features"] == list(FEATURES) == fitted["features"]
    assert shipped["trained_through"] == measure.TRAINED_THROUGH.isoformat()
    assert shipped["fitted_on_days"] == fitted["fitted_on_days"]
    for name in ("mean", "scale", "weights"):
        assert shipped[name] == pytest.approx(fitted[name], abs=1e-5), name
    assert shipped["bias"] == pytest.approx(fitted["bias"], abs=1e-5)
    assert load_model().climatology == pytest.approx(fitted["climatology"], abs=1e-6)


def test_no_year_is_forecast_by_a_model_that_had_seen_it() -> None:
    """The walk is forward: the scores start where there are years before to fit on."""
    walk = EVIDENCE["walk_forward"]

    assert walk["first_year"] == measure.FIRST_FORECAST_YEAR == 1960
    assert walk["last_year"] == measure.TRAINED_THROUGH.year
    assert date.fromisoformat(EVIDENCE["after_the_fit"]["first"]) > measure.TRAINED_THROUGH
    assert walk["all"]["forecasts"] == sum(
        period["forecasts"] for period in walk["by_period"].values()
    )


def test_it_beats_both_references_in_every_period_and_after_the_fit() -> None:
    """Modestly: this is what the forecast is worth, and all it is worth."""
    walk = EVIDENCE["walk_forward"]

    for told in (walk["all"], *walk["by_period"].values(), EVIDENCE["after_the_fit"]):
        assert told["references"]["climatology"]["skill"] > 0.05
        assert told["references"]["persistence"]["skill"] > 0.05
    assert 0.10 < walk["all"]["references"]["climatology"]["skill"] < 0.20


def test_what_it_says_at_each_probability_is_about_how_often_it_happened() -> None:
    for row in EVIDENCE["walk_forward"]["all"]["reliability"]:
        if row["forecasts"] >= 300:
            assert abs(row["mean_probability"] - row["observed_rate"]) < 0.07, row


def test_the_document_gives_the_measured_figures() -> None:
    walk, after = EVIDENCE["walk_forward"]["all"], EVIDENCE["after_the_fit"]

    for figure in (
        f"{walk['forecasts']:,} days",
        f"Brier score of {walk['brier']}",
        f"{walk['references']['climatology']['brier']} for the long-run rate",
        f"{walk['references']['persistence']['brier']} for persistence",
        f"{walk['references']['climatology']['skill']:.1%}",
        f"{walk['references']['persistence']['skill']:.1%}",
        f"{after['forecasts']} days",
        f"{after['references']['climatology']['skill']:.1%}",
        f"{after['references']['persistence']['skill']:.1%}",
        f"{EVIDENCE['record']['storm_day_rate']:.1%}",
    ):
        assert figure in DOC, figure


def test_the_record_is_whole_and_says_whose_it_is() -> None:
    record, stated = measure.read_record()
    head = measure.RECORD.read_text(encoding="utf-8")[:900]

    assert stated["first"] == date(1932, 1, 1) and len(record) == EVIDENCE["record"]["days"]
    assert not any(value is None for values in record.values() for value in values)
    assert "CC BY 4.0" in head and "10.1029/2020SW002641" in head
