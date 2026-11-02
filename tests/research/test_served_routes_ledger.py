# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The adversarial ledger of the served routes is current, and its floors hold.

``research/adversarial/served_routes.py`` measures the served detection routes
on seeded adversarial scenarios and commits the numbers to
``served_routes_results.json`` and ``docs/DETECTION_QUALITY.md``. Three guards:

* the ledger records the git blob of every file whose content decides the
  numbers; recomputed here, so a change to any of them fails until the suite
  is re-run (fast, every lane);
* the committed numbers must clear the floors below (fast, every lane);
* with ``MERCURY_FULL_LEDGER_CHECK=1`` (the ML lane) the whole suite is
  re-run and must reproduce the ledger and the page to the digit.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytest.importorskip("fastapi", reason="fastapi (optional '[api]' extra) is required")

_REPO = Path(__file__).resolve().parents[2]


def _module() -> ModuleType:
    sys.path.insert(0, str(_REPO / "research" / "adversarial"))
    try:
        return importlib.import_module("served_routes")
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def ledger() -> dict[str, Any]:
    path = _REPO / "research" / "adversarial" / "served_routes_results.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def test_ledger_inputs_are_the_files_on_disk(ledger: dict[str, Any]) -> None:
    """Every file that decides the numbers hashes as the ledger recorded it."""
    suite = _module()
    recorded: dict[str, str] = ledger["provenance"]["inputs"]
    inputs: tuple[str, ...] = suite.INPUTS
    assert set(recorded) == set(inputs)
    stale = [p for p in inputs if str(suite.blob_hash(p)) != recorded[p]]
    assert not stale, (
        f"{stale} changed since the ledger was measured; run "
        "`python research/adversarial/served_routes.py` and commit the ledger and page"
    )


def test_page_is_rendered_from_the_ledger(ledger: dict[str, Any]) -> None:
    suite = _module()
    page = (_REPO / "docs" / "DETECTION_QUALITY.md").read_text(encoding="utf-8")
    rendered: str = suite.render_doc(ledger)
    assert page == rendered


def test_width_false_positives_are_nominal(ledger: dict[str, Any]) -> None:
    for width, row in ledger["width"].items():
        assert row["multivariate_fpr"] <= 0.002, (width, row)
        assert row["fusion_series_flagged"] == 0, (width, row)


@pytest.mark.parametrize(
    ("scenario", "route", "floor"),
    [
        ("point_spike", "univariate_zscore", 1.0),
        ("point_spike", "fusion_rows", 1.0),
        ("seasonal_masking", "univariate_zscore", 1.0),
        ("seasonal_masking", "multivariate", 1.0),
        ("variance_burst", "univariate_zscore", 1.0),
        ("distribution_shift", "fusion_rows", 1.0),
        ("bounded_evasion", "fusion_rows", 0.8),
        ("slow_ramp", "fusion_rows", 0.8),
    ],
)
def test_recall_floors(ledger: dict[str, Any], scenario: str, route: str, floor: float) -> None:
    assert ledger["shape"][scenario][route]["recall"] >= floor


@pytest.mark.parametrize(
    "scenario",
    [
        "point_spike",
        "slow_ramp",
        "level_shift",
        "seasonal_masking",
        "variance_burst",
        "bounded_evasion",
        "distribution_shift",
    ],
)
def test_every_scenario_reaches_a_series_verdict(ledger: dict[str, Any], scenario: str) -> None:
    row = ledger["shape"][scenario]
    assert row["fusion_series_flagged"] == row["runs"], (scenario, row)


def test_rate_independence_is_measured_and_stated(ledger: dict[str, Any]) -> None:
    rate = ledger["rate_independence"]
    # The tier's contamination flags are the parameter by construction ...
    assert rate["flag_rate_slope_vs_contamination"] == pytest.approx(1.0, abs=0.05)
    # ... and the conformal flags do not move with it.
    conformal = [row["conformal_flag_rate"] for row in rate["sweep"].values()]
    assert max(conformal) - min(conformal) <= 0.01


def test_attribution_names_uninformative_members(ledger: dict[str, Any]) -> None:
    members = ledger["attribution"]["members"]
    assert members["rca"]["own_auroc"] == pytest.approx(0.5, abs=0.01)
    assert members["frequent_pattern"]["own_auroc"] == pytest.approx(0.5, abs=0.01)
    assert max(m["marginal_gain"] for m in members.values()) > 0.01


@pytest.mark.skipif(
    os.getenv("MERCURY_FULL_LEDGER_CHECK") != "1",
    reason="full recomputation of the adversarial ledger runs on the ML lane (MERCURY_FULL_LEDGER_CHECK=1)",
)
def test_ledger_reproduces_to_the_digit() -> None:
    suite = _module()
    exit_code: int = suite.main(["--check"])
    assert exit_code == 0
