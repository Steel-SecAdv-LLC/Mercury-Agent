# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin the ADBench base-vs-current comparison behind the committed artifacts.

``research/omni_equation/_compare.py`` is engine-free, so this suite imports the
diff logic directly (not through the detector) and asserts that the committed base run
(``adbench_base_e118e1f.json``) diffed against the committed hardened run
(``adbench_results.json``) yields the PR-302 headline — 14 W / 2 tie / 2 L,
+0.0237 — and that the committed ``adbench_base_vs_current.json`` artifact stays
in sync with a fresh recompute from those two sources.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO = Path(__file__).resolve().parents[2]
_RESEARCH = _REPO / "research" / "omni_equation"


def _load_compare() -> Callable[..., dict[str, Any]]:
    """Import ``compare_to_baseline`` from the (non-package) research dir by path."""
    spec = importlib.util.spec_from_file_location("omni_eq_compare", _RESEARCH / "_compare.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast("Callable[..., dict[str, Any]]", module.compare_to_baseline)


compare_to_baseline = _load_compare()


def _results(filename: str) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", json.loads((_RESEARCH / filename).read_text())["results"])


class TestCommittedHeadline:
    """The committed JSON pair must reproduce the headline the PR/docs cite."""

    def test_base_vs_current_is_14_2_2(self) -> None:
        comparison = compare_to_baseline(
            _results("adbench_results.json"), _results("adbench_base_e118e1f.json")
        )
        summary = comparison["summary"]
        assert (summary["wins"], summary["ties"], summary["losses"]) == (14, 2, 2)
        assert summary["mean_baseline"] == 0.7397
        assert summary["mean_current"] == 0.7634
        assert summary["mean_delta"] == 0.0237

    def test_named_ties_and_losses(self) -> None:
        comparison = compare_to_baseline(
            _results("adbench_results.json"), _results("adbench_base_e118e1f.json")
        )
        verdict = {row["dataset"]: row["verdict"] for row in comparison["per_set"]}
        assert verdict["Hepatitis"] == "tie"
        assert verdict["Lymphography"] == "tie"
        assert verdict["Waveform"] == "loss"
        assert verdict["WPBC"] == "loss"

    def test_committed_comparison_artifact_in_sync(self) -> None:
        """adbench_base_vs_current.json must equal a fresh recompute from sources."""
        fresh = compare_to_baseline(
            _results("adbench_results.json"), _results("adbench_base_e118e1f.json")
        )
        committed = json.loads((_RESEARCH / "adbench_base_vs_current.json").read_text())
        assert committed["per_set"] == fresh["per_set"]
        for key in (
            "n_scored",
            "mean_baseline",
            "mean_current",
            "mean_delta",
            "wins",
            "ties",
            "losses",
            "tie_tol",
        ):
            assert committed["summary"][key] == fresh["summary"][key]


class TestVerdictRules:
    """Classification edges that the committed ledger depends on."""

    def test_sub_noise_regression_counts_as_loss(self) -> None:
        # The real Waveform delta: -0.0003 is negligible but still a loss at tie_tol=0.
        comparison = compare_to_baseline(
            [{"dataset": "x", "auroc": 0.5867}], [{"dataset": "x", "auroc": 0.5870}]
        )
        assert comparison["per_set"][0]["verdict"] == "loss"
        assert comparison["summary"]["losses"] == 1

    def test_exact_zero_delta_is_tie(self) -> None:
        comparison = compare_to_baseline(
            [{"dataset": "x", "auroc": 0.6820}], [{"dataset": "x", "auroc": 0.6820}]
        )
        assert comparison["per_set"][0]["verdict"] == "tie"

    def test_only_sets_scored_in_both_runs_are_compared(self) -> None:
        comparison = compare_to_baseline(
            [{"dataset": "a", "auroc": 0.7}, {"dataset": "b", "auroc": 0.9}],
            [{"dataset": "a", "auroc": 0.6}, {"dataset": "b", "error": "load failed"}],
        )
        assert comparison["summary"]["n_scored"] == 1
        assert [row["dataset"] for row in comparison["per_set"]] == ["a"]

    def test_per_set_sorted_by_delta_descending(self) -> None:
        comparison = compare_to_baseline(
            [{"dataset": "big", "auroc": 0.9}, {"dataset": "small", "auroc": 0.61}],
            [{"dataset": "big", "auroc": 0.5}, {"dataset": "small", "auroc": 0.60}],
        )
        deltas = [row["delta"] for row in comparison["per_set"]]
        assert deltas == sorted(deltas, reverse=True)
