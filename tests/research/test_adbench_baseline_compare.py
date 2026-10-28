# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pin the ADBench base-vs-current comparison behind the committed artifacts.

``research/omni_equation/_compare.py`` is engine-free, so this suite imports the
diff logic directly (not through the detector) and asserts that the committed
base run (``adbench_base_e118e1f.json``) diffed against the committed current
run (``adbench_results.json``) yields the ledger the README cites, and that
``adbench_base_vs_current.json`` stays in sync with a fresh recompute from
those two sources.

**The ledger moved in the v3 audit, and this suite is why that was visible.**
It pinned the PR-302 headline (14 W / 2 tie / 2 L, +0.0237 to a mean of
0.7634), which was recorded at ``bdcdf6a``. Seven commits of
``detectors/statistical.py`` (+1,199 / −105 lines) later, re-running the
harness measured 0.7745. The base was re-measured too, in a matched
environment (numpy 2.5.3, scipy 1.18.1, AMA 3.2.0 — the release the base tree
pins), and it reproduced the committed 0.7397 on all 18 sets byte-for-byte,
which rules the environment out and attributes the move to the detector. The
refreshed ledger is a larger mean gain on a worse win rate: 13 W / 1 tie / 4 L,
+0.0348. ``cardio`` (−0.0855) and ``optdigits`` (−0.0624) are the regressions
to look at.
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

    def test_base_vs_current_is_13_1_4(self) -> None:
        comparison = compare_to_baseline(
            _results("adbench_results.json"), _results("adbench_base_e118e1f.json")
        )
        summary = comparison["summary"]
        assert (summary["wins"], summary["ties"], summary["losses"]) == (13, 1, 4)
        assert summary["mean_baseline"] == 0.7397
        assert summary["mean_current"] == 0.7745
        assert summary["mean_delta"] == 0.0348

    def test_the_current_run_records_the_detector_it_measured(self) -> None:
        """Both sides date themselves, so a stale ledger cannot hide again.

        The PR-302 current-side run carried no provenance at all, which is why
        it outlived its detector unnoticed; ``harness_adbench.py`` now stamps
        commit, detector commit, timestamp and library versions on every run.
        """
        for filename in ("adbench_results.json", "adbench_base_e118e1f.json"):
            summary = json.loads((_RESEARCH / filename).read_text())["summary"]
            assert summary.get("detector_commit"), f"{filename} records no detector_commit"

    def test_named_ties_and_losses(self) -> None:
        comparison = compare_to_baseline(
            _results("adbench_results.json"), _results("adbench_base_e118e1f.json")
        )
        verdict = {row["dataset"]: row["verdict"] for row in comparison["per_set"]}
        assert verdict["Hepatitis"] == "tie"
        # The four regressions, named so a later run cannot quietly add a fifth.
        assert verdict["cardio"] == "loss"
        assert verdict["optdigits"] == "loss"
        assert verdict["Cardiotocography"] == "loss"
        assert verdict["vertebral"] == "loss"
        # Both were losses under the PR-302 ledger and are wins now.
        assert verdict["Waveform"] == "win"
        assert verdict["WPBC"] == "win"

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
        # Was the real Waveform delta under the PR-302 ledger (-0.0003:
        # negligible, still a loss at tie_tol=0); Waveform is a +0.1289 win in
        # the refreshed one, so these are now illustrative literals for the
        # rule itself rather than a reference to a live row.
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
