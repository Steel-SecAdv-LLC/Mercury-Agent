# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the offline benchmark-integrity gate (ROADMAP row 17 fallback).

Exercises ``scripts/check_benchmark_integrity.py``: the structural
summary/per-dataset invariants, the headline AUC/F1 *recomputation*
(not restatement), the README-parity check, and the fail-closed error
handling. A final integration test runs the gate against the real
committed ``benchmarks/mercury_benchmark_results.json`` + ``README.md``
so a genuine future drift is caught here, not only by CI.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.check_benchmark_integrity import (
    IntegrityError,
    check,
    main,
)


def _valid_results() -> dict[str, Any]:
    """A minimal internally-consistent results document."""
    per_dataset = [
        {"name": "A", "label_source": "ground_truth", "ensemble_auc": 0.90, "oracle_f1": 0.70},
        {"name": "B", "label_source": "ground_truth", "ensemble_auc": 0.80, "oracle_f1": 0.60},
        {"name": "C", "label_source": "statistical", "ensemble_auc": 0.99, "oracle_f1": 0.99},
        {"name": "D", "label_source": "ground_truth", "ensemble_auc": None, "error": "unavailable"},
    ]
    gt_auc = [0.90, 0.80]
    gt_f1 = [0.70, 0.60]
    summary = {
        "total_datasets": 4,
        "successful": 3,
        "failed": 1,
        "headline_label_policy": "genuine_labels_only (ground_truth | expert_annotated)",
        "n_genuine_labeled": 2,
        "mean_auc": statistics.mean(gt_auc),
        "median_auc": statistics.median(gt_auc),
        "mean_oracle_f1": statistics.mean(gt_f1),
        "median_oracle_f1": statistics.median(gt_f1),
    }
    metadata = {"timestamp": "2026-01-02T03:04:05+00:00", "git_commit": "abcdef1234567890"}
    return {"summary": summary, "per_dataset": per_dataset, "metadata": metadata}


def _readme_for(results: dict[str, Any]) -> str:
    """A README carrying a benchmark block consistent with *results*."""
    s = results["summary"]
    m = results["metadata"]
    return (
        "# Title\n\n"
        "<!-- BENCHMARK:START -->\n"
        "## Latest Benchmark Results\n\n"
        "| Metric | Current | Previous | Δ |\n"
        "|---|---|---|---|\n"
        f"| Mean ROC-AUC | {s['mean_auc']:.4f} | x | x |\n"
        f"| Median ROC-AUC | {s['median_auc']:.4f} | x | x |\n"
        f"| Mean Oracle F1 | {s['mean_oracle_f1']:.4f} | x | x |\n"
        f"| Datasets (successful / total) | {s['successful']} / {s['total_datasets']} | x | x |\n"
        f"| Run timestamp (UTC) | {m['timestamp']} | x | — |\n"
        f"| Commit | `{m['git_commit'][:7]}` | x | — |\n"
        "<!-- BENCHMARK:END -->\n"
    )


def _write(tmp_path: Path, results: dict[str, Any], readme: str | None = None) -> tuple[Path, Path]:
    """Materialise results JSON + README to disk, return their paths."""
    rp = tmp_path / "results.json"
    rp.write_text(json.dumps(results), encoding="utf-8")
    mp = tmp_path / "README.md"
    mp.write_text(readme if readme is not None else _readme_for(results), encoding="utf-8")
    return rp, mp


class TestCleanDocument:
    def test_consistent_document_has_no_violations(self, tmp_path: Path) -> None:
        rp, mp = _write(tmp_path, _valid_results())
        assert check(rp, mp) == []

    def test_main_returns_zero_on_clean_document(self, tmp_path: Path) -> None:
        rp, mp = _write(tmp_path, _valid_results())
        assert main(["--results", str(rp), "--readme", str(mp)]) == 0


class TestStructuralInvariants:
    def test_total_not_equal_successful_plus_failed(self, tmp_path: Path) -> None:
        results = _valid_results()
        results["summary"]["successful"] = 2  # 2 + 1 != 4
        rp, mp = _write(tmp_path, results)
        violations = check(rp, mp)
        assert any("successful" in v for v in violations)

    def test_failed_count_must_match_error_records(self, tmp_path: Path) -> None:
        results = _valid_results()
        results["summary"]["failed"] = 0
        results["summary"]["successful"] = 4
        rp, mp = _write(tmp_path, results)
        assert any("carry an error" in v or "failed" in v for v in check(rp, mp))

    def test_n_genuine_labeled_must_match(self, tmp_path: Path) -> None:
        results = _valid_results()
        results["summary"]["n_genuine_labeled"] = 3  # only 2 ground_truth succeed
        rp, mp = _write(tmp_path, results)
        assert any("n_genuine_labeled" in v for v in check(rp, mp))


class TestHeadlineRecomputation:
    def test_fabricated_mean_auc_is_caught(self, tmp_path: Path) -> None:
        results = _valid_results()
        results["summary"]["mean_auc"] = 0.99  # does not follow from per_dataset
        rp, mp = _write(tmp_path, results)
        assert any("mean_auc" in v for v in check(rp, mp))

    def test_fabricated_oracle_f1_is_caught(self, tmp_path: Path) -> None:
        results = _valid_results()
        results["summary"]["median_oracle_f1"] = 0.10
        rp, mp = _write(tmp_path, results)
        assert any("median_oracle_f1" in v for v in check(rp, mp))

    def test_statistical_labels_excluded_from_headline(self, tmp_path: Path) -> None:
        # The 0.99 statistical dataset must NOT move the headline; a summary
        # that included it would fail the recompute.
        results = _valid_results()
        rp, mp = _write(tmp_path, results)
        assert check(rp, mp) == []


class TestReadmeParity:
    def test_readme_mismatch_is_caught(self, tmp_path: Path) -> None:
        results = _valid_results()
        bad_readme = _readme_for(results).replace(
            "| Mean ROC-AUC | 0.8500", "| Mean ROC-AUC | 0.9999"
        )
        rp, mp = _write(tmp_path, results, readme=bad_readme)
        assert any("Mean ROC-AUC" in v for v in check(rp, mp))

    def test_missing_benchmark_block_raises(self, tmp_path: Path) -> None:
        results = _valid_results()
        rp, mp = _write(tmp_path, results, readme="# no block here\n")
        with pytest.raises(IntegrityError, match="BENCHMARK"):
            check(rp, mp)


class TestFailClosed:
    def test_missing_results_file_returns_two(self, tmp_path: Path) -> None:
        assert (
            main(["--results", str(tmp_path / "nope.json"), "--readme", str(tmp_path / "R.md")])
            == 2
        )

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        rp = tmp_path / "bad.json"
        rp.write_text("{not json", encoding="utf-8")
        with pytest.raises(IntegrityError, match="not valid JSON"):
            check(rp, tmp_path / "README.md")


class TestRealCommittedArtifacts:
    def test_committed_benchmark_json_and_readme_are_consistent(self) -> None:
        results = _REPO_ROOT / "benchmarks" / "mercury_benchmark_results.json"
        readme = _REPO_ROOT / "README.md"
        if not results.is_file():  # pragma: no cover - defensive
            pytest.skip("committed benchmark results not present")
        assert check(results, readme) == []
