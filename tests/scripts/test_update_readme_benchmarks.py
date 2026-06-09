# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for ``scripts/update_readme_benchmarks.py``.

The README updater historically read ``data["commit"]`` and
``data["timestamp"]``, but ``benchmarks/mercury_benchmark.py`` writes
those provenance fields under ``data["metadata"]`` (``git_commit`` and
``timestamp``).  These tests pin the contract that the script reads
the canonical nested location first while still tolerating older
flat-layout result files.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "update_readme_benchmarks.py"


@pytest.fixture(scope="module")
def update_readme_benchmarks() -> object:
    """Import the script as a module so we can call ``_summary`` directly."""
    spec = importlib.util.spec_from_file_location(
        "_update_readme_benchmarks_under_test", _SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSummaryNestedMetadata:
    """``_summary`` must read provenance from ``data["metadata"]`` first."""

    def test_reads_nested_metadata_from_canonical_schema(
        self, update_readme_benchmarks: object
    ) -> None:
        """Canonical schema written by ``mercury_benchmark.py``.

        ``benchmarks/mercury_benchmark.py`` writes::

            {
              "metadata": {"git_commit": "abc123", "timestamp": "2026-..."},
              "summary":  {"mean_auc": 0.81, ...},
            }

        The script must surface the nested values, not silently fall
        back to ``data["commit"]`` / ``data["timestamp"]`` (which are
        absent in the canonical schema).
        """
        data = {
            "metadata": {
                "git_commit": "deadbeefcafe",
                "timestamp": "2026-05-04T12:00:00+00:00",
                "python_version": "3.12.8",
            },
            "summary": {
                "mean_auc": 0.812,
                "median_auc": 0.81,
                "mean_oracle_f1": 0.59,
                "successful": 7,
                "total": 7,
            },
        }

        summary = update_readme_benchmarks._summary(data)  # type: ignore[attr-defined]

        assert summary["commit"] == "deadbeefcafe"
        assert summary["timestamp"] == "2026-05-04T12:00:00+00:00"
        assert summary["mean_auc"] == 0.812
        assert summary["successful"] == 7

    def test_falls_back_to_flat_top_level(self, update_readme_benchmarks: object) -> None:
        """Older fixtures stored provenance at the top level — still read it."""
        data = {
            "commit": "legacy-flat-sha",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "summary": {
                "mean_auc": 0.70,
                "mean_oracle_f1": 0.50,
                "successful": 3,
                "total": 3,
            },
        }

        summary = update_readme_benchmarks._summary(data)  # type: ignore[attr-defined]

        assert summary["commit"] == "legacy-flat-sha"
        assert summary["timestamp"] == "2024-01-01T00:00:00+00:00"

    def test_metadata_takes_precedence_over_flat_keys(
        self, update_readme_benchmarks: object
    ) -> None:
        """Mixed-schema files prefer the canonical nested location."""
        data = {
            "metadata": {
                "git_commit": "nested-wins",
                "timestamp": "2026-05-04T12:00:00+00:00",
            },
            "commit": "flat-loses",
            "timestamp": "1999-01-01T00:00:00+00:00",
            "summary": {"mean_auc": 0.8, "mean_oracle_f1": 0.5, "successful": 1},
        }

        summary = update_readme_benchmarks._summary(data)  # type: ignore[attr-defined]

        assert summary["commit"] == "nested-wins"
        assert summary["timestamp"] == "2026-05-04T12:00:00+00:00"

    def test_missing_provenance_returns_none(self, update_readme_benchmarks: object) -> None:
        """When neither nested nor flat fields are present, return None."""
        data = {
            "summary": {"mean_auc": 0.5, "mean_oracle_f1": 0.4, "successful": 1},
        }

        summary = update_readme_benchmarks._summary(data)  # type: ignore[attr-defined]

        assert summary["commit"] is None
        assert summary["timestamp"] is None


class TestTotalDatasetResolution:
    """``_summary`` must surface the *attempted* dataset count, not collapse it.

    ``benchmarks/mercury_benchmark.py`` writes the attempted count under
    ``summary["total_datasets"]`` (next to ``successful`` / ``failed``).  If
    that key is not in the resolution chain the ratio silently degrades to
    ``successful / successful`` — the exact ``65 / 65`` bug this pins against
    (the committed results carry ``total_datasets=75``, ``successful=65``).
    """

    def test_reads_total_datasets_key(self, update_readme_benchmarks: Any) -> None:
        """The canonical ``total_datasets`` key resolves to the attempted total."""
        data = {
            "summary": {
                "mean_auc": 0.8466,
                "mean_oracle_f1": 0.6428,
                "successful": 65,
                "failed": 10,
                "total_datasets": 75,
            },
        }

        summary = update_readme_benchmarks._summary(data)

        assert summary["successful"] == 65
        assert summary["total"] == 75, "must not collapse to successful/successful (65/65)"

    def test_explicit_total_wins_over_total_datasets(self, update_readme_benchmarks: Any) -> None:
        """An explicit ``total`` (older/external fixtures) still takes precedence."""
        data = {
            "summary": {
                "mean_auc": 0.8,
                "mean_oracle_f1": 0.5,
                "successful": 7,
                "total": 7,
                "total_datasets": 999,
            },
        }

        summary = update_readme_benchmarks._summary(data)

        assert summary["total"] == 7

    def test_falls_back_to_successful_when_no_total(self, update_readme_benchmarks: Any) -> None:
        """With no total/total_datasets/n_datasets, fall back to ``successful``."""
        data = {
            "summary": {"mean_auc": 0.5, "mean_oracle_f1": 0.4, "successful": 3},
        }

        summary = update_readme_benchmarks._summary(data)

        assert summary["total"] == 3


class TestCommittedReadmeMatchesCommittedJSON:
    """Drift gate: the committed README block must match the committed results JSON.

    ``benchmark.yml`` regenerates the ``<!-- BENCHMARK:START -->`` block on every
    push to ``main`` via ``render_block``, but nothing else stops a hand-edit (or
    a stale commit) from leaving the README headline out of sync with
    ``benchmarks/mercury_benchmark_results.json``.  This renders the canonical
    figures through the *same* generator helpers (``_summary`` / ``_fmt``) and
    asserts they appear in the committed block, so CI fails on benchmark-number
    drift instead of shipping a wrong headline.
    """

    def test_block_carries_canonical_committed_figures(
        self, update_readme_benchmarks: Any
    ) -> None:
        """The committed README block must carry the committed JSON's figures."""
        results = json.loads(
            (_REPO_ROOT / "benchmarks" / "mercury_benchmark_results.json").read_text(
                encoding="utf-8"
            )
        )
        summary = update_readme_benchmarks._summary(results)

        readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
        start = readme.index(update_readme_benchmarks.START_MARKER)
        end = readme.index(update_readme_benchmarks.END_MARKER)
        block = readme[start:end]

        fmt = update_readme_benchmarks._fmt
        for key in ("mean_auc", "median_auc", "mean_oracle_f1"):
            rendered = fmt(summary[key], kind="float")
            assert rendered in block, (
                f"README benchmark block is stale: missing {key}={rendered} from "
                "benchmarks/mercury_benchmark_results.json. Regenerate via "
                "`python scripts/update_readme_benchmarks.py`."
            )

        ratio = (
            f"{fmt(summary['successful'], kind='int')} / "
            f"{fmt(summary['total'], kind='int')}"
        )
        assert ratio in block, (
            f"README benchmark block is stale: missing dataset ratio {ratio}. "
            "Regenerate via `python scripts/update_readme_benchmarks.py`."
        )
