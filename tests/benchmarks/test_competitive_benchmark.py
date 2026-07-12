# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the competitive benchmark aggregation.

Focus: ``summarize`` must tolerate the three terminal per-cell states the
per-(dataset, method) wall-clock guard can produce -- a finite score, a recorded
``error``, and a recorded ``deferred`` overrun. A deferred cell carries neither
``error`` nor ``roc_auc``; an earlier revision accessed ``res["roc_auc"]``
unconditionally and crashed the whole run at the final aggregation once any cell
deferred. This pins the fix.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CB = _REPO / "benchmarks" / "competitive_benchmark.py"


def _load_cb() -> Any:
    """Import benchmarks/competitive_benchmark.py by path (it self-configures sys.path)."""
    spec = importlib.util.spec_from_file_location("competitive_benchmark", _CB)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.skip("cannot load competitive_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summarize_handles_deferred_cells_without_crashing() -> None:
    """A deferred cell is recorded (per-method + summary.deferred_cells), not a KeyError."""
    cb = _load_cb()
    method_order = ["mercury_tier", "mercury_fusion", "isolation_forest"]
    per_dataset = [
        {
            "name": "big_highdim",
            "methods": {
                "mercury_tier": {"roc_auc": 0.9, "average_precision": 0.5},
                # deferral: neither "error" nor "roc_auc"
                "mercury_fusion": {"deferred": "exceeded 300s wall budget", "wall_seconds": 300.1},
                "isolation_forest": {"roc_auc": 0.7, "average_precision": 0.4},
            },
        },
        {
            "name": "small",
            "methods": {
                "mercury_tier": {"roc_auc": 0.8, "average_precision": 0.4},
                "mercury_fusion": {"roc_auc": 0.75, "average_precision": 0.35},
                "isolation_forest": {"roc_auc": 0.6, "average_precision": 0.3},
            },
        },
        {
            "name": "errset",
            "methods": {
                "mercury_tier": {"roc_auc": 0.5, "average_precision": 0.2},
                "mercury_fusion": {"roc_auc": 0.5, "average_precision": 0.2},
                "isolation_forest": {"error": "SomeError: boom"},
            },
        },
    ]

    summary = cb.summarize(per_dataset, method_order)  # must not raise

    # The deferred cell is surfaced verbatim in the top-level list.
    assert summary["deferred_cells"] == [
        {"dataset": "big_highdim", "method": "mercury_fusion", "wall_seconds": 300.1}
    ]
    # Fusion scored on two datasets (small, errset); the deferral is excluded from
    # the mean and recorded in the per-method "deferred" list instead.
    fusion = summary["per_method"]["mercury_fusion"]
    assert fusion["n_datasets"] == 2
    assert fusion["deferred"] == [{"dataset": "big_highdim", "wall_seconds": 300.1}]
    # The error cell is recorded, never counted toward a score.
    iforest = summary["per_method"]["isolation_forest"]
    assert iforest["n_datasets"] == 2
    assert [e["dataset"] for e in iforest["errors"]] == ["errset"]
    # Only "small" has a finite AUC for every method, so it alone is rank-complete.
    assert summary["n_datasets_complete_for_ranks"] == 1
    # Head-to-head compares only datasets where BOTH methods scored: fusion vs
    # isolation_forest excludes big_highdim (fusion deferred) and errset (iforest error).
    assert summary["head_to_head"]["mercury_fusion"]["isolation_forest"]["n_compared"] == 1
