# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the out-of-fold (OOF) ECE/Brier rolling-corpus evaluation harness.

The OOF math (folding, metrics, regression check) is exercised with synthetic
data and needs no PQC backend; a guarded end-to-end ``evaluate()`` smoke test
runs the real gate features when AMA/PQC is available.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks import rolling_corpus_eval as rce

if TYPE_CHECKING:
    from pathlib import Path


class TestMetrics:
    def test_perfect_predictions(self) -> None:
        y = np.array([0.0, 0.0, 1.0, 1.0])
        p = np.array([0.01, 0.02, 0.98, 0.99])
        m = rce._metrics(y, p)
        assert m["auroc"] == 1.0
        assert m["fn_rate"] == 0.0
        assert m["fp_rate"] == 0.0
        assert m["brier"] < 0.01

    def test_all_wrong_has_fn_and_fp(self) -> None:
        y = np.array([0.0, 1.0])
        p = np.array([0.9, 0.1])
        m = rce._metrics(y, p)
        assert m["fn_rate"] == 1.0
        assert m["fp_rate"] == 1.0


class TestKFoldOOF:
    def test_separable_data_scores_well(self) -> None:
        # Feature 0 is a clean signal; OOF AUROC should be high and every
        # prediction is out-of-sample.
        n = 60
        y = np.array([float(i % 2) for i in range(n)])
        x = np.column_stack([y * 4.0 - 2.0, np.zeros(n), np.zeros(n)])
        texts = [f"example number {i} about a distinct topic" for i in range(n)]
        m = rce.kfold_oof(x, y, texts, k=5)
        assert m["n"] == n
        assert m["auroc"] > 0.95
        assert 0.0 <= m["ece"] <= 1.0

    def test_all_single_class_yields_nan_not_zero_predictions(self) -> None:
        # Every fold's training complement is single-class -> nothing is scored.
        # The metric must be NaN over 0 rows, NOT a corrupted all-p=0 result.
        n = 20
        y = np.ones(n)  # single class -> every fold complement is single-class
        x = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n)])
        texts = [f"row {i}" for i in range(n)]
        m = rce.kfold_oof(x, y, texts, k=5)
        assert m["n"] == 0
        assert m["fn_rate"] != m["fn_rate"]  # NaN (never counted p=0 as a false negative)


class TestRollingOrigin:
    def test_returns_windowed_metrics(self) -> None:
        n = 60
        y = np.array([float(i % 2) for i in range(n)])
        x = np.column_stack([y * 4.0 - 2.0, np.zeros(n), np.zeros(n)])
        order = np.arange(n)
        m = rce.rolling_origin(x, y, order, windows=5)
        assert m["windows"] >= 1
        assert "ece" in m and "brier" in m


class TestRegressionCheck:
    def _baseline(self, tmp_path: Path, **over: Any) -> Path:
        base = {
            "oof_ece": 0.03,
            "oof_brier": 0.002,
            "oof_auroc": 1.0,
            "adversarial_recall": 0.48,
            **over,
        }
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(base), encoding="utf-8")
        return p

    def test_within_margins_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rce, "BASELINE_PATH", self._baseline(tmp_path))
        measured = {"oof_ece": 0.05, "oof_brier": 0.004, "oof_auroc": 0.99, "adversarial_recall": 0.46}
        assert rce.check(measured) == []

    def test_ece_regression_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rce, "BASELINE_PATH", self._baseline(tmp_path))
        measured = {"oof_ece": 0.20, "oof_brier": 0.002, "oof_auroc": 1.0, "adversarial_recall": 0.48}
        problems = rce.check(measured)
        assert any("oof_ece" in p for p in problems)

    def test_recall_regression_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rce, "BASELINE_PATH", self._baseline(tmp_path))
        measured = {"oof_ece": 0.03, "oof_brier": 0.002, "oof_auroc": 1.0, "adversarial_recall": 0.20}
        problems = rce.check(measured)
        assert any("adversarial_recall" in p for p in problems)

    def test_missing_baseline_reports(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rce, "BASELINE_PATH", tmp_path / "nope.json")
        problems = rce.check({"oof_ece": 0.03})
        assert problems and "missing baseline" in problems[0]

    def test_nan_metric_fails_closed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A NaN metric (missing adversarial slice / undefined AUROC) must be a
        # regression, not a silent pass.
        monkeypatch.setattr(rce, "BASELINE_PATH", self._baseline(tmp_path))
        measured = {
            "oof_ece": 0.03,
            "oof_brier": 0.002,
            "oof_auroc": 1.0,
            "adversarial_recall": float("nan"),
        }
        problems = rce.check(measured)
        assert any("adversarial_recall" in p and "NaN" in p for p in problems)


class TestEvaluateEndToEnd:
    def test_real_evaluate_smoke(self) -> None:
        pytest.importorskip(
            "omni_mercury_engine.cognitive.ethical_bounding",
            reason="requires the AMA/PQC backend for real gate features",
        )
        metrics = rce.evaluate()
        assert metrics["n_cases"] >= 300
        assert 0.0 <= metrics["oof_ece"] <= 1.0
        assert 0.0 <= metrics["oof_brier"] <= 1.0
        assert metrics["adversarial_n"] >= 1
