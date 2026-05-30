"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Network-free unit tests for the neuro-symbolic ablation harness math
(``benchmarks.neurosymbolic_ablation``): the false-positive-rate-at-recall
metric and the transparent verdict logic. The full paired run requires real
ADBench downloads and is exercised separately/manually.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.neurosymbolic_ablation import derive_verdict, fpr_at_recall


class TestFprAtRecall:
    """fpr_at_recall must report a correct, honest operating point."""

    def test_perfect_separation_has_zero_fpr(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert fpr_at_recall(y, score, 0.9) == pytest.approx(0.0)

    def test_random_overlap_has_positive_fpr(self) -> None:
        # To recall all 2 positives we must cross a negative -> FPR > 0.
        y = np.array([0, 1, 0, 1, 0])
        score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        assert fpr_at_recall(y, score, 1.0) > 0.0

    def test_single_class_returns_nan(self) -> None:
        y = np.array([0, 0, 0])
        score = np.array([0.1, 0.2, 0.3])
        assert np.isnan(fpr_at_recall(y, score))

    def test_lower_recall_target_needs_no_more_false_positives(self) -> None:
        y = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([0.95, 0.9, 0.6, 0.55, 0.4, 0.2])
        fpr_low = fpr_at_recall(y, score, 0.5)
        fpr_high = fpr_at_recall(y, score, 1.0)
        assert fpr_low <= fpr_high


def _fraction(frac: float, d_auc: float, d_fpr: float, agree: int, n: int = 3) -> dict:
    return {
        "fraction": frac,
        "delta_auc_mean": d_auc,
        "delta_fpr_mean": d_fpr,
        "seeds_auc_better": agree,
        "n_seeds": n,
    }


class TestDeriveVerdict:
    """The verdict must follow the measured deltas, not a hard-coded pass."""

    def test_auc_gate_passes_on_consistent_improvement(self) -> None:
        results = [{"dataset": "d", "fractions": [_fraction(1.0, 0.01, 0.0, 3)]}]
        verdict = derive_verdict(results)
        assert verdict["gate_auc_up"] is True
        assert verdict["passed"] is True

    def test_no_improvement_quarantines(self) -> None:
        results = [{"dataset": "d", "fractions": [_fraction(1.0, -0.01, -0.02, 0)]}]
        verdict = derive_verdict(results)
        assert verdict["passed"] is False
        assert "QUARANTINE" in verdict["verdict"]

    def test_false_positive_gate_independent_of_auc(self) -> None:
        # AUC flat but false positives clearly down, with seeds agreeing.
        results = [{"dataset": "d", "fractions": [_fraction(1.0, 0.0, 0.05, 2)]}]
        verdict = derive_verdict(results)
        assert verdict["gate_false_positives_down"] is True
        assert verdict["passed"] is True

    def test_sample_efficiency_gate_rewards_low_data_gain(self) -> None:
        results = [
            {
                "dataset": "d",
                "fractions": [
                    _fraction(0.1, 0.02, 0.0, 3),
                    _fraction(0.25, 0.015, 0.0, 3),
                    _fraction(1.0, 0.0, 0.0, 1),
                ],
            }
        ]
        verdict = derive_verdict(results)
        assert verdict["gate_sample_efficiency_up"] is True
        assert verdict["passed"] is True
