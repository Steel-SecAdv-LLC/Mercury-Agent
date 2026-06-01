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


def _fraction(
    frac: float, d_auc: float, d_fpr: float, agree: int, n: int = 3
) -> dict[str, float | int | dict[str, list[float]]]:
    return {
        "fraction": frac,
        "delta_auc_mean": d_auc,
        "delta_fpr_mean": d_fpr,
        "seeds_auc_better": agree,
        "n_seeds": n,
        "neural": {"aucs": [0.70] * n},
        "symbolic": {"aucs": [0.70 + d_auc] * n},
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

    def test_false_positive_gate_passes_with_flat_auc(self) -> None:
        # FP clearly down and full-data AUC not regressing -> the FP gate carries.
        results = [{"dataset": "d", "fractions": [_fraction(1.0, 0.0, 0.05, 2)]}]
        verdict = derive_verdict(results)
        assert verdict["gate_false_positives_down"] is True
        assert verdict["passed"] is True

    def test_false_positive_with_auc_regression_quarantines(self) -> None:
        # FP down but full-data AUC regresses beyond the noise floor: a constant
        # weight must not be kept on an FP gain bought with an AUC regression.
        results = [{"dataset": "d", "fractions": [_fraction(1.0, -0.01, 0.05, 2)]}]
        verdict = derive_verdict(results)
        assert verdict["gate_false_positives_down"] is False
        assert verdict["passed"] is False
        assert "QUARANTINE" in verdict["verdict"]

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


from benchmarks.neurosymbolic_ablation import derive_adaptive_verdict


def _afraction(
    frac: float, d_auc: float, d_fpr: float, agree: int, n: int = 3
) -> dict[str, float | int | dict[str, list[float]]]:
    """Fraction dict carrying the adaptive arm (vs the same neural baseline)."""
    return {
        "fraction": frac,
        "delta_auc_adaptive_mean": d_auc,
        "delta_fpr_adaptive_mean": d_fpr,
        "seeds_auc_adaptive_better": agree,
        "n_seeds": n,
        "neural": {"aucs": [0.80] * n},
        "adaptive": {"aucs": [0.80 + d_auc] * n},
    }


class TestDeriveAdaptiveVerdict:
    """The adaptive arm must earn default-on by dominance, honestly."""

    def test_keep_on_low_lift_without_full_regression(self) -> None:
        results = [
            {
                "dataset": "d",
                "fractions": [
                    _afraction(0.1, 0.004, 0.0, 3),
                    _afraction(0.25, 0.003, 0.0, 2),
                    _afraction(1.0, 0.0, 0.0, 2),  # schedule ~ neural at full data
                ],
            }
        ]
        v = derive_adaptive_verdict(results)
        assert v["gate_no_full_data_regression"] is True
        assert v["gate_low_data_lift"] is True
        assert v["passed"] is True
        assert "KEEP" in v["verdict"]

    def test_quarantine_on_full_data_regression(self) -> None:
        results = [
            {
                "dataset": "d",
                "fractions": [
                    _afraction(0.1, 0.004, 0.0, 3),
                    _afraction(1.0, -0.01, -0.02, 0),  # clear full-data regression
                ],
            }
        ]
        v = derive_adaptive_verdict(results)
        assert v["gate_no_full_data_regression"] is False
        assert v["passed"] is False
        assert "QUARANTINE" in v["verdict"]

    def test_quarantine_when_no_low_data_lift(self) -> None:
        results = [
            {
                "dataset": "d",
                "fractions": [
                    _afraction(0.1, -0.003, 0.0, 1),  # adaptive does not help when scarce
                    _afraction(1.0, 0.0, 0.0, 2),
                ],
            }
        ]
        v = derive_adaptive_verdict(results)
        assert v["gate_low_data_lift"] is False
        assert v["passed"] is False

    def test_lift_must_be_seed_agreed(self) -> None:
        # Positive mean low-data lift but only a minority of seeds agree -> no KEEP.
        results = [
            {
                "dataset": "d",
                "fractions": [
                    _afraction(0.1, 0.004, 0.0, 0),
                    _afraction(0.25, 0.004, 0.0, 1),
                    _afraction(1.0, 0.0, 0.0, 2),
                ],
            }
        ]
        v = derive_adaptive_verdict(results)
        assert v["gate_low_data_lift"] is False
        assert v["passed"] is False
