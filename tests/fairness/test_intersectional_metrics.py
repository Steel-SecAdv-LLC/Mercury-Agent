# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Intersectional (joint-subgroup) fairness metrics — ROADMAP row 6.

Covers:
- ``build_intersectional_groups``: joint-label construction from mapping,
  2-D array, and 1-D array inputs, plus every input-validation raise path
- ``compute_intersectional_parity``: the Simpson's-paradox case (marginal
  parity holds on every feature while a joint cell is disadvantaged),
  small-cell exclusion, worst-group identification, and the
  ``insufficient_data`` fail-safe
- ``compute_intersectional_equalized_odds``: joint TPR/FPR gaps, cells
  without positives/negatives excluded rather than fabricated as 0.0
- ``FairnessAuditor.audit``: multi-feature mapping input (the engine's
  ``_audit_fairness`` shape), per-feature marginal keys, intersectional
  keys, and unchanged single-feature report shape
- Property-based invariants (bounded scores, permutation invariance)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.ml.fairness import (
    BiasAuditConfig,
    FairnessAuditor,
    FairnessMetric,
    build_intersectional_groups,
)

# =============================================================================
# build_intersectional_groups
# =============================================================================


class TestBuildIntersectionalGroups:
    """Joint-label construction across the three accepted input shapes."""

    def test_mapping_input_builds_named_joint_labels(self) -> None:
        labels, names = build_intersectional_groups(
            {
                "race": np.array(["A", "B", "A"]),
                "gender": np.array(["F", "F", "M"]),
            }
        )
        assert names == ["race", "gender"]
        assert labels.tolist() == ["race=A|gender=F", "race=B|gender=F", "race=A|gender=M"]

    def test_2d_array_with_feature_names(self) -> None:
        array = np.array([["A", "F"], ["B", "M"]])
        labels, names = build_intersectional_groups(array, feature_names=["race", "gender"])
        assert names == ["race", "gender"]
        assert labels.tolist() == ["race=A|gender=F", "race=B|gender=M"]

    def test_2d_array_default_names(self) -> None:
        labels, names = build_intersectional_groups(np.array([[0, 1], [1, 0]]))
        assert names == ["feature_0", "feature_1"]
        assert labels.tolist() == ["feature_0=0|feature_1=1", "feature_0=1|feature_1=0"]

    def test_1d_array_is_single_feature(self) -> None:
        labels, names = build_intersectional_groups(np.array([1, 2, 1]))
        assert names == ["feature_0"]
        assert labels.tolist() == ["feature_0=1", "feature_0=2", "feature_0=1"]

    def test_integer_features_stringified_consistently(self) -> None:
        labels, _ = build_intersectional_groups({"a": np.array([0, 1]), "b": np.array([2, 3])})
        assert labels.tolist() == ["a=0|b=2", "a=1|b=3"]

    def test_empty_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_intersectional_groups({})

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            build_intersectional_groups({"a": np.array([1, 2, 3]), "b": np.array([1, 2])})

    def test_wrong_feature_name_count_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_names"):
            build_intersectional_groups(np.array([[1, 2], [3, 4]]), feature_names=["only_one"])

    def test_3d_array_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            build_intersectional_groups(np.zeros((2, 2, 2)))

    def test_zero_column_array_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one feature column"):
            build_intersectional_groups(np.zeros((3, 0)))


# =============================================================================
# compute_intersectional_parity
# =============================================================================


def _simpsons_paradox_dataset() -> tuple[np.ndarray[Any, Any], dict[str, np.ndarray[Any, Any]]]:
    """Dataset where every marginal is fair but the joint cells are not.

    Four equal-size joint cells of 20 samples.  Selection rates:
    ``(A,F)=0.8, (A,M)=0.2, (B,F)=0.2, (B,M)=0.8`` — both marginal rates
    are exactly 0.5 for every group of each feature, yet each joint cell
    deviates from the overall 0.5 rate by 0.3.
    """
    cell = 20
    rates = {("A", "F"): 0.8, ("A", "M"): 0.2, ("B", "F"): 0.2, ("B", "M"): 0.8}
    predictions: list[float] = []
    race: list[str] = []
    gender: list[str] = []
    for (r, g), rate in rates.items():
        n_pos = int(cell * rate)
        predictions.extend([1.0] * n_pos + [0.0] * (cell - n_pos))
        race.extend([r] * cell)
        gender.extend([g] * cell)
    return np.array(predictions), {"race": np.array(race), "gender": np.array(gender)}


class TestIntersectionalParity:
    """Joint-subgroup demographic parity."""

    def test_simpsons_paradox_detected_only_intersectionally(self) -> None:
        """Marginal parity is perfect while joint disparity is 0.3."""
        predictions, features = _simpsons_paradox_dataset()
        auditor = FairnessAuditor()

        for column in features.values():
            marginal = auditor.compute_demographic_parity(predictions, column)
            assert marginal["max_disparity"] == pytest.approx(0.0)

        joint = auditor.compute_intersectional_parity(predictions, features)
        assert joint["max_disparity"] == pytest.approx(0.3)
        assert joint["parity_score"] == pytest.approx(0.7)
        assert not joint["insufficient_data"]

    def test_worst_group_identified(self) -> None:
        predictions, features = _simpsons_paradox_dataset()
        joint = FairnessAuditor().compute_intersectional_parity(predictions, features)
        # All four cells deviate by exactly 0.3; the worst group must be
        # one of them and its reported rate must match its construction.
        assert joint["worst_group"] in joint["group_rates"]
        worst_rate = joint["group_rates"][joint["worst_group"]]
        assert worst_rate in (pytest.approx(0.8), pytest.approx(0.2))

    def test_small_cells_excluded_and_reported(self) -> None:
        """A 2-sample cell with an extreme rate must not drive the gap."""
        predictions = np.array([1.0] * 10 + [0.0] * 10 + [1.0, 1.0])
        features = {
            "a": np.array(["x"] * 10 + ["y"] * 10 + ["z"] * 2),
            "b": np.array(["u"] * 22),
        }
        auditor = FairnessAuditor(BiasAuditConfig(intersectional_min_group_size=5))
        joint = auditor.compute_intersectional_parity(predictions, features)
        assert joint["small_groups"] == {"a=z|b=u": 2}
        assert "a=z|b=u" not in joint["group_rates"]
        assert set(joint["group_sizes"].values()) == {10}

    def test_all_cells_small_flags_insufficient_data(self) -> None:
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        features = {"a": np.array(["w", "x", "y", "z"]), "b": np.array(["1", "2", "3", "4"])}
        joint = FairnessAuditor().compute_intersectional_parity(predictions, features)
        assert joint["insufficient_data"] is True
        assert joint["parity_score"] == pytest.approx(1.0)
        assert joint["group_rates"] == {}
        assert len(joint["small_groups"]) == 4

    def test_min_group_size_override(self) -> None:
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        features = {"a": np.array(["x", "x", "y", "y"]), "b": np.array(["u", "u", "u", "u"])}
        joint = FairnessAuditor().compute_intersectional_parity(
            predictions, features, min_group_size=2
        )
        assert not joint["insufficient_data"]
        assert len(joint["group_rates"]) == 2

    def test_prediction_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            FairnessAuditor().compute_intersectional_parity(
                np.array([1.0, 0.0]),
                {"a": np.array(["x", "y", "z"]), "b": np.array(["u", "v", "w"])},
            )


# =============================================================================
# compute_intersectional_equalized_odds
# =============================================================================


class TestIntersectionalEqualizedOdds:
    """Joint-subgroup TPR/FPR gap measurement."""

    def test_joint_tpr_gap_measured(self) -> None:
        # Two joint cells of 20: cell 1 perfect recall, cell 2 half recall.
        predictions = np.array(([1.0] * 10 + [0.0] * 10) + ([1.0] * 5 + [0.0] * 15))
        labels = np.array(([1] * 10 + [0] * 10) + ([1] * 10 + [0] * 10))
        features = {
            "a": np.array(["x"] * 20 + ["y"] * 20),
            "b": np.array(["u"] * 40),
        }
        result = FairnessAuditor().compute_intersectional_equalized_odds(
            predictions, labels, features
        )
        assert result["group_tpr"]["a=x|b=u"] == pytest.approx(1.0)
        assert result["group_tpr"]["a=y|b=u"] == pytest.approx(0.5)
        assert result["tpr_difference"] == pytest.approx(0.5)
        assert result["worst_tpr_group"] == "a=y|b=u"
        assert result["equalized_odds_score"] == pytest.approx(0.5)

    def test_cell_without_positives_excluded_from_tpr(self) -> None:
        """No fabricated 0.0 TPR for a cell that has no positive labels."""
        predictions = np.array([1.0] * 10 + [0.0] * 10)
        labels = np.array([1] * 10 + [0] * 10)  # second cell all-negative
        features = {
            "a": np.array(["x"] * 10 + ["y"] * 10),
            "b": np.array(["u"] * 20),
        }
        result = FairnessAuditor().compute_intersectional_equalized_odds(
            predictions, labels, features
        )
        assert "a=y|b=u" not in result["group_tpr"]
        assert result["tpr_difference"] == pytest.approx(0.0)
        assert result["group_fpr"]["a=y|b=u"] == pytest.approx(0.0)

    def test_insufficient_data_when_fewer_than_two_cells(self) -> None:
        predictions = np.array([1.0, 0.0, 1.0])
        labels = np.array([1, 0, 1])
        features = {"a": np.array(["x", "y", "z"]), "b": np.array(["1", "2", "3"])}
        result = FairnessAuditor().compute_intersectional_equalized_odds(
            predictions, labels, features
        )
        assert result["insufficient_data"] is True
        assert result["equalized_odds_score"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            FairnessAuditor().compute_intersectional_equalized_odds(
                np.array([1.0, 0.0]),
                np.array([1]),
                {"a": np.array(["x", "y"]), "b": np.array(["u", "v"])},
            )


# =============================================================================
# FairnessAuditor.audit — multi-feature integration
# =============================================================================


class TestAuditIntersectional:
    """audit() with the engine's mapping shape and with 2-D arrays."""

    def test_mapping_input_produces_marginal_and_intersectional_keys(self) -> None:
        predictions, features = _simpsons_paradox_dataset()
        report = FairnessAuditor().audit(predictions=predictions, sensitive_features=features)

        assert "demographic_parity:race" in report.metric_scores
        assert "demographic_parity:gender" in report.metric_scores
        assert "intersectional_parity" in report.metric_scores
        assert report.metric_scores["demographic_parity:race"] == pytest.approx(1.0)
        assert report.metric_scores["intersectional_parity"] == pytest.approx(0.7)

    def test_simpsons_paradox_flagged_as_violation(self) -> None:
        """The audit must fail a marginally-fair but jointly-unfair model."""
        predictions, features = _simpsons_paradox_dataset()
        report = FairnessAuditor().audit(predictions=predictions, sensitive_features=features)
        assert not report.is_fair
        assert any("Intersectional parity violation" in v for v in report.violations)
        assert report.details["intersectional_parity"]["worst_group"] is not None

    def test_intersectional_equalized_odds_in_audit(self) -> None:
        predictions = np.array(([1.0] * 10 + [0.0] * 10) + ([1.0] * 5 + [0.0] * 15))
        labels = np.array(([1] * 10 + [0] * 10) + ([1] * 10 + [0] * 10))
        features = {
            "a": np.array(["x"] * 20 + ["y"] * 20),
            "b": np.array(["u"] * 40),
        }
        report = FairnessAuditor().audit(
            predictions=predictions, labels=labels, sensitive_features=features
        )
        assert "intersectional_equalized_odds" in report.metric_scores
        assert any("Intersectional equalized odds violation" in v for v in report.violations)

    def test_2d_array_with_feature_names(self) -> None:
        predictions, features = _simpsons_paradox_dataset()
        matrix = np.column_stack([features["race"], features["gender"]])
        report = FairnessAuditor().audit(
            predictions=predictions,
            sensitive_features=matrix,
            feature_names=["race", "gender"],
        )
        assert "demographic_parity:race" in report.metric_scores
        assert report.metric_scores["intersectional_parity"] == pytest.approx(0.7)

    def test_single_feature_report_shape_unchanged(self) -> None:
        """Historical 1-D behaviour: un-suffixed keys, no intersectional."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0] * 5)
        sensitive = np.array(["a", "a", "b", "b"] * 5)
        report = FairnessAuditor().audit(predictions=predictions, sensitive_features=sensitive)
        assert "demographic_parity" in report.metric_scores
        assert "intersectional_parity" not in report.metric_scores
        assert all(":" not in key for key in report.metric_scores)
        assert isinstance(report.details["n_groups"], int)

    def test_single_entry_mapping_keeps_historical_keys(self) -> None:
        """The engine may pass a one-feature dict; keys stay un-suffixed."""
        predictions = np.array([1.0, 0.0] * 10)
        report = FairnessAuditor().audit(
            predictions=predictions,
            sensitive_features={"region": np.array(["n", "s"] * 10)},
        )
        assert "demographic_parity" in report.metric_scores
        assert "intersectional_parity" not in report.metric_scores

    def test_empty_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            FairnessAuditor().audit(predictions=np.array([1.0]), sensitive_features={})

    def test_insufficient_joint_data_recommends_not_violates(self) -> None:
        """Sparse joint cells must not produce a fabricated violation."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        features = {"a": np.array(["w", "x", "y", "z"]), "b": np.array(["1", "2", "3", "4"])}
        config = BiasAuditConfig(
            metrics=[FairnessMetric.INTERSECTIONAL_PARITY],
        )
        report = FairnessAuditor(config).audit(predictions=predictions, sensitive_features=features)
        assert not any("Intersectional" in v for v in report.violations)
        assert any("indeterminate" in r for r in report.recommendations)
        assert report.details["intersectional_parity"]["insufficient_data"] is True


# =============================================================================
# Property-based invariants
# =============================================================================


class TestIntersectionalProperties:
    """Adversarial/property invariants over random inputs."""

    @given(
        data=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=1),
                st.integers(min_value=0, max_value=2),
                st.integers(min_value=0, max_value=2),
            ),
            min_size=1,
            max_size=200,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_parity_score_always_bounded(self, data: list[tuple[int, int, int]]) -> None:
        predictions = np.array([row[0] for row in data], dtype=float)
        features = {
            "f1": np.array([row[1] for row in data]),
            "f2": np.array([row[2] for row in data]),
        }
        result = FairnessAuditor().compute_intersectional_parity(
            predictions, features, min_group_size=1
        )
        assert 0.0 <= result["parity_score"] <= 1.0
        assert 0.0 <= result["max_disparity"] <= 1.0

    @given(seed=st.integers(min_value=0, max_value=2**32 - 1))
    @settings(max_examples=25, deadline=None)
    def test_permutation_invariance(self, seed: int) -> None:
        """Reordering samples must not change any intersectional result."""
        rng = np.random.default_rng(seed)
        n = 80
        predictions = rng.integers(0, 2, n).astype(float)
        features = {
            "f1": rng.integers(0, 2, n),
            "f2": rng.integers(0, 2, n),
        }
        base = FairnessAuditor().compute_intersectional_parity(
            predictions, features, min_group_size=1
        )
        perm = rng.permutation(n)
        shuffled = FairnessAuditor().compute_intersectional_parity(
            predictions[perm],
            {name: column[perm] for name, column in features.items()},
            min_group_size=1,
        )
        assert base["group_rates"] == shuffled["group_rates"]
        assert base["max_disparity"] == pytest.approx(shuffled["max_disparity"])
