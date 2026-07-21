# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bias-mitigation processor — threshold matching + real reweighing.

Covers two defects fixed alongside the intersectional-metrics work:

- ``_apply_threshold_optimization`` compared the raw sensitive-feature
  array against **stringified** group keys, so integer-typed groups
  matched nothing and every prediction silently became 0.0; unseen
  groups were likewise zero-filled instead of falling back to a default
  threshold.
- ``_fit_reweighting`` was a placeholder (every threshold hard-coded to
  0.5, no weights produced).  It now computes Kamiran–Calders reweighing
  weights ``P(g)·P(y)/P(g,y)`` exposed via ``get_sample_weights``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.ml.fairness import BiasmitigationProcessor, MitigationStrategy

# =============================================================================
# Threshold optimization
# =============================================================================


class TestThresholdOptimization:
    """Group-threshold fitting and application."""

    def test_integer_groups_regression(self) -> None:
        """Integer-typed group arrays must not silently zero all output.

        Before the fix, ``fit`` stored ``str(group)`` keys while
        ``transform`` compared the raw integer array against them, so no
        sample ever matched and the zeros-initialised output was returned
        unchanged.
        """
        predictions = np.array([0.9, 0.8, 0.7, 0.2, 0.9, 0.8, 0.7, 0.2])
        labels = np.array([1, 1, 1, 0, 1, 1, 1, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        processor = BiasmitigationProcessor(strategy=MitigationStrategy.THRESHOLD_OPTIMIZATION)
        processor.fit(predictions, labels, groups)
        adjusted = processor.transform(predictions, groups)

        assert np.any(adjusted > 0), "integer-group transform must not zero every prediction"
        # High-scoring samples clear their group threshold in both groups.
        assert adjusted[0] == pytest.approx(1.0)
        assert adjusted[4] == pytest.approx(1.0)

    def test_unseen_group_gets_default_threshold(self) -> None:
        """Samples from a group unseen at fit time keep their signal."""
        predictions = np.array([0.9, 0.1, 0.9, 0.1])
        labels = np.array([1, 0, 1, 0])
        groups = np.array(["a", "a", "a", "a"])

        processor = BiasmitigationProcessor(strategy=MitigationStrategy.THRESHOLD_OPTIMIZATION)
        processor.fit(predictions, labels, groups)

        new_groups = np.array(["a", "a", "b", "b"])
        adjusted = processor.transform(predictions, new_groups)
        # Unseen group "b": default 0.5 threshold — 0.9 selects, 0.1 does not.
        assert adjusted[2] == pytest.approx(1.0)
        assert adjusted[3] == pytest.approx(0.0)

    def test_transform_output_is_binary(self) -> None:
        rng = np.random.default_rng(7)
        predictions = rng.random(50)
        labels = rng.integers(0, 2, 50)
        groups = rng.integers(0, 3, 50)
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.THRESHOLD_OPTIMIZATION)
        processor.fit(predictions, labels, groups)
        adjusted = processor.transform(predictions, groups)
        assert set(np.unique(adjusted)).issubset({0.0, 1.0})

    def test_non_threshold_strategy_passes_predictions_through(self) -> None:
        predictions = np.array([0.9, 0.1])
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        result = processor.transform(predictions, np.array(["a", "b"]))
        assert result is predictions


# =============================================================================
# Kamiran–Calders reweighing
# =============================================================================


class TestReweighing:
    """Real reweighing weights replace the historical placeholder."""

    def test_weights_match_hand_computed_values(self) -> None:
        """Weights follow the closed form ``P(g)·P(y)/P(g,y)`` exactly.

        8 samples, two groups of 4: group ``a`` has 2 positives, group
        ``b`` has 1 — every weight below is asserted against the formula
        evaluated on those raw counts.
        """
        labels = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        processor.fit(np.zeros(8), labels, groups)

        n = 8
        p_a = 4 / n
        p_b = 4 / n
        p_pos = 3 / n
        p_neg = 5 / n
        assert processor.sample_weight_map[("a", 1)] == pytest.approx(p_a * p_pos / (2 / n))
        assert processor.sample_weight_map[("a", 0)] == pytest.approx(p_a * p_neg / (2 / n))
        assert processor.sample_weight_map[("b", 1)] == pytest.approx(p_b * p_pos / (1 / n))
        assert processor.sample_weight_map[("b", 0)] == pytest.approx(p_b * p_neg / (3 / n))

    def test_get_sample_weights_alignment(self) -> None:
        labels = np.array([1, 0, 1, 0])
        groups = np.array(["a", "a", "b", "b"])
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        processor.fit(np.zeros(4), labels, groups)
        weights = processor.get_sample_weights(labels, groups)
        assert weights.shape == (4,)
        assert weights[0] == pytest.approx(processor.sample_weight_map[("a", 1)])
        assert weights[3] == pytest.approx(processor.sample_weight_map[("b", 0)])

    def test_unseen_combination_gets_neutral_weight(self) -> None:
        labels = np.array([1, 1, 0, 0])
        groups = np.array(["a", "a", "a", "a"])
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        processor.fit(np.zeros(4), labels, groups)
        weights = processor.get_sample_weights(np.array([1]), np.array(["never_seen"]))
        assert weights[0] == pytest.approx(1.0)

    def test_get_sample_weights_before_fit_raises(self) -> None:
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        with pytest.raises(RuntimeError, match="requires fit"):
            processor.get_sample_weights(np.array([1]), np.array(["a"]))

    def test_length_mismatch_raises(self) -> None:
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        processor.fit(np.zeros(4), np.array([1, 0, 1, 0]), np.array(["a", "a", "b", "b"]))
        with pytest.raises(ValueError, match="length mismatch"):
            processor.get_sample_weights(np.array([1, 0]), np.array(["a"]))

    def test_empty_fit_raises(self) -> None:
        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        with pytest.raises(ValueError, match="empty"):
            processor.fit(np.zeros(0), np.zeros(0), np.zeros(0))

    @given(seed=st.integers(min_value=0, max_value=2**32 - 1))
    @settings(max_examples=25, deadline=None)
    def test_reweighing_balances_joint_distribution(self, seed: int) -> None:
        """Invariant: weighted joint frequency equals product of marginals.

        This is the defining property of Kamiran–Calders reweighing —
        after weighting, group membership and label are statistically
        independent.
        """
        rng = np.random.default_rng(seed)
        n = 120
        labels = rng.integers(0, 2, n)
        groups = rng.integers(0, 3, n)
        # Ensure every (group, label) combination occurs so weights exist.
        labels[:6] = [0, 1, 0, 1, 0, 1]
        groups[:6] = [0, 0, 1, 1, 2, 2]

        processor = BiasmitigationProcessor(strategy=MitigationStrategy.REWEIGHTING)
        processor.fit(np.zeros(n), labels, groups)
        weights = processor.get_sample_weights(labels, groups)

        total = float(np.sum(weights))
        for group in np.unique(groups):
            group_mask = groups == group
            for label_value in (0, 1):
                mask = group_mask & (labels == label_value)
                weighted_joint = float(np.sum(weights[mask])) / total
                weighted_group = float(np.sum(weights[group_mask])) / total
                weighted_label = float(np.sum(weights[labels == label_value])) / total
                assert weighted_joint == pytest.approx(weighted_group * weighted_label, abs=1e-9)
