# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the root-cause graph localisation detector."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.rca import RootCauseGraphDetector


def _matrix_with_node_fault(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (200, 6))
    x[120:140, 3] += 8.0  # fault localised on node 3
    return x.astype(np.float64)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = RootCauseGraphDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=(200, 6)))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _matrix_with_node_fault()
        scores = np.asarray(RootCauseGraphDetector().fit(x).detect(x)["scores"])
        assert scores.shape == (x.shape[0],)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _matrix_with_node_fault()
        feats = RootCauseGraphDetector().fit(x).extract_features(x)
        assert feats.shape == (x.shape[0], 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            RootCauseGraphDetector(damping=1.5)
        with pytest.raises(ValueError):
            RootCauseGraphDetector(walk_iters=0)

    def test_deterministic(self) -> None:
        x = _matrix_with_node_fault(7)
        a = RootCauseGraphDetector().fit(x).detect(x)["scores"]
        b = RootCauseGraphDetector().fit(x).detect(x)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_fault_flagged_and_ranked(self) -> None:
        train = np.random.default_rng(1).normal(size=(400, 6))
        x = _matrix_with_node_fault(2)
        det = RootCauseGraphDetector().fit(train)
        scores = np.asarray(det.detect(x)["scores"])
        assert float(scores[120:140].max()) > 0.5
        # Localise a faulted row (row 130): node 3 should rank first.
        ranked = det.rank_root_causes(x[:131])
        assert ranked[0][0] == 3

    def test_low_false_positive_rate(self) -> None:
        train = np.random.default_rng(3).normal(size=(400, 6))
        test = np.random.default_rng(4).normal(size=(400, 6))
        scores = np.asarray(
            RootCauseGraphDetector(calibration_quantile=0.98).fit(train).detect(test)["scores"]
        )
        assert (scores > 0.5).mean() < 0.08


class TestDegenerateInput:
    def test_empty_input_no_crash(self) -> None:
        # Regression: empty input coerced to (n, 0) / (0, n) made _peak_residuals'
        # resid.max(axis=1) raise ValueError on the size-0 node axis.
        det = RootCauseGraphDetector()
        for empty in (np.array([]), np.zeros((0, 4)), np.zeros((3, 0))):
            scores = np.asarray(det.detect(empty)["scores"], dtype=float)
            assert np.all(np.isfinite(scores))
        RootCauseGraphDetector().fit(np.array([]))  # empty fit must not crash

    def test_rank_before_fit_inferred_graph_raises_clearly(self) -> None:
        # Regression: rank_root_causes before fit in inferred-graph mode tripped a
        # bare assert in _walk; it must raise a clear RuntimeError instead.
        with pytest.raises(RuntimeError, match="must be fit"):
            RootCauseGraphDetector().rank_root_causes(np.random.default_rng(0).normal(size=(5, 3)))


class TestDeclaredModality:
    """``rca`` is a ``matrix_2d`` detector: ``(n_rows, n_nodes)``, one verdict per row."""

    def test_the_manifest_declares_the_modality_the_class_can_consume(self) -> None:
        """It was declared ``series_1d``.

        ``DetectorRegistry.extract_all_features`` filters on that declaration
        precisely to keep a detector from being driven with data it cannot
        read, so the declaration being wrong defeats the filter.
        """
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        entry = next(e for e in DETECTOR_MANIFEST if e.name == "rca")
        assert entry.modality == "matrix_2d"

    def test_node_spreads_are_learned_across_rows(self) -> None:
        """A 1-D series reshapes to one row, which collapses every node's spread.

        ``_to_2d_f64`` reshapes a 1-D input to ``(1, n)``, so a 200-sample
        series became one row of 200 nodes: ``rows.std(axis=0)`` over a single
        row is 0, every ``_node_std`` sat at its 1e-9 floor, the squash scale
        came out 1.4e-9, and every later row scored 1.0.  Measured on a
        stationary series: 100% of a held-out half flagged, against the 2% its
        ``calibration_quantile`` targets.
        """
        rows = np.random.default_rng(7).normal(size=(400, 8))
        detector = RootCauseGraphDetector().fit(rows)

        assert detector._node_std is not None
        assert detector._node_std.shape == (8,)
        assert detector._node_std.min() > 0.5
        assert detector._scale > 1.0

    def test_a_stationary_held_out_half_is_not_flagged_wholesale(self) -> None:
        rows = np.random.default_rng(7).normal(size=(800, 8))
        detector = RootCauseGraphDetector().fit(rows[:400])

        flagged = np.asarray(detector.detect(rows[400:])["is_anomaly"], dtype=bool)

        assert flagged.mean() <= 0.10, f"{flagged.mean():.4f} against a nominal 0.02"
