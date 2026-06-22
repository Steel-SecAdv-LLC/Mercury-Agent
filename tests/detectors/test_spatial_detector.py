# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for SpatialAnomalyDetector."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import pytest
import torch

from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector, _NativeLOF


class TestSpatialAnomalyDetector:
    """Tests for SpatialAnomalyDetector."""

    @pytest.fixture
    def spatial_data(self, deterministic_rng: Any) -> Any:
        """Generate sample spatial data (2D coordinates)."""
        return deterministic_rng.randn(100, 2) * 10

    @pytest.fixture
    def spatial_data_3d(self, deterministic_rng: Any) -> Any:
        """Generate sample 3D spatial data."""
        return deterministic_rng.randn(100, 3) * 10

    def test_initialization_default(self) -> None:
        """Test initialization with default config."""
        detector = SpatialAnomalyDetector()
        assert detector.n_neighbors == 20
        assert detector.contamination == 0.1
        assert not detector._is_fitted

    def test_initialization_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = {"n_neighbors": 10, "contamination": 0.05}
        detector = SpatialAnomalyDetector(config=config)
        assert detector.n_neighbors == 10
        assert detector.contamination == 0.05

    def test_fit_numpy_array(self, spatial_data: Any) -> None:
        """Test fitting with numpy array."""
        detector = SpatialAnomalyDetector()
        result = detector.fit(spatial_data)

        assert result is detector
        assert detector._is_fitted
        assert detector.center is not None
        assert detector.radius_threshold is not None

    def test_fit_torch_tensor(self, spatial_data: Any) -> None:
        """Test fitting with torch tensor."""
        tensor_data = torch.tensor(spatial_data, dtype=torch.float32)
        detector = SpatialAnomalyDetector()
        result = detector.fit(tensor_data)

        assert result is detector
        assert detector._is_fitted

    def test_fit_computes_center(self, spatial_data: Any) -> None:
        """Test that fitting computes correct center."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)

        expected_center = np.mean(spatial_data, axis=0)
        assert detector.center is not None
        np.testing.assert_array_almost_equal(detector.center, expected_center)

    def test_fit_computes_radius_threshold(self, spatial_data: Any) -> None:
        """Test that fitting computes 95th percentile radius."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)

        center = np.mean(spatial_data, axis=0)
        distances = np.linalg.norm(spatial_data - center, axis=1)
        expected_threshold = np.percentile(distances, 95)

        assert abs(detector.radius_threshold - expected_threshold) < 1e-6

    def test_fit_insufficient_dimensions_raises(self) -> None:
        """Test that 1D data raises exception."""
        data = np.random.randn(100, 1)
        detector = SpatialAnomalyDetector()

        with pytest.raises(DetectorException, match="at least 2 dimensions"):
            detector.fit(data)

    def test_detect_unfitted_raises(self, spatial_data: Any) -> None:
        """Test detection on unfitted detector raises exception."""
        detector = SpatialAnomalyDetector()

        with pytest.raises(DetectorException, match="must be fitted"):
            detector.detect(spatial_data)

    def test_detect_numpy_array(self, spatial_data: Any) -> None:
        """Test detection with numpy input."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        result = detector.detect(spatial_data)

        assert "is_anomaly" in result
        assert "scores" in result
        assert "distance_scores" in result
        assert "lof_scores" in result
        assert "detector_type" in result
        assert result["detector_type"] == "spatial"
        assert len(result["scores"]) == len(spatial_data)

    def test_detect_torch_tensor(self, spatial_data: Any) -> None:
        """Test detection with torch tensor input."""
        tensor_data = torch.tensor(spatial_data, dtype=torch.float32)
        detector = SpatialAnomalyDetector()
        detector.fit(tensor_data)
        result = detector.detect(tensor_data)

        assert "is_anomaly" in result
        assert len(result["scores"]) == len(spatial_data)

    def test_detect_identifies_far_outliers(self, spatial_data: Any) -> None:
        """Test that far outliers are detected."""
        detector = SpatialAnomalyDetector(config={"threshold": 0.3})
        detector.fit(spatial_data)

        # Create data with far outliers
        outliers = np.array([[100, 100], [-100, -100]])
        test_data = np.vstack([spatial_data[:5], outliers])

        result = detector.detect(test_data)

        # The outliers should have higher scores
        outlier_scores = result["scores"][-2:]
        normal_scores = result["scores"][:-2]
        assert np.mean(outlier_scores) > np.mean(normal_scores)

    def test_detect_3d_data(self, spatial_data_3d: Any) -> None:
        """Test detection with 3D spatial data."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data_3d)
        result = detector.detect(spatial_data_3d)

        assert len(result["scores"]) == len(spatial_data_3d)

    def test_extract_features_unfitted(self, spatial_data: Any) -> None:
        """Test feature extraction auto-fits if not fitted."""
        detector = SpatialAnomalyDetector()
        features = detector.extract_features(spatial_data)

        assert detector._is_fitted
        assert isinstance(features, torch.Tensor)

    def test_extract_features_numpy(self, spatial_data: Any) -> None:
        """Test feature extraction with numpy input."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        features = detector.extract_features(spatial_data)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == len(spatial_data)
        # Should be padded to at least 32 features
        assert features.shape[1] >= 32

    def test_extract_features_torch(self, spatial_data: Any) -> None:
        """Test feature extraction with torch tensor input."""
        tensor_data = torch.tensor(spatial_data, dtype=torch.float32)
        detector = SpatialAnomalyDetector()
        detector.fit(tensor_data)
        features = detector.extract_features(tensor_data)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == len(spatial_data)

    def test_extract_features_content(self, spatial_data: Any) -> None:
        """Test that extracted features contain expected components."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        features = detector.extract_features(spatial_data)

        # Features should include: coordinates (2), distance (1), angle (1)
        # Total = 4, padded to 32
        assert features.shape[1] == 32

    def test_compute_distance_scores(self, spatial_data: Any) -> None:
        """Test distance score computation."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)

        scores = detector._compute_distance_scores(spatial_data)

        assert len(scores) == len(spatial_data)
        assert np.all(scores >= 0)

    def test_scores_normalized(self, spatial_data: Any) -> None:
        """Test that combined scores are properly normalized."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        result = detector.detect(spatial_data)

        assert np.min(result["scores"]) >= 0
        assert np.max(result["scores"]) <= 1


class TestNativeLOFDuplicateClusterContract:
    """Lock the LOF train/inference symmetry contract.

    Regression coverage for the ``_NativeLOF._REACH_FLOOR`` symmetry: an
    earlier implementation floored ``mean_reach`` at ``np.finfo.eps`` in
    fit and at ``1e-10`` in ``decision_function``. The 5-orders-of-magnitude
    mismatch broke the LOF ratio on duplicate-cluster queries, mapping them
    to ``decision ~= -4.5e5`` (massive outlier) instead of ``~0`` (inlier).
    These tests fail closed if the floors are ever unmatched again.
    """

    @pytest.mark.parametrize(
        "X_train, label",
        [
            (np.zeros((30, 2), dtype=np.float64), "30 pure duplicates"),
            (
                np.vstack(
                    [
                        np.random.RandomState(13).randn(50, 2) * 5 + np.array([10.0, 10.0]),
                        np.tile(np.zeros(2, dtype=np.float64), (8, 1)),
                    ]
                ),
                "8 duplicates + 50 normal points",
            ),
            (
                np.vstack(
                    [
                        np.zeros((10, 3), dtype=np.float64),
                        np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=np.float64),
                    ]
                ),
                "10 duplicates + 2 distant points",
            ),
        ],
    )
    def test_duplicate_cluster_query_scores_as_inlier(
        self, X_train: np.ndarray[Any, Any], label: str
    ) -> None:
        """Query inside a duplicate cluster must score as an inlier (decision >= -0.5).

        Previously this was -4.5e5 due to train/inference floor mismatch.
        """
        lof = _NativeLOF(n_neighbors=5).fit(X_train)
        query = np.zeros((1, X_train.shape[1]), dtype=np.float64)
        decision = float(lof.decision_function(query)[0])
        assert decision >= -0.5, (
            f"[{label}] duplicate-cluster query mis-scored as anomaly: "
            f"decision={decision:.3e} (expected ~0). The fit and inference "
            f"reach-floor symmetry is broken; see _NativeLOF._REACH_FLOOR."
        )

    def test_far_isolated_query_scores_as_outlier(self) -> None:
        """A query far from any training cluster must score negative."""
        X_train = np.vstack(
            [
                np.zeros((10, 3), dtype=np.float64),
                np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=np.float64),
            ]
        )
        lof = _NativeLOF(n_neighbors=5).fit(X_train)
        far = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
        decision = float(lof.decision_function(far)[0])
        assert decision < 0, f"far-isolated query mis-scored as inlier: decision={decision:.3e}"

    def test_train_inference_floor_symmetry(self) -> None:
        """The fit and inference paths must use the same _REACH_FLOOR constant.

        If the two ever drift apart, duplicate-cluster classification regresses.
        """
        assert _NativeLOF._REACH_FLOOR == 1e-10, (
            "The _REACH_FLOOR must remain 1e-10 (sklearn convention) and the "
            "single constant must be used by both fit and decision_function. "
            "Any value is fine in principle as long as fit and decision_function "
            "share it byte-for-byte; differing floors break the LOF ratio."
        )

    def test_sign_convention_on_unambiguous_cases(self) -> None:
        """decision_function must honour the LOF sign convention (positive/zero ~
        inlier, negative ~ outlier) on cases where the answer is analytically
        unambiguous — no competitor oracle required.

        We restrict to two unambiguous queries — the centre of a dense duplicate
        cluster (clear inlier) and a point 100 stds from any training sample
        (clear outlier) — because near-cluster queries depend on density details
        where ``inlier vs outlier`` is judgement-dependent. The head-to-head
        magnitude comparison against scikit-learn's reference LOF lives in
        ``benchmarks/`` (sklearn is a competitor, never imported here). The -0.5
        floor on the inlier matches sklearn's convention (a true inlier scores
        ~0) without needing sklearn present to tell us so.
        """
        rng = np.random.RandomState(7)
        X_train = np.vstack(
            [rng.randn(50, 2) * 3.0, np.tile(np.zeros(2, dtype=np.float64), (8, 1))]
        )
        native = _NativeLOF(n_neighbors=5).fit(X_train)

        # Unambiguous inlier: a query placed exactly at the duplicate cluster
        # must score as non-anomalous (decision >= -0.5; a true inlier is ~0).
        q_dup = np.array([[0.0, 0.0]], dtype=np.float64)
        n_dup = float(native.decision_function(q_dup)[0])
        assert n_dup >= -0.5, f"duplicate-cluster query mis-scored as outlier: native={n_dup:.3e}"

        # Unambiguous outlier: a query 100 stds away must score negative.
        q_far = np.array([[100.0, 100.0]], dtype=np.float64)
        n_far = float(native.decision_function(q_far)[0])
        assert n_far < 0, f"far-isolated query mis-scored as inlier: native={n_far:.3e}"


class TestLOFKEqualsOneRegression:
    """k == 1 LOF queries must not crash (defect found 2026-06-11).

    scipy's ``KDTree.query`` squeezes the neighbor axis when ``k == 1`` —
    fitting on a single sample, or configuring ``n_neighbors=1``, made every
    ``decision_function`` call raise ``AxisError`` at inference.
    """

    def test_single_training_sample_scores(self) -> None:
        detector = SpatialAnomalyDetector()
        detector.fit(np.array([[0.0, 0.0, 0.0]]))
        result = detector.detect(np.array([[0.1, 0.0, 0.0], [9.0, 9.0, 9.0]]))
        assert result["scores"].shape == (2,)
        assert np.all(np.isfinite(result["scores"]))

    def test_n_neighbors_one_scores(self) -> None:
        rng = np.random.RandomState(3)
        detector = SpatialAnomalyDetector({"n_neighbors": 1})
        detector.fit(rng.randn(40, 3))
        result = detector.detect(rng.randn(12, 3))
        assert result["scores"].shape == (12,)
        assert np.all(np.isfinite(result["scores"]))

    def test_k_one_lof_far_query_is_anomalous(self) -> None:
        """The restored axis must preserve LOF semantics, not just shape."""
        rng = np.random.RandomState(5)
        lof = _NativeLOF(n_neighbors=1).fit(rng.randn(30, 2))
        near = float(lof.decision_function(rng.randn(1, 2))[0])
        far = float(lof.decision_function(np.array([[80.0, 80.0]]))[0])
        assert far < near


class TestSpatialNonFiniteInputRegression:
    """Non-finite queries must be sanitized, not crash (defect found 2026-06-11).

    ``KDTree.query`` refuses NaN/Inf outright, which crashed ``detect()``
    before its own downstream NaN guards could run — the temporal detector
    already sanitizes at the same boundary.
    """

    def test_nan_query_scores(self) -> None:
        rng = np.random.RandomState(11)
        detector = SpatialAnomalyDetector()
        detector.fit(rng.randn(60, 4))
        queries = rng.randn(5, 4)
        queries[2, 1] = np.nan
        queries[4, 3] = np.inf
        result = detector.detect(queries)
        assert result["scores"].shape == (5,)
        assert np.all(np.isfinite(result["scores"]))

    def test_finite_rows_unchanged_by_sanitization(self) -> None:
        """Sanitization is a no-op for finite inputs (bit-identical scores)."""
        rng = np.random.RandomState(13)
        detector = SpatialAnomalyDetector()
        detector.fit(rng.randn(60, 4))
        queries = rng.randn(8, 4)
        baseline = detector.detect(queries)["scores"]
        np.testing.assert_array_equal(detector.detect(queries)["scores"], baseline)
