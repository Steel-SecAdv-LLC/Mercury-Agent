# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Information Geometry integration."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.info_geometry import InformationGeometryDetector


class TestInformationGeometryDetector:
    """Test InformationGeometryDetector class."""

    def test_detector_initialization(self) -> None:
        """Test detector initialization."""
        detector = InformationGeometryDetector()
        assert detector.distance_metric == "fisher_rao"
        assert detector.manifold_dim == 10
        assert detector.approximation_method == "closed_form"
        assert detector.reference_distribution is None
        assert detector.fisher_matrix is None

    def test_detector_custom_config(self) -> None:
        """Test detector with custom configuration."""
        config = {
            "distance_metric": "kl_divergence",
            "manifold_dim": 20,
            "approximation_method": "sampling",
        }
        detector = InformationGeometryDetector(config)
        assert detector.distance_metric == "kl_divergence"
        assert detector.manifold_dim == 20
        assert detector.approximation_method == "sampling"

    def test_fit_reference_distribution(self) -> None:
        """Test fitting reference distribution."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.reference_distribution is not None
        assert "mean" in detector.reference_distribution
        assert "cov" in detector.reference_distribution
        assert detector.fisher_matrix is not None

    def test_reference_mean_shape(self) -> None:
        """Test reference distribution mean shape."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.reference_distribution is not None
        assert detector.reference_distribution["mean"].shape == (10,)

    def test_reference_cov_shape(self) -> None:
        """Test reference distribution covariance shape."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.reference_distribution is not None
        assert detector.reference_distribution["cov"].shape == (10, 10)

    def test_fisher_matrix_shape(self) -> None:
        """Test Fisher matrix shape."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.fisher_matrix is not None
        assert detector.fisher_matrix.shape == (10, 10)

    def test_compute_fisher_matrix(self) -> None:
        """Test Fisher matrix computation."""
        detector = InformationGeometryDetector()
        mean = np.zeros(5)
        cov = np.eye(5)

        fisher = detector._compute_fisher_matrix(mean, cov)

        assert fisher.shape == (5, 5)
        assert np.all(np.isfinite(fisher))

    def test_fisher_matrix_invertible(self) -> None:
        """Test that Fisher matrix is invertible."""
        detector = InformationGeometryDetector()
        mean = np.zeros(5)
        cov = np.eye(5)

        fisher = detector._compute_fisher_matrix(mean, cov)

        det = np.linalg.det(fisher)
        assert det > 0

    def test_fisher_rao_distance_same_distribution(self) -> None:
        """Test Fisher-Rao distance is zero for same distribution."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        dist1 = {"mean": np.zeros(10), "cov": np.eye(10)}
        dist2 = {"mean": np.zeros(10), "cov": np.eye(10)}

        distance = detector.fisher_rao_distance(dist1, dist2)
        assert distance == 0.0

    def test_fisher_rao_distance_different_distributions(self) -> None:
        """Test Fisher-Rao distance for different distributions."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        dist1 = {"mean": np.zeros(10), "cov": np.eye(10)}
        dist2 = {"mean": np.ones(10), "cov": np.eye(10)}

        distance = detector.fisher_rao_distance(dist1, dist2)
        assert distance > 0.0

    def test_fisher_rao_distance_positive(self) -> None:
        """Test Fisher-Rao distance is always non-negative."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        dist1 = {"mean": np.random.randn(10), "cov": np.eye(10)}
        dist2 = {"mean": np.random.randn(10), "cov": np.eye(10)}

        distance = detector.fisher_rao_distance(dist1, dist2)
        assert distance >= 0.0

    def test_detect_ood_without_fit_raises_error(self) -> None:
        """Test that detect_ood raises error if not fitted."""
        detector = InformationGeometryDetector()
        test_data = np.random.randn(50, 10)

        with pytest.raises(ValueError, match="Must fit reference distribution first"):
            detector.detect_ood(test_data)

    def test_detect_ood_basic(self) -> None:
        """Test basic OOD detection."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert "ood_score" in results
        assert "is_ood" in results
        assert "threshold" in results
        assert "method" in results

    def test_detect_ood_method_label(self) -> None:
        """Test that method label is correct."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert results["method"] == "fisher_rao_geometry"

    def test_detect_ood_custom_threshold(self) -> None:
        """Test OOD detection with custom threshold."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data, threshold=5.0)

        assert results["threshold"] == 5.0

    def test_detect_ood_default_threshold(self) -> None:
        """Test OOD detection with default threshold."""
        # With adaptive_threshold disabled, fallback is 3.0
        detector = InformationGeometryDetector(config={"adaptive_threshold": False})
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert results["threshold"] == 3.0

    def test_detect_ood_adaptive_threshold(self) -> None:
        """Test OOD detection with adaptive threshold (default behavior)."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        # Adaptive threshold is FIM-derived, must be positive
        assert results["threshold"] > 0
        assert results["adaptive"] is True

    def test_detect_ood_score_type(self) -> None:
        """Test OOD score is float."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert isinstance(results["ood_score"], float)

    def test_detect_ood_is_ood_type(self) -> None:
        """Test is_ood is boolean."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert isinstance(results["is_ood"], (bool, np.bool_))

    def test_detect_ood_single_sample(self) -> None:
        """Test OOD detection with single sample."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(1, 10)
        results = detector.detect_ood(test_data)

        assert "ood_score" in results

    def test_distance_metric_config(self) -> None:
        """Test distance metric configuration."""
        config = {"distance_metric": "kl_divergence"}
        detector = InformationGeometryDetector(config)
        assert detector.distance_metric == "kl_divergence"

    def test_manifold_dim_config(self) -> None:
        """Test manifold dimension configuration."""
        config = {"manifold_dim": 15}
        detector = InformationGeometryDetector(config)
        assert detector.manifold_dim == 15

    def test_approximation_method_config(self) -> None:
        """Test approximation method configuration."""
        config = {"approximation_method": "sampling"}
        detector = InformationGeometryDetector(config)
        assert detector.approximation_method == "sampling"


class TestFisherMatrixCorrectness:
    """Mathematical correctness tests for Fisher Information Matrix."""

    def test_fim_equals_cov_inverse_for_identity(self) -> None:
        """For Gaussian with identity covariance, FIM should be identity."""
        detector = InformationGeometryDetector()
        mean = np.zeros(5)
        cov = np.eye(5)
        fisher = detector._compute_fisher_matrix(mean, cov)
        # FIM = cov^{-1} + Tikhonov regularization
        # For identity cov, FIM ≈ I + lambda*I = (1+lambda)*I
        # The diagonal should be close to 1.0 (regularization is small)
        diag = np.diag(fisher)
        assert np.allclose(
            diag, diag[0], atol=1e-6
        ), "FIM diagonal should be uniform for identity cov"
        assert diag[0] > 0.99, "FIM diagonal for identity cov should be close to 1.0"

    def test_fim_equals_cov_inverse_for_scaled_identity(self) -> None:
        """For Gaussian with sigma^2 * I covariance, FIM diagonal = 1/sigma^2."""
        detector = InformationGeometryDetector()
        sigma_sq = 4.0
        mean = np.zeros(3)
        cov = sigma_sq * np.eye(3)
        fisher = detector._compute_fisher_matrix(mean, cov)
        expected_diag = 1.0 / sigma_sq  # = 0.25
        diag = np.diag(fisher)
        assert np.allclose(
            diag, expected_diag, atol=0.01
        ), f"FIM diagonal for {sigma_sq}*I should be ~{expected_diag}, got {diag}"

    def test_fim_symmetry(self) -> None:
        """FIM must be symmetric."""
        detector = InformationGeometryDetector()
        cov = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.3], [0.1, 0.3, 1.0]])
        fisher = detector._compute_fisher_matrix(np.zeros(3), cov)
        assert np.allclose(fisher, fisher.T, atol=1e-10), "FIM must be symmetric"

    def test_fim_positive_definite(self) -> None:
        """FIM must be positive definite (all eigenvalues > 0)."""
        detector = InformationGeometryDetector()
        cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        fisher = detector._compute_fisher_matrix(np.zeros(2), cov)
        eigenvalues = np.linalg.eigvalsh(fisher)
        assert np.all(eigenvalues > 0), f"FIM eigenvalues must all be positive, got {eigenvalues}"

    def test_fisher_rao_distance_symmetry(self) -> None:
        """Fisher-Rao distance must be symmetric: d(P,Q) = d(Q,P)."""
        detector = InformationGeometryDetector()
        detector.fit_reference_distribution(np.random.randn(100, 5))
        dist1 = {"mean": np.array([1.0, 0.0, 0.0, 0.0, 0.0]), "cov": np.eye(5)}
        dist2 = {"mean": np.array([0.0, 1.0, 0.0, 0.0, 0.0]), "cov": np.eye(5)}
        d12 = detector.fisher_rao_distance(dist1, dist2)
        d21 = detector.fisher_rao_distance(dist2, dist1)
        assert abs(d12 - d21) < 1e-10, f"Distance not symmetric: {d12} vs {d21}"

    def test_fisher_rao_distance_scales_with_separation(self) -> None:
        """Larger mean separation should yield larger Fisher-Rao distance."""
        detector = InformationGeometryDetector()
        detector.fit_reference_distribution(np.random.randn(100, 3))
        base = {"mean": np.zeros(3), "cov": np.eye(3)}
        near = {"mean": np.array([0.5, 0.0, 0.0]), "cov": np.eye(3)}
        far = {"mean": np.array([5.0, 0.0, 0.0]), "cov": np.eye(3)}
        d_near = detector.fisher_rao_distance(base, near)
        d_far = detector.fisher_rao_distance(base, far)
        assert d_far > d_near, f"Farther point should have larger distance: {d_far} vs {d_near}"

    def test_ood_detects_shifted_distribution(self) -> None:
        """OOD detector should flag data shifted far from reference."""
        np.random.seed(42)
        detector = InformationGeometryDetector(config={"adaptive_threshold": False})
        reference = np.random.randn(200, 5)
        detector.fit_reference_distribution(reference)
        # Test with data shifted very far from reference
        shifted = np.random.randn(50, 5) + 100.0
        results = detector.detect_ood(shifted)
        assert results["ood_score"] > 0, "Shifted data should have positive OOD score"
        assert results["is_ood"], "Data shifted by 100 sigma should be flagged OOD"

    def test_ood_in_distribution_data_low_score(self) -> None:
        """In-distribution data should have low OOD scores."""
        np.random.seed(42)
        detector = InformationGeometryDetector(config={"adaptive_threshold": False})
        reference = np.random.randn(200, 5)
        detector.fit_reference_distribution(reference)
        # Test with data from same distribution
        in_dist = np.random.randn(50, 5)
        results = detector.detect_ood(in_dist, threshold=100.0)
        assert not results["is_ood"], "In-distribution data should not be flagged as OOD"
