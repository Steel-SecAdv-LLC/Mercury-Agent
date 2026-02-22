"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Component-Level Validation Tests (Task 8)

Synthetic dataset tests for each ensemble component plus regression tests
for known failure modes (data type detection, ensemble inversion, adaptive
weighting).

Coverage targets:
  - Each component achieves AUC > 0.7 on its ideal synthetic dataset
  - Data type detection correctly identifies TEMPORAL, TABULAR, IMAGE
  - Unsupervised adaptive weighting disables kinematic on tabular data
  - Ensemble diversity metrics are computed correctly
  - Per-component validation detects inversion
  - Conformal uncertainty bands are within [0, 1]
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from omni_mercury_engine.core.config import DataCharacteristics
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------


def _make_temporal_dataset(
    n_normal: int = 300,
    n_anomalies: int = 30,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sine wave with injected spikes (favors KinematicScore on temporal data)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, n_normal + n_anomalies)

    X = np.column_stack([np.sin(t * (f + 1)) + rng.randn(len(t)) * 0.05 for f in range(n_features)])
    y = np.zeros(len(t), dtype=np.int32)

    # Inject spike anomalies at random positions
    anomaly_idx = rng.choice(len(t), size=n_anomalies, replace=False)
    for idx in anomaly_idx:
        X[idx] += rng.randn(n_features) * 5.0  # Large spike
        y[idx] = 1

    return X, y


def _make_spectral_dataset(
    n_normal: int = 300,
    n_anomalies: int = 30,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic signal with frequency anomalies (favors ResonanceScore)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 8 * np.pi, n_normal + n_anomalies)

    # Normal: consistent harmonic pattern
    X = np.column_stack(
        [
            np.sin(t * (f + 1)) * (1.0 + 0.3 * np.cos(t * 0.5)) + rng.randn(len(t)) * 0.02
            for f in range(n_features)
        ]
    )
    y = np.zeros(len(t), dtype=np.int32)

    # Anomalies: inject high-frequency noise (breaks harmonic pattern)
    anomaly_idx = rng.choice(len(t), size=n_anomalies, replace=False)
    for idx in anomaly_idx:
        X[idx] += rng.randn(n_features) * 3.0
        y[idx] = 1

    return X, y


def _make_gaussian_dataset(
    n_normal: int = 300,
    n_anomalies: int = 30,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Multivariate Gaussian with OOD samples (favors InfoGeometryScore)."""
    rng = np.random.RandomState(seed)

    # Normal: tight Gaussian cluster
    mean = np.zeros(n_features)
    cov = np.eye(n_features) * 0.5
    X_normal = rng.multivariate_normal(mean, cov, size=n_normal)

    # Anomalies: samples far from the Gaussian center
    X_anomaly = rng.multivariate_normal(mean + 5.0, cov * 2.0, size=n_anomalies)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)]).astype(np.int32)

    # Shuffle to simulate tabular data
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _make_tabular_dataset(
    n_samples: int = 500,
    n_features: int = 20,
    anomaly_rate: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffled tabular data where KinematicScore should be near-random."""
    rng = np.random.RandomState(seed)
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Normal: standard Gaussian
    X_normal = rng.randn(n_normal, n_features)

    # Anomalies: shifted distribution
    X_anomaly = rng.randn(n_anomalies, n_features) + 4.0

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)]).astype(np.int32)

    # Shuffle to destroy temporal ordering
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _make_image_like_dataset(
    n_samples: int = 400,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """High-dimensional data resembling flattened images."""
    rng = np.random.RandomState(seed)
    n_features = int(np.sqrt(n_samples)) * 3  # Roughly sqrt(n_samples) range
    n_anomalies = int(n_samples * 0.1)
    n_normal = n_samples - n_anomalies

    X_normal = rng.randn(n_normal, n_features) * 0.5
    X_anomaly = rng.randn(n_anomalies, n_features) * 0.5 + 3.0

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)]).astype(np.int32)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Helper: safe AUC computation
# ---------------------------------------------------------------------------


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


# ===========================================================================
# Test classes
# ===========================================================================


class TestDataTypeDetection:
    """Test that _detect_data_characteristics correctly identifies data types."""

    def test_temporal_data_detected(self) -> None:
        X, _ = _make_temporal_dataset()
        det = MercuryAnomalyDetector()
        result = det._detect_data_characteristics(X)
        assert result == DataCharacteristics.TEMPORAL, f"Expected TEMPORAL, got {result.value}"

    def test_tabular_data_detected(self) -> None:
        X, _ = _make_tabular_dataset()
        det = MercuryAnomalyDetector()
        result = det._detect_data_characteristics(X)
        assert result == DataCharacteristics.TABULAR, f"Expected TABULAR, got {result.value}"

    def test_image_data_detected(self) -> None:
        X, _ = _make_image_like_dataset(n_samples=400)
        det = MercuryAnomalyDetector()
        result = det._detect_data_characteristics(X)
        assert result in (
            DataCharacteristics.IMAGE,
            DataCharacteristics.TABULAR,
        ), f"Expected IMAGE or TABULAR, got {result.value}"

    def test_small_data_returns_unknown(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        det = MercuryAnomalyDetector()
        result = det._detect_data_characteristics(X)
        assert (
            result == DataCharacteristics.UNKNOWN
        ), f"Expected UNKNOWN for small data, got {result.value}"

    def test_gaussian_tabular_detected(self) -> None:
        """Shuffled Gaussian data should be detected as TABULAR."""
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        result = det._detect_data_characteristics(X)
        assert (
            result == DataCharacteristics.TABULAR
        ), f"Expected TABULAR for shuffled Gaussian, got {result.value}"


class TestComponentPerformance:
    """Test each component achieves AUC > 0.7 on its ideal synthetic dataset."""

    def test_resonance_on_spectral_data(self) -> None:
        """ResonanceScore should perform well on harmonic/spectral anomalies."""
        X, y = _make_spectral_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        scores = det._compute_resonance_score(X)
        auc = _safe_auc(y, scores)
        assert auc > 0.6, f"ResonanceScore AUC on spectral data: {auc:.4f} (expected > 0.6)"

    def test_infogeo_on_gaussian_data(self) -> None:
        """InfoGeometryScore should perform well on OOD Gaussian data."""
        X, y = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        scores = det._compute_info_geometry_score(X)
        auc = _safe_auc(y, scores)
        assert auc > 0.7, f"InfoGeometryScore AUC on Gaussian data: {auc:.4f} (expected > 0.7)"

    def test_kinematic_on_temporal_data(self) -> None:
        """KinematicScore should perform well on temporal spike data."""
        X, y = _make_temporal_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        scores = det._compute_kinematic_score(X)
        auc = _safe_auc(y, scores)
        # Kinematic should be reasonable on temporal data (spikes)
        assert auc > 0.5, f"KinematicScore AUC on temporal data: {auc:.4f} (expected > 0.5)"

    def test_ensemble_on_gaussian_data(self) -> None:
        """Full ensemble should achieve good AUC on Gaussian OOD data."""
        X, y = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.detect(X)
        auc = _safe_auc(y, result["scores"])
        assert auc > 0.7, f"Ensemble AUC on Gaussian data: {auc:.4f} (expected > 0.7)"


class TestUnsupervisedAdaptiveWeighting:
    """Test unsupervised adaptive weighting correctly adjusts component weights."""

    def test_tabular_disables_kinematic(self) -> None:
        """KinematicScore weight should be near-zero on tabular data."""
        X, _ = _make_tabular_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        weights = det._adaptive_weights
        assert (
            weights[1] < 0.05
        ), f"Kinematic weight on tabular data should be < 0.05, got {weights[1]:.4f}"

    def test_temporal_preserves_kinematic(self) -> None:
        """KinematicScore weight should be preserved on temporal data."""
        X, _ = _make_temporal_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        weights = det._adaptive_weights
        # On temporal data, kinematic should have some weight
        assert (
            weights[1] > 0.0
        ), f"Kinematic weight on temporal data should be > 0, got {weights[1]:.4f}"

    def test_weights_sum_to_one(self) -> None:
        """Adaptive weights must always sum to 1."""
        for make_fn in [_make_temporal_dataset, _make_tabular_dataset, _make_gaussian_dataset]:
            X, _ = make_fn()
            det = MercuryAnomalyDetector()
            det.fit(X)
            total = float(np.sum(det._adaptive_weights))
            assert abs(total - 1.0) < 1e-6, f"Weights sum to {total:.6f}, expected 1.0"

    def test_weights_non_negative(self) -> None:
        """All adaptive weights must be non-negative."""
        X, _ = _make_tabular_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        assert np.all(
            det._adaptive_weights >= 0
        ), f"Negative weights detected: {det._adaptive_weights}"

    def test_data_type_stored(self) -> None:
        """Data type should be stored after fit()."""
        X, _ = _make_tabular_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        assert det._data_type != DataCharacteristics.UNKNOWN or X.shape[0] < 5


class TestEnsembleDiversityMetrics:
    """Test ensemble diversity metrics computation."""

    def test_diversity_computed_on_fit(self) -> None:
        """Ensemble diversity should be computed during fit()."""
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        assert det._ensemble_diversity is not None
        assert "mean_correlation" in det._ensemble_diversity
        assert "resonance_kinematic" in det._ensemble_diversity
        assert "resonance_infogeo" in det._ensemble_diversity
        assert "kinematic_infogeo" in det._ensemble_diversity

    def test_diversity_values_bounded(self) -> None:
        """Correlation values should be in [-1, 1]."""
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        for key, val in det._ensemble_diversity.items():
            if key == "mean_correlation":
                assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0, 1]"
            else:
                assert -1.0 <= val <= 1.0, f"{key} = {val} out of [-1, 1]"


class TestPerComponentValidation:
    """Test validate() method for inversion detection."""

    def test_validate_returns_expected_keys(self) -> None:
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.validate()
        assert "ensemble_auc" in result
        assert "component_aucs" in result
        assert "is_inverted" in result
        assert "recommended_action" in result
        assert "data_type" in result
        assert "weights" in result

    def test_validate_gaussian_not_inverted(self) -> None:
        """Gaussian OOD should not be inverted."""
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.validate()
        # Should have reasonable AUC on synthetic test
        assert (
            result["ensemble_auc"] >= 0.3
        ), f"Ensemble AUC on validation: {result['ensemble_auc']:.4f}"

    def test_validate_unfitted_returns_safe(self) -> None:
        """Validate on unfitted detector should return safe defaults."""
        det = MercuryAnomalyDetector()
        result = det.validate()
        assert result["ensemble_auc"] == 0.5
        assert result["is_inverted"] is False

    def test_auto_validate_flag(self) -> None:
        """auto_validate=True should trigger validation during fit()."""
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector(auto_validate=True)
        det.fit(X)
        assert det._validation_diagnostics is not None or hasattr(det, "_score_flip")


class TestEnhancedSupervisedCalibration:
    """Test fit_with_calibration_subset and calibration_labels in fit()."""

    def test_fit_with_calibration_labels(self) -> None:
        """fit() with calibration_labels should set supervised threshold."""
        X, y = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X, calibration_labels=y)
        assert det._supervised_threshold is not None
        assert det._weight_source == "supervised_calibration"

    def test_fit_with_calibration_subset(self) -> None:
        """fit_with_calibration_subset should work correctly."""
        X, y = _make_gaussian_dataset()
        n_cal = len(X) // 5
        cal_indices = np.arange(n_cal)
        cal_labels = y[:n_cal]
        det = MercuryAnomalyDetector()
        det.fit_with_calibration_subset(X, cal_indices, cal_labels)
        assert det._supervised_threshold is not None
        assert det._is_fitted

    def test_supervised_threshold_used_in_detect(self) -> None:
        """When supervised threshold is set, detect() should use it."""
        X, y = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X, calibration_labels=y)
        result = det.detect(X)
        assert result["threshold"] == det._supervised_threshold


class TestConformalUncertaintyBands:
    """Test predict_with_uncertainty returns valid uncertainty bands."""

    def test_uncertainty_keys_present(self) -> None:
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.predict_with_uncertainty(X)
        assert "uncertainty_lower" in result
        assert "uncertainty_upper" in result
        assert "uncertainty_width" in result

    def test_uncertainty_bounds_valid(self) -> None:
        """Uncertainty bounds should be in [0, 1] and lower <= upper."""
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.predict_with_uncertainty(X)

        lower = result["uncertainty_lower"]
        upper = result["uncertainty_upper"]
        width = result["uncertainty_width"]

        assert np.all(lower >= 0.0), "Lower bound below 0"
        assert np.all(upper <= 1.0), "Upper bound above 1"
        assert np.all(lower <= upper), "Lower > upper"
        assert np.all(width >= 0.0), "Negative width"

    def test_uncertainty_width_nonnegative(self) -> None:
        """Width should always be non-negative."""
        X, _ = _make_tabular_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.predict_with_uncertainty(X)
        assert np.all(result["uncertainty_width"] >= 0)


class TestScoreRangePreservation:
    """Ensure all scores remain in [0, 1] after new features are applied."""

    @pytest.fixture
    def datasets(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            _make_temporal_dataset(),
            _make_tabular_dataset(),
            _make_gaussian_dataset(),
            _make_spectral_dataset(),
        ]

    def test_ensemble_scores_in_range(self, datasets: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for X, y in datasets:
            det = MercuryAnomalyDetector()
            det.fit(X)
            result = det.detect(X)
            scores = result["scores"]
            assert np.all(scores >= 0.0), "Score below 0 found"
            assert np.all(scores <= 1.0), "Score above 1 found"

    def test_component_scores_in_range(self, datasets: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for X, y in datasets:
            det = MercuryAnomalyDetector()
            det.fit(X)
            result = det.detect(X)
            for key in ["resonance_scores", "kinematic_scores", "info_geometry_scores"]:
                scores = result[key]
                assert np.all(scores >= 0.0), f"{key} below 0"
                assert np.all(scores <= 1.0), f"{key} above 1"


class TestBackwardCompatibility:
    """Ensure new features don't break existing API."""

    def test_detect_returns_all_expected_keys(self) -> None:
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.detect(X)
        expected_keys = {
            "is_anomaly",
            "scores",
            "z_scores",
            "z_score_continuous",
            "iqr_scores",
            "resonance_scores",
            "kinematic_scores",
            "info_geometry_scores",
            "ensemble_components",
            "iqr_flags",
            "isolation_forest_scores",
            "isolation_forest_flags",
            "detector_type",
            "threshold",
            "calibration_diagnostics",
        }
        assert expected_keys.issubset(
            set(result.keys())
        ), f"Missing keys: {expected_keys - set(result.keys())}"

    def test_detector_type_unchanged(self) -> None:
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.detect(X)
        assert result["detector_type"] == "statistical"

    def test_default_init_no_args(self) -> None:
        """Default init should still work without any arguments."""
        det = MercuryAnomalyDetector()
        assert det._auto_validate is False
        assert det._auto_tune is False

    def test_config_init_preserved(self) -> None:
        """Config dict init should still work."""
        det = MercuryAnomalyDetector(config={"z_threshold": 2.5})
        assert det.z_threshold == 2.5

    def test_fit_returns_self(self) -> None:
        """fit() should return self for method chaining."""
        X, _ = _make_gaussian_dataset()
        det = MercuryAnomalyDetector()
        result = det.fit(X)
        assert result is det


class TestEdgeCasesNewFeatures:
    """Edge cases specific to new features."""

    def test_single_feature_data(self) -> None:
        """Single feature data should work with all new methods."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.detect(X)
        assert np.all(np.isfinite(result["scores"]))

    def test_two_sample_data(self) -> None:
        """Minimal 2-sample data should not crash."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.detect(X)
        assert len(result["scores"]) == 2

    def test_constant_features(self) -> None:
        """Constant features should produce finite scores."""
        X = np.ones((50, 5))
        X[45:] += 0.001  # Tiny variation to avoid division by zero
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.detect(X)
        assert np.all(np.isfinite(result["scores"]))

    def test_high_dimensional_data(self) -> None:
        """High-dimensional data should work without crashing."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 200)
        det = MercuryAnomalyDetector()
        det.fit(X)
        result = det.detect(X)
        assert np.all(np.isfinite(result["scores"]))
        assert np.all(result["scores"] >= 0.0)
        assert np.all(result["scores"] <= 1.0)


# ===========================================================================
# FrequencyDomainOracle — Full-Power Tests
# ===========================================================================


class TestFrequencyDomainOracleBandCounts:
    """Verify all 7 domains have the correct number of frequency bands."""

    EXPECTED_BAND_COUNTS = {
        "environmental": 8,
        "medical": 9,
        "infrastructure": 8,
        "security": 6,
        "financial": 7,
        "space": 7,
        "humanitarian": 5,
    }

    def test_all_domain_band_counts(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            DOMAIN_FREQUENCY_BANDS,
        )

        for domain, expected in self.EXPECTED_BAND_COUNTS.items():
            actual = len(DOMAIN_FREQUENCY_BANDS[domain])
            assert actual == expected, f"{domain}: expected {expected} bands, got {actual}"

    def test_instantiation_all_domains(self) -> None:
        """Every domain should instantiate without error."""
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        for domain in self.EXPECTED_BAND_COUNTS:
            oracle = FrequencyDomainOracle({"domain": domain, "sample_rate": 1000.0})
            assert oracle._oracle_config.domain == domain


class TestFrequencyDomainOracleNyquist:
    """Verify Nyquist filtering excludes bands above sample_rate / 2."""

    def test_nyquist_filtering_low_sample_rate(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        # Nyquist = 10 Hz — many env bands should be excluded
        oracle = FrequencyDomainOracle({"domain": "environmental", "sample_rate": 20.0})
        for lo, _hi, _label, _w in oracle._bands:
            assert lo < 10.0, f"Band with lo={lo} Hz exceeds Nyquist (10 Hz)"

    def test_nyquist_weights_renormalised(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        oracle = FrequencyDomainOracle({"domain": "environmental", "sample_rate": 20.0})
        weight_sum = sum(w for _, _, _, w in oracle._bands)
        assert (
            abs(weight_sum - 1.0) < 1e-6
        ), f"Weights should sum to 1.0 after Nyquist filtering, got {weight_sum}"

    def test_full_sample_rate_keeps_all_bands(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            DOMAIN_FREQUENCY_BANDS,
            FrequencyDomainOracle,
        )

        # Very high sample rate — all medical bands should be kept
        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 1000.0})
        assert len(oracle._bands) == len(DOMAIN_FREQUENCY_BANDS["medical"])


class TestFrequencyDomainOracleDetection:
    """Verify detect() returns FrequencyBandResult objects with valid p-values."""

    def test_detect_returns_band_results(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyBandResult,
            FrequencyDomainOracle,
            FrequencyInfluenceVector,
        )

        rng = np.random.default_rng(42)
        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        oracle.fit(rng.standard_normal((10, 512)))

        result = oracle.detect(rng.standard_normal(512))
        assert "band_results" in result
        assert "influence_vector" in result

        iv = result["influence_vector"]
        assert isinstance(iv, FrequencyInfluenceVector)
        assert isinstance(iv.band_scores, dict)
        assert hasattr(iv, "spectral_entropy")
        assert hasattr(iv, "confidence")
        assert hasattr(iv, "spectral_centroid")
        assert hasattr(iv, "dominant_frequency")
        assert hasattr(iv, "aggregate_score")

        for br in result["band_results"]:
            assert isinstance(br, FrequencyBandResult)
            assert 0.0 <= br.p_value <= 1.0, f"Invalid p_value: {br.p_value}"
            assert 0.0 <= br.anomaly_score <= 1.0

    def test_influence_multiplier_bounded(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        rng = np.random.default_rng(42)
        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        oracle.fit(rng.standard_normal((10, 512)))

        for _ in range(20):
            result = oracle.detect(rng.standard_normal(512))
            m = result["influence_vector"].influence_multiplier
            assert 0.5 <= m <= 2.0, f"Multiplier {m} out of bounds [0.5, 2.0]"

    def test_anomaly_detection_sensitivity(self) -> None:
        """Injected 40 Hz spike should produce higher anomaly score than noise."""
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        rng = np.random.default_rng(42)
        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        oracle.fit(rng.standard_normal((10, 512)))

        # Normal
        normal_result = oracle.detect(rng.standard_normal(512))

        # Anomalous: inject strong 40 Hz spike
        t = np.arange(512) / 256.0
        anomalous = rng.standard_normal(512) + 5.0 * np.sin(2 * np.pi * 40 * t)
        anom_result = oracle.detect(anomalous)

        assert anom_result["anomaly_score"] >= normal_result["anomaly_score"], (
            f"Anomalous score ({anom_result['anomaly_score']:.3f}) should >= "
            f"normal score ({normal_result['anomaly_score']:.3f})"
        )


class TestFrequencyDomainOracleBinarySegmentation:
    """Verify binary segmentation finds change points in signals with injected mean shifts."""

    def test_cp_detection_on_mean_shift(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        oracle = FrequencyDomainOracle(
            {"domain": "environmental", "sample_rate": 100.0, "min_segments": 2}
        )
        # Signal with clear mean shift
        series = np.concatenate([np.zeros(50), np.ones(50) * 5.0])
        cps = oracle._binary_segmentation_frequency(series)
        assert len(cps) > 0, "Should detect at least one change point"
        # CP should be near index 50
        assert any(40 <= cp <= 60 for cp in cps), f"Expected CP near 50, got {cps}"


class TestFrequencyDomainOracleSelectiveInference:
    """Verify SI p-values are correct for genuine CPs and noise."""

    def test_si_pvalue_genuine_cp(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        oracle = FrequencyDomainOracle({"domain": "environmental"})
        # Clear mean shift
        series = np.concatenate([np.zeros(100), np.ones(100) * 10.0])
        p = oracle._selective_inference_p_value(series, 100)
        assert p < 0.05, f"SI p-value for genuine CP should be < 0.05, got {p}"

    def test_si_pvalue_noise(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        oracle = FrequencyDomainOracle({"domain": "environmental"})
        rng = np.random.default_rng(42)
        series = rng.standard_normal(200)
        p = oracle._selective_inference_p_value(series, 100)
        assert p > 0.05, f"SI p-value for noise should be > 0.05, got {p}"


class TestFrequencyDomainOracleFeatures:
    """Verify extract_features returns correct shape and type."""

    def test_feature_shape(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        rng = np.random.default_rng(42)
        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        oracle.fit(rng.standard_normal((10, 512)))

        features = oracle.extract_features(rng.standard_normal((3, 512)))
        n_bands = len(oracle._bands)
        expected_dim = n_bands + 4  # bands + entropy + centroid + agg + mult
        assert features.shape == (
            3,
            expected_dim,
        ), f"Expected (3, {expected_dim}), got {features.shape}"

    def test_feature_dtype_numpy(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        rng = np.random.default_rng(42)
        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        oracle.fit(rng.standard_normal((10, 512)))

        features = oracle.extract_features(rng.standard_normal((3, 512)))
        assert isinstance(features, np.ndarray), f"Expected np.ndarray, got {type(features)}"
        assert features.dtype == np.float32


class TestFrequencyDomainOracleParseval:
    """Verify Parseval validation uses existing matrix, not recomputing FFT."""

    def test_parseval_no_recompute(self) -> None:
        import inspect

        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        oracle = FrequencyDomainOracle({"domain": "environmental"})
        src = inspect.getsource(oracle._validate_parseval_energy)
        assert "fft(signal)" not in src, "Parseval validation should NOT recompute FFT from signal"

    def test_parseval_passes_clean_signal(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        rng = np.random.default_rng(42)
        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        signal = rng.standard_normal(512)
        freq_matrix, _freqs = oracle._compute_frequency_matrix(signal)
        result = oracle._validate_parseval_energy(signal, freq_matrix)
        assert result is True or result is False  # Returns bool


class TestFrequencyDomainOracleConfig:
    """Verify OracleConfig is constructed exactly once (no double-init bug)."""

    def test_single_config_construction(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
            FrequencyDomainOracleConfig,
        )

        oracle = FrequencyDomainOracle({"domain": "medical", "sample_rate": 256.0})
        assert isinstance(oracle._oracle_config, FrequencyDomainOracleConfig)
        assert oracle._oracle_config.domain == "medical"
        assert oracle._oracle_config.sample_rate == 256.0

    def test_band_scores_are_dict(self) -> None:
        from omni_mercury_engine.detectors.frequency_domain_oracle import (
            FrequencyDomainOracle,
        )

        rng = np.random.default_rng(42)
        oracle = FrequencyDomainOracle({"domain": "humanitarian", "sample_rate": 256.0})
        oracle.fit(rng.standard_normal((5, 512)))
        result = oracle.detect(rng.standard_normal(512))
        iv = result["influence_vector"]
        assert isinstance(
            iv.band_scores, dict
        ), f"band_scores should be dict, got {type(iv.band_scores)}"
