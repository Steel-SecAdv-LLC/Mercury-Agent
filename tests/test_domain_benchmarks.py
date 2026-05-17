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

Tests for the domain benchmark base infrastructure:
  - compute_auc (trapezoidal AUC-ROC without sklearn)
  - compute_f1_precision_recall (F1 / precision / recall without sklearn)
  - run_domain_benchmark (end-to-end with a mock loader)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmarks.domain_benchmark_base import compute_auc, compute_f1_precision_recall

# ======================================================================
# compute_auc
# ======================================================================


class TestComputeAuc:
    """Tests for the trapezoidal AUC-ROC implementation."""

    def test_perfect_separation(self) -> None:
        """All positive scores higher than all negative scores -> AUC = 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(1.0)

    def test_perfect_inverse_separation(self) -> None:
        """All positive scores lower than all negative scores -> AUC = 0.0."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(0.0)

    def test_random_scores_near_half(self) -> None:
        """Random scores should produce AUC near 0.5 for large samples."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=5000)
        y_scores = rng.rand(5000)
        auc = compute_auc(y_true, y_scores)
        assert 0.45 <= auc <= 0.55

    def test_all_same_class_positive(self) -> None:
        """If all labels are 1, AUC is undefined and returns 0.5."""
        y_true = np.array([1, 1, 1, 1])
        y_scores = np.array([0.1, 0.5, 0.7, 0.9])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(0.5)

    def test_all_same_class_negative(self) -> None:
        """If all labels are 0, AUC is undefined and returns 0.5."""
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.5, 0.7, 0.9])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(0.5)

    def test_length_mismatch_raises(self) -> None:
        """Mismatched array lengths must raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            compute_auc(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))

    def test_two_samples(self) -> None:
        """Minimal valid case: one positive and one negative sample."""
        y_true = np.array([0, 1])
        y_scores = np.array([0.2, 0.8])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(1.0)

    def test_two_samples_tied_scores(self) -> None:
        """Tied scores: positive and negative have identical score."""
        y_true = np.array([0, 1])
        y_scores = np.array([0.5, 0.5])
        auc = compute_auc(y_true, y_scores)
        # With tied scores the area depends on sort order, but must be in [0,1]
        assert 0.0 <= auc <= 1.0

    def test_auc_returns_float(self) -> None:
        """Return type must be a plain float."""
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.9, 0.2, 0.8])
        auc = compute_auc(y_true, y_scores)
        assert isinstance(auc, float)

    def test_auc_bounded(self) -> None:
        """AUC must always be in [0, 1]."""
        rng = np.random.RandomState(7)
        for _ in range(20):
            n = rng.randint(10, 200)
            y_true = rng.randint(0, 2, size=n)
            # Ensure both classes present
            if y_true.sum() == 0 or y_true.sum() == n:
                y_true[0] = 0
                y_true[1] = 1
            y_scores = rng.rand(n)
            auc = compute_auc(y_true, y_scores)
            assert 0.0 <= auc <= 1.0

    def test_auc_accepts_lists(self) -> None:
        """Function should accept plain Python lists via np.asarray."""
        auc = compute_auc(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.4, 0.6, 0.9]),
        )
        assert 0.0 <= auc <= 1.0

    def test_known_three_point_auc(self) -> None:
        """Hand-computed AUC for a small example."""
        # Labels:  [0, 1, 0, 1]
        # Scores:  [0.1, 0.4, 0.35, 0.8]
        # Sorted descending by score: indices [3, 1, 2, 0]
        # y_true_sorted = [1, 1, 0, 0]
        # tps = [1, 2, 2, 2], fps = [0, 0, 1, 2]
        # tpr = [0.5, 1.0, 1.0, 1.0], fpr = [0.0, 0.0, 0.5, 1.0]
        # With prepended (0, 0):
        #   fpr = [0, 0, 0, 0.5, 1.0]
        #   tpr = [0, 0.5, 1.0, 1.0, 1.0]
        # Trapz area = 1.0
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc = compute_auc(y_true, y_scores)
        assert auc == pytest.approx(1.0)


# ======================================================================
# compute_f1_precision_recall
# ======================================================================


class TestComputeF1PrecisionRecall:
    """Tests for F1 / precision / recall computation."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction: precision=recall=f1=1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        result = compute_f1_precision_recall(y_true, y_pred)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        """All predictions inverted: precision=0, recall=0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = compute_f1_precision_recall(y_true, y_pred)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_no_positives_predicted(self) -> None:
        """Predicting all 0 when some are 1: recall=0, precision=0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        result = compute_f1_precision_recall(y_true, y_pred)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_no_actual_positives(self) -> None:
        """No true positives exist: recall denominator is 0."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1])
        result = compute_f1_precision_recall(y_true, y_pred)
        assert result["recall"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)

    def test_known_values(self) -> None:
        """Hand-computed: tp=2, fp=1, fn=1 -> P=2/3, R=2/3, F1=2/3."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0])
        result = compute_f1_precision_recall(y_true, y_pred)
        assert result["precision"] == pytest.approx(2.0 / 3.0)
        assert result["recall"] == pytest.approx(2.0 / 3.0)
        assert result["f1"] == pytest.approx(2.0 / 3.0)

    def test_returns_dict_keys(self) -> None:
        """Result dict must contain exactly f1, precision, recall."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = compute_f1_precision_recall(y_true, y_pred)
        assert set(result.keys()) == {"f1", "precision", "recall"}

    def test_all_values_float(self) -> None:
        """All returned values must be plain floats."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = compute_f1_precision_recall(y_true, y_pred)
        for v in result.values():
            assert isinstance(v, float)

    def test_high_precision_low_recall(self) -> None:
        """Only one positive predicted, and it is correct: P=1.0, R<1.0."""
        y_true = np.array([1, 1, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 0])
        result = compute_f1_precision_recall(y_true, y_pred)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.25)

    def test_single_sample_tp(self) -> None:
        """Single sample, true positive."""
        result = compute_f1_precision_recall(np.array([1]), np.array([1]))
        assert result["f1"] == pytest.approx(1.0)

    def test_single_sample_fn(self) -> None:
        """Single sample, false negative."""
        result = compute_f1_precision_recall(np.array([1]), np.array([0]))
        assert result["f1"] == pytest.approx(0.0)


# ======================================================================
# run_domain_benchmark (with mock loader and detector)
# ======================================================================


class _MockLoader:
    """Minimal mock that satisfies the loader protocol used by run_domain_benchmark."""

    SOURCE_URL = "https://mock.example.com/data"

    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        n_samples: int = 100,
        n_features: int = 3,
    ) -> None:
        self._events = events if events is not None else [{"event_id": "mock_event_1"}]
        self._n_samples = n_samples
        self._n_features = n_features

    def list_events(self) -> list[dict[str, Any]]:
        return self._events

    def fetch_historical(self, event_id: str) -> np.ndarray:
        rng = np.random.RandomState(hash(event_id) % 2**31)
        return rng.randn(self._n_samples, self._n_features)

    def engineer_features(self, raw_data: np.ndarray) -> np.ndarray:
        return raw_data

    def get_ground_truth(self, event_id: str) -> np.ndarray:
        rng = np.random.RandomState(hash(event_id) % 2**31 + 1)
        gt = np.zeros(self._n_samples, dtype=np.int64)
        anomaly_idx = rng.choice(self._n_samples, size=self._n_samples // 10, replace=False)
        gt[anomaly_idx] = 1
        return gt

    def get_provenance(self, event_id: str, features: np.ndarray) -> dict[str, Any]:
        return {"source": "mock", "event_id": event_id}


_torch_available = False
try:
    import torch  # noqa: F401

    _torch_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _torch_available, reason="torch not available")
class TestRunDomainBenchmark:
    """Tests for the full run_domain_benchmark pipeline with mocked components."""

    @patch("omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector")
    def test_returns_results_dict(self, mock_detector_cls, tmp_path) -> None:
        """Benchmark should return a dict with domain, events, summary."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        # Set up mock detector
        mock_instance = MagicMock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.detect.return_value = {
            "scores": np.random.rand(100),
            "is_anomaly": np.random.rand(100) > 0.5,
        }
        mock_detector_cls.return_value = mock_instance

        loader = _MockLoader()
        output_path = tmp_path / "results.json"
        result = run_domain_benchmark("test_domain", loader, output_path=output_path)

        assert isinstance(result, dict)
        assert result["domain"] == "test_domain"
        assert "events" in result
        assert "summary" in result
        assert result["summary"]["status"] == "complete"
        assert output_path.exists()

    @patch("omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector")
    def test_multiple_events(self, mock_detector_cls, tmp_path) -> None:
        """Benchmark should process all events from the loader."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        mock_instance = MagicMock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.detect.return_value = {
            "scores": np.random.rand(100),
            "is_anomaly": np.random.rand(100) > 0.5,
        }
        mock_detector_cls.return_value = mock_instance

        events = [
            {"event_id": "event_a"},
            {"event_id": "event_b"},
            {"event_id": "event_c"},
        ]
        loader = _MockLoader(events=events)
        output_path = tmp_path / "results.json"
        result = run_domain_benchmark("multi", loader, output_path=output_path)

        assert result["summary"]["events_benchmarked"] == 3
        assert result["summary"]["events_attempted"] == 3
        assert "event_a" in result["events"]
        assert "event_b" in result["events"]
        assert "event_c" in result["events"]

    @patch("omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector")
    def test_event_result_has_metrics(self, mock_detector_cls, tmp_path) -> None:
        """Each successful event result should contain AUC, F1, precision, recall."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        mock_instance = MagicMock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.detect.return_value = {
            "scores": np.random.rand(100),
            "is_anomaly": np.random.rand(100) > 0.5,
        }
        mock_detector_cls.return_value = mock_instance

        loader = _MockLoader()
        output_path = tmp_path / "results.json"
        result = run_domain_benchmark("test", loader, output_path=output_path)

        event_result = result["events"]["mock_event_1"]
        assert event_result["status"] == "success"
        for key in ("auc", "f1", "precision", "recall", "n_samples"):
            assert key in event_result, f"Missing key: {key}"

    def test_no_events_exits(self, tmp_path) -> None:
        """If loader returns no events, run_domain_benchmark should sys.exit(1)."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        loader = _MockLoader(events=[])
        output_path = tmp_path / "results.json"
        with pytest.raises(SystemExit) as exc_info:
            run_domain_benchmark("empty", loader, output_path=output_path)
        assert exc_info.value.code == 1

    @patch("omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector")
    def test_summary_has_mean_auc(self, mock_detector_cls, tmp_path) -> None:
        """Summary should include aggregated metrics."""
        from benchmarks.domain_benchmark_base import run_domain_benchmark

        mock_instance = MagicMock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.detect.return_value = {
            "scores": np.linspace(0.0, 1.0, 100),
            "is_anomaly": np.array([False] * 80 + [True] * 20),
        }
        mock_detector_cls.return_value = mock_instance

        loader = _MockLoader()
        output_path = tmp_path / "results.json"
        result = run_domain_benchmark("agg", loader, output_path=output_path)

        summary = result["summary"]
        assert "mean_auc" in summary
        assert "mean_f1" in summary
        assert "min_auc" in summary
        assert "max_auc" in summary
        assert isinstance(summary["mean_auc"], float)
