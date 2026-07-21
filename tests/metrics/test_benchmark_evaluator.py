# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the benchmark evaluation framework.

Covers:
- ``EvaluationResult``: ``to_dict`` shape, JSON save/load round trip with
  parent-directory creation, ``__str__`` formatting, default timestamp,
  and independent default-factory containers
- ``BenchmarkEvaluator.evaluate`` on hand-computable constructed datasets:
  exact AUROC/AUPRC/F1-max values verified against pen-and-paper results,
  score extraction from plain floats, ndarray means, and torch-like duck-typed
  tensors, image vs. video sample handling, skipping of unusable samples,
  detector-failure resilience, mask/anomaly-map pixel metrics, per-category
  metrics, prediction persistence (``.npz``), progress logging, and both
  ``tune_on`` policies (val-split and the small-data in-sample fallback)
- The documented sharp edge that an all-skipped dataset raises ``ValueError``
  from the underlying metric reductions
- ``BenchmarkEvaluator.compare``: table layout, N/A cells for missing
  detector/dataset combos, missing-metric default of 0.0, best/mean summary
- ``BenchmarkEvaluator.generate_report``: markdown structure and metric rows
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.metrics.benchmark_evaluator import BenchmarkEvaluator, EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

EVALUATOR_LOGGER = "omni_mercury_engine.metrics.benchmark_evaluator"
METRICS_LOGGER = "omni_mercury_engine.metrics.anomaly_metrics"


# =============================================================================
# Test doubles for the detector/dataset collaborators (NOT the unit under test)
# =============================================================================


class _FakeImage:
    """Tensor stand-in carrying the anomaly score the detector should emit."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.batched = False

    def unsqueeze(self, dim: int) -> _FakeImage:
        """Mimic ``torch.Tensor.unsqueeze`` — returns a batched copy."""
        assert dim == 0
        batched = _FakeImage(self.score)
        batched.batched = True
        return batched


class _FakeVideo:
    """Video-clip stand-in.  Deliberately has NO ``unsqueeze`` method: the
    evaluator contract passes videos through unbatched, so calling
    ``unsqueeze`` on it would raise ``AttributeError`` and fail the test."""

    def __init__(self, score: float) -> None:
        self.score = score


class _FakeMask:
    """Ground-truth mask stand-in exposing torch's ``.numpy()`` accessor."""

    def __init__(self, mask: NDArray[np.float64]) -> None:
        self._mask = mask

    def numpy(self) -> NDArray[np.float64]:
        return self._mask


class _FakeTensorScalar:
    """Result of ``tensor.mean()`` — exposes ``.item()``."""

    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeTensorScore:
    """Torch-tensor duck type: has ``cpu`` and ``mean().item()``, matching the
    hasattr-based dispatch the evaluator documents for score extraction."""

    def __init__(self, value: float) -> None:
        self._value = value

    def cpu(self) -> _FakeTensorScore:
        return self

    def mean(self) -> _FakeTensorScalar:
        return _FakeTensorScalar(self._value)


class _ScoreDetector:
    """Emits the score planted on each sample's tensor stand-in."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"kind": "planted-score"}
        self.seen: list[Any] = []

    def detect(self, data: Any) -> dict[str, Any]:
        self.seen.append(data)
        return {"score": float(data.score)}


class _NoConfigDetector:
    """Detector without a ``config`` attribute (tests the getattr default)."""

    def detect(self, data: Any) -> dict[str, Any]:
        return {"score": float(data.score)}


class _ArrayScoreDetector:
    """Returns a ``scores`` ndarray whose mean is the planted score."""

    def detect(self, data: Any) -> dict[str, Any]:
        planted = float(data.score)
        return {"scores": np.array([planted - 0.05, planted + 0.05])}


class _TensorScoreDetector:
    """Returns a torch-like tensor score to exercise the ``cpu`` duck path."""

    def detect(self, data: Any) -> dict[str, Any]:
        return {"score": _FakeTensorScore(float(data.score))}


class _FlakyDetector:
    """Raises for samples whose planted score is negative."""

    def detect(self, data: Any) -> dict[str, Any]:
        score = float(data.score)
        if score < 0:
            raise RuntimeError("sensor exploded")
        return {"score": score}


class _MaskDetector:
    """Returns both a score and an anomaly map identical to the true mask."""

    def __init__(self) -> None:
        self.maps: dict[float, NDArray[np.float64]] = {}

    def detect(self, data: Any) -> dict[str, Any]:
        score = float(data.score)
        return {"score": score, "anomaly_maps": self.maps[score]}


def _image_dataset(
    labels: Sequence[int],
    scores: Sequence[float],
    categories: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a list-style dataset of image samples with planted scores."""
    samples: list[dict[str, Any]] = []
    for i, (label, score) in enumerate(zip(labels, scores, strict=True)):
        sample: dict[str, Any] = {"image": _FakeImage(score), "label": label}
        if categories is not None:
            sample["category"] = categories[i]
        samples.append(sample)
    return samples


def _evaluator(tmp_path: Path, **kwargs: Any) -> BenchmarkEvaluator:
    """Evaluator writing under tmp_path; in-sample tuning for exact values."""
    kwargs.setdefault("tune_on", "in_sample")
    return BenchmarkEvaluator(output_dir=tmp_path / "results", **kwargs)


def _result(
    detector: str,
    dataset: str,
    metrics: dict[str, float],
) -> EvaluationResult:
    """Shorthand for a hand-built EvaluationResult."""
    return EvaluationResult(detector_name=detector, dataset_name=dataset, metrics=metrics)


# Hand-computed reference dataset (see assertions for the arithmetic):
# labels [0, 0, 1, 1, 0, 1], scores [0.1, 0.4, 0.35, 0.8, 0.2, 0.7]
MIXED_LABELS = [0, 0, 1, 1, 0, 1]
MIXED_SCORES = [0.1, 0.4, 0.35, 0.8, 0.2, 0.7]

PERFECT_LABELS = [0, 0, 0, 1, 1, 1]
PERFECT_SCORES = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]


# =============================================================================
# EvaluationResult
# =============================================================================


class TestEvaluationResult:
    """Serialization, persistence, and string formatting."""

    def test_to_dict_carries_all_fields(self) -> None:
        result = EvaluationResult(
            detector_name="det",
            dataset_name="ds",
            metrics={"auroc": 0.9},
            per_category={"cat": {"auroc": 0.8}},
            timestamp="2026-07-21T00:00:00",
            config={"lr": 0.01},
        )
        assert result.to_dict() == {
            "detector_name": "det",
            "dataset_name": "ds",
            "metrics": {"auroc": 0.9},
            "per_category": {"cat": {"auroc": 0.8}},
            "timestamp": "2026-07-21T00:00:00",
            "config": {"lr": 0.01},
        }

    def test_save_creates_parent_dirs_and_load_round_trips(self, tmp_path: Path) -> None:
        result = _result("det", "ds", {"auroc": 0.75, "auprc": 0.5})
        path = tmp_path / "deep" / "nested" / "result.json"
        result.save(path)
        assert path.is_file()

        loaded = EvaluationResult.load(path)
        assert loaded.detector_name == "det"
        assert loaded.dataset_name == "ds"
        assert loaded.metrics == {"auroc": 0.75, "auprc": 0.5}
        assert loaded.timestamp == result.timestamp
        assert loaded.to_dict() == result.to_dict()

    def test_str_lists_metrics_sorted_with_four_decimals(self) -> None:
        result = _result("MyDet", "MyData", {"zeta": 0.5, "auroc": 8 / 9})
        text = str(result)
        assert "Evaluation: MyDet on MyData" in text
        assert "auroc: 0.8889" in text
        assert "zeta: 0.5000" in text
        assert text.index("auroc") < text.index("zeta")

    def test_default_timestamp_is_isoformat(self) -> None:
        result = _result("d", "s", {})
        # Raises ValueError if not a valid ISO-8601 timestamp
        datetime.fromisoformat(result.timestamp)

    def test_default_factory_containers_are_independent(self) -> None:
        first = _result("d", "s", {})
        second = _result("d", "s", {})
        first.per_category["cat"] = {"auroc": 1.0}
        first.config["k"] = "v"
        assert second.per_category == {}
        assert second.config == {}


# =============================================================================
# BenchmarkEvaluator — construction
# =============================================================================


class TestEvaluatorInit:
    """Constructor side effects and stored policy."""

    def test_creates_nested_output_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        evaluator = BenchmarkEvaluator(output_dir=target)
        assert target.is_dir()
        assert evaluator.output_dir == target

    def test_stores_policy_flags(self, tmp_path: Path) -> None:
        evaluator = BenchmarkEvaluator(
            output_dir=tmp_path, save_predictions=True, tune_on="in_sample"
        )
        assert evaluator.save_predictions is True
        assert evaluator.tune_on == "in_sample"

    def test_default_policy_is_val_tuning_without_prediction_dumps(self, tmp_path: Path) -> None:
        evaluator = BenchmarkEvaluator(output_dir=tmp_path)
        assert evaluator.save_predictions is False
        assert evaluator.tune_on == "val"


# =============================================================================
# BenchmarkEvaluator.evaluate — metric correctness
# =============================================================================


class TestEvaluateMetrics:
    """Hand-computable metric values on tiny constructed datasets."""

    def test_perfectly_separable_dataset_scores_all_ones(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(), _image_dataset(PERFECT_LABELS, PERFECT_SCORES)
        )
        assert result.metrics["auroc"] == 1.0
        assert result.metrics["auprc"] == 1.0
        assert result.metrics["f1_max"] == 1.0
        # Any threshold in (0.3, 0.7] separates the classes perfectly.
        assert 0.3 < result.metrics["optimal_threshold"] <= 0.7

    def test_hand_computed_mixed_dataset(self, tmp_path: Path) -> None:
        # Positives score {0.35, 0.8, 0.7}, negatives {0.1, 0.4, 0.2}.
        # AUROC: of the 9 (pos, neg) pairs only (0.35 vs 0.4) is misordered
        #   => 8/9.
        # AUPRC (step-wise AP): descending scores 0.8+, 0.7+, 0.4-, 0.35+,
        #   precisions at hits 1/1, 2/2, 3/4, each covering recall 1/3
        #   => (1 + 1 + 0.75) / 3 = 11/12.
        # F1-max: threshold in (0.2, 0.35] predicts {0.8, 0.7, 0.4, 0.35}
        #   => tp=3, fp=1, fn=0 => precision 3/4, recall 1 => F1 = 6/7.
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(), _image_dataset(MIXED_LABELS, MIXED_SCORES)
        )
        assert result.metrics["auroc"] == pytest.approx(8 / 9)
        assert result.metrics["auprc"] == pytest.approx(11 / 12)
        assert result.metrics["f1_max"] == pytest.approx(6 / 7)
        assert 0.2 < result.metrics["optimal_threshold"] <= 0.35

    def test_inverted_detector_scores_zero_auroc(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(),
            _image_dataset(PERFECT_LABELS, list(reversed(PERFECT_SCORES))),
        )
        assert result.metrics["auroc"] == 0.0


class TestEvaluateNamesAndPersistence:
    """Naming defaults, config capture, and result-file persistence."""

    def test_names_default_to_class_names(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(), _image_dataset(PERFECT_LABELS, PERFECT_SCORES)
        )
        assert result.detector_name == "_ScoreDetector"
        assert result.dataset_name == "list"

    def test_explicit_names_override_defaults(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(),
            _image_dataset(PERFECT_LABELS, PERFECT_SCORES),
            detector_name="MercuryDet",
            dataset_name="TinyBench",
        )
        assert result.detector_name == "MercuryDet"
        assert result.dataset_name == "TinyBench"

    def test_detector_config_captured(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(), _image_dataset(PERFECT_LABELS, PERFECT_SCORES)
        )
        assert result.config == {"kind": "planted-score"}

    def test_missing_detector_config_defaults_to_empty(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _NoConfigDetector(), _image_dataset(PERFECT_LABELS, PERFECT_SCORES)
        )
        assert result.config == {}

    def test_result_json_written_and_loadable(self, tmp_path: Path) -> None:
        evaluator = _evaluator(tmp_path)
        result = evaluator.evaluate(
            _ScoreDetector(),
            _image_dataset(MIXED_LABELS, MIXED_SCORES),
            detector_name="det",
            dataset_name="ds",
        )
        saved = list(evaluator.output_dir.glob("det_ds_*.json"))
        assert len(saved) == 1
        loaded = EvaluationResult.load(saved[0])
        assert loaded.metrics == result.metrics

    def test_save_predictions_writes_npz(self, tmp_path: Path) -> None:
        evaluator = _evaluator(tmp_path, save_predictions=True)
        evaluator.evaluate(
            _ScoreDetector(),
            _image_dataset(MIXED_LABELS, MIXED_SCORES, categories=["a"] * 6),
        )
        npz_files = list(evaluator.output_dir.glob("*.npz"))
        assert len(npz_files) == 1
        with np.load(npz_files[0]) as data:
            assert data["scores"].tolist() == MIXED_SCORES
            assert data["labels"].tolist() == MIXED_LABELS
            assert data["categories"].tolist() == ["a"] * 6

    def test_predictions_not_written_by_default(self, tmp_path: Path) -> None:
        evaluator = _evaluator(tmp_path)
        evaluator.evaluate(_ScoreDetector(), _image_dataset(PERFECT_LABELS, PERFECT_SCORES))
        assert list(evaluator.output_dir.glob("*.npz")) == []


class TestEvaluateSampleHandling:
    """Score extraction paths, sample skipping, and failure resilience."""

    def test_images_are_batched_before_detection(self, tmp_path: Path) -> None:
        detector = _ScoreDetector()
        _evaluator(tmp_path).evaluate(detector, _image_dataset(PERFECT_LABELS, PERFECT_SCORES))
        assert len(detector.seen) == 6
        assert all(image.batched for image in detector.seen)

    def test_video_samples_passed_through_unbatched(self, tmp_path: Path) -> None:
        videos: list[dict[str, Any]] = [
            {"video": _FakeVideo(score), "label": label}
            for label, score in zip(PERFECT_LABELS, PERFECT_SCORES, strict=True)
        ]
        detector = _ScoreDetector()
        result = _evaluator(tmp_path).evaluate(detector, videos)
        # The exact _FakeVideo instances reach the detector (no unsqueeze).
        assert detector.seen == [sample["video"] for sample in videos]
        assert result.metrics["auroc"] == 1.0

    def test_ndarray_scores_reduced_by_mean(self, tmp_path: Path) -> None:
        evaluator = _evaluator(tmp_path, save_predictions=True)
        evaluator.evaluate(_ArrayScoreDetector(), _image_dataset(PERFECT_LABELS, PERFECT_SCORES))
        with np.load(next(evaluator.output_dir.glob("*.npz"))) as data:
            np.testing.assert_allclose(data["scores"], PERFECT_SCORES)

    def test_torch_like_tensor_scores_extracted_via_duck_typing(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _TensorScoreDetector(), _image_dataset(MIXED_LABELS, MIXED_SCORES)
        )
        assert result.metrics["auroc"] == pytest.approx(8 / 9)

    def test_samples_without_image_or_video_are_skipped(self, tmp_path: Path) -> None:
        dataset = _image_dataset(PERFECT_LABELS, PERFECT_SCORES)
        dataset.insert(2, {"label": 1, "text": "not a visual sample"})
        evaluator = _evaluator(tmp_path, save_predictions=True)
        result = evaluator.evaluate(_ScoreDetector(), dataset)
        with np.load(next(evaluator.output_dir.glob("*.npz"))) as data:
            assert data["scores"].tolist() == PERFECT_SCORES
        assert result.metrics["auroc"] == 1.0

    def test_detector_failure_skips_sample_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        labels = [0, 0, 0, 1, 1, 1, 0, 1]
        scores = [0.1, -1.0, 0.2, 0.9, -1.0, 0.8, 0.3, 0.7]
        evaluator = _evaluator(tmp_path, save_predictions=True)
        with caplog.at_level(logging.WARNING, logger=EVALUATOR_LOGGER):
            result = evaluator.evaluate(_FlakyDetector(), _image_dataset(labels, scores))
        assert "Detection failed for sample 1" in caplog.text
        assert "Detection failed for sample 4" in caplog.text
        with np.load(next(evaluator.output_dir.glob("*.npz"))) as data:
            assert data["scores"].tolist() == [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
            assert data["labels"].tolist() == [0, 0, 1, 1, 0, 1]
        assert result.metrics["auroc"] == 1.0

    def test_progress_logged_every_100_samples(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        labels = [0] * 50 + [1] * 50
        scores = [float(i) / 200 for i in range(50)] + [0.5 + i / 200 for i in range(50)]
        with caplog.at_level(logging.INFO, logger=EVALUATOR_LOGGER):
            result = _evaluator(tmp_path).evaluate(_ScoreDetector(), _image_dataset(labels, scores))
        assert "Processed 100/100 samples" in caplog.text
        assert result.metrics["auroc"] == 1.0

    def test_all_samples_skipped_raises_value_error(self, tmp_path: Path) -> None:
        # Documented sharp edge: with zero usable samples the empty score
        # array reaches numpy reductions inside AnomalyMetrics and raises
        # ValueError ("zero-size array to reduction operation ...") rather
        # than a domain-specific error.  Pinned so a future change to a
        # clearer exception is a conscious contract change.
        dataset: list[dict[str, Any]] = [{"label": 0, "text": "no visuals"}] * 4
        with pytest.raises(ValueError, match="zero-size"):
            _evaluator(tmp_path).evaluate(_ScoreDetector(), dataset)


class TestEvaluateMasksAndCategories:
    """Pixel-level metrics and per-category breakdowns."""

    def test_masks_with_anomaly_maps_yield_perfect_pixel_metrics(self, tmp_path: Path) -> None:
        anomalous_mask = np.array([[1.0, 0.0], [0.0, 0.0]])
        normal_mask = np.zeros((2, 2))
        detector = _MaskDetector()
        dataset: list[dict[str, Any]] = []
        for label, score in zip([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], strict=True):
            mask = anomalous_mask if label == 1 else normal_mask
            detector.maps[score] = mask  # predicted map == ground truth
            dataset.append({"image": _FakeImage(score), "label": label, "mask": _FakeMask(mask)})

        result = _evaluator(tmp_path).evaluate(detector, dataset)
        assert result.metrics["pixel_auroc"] == 1.0
        assert result.metrics["pro"] == 1.0
        assert result.metrics["auroc"] == 1.0

    def test_masks_ignored_when_detector_has_no_anomaly_maps(self, tmp_path: Path) -> None:
        dataset = _image_dataset(PERFECT_LABELS, PERFECT_SCORES)
        for sample in dataset:
            sample["mask"] = _FakeMask(np.zeros((2, 2)))
        result = _evaluator(tmp_path).evaluate(_ScoreDetector(), dataset)
        assert "pixel_auroc" not in result.metrics
        assert "pro" not in result.metrics

    def test_per_category_metrics_computed_for_multiple_categories(self, tmp_path: Path) -> None:
        # "glass" is perfectly ranked, "steel" is perfectly inverted, and
        # "empty" is single-class (flagged, not scored).
        labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        scores = [0.1, 0.2, 0.8, 0.9, 0.8, 0.9, 0.1, 0.2, 0.4, 0.5]
        categories = ["glass"] * 4 + ["steel"] * 4 + ["empty"] * 2
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(), _image_dataset(labels, scores, categories=categories)
        )
        assert set(result.per_category) == {"glass", "steel", "empty"}
        assert result.per_category["glass"]["auroc"] == 1.0
        assert result.per_category["steel"]["auroc"] == 0.0
        assert result.per_category["empty"] == {"auroc": 0.5, "note": "single_class"}

    def test_single_category_yields_empty_per_category(self, tmp_path: Path) -> None:
        result = _evaluator(tmp_path).evaluate(
            _ScoreDetector(),
            _image_dataset(PERFECT_LABELS, PERFECT_SCORES, categories=["only"] * 6),
        )
        assert result.per_category == {}

    def test_missing_category_defaults_to_default(self, tmp_path: Path) -> None:
        evaluator = _evaluator(tmp_path, save_predictions=True)
        evaluator.evaluate(_ScoreDetector(), _image_dataset(PERFECT_LABELS, PERFECT_SCORES))
        with np.load(next(evaluator.output_dir.glob("*.npz"))) as data:
            assert data["categories"].tolist() == ["default"] * 6


class TestEvaluateTuneOnPolicies:
    """The tune_on policy is forwarded to AnomalyMetrics.compute_all."""

    def test_val_policy_falls_back_in_sample_on_tiny_data(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # At N=4 the stratified val split cannot hold two classes, so the
        # "val" policy must fall back to in-sample metrics with a warning.
        evaluator = BenchmarkEvaluator(output_dir=tmp_path / "results")  # tune_on="val"
        with caplog.at_level(logging.WARNING, logger=METRICS_LOGGER):
            result = evaluator.evaluate(
                _ScoreDetector(), _image_dataset([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
            )
        assert "split infeasible" in caplog.text
        # Fallback reports the in-sample metric set (no operating-point "f1")
        assert result.metrics["auroc"] == 1.0
        assert "f1" not in result.metrics
        assert "f1_max" in result.metrics

    def test_val_policy_splits_when_data_is_large_enough(self, tmp_path: Path) -> None:
        labels = [0] * 30 + [1] * 30
        scores = list(np.linspace(0.0, 0.4, 30)) + list(np.linspace(0.6, 1.0, 30))
        evaluator = BenchmarkEvaluator(output_dir=tmp_path / "results")
        result = evaluator.evaluate(_ScoreDetector(), _image_dataset(labels, scores))
        # Perfect separation survives any split; the val path additionally
        # exposes the transparent operating-point "f1" alias.
        assert result.metrics["auroc"] == 1.0
        assert result.metrics["f1"] == 1.0
        assert result.metrics["f1_max"] == result.metrics["f1"]
        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["precision"] == 1.0
        assert result.metrics["recall"] == 1.0


# =============================================================================
# compare / generate_report
# =============================================================================


class TestCompare:
    """Comparison-table rendering."""

    def test_empty_results_message(self, tmp_path: Path) -> None:
        assert _evaluator(tmp_path).compare([]) == "No results to compare"

    def test_full_grid_table_with_summary(self, tmp_path: Path) -> None:
        results = [
            _result("alpha", "ds_a", {"auroc": 0.9}),
            _result("alpha", "ds_b", {"auroc": 0.8}),
            _result("beta", "ds_a", {"auroc": 0.7}),
            _result("beta", "ds_b", {"auroc": 0.6}),
        ]
        table = _evaluator(tmp_path).compare(results)
        assert "ds_a" in table
        assert "ds_b" in table
        for value in ("0.9000", "0.8000", "0.7000", "0.6000"):
            assert value in table
        assert "Best auroc: 0.9000" in table
        assert "Mean auroc: 0.7500" in table
        # Detectors are listed sorted
        assert table.index("alpha") < table.index("beta")

    def test_missing_combo_rendered_as_na(self, tmp_path: Path) -> None:
        results = [
            _result("alpha", "ds_a", {"auroc": 0.9}),
            _result("alpha", "ds_b", {"auroc": 0.8}),
            _result("beta", "ds_a", {"auroc": 0.7}),
        ]
        table = _evaluator(tmp_path).compare(results)
        assert "N/A" in table
        assert "Best auroc: 0.9000" in table
        assert "Mean auroc: 0.8000" in table

    def test_missing_metric_defaults_to_zero(self, tmp_path: Path) -> None:
        results = [
            _result("alpha", "ds_a", {"f1": 0.5}),
            _result("beta", "ds_a", {"auroc": 0.9}),  # no "f1"
        ]
        table = _evaluator(tmp_path).compare(results, metric="f1")
        assert "0.5000" in table
        assert "0.0000" in table
        assert "Best f1: 0.5000" in table
        assert "Mean f1: 0.2500" in table


class TestGenerateReport:
    """Markdown report structure."""

    def test_report_contains_summary_comparison_and_tables(self, tmp_path: Path) -> None:
        results = [
            _result("alpha", "ds_a", {"auroc": 8 / 9, "f1_max": 6 / 7}),
            _result("beta", "ds_a", {"auroc": 0.75}),
            _result("alpha", "ds_b", {"auroc": 1.0}),
        ]
        report_path = tmp_path / "report.md"
        _evaluator(tmp_path).generate_report(results, report_path)

        text = report_path.read_text()
        assert "# Anomaly Detection Evaluation Report" in text
        assert "- Detectors evaluated: 2" in text
        assert "- Datasets used: 2" in text
        assert "- Total evaluations: 3" in text
        assert "## Overall Comparison" in text
        assert "### ds_a" in text
        assert "### ds_b" in text
        assert "#### alpha" in text
        assert "#### beta" in text
        assert "| auroc | 0.8889 |" in text
        assert "| f1_max | 0.8571 |" in text
        assert "| auroc | 0.7500 |" in text
