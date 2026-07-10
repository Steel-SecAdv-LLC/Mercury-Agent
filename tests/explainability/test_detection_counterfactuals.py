# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for detection-path counterfactual reasoning.

Covers the core :func:`explain_detection_counterfactual` API (every method,
both flip directions, determinism), the verified greedy-minimality and
re-scored-correctness properties, and the three production adapters
(statistical detector, tier ensemble, symbolic rule module) on real
detections.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.explainability.detection_counterfactuals import (  # type: ignore[import-not-found,unused-ignore]
    DETECTION_COUNTERFACTUAL_METHODS,
    DetectionCounterfactual,
    explain_detection_counterfactual,
    make_statistical_score_fn,
    make_symbolic_score_fn,
    make_tier_score_fn,
)

# ---------------------------------------------------------------------------
# Analytic score functions (exact ground truth for correctness / minimality)
# ---------------------------------------------------------------------------


def _linear_score(x: np.ndarray) -> np.ndarray:
    """Score in (0, 1): anomalous iff x0 + 0.5*x1 > 1 (features 2+ irrelevant)."""
    z = 2.0 * (x[:, 0] + 0.5 * x[:, 1] - 1.0)
    return 1.0 / (1.0 + np.exp(-z))


def _single_feature_score(x: np.ndarray) -> np.ndarray:
    """Score depends ONLY on feature 0: anomalous iff x0 > 1."""
    z = 3.0 * (x[:, 0] - 1.0)
    return 1.0 / (1.0 + np.exp(-z))


class TestCoreAPI:
    def test_flips_flagged_point_to_normal(self) -> None:
        x = np.array([2.0, 1.0, 0.3, -0.7])
        result = explain_detection_counterfactual(_linear_score, x, 0.5, method="wachter", seed=0)
        assert isinstance(result, DetectionCounterfactual)
        assert result.score_before > 0.5
        assert result.flipped is True
        assert result.score_after <= 0.5
        # Re-score independently: the reported score must be the real one.
        rescored = float(_linear_score(result.counterfactual_x.reshape(1, -1))[0])
        assert rescored == pytest.approx(result.score_after)
        assert rescored <= 0.5

    def test_flips_normal_point_to_flagged(self) -> None:
        x = np.array([-1.0, 0.0, 2.0])
        result = explain_detection_counterfactual(_linear_score, x, 0.5, method="wachter", seed=0)
        assert result.score_before <= 0.5
        assert result.flipped is True
        assert result.score_after > 0.5

    @pytest.mark.parametrize("method", ["wachter", "dice", "growing_spheres"])
    def test_every_searchless_method_flips_and_is_minimal(self, method: str) -> None:
        x = np.array([2.0, 1.0, 0.3, -0.7])
        result = explain_detection_counterfactual(_linear_score, x, 0.5, method=method, seed=0)
        assert result.flipped is True, f"{method} failed to flip"
        assert result.minimal is True
        assert result.method == method
        # Irrelevant features (2, 3) must have been reverted by minimization.
        changed_names = {c.name for c in result.changed_features}
        assert changed_names <= {"feature_0", "feature_1"}

    def test_prototype_method_flips(self) -> None:
        rng = np.random.default_rng(0)
        normals = rng.normal(0.0, 0.3, size=(40, 3))
        anomalies = rng.normal(0.0, 0.3, size=(10, 3))
        anomalies[:, 0] += 3.0
        data = np.vstack([normals, anomalies])
        labels = np.array([0] * 40 + [1] * 10)
        x = np.array([2.5, 0.1, -0.2])
        result = explain_detection_counterfactual(
            _single_feature_score,
            x,
            0.5,
            method="prototype",
            training_data=data,
            training_labels=labels,
            seed=0,
        )
        assert result.flipped is True
        assert result.minimal is True

    def test_changed_features_records_old_new_delta(self) -> None:
        x = np.array([2.0, 1.0, 0.3])
        result = explain_detection_counterfactual(_linear_score, x, 0.5, seed=0)
        assert result.flipped
        assert result.sparsity == len(result.changed_features) > 0
        for change in result.changed_features:
            idx = result.feature_names.index(change.name)
            assert change.old == pytest.approx(float(x[idx]))
            assert change.new == pytest.approx(float(result.counterfactual_x[idx]))
            assert change.delta == pytest.approx(change.new - change.old)

    def test_feature_names_are_threaded(self) -> None:
        x = np.array([2.0, 1.0])
        result = explain_detection_counterfactual(
            _linear_score, x, 0.5, feature_names=["cpu", "mem"], seed=0
        )
        assert result.feature_names == ["cpu", "mem"]
        assert {c.name for c in result.changed_features} <= {"cpu", "mem"}

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        x = np.array([2.0, 1.0, 0.0])
        result = explain_detection_counterfactual(_linear_score, x, 0.5, seed=0)
        payload = json.loads(json.dumps(result.to_dict()))
        assert set(payload) >= {
            "original_x",
            "counterfactual_x",
            "changed_features",
            "score_before",
            "score_after",
            "threshold",
            "flipped",
            "sparsity",
            "distance",
            "method",
            "minimal",
        }

    def test_deterministic_same_seed_same_counterfactual(self) -> None:
        x = np.array([2.0, 1.0, 0.3, -0.7])
        a = explain_detection_counterfactual(_linear_score, x, 0.5, method="wachter", seed=7)
        b = explain_detection_counterfactual(_linear_score, x, 0.5, method="wachter", seed=7)
        assert np.array_equal(a.counterfactual_x, b.counterfactual_x)
        assert a.score_after == b.score_after
        assert a.sparsity == b.sparsity

    def test_unflippable_score_reports_honest_failure(self) -> None:
        def constant(x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], 0.9)

        x = np.array([1.0, 2.0])
        result = explain_detection_counterfactual(
            constant, x, 0.5, method="growing_spheres", seed=0, max_iterations=3, n_samples=20
        )
        assert result.flipped is False
        assert result.minimal is False


class TestMinimalityProperty:
    def test_only_the_causal_feature_survives_minimization(self) -> None:
        """Score depends only on x0 => a verified-minimal CF changes only x0."""
        x = np.array([2.0, 5.0, -3.0, 0.5])
        for method in ("wachter", "dice", "growing_spheres"):
            result = explain_detection_counterfactual(
                _single_feature_score, x, 0.5, method=method, seed=1
            )
            assert result.flipped is True, method
            assert result.minimal is True, method
            assert [c.name for c in result.changed_features] == ["feature_0"], method
            assert result.sparsity == 1, method

    def test_minimality_is_verified_by_rescoring(self) -> None:
        """Reverting any single changed feature of a minimal CF must unflip it."""
        x = np.array([2.0, 1.5, 0.0])

        def score(m: np.ndarray) -> np.ndarray:
            # Anomalous iff x0 > 1 AND x1 > 1 (both features jointly causal).
            z0 = 4.0 * (m[:, 0] - 1.0)
            z1 = 4.0 * (m[:, 1] - 1.0)
            s0 = 1.0 / (1.0 + np.exp(-z0))
            s1 = 1.0 / (1.0 + np.exp(-z1))
            return np.minimum(s0, s1)

        result = explain_detection_counterfactual(score, x, 0.5, method="wachter", seed=0)
        assert result.flipped and result.minimal
        for change in result.changed_features:
            idx = result.feature_names.index(change.name)
            reverted = result.counterfactual_x.copy()
            reverted[idx] = x[idx]
            rescored = float(score(reverted.reshape(1, -1))[0])
            assert (
                rescored > 0.5
            ), f"reverting {change.name} kept the flip -- counterfactual was not minimal"


class TestFailLoud:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown detection counterfactual method"):
            explain_detection_counterfactual(_linear_score, np.ones(2), 0.5, method="oracle")

    def test_prototype_without_training_data_raises(self) -> None:
        with pytest.raises(ValueError, match="prototype"):
            explain_detection_counterfactual(_linear_score, np.ones(2), 0.5, method="prototype")

    def test_non_finite_scores_raise(self) -> None:
        def bad(x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], np.nan)

        with pytest.raises(ValueError, match="non-finite"):
            explain_detection_counterfactual(bad, np.ones(2), 0.5)

    def test_misshaped_score_output_raises(self) -> None:
        def bad(x: np.ndarray) -> np.ndarray:
            return np.ones(x.shape[0] + 1)

        with pytest.raises(ValueError, match="scores for"):
            explain_detection_counterfactual(bad, np.ones(2), 0.5)

    def test_non_finite_x_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            explain_detection_counterfactual(_linear_score, np.array([1.0, np.nan]), 0.5)

    def test_mismatched_feature_names_raise(self) -> None:
        with pytest.raises(ValueError, match="feature_names length"):
            explain_detection_counterfactual(
                _linear_score, np.ones(3), 0.5, feature_names=["only_one"]
            )

    def test_method_registry_matches_module_constant(self) -> None:
        assert DETECTION_COUNTERFACTUAL_METHODS == (
            "wachter",
            "dice",
            "growing_spheres",
            "prototype",
            "genetic",
        )


# ---------------------------------------------------------------------------
# Adapters against real production scorers
# ---------------------------------------------------------------------------


class TestStatisticalAdapter:
    @pytest.fixture(scope="class")
    def fitted(self) -> tuple[Any, ...]:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.default_rng(0)
        train = rng.normal(size=(300, 6))
        context = rng.normal(size=(80, 6))
        context[5] += 6.0  # a clear injected anomaly row
        detector = MercuryAnomalyDetector()
        detector.fit(train)
        result = detector.detect(context)
        return detector, context, result

    def test_adapter_scores_match_real_detect_path(self, fitted: tuple[Any, ...]) -> None:
        detector, context, result = fitted
        score_fn = make_statistical_score_fn(detector, context, row_index=5)
        # Scoring the unchanged row must reproduce the real detection score.
        rescored = float(score_fn(context[5].reshape(1, -1))[0])
        assert rescored == pytest.approx(float(result["scores"][5]))

    def test_counterfactual_flips_real_statistical_detection(self, fitted: tuple[Any, ...]) -> None:
        detector, context, result = fitted
        assert bool(result["is_anomaly"][5]), "fixture row must be a real detection"
        threshold = float(result["threshold"])
        score_fn = make_statistical_score_fn(detector, context, row_index=5)
        cf = explain_detection_counterfactual(
            score_fn, context[5], threshold, method="wachter", seed=0
        )
        assert cf.flipped is True
        assert cf.minimal is True
        assert cf.score_before > threshold
        assert cf.score_after <= threshold
        assert 1 <= cf.sparsity <= context.shape[1]

    def test_adapter_rejects_bad_shapes(self, fitted: tuple[Any, ...]) -> None:
        detector, context, _ = fitted
        with pytest.raises(ValueError, match="row_index"):
            make_statistical_score_fn(detector, context, row_index=999)
        with pytest.raises(ValueError, match="2-D"):
            make_statistical_score_fn(detector, context[0], row_index=0)
        score_fn = make_statistical_score_fn(detector, context, row_index=5)
        with pytest.raises(ValueError, match="width"):
            score_fn(np.ones((1, 3)))


class TestTierAdapter:
    @pytest.fixture(scope="class")
    def fitted(self) -> tuple[Any, ...]:
        from omni_mercury_engine.detectors.detection_tier import (
            StreamingScoreEnsemble,
            build_tier_detectors,
        )

        rng = np.random.default_rng(0)
        series = rng.normal(0, 1, 200)
        series[100] += 8.0  # single-point spike: its value alone carries the anomaly
        labels = np.zeros(200, dtype=int)
        labels[100] = 1
        detectors = build_tier_detectors(("spectral_residual", "spot_evt", "bocpd"))
        ensemble = StreamingScoreEnsemble(detectors, method="stacking", contamination=0.05)
        ensemble.fit(series, labels=labels)
        return ensemble, series

    def test_counterfactual_flips_real_tier_detection(self, fitted: tuple[Any, ...]) -> None:
        # ``growing_spheres``: the tier's calibrated score is piecewise
        # constant in the point value (ECDF/rank calibration), so a sampling
        # search is the structurally correct method -- gradient-based Wachter
        # sees a zero gradient on the plateaus and honestly reports no flip.
        ensemble, series = fitted
        scores = np.asarray(ensemble.score(series))
        flags = np.asarray(ensemble.predict(series))
        assert flags.sum() > 0, "fixture burst must be flagged by the real tier"
        index = int(np.argmax(np.where(flags > 0, scores, -np.inf)))
        score_fn, x_window, names = make_tier_score_fn(ensemble, series, index)
        cf = explain_detection_counterfactual(
            score_fn,
            x_window,
            float(ensemble.threshold),
            feature_names=names,
            method="growing_spheres",
            seed=0,
            # Windowed feature space over a piecewise-constant calibrated
            # score: a sampling search is the structurally correct method,
            # and each evaluation is a full ensemble re-score of the series.
            n_samples=40,
            step_size=1.0,
            max_iterations=25,
        )
        assert cf.flipped is True
        assert cf.minimal is True
        # Contextual detection: the flip may legitimately require moving the
        # burst neighbors, so sparsity is bounded by the window, not by 1.
        assert 1 <= cf.sparsity <= len(names)
        assert all(name in names for name in (c.name for c in cf.changed_features))
        # Independently re-score through the real ensemble.
        modified = np.asarray(series, dtype=float).copy()
        lo = index - (len(names) - 1) // 2
        modified[lo : lo + len(names)] = np.asarray(cf.counterfactual_x, dtype=float)
        assert float(ensemble.score(modified)[index]) <= float(ensemble.threshold)

    def test_adapter_rejects_bad_index_and_width(self, fitted: tuple[Any, ...]) -> None:
        ensemble, series = fitted
        with pytest.raises(ValueError, match="out of range"):
            make_tier_score_fn(ensemble, series, index=10_000)
        score_fn, x_window, _names = make_tier_score_fn(ensemble, series, index=100)
        with pytest.raises(ValueError, match="width"):
            score_fn(np.ones((1, x_window.size + 3)))


class TestSymbolicAdapter:
    def test_counterfactual_flips_rule_consensus(self) -> None:
        torch = pytest.importorskip("torch")
        from omni_mercury_engine.explainability.counterfactuals import FeatureConstraint
        from omni_mercury_engine.ml.symbolic_constraint import SymbolicConstraintModule

        module = SymbolicConstraintModule(num_detectors=3)
        score_fn = make_symbolic_score_fn(module)
        x = np.array([0.9, 0.85, 0.95])  # strong joint detector support
        assert float(score_fn(x.reshape(1, -1))[0]) > 0.5
        constraints = [
            FeatureConstraint(name=f"det_{i}", feature_idx=i, min_value=0.0, max_value=1.0)
            for i in range(3)
        ]
        cf = explain_detection_counterfactual(
            score_fn,
            x,
            0.5,
            feature_names=["det_0", "det_1", "det_2"],
            method="wachter",
            feature_constraints=constraints,
            seed=0,
        )
        assert cf.flipped is True
        assert cf.minimal is True
        # The counterfactual must live in the valid detector-score box.
        assert np.all(cf.counterfactual_x >= -1e-9)
        assert np.all(cf.counterfactual_x <= 1.0 + 1e-9)
        # Consensus recomputed through the real module confirms the flip.
        with torch.no_grad():
            consensus = float(
                module.predict(
                    torch.as_tensor(
                        np.clip(cf.counterfactual_x, 0.0, 1.0).reshape(1, -1),
                        dtype=torch.float32,
                    )
                )[0]
            )
        assert consensus <= 0.5 + 1e-6


class TestCommittedValidationResults:
    """The committed validation results must meet the pre-registered bar.

    ``benchmarks/counterfactual_validation.py`` (real ADBench WBC
    true-positive detections, all five methods, seeded): wachter and
    genetic must hold flip-rate >= 0.9 with 100% re-scored minimality.
    dice / growing_spheres underperforming on this piecewise black-box
    regime is a recorded structural result, not a gate.
    """

    def test_results_meet_preregistered_bar(self) -> None:
        import json
        from pathlib import Path

        path = (
            Path(__file__).parent.parent.parent
            / "benchmarks"
            / "counterfactual_validation_results.json"
        )
        results = json.loads(path.read_text())
        assert results["dataset"] == "WBC"
        assert results["provenance"]["commit"]
        for method in ("wachter", "genetic", "prototype"):
            record = results["per_method"][method]
            assert record["flip_rate"] >= 0.9, method
            assert record["minimality_verified_rate"] == 1.0, method
        assert set(results["per_method"]) == {
            "wachter",
            "dice",
            "growing_spheres",
            "prototype",
            "genetic",
        }
