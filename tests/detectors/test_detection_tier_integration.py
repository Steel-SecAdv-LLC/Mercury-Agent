# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration tests for the streaming detector tier wiring (detection_tier)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_tier import (
    STREAMING_TIER,
    TIER_PARADIGMS,
    StreamingScoreEnsemble,
    TierStreamingScorer,
    align_point_scores,
    build_tier_detectors,
    conformal_flags,
    conformal_threshold,
    rca_localize,
    store_tier_features,
)


def _labelled_series(
    seed: int = 0, n: int = 600, rate: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """A 1-D series with additive point anomalies and per-point 0/1 labels."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    labels = np.zeros(n, dtype=int)
    n_anom = max(1, int(n * rate))
    idx = rng.choice(np.arange(50, n), size=n_anom, replace=False)
    x[idx] += rng.choice([-1.0, 1.0], size=n_anom) * rng.uniform(5.0, 9.0, size=n_anom)
    labels[idx] = 1
    return x, labels


class TestTierManifest:
    def test_tier_covers_paradigms(self) -> None:
        flat = [name for group in TIER_PARADIGMS.values() for name in group]
        assert set(STREAMING_TIER) == set(flat)
        assert len(STREAMING_TIER) == len(flat)  # no duplicates

    def test_build_tier_detectors_all(self) -> None:
        built = build_tier_detectors()
        assert set(built) == set(STREAMING_TIER)
        for det in built.values():
            assert hasattr(det, "fit") and hasattr(det, "detect")

    def test_build_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            build_tier_detectors(["not_a_detector"])


class TestAlignment:
    def test_align_point_scores_length(self) -> None:
        x, _ = _labelled_series()
        det = build_tier_detectors(["spectral_residual"])["spectral_residual"].fit(x)
        scores = align_point_scores(det, x)
        assert scores.shape == (x.shape[0],)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_align_attributes_finite_guard_to_member(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression: the defence-in-depth finite guard must be labelled with the
        # member detector's name (not a generic "align"), so the
        # omni_detector_nonfinite_corrected metric/log names the misbehaving member.
        import omni_mercury_engine.detectors.detection_tier as tier

        captured: list[str] = []
        real = tier.finite_scores

        def spy(values: Any, **kwargs: Any) -> Any:
            captured.append(kwargs.get("detector", ""))
            return real(values, **kwargs)

        monkeypatch.setattr(tier, "finite_scores", spy)
        x, _ = _labelled_series()
        det = build_tier_detectors(["spectral_residual"])["spectral_residual"].fit(x)
        align_point_scores(det, x)
        assert captured, "finite_scores was not exercised by align_point_scores"
        assert all(name == det.name for name in captured)
        assert "align" not in captured


class TestEnsemble:
    _MEMBERS = ["spectral_residual", "bocpd", "gaussian_process", "particle_filter"]

    def test_stacking_calibrated_and_competitive(self) -> None:
        x_tr, y_tr = _labelled_series(1)
        x_te, y_te = _labelled_series(2)
        det = build_tier_detectors(self._MEMBERS)
        ens = StreamingScoreEnsemble(det, method="stacking").fit(x_tr, y_tr)
        proba = ens.score(x_te)
        assert proba.shape == (x_te.shape[0],)
        assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0
        # The stacked ensemble ranks true anomalies above normal points.
        assert proba[y_te == 1].mean() > proba[y_te == 0].mean() + 0.1

    def test_bma_weights_normalised_with_uncertainty(self) -> None:
        x, y = _labelled_series(3)
        ens = StreamingScoreEnsemble(build_tier_detectors(self._MEMBERS), method="bma").fit(x, y)
        weights = ens.bma_weights()
        assert set(weights) == set(self._MEMBERS)
        total = sum(w for w, _ in weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        assert all(unc >= 0.0 for _, unc in weights.values())

    def test_average_is_label_free(self) -> None:
        x, _ = _labelled_series(4)
        ens = StreamingScoreEnsemble(build_tier_detectors(self._MEMBERS), method="average").fit(x)
        proba = ens.score(x)
        assert proba.shape == (x.shape[0],)
        unc = ens.ensemble_uncertainty(x)
        assert unc.shape == (x.shape[0],)
        assert float(unc.min()) >= 0.0

    def test_stacking_requires_labels(self) -> None:
        x, _ = _labelled_series(5)
        with pytest.raises(ValueError):
            StreamingScoreEnsemble(build_tier_detectors(self._MEMBERS), method="stacking").fit(x)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            StreamingScoreEnsemble(build_tier_detectors(["bocpd"]), method="nonsense")

    def test_empty_detectors(self) -> None:
        with pytest.raises(ValueError):
            StreamingScoreEnsemble({}, method="average")

    def test_explicit_warmup_validated_like_config(self) -> None:
        # Regression: an explicit ``warmup`` constructor arg must clear the same
        # validation as ``DetectionConfig.ensemble_warmup`` (reject bool, a float
        # outside (0, 1], or an int < 2) instead of bypassing it.
        det = build_tier_detectors(["bocpd"])
        for bad in (True, 1, 1.5, 0.0, -0.2):
            with pytest.raises(ValueError):
                StreamingScoreEnsemble(det, method="average", warmup=bad)

    def test_valid_warmup_accepted(self) -> None:
        det = build_tier_detectors(["bocpd"])
        for good in (2, 50, 0.25, 1.0):
            ens = StreamingScoreEnsemble(det, method="average", warmup=good)
            assert ens.warmup == good


class TestConformal:
    def test_bounds_false_positive_rate(self) -> None:
        rng = np.random.default_rng(11)
        calib = rng.uniform(0.0, 1.0, size=4000)  # exchangeable normal scores
        test_normal = rng.uniform(0.0, 1.0, size=4000)
        alpha = 0.05
        thr = conformal_threshold(calib, alpha=alpha)
        fpr = float((test_normal > thr).mean())
        assert fpr <= alpha + 0.02  # empirical FPR bounded by alpha (+ sampling slack)

    def test_conformal_flags_shape(self) -> None:
        rng = np.random.default_rng(12)
        calib = rng.uniform(0.0, 1.0, size=1000)
        scores = rng.uniform(0.0, 1.0, size=200)
        flags = conformal_flags(scores, calib, alpha=0.1)
        assert flags.shape == (200,)
        assert flags.dtype == bool

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            conformal_threshold(np.arange(10.0), alpha=1.5)


class TestRCA:
    def test_localises_injected_root_cause(self) -> None:
        rng = np.random.default_rng(21)
        n, nodes = 400, 5
        base = rng.normal(size=(n, nodes))
        # node 0 drives nodes 1 and 2 (causal chain); adjacency 0 -> {1,2}
        base[:, 1] += 0.8 * base[:, 0]
        base[:, 2] += 0.8 * base[:, 0]
        adjacency = np.zeros((nodes, nodes))
        adjacency[0, 1] = adjacency[0, 2] = 1.0
        train = base[:350]
        obs = base[350:].copy()
        obs[-1, 0] += 12.0  # inject a spike at the true root cause (node 0)
        obs[-1, 1] += 9.6
        obs[-1, 2] += 9.6
        ranked = rca_localize(obs, adjacency=adjacency, train=train, top_k=3)
        assert len(ranked) == 3
        top_nodes = [node for node, _ in ranked]
        assert 0 in top_nodes  # the true root cause is surfaced in the top-3


class TestStreamingAdapter:
    def test_scores_stream_and_flags_spike(self) -> None:
        from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector

        scorer = TierStreamingScorer(
            SpectralResidualDetector(), name="sr", window_size=128, min_samples=32
        )
        rng = np.random.default_rng(9)
        results = []
        for i in range(300):
            value = float(rng.normal())
            if i == 250:
                value += 12.0  # a clear point anomaly on the stream
            results.append(scorer({"value": value}))
        # warm-up points are not flagged
        assert results[0]["warmup"] is True
        assert results[0]["is_anomaly"] is False
        # every emitted score is a valid probability
        assert all(0.0 <= r["anomaly_score"] <= 1.0 for r in results)
        # the spike neighbourhood scores above the stream's median
        spike = max(r["anomaly_score"] for r in results[248:256])
        median = float(np.median([r["anomaly_score"] for r in results if not r["warmup"]]))
        assert spike > median

    def test_usable_as_pipeline_detector(self) -> None:
        # The scorer satisfies the StreamingAnomalyPipeline detector contract.
        from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector
        from omni_mercury_engine.infrastructure.streaming import StreamingAnomalyPipeline

        scorer = TierStreamingScorer(SpectralResidualDetector())
        pipeline = StreamingAnomalyPipeline("in", "out", detector=scorer)
        assert pipeline.detector is scorer
        result = scorer({"value": 1.0})
        assert {"is_anomaly", "anomaly_score", "score"} <= set(result)

    def test_invalid_params(self) -> None:
        from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector

        with pytest.raises(ValueError):
            TierStreamingScorer(SpectralResidualDetector(), window_size=1)

    def test_value_less_message_after_warmup_is_not_warmup(self) -> None:
        # A message carrying no finite numeric value must not regress a ready
        # scorer back to warm-up: ``warmup`` reflects the buffered history, not
        # the empty message, so downstream alerting is not wrongly suppressed.
        from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector

        scorer = TierStreamingScorer(
            SpectralResidualDetector(), name="sr", window_size=64, min_samples=16
        )
        rng = np.random.default_rng(3)
        for _ in range(20):  # buffer past min_samples (16)
            scorer({"value": float(rng.normal())})

        # NaN is not finite and "tag" is non-numeric -> nothing to score.
        result = scorer({"value": float("nan"), "tag": "note"})
        assert result["warmup"] is False
        assert result["is_anomaly"] is False
        assert result["anomaly_score"] == 0.0

    def test_value_less_message_during_warmup_stays_warmup(self) -> None:
        # Before ``min_samples`` accumulate, a value-less message stays warm-up.
        from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector

        scorer = TierStreamingScorer(
            SpectralResidualDetector(), name="sr", window_size=64, min_samples=16
        )
        assert scorer({"tag": "no-number-here"})["warmup"] is True


class TestFeatureStoreProvenance:
    def test_store_and_schema_roundtrip(self) -> None:
        from omni_mercury_engine.core.feature_pipeline import FeatureStore, FeatureVersionManager
        from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector

        x, _ = _labelled_series()
        det = SpectralResidualDetector().fit(x)
        store = FeatureStore(backend="memory")
        vm = FeatureVersionManager()
        features, schema = store_tier_features(store, det, "sr", x, version_manager=vm)
        # provenance schema is well-formed and registered
        assert schema.name == "sr" and schema.n_features >= 1
        assert vm.get_schema("sr") is not None
        # features round-trip through the store (flattened float64)
        cached = store.get("sr", np.asarray(x, dtype=np.float64))
        assert cached is not None
        np.testing.assert_allclose(cached.ravel(), features.astype(np.float64).ravel())


class TestObservability:
    def test_record_detector_score_is_safe(self) -> None:
        from omni_mercury_engine.core.metrics import is_prometheus_available, record_detector_score

        # No-op safe whether or not prometheus_client is installed.
        record_detector_score("sr", 0.73)
        assert isinstance(is_prometheus_available(), bool)

    def test_record_detector_score_tolerates_bad_values(self) -> None:
        # Metrics emission is fail-safe: a NaN/inf/out-of-range/non-numeric score
        # from a misbehaving detector must never raise into the hot streaming
        # path -- non-finite is dropped, out-of-range is clamped to [0, 1].
        from omni_mercury_engine.core.metrics import record_detector_score

        bad_values: list[Any] = [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.5,
            1.5,
            "not-a-number",
            None,
        ]
        for bad in bad_values:
            record_detector_score("sr", bad)  # must not raise
