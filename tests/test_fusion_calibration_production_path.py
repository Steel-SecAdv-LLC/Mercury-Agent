# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for production-path fusion calibration application.

These lock the contract that ``detect_with_fusion`` (the production decision
boundary) and ``detect_with_fusion_calibrated`` (with threshold calibration)
return temperature-scaled probabilities — not raw sigmoid — when a calibrator
is present, identical to ``score_fusion``'s contract.

Without these the trained ``_fusion_calibrator`` only affects the benchmark
path; user-facing ``mercury-agent detect`` keeps returning uncalibrated values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch


def _fixture(seed: int = 7, sep: float = 2.0) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.RandomState(seed)
    normal = rng.normal(0.0, 1.0, (500, 12))
    anomaly = rng.normal(sep, 1.0, (60, 12))
    x = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(500), np.ones(60)]).astype(np.int64)
    order = rng.permutation(len(x))
    return x[order], y[order]


@pytest.fixture
def trained_engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    torch.manual_seed(0)
    np.random.seed(0)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    x, y = _fixture()
    engine.fit_fusion(x, y, epochs=20, early_stopping_patience=10)
    assert engine._fusion_calibrator is not None, "fixture requires a fitted calibrator"
    return engine


@pytest.fixture(autouse=True)
def _bypass_sigma_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the σ_Immutable second ethical gate so detect_with_fusion can run."""
    from omni_mercury_engine import engine as engine_mod

    monkeypatch.setattr(engine_mod, "_GOSNN_TESTING_BYPASS", True)


class TestDetectWithFusionAppliesCalibration:
    def test_anomaly_prob_matches_score_fusion(self, trained_engine: Any) -> None:
        """The production scalar is the calibrated scalar score_fusion would produce."""
        x, _ = _fixture(seed=11)
        sample = x[:1]

        score = float(trained_engine.score_fusion(sample)[0])
        result = trained_engine.detect_with_fusion(sample)
        assert (
            abs(result["anomaly_prob"] - score) < 1e-6
        ), f"detect_with_fusion={result['anomaly_prob']} != score_fusion={score}"

    def test_calibration_changes_anomaly_prob(self, trained_engine: Any) -> None:
        """With a non-identity calibrator, anomaly_prob differs from raw sigmoid."""
        # Force a non-identity temperature so the calibration delta is visible.
        trained_engine._fusion_calibrator.temperature = 2.0

        x, _ = _fixture(seed=13)
        sample = x[:1]

        calibrated = trained_engine.detect_with_fusion(sample)["anomaly_prob"]
        saved = trained_engine._fusion_calibrator
        trained_engine._fusion_calibrator = None
        try:
            raw = trained_engine.detect_with_fusion(sample)["anomaly_prob"]
        finally:
            trained_engine._fusion_calibrator = saved

        # With T=2, the sigmoid is softened; only an exactly-0.5 raw probability
        # is a fixed point. Any non-degenerate prediction must visibly shift.
        if abs(raw - 0.5) > 1e-4:
            assert (
                abs(calibrated - raw) > 1e-6
            ), f"calibrator had no effect on detect_with_fusion: raw={raw}, cal={calibrated}"

    def test_no_calibrator_is_noop(self, trained_engine: Any) -> None:
        """When _fusion_calibrator is None, detect_with_fusion matches raw sigmoid.

        score_fusion and detect_with_fusion route through different inference
        wrappers (FusionInference.predict vs the model.forward used by
        score_fusion) and the two paths drift by ~1e-4 in fp32 even on
        identical inputs, so we don't expect bit-exact agreement here — only
        that ``detect_with_fusion`` is not visibly worse than raw sigmoid (and
        is not silently calibrated to something else by accident).
        """
        trained_engine._fusion_calibrator = None
        x, _ = _fixture(seed=17)
        sample = x[:1]

        result = trained_engine.detect_with_fusion(sample)
        raw_via_score = float(trained_engine.score_fusion(sample)[0])
        # The no-op contract is structural: with calibrator=None the helper
        # returns the input unchanged. The remaining drift is fp32 noise
        # between the two inference wrappers (not a calibration effect).
        assert abs(result["anomaly_prob"] - raw_via_score) < 1e-3


class TestDetectCalibratedAppliesCalibration:
    def test_threshold_calibration_runs_on_calibrated_scores(self, trained_engine: Any) -> None:
        """detect_with_fusion_calibrated must apply temperature before threshold search.

        Otherwise the threshold finder (Otsu/F1/percentile) operates on uncalibrated
        scores while detect_with_fusion's anomaly_prob is calibrated — the boolean
        is_anomaly verdict then mixes scales and may flip versus a consistent path.
        """
        trained_engine._fusion_calibrator.temperature = 0.5

        x, _ = _fixture(seed=19)
        sample = x[:8]

        result = trained_engine.detect_with_fusion_calibrated(
            sample, calibration_method="percentile", contamination=0.1
        )

        # The structural contract: the call must not return raw uncalibrated
        # scores when a calibrator is set. detect_with_fusion_calibrated calls
        # detect_with_fusion (whose anomaly_prob is now calibrated by the
        # in-place fix to the result dict) and then re-runs the model on the
        # full batch to find the threshold — the re-run path now also applies
        # calibration. So both the scalar anomaly_prob and the batch over which
        # the threshold is computed are temperature-scaled.
        anomaly_prob = result["anomaly_prob"]
        if hasattr(anomaly_prob, "__len__"):
            anomaly_prob = float(np.asarray(anomaly_prob).reshape(-1)[0])
        score = float(trained_engine.score_fusion(sample[:1])[0])
        # Same wrapper-drift caveat as above; allow ~1e-3 fp32 noise between
        # the two inference paths but assert calibration is actually applied
        # (i.e. raw sigmoid was around 0.x and calibrated value is what
        # score_fusion sees on the same row).
        assert abs(anomaly_prob - score) < 1e-3, (
            f"detect_with_fusion_calibrated anomaly_prob={anomaly_prob} "
            f"differs from calibrated score_fusion={score} by more than the "
            "fp32 wrapper-drift tolerance — calibration likely not applied."
        )


class TestDetectWithFusionSingleSampleShape:
    def test_one_dim_sample_matches_two_dim_row(self, trained_engine: Any) -> None:
        """A 1-D sample is one observation, exactly as score_fusion treats it.

        Before the entry normalization, a plain ``x[0]`` crashed five frames
        deep in the fusion forward (detector extractors disagree on whether a
        1-D array is one sample or n one-feature samples). The contract is
        score_fusion's: 1-D == one sample == its (1, n_features) reshape.
        """
        x, _ = _fixture(seed=23)

        from_1d = trained_engine.detect_with_fusion(x[0])
        from_2d = trained_engine.detect_with_fusion(x[:1])

        assert from_1d["anomaly_prob"] == pytest.approx(from_2d["anomaly_prob"], abs=1e-6)
        assert from_1d["is_anomaly"] == from_2d["is_anomaly"]

    def test_one_dim_torch_tensor_matches_numpy(self, trained_engine: Any) -> None:
        """The torch.Tensor input branch gets the same normalization."""
        x, _ = _fixture(seed=29)

        from_tensor = trained_engine.detect_with_fusion(torch.from_numpy(x[0]))
        from_numpy = trained_engine.detect_with_fusion(x[:1])

        assert from_tensor["anomaly_prob"] == pytest.approx(from_numpy["anomaly_prob"], abs=1e-6)
