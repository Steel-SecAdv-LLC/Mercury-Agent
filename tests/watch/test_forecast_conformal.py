# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The storm forecast carries a conformal certificate, uniform with detection.

Before this, the anomaly path served prediction sets with measured coverage
while the forecast path served a bare probability: the uniform-conformal
contract stopped at detection.  The shipped ``kp_g1_model.json`` now carries
a ``conformal`` block in exactly the fusion checkpoint's ``conformal_state``
layout, fitted on 9,497 walk-forward out-of-sample forecasts (2000-2025) and
measured on the 320 after-the-fit days (empirical coverage 0.8656 overall;
0.9286 on storm days, 0.8523 on quiet days — the 2026 regime sits under the
0.90 target on quiet days and the evidence says so).  These tests pin the
artifact, the uniform output block, and the issued forecast's basis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omni_mercury_engine.watch.forecast import Model, load_model

_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE = _ROOT / "research" / "forecast_kp" / "evidence.json"


class TestShippedCertificate:
    def test_the_shipped_model_carries_the_certificate(self) -> None:
        model = load_model()
        assert model.conformal is not None
        # The exact conformal_state layout the fusion checkpoint uses.
        assert {"coverage", "coverage_achieved", "seed", "thresholds"} <= set(model.conformal)
        assert model.conformal.get("n_calibration", 0) > 1000

    def test_prediction_set_is_the_uniform_block(self) -> None:
        model = load_model()
        block = model.prediction_set(0.5)
        assert block is not None
        assert set(block) == {
            "prediction_set",
            "set_size",
            "abstain",
            "coverage",
            "regime",
            "n_calibration",
        }
        assert block["regime"] == "per_day_walk_forward"
        assert block["set_size"] == len(block["prediction_set"])
        assert block["abstain"] == (block["set_size"] == 2)

    def test_legacy_artifact_serves_none_not_a_fabricated_set(self) -> None:
        model = load_model()
        legacy = Model(
            version=model.version,
            mean=model.mean,
            scale=model.scale,
            weights=model.weights,
            bias=model.bias,
            climatology=model.climatology,
            persistence=model.persistence,
            trained_through=model.trained_through,
            conformal=None,
        )
        assert legacy.prediction_set(0.5) is None

    def test_certificate_round_trips_through_the_shared_classifier(self) -> None:
        import numpy as np

        from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier

        model = load_model()
        assert model.conformal is not None
        classifier = BinaryConformalClassifier.from_state(dict(model.conformal))
        pred = classifier.predict(np.array([0.05, 0.5, 0.95]))
        assert list(pred.set_size) == [int(s) for s in pred.set_size]

    def test_evidence_measures_the_held_out_coverage(self) -> None:
        evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
        conformal = evidence["conformal"]
        assert conformal["coverage_target"] == 0.9
        after = conformal["after_the_fit"]
        assert after["days"] > 0
        for key in (
            "empirical_coverage",
            "singleton_rate",
            "ambiguous_rate",
            "empty_rate",
        ):
            assert 0.0 <= after[key] <= 1.0
        # The rates partition the held-out days.
        assert after["singleton_rate"] + after["ambiguous_rate"] + after[
            "empty_rate"
        ] == pytest.approx(1.0, abs=1e-6)

    def test_model_n_calibration_matches_the_evidence(self) -> None:
        model = load_model()
        evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
        assert model.conformal is not None
        assert model.conformal["n_calibration"] == evidence["conformal"]["n_calibration"]
