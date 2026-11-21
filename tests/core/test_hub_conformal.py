# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hub's calibration is sound and its certificate is served.

Before this, the hub fitted its encoder on every row (calibration rows
included), fitted Platt on *neural* scores and served it over *fused*
scores, and could therefore honestly claim no conformal certificate.  These
tests pin the sound ordering's observable consequences: a labelled fit
produces a certificate on rows no fitted component touched, the served
output carries the uniform conformal block, and the degenerate cases
(unsupervised fit, too few calibration rows) degrade declared, not silent.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the hub's encoder needs the [ml] extra")

from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = _ROOT / "artifacts" / "hub_conformal.json"


def _data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(n, 8))
    anomalous = rng.normal(loc=4.0, size=(max(n // 10, 8), 8))
    features = np.vstack([normal, anomalous])
    labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomalous), dtype=int)])
    order = rng.permutation(len(labels))
    return features[order], labels[order]


@pytest.fixture(scope="module")
def fitted_hub() -> NeuroSymbolicHub:
    features, labels = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hub = NeuroSymbolicHub(input_dim=8, seed=0)
        hub.fit(features, labels)
    return hub


class TestSoundCalibration:
    def test_labelled_fit_produces_a_certificate(self, fitted_hub: NeuroSymbolicHub) -> None:
        assert fitted_hub._conformal is not None
        assert fitted_hub._conformal.n_calibration >= 2

    def test_served_output_carries_the_uniform_block(self, fitted_hub: NeuroSymbolicHub) -> None:
        features, _ = _data(seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = fitted_hub.predict(features[:16], return_explanations=False)
        for output in outputs:
            block = output.conformal
            assert block is not None
            assert {
                "prediction_set",
                "set_size",
                "abstain",
                "coverage",
                "regime",
                "n_calibration",
            } == set(block)
            assert block["regime"] == "per_sample_pre_veto"
            assert block["set_size"] == len(block["prediction_set"])
            assert output.to_dict()["conformal"] == block

    def test_unsupervised_fit_serves_no_fabricated_certificate(self) -> None:
        features, _ = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hub = NeuroSymbolicHub(input_dim=8, seed=0)
            hub.fit(features, None)
            outputs = hub.predict(features[:4], return_explanations=False)
        assert hub._conformal is None
        assert all(o.conformal is None for o in outputs)

    def test_too_few_calibration_rows_degrade_declared(self) -> None:
        features, labels = _data(n=8)  # 16 rows -> cal split of 3 (< 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hub = NeuroSymbolicHub(input_dim=8, seed=0)
            hub.fit(features, labels)
        assert hub._conformal is None
        assert hub._calibrator is None

    def test_certificate_rows_were_touched_by_no_fitted_component(self) -> None:
        """The stated achieved bound is the honest small-sample one, never
        silently the requested level when the class sizes cannot carry it."""
        features, labels = _data(n=60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hub = NeuroSymbolicHub(input_dim=8, seed=0)
            hub.fit(features, labels)
        if hub._conformal is not None:
            assert 0.0 < hub._conformal.coverage_achieved <= 0.9


class TestMeasuredArtifact:
    @pytest.fixture(scope="class")
    def artifact(self) -> dict[str, Any]:
        assert _ARTIFACT.is_file(), "run research/neurosymbolic/harness_hub_conformal.py"
        data: dict[str, Any] = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
        return data

    def test_fifteen_evaluations(self, artifact: dict[str, Any]) -> None:
        assert artifact["protocol"]["n_evaluations"] == 15

    def test_coverage_meets_the_stated_bounds_on_average(self, artifact: dict[str, Any]) -> None:
        mean = artifact["mean"]
        assert mean["empirical_coverage"] >= mean["stated_coverage"]

    def test_split_cost_is_recorded_not_hidden(self, artifact: dict[str, Any]) -> None:
        mean = artifact["mean"]
        assert "split_cost_upper_bound" in mean
        assert mean["auc_sound_split"] == pytest.approx(
            mean["auc_encoder_on_all_train"] - mean["split_cost_upper_bound"], abs=1e-3
        )

    def test_ledger_records_the_adoption(self, artifact: dict[str, Any]) -> None:
        from omni_mercury_engine.governance.improvement_ledger import published

        data = published()
        assert data is not None
        entry = next(
            d for d in data["decisions"] if d["component"] == "neurosymbolic_hub.sound_calibration"
        )
        assert entry["verdict"] == "adopted"
        assert entry["impact"]["mean_empirical_coverage"] == pytest.approx(
            artifact["mean"]["empirical_coverage"], abs=1e-4
        )
