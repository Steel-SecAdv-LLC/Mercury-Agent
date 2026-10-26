# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The default labelled fit carries a conformal coverage certificate.

The decision layer was built around a distribution-free coverage
certificate (its ``calibrated``/``coverage`` signals and the conformal
prediction sets), yet nothing on the default path ever fitted one:
``calibrate_fusion_conformal()`` existed only as a separate opt-in, so every
decision read "margin heuristic, no coverage guarantee" even when labels
were available to certify. These tests pin the automatic wiring: a
stratified slice is held out before detectors or the net see it, the
certificate is fitted on it, serving attaches prediction sets, the decision
signals energise, and the certificate survives the checkpoint round-trip.
They equally pin the refusals: consensus pseudo-labels never certify (a
coverage guarantee against labels the detectors manufactured is circular),
tiny classes skip with a log instead of certifying on nothing, and
``conformal_calibrate=False`` restores the prior behaviour exactly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.engine import OmniMercuryEngine

if TYPE_CHECKING:
    from pathlib import Path

WIDTH = 8


def _data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(size=(200, WIDTH)), rng.normal(size=(40, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(200), np.ones(40)])
    return x, y


@pytest.fixture(scope="module")
def labelled_engine() -> tuple[OmniMercuryEngine, dict[str, object]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion")
        x, y = _data()
        metrics = engine.fit_fusion(x, y, epochs=2)
    return engine, metrics


def test_labelled_fit_fits_the_certificate(
    labelled_engine: tuple[OmniMercuryEngine, dict[str, object]],
) -> None:
    engine, metrics = labelled_engine
    assert engine._fusion_conformal is not None
    conformal = metrics["conformal"]
    assert isinstance(conformal, dict)
    assert conformal["coverage"] == 0.9
    assert conformal["n_calibration"] >= 10  # at least 5 per class


def test_serving_attaches_sets_and_decision_signals_energise(
    labelled_engine: tuple[OmniMercuryEngine, dict[str, object]],
) -> None:
    engine, _ = labelled_engine
    probe = np.random.default_rng(1).normal(size=(10, WIDTH))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = engine.detect_with_fusion(probe)
    assert "conformal" in out
    assert out["conformal"]["coverage"] == 0.9
    signals = out["decision"]["signals"]
    assert signals["calibrated"] is True
    assert signals["coverage"] == 0.9


def test_certificate_survives_the_checkpoint_round_trip(
    labelled_engine: tuple[OmniMercuryEngine, dict[str, object]], tmp_path: Path
) -> None:
    engine, _ = labelled_engine
    path = str(tmp_path / "certified.mdl")
    engine.save_model(path)
    restored = OmniMercuryEngine(mode="fusion")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restored.load_model(path)
        out = restored.detect_with_fusion(np.random.default_rng(2).normal(size=(5, WIDTH)))
    assert restored._fusion_conformal is not None
    assert "conformal" in out


def test_consensus_pseudo_labels_never_certify() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion")
        x, _ = _data(seed=3)
        metrics = engine.fit_fusion(x, epochs=2)  # no labels -> consensus path
    assert engine._fusion_conformal is None
    assert "conformal" not in metrics


def test_tiny_class_skips_instead_of_certifying_on_nothing() -> None:
    rng = np.random.default_rng(4)
    x = np.vstack([rng.normal(size=(120, WIDTH)), rng.normal(size=(3, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(120), np.ones(3)])  # 3 positives < 5 holdout floor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion")
        metrics = engine.fit_fusion(x, y, epochs=2)
    assert engine._fusion_conformal is None
    assert "conformal" not in metrics


def test_opt_out_restores_prior_behaviour() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion")
        x, y = _data(seed=5)
        metrics = engine.fit_fusion(x, y, epochs=2, conformal_calibrate=False)
    assert engine._fusion_conformal is None
    assert "conformal" not in metrics


def test_realized_coverage_holds_on_fresh_data() -> None:
    """Per-class coverage on a fresh draw stays at or above target - slack.

    This one trains properly (default epochs, larger classes): the
    certificate's guarantee rests on exchangeability between calibration and
    serving scores, and a barely-trained net (the cheap wiring fixture's 2
    epochs) is dominated by batch-relative feature noise — measuring coverage
    there measures the noise, not the wiring.
    """
    rng = np.random.default_rng(6)
    x = np.vstack([rng.normal(size=(500, WIDTH)), rng.normal(size=(60, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(500), np.ones(60)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion")
        engine.fit_fusion(x, y)
    x_test = np.vstack([rng.normal(size=(150, WIDTH)), rng.normal(size=(30, WIDTH)) + 3.0])
    y_test = np.concatenate([np.zeros(150), np.ones(30)]).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = np.asarray(engine.score_fusion(x_test), dtype=float).ravel()
    assert engine._fusion_conformal is not None
    sets = engine._fusion_conformal.predict(probs)
    labels = sets.label_sets()
    for cls in (0, 1):
        hits = [cls in lab for yy, lab in zip(y_test, labels, strict=True) if yy == cls]
        assert np.mean(hits) >= 0.8, f"class {cls} realized coverage {np.mean(hits):.2f}"
