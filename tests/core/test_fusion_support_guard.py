# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Out-of-support rows cannot be voted "normal" by the extrapolating fusion net.

Measured before the guard: a semi-supervised ``fit_fusion`` over pure normals
(consensus pseudo-labels put the decision boundary inside the normal cloud)
assigned near-zero anomaly probability to 5-sigma outliers while
in-distribution rows averaged higher — an inverted ranking exactly where the
input is most clearly anomalous, because outside the training envelope the
net's output is extrapolation, not inference. These tests pin the fix:
``detect_with_fusion`` floors such rows at the base detectors' consensus
(whose analytic channels are monotone in deviation), leaves in-support rows
bit-identical, reports the floored rows in ``out_of_support_rows``, and
round-trips the recorded envelope through ``save_model``/``load_model`` —
with a legacy checkpoint (no envelope) restoring the unguarded behaviour.
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
N_SHIFTED = 5


@pytest.fixture(scope="module")
def fitted_engine() -> OmniMercuryEngine:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion")
        rng = np.random.default_rng(0)
        engine.fit_fusion(rng.normal(size=(120, WIDTH)), epochs=2, calibrate=False)
    return engine


@pytest.fixture(scope="module")
def shifted_probe() -> np.ndarray:
    rng = np.random.default_rng(1)
    probe = rng.normal(size=(40, WIDTH))
    probe[:N_SHIFTED] += 6.0
    return probe


def test_fit_records_the_training_envelope(fitted_engine: OmniMercuryEngine) -> None:
    assert fitted_engine._fusion_support_low is not None
    assert fitted_engine._fusion_support_high is not None
    assert fitted_engine._fusion_support_scale is not None
    assert fitted_engine._fusion_support_low.shape == (WIDTH,)
    assert np.all(fitted_engine._fusion_support_scale > 0)


def test_far_outliers_are_floored_at_the_detector_consensus(
    fitted_engine: OmniMercuryEngine, shifted_probe: np.ndarray
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fitted_engine.detect_with_fusion(shifted_probe)
    probs = np.asarray(out["anomaly_probs"], dtype=float)
    flagged = set(out["out_of_support_rows"])
    assert set(range(N_SHIFTED)) <= flagged, "6-sigma rows must be recognised as out-of-support"
    # The floor restores the ranking the net inverted: every shifted row above
    # the in-distribution mean.
    normal_mean = probs[N_SHIFTED:].mean()
    assert (
        probs[:N_SHIFTED].min() > normal_mean
    ), f"shifted rows {probs[:N_SHIFTED]} do not rank above normal mean {normal_mean:.4f}"


def test_in_support_rows_pass_through_unchanged(fitted_engine: OmniMercuryEngine) -> None:
    """Rows inside the envelope keep the net's probabilities bit-identically."""
    low = fitted_engine._fusion_support_low
    high = fitted_engine._fusion_support_high
    assert low is not None and high is not None
    center = (low + high) / 2.0
    inside = np.tile(center, (6, 1)) + np.random.default_rng(2).normal(size=(6, WIDTH)) * 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fitted_engine.detect_with_fusion(inside)
    assert out["out_of_support_rows"] == []
    raw_probs = np.asarray(out["anomaly_probs"], dtype=float)
    guarded, rows = fitted_engine._apply_fusion_support_guard(raw_probs, inside, {})
    assert rows == [] and np.array_equal(guarded, raw_probs)


def test_guard_survives_the_checkpoint_round_trip(
    fitted_engine: OmniMercuryEngine, shifted_probe: np.ndarray, tmp_path: Path
) -> None:
    path = str(tmp_path / "guarded.mdl")
    fitted_engine.save_model(path)
    restored = OmniMercuryEngine(mode="fusion")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restored.load_model(path)
        out_a = fitted_engine.detect_with_fusion(shifted_probe)
        out_b = restored.detect_with_fusion(shifted_probe)
    assert out_b["out_of_support_rows"] == out_a["out_of_support_rows"]
    assert np.allclose(
        np.asarray(out_a["anomaly_probs"], dtype=float),
        np.asarray(out_b["anomaly_probs"], dtype=float),
        atol=1e-9,
    )


def test_legacy_state_without_envelope_is_a_no_op(shifted_probe: np.ndarray) -> None:
    engine = OmniMercuryEngine(mode="fusion")
    assert engine._fusion_support_low is None
    probs = np.linspace(0.0, 1.0, len(shifted_probe))
    guarded, rows = engine._apply_fusion_support_guard(probs, shifted_probe, {})
    assert rows == [] and np.array_equal(guarded, probs)


def test_non_tabular_input_is_a_no_op(fitted_engine: OmniMercuryEngine) -> None:
    probs = np.array([0.1, 0.2])
    guarded, rows = fitted_engine._apply_fusion_support_guard(probs, {"text": ["payload"]}, {})
    assert rows == [] and np.array_equal(guarded, probs)
