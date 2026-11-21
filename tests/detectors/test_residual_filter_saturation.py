# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The residual frequency filter must not tie the scores it lifts over 1.

The filter blends each score with a smoothed copy of the series, and on a
sustained run of high scores the blend exceeds 1. It was hard-clipped there,
so the run came out tied at exactly 1.0. When more than the false-alarm budget
of the fit-time reference was in that tie, the split-conformal cut was 1.0
and ``score > cut`` flagged nothing: the detector went silent on the batches
holding its most extreme rows (measured on a recorded FEMA declarations
series: 16 of 229 rows tied, 0 flags, where 10 rows are flagged once the tie
is resolved).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.detectors._calibration import conformal_cut
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

ALPHA = 0.05


def _series_with_a_sustained_run() -> np.ndarray[Any, Any]:
    """Unremarkable scores with an 80-row run just under the ceiling."""
    rng = np.random.default_rng(0)
    scores = 0.10 + 0.30 * rng.random(400)
    scores[150:230] = 0.9990 + 0.0009 * rng.random(80)
    return scores


def _raw_blend(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """What the filter computes before it bounds the result, re-derived here."""
    kernel_size = max(5, len(scores) // 20)
    kernel_size += kernel_size % 2 == 0
    baseline = np.convolve(scores, np.ones(kernel_size) / kernel_size, mode="same")
    spectrum = np.fft.rfft(scores - baseline)
    power = np.abs(spectrum) ** 2
    cumulative = np.cumsum(power) / power.sum()
    kept = np.zeros_like(spectrum)
    low, high = int(np.searchsorted(cumulative, 0.05)), int(np.searchsorted(cumulative, 0.75))
    kept[low:high] = spectrum[low:high]
    blend: np.ndarray[Any, Any] = 0.7 * scores + 0.3 * (
        baseline + np.fft.irfft(kept, n=len(scores))
    )
    return blend


def test_the_fixture_is_one_the_blend_lifts_over_the_ceiling() -> None:
    raw = _raw_blend(_series_with_a_sustained_run())
    assert (raw >= 1.0).mean() > ALPHA, "more than the alarm budget must exceed 1 for this test"


def test_scores_lifted_over_the_ceiling_are_not_tied() -> None:
    scores = _series_with_a_sustained_run()
    raw = _raw_blend(scores)
    out = MercuryAnomalyDetector._residual_frequency_filter(scores.copy())

    assert out.max() < 1.0
    lifted = raw >= 1.0
    assert len(np.unique(out[lifted])) == len(np.unique(raw[lifted]))
    # The blend's order is the output's order.
    np.testing.assert_array_equal(np.argsort(out, kind="stable"), np.argsort(raw, kind="stable"))


def test_scores_below_the_knee_are_what_the_clip_returned() -> None:
    scores = _series_with_a_sustained_run()
    raw = _raw_blend(scores)
    out = MercuryAnomalyDetector._residual_frequency_filter(scores.copy())

    below = raw <= MercuryAnomalyDetector._FILTER_SATURATION_KNEE
    assert below.sum() > 300
    np.testing.assert_allclose(out[below], np.clip(raw[below], 0.0, 1.0), rtol=0, atol=1e-12)
    assert np.abs(out - np.clip(raw, 0.0, 1.0)).max() <= 1.0 - (
        MercuryAnomalyDetector._FILTER_SATURATION_KNEE
    )


def test_the_conformal_cut_can_still_flag_the_run() -> None:
    """A reference whose top is one tie puts the cut on the tie, and nothing exceeds it."""
    reference = MercuryAnomalyDetector._residual_frequency_filter(
        _series_with_a_sustained_run().copy()
    )
    cut = conformal_cut(np.sort(reference), ALPHA)
    assert cut is not None

    flagged = reference > cut
    assert cut < reference.max()
    assert 0 < flagged.sum() <= np.ceil(ALPHA * len(reference))
    assert flagged[150:230].sum() == flagged.sum()


def test_the_saturation_is_the_identity_below_its_knee_and_increasing_above() -> None:
    values = np.array([-0.5, 0.0, 0.4, 0.999, 1.0, 1.5, 40.0])
    out = MercuryAnomalyDetector._saturate_upper(values, knee=0.999)

    np.testing.assert_array_equal(out[:4], [0.0, 0.0, 0.4, 0.999])
    assert np.all(np.diff(out[3:]) > 0) and out[-1] < 1.0
    np.testing.assert_array_equal(
        MercuryAnomalyDetector._saturate_upper(values),
        MercuryAnomalyDetector._saturate_upper(
            values, knee=MercuryAnomalyDetector._KNN_SATURATION_KNEE
        ),
    )
