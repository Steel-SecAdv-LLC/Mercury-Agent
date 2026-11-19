# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The causal fixed-window execution mode of the 21-probe ensemble, pinned.

``CausalMathArrest`` runs the identical instruments — the same probes, the
same fusion and decorrelator, the same conformal alarm cut — in strictly
causal order: each point is the last element of its trailing window. This
module pins the three properties that make it an execution mode rather than
a new algorithm:

* **Causality**: a point's score never changes when later samples change.
* **Instrument identity**: one update's score equals ``AnomalyMathArrest``
  scoring the same trailing window directly.
* **The measured triple** on the four recorded NAB files, the same metrics
  the offline path pins (``tests/detectors/test_recorded_nab_series.py``,
  math_arrest row 0.749 / 33.7 / 0.017): the causal mode measures mean AUROC
  0.6709, NAB standard score 43.93 and mean normal-point alarm rate 0.0117.
  Causal windowing costs ranking AUROC and buys earliness — the NAB score,
  which rewards an early alarm inside a window and charges for every false
  one, is *higher* than the offline batch mode's.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("causal_math_arrest"),
    pytest.mark.timeout(900),
]

from benchmarks.nab_streaming_benchmark import run_causal_math_arrest
from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.streaming.causal_math_arrest import CausalMathArrest

#: The measured causal triple (mean AUROC, NAB standard score, alarm rate),
#: recorded 2026-11-19 at window 256. Tolerances mirror the offline pins:
#: AUROC +-0.05, NAB +-max(7, 10%), alarm rate +-max(0.01, 25%).
MEASURED = (0.6709, 43.93, 0.0117)


def _warmup_and_stream() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    series = np.sin(np.linspace(0.0, 30.0, 700)) + 0.05 * rng.standard_normal(700)
    return series[:200], series[200:]


def test_scores_are_causal() -> None:
    """Changing the future changes nothing about the past."""
    warmup, stream = _warmup_and_stream()
    altered = stream.copy()
    altered[300:] += 9.0  # violent future

    base = CausalMathArrest(window=64).fit(warmup)
    other = CausalMathArrest(window=64).fit(warmup)
    base_scores, _ = base.score_stream(stream)
    other_scores, _ = other.score_stream(altered)
    np.testing.assert_array_equal(base_scores[:300], other_scores[:300])


def test_one_update_scores_the_trailing_window_through_the_instrument() -> None:
    """No new algorithm: an update is the instrument on the trailing window."""
    warmup, stream = _warmup_and_stream()
    detector = CausalMathArrest(window=64).fit(warmup)
    for value in stream[:80]:
        result = detector.update(float(value))

    reference = AnomalyMathArrest()
    reference.fit(warmup)
    window = np.concatenate([warmup, stream[:80]])[-64:]
    expected = float(np.asarray(reference.detect(window), dtype=float).ravel()[-1])
    assert result["score"] == pytest.approx(expected, abs=1e-12)
    assert result["window_used"] == 64


def test_update_before_fit_is_refused() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted"):
        CausalMathArrest().update(1.0)
    with pytest.raises(ValueError, match="window must be"):
        CausalMathArrest(window=4)
    with pytest.raises(ValueError, match="alpha must be"):
        CausalMathArrest(alpha=1.5)


def test_the_measured_triple_on_the_recorded_nab_corpus() -> None:
    """The causal mode's triple, measured through the streaming harness."""
    result = run_causal_math_arrest()
    pinned_auroc, pinned_nab, pinned_rate = MEASURED
    assert result["mean_auroc"] == pytest.approx(pinned_auroc, abs=0.05)
    assert result["nab_standard_score"] == pytest.approx(
        pinned_nab, abs=max(7.0, 0.10 * pinned_nab)
    )
    rate = result["mean_alarm_rate_normal_points"]
    assert rate == pytest.approx(pinned_rate, abs=max(0.01, 0.25 * pinned_rate))
    # The conformal promise carries over to the causal execution: the mean
    # alarm rate on normal points stays at or under the cut's alpha.
    assert rate <= 0.05
