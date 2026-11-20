# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The NAB streaming benchmark measures real detection on real labeled data.

Pins the properties that make it a real benchmark rather than a green
checkmark: it beats chance on every real stream by a margin, the signal
is destroyed when the labels are shuffled (so the AUROC tracks the actual
anomaly windows, not an artifact), and the protocol never fits on or
scores a leaked anomaly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.nab_streaming_benchmark import (
    _CAUSAL_WINDOW,
    _DEFAULT_FIXTURE as FIXTURE,
    _STREAMS,
    _WARMUP_FRACTION,
    auroc,
    causal_features,
    causal_scores,
    evaluate_stream,
    load_stream,
    run_benchmark,
)

if TYPE_CHECKING:
    from pathlib import Path

# Floors sit just under the recorded causal run (0.817/0.692/0.665/0.749,
# mean 0.731); they assert real detection with headroom for numeric noise,
# not a tuned number.
_PER_STREAM_FLOOR = 0.60
_MEAN_FLOOR = 0.68


def test_every_real_stream_beats_chance_by_a_margin() -> None:
    results = run_benchmark(FIXTURE)
    assert results["mean_auroc"] >= _MEAN_FLOOR
    for key, r in results["per_stream"].items():
        assert r["auroc"] >= _PER_STREAM_FLOOR, f"{key} AUROC {r['auroc']:.3f} below floor"


def test_shuffling_labels_destroys_the_signal() -> None:
    """If AUROC tracked a real label, a random permutation must collapse it.

    This is the anti-artifact control: a benchmark that stays 'passing'
    under shuffled labels is measuring nothing. The real signal must drop
    to ~0.5 (|AUROC-0.5| small) when labels are permuted.
    """
    values, labels = load_stream(FIXTURE, *_STREAMS[0])  # realTraffic
    features = causal_features(values)
    warmup = int(_WARMUP_FRACTION * len(features))
    scores = causal_scores(features, warmup)
    scored = np.arange(len(features)) >= warmup
    real = auroc(labels[scored], scores[scored])

    rng = np.random.default_rng(0)
    shuffled = [auroc(rng.permutation(labels[scored]), scores[scored]) for _ in range(20)]
    mean_shuffled = float(np.mean([abs(a - 0.5) for a in shuffled]))
    assert real >= 0.75  # the true AUROC is strong on this stream
    assert mean_shuffled < 0.08, (
        f"shuffled-label AUROC deviates {mean_shuffled:.3f} from 0.5; the metric "
        "is not actually tracking the real anomaly windows"
    )


def test_protocol_is_leak_free() -> None:
    """The protocol must be strictly causal, not merely clean at warm-up.

    Two things make it leak-free: (1) the fitted warmup prefix contains no
    labeled anomaly and the scored region excludes that prefix, and (2) the
    scoring is prefix-invariant -- a point's causal score depends only on
    samples at or before it, so the scores of ``F[:k]`` equal the first ``k``
    scores of ``F`` to machine precision. Property (2) is what the earlier
    whole-batch call failed: ``MercuryAnomalyDetector``'s non-causal stages
    let later samples shift an earlier point's score.
    """
    for key, rel in _STREAMS:
        values, labels = load_stream(FIXTURE, key, rel)
        warmup = int(_WARMUP_FRACTION * len(labels))
        assert not labels[:warmup].any(), f"{key}: anomaly leaked into the fitted warmup prefix"
        # evaluate_stream raises if that invariant is violated.
        result = evaluate_stream(values, labels)
        assert result["warmup"] == warmup
        assert result["window"] == _CAUSAL_WINDOW


def test_causal_scoring_is_prefix_invariant() -> None:
    """Scores of ``F[:k]`` equal the first ``k`` scores of ``F`` exactly.

    This is the real test of causality: under the trailing-window protocol a
    score can only depend on the past, so truncating the future must not move
    any earlier score (atol 1e-9). The superseded whole-batch call is NOT
    prefix-invariant -- shown below -- which is why its "leak_free" label was
    false.
    """
    values, _ = load_stream(FIXTURE, *_STREAMS[0])  # realTraffic
    features = causal_features(values)[:500]
    warmup = int(_WARMUP_FRACTION * len(features))
    window = 64
    k = 350

    full = causal_scores(features, warmup, window)
    prefix = causal_scores(features[:k], warmup, window)
    assert np.allclose(
        full[:k], prefix, atol=1e-9, rtol=0.0
    ), "causal scores changed when the future was truncated; protocol is not causal"

    # The old whole-batch protocol was not prefix-invariant: scoring F[:k]
    # alone differs from scoring it as a prefix of F.
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    det = MercuryAnomalyDetector()
    det.fit(features[:warmup])
    batch_full = np.asarray(det.detect(features)["scores"], dtype=np.float64)
    batch_prefix = np.asarray(det.detect(features[:k])["scores"], dtype=np.float64)
    assert not np.allclose(
        batch_full[:k], batch_prefix, atol=1e-9
    ), "whole-batch scoring was expected to leak future information but did not"


def test_results_are_deterministic() -> None:
    first = run_benchmark(FIXTURE)["per_stream"]
    second = run_benchmark(FIXTURE)["per_stream"]
    for key in first:
        assert abs(first[key]["auroc"] - second[key]["auroc"]) < 1e-9


def test_labels_come_from_committed_windows_not_fabricated() -> None:
    """Each stream has real, non-trivial anomaly coverage from NAB windows."""
    for key, rel in _STREAMS:
        _, labels = load_stream(FIXTURE, key, rel)
        rate = float(labels.mean())
        assert 0.0 < rate < 0.5, f"{key}: implausible anomaly rate {rate:.3f}"


def test_missing_fixture_raises_rather_than_inventing_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stream(tmp_path, *_STREAMS[0])
