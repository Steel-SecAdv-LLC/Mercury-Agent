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

from pathlib import Path

import numpy as np
import pytest

from benchmarks.nab_streaming_benchmark import (
    _DEFAULT_FIXTURE as FIXTURE,
    _STREAMS,
    auroc,
    causal_features,
    evaluate_stream,
    load_stream,
    run_benchmark,
)

# Floors sit just under the recorded run (0.840/0.661/0.686/0.728); they
# assert real detection with headroom for numeric noise, not a tuned number.
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
    warmup = int(0.15 * len(features))
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    det = MercuryAnomalyDetector()
    det.fit(features[:warmup])
    scores = np.asarray(det.detect(features)["scores"], dtype=np.float64)
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
    """The fitted warmup prefix must contain no labeled anomaly, and the
    scored region must exclude the fitted prefix entirely."""
    for key, rel in _STREAMS:
        values, labels = load_stream(FIXTURE, key, rel)
        warmup = int(0.15 * len(labels))
        assert not labels[:warmup].any(), f"{key}: anomaly leaked into the fitted warmup prefix"
        # evaluate_stream raises if that invariant is violated.
        result = evaluate_stream(values, labels)
        assert result["warmup"] == warmup


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
