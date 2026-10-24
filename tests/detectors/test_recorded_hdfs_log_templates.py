# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``deeplog_sequence`` on recorded HDFS log templates, measured and pinned.

``tests/fixtures/loghub`` holds the Drain-parsed event-template keys of 2000
real Hadoop HDFS log lines in file order -- LogHub's ``HDFS_2k`` sample, 14
templates (``scripts/record_hdfs_log_fixtures.py``; He, Zhu, He & Lyu 2020,
arXiv:2008.06448).  That key sequence is the input
:class:`DeepLogSequenceDetector` takes, and this is its first measurement on
recorded data of its own kind: it was previously pinned only on NAB's
*continuous* series, where a categorical next-key model has no grammar to learn.

The lines are not anomaly-labelled -- LogHub's block-level labels ship with the
full 11-million-line ``HDFS_1`` release -- so what is measured here is the
false-alarm rate on a held-out stretch of the same log, and the detection of a
template the training half never shows.  No AUROC is claimed.

The sessions in this sample interleave (2200 block ids over 2000 lines, median
one event each), so the next-key distribution is genuinely high-entropy and the
rate below is not low.  That is the data, not a calibration choice: two of the
14 templates appear only in the held-out half.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.deeplog_sequence import DeepLogSequenceDetector
from tests.recorded_data import hdfs_log_templates

TRAIN = 1000
#: Share of the held-out half flagged, as measured.
MEASURED_FALSE_ALARM_RATE = 0.1730
#: What a 0.5 cut on the miss probability flags on the same half.
MISS_PROBABILITY_RULE_RATE = 0.8270
RATE_TOLERANCE = 0.02


def _fitted() -> tuple[DeepLogSequenceDetector, np.ndarray, list[str]]:
    keys, events = hdfs_log_templates()
    return DeepLogSequenceDetector().fit(keys[:TRAIN]), keys, events


def test_the_fixture_is_the_recorded_sample() -> None:
    keys, events = hdfs_log_templates()

    assert keys.shape == (2000,)
    assert keys.dtype == np.int64
    assert len(events) == 14
    assert events[0] == "E1" and events[-1] == "E14"
    assert set(keys.tolist()) <= set(range(len(events)))


def test_the_false_alarm_rate_on_a_held_out_stretch_matches_the_measurement() -> None:
    detector, keys, _ = _fitted()

    result = detector.detect(keys[TRAIN:])
    flagged = np.asarray(result["is_anomaly"], dtype=bool)

    assert flagged.mean() == pytest.approx(MEASURED_FALSE_ALARM_RATE, abs=RATE_TOLERANCE)
    assert result["metadata"]["top_g_violations"] == int(flagged.sum())


def test_the_paper_criterion_beats_a_cut_on_the_miss_probability() -> None:
    """``1 - P`` falls under 0.5 only where one successor holds half the mass.

    With 14 templates and interleaved sessions there is rarely such a
    successor, so the miss-probability cut this detector used to decide on
    flags 0.827 of the same held-out stretch that DeepLog's own top-g criterion
    flags 0.173.
    """
    detector, keys, _ = _fitted()
    held_out = keys[TRAIN:]

    result = detector.detect(held_out)
    _, probabilities, outside_top_g = detector._eval(detector._to_1d_int(held_out))

    miss_rule = (1.0 - probabilities) > detector.threshold
    assert miss_rule.mean() == pytest.approx(MISS_PROBABILITY_RULE_RATE, abs=RATE_TOLERANCE)
    assert np.asarray(result["is_anomaly"], dtype=bool).mean() < miss_rule.mean() / 4.0
    # The verdict is the paper's rule, conjoined with the configured threshold.
    np.testing.assert_array_equal(
        np.asarray(result["is_anomaly"], dtype=bool), outside_top_g & miss_rule
    )


def test_a_template_the_training_half_never_shows_is_flagged() -> None:
    detector, keys, events = _fitted()
    held_out = keys[TRAIN:].copy()
    unseen = len(events)  # one past the recorded vocabulary
    held_out[400:410] = unseen

    flagged = np.asarray(detector.detect(held_out)["is_anomaly"], dtype=bool)

    assert flagged[400:410].all()
    elsewhere = flagged[np.r_[0:400, 410 : flagged.size]]
    assert elsewhere.mean() == pytest.approx(MEASURED_FALSE_ALARM_RATE, abs=RATE_TOLERANCE)


def test_the_per_position_top_g_array_is_reported() -> None:
    """Only the *count* used to survive, so a caller could not localise a violation."""
    detector, keys, _ = _fitted()

    metadata = detector.detect(keys[TRAIN:])["metadata"]

    outside = np.asarray(metadata["top_g_outside"], dtype=bool)
    assert outside.shape == (keys.size - TRAIN,)
    assert int(outside.sum()) == metadata["top_g_violations"]


def test_the_manifest_declares_the_modality_this_detector_takes() -> None:
    """It was ``series_1d``; a next-key model has no grammar in a continuous series."""
    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

    entry = next(e for e in DETECTOR_MANIFEST if e.name == "deeplog_sequence")
    assert entry.modality == "event"
