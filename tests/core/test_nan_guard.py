# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""No detector takes a NaN silently, and none serves a NaN score.

Measured before the shared guard (``core.base._nan_guard``): of the
engine-seam detectors, 25 accepted a NaN sample without saying so (it
failed every comparison and biased every statistic it reached) and 6
returned a NaN anomaly score. This suite sweeps every engine-seam
``BaseDetector`` in the manifest with a one-NaN probe and fails on any
silent acceptance or non-finite score, so the guard cannot regress
detector by detector.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST


def _engine_seam_base_detectors() -> list[Any]:
    entries = []
    for entry in DETECTOR_MANIFEST:
        if "engine" not in entry.seams:
            continue
        cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
        if isinstance(cls, type) and issubclass(cls, BaseDetector):
            entries.append(pytest.param(entry, id=entry.name))
    return entries


TRAIN = np.random.default_rng(0).standard_normal((120, 4))
PROBE = np.random.default_rng(1).standard_normal((24, 4))
PROBE_NAN = PROBE.copy()
PROBE_NAN[5, 2] = np.nan


@pytest.mark.parametrize("entry", _engine_seam_base_detectors())
def test_nan_sample_is_never_silent_and_never_a_nan_score(entry: Any) -> None:
    """One NaN sample: reported (partial/nan_rows) or refused -- never silent."""
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    detector = cls()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            detector.fit(TRAIN)
            result = detector.detect(PROBE_NAN)
        except Exception:
            # A loud refusal of the whole input satisfies "never silent"
            # (the per-detector fit/modality contracts stay theirs).
            return

    score = result.get("anomaly_score", result.get("anomaly_prob"))
    if score is not None:
        assert np.isfinite(float(score)), f"{entry.name} served a non-finite headline score"
    scores = np.asarray(result.get("scores", [0.0]), dtype=np.float64)
    assert np.all(np.isfinite(scores)), f"{entry.name} served non-finite per-sample scores"

    reported = (
        bool(result.get("partial"))
        or bool(np.asarray(result.get("nan_rows", []), dtype=bool).any())
        or result.get("unmeasured") is not None
    )
    assert reported, (
        f"{entry.name} accepted a NaN sample silently: no partial flag, no "
        "nan_rows, no unmeasured record"
    )


def test_all_nonfinite_input_is_refused_loudly() -> None:
    """Nothing measurable: a guarded detector raises, never grades quiet."""
    from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

    detector = TemporalAnomalyDetector()
    detector.fit(TRAIN[:, 0])
    with pytest.raises(ValueError, match="non-finite"):
        detector.detect(np.full(16, np.nan))


def test_guard_preserves_alignment_and_marks_unmeasured() -> None:
    """Scores stay row-aligned; the NaN row is neutral and named."""
    from omni_mercury_engine.detectors.kmeans_distance import KMeansDistanceDetector

    detector = KMeansDistanceDetector()
    detector.fit(TRAIN)
    result = detector.detect(PROBE_NAN)
    scores = np.asarray(result["scores"], dtype=np.float64)
    assert scores.shape[0] == PROBE_NAN.shape[0]
    assert scores[5] == pytest.approx(0.5)
    assert result["partial"] is True
    assert result["unmeasured"]["indices"] == [5]
    assert np.asarray(result["nan_rows"], dtype=bool)[5]
