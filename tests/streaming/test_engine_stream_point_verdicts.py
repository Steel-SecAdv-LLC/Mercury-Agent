# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stream verdict is about the newest sample, scored against what came before it.

Two defects in ``EngineStreamSession.observe``:

* On a refit tick the session fitted its detectors on the window *including*
  the sample it was about to score, so that sample helped define the "normal"
  it was then compared with.
* When a detector returned no per-row scores of the window's length, the
  session took the batch-level ``anomaly_score`` (for some detectors the
  window maximum) as the newest sample's score and ``any()`` of its flags as
  the newest sample's verdict -- so one anomaly anywhere in the window flagged
  every tick until it scrolled out, up to ``window_size`` samples later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector


@pytest.fixture()
def engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(require_explicit_fit=False)


def test_the_scored_sample_is_never_in_the_window_it_is_fitted_on(engine: Any) -> None:
    """Every fit -- warm-up and each refit -- excludes the sample being scored."""
    session = engine.stream_session(
        detector_types=["statistical"], min_samples=8, window_size=32, refit_interval=5
    )
    detector = session._detectors["statistical"]
    real_fit = detector.fit
    current: list[float] = []
    fits: list[tuple[float, np.ndarray[Any, Any]]] = []

    def recording_fit(data: Any, *args: Any, **kwargs: Any) -> Any:
        fits.append((current[-1], np.array(data, dtype=np.float64, copy=True)))
        return real_fit(data, *args, **kwargs)

    detector.fit = recording_fit
    # Unique marker values: sample i is i + 0.125, so membership is exact.
    for i in range(40):
        current.append(i + 0.125)
        session.observe(current[-1])

    assert len(fits) >= 5, "expected a warm-up fit and several refits"
    for scored, fitted in fits:
        assert scored not in fitted.reshape(
            -1
        ), f"sample {scored} was in the window its own reference was fitted on"
        # The reference is exactly the samples that preceded it.
        assert fitted.reshape(-1).max() == scored - 1.0


def test_the_first_verdict_still_arrives_at_min_samples(engine: Any) -> None:
    """Warm-up length is unchanged: the fit just uses the samples before it."""
    session = engine.stream_session(detector_types=["statistical"], min_samples=8, window_size=32)
    rng = np.random.default_rng(3)
    statuses = [session.observe(float(rng.standard_normal()))["status"] for _ in range(9)]
    assert statuses == ["warming_up"] * 7 + ["ok", "ok"]


class _WindowLevelDetector(BaseDetector):
    """Reports a window-level result only: no per-row scores for the window.

    Flags the whole window while the marker value 99.0 is anywhere in it, as
    a detector summarising its batch would.
    """

    def __init__(self) -> None:
        super().__init__({"threshold": 0.5})

    def fit(self, data: Any) -> _WindowLevelDetector:
        self._is_fitted = True
        return self

    def detect(self, data: Any) -> dict[str, Any]:
        seen = bool(np.any(np.asarray(data) == 99.0))
        return {"anomaly_score": 0.99 if seen else 0.01, "is_anomaly": [seen]}

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        return np.zeros((int(np.asarray(data).shape[0]), 1), dtype=np.float32)


class _PerRowDetector(BaseDetector):
    """Scores each row on its own: 0.99 for the marker value, 0.01 otherwise."""

    def __init__(self) -> None:
        super().__init__({"threshold": 0.5})

    def fit(self, data: Any) -> _PerRowDetector:
        self._is_fitted = True
        return self

    def detect(self, data: Any) -> dict[str, Any]:
        hit = np.asarray(data).reshape(len(data), -1)[:, 0] == 99.0
        return {
            "anomaly_score": float(np.where(hit, 0.99, 0.01).max()),
            "scores": np.where(hit, 0.99, 0.01),
            "is_anomaly": hit,
        }

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        return np.zeros((int(np.asarray(data).shape[0]), 1), dtype=np.float32)


def test_a_window_level_detector_does_not_latch_the_verdict(engine: Any) -> None:
    """One early anomaly must not flag the clean ticks that follow it."""
    engine.register_detector("per_row", _PerRowDetector())
    engine.register_detector("window_level", _WindowLevelDetector())
    session = engine.stream_session(
        detector_types=["per_row", "window_level"], min_samples=8, window_size=32
    )
    rng = np.random.default_rng(5)
    for _ in range(12):
        session.observe(float(rng.standard_normal() * 0.1))
    spike = session.observe(99.0)
    later = [session.observe(float(rng.standard_normal() * 0.1)) for _ in range(15)]

    # The spike's own tick is flagged, by the detector that scored that sample.
    assert spike["is_anomaly"] is True
    # The clean ticks after it are not: the spike is still in the window, and
    # only a window-level detector says so.
    assert not any(
        v["is_anomaly"] for v in later
    ), "clean ticks were flagged because an earlier sample is still in the window"
    assert all(v["score"] < 0.5 for v in later)

    clean = later[-1]
    assert clean["status"] == "ok"
    entry = clean["per_detector"]["window_level"]
    assert entry["point_resolved"] is False
    assert entry["score"] is None and entry["is_anomaly"] is None
    # The window-level statistic is still there, named for what it is.
    assert entry["window_score"] == pytest.approx(0.99)
    assert entry["window_is_anomaly"] is True
    # The tick's verdict comes from the point-resolved detector alone.
    assert clean["per_detector"]["per_row"]["point_resolved"] is True
    assert clean["is_anomaly"] is clean["per_detector"]["per_row"]["is_anomaly"]
    assert clean["score"] == pytest.approx(clean["per_detector"]["per_row"]["score"])


def test_a_tick_with_no_point_resolved_detector_abstains(engine: Any) -> None:
    """Window-level evidence alone is an abstention, never a clear and never a flag."""
    engine.register_detector("window_level", _WindowLevelDetector())
    session = engine.stream_session(detector_types=["window_level"], min_samples=4, window_size=16)
    verdicts = [session.observe(float(i)) for i in range(6)]
    verdicts.append(session.observe(99.0))
    verdicts.append(session.observe(1.0))
    for verdict in verdicts[4:]:
        assert verdict["status"] == "not_point_resolved"
        assert verdict["partial"] is True
        assert verdict["is_anomaly"] is None
        assert verdict["score"] is None
        assert verdict["per_detector"]["window_level"]["point_resolved"] is False
    assert verdicts[-1]["per_detector"]["window_level"]["window_is_anomaly"] is True
    assert verdicts[5]["per_detector"]["window_level"]["window_is_anomaly"] is False


def test_per_row_detectors_are_point_resolved(engine: Any) -> None:
    session = engine.stream_session(detector_types=["statistical"], min_samples=8, window_size=32)
    rng = np.random.default_rng(7)
    verdict: dict[str, Any] = {}
    for _ in range(12):
        verdict = session.observe(float(rng.standard_normal()))
    entry = verdict["per_detector"]["statistical"]
    assert entry["point_resolved"] is True
    assert isinstance(entry["is_anomaly"], bool)
    assert 0.0 <= entry["score"] <= 1.0


def _level_shift_session(engine: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """150 samples around 0, then 150 around 8, with the interval refit out of reach."""
    session = engine.stream_session(
        detector_types=["statistical"], min_samples=16, window_size=64, refit_interval=100_000
    )
    rng = np.random.default_rng(21)
    before = [session.observe(float(rng.normal(0.0, 0.5))) for _ in range(150)]
    after = [session.observe(float(rng.normal(8.0, 0.5))) for _ in range(150)]
    return before, after


def test_a_detected_drift_refits_on_the_post_drift_window(engine: Any) -> None:
    """A level shift stops being an anomaly once the monitor has seen it.

    The drift monitor only counted: the detectors kept the pre-shift
    reference until the fixed interval came round, so every sample of the
    new level was flagged in the meantime.
    """
    before, after = _level_shift_session(engine)
    quiet = sum(bool(v["is_anomaly"]) for v in before[50:]) / 100.0

    # Once adapted, the new level is flagged no more often than the old one
    # was (plus slack), not on every tick.
    settled = after[60:]
    flagged = sum(bool(v["is_anomaly"]) for v in settled) / len(settled)
    assert flagged <= quiet + 0.15, f"post-shift flag rate {flagged:.2f} vs {quiet:.2f} before"

    # The adaptation is a drift refit, within the monitor's detection delay.
    detections = [i for i, v in enumerate(after) if v["drift_detected"]]
    refits = [i for i, v in enumerate(after) if v["refit_reason"] == "drift"]
    assert detections and detections[0] < 40, detections[:3]
    assert refits and refits[0] == detections[0] + 1, (detections[:3], refits[:3])
    assert all(v["refit_reason"] in (None, "drift") for v in after)
    # The shift itself is flagged before the refit: the anomaly is reported, then absorbed.
    assert any(v["is_anomaly"] for v in after[: refits[0]])


def test_a_drift_refit_excludes_the_sample_being_scored(engine: Any) -> None:
    session = engine.stream_session(
        detector_types=["statistical"], min_samples=16, window_size=64, refit_interval=100_000
    )
    detector = session._detectors["statistical"]
    real_fit = detector.fit
    current: list[float] = []
    fits: list[tuple[float, np.ndarray[Any, Any]]] = []

    def recording_fit(data: Any, *args: Any, **kwargs: Any) -> Any:
        fits.append((current[-1], np.array(data, dtype=np.float64, copy=True).reshape(-1)))
        return real_fit(data, *args, **kwargs)

    detector.fit = recording_fit
    rng = np.random.default_rng(22)
    reasons = []
    for i in range(260):
        level = 0.0 if i < 130 else 8.0
        current.append(float(level + rng.normal(0.0, 0.5)))
        reasons.append(session.observe(current[-1])["refit_reason"])

    assert reasons.count("warmup") == 1 and "drift" in reasons
    assert len(fits) == sum(reason is not None for reason in reasons)
    for scored, fitted in fits:
        assert scored not in fitted
    # A drift refit is fitted on the post-shift samples: the monitor's cut may
    # leave one boundary sample of the old level, never the old window.
    drift_fits = [fitted for _, fitted in fits[1:]]
    assert len(drift_fits) >= 2, "the short post-drift reference was never consolidated"
    for fitted in drift_fits:
        assert int((fitted < 4.0).sum()) <= 1, "a drift refit kept the pre-shift window"
        assert float(np.median(fitted)) > 4.0
    assert len(drift_fits[-1]) >= 2 * len(drift_fits[0])


def test_a_stationary_stream_is_not_refitted_for_drift(engine: Any) -> None:
    session = engine.stream_session(
        detector_types=["statistical"], min_samples=16, window_size=64, refit_interval=100_000
    )
    rng = np.random.default_rng(23)
    ticks = [session.observe(float(rng.normal(0.0, 0.5))) for _ in range(300)]
    assert [v["refit_reason"] for v in ticks].count("warmup") == 1
    assert not any(v["refit_reason"] == "drift" for v in ticks)
    assert ticks[-1]["window_fill"] == 64
