# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A streamed point is scored against a reference it is not part of.

Three behaviours of :class:`StreamingDetector`, each wrong before:

* the new point was appended to the window and the detector re-fitted on that
  window before the point was scored, so on every refit tick an anomaly helped
  define the "normal" it was then compared with (and likewise for a batch);
* a detection or refit failure returned ``None`` exactly as warm-up does, with
  nothing to tell an outage from a cold start;
* a failed refit never reset its counter, so every later tick paid for a
  full-window fit that was going to fail again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.streaming.streaming_detector import StreamingDetector

if TYPE_CHECKING:
    import pytest


def _spy_on_fit(sd: StreamingDetector) -> list[np.ndarray]:
    """Record a copy of every array the wrapped detector is fitted on."""
    fitted: list[np.ndarray] = []
    original = sd._detector.fit

    def recording_fit(data: Any, *args: Any, **kwargs: Any) -> Any:
        fitted.append(np.array(data, copy=True))
        return original(data, *args, **kwargs)

    sd._detector.fit = recording_fit  # type: ignore[method-assign]
    return fitted


class TestReferenceExcludesTheScoredPoint:
    def test_single_point_is_never_in_its_own_reference(self) -> None:
        sd = StreamingDetector(window_size=200, refit_interval=7, min_samples=20)
        fitted = _spy_on_fit(sd)
        rng = np.random.default_rng(0)
        for tick in range(120):
            marker = 1000.0 + tick  # unique per tick, so membership is unambiguous
            before = len(fitted)
            sd.ingest([marker, float(rng.normal())])
            for reference in fitted[before:]:
                assert marker not in reference[:, 0], (
                    f"tick {tick}: the point being scored was part of the window the "
                    "detector was re-fitted on"
                )
        assert len(fitted) >= 10, "the stream should have refitted repeatedly"

    def test_batch_is_not_in_its_own_reference_once_warm(self) -> None:
        sd = StreamingDetector(window_size=500, refit_interval=10, min_samples=30)
        fitted = _spy_on_fit(sd)
        rng = np.random.default_rng(1)

        cold = sd.ingest_batch(rng.normal(size=(60, 2)))
        assert cold is not None
        # No earlier reference exists at a cold start; the result says so.
        assert cold["streaming_metadata"]["reference_includes_scored_points"] is True

        batch = np.column_stack([np.full(20, 777.0), rng.normal(size=20)])
        before = len(fitted)
        warm = sd.ingest_batch(batch)
        assert warm is not None
        assert warm["streaming_metadata"]["reference_includes_scored_points"] is False
        assert len(fitted) > before, "the second batch was due a refit"
        for reference in fitted[before:]:
            assert 777.0 not in reference[:, 0]


class TestFailuresAreNotWarmup:
    @staticmethod
    def _warm() -> StreamingDetector:
        sd = StreamingDetector(window_size=200, refit_interval=10, min_samples=20)
        rng = np.random.default_rng(2)
        for _ in range(40):
            sd.ingest(rng.normal(size=2).tolist())
        assert sd.last_status == "ok"
        return sd

    def test_cold_start_reports_warmup(self) -> None:
        sd = StreamingDetector(min_samples=20)
        assert sd.ingest([0.0, 1.0]) is None
        assert sd.last_status == "warmup"
        assert sd.last_error is None

    def test_detection_failure_is_distinguishable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = self._warm()

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("sensor bus down")

        monkeypatch.setattr(sd._detector, "detect", boom)
        assert sd.ingest([0.0, 0.0]) is None
        assert sd.last_status == "detect_failed"
        assert sd.last_error is not None and "sensor bus down" in sd.last_error
        stats = sd.get_stats()
        assert stats["last_status"] == "detect_failed"

    def test_recovery_clears_the_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sd = self._warm()
        original = sd._detector.detect
        monkeypatch.setattr(
            sd._detector, "detect", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        assert sd.ingest([0.0, 0.0]) is None
        monkeypatch.setattr(sd._detector, "detect", original)
        assert sd.ingest([0.0, 0.0]) is not None
        assert sd.last_status == "ok"
        assert sd.last_error is None


class TestFailedRefitIsNotRetriedEveryTick:
    def test_fitted_detector_keeps_serving_and_waits_for_the_next_interval(self) -> None:
        sd = StreamingDetector(window_size=200, refit_interval=10, min_samples=20)
        rng = np.random.default_rng(3)
        for _ in range(30):
            sd.ingest(rng.normal(size=2).tolist())
        assert sd.last_status == "ok"

        attempts = 0

        def failing_fit(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            raise ValueError("degenerate window")

        sd._detector.fit = failing_fit  # type: ignore[method-assign]
        served = [sd.ingest(rng.normal(size=2).tolist()) for _ in range(50)]

        # One attempt per refit interval, not one per point (which was 41+).
        assert attempts <= 6, f"{attempts} refit attempts over 50 points"
        assert all(result is not None for result in served), "the previous fit keeps serving"

    def test_cold_detector_backs_off_between_attempts(self) -> None:
        sd = StreamingDetector(window_size=200, refit_interval=10, min_samples=20)
        attempts = 0

        def failing_fit(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            raise ValueError("cannot fit")

        sd._detector.fit = failing_fit  # type: ignore[method-assign]
        rng = np.random.default_rng(4)
        results = [sd.ingest(rng.normal(size=2).tolist()) for _ in range(60)]

        assert all(result is None for result in results)
        assert sd.last_status == "refit_failed"
        # 40 eligible ticks; one attempt per tick before, a backoff of 2 now.
        assert attempts <= 22, f"{attempts} cold refit attempts over 40 eligible points"
