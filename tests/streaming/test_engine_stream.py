# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-sample-at-a-time detection through the engine.

Regression: nothing took live data one sample at a time *through the
engine* -- the streaming helpers each drove a single detector, the engine
only served batches, and the cascade adapters rebuilt their cascade for
every record, wiping staged evidence between records.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="module")
def engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    return OmniMercuryEngine(require_explicit_fit=False)


class TestEngineStreamSession:
    def test_warmup_is_partial_and_never_an_all_clear(self, engine: Any) -> None:
        session = engine.stream_session(
            detector_types=["statistical"], min_samples=8, window_size=32
        )
        rng = np.random.default_rng(3)
        for _ in range(7):
            verdict = session.observe(float(rng.standard_normal()))
            assert verdict["status"] == "warming_up"
            assert verdict["partial"] is True
            assert verdict["is_anomaly"] is None
        verdict = session.observe(float(rng.standard_normal()))
        assert verdict["status"] == "ok"
        assert verdict["partial"] is False
        assert isinstance(verdict["is_anomaly"], bool)

    def test_spike_flags_and_scores_higher_than_calm(self, engine: Any) -> None:
        session = engine.stream_session(
            detector_types=["statistical"], min_samples=16, window_size=64
        )
        rng = np.random.default_rng(5)
        calm = None
        for _ in range(30):
            calm = session.observe(float(rng.standard_normal() * 0.1))
        assert calm is not None and calm["status"] == "ok"
        spike = session.observe(25.0)
        assert spike["score"] > calm["score"]
        assert spike["per_detector"]["statistical"]["score"] > 0.5

    def test_nan_sample_is_unmeasured_not_scored(self, engine: Any) -> None:
        session = engine.stream_session(
            detector_types=["statistical"], min_samples=4, window_size=16
        )
        rng = np.random.default_rng(7)
        for _ in range(6):
            session.observe(float(rng.standard_normal()))
        before = session.samples_unmeasured
        verdict = session.observe(float("nan"))
        assert verdict["status"] == "unmeasured"
        assert verdict["partial"] is True
        assert verdict["is_anomaly"] is None
        assert session.samples_unmeasured == before + 1
        # The window was not poisoned: the next finite sample scores normally.
        after = session.observe(0.0)
        assert after["status"] == "ok"
        assert not np.isnan(after["score"])

    def test_detectors_are_not_refit_per_sample(self, engine: Any) -> None:
        session = engine.stream_session(
            detector_types=["statistical"], min_samples=4, window_size=32, refit_interval=10
        )
        detector = engine.detectors["statistical"]
        fits = 0
        original_fit = type(detector).fit

        def counting_fit(self: Any, data: Any) -> Any:
            nonlocal fits
            fits += 1
            return original_fit(self, data)

        type(detector).fit = counting_fit  # type: ignore[method-assign]
        try:
            rng = np.random.default_rng(11)
            for _ in range(24):
                session.observe(float(rng.standard_normal()))
        finally:
            type(detector).fit = original_fit  # type: ignore[method-assign]
        # One warm-up fit plus interval refits -- never one per sample.
        assert 1 <= fits <= 3

    def test_mismatched_dimensionality_is_refused(self, engine: Any) -> None:
        session = engine.stream_session(detector_types=["statistical"], min_samples=2)
        session.observe([1.0, 2.0])
        with pytest.raises(ValueError, match="features"):
            session.observe([1.0, 2.0, 3.0])

    def test_verdict_carries_real_uncertainty(self, engine: Any) -> None:
        session = engine.stream_session(
            detector_types=["statistical"], min_samples=8, window_size=32
        )
        rng = np.random.default_rng(13)
        verdict = None
        for _ in range(12):
            verdict = session.observe(float(rng.standard_normal()))
        assert verdict is not None
        assert 0.0 <= verdict["uncertainty"] <= 1.0
        assert 0.0 <= verdict["per_detector"]["statistical"]["uncertainty"] <= 1.0


class TestPersistentCascadeAdapters:
    """Cascade evidence must survive across records in live-feed mode."""

    HURRICANE_RECORD = {
        "hurricane": {
            "pressure_data": {
                "central_pressure_mb": 940.0,
                "environmental_pressure_mb": 1010.0,
            },
            "sst_data": {"sst_celsius": 29.0},
        }
    }

    @staticmethod
    def _surge_and_river_record() -> dict[str, Any]:
        rows = [{"t": f"2026-07-08 {i:02d}:00", "v": f"{v:.3f}"} for i, v in enumerate([1.0, 2.2])]
        tide = [{"t": f"2026-07-08 {i:02d}:00", "v": f"{v:.3f}"} for i, v in enumerate([1.0, 1.1])]
        return {
            "coops_observed": {"data": rows},
            "coops_predicted": {"predictions": tide},
            "station_id": "8518750",
            "river_gauge": {
                "lid": "TESTG1",
                "status": {"observed": {"primary": 21.4, "floodCategory": "major"}},
                "flood": {"categories": {"minor": {"stage": 14.0}}},
            },
        }

    def test_default_adapter_restarts_per_record(self) -> None:
        from omni_mercury_engine.detectors.adapters.earth import SurgeFloodCascadeAdapter

        adapter = SurgeFloodCascadeAdapter()
        adapter.detect(self.HURRICANE_RECORD)
        out = adapter.detect(self._surge_and_river_record())
        # Without the hurricane evidence in the same record the stage cannot
        # pass WATCH, so the fresh cascade grades below COMPOUND_FLOOD (3).
        assert float(out["scores"][0]) < 1.0

    def test_persistent_adapter_composes_evidence_across_records(self) -> None:
        from omni_mercury_engine.detectors.adapters.earth import SurgeFloodCascadeAdapter

        adapter = SurgeFloodCascadeAdapter(config={"persistent_cascade": True})
        first = adapter.detect(self.HURRICANE_RECORD)
        assert float(first["scores"][0]) == pytest.approx(1.0 / 3.0)  # WATCH
        second = adapter.detect(self._surge_and_river_record())
        assert float(second["scores"][0]) == pytest.approx(1.0)  # COMPOUND_FLOOD
        assert bool(second["flags"][0]) is True

    def test_persistent_eq_tsunami_accumulates_events(self) -> None:
        from omni_mercury_engine.detectors.adapters.earth import EqTsunamiCascadeAdapter

        watch_event = {"magnitude": 8.1, "depth_km": 12.0}
        persistent = EqTsunamiCascadeAdapter(config={"persistent_cascade": True})
        persistent.detect({"events": [watch_event], "offshore": True})
        confirmation = {
            "events": [{"magnitude": 5.0, "depth_km": 30.0}],
            "offshore": True,
        }
        follow_up = persistent.detect(confirmation)
        fresh = EqTsunamiCascadeAdapter().detect(confirmation)
        # The persistent cascade still holds the M8.1 watch; a fresh one only
        # sees the M5 aftershock and grades below watch.
        assert float(follow_up["scores"][0]) > float(fresh["scores"][0])


class TestStreamDriftMonitor:
    """The ADWIN online drift monitor rides the stream session (D16 wiring)."""

    def test_mean_shift_is_detected_and_counted(self, engine: Any) -> None:
        session = engine.stream_session(
            detector_types=["statistical"], min_samples=8, window_size=64
        )
        rng = np.random.default_rng(21)
        events = 0
        for _ in range(150):
            verdict = session.observe(float(rng.normal(0.0, 0.5)))
        for _ in range(150):
            verdict = session.observe(float(rng.normal(8.0, 0.5)))
            events = verdict["drift"]["events"]
        assert events >= 1
        assert set(verdict["drift"]) == {"detected", "events", "message"}

    def test_monitor_can_be_disabled(self, engine: Any) -> None:
        session = engine.stream_session(
            detector_types=["statistical"], min_samples=4, window_size=16, monitor_drift=False
        )
        verdict = None
        for value in (0.1, 0.2, 0.1, 0.3, 0.2):
            verdict = session.observe(value)
        assert verdict is not None and "drift" not in verdict
