# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-tick decision churn over the stream's point-resolved verdicts.

A stream whose verdict flips every tick is operationally useless even when
each tick is individually defensible, and nothing measured it: the session
reported scores and flags but never how unstable they were. Each tick now
carries ``churn`` -- the fraction of verdict flips among the last
``churn_window`` consecutive point-resolved verdict pairs -- ``None`` until
two resolved verdicts exist, warmup ticks excluded.

The mechanics are pinned against a deterministic step detector (sample > 1.0
is an anomaly, exactly), so a constant-normal stream *provably* reads 0.0 and
an alternating one 1.0; a real-engine session pins the surface end to end.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.streaming.engine_stream import EngineStreamSession


class _StepDetector:
    """Deterministic point detector: a sample above 1.0 is an anomaly."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def fit(self, window: Any) -> None:
        """Stateless reference; nothing to fit."""

    def get_effective_threshold(self) -> float:
        return 0.5


class _FakeEngine:
    """The minimal engine surface a stream session consumes, deterministic."""

    def __init__(self) -> None:
        self.detectors: dict[str, Any] = {"step": _StepDetector()}

    def detect(
        self,
        window: Any,
        detector_types: list[str] | None = None,
        detector_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows = np.asarray(window, dtype=np.float64)
        scores = (rows[:, 0] > 1.0).astype(np.float64)
        return {
            "detectors": {
                "step": {"scores": scores.tolist(), "is_anomaly": (scores > 0.5).tolist()}
            }
        }


def _session(**over: Any) -> EngineStreamSession:
    kwargs: dict[str, Any] = {
        "detector_types": ["step"],
        "min_samples": 4,
        "window_size": 64,
        "refit_interval": 10_000,
        "monitor_drift": False,
    }
    kwargs.update(over)
    return EngineStreamSession(_FakeEngine(), **kwargs)  # type: ignore[arg-type]


class TestStreamChurnDeterministic:
    def test_warmup_ticks_are_excluded_and_churn_starts_none(self) -> None:
        session = _session()
        for _ in range(3):
            verdict = session.observe(0.0)
            assert verdict["status"] == "warming_up"
            assert verdict["churn"] is None
        # First point-resolved verdict: one verdict is no pair yet.
        verdict = session.observe(0.0)
        assert verdict["status"] == "ok"
        assert verdict["churn"] is None
        # Second resolved verdict: one pair exists.
        assert session.observe(0.0)["churn"] == 0.0

    def test_constant_normal_stream_has_zero_churn(self) -> None:
        session = _session()
        verdict: dict[str, Any] = {}
        for _ in range(40):
            verdict = session.observe(0.0)
        assert verdict["status"] == "ok"
        assert verdict["is_anomaly"] is False
        assert verdict["churn"] == 0.0

    def test_alternating_verdict_stream_has_churn_one(self) -> None:
        session = _session()
        verdict: dict[str, Any] = {}
        for i in range(40):
            verdict = session.observe(2.0 if i % 2 == 0 else 0.0)
            if i >= 5:  # two resolved verdicts exist from tick 5 on
                assert verdict["churn"] == 1.0
        assert verdict["status"] == "ok"
        assert verdict["churn"] == 1.0

    def test_churn_window_bounds_the_history(self) -> None:
        session = _session(churn_window=4)
        verdict: dict[str, Any] = {}
        for i in range(16):
            verdict = session.observe(2.0 if i % 2 == 0 else 0.0)
        assert verdict["churn"] == 1.0
        # Five stable ticks fill the 5-verdict history: churn drops to 0.
        for _ in range(5):
            verdict = session.observe(0.0)
        assert verdict["churn"] == 0.0

    def test_unmeasured_ticks_do_not_disturb_churn(self) -> None:
        session = _session()
        for i in range(10):
            session.observe(2.0 if i % 2 == 0 else 0.0)
        before = session.observe(2.0)["churn"]
        nan_tick = session.observe(float("nan"))
        assert nan_tick["status"] == "unmeasured"
        assert nan_tick["churn"] == before
        # The next resolved verdict pairs with the last resolved one, the
        # unmeasured tick having never entered the history.
        after = session.observe(0.0)
        assert after["churn"] == before

    def test_churn_window_is_validated(self) -> None:
        with pytest.raises(ValueError):
            _session(churn_window=0)


class TestStreamChurnThroughTheEngine:
    """The real engine surface carries the field end to end."""

    def test_engine_session_reports_churn(self) -> None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(require_explicit_fit=False)
        session = engine.stream_session(
            detector_types=["statistical"],
            min_samples=8,
            window_size=64,
            refit_interval=10_000,
            monitor_drift=False,
            churn_window=16,
        )
        rng = np.random.default_rng(3)
        verdict: dict[str, Any] = {}
        for i in range(7):
            verdict = session.observe(float(rng.standard_normal() * 0.1))
            assert verdict["churn"] is None  # warmup ticks are excluded
        verdict = session.observe(float(rng.standard_normal() * 0.1))
        assert verdict["status"] == "ok"
        assert verdict["churn"] is None  # one resolved verdict is no pair
        for _ in range(20):
            verdict = session.observe(float(rng.standard_normal() * 0.1))
        assert verdict["status"] == "ok"
        assert isinstance(verdict["churn"], float)
        assert 0.0 <= verdict["churn"] <= 1.0
