# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hysteresis verdict gate: semantics, wiring, and the measured decision.

Before this, streaming verdicts were a single-threshold comparison per tick:
scores oscillating around the alarm cut flipped the verdict every few samples
(measured mean churn 0.0235 on the four committed NAB streams), the decision
layer *measured* that churn but nothing could reduce it, and no streaming
surface offered a stability policy at all.  These tests pin the gate's
semantics, its opt-in wiring into ``CausalMathArrest`` (released
single-threshold behaviour unchanged by default), and the recorded decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.streaming.hysteresis import HysteresisGate, hysteresis_verdicts


class TestGateSemantics:
    def test_enters_high_holds_in_band_clears_low(self) -> None:
        gate = HysteresisGate(0.8, 0.5)
        assert gate.update(0.7) is False  # below enter: stays clear
        assert gate.update(0.9) is True  # enters
        assert gate.update(0.6) is True  # in the band: holds
        assert gate.update(0.79) is True  # still in the band
        assert gate.update(0.5) is False  # at the exit cut: clears
        assert gate.update(0.6) is False  # back in the band from below: clear

    def test_single_threshold_degenerate_band_matches_raw(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 200)
        verdicts = hysteresis_verdicts(scores, 0.7, 0.7)
        raw = np.zeros(200, dtype=bool)
        on = False
        for i, s in enumerate(scores):
            if not on and s > 0.7:
                on = True
            elif on and s <= 0.7:
                on = False
            raw[i] = on
        assert np.array_equal(verdicts, raw)

    def test_non_finite_score_does_not_move_the_gate(self) -> None:
        gate = HysteresisGate(0.8, 0.5)
        gate.update(0.9)
        assert gate.update(float("nan")) is True
        assert gate.active is True
        gate.reset()
        assert gate.update(float("inf")) is False

    def test_misordered_band_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            HysteresisGate(0.5, 0.8)
        with pytest.raises(ValueError, match="finite"):
            HysteresisGate(float("nan"), 0.1)

    def test_batch_form_equals_stateful_form(self) -> None:
        rng = np.random.default_rng(3)
        scores = rng.uniform(0, 1, 500)
        batch = hysteresis_verdicts(scores, 0.9, 0.4)
        gate = HysteresisGate(0.9, 0.4)
        stateful = np.array([gate.update(s) for s in scores])
        assert np.array_equal(batch, stateful)

    def test_churn_never_increases(self) -> None:
        """The gate can only merge flip pairs, never create one."""
        rng = np.random.default_rng(7)
        for trial in range(20):
            scores = rng.uniform(0, 1, 300)
            enter = float(rng.uniform(0.5, 0.95))
            exit_ = float(rng.uniform(0.05, enter))
            raw = scores > enter
            smooth = hysteresis_verdicts(scores, enter, exit_)
            churn_raw = int(np.sum(raw[1:] != raw[:-1]))
            churn_smooth = int(np.sum(smooth[1:] != smooth[:-1]))
            assert churn_smooth <= churn_raw


class TestCausalMathArrestWiring:
    @pytest.fixture(scope="class")
    def warmup(self) -> np.ndarray:
        return np.random.default_rng(0).normal(size=300)

    def test_default_behaviour_is_unchanged(self, warmup: np.ndarray) -> None:
        from omni_mercury_engine.streaming.causal_math_arrest import CausalMathArrest

        detector = CausalMathArrest(window=64, probes="minimal")
        detector.fit(warmup)
        verdict = detector.update(5.0)
        assert "exit_threshold" not in verdict
        assert verdict["is_anomaly"] == (verdict["score"] > verdict["threshold"])

    def test_opt_in_gate_holds_between_cuts(self, warmup: np.ndarray) -> None:
        from omni_mercury_engine.streaming.causal_math_arrest import CausalMathArrest

        detector = CausalMathArrest(window=64, probes="minimal", exit_alpha=0.20)
        detector.fit(warmup)
        assert detector._exit_threshold is not None
        assert detector.threshold is not None
        assert detector._exit_threshold <= detector.threshold
        saw_hold = False
        prev = False
        for value in np.random.default_rng(1).normal(scale=4.0, size=200):
            verdict = detector.update(float(value))
            assert verdict["exit_threshold"] == pytest.approx(detector._exit_threshold)
            score = verdict["score"]
            if (
                prev
                and verdict["is_anomaly"]
                and detector._exit_threshold < score <= verdict["threshold"]
            ):
                # Raw thresholding would have cleared here; the gate held.
                saw_hold = True
            prev = verdict["is_anomaly"]
        assert saw_hold, "no in-band hold was exercised; widen the test stimulus"

    def test_exit_alpha_must_exceed_alpha(self) -> None:
        from omni_mercury_engine.streaming.causal_math_arrest import CausalMathArrest

        with pytest.raises(ValueError, match="exit_alpha"):
            CausalMathArrest(exit_alpha=0.05)
        with pytest.raises(ValueError, match="exit_alpha"):
            CausalMathArrest(alpha=0.1, exit_alpha=0.1)


class TestRecordedDecision:
    def test_ledger_records_the_adoption_with_numbers(self) -> None:
        from omni_mercury_engine.governance.improvement_ledger import published

        data = published()
        assert data is not None
        entry = next(d for d in data["decisions"] if d["component"] == "streaming.hysteresis_gate")
        assert entry["verdict"] == "adopted_opt_in"
        impact = entry["impact"]
        assert impact["mean_churn_before"] == pytest.approx(0.0235, abs=1e-3)
        assert impact["mean_churn_after"] == pytest.approx(0.0153, abs=1e-3)
        assert impact["mean_f1_before"] == pytest.approx(0.3123, abs=1e-3)
        assert impact["mean_f1_after"] == pytest.approx(0.3722, abs=1e-3)
        assert "speed_7578" in str(entry)  # the one measured cost is named


class TestDefaultFlipRefusal:
    def test_the_alarm_path_default_flip_is_refused_with_numbers(self) -> None:
        from omni_mercury_engine.governance.improvement_ledger import published

        data = published()
        assert data is not None
        entry = next(
            d for d in data["decisions"] if d["component"] == "streaming.hysteresis_default"
        )
        assert entry["verdict"] == "refused"
        impact = entry["impact"]
        # Both halves of the refusal are pinned: the NAB score falls and the
        # false-alarm rate rises under the flipped default.
        assert impact["nab_standard_score_exit_alpha_0_20"] < impact["nab_standard_score_default"]
        assert (
            impact["alarm_rate_on_normals_exit_alpha_0_20"]
            > impact["alarm_rate_on_normals_default"]
        )

    def test_the_released_default_is_still_single_threshold(self) -> None:
        import inspect

        from omni_mercury_engine.streaming.causal_math_arrest import CausalMathArrest

        signature = inspect.signature(CausalMathArrest.__init__)
        assert signature.parameters["exit_alpha"].default is None
