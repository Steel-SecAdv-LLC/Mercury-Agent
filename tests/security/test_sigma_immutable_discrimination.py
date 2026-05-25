"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC.

Pins the σ_Immutable discrimination finding and the deterministic
critical-ethical floor that cures it.

Two layers:

* :class:`TestCriticalEthicalFloor` -- the deterministic floor in
  isolation (no torch required): a collapsed ethical anchor is a
  categorical refusal, the healthy default passes, and the narrative
  tuning scalars are correctly excluded.
* :class:`TestDiscriminationProbe` -- the full probe through the real
  trained gate (torch required): asserts the gate DISCRIMINATES and that
  there are no fail-open false-assurance leaks.  This is the regression
  guard for the headline finding of #242: if a future change deletes the
  floor (or otherwise re-opens a leak), the probe's verdict / leak set
  moves and this test fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
)
from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    reset_global_network,
)
from omni_mercury_engine.security.sigma_immutable_gate import (
    CRITICAL_ETHICAL_FLOOR,
    SigmaImmutableGate,
)
from scripts import sigma_immutable_discrimination_check as probe


@pytest.fixture(autouse=True)
def _reset_gosnn() -> None:
    reset_global_network()
    yield
    reset_global_network()


class TestCriticalEthicalFloor:
    """The deterministic floor -- no torch needed (it never touches the network)."""

    def _floor_gate(self) -> SigmaImmutableGate:
        # verify_corpus=False keeps this unit test free of the signed-corpus
        # dependency; the floor logic is independent of corpus + network.
        return SigmaImmutableGate(verify_corpus=False)

    def test_floor_is_the_trainer_ethical_band_threshold(self) -> None:
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SIGMA_IMMUTABLE_DEFAULT_THRESHOLD,
        )

        # The per-anchor floor is the trainer's ethical-band lower bound
        # (0.93), not the lower absolute benevolence baseline (0.70) -- the
        # latter leaves a [0.70, 0.93) gap the network does not catch.
        assert CRITICAL_ETHICAL_FLOOR == SIGMA_IMMUTABLE_DEFAULT_THRESHOLD
        assert CRITICAL_ETHICAL_FLOOR == 0.93

    def test_anchors_exclude_narrative_scalars(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        anchors = gosnn.critical_ethical_anchors()
        for knob in ("omniforgiveness", "omnitransparency", "omniexplainability"):
            assert knob not in anchors
        # The genuine anchors are present.
        for anchor in ("omnibenevolence", "omnimorality", "omnijustice"):
            assert anchor in anchors

    def test_healthy_default_passes_the_floor(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        gate = self._floor_gate()
        # No violations on the unmodified default operating point.
        assert gate.critical_ethical_floor_violations(gosnn.critical_ethical_anchors()) == []
        # And enforce_ethical_floor does not raise.
        gate.enforce_ethical_floor("test", gosnn.critical_ethical_anchors())

    def test_low_narrative_defaults_do_not_trip_the_floor(self) -> None:
        # forgiveness=0.10 / transparency=0.18 are below the floor but are
        # narrative knobs -- excluded, so the healthy default still passes.
        gosnn = GlobalOmniScalarNetwork()
        anchors = gosnn.critical_ethical_anchors()
        assert all(v >= CRITICAL_ETHICAL_FLOOR for v in anchors.values())

    def test_zeroed_benevolence_is_a_categorical_refusal(self) -> None:
        gate = self._floor_gate()
        anchors = {"omnibenevolence": 0.0, "omnimorality": 1.20}
        with pytest.raises(EthicalConstraintViolationError) as exc:
            gate.enforce_ethical_floor("test", anchors)
        assert exc.value.check == "sigma_immutable_ethical_floor"
        assert "omnibenevolence" in exc.value.details["floor_violations"]

    def test_below_floor_anchor_is_refused(self) -> None:
        gate = self._floor_gate()
        # 0.10 is below 0.70 even though above 0.0 -- still a breach.
        violations = gate.critical_ethical_floor_violations({"omnijustice": 0.10})
        assert violations == [("omnijustice", 0.10)]


@pytest.mark.usefixtures("_reset_gosnn")
class TestDiscriminationProbe:
    """Full probe through the live trained gate (torch + weights required)."""

    def setup_method(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.security.sigma_immutable_gate import (
            get_sigma_immutable_gate,
        )

        if not get_sigma_immutable_gate().is_trained:
            pytest.skip("σ_Immutable gate untrained (weights absent); probe needs the real network")

    def test_gate_discriminates_with_no_false_assurance(self) -> None:
        summary = probe.run_discrimination_check()

        # The headline binary verdict.
        assert summary.verdict == "discriminates", (
            "σ_Immutable is near-constant-PASS -- false assurance on the "
            f"anomaly path: {summary.confusion}"
        )
        # No fail-open leaks beyond the (now empty) documented set.
        assert summary.unexpected_leaks == [], (
            "NEW false-assurance leak(s) -- a fail-open regression: " f"{summary.unexpected_leaks}"
        )
        assert set(summary.false_assurance) <= probe.KNOWN_FALSE_ASSURANCE
        # Robust separation signals (not knife-edge).
        assert summary.good_pass_rate == 1.0
        assert summary.bad_pass_rate == 0.0
        assert summary.score_range > 0.5

    def test_specific_catastrophic_breaches_refuse(self) -> None:
        summary = probe.run_discrimination_check()
        by_name = {r.name: r for r in summary.results}
        for name in (
            "benevolence_zeroed",
            "benevolence_below_floor",
            "single_critical_zeroed",
            "contradictory_opaque",
            "ethical_collapse_zero",
        ):
            assert not by_name[name].passes, f"{name} must be refused"
