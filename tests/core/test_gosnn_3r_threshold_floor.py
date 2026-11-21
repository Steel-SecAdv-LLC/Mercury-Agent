# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""3R feedback may only ever tighten the GOSNN ethical gate, never loosen it.

``GOSNN3RIntegration._adjust_gosnn_from_3r`` used to lower the process-global
``ethical_gate.threshold`` / ``sigma_immutable_threshold`` from *input-derived*
state (``max(0.90, threshold - adjustment)`` on a stable low-score stream),
below the configured domain threshold, and it overwrote the shared singleton's
threshold for every domain with this instance's state.  These tests pin the
restored contract: the configured gate threshold is a floor (raising is
permitted, instance-locally), and another domain's GOSNN threshold is never
touched by this instance's feedback.

That fix, on its own, also removed every way the feedback TIGHTENED a gate
that is enforced: the global threshold was the only thing the working
threshold had ever reached, and ``process`` threw away the verdict of the gate
it evaluated.  The working threshold kept climbing to 0.99 and decided
nothing.  The second half of this module pins the tightening as an
enforcement: ``process`` holds the gate's score against the larger of the
configured threshold and this instance's working threshold, and a score that
fails it gets no fused score.  Nothing is written to the shared GOSNN.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.gosnn_3r_integration import GOSNN3RIntegration


def _feed_steady_low_scores(integration: GOSNN3RIntegration, steps: int = 80) -> None:
    """A stable, low fusion-score stream: the exact input that used to lower the gate."""
    rng = np.random.default_rng(0)
    for _ in range(steps):
        integration.process(0.1 + 0.001 * float(rng.standard_normal()), 0.1, 0.1)


def _feed_volatile_scores(integration: GOSNN3RIntegration, steps: int = 60) -> None:
    """Fused scores swinging between 0 and 1: the input the tightening branch is for."""
    for step in range(steps):
        level = float(step % 2)
        integration.process(level, level, level)


def _score_the_gate_at(
    integration: GOSNN3RIntegration, score: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Have the GOSNN gate's network report ``score``; its own threshold test is kept.

    The trained network scores the intact configuration at 0.9999, above any
    threshold the feedback can reach, so a score that falls *between* two
    thresholds has to be supplied. Only the number is supplied: the gate still
    compares it with its own (configured) threshold.
    """
    gate = integration.gosnn.ethical_gate

    def _evaluate(scalar_vector: Any) -> tuple[bool, float]:
        return score >= gate.threshold, score

    monkeypatch.setattr(gate, "evaluate", _evaluate)


def _released(result: Any) -> bool:
    """A fused score was released: the gate passed and the score is a number."""
    return getattr(result, "ethical_gate_passed", None) is True and math.isfinite(
        result.fusion_score
    )


def _withheld(result: Any) -> bool:
    """The gate refused, and no fused score came back."""
    return getattr(result, "ethical_gate_passed", None) is False and math.isnan(result.fusion_score)


class TestEthicalThresholdFloor:
    def test_steady_low_scores_never_lower_the_gate_below_configured(self) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        gosnn = integration.gosnn
        configured_gate = gosnn.ethical_gate.threshold
        configured_sigma = gosnn.sigma_immutable_threshold

        _feed_steady_low_scores(integration)

        assert gosnn.ethical_gate.threshold >= configured_gate, (
            "input-derived feedback lowered the process-global ethical gate "
            f"threshold from {configured_gate} to {gosnn.ethical_gate.threshold}"
        )
        assert gosnn.sigma_immutable_threshold >= configured_sigma
        # The instance-local working threshold is clamped at the configured
        # floor too: it feeds the fusion equation's ethical override.
        assert integration.state.ethical_threshold >= configured_sigma

    def test_state_threshold_can_still_rise_on_unstable_scores(self) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        start = integration.state.ethical_threshold
        rng = np.random.default_rng(1)
        # Highly variable fusion scores: the documented tightening branch.
        for _ in range(40):
            integration.process(float(rng.uniform(0.0, 1.0)), 0.1, 0.1)
        # Strictly: ``>=`` is also satisfied by a threshold that never moves.
        assert integration.state.ethical_threshold > start
        assert integration.state.ethical_threshold <= 0.99
        # And the gate it tightens is not the shared one.
        assert integration.gosnn.ethical_gate.threshold == start


class TestNoCrossDomainOverwrite:
    def test_feedback_does_not_overwrite_another_domains_threshold(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            get_sigma_immutable_threshold,
        )

        # A GOSNN resolved for the medical domain carries the medical
        # threshold (0.93 by design, to avoid false negatives in critical
        # scenarios).  Running another integration's feedback over the shared
        # network must not replace that domain threshold with this instance's
        # state default.
        medical_gosnn = GlobalOmniScalarNetwork(domain="medical")
        medical_threshold = get_sigma_immutable_threshold("medical")
        assert medical_gosnn.sigma_immutable_threshold == medical_threshold

        integration = GOSNN3RIntegration(gosnn=medical_gosnn, enable_sliding_window=False)
        _feed_steady_low_scores(integration, steps=30)

        assert medical_gosnn.sigma_immutable_threshold == medical_threshold, (
            "3R feedback overwrote the medical domain's σ_Immutable threshold "
            f"({medical_threshold}) with {medical_gosnn.sigma_immutable_threshold}"
        )
        assert medical_gosnn.ethical_gate.threshold == medical_threshold
        # The instance-local threshold never drops below the domain floor.
        assert integration.state.ethical_threshold >= medical_threshold

    def test_tightening_does_not_overwrite_another_domains_threshold_either(self) -> None:
        """The direction that is allowed is instance-local too."""
        pytest.importorskip("torch")
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            get_sigma_immutable_threshold,
        )

        medical_gosnn = GlobalOmniScalarNetwork(domain="medical")
        medical_threshold = get_sigma_immutable_threshold("medical")
        integration = GOSNN3RIntegration(gosnn=medical_gosnn, enable_sliding_window=False)

        _feed_volatile_scores(integration, steps=400)

        assert integration.enforced_ethical_threshold() == pytest.approx(0.99)
        assert medical_gosnn.ethical_gate.threshold == medical_threshold
        assert medical_gosnn.sigma_immutable_threshold == medical_threshold

    def test_one_integrations_tightening_is_not_anothers(self) -> None:
        """Two integrations over the one shared GOSNN keep separate working thresholds."""
        volatile = GOSNN3RIntegration(enable_sliding_window=False)
        bystander = GOSNN3RIntegration(gosnn=volatile.gosnn, enable_sliding_window=False)
        untouched = bystander.enforced_ethical_threshold()

        _feed_volatile_scores(volatile)

        assert volatile.enforced_ethical_threshold() > untouched
        assert bystander.enforced_ethical_threshold() == untouched


class TestTighteningIsEnforced:
    """The working threshold decides something: a score under it gets no fused score."""

    def test_volatile_scores_raise_the_enforced_threshold_and_it_rejects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        configured = integration.gosnn.ethical_gate.threshold

        _feed_volatile_scores(integration)
        tightened = integration.state.ethical_threshold
        assert tightened > configured

        # A score the configured gate passes and the tightened one does not.
        between = (configured + tightened) / 2.0
        _score_the_gate_at(integration, between, monkeypatch)
        result = integration.process(0.5, 0.5, 0.5)

        assert _withheld(result), (
            f"gate score {between:.4f} is under the tightened threshold {tightened:.4f} "
            f"(configured {configured:.2f}) and a fused score was still released: "
            f"{result.fusion_score!r}"
        )
        assert result.ethical_score == between
        assert configured < result.ethical_threshold_enforced <= 0.99
        assert result.ethical_threshold_enforced == pytest.approx(tightened, abs=1e-3)
        assert integration.gosnn.ethical_gate.threshold == configured

    def test_a_score_over_the_tightened_threshold_is_released(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        _feed_volatile_scores(integration)
        _score_the_gate_at(integration, 0.995, monkeypatch)

        result = integration.process(0.5, 0.5, 0.5)

        assert _released(result)
        assert result.fusion_score == pytest.approx(0.5)
        assert result.ethical_compliance_threshold == result.ethical_threshold_enforced

    def test_the_first_call_already_enforces_the_working_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Medical GOSNN: configured 0.93, working threshold 0.96 from the start."""
        pytest.importorskip("torch")
        from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

        medical_gosnn = GlobalOmniScalarNetwork(domain="medical")
        integration = GOSNN3RIntegration(gosnn=medical_gosnn, enable_sliding_window=False)
        assert medical_gosnn.ethical_gate.threshold == 0.93
        _score_the_gate_at(integration, 0.945, monkeypatch)

        result = integration.process(0.5, 0.5, 0.5)

        assert _withheld(result)
        assert result.ethical_threshold_enforced == pytest.approx(0.96)

    def test_feedback_keeps_tightening_while_the_gate_is_refusing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused call still feeds the stability history: refusal is not a way out."""
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        _score_the_gate_at(integration, 0.5, monkeypatch)

        _feed_volatile_scores(integration)

        assert integration.state.ethical_threshold > integration.gosnn.ethical_gate.threshold
        assert integration.state.last_ethical_gate_passed is False
        assert math.isnan(integration.state.last_fusion_score)

    def test_the_verdict_and_the_enforced_threshold_are_reported(self) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        _feed_volatile_scores(integration)

        report = integration.get_integration_state()

        assert report["thresholds"]["ethical_enforced"] == integration.enforced_ethical_threshold()
        assert report["thresholds"]["ethical_enforced"] > report["thresholds"]["ethical_configured"]
        assert report["performance"]["last_ethical_gate_passed"] in (True, False)


class TestEnforcementNeverDropsBelowTheConfiguredGate:
    def test_steady_low_scores_leave_enforcement_at_the_configured_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        configured = integration.gosnn.ethical_gate.threshold

        _feed_steady_low_scores(integration)

        assert integration.enforced_ethical_threshold() == configured
        _score_the_gate_at(integration, configured - 1e-4, monkeypatch)
        assert _withheld(integration.process(0.1, 0.1, 0.1))
        _score_the_gate_at(integration, configured, monkeypatch)
        assert _released(integration.process(0.1, 0.1, 0.1))

    def test_relaxing_a_tightening_stops_at_the_domain_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Medical: 0.96 working threshold relaxes toward 0.93 and no further."""
        pytest.importorskip("torch")
        from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

        medical_gosnn = GlobalOmniScalarNetwork(domain="medical")
        integration = GOSNN3RIntegration(gosnn=medical_gosnn, enable_sliding_window=False)

        _feed_steady_low_scores(integration, steps=700)

        assert integration.enforced_ethical_threshold() == 0.93
        _score_the_gate_at(integration, 0.9299, monkeypatch)
        assert _withheld(integration.process(0.1, 0.1, 0.1))

    def test_a_working_threshold_set_below_the_floor_is_not_enforced_as_such(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        configured = integration.gosnn.ethical_gate.threshold
        integration.state.ethical_threshold = 0.10

        assert integration.enforced_ethical_threshold() == configured
        _score_the_gate_at(integration, 0.50, monkeypatch)
        assert _withheld(integration.process(0.1, 0.1, 0.1))

    def test_a_shared_gate_lowered_elsewhere_does_not_lower_this_enforcement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        configured = integration.gosnn.ethical_gate.threshold
        monkeypatch.setattr(integration.gosnn.ethical_gate, "threshold", 0.10)
        _score_the_gate_at(integration, 0.50, monkeypatch)

        result = integration.process(0.1, 0.1, 0.1)

        assert _withheld(result)
        assert result.ethical_threshold_enforced == configured

    def test_a_shared_gate_raised_elsewhere_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        monkeypatch.setattr(integration.gosnn.ethical_gate, "threshold", 0.985)
        _score_the_gate_at(integration, 0.98, monkeypatch)

        assert _withheld(integration.process(0.1, 0.1, 0.1))

    @pytest.mark.parametrize("working", [float("nan"), float("inf")])
    def test_an_unusable_working_threshold_passes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, working: float
    ) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        integration.state.ethical_threshold = working
        _score_the_gate_at(integration, 1.0, monkeypatch)

        assert _withheld(integration.process(0.1, 0.1, 0.1))

    def test_the_gates_own_refusal_stands_whatever_the_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate fails closed on a non-finite scalar vector; that is not re-judged."""
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        monkeypatch.setattr(
            integration.gosnn.ethical_gate, "evaluate", lambda scalar_vector: (False, 1.0)
        )

        assert _withheld(integration.process(0.1, 0.1, 0.1))

    def test_a_score_that_is_not_a_number_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        integration = GOSNN3RIntegration(enable_sliding_window=False)
        monkeypatch.setattr(
            integration.gosnn.ethical_gate,
            "evaluate",
            lambda scalar_vector: (True, float("nan")),
        )

        assert _withheld(integration.process(0.1, 0.1, 0.1))
