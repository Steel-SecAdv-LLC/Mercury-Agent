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
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.gosnn_3r_integration import GOSNN3RIntegration


def _feed_steady_low_scores(integration: GOSNN3RIntegration, steps: int = 80) -> None:
    """A stable, low fusion-score stream: the exact input that used to lower the gate."""
    rng = np.random.default_rng(0)
    for _ in range(steps):
        integration.process(0.1 + 0.001 * float(rng.standard_normal()), 0.1, 0.1)


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
        assert integration.state.ethical_threshold >= start


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
