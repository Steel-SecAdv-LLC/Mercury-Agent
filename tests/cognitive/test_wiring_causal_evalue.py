# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired E-value sensitivity analysis.

Hand-computed against VanderWeele & Ding (2017): ``RR = exp(0.91 d)`` with the
standardized effect ``d`` using the treatment-conditional pooled SD for a
binary treatment, then ``E = RR + sqrt(RR (RR - 1))``, capped at 3.0.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

# Binary treatment (col 0); outcome (col 1) has within-group variance 4.0 in
# both arms, so the pooled SD is exactly 2.0.
_DATA = np.array(
    [
        [0.0, 0.0],
        [0.0, 2.0],
        [0.0, 4.0],
        [1.0, 10.0],
        [1.0, 12.0],
        [1.0, 14.0],
    ]
)


def _evalue(effect: float, data: np.ndarray = _DATA) -> float:
    engine = CausalDiscoveryEngine()
    return engine._sensitivity_analysis(
        data, treatment_idx=0, outcome_idx=1, observed_effect=effect
    )


def _closed_form(effect: float, outcome_std: float) -> float:
    d = abs(effect) / outcome_std
    rr = np.exp(0.91 * d)
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def test_pooled_sd_and_evalue_formula() -> None:
    # Within-group variance is 4.0 in both arms, so the pooled SD is 2.0 and
    # d = 1.0 / 2.0 = 0.5. The result must match the closed form at that SD...
    assert abs(_evalue(1.0) - _closed_form(1.0, outcome_std=2.0)) < 1e-9
    # ...and must NOT match the marginal-SD version (col std ~= 5.26), which
    # confirms the treatment-conditional pooling is what drives d.
    marginal_std = float(np.std(_DATA[:, 1]))
    assert abs(_evalue(1.0) - _closed_form(1.0, outcome_std=marginal_std)) > 0.5


def test_null_effect_gives_unit_evalue() -> None:
    # d = 0 -> RR = 1 -> E = 1 + sqrt(0) = 1.0 (any confounding nullifies it).
    assert abs(_evalue(0.0) - 1.0) < 1e-9


def test_large_effect_is_capped_at_three() -> None:
    # d = 10/2 = 5 -> E is enormous; the consumer-range cap holds it at 3.0.
    assert _evalue(10.0) == 3.0


def test_zero_variance_outcome_returns_one() -> None:
    flat = np.array([[0.0, 5.0], [0.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
    # No outcome spread for a confounder to act against -> E = 1.0.
    assert _evalue(3.0, data=flat) == 1.0


def test_evalue_is_monotone_in_effect() -> None:
    assert _evalue(0.5) < _evalue(1.0) < _evalue(1.8)
