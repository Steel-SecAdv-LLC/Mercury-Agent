"""Stage 3 R6: opt-in, default-off decoupling of the soft eta^Phi multiplier.

Default-off MUST be byte-identical (the eta^Phi multiplier stays in the score
path); when on, the multiplier is removed so a proper-scored monotone calibrator
(MCA) can own the probability. The two fail-closed HARD gates are untouched (I1)
-- this only removes the *soft* in-score multiplier.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation


def _weighted(eq: OmniAvaEquation, r: float, h: float, o: float) -> float:
    w = eq.weights
    return w["w_R"] * r + w["w_H"] * h + w["w_O"] * o


def test_default_off_is_byte_identical() -> None:
    a = OmniAvaEquation().compute(0.8, 0.6, 0.4)
    b = OmniAvaEquation(decouple_ethical_scaling=False).compute(0.8, 0.6, 0.4)
    assert a.fusion_score == b.fusion_score
    # Default path keeps the eta^Phi multiplier.
    eq = OmniAvaEquation()
    expected = _weighted(eq, 0.8, 0.6, 0.4) * (eq.ethical_compliance_threshold**eq.ethical_exponent)
    assert a.fusion_score == pytest.approx(expected, abs=1e-12)


def test_decoupled_removes_eta_multiplier() -> None:
    eq = OmniAvaEquation(decouple_ethical_scaling=True)
    r = eq.compute(0.8, 0.6, 0.4)
    # Decoupled score is exactly the weighted sum (no eta^Phi).
    assert r.fusion_score == pytest.approx(_weighted(eq, 0.8, 0.6, 0.4), abs=1e-12)
    # And it differs from the default (eta < 1 -> eta^Phi shrinks the default score).
    assert r.fusion_score != OmniAvaEquation().compute(0.8, 0.6, 0.4).fusion_score
    # eta is still reported (the hard gates remain the enforcement).
    assert 0.0 < r.ethical_compliance_threshold <= 1.0
