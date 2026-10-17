# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The soft ``eta ** Phi`` multiplier is out of the fused-score path, for good.

It used to be removable only via an opt-in ``decouple_ethical_scaling`` flag that
shipped default-off, so the shipped behaviour was still ``weighted_sum * eta**Phi``.
That factor enforced nothing: ``eta`` is the configured
``ethical_compliance_threshold`` on every production path (``benevolence_score``
is passed only from tests), so it was a per-instance constant --
``0.96 ** 1.618 = 0.9359`` -- shaving 6.4 % off every score under a name that
implied an ethics control was acting. Being constant it could not add
discrimination; it only moved absolute scores against fixed thresholds.

The flag is gone along with the multiplier, so there is no configuration in which
the old behaviour returns. Enforcement is unchanged and lives in
``cognitive/decision_gate.py``.
"""

from __future__ import annotations

import inspect

import pytest

from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation


def _weighted(eq: OmniAvaEquation, r: float, h: float, o: float) -> float:
    w = eq.weights
    return w["w_R"] * r + w["w_H"] * h + w["w_O"] * o


def test_fused_score_is_the_weighted_sum() -> None:
    eq = OmniAvaEquation()
    result = eq.compute(0.8, 0.6, 0.4)
    assert result.fusion_score == pytest.approx(_weighted(eq, 0.8, 0.6, 0.4), abs=1e-12)


def test_the_eta_multiplier_is_not_applied() -> None:
    """Explicitly assert the old value is *not* what comes back."""
    eq = OmniAvaEquation()
    superseded = _weighted(eq, 0.8, 0.6, 0.4) * (
        eq.ethical_compliance_threshold**eq.ethical_exponent
    )

    result = eq.compute(0.8, 0.6, 0.4)

    assert result.fusion_score != pytest.approx(superseded, abs=1e-12)
    assert result.fusion_score > superseded  # eta < 1, so the factor shrank scores


def test_there_is_no_flag_that_restores_the_multiplier() -> None:
    """The removal is unconditional; no constructor argument brings it back."""
    params = set(inspect.signature(OmniAvaEquation.__init__).parameters)
    assert "decouple_ethical_scaling" not in params
    assert not any("decouple" in p for p in params)


@pytest.mark.parametrize("threshold", [0.93, 0.95, 0.96, 0.99])
def test_the_configured_threshold_no_longer_scales_the_score(threshold: float) -> None:
    """Two instances differing only in eta must produce the same fused score.

    This is the property the multiplier broke: eta is a governance setting, and a
    governance setting must not silently move a detection number.
    """
    baseline = OmniAvaEquation(ethical_compliance_threshold=0.96)
    other = OmniAvaEquation(ethical_compliance_threshold=threshold)

    assert other.compute(0.8, 0.6, 0.4).fusion_score == pytest.approx(
        baseline.compute(0.8, 0.6, 0.4).fusion_score, abs=1e-12
    )


def test_eta_is_still_reported_as_provenance() -> None:
    """Removing it from the arithmetic must not remove it from the record."""
    result = OmniAvaEquation().compute(0.8, 0.6, 0.4)
    assert 0.0 < result.ethical_compliance_threshold <= 1.0


def test_dominance_proof_describes_the_equation_as_implemented() -> None:
    """The proof dict used to report the removed factor and an unmeasured claim.

    It listed ``* η_Ethical^Φ`` in the equation, presented the NSL-KDD F1 target
    (0.92 vs 0.797) as an "improvement_factor" and claimed a "10-15%"
    false-positive reduction. It now matches :meth:`OmniAvaEquation.compute`.
    """
    eq = OmniAvaEquation()
    proof = eq.get_dominance_proof()
    assert proof["equation"] == "A = w_R * R(x) + w_H * H(omega) + w_O * O(theta)"
    assert proof["ethical_scaling"]["applied"] is False
    assert proof["performance_claims"] is None
    flat = repr(proof)
    for retired in ("η_Ethical^Φ", "improvement_factor", "fp_reduction", "nsl_kdd"):
        assert retired not in flat
    # The equation string is what compute() returns for these inputs.
    assert eq.compute(0.8, 0.6, 0.4).fusion_score == pytest.approx(
        _weighted(eq, 0.8, 0.6, 0.4), abs=1e-12
    )
