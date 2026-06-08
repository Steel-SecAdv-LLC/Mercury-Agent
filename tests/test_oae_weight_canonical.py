"""R5: one canonical OAE weight derivation (Φ:1:1, phi_sum = Φ + 2).

Pins the single canonical derivation across spec + code and proves the default
fused score is byte-identical to the canonical-weights formula (the R5 change
repointed dead constants / docs / drifted initialisers to this derivation
without touching the runtime fusion path).
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation

PHI = 1.618033988749895
PHI_SUM = PHI + 2.0  # canonical Φ:1:1 denominator


def test_oae_default_weights_are_canonical() -> None:
    eq = OmniAvaEquation()
    assert eq.weights["w_R"] == pytest.approx(PHI / PHI_SUM)
    assert eq.weights["w_H"] == pytest.approx(1.0 / PHI_SUM)
    assert eq.weights["w_O"] == pytest.approx(1.0 / PHI_SUM)
    # H and O are equal unit shares; R leads.
    assert eq.weights["w_H"] == pytest.approx(eq.weights["w_O"])
    assert eq.weights["w_R"] > eq.weights["w_H"]
    assert sum(eq.weights.values()) == pytest.approx(1.0)


def test_centralized_constants_match_runtime_weights() -> None:
    from omni_mercury_engine.core.centralized_constants import FUSION

    eq = OmniAvaEquation()
    assert pytest.approx(eq.weights["w_R"], abs=5e-4) == FUSION.OAE_WEIGHT_R
    assert pytest.approx(eq.weights["w_H"], abs=5e-4) == FUSION.OAE_WEIGHT_H
    assert pytest.approx(eq.weights["w_O"], abs=5e-4) == FUSION.OAE_WEIGHT_O
    # AAFE aliases are kept in lock-step.
    assert FUSION.AAFE_WEIGHT_R == FUSION.OAE_WEIGHT_R
    assert FUSION.AAFE_WEIGHT_H == FUSION.OAE_WEIGHT_H
    assert FUSION.AAFE_WEIGHT_O == FUSION.OAE_WEIGHT_O


def test_oae_fused_score_byte_identical_to_canonical_formula() -> None:
    eq = OmniAvaEquation()
    r = eq.compute(0.8, 0.6, 0.4)
    w = eq.weights
    weighted = w["w_R"] * 0.8 + w["w_H"] * 0.6 + w["w_O"] * 0.4
    expected = weighted * (eq.ethical_compliance_threshold**eq.ethical_exponent)
    assert r.fusion_score == pytest.approx(expected, abs=1e-12)


def test_oae_fused_score_is_deterministic() -> None:
    a = OmniAvaEquation().compute(0.8, 0.6, 0.4).fusion_score
    b = OmniAvaEquation().compute(0.8, 0.6, 0.4).fusion_score
    assert a == b
