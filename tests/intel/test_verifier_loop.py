# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verifier-in-the-loop: oracle-refuted claims block emission (hard mode).

Covers all four symbolic verifier families (primality, Collatz, propositional,
physics), the hard/soft disposition, and the stream's value metric (every
oracle-refuted claim blocked in hard mode).
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.intel.value_metrics import VALUE_METRICS
from omni_mercury_engine.intel.verifier_loop import (
    ClaimStatus,
    VerifierLoop,
    VerifierMode,
    extract_and_verify,
    false_claim_block_rate,
)

# One true and one false claim per verifier family.
_TRUE_CLAIMS = [
    "7 is prime.",
    "The Collatz trajectory of 6 reaches 1 quickly.",
    "Clearly P or not P is a tautology.",
    "The relation E = mc^2 is dimensionally consistent.",
]
_FALSE_CLAIMS = [
    "91 is prime and useful.",  # 91 = 7*13
    "The Collatz sequence of 27 never reaches 1.",
    "Note that P and not P is a tautology.",
    "As shown, E = mc is dimensionally consistent.",  # dimensionally wrong
]


@pytest.mark.parametrize("text", _TRUE_CLAIMS)
def test_true_claims_are_allowed(text: str) -> None:
    loop = VerifierLoop(mode=VerifierMode.HARD)
    decision = loop.guard_emission(text)
    assert decision.allowed
    assert all(v.status is ClaimStatus.CONFIRMED for v in decision.verdicts)


@pytest.mark.parametrize("text", _FALSE_CLAIMS)
def test_false_claims_block_in_hard_mode(text: str) -> None:
    loop = VerifierLoop(mode=VerifierMode.HARD)
    decision = loop.guard_emission(text)
    assert not decision.allowed
    assert decision.blocked_claims
    assert all(v.status is ClaimStatus.REFUTED for v in decision.refuted)


@pytest.mark.parametrize("text", _FALSE_CLAIMS)
def test_false_claims_flagged_but_allowed_in_soft_mode(text: str) -> None:
    loop = VerifierLoop(mode=VerifierMode.SOFT)
    decision = loop.guard_emission(text)
    assert decision.allowed
    assert decision.flagged_claims  # flagged, not blocked


def test_each_family_covered() -> None:
    kinds = {v.kind for text in _FALSE_CLAIMS for v in extract_and_verify(text)}
    assert kinds == {"primality", "collatz", "propositional", "physics"}


def test_unavailable_claim_never_blocks() -> None:
    # A Collatz claim with an out-of-domain n is unavailable, not refuted.
    loop = VerifierLoop(mode=VerifierMode.HARD)
    decision = loop.guard_emission("The Collatz sequence of 0 reaches 1.")
    assert decision.allowed
    assert all(v.status is not ClaimStatus.REFUTED for v in decision.verdicts)


def test_value_metric_block_rate() -> None:
    hard = VerifierLoop(mode=VerifierMode.HARD)
    soft = VerifierLoop(mode=VerifierMode.SOFT)
    target = VALUE_METRICS["verifier_in_loop"].target
    assert false_claim_block_rate(hard, _FALSE_CLAIMS) == target == 1.0
    assert false_claim_block_rate(soft, _FALSE_CLAIMS) == 0.0


def test_mode_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_VERIFIER_MODE", "soft")
    assert VerifierMode.from_env() is VerifierMode.SOFT
    monkeypatch.setenv("MERCURY_VERIFIER_MODE", "nonsense")
    assert VerifierMode.from_env() is VerifierMode.HARD  # unknown -> hard
    monkeypatch.delenv("MERCURY_VERIFIER_MODE", raising=False)
    assert VerifierMode.from_env() is VerifierMode.HARD
