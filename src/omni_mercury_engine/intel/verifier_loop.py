# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verifier-in-the-loop: block a generative claim an oracle can refute.

A generative model will state checkable falsehoods with total confidence -- "91
is prime", "the Collatz sequence of 27 never reaches 1", "P and not P is a
tautology", "E = mc is dimensionally consistent". Mercury already ships the
oracle-validated verifier family (:mod:`omni_mercury_engine.verifiers`); this
module routes *applicable* claims found in generated text through those oracles
and, in ``hard`` mode, **blocks emission** when an oracle refutes one.

The design mirrors the existing gate's honesty contract:

* Only claims an oracle can *decide* participate. A claim the oracle cannot
  settle this run (a Collatz budget overrun, an unparseable formula) is recorded
  ``unavailable`` and never blocks -- the loop refutes, it never guesses.
* ``MERCURY_VERIFIER_MODE`` selects the disposition of a refuted claim:
  ``hard`` (default) blocks emission; ``soft`` annotates and allows. Every
  decision is durably audited via
  :func:`~omni_mercury_engine.cognitive.gate_audit.record_gate_decision`.

The measured value (:data:`value_metrics.VALUE_METRICS['verifier_in_loop']`) is
the fraction of oracle-refuted claims blocked -- ``0`` without the loop, ``1`` in
hard mode.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision
from omni_mercury_engine.intel.propositional_claims import (
    Node,
    PropositionalParseError,
    node_is_satisfiable,
    node_is_tautology,
    parse_trailing,
)
from omni_mercury_engine.verifiers import collatz, physics
from omni_mercury_engine.verifiers.primality import is_prime

logger = logging.getLogger(__name__)

#: Bound the Collatz search so a claim never hangs the loop; an overrun is
#: ``unavailable`` (decidable with a larger budget), not a refutation.
_COLLATZ_MAX_STEPS = 200_000


class VerifierMode(Enum):
    """How a refuted claim is dispositioned."""

    HARD = "hard"  # block emission
    SOFT = "soft"  # annotate, allow

    @classmethod
    def from_env(cls, default: VerifierMode | None = None) -> VerifierMode:
        """Resolve the mode from ``MERCURY_VERIFIER_MODE`` (default ``hard``)."""
        raw = os.environ.get("MERCURY_VERIFIER_MODE", "").strip().lower()
        if not raw:
            return default or cls.HARD
        try:
            return cls(raw)
        except ValueError:
            logger.warning("unknown MERCURY_VERIFIER_MODE=%r; defaulting to hard", raw)
            return cls.HARD


class ClaimStatus(Enum):
    """The oracle verdict on a single claim."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNAVAILABLE = "unavailable"  # decidable, but not settled this run


@dataclass(frozen=True)
class ClaimVerdict:
    """An adjudicated claim with full provenance."""

    kind: str  # "primality" | "collatz" | "propositional" | "physics"
    claim_text: str
    status: ClaimStatus
    reason: str
    checker: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping."""
        return {
            "kind": self.kind,
            "claim_text": self.claim_text,
            "status": self.status.value,
            "reason": self.reason,
            "checker": self.checker,
        }


# --------------------------------------------------------------------------- #
# Claim extraction. Each pattern yields a (kind, claim_text, verify()) triple.
# Extraction is conservative: a pattern that does not confidently match is not
# emitted as a claim (better to miss a claim than to fabricate one).
# --------------------------------------------------------------------------- #
_PRIME_RE = re.compile(
    r"\b(\d{1,18})\s+is\s+(?:a\s+)?(not\s+(?:a\s+)?prime|composite|prime)\b",
    re.IGNORECASE,
)
_COLLATZ_RE = re.compile(
    r"[Cc]ollatz\s+(?:sequence|trajectory|orbit)\s+of\s+(\d{1,9})\s+"
    r"(never\s+reaches|does\s+not\s+reach|always\s+reaches|reaches|eventually\s+reaches)\s+1",
    re.IGNORECASE,
)
_TAUT_RE = re.compile(r"([A-Za-z0-9_()~!&|^<>=\-\s]+?)\s+is\s+a\s+tautology\b", re.IGNORECASE)
_CONTRA_RE = re.compile(
    r"([A-Za-z0-9_()~!&|^<>=\-\s]+?)\s+is\s+(?:a\s+contradiction|unsatisfiable)\b",
    re.IGNORECASE,
)
_SAT_RE = re.compile(r"([A-Za-z0-9_()~!&|^<>=\-\s]+?)\s+is\s+satisfiable\b", re.IGNORECASE)

#: Named physics relations recognized in text, mapped to a checkable relation.
_PHYSICS_RELATIONS = {
    "mass_energy": ("E = mc^2", physics.mass_energy_equivalence),
    "wrong_mass_energy": ("E = mc", physics.dimensionally_wrong_mass_energy),
    "newton": ("F = ma", physics.newtons_second_law),
}
_PHYSICS_RE = re.compile(
    r"(E\s*=\s*mc\s*\^?\s*2|E\s*=\s*mc(?!\s*\^?\s*2)|F\s*=\s*ma)\b[^.]*?"
    r"dimensionally\s+consistent",
    re.IGNORECASE,
)


def _verify_primality(n: int, asserted_prime: bool, text: str) -> ClaimVerdict:
    actual = is_prime(n)
    status = ClaimStatus.CONFIRMED if actual == asserted_prime else ClaimStatus.REFUTED
    reason = (
        f"{n} is {'prime' if actual else 'composite'}; "
        f"claim asserted {'prime' if asserted_prime else 'composite'}"
    )
    return ClaimVerdict("primality", text, status, reason, "deterministic_miller_rabin")


def _verify_collatz(n: int, asserted_reaches_one: bool, text: str) -> ClaimVerdict:
    if n < 1:
        return ClaimVerdict(
            "collatz", text, ClaimStatus.UNAVAILABLE, f"n={n} out of domain (n>=1)", "collatz_map"
        )
    traj = collatz.compute_trajectory(n, max_steps=_COLLATZ_MAX_STEPS)
    if traj is None:
        # Budget exhausted: decidable with more steps -> unavailable, never a refutation.
        return ClaimVerdict(
            "collatz",
            text,
            ClaimStatus.UNAVAILABLE,
            f"{n} did not reach 1 within {_COLLATZ_MAX_STEPS} steps (budget)",
            "collatz_map",
        )
    reaches_one = traj[-1] == 1
    status = ClaimStatus.CONFIRMED if reaches_one == asserted_reaches_one else ClaimStatus.REFUTED
    reason = (
        f"trajectory of {n} {'reaches' if reaches_one else 'did not reach'} 1 "
        f"in {len(traj) - 1} steps; claim asserted "
        f"{'reaches' if asserted_reaches_one else 'never reaches'} 1"
    )
    return ClaimVerdict("collatz", text, status, reason, "collatz_map")


def _verify_propositional(node: Node, asserted: bool, prop: str, text: str) -> ClaimVerdict:
    """Adjudicate a parsed propositional claim (``prop`` in {tautology, satisfiable})."""
    try:
        actual = node_is_tautology(node) if prop == "tautology" else node_is_satisfiable(node)
    except PropositionalParseError as exc:  # bounded-variable overflow -> unavailable
        return ClaimVerdict("propositional", text, ClaimStatus.UNAVAILABLE, str(exc), "dpll")
    status = ClaimStatus.CONFIRMED if actual == asserted else ClaimStatus.REFUTED
    reason = f"formula {prop}={actual}; claim asserted {prop}={asserted}"
    return ClaimVerdict("propositional", text, status, reason, "dpll")


def _verify_physics(relation_key: str, text: str) -> ClaimVerdict:
    label, factory = _PHYSICS_RELATIONS[relation_key]
    verdict = physics.verify_relation(factory())
    # The claim asserts the relation IS dimensionally consistent.
    status = ClaimStatus.CONFIRMED if verdict.valid else ClaimStatus.REFUTED
    return ClaimVerdict("physics", text, status, verdict.reason, verdict.checker)


def _match_physics_key(formula: str) -> str:
    """Map a matched physics formula string to a relation key."""
    compact = re.sub(r"\s+", "", formula.lower())
    if compact in ("e=mc^2", "e=mc2"):
        return "mass_energy"
    if compact == "e=mc":
        return "wrong_mass_energy"
    return "newton"  # F = ma


def extract_and_verify(text: str) -> list[ClaimVerdict]:
    """Extract every recognized claim from ``text`` and adjudicate each by oracle."""
    verdicts: list[ClaimVerdict] = []

    for match in _PRIME_RE.finditer(text):
        n = int(match.group(1))
        asserted_prime = match.group(2).lower() == "prime"
        verdicts.append(_verify_primality(n, asserted_prime, match.group(0)))

    for match in _COLLATZ_RE.finditer(text):
        n = int(match.group(1))
        verb = match.group(2).lower()
        asserted_reaches_one = "reaches" in verb and "never" not in verb and "not" not in verb
        verdicts.append(_verify_collatz(n, asserted_reaches_one, match.group(0)))

    for regex, prop, asserted in (
        (_TAUT_RE, "tautology", True),
        (_CONTRA_RE, "satisfiable", False),  # "contradiction/unsatisfiable" == not satisfiable
        (_SAT_RE, "satisfiable", True),
    ):
        for match in regex.finditer(text):
            # Trim leading prose to the longest parseable trailing formula; a
            # capture that is not a formula at all yields no claim (conservative).
            node = parse_trailing(match.group(1))
            if node is None:
                continue
            verdicts.append(_verify_propositional(node, asserted, prop, match.group(0)))

    for match in _PHYSICS_RE.finditer(text):
        key = _match_physics_key(match.group(1))
        verdicts.append(_verify_physics(key, match.group(0)))

    return verdicts


# --------------------------------------------------------------------------- #
# The loop.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmissionDecision:
    """The verifier loop's disposition of a candidate emission."""

    allowed: bool
    mode: VerifierMode
    verdicts: tuple[ClaimVerdict, ...] = ()
    blocked_claims: tuple[ClaimVerdict, ...] = ()
    flagged_claims: tuple[ClaimVerdict, ...] = ()

    @property
    def refuted(self) -> tuple[ClaimVerdict, ...]:
        """All oracle-refuted claims found (regardless of mode)."""
        return tuple(v for v in self.verdicts if v.status is ClaimStatus.REFUTED)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly summary."""
        return {
            "allowed": self.allowed,
            "mode": self.mode.value,
            "n_claims": len(self.verdicts),
            "n_refuted": len(self.refuted),
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


@dataclass
class VerifierLoop:
    """Routes generative claims through oracles and gates emission on refutation."""

    mode: VerifierMode = field(default_factory=VerifierMode.from_env)

    def review(self, text: str) -> tuple[ClaimVerdict, ...]:
        """Adjudicate every recognized claim in ``text`` (no gating)."""
        return tuple(extract_and_verify(text))

    def guard_emission(self, text: str, *, source: str = "generation") -> EmissionDecision:
        """Adjudicate ``text`` and decide whether it may be emitted.

        In ``hard`` mode a single oracle-refuted claim blocks emission
        (``allowed=False``); in ``soft`` mode refuted claims are flagged but the
        emission is allowed. Unavailable claims never block. **Every** disposition
        -- pass, hard block, and soft flag -- is durably audited, so the audit
        trail matches the module contract and an auditor can distinguish "checked
        and passed" from "never checked".
        """
        verdicts = self.review(text)
        refuted = tuple(v for v in verdicts if v.status is ClaimStatus.REFUTED)

        if not refuted:
            # Audit the clean disposition too, not only refusals: without this the
            # (common) allowed path is silently absent from the trail, contradicting
            # the module's "every decision is durably audited" contract.
            record_gate_decision(
                decision="verifier_pass",
                source=f"verifier_loop:{source}",
                disposition="allow",
                signals=("verifier_in_loop",),
                reason=f"{len(verdicts)} oracle-checkable claim(s); none refuted (emission allowed)",
                extra={"claims": [v.as_dict() for v in verdicts]},
            )
            return EmissionDecision(allowed=True, mode=self.mode, verdicts=verdicts)

        if self.mode is VerifierMode.HARD:
            record_gate_decision(
                decision="verifier_block",
                source=f"verifier_loop:{source}",
                disposition="hard_refuse",
                signals=("verifier_in_loop", "oracle_refuted"),
                reason=f"{len(refuted)} oracle-refuted claim(s); emission blocked",
                extra={"claims": [v.as_dict() for v in refuted]},
            )
            return EmissionDecision(
                allowed=False, mode=self.mode, verdicts=verdicts, blocked_claims=refuted
            )

        record_gate_decision(
            decision="verifier_flag",
            source=f"verifier_loop:{source}",
            disposition="allow_log",
            signals=("verifier_in_loop", "oracle_refuted", "soft_mode"),
            reason=f"{len(refuted)} oracle-refuted claim(s); flagged (soft mode)",
            extra={"claims": [v.as_dict() for v in refuted]},
        )
        return EmissionDecision(
            allowed=True, mode=self.mode, verdicts=verdicts, flagged_claims=refuted
        )


def false_claim_block_rate(loop: VerifierLoop, refutable_texts: Iterable[str]) -> float:
    """Fraction of known-refutable emissions the loop blocks (the value metric).

    Each text in ``refutable_texts`` is expected to contain at least one
    oracle-refutable claim; the rate is how many the loop actually blocks. In
    hard mode a correct loop returns ``1.0``; in soft mode ``0.0`` (flags, never
    blocks). A text with no refutable claim is skipped (does not dilute the rate).
    """
    texts = list(refutable_texts)
    considered = 0
    blocked = 0
    for text in texts:
        decision = loop.guard_emission(text)
        if not decision.refuted:
            continue
        considered += 1
        if not decision.allowed:
            blocked += 1
    return blocked / considered if considered else 0.0


__all__ = [
    "ClaimStatus",
    "ClaimVerdict",
    "EmissionDecision",
    "VerifierLoop",
    "VerifierMode",
    "extract_and_verify",
    "false_claim_block_rate",
]
