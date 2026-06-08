# Copyright (C) 2025 Steel Security Advisors LLC
"""The unified three-state honesty contract -- one invariant across both repos.

``GROUNDED`` / ``UNAVAILABLE`` / ``UNDECIDABLE`` is the single vocabulary that
governs both the **oracle side** (this repo's :mod:`omni_mercury_engine.verifiers`)
and the **governance side** (the companion ``FIND-YOU-ARC-CODE`` repo, which
mirrors :class:`ThreeState` member-for-member).  The string values are the wire
format, so a serialised state round-trips across the repo boundary unchanged.

The oracle side historically spoke a finer four-word vocabulary --
``confirmed`` / ``refuted`` / ``inconclusive`` / ``unavailable`` -- which
*collapsed* two genuinely different reasons for not deciding into one bucket.
:func:`three_state_of` reconciles that vocabulary onto the invariant and the
collapse is undone by **cause** (see its docstring for the per-call-site
citations):

* ``confirmed`` and ``refuted`` are both :attr:`ThreeState.GROUNDED` -- a
  decision was reached; the value (1.0 / 0.0) carries it.
* ``inconclusive`` / ``unavailable`` are :attr:`ThreeState.UNAVAILABLE`
  *whenever they mean "decidable, but not produced this run"* -- a bounded
  search hit its budget, an input was absent, or a checker was not installed.
* :attr:`ThreeState.UNDECIDABLE` is reserved for claims with **no decision
  procedure in principle** -- the universally-quantified open conjectures an
  instance oracle can never settle.  The verifier family only ever checks
  *instances* (all decidable), so it never emits this from the instance
  paths; it is produced by :meth:`MysteryRegistry.submit_undecidable`.
"""

from __future__ import annotations

from enum import Enum


class ThreeState(Enum):
    """The cross-repo verdict invariant: a decision, a deferral, or an impossibility.

    Mutually exclusive and exhaustive over every verdict an oracle can
    reach.  Mirrored member-for-member (names + string values) in the
    companion ``FIND-YOU-ARC-CODE`` repo so the contract is one vocabulary,
    not two lookalike enums.
    """

    #: A decision was reached and the value carries it.  On the oracle side
    #: this is both ``confirmed`` (oracle proved the claim; grounds 1.0) and
    #: ``refuted`` (oracle disproved it; grounds 0.0): in both the oracle
    #: *decided*, so a scalar is genuinely grounded in that decision.
    GROUNDED = "grounded"

    #: Decidable in principle, but not produced this run.  The decision
    #: procedure exists and the capability is real, but the oracle/checker
    #: was absent, the input was not fed, or a bounded search exhausted its
    #: budget without reaching the committed frontier.  Registers nothing
    #: THIS run; re-running with the missing piece (or a larger budget)
    #: could decide it.
    UNAVAILABLE = "unavailable"

    #: No decision procedure exists in principle.  The oracle can never
    #: decide this -- a universally-quantified claim over an infinite domain
    #: that an instance checker cannot settle (Collatz-in-general, the
    #: infinitude of twin primes, any Millennium-class open problem).
    #: Registers nothing, EVER.
    UNDECIDABLE = "undecidable"


#: Canonical problems with no decision procedure available to an instance
#: oracle -- the universal/general statements behind the instance verifiers.
#: Used by :meth:`MysteryRegistry.submit_undecidable` so a known-undecidable
#: claim is recorded as :attr:`ThreeState.UNDECIDABLE` (registers nothing,
#: ever), never confused with an UNAVAILABLE instance that simply was not
#: run this time.
KNOWN_UNDECIDABLE_IN_GENERAL: dict[str, str] = {
    "collatz_general": (
        "Does every positive integer's Collatz trajectory reach 1? Open; an "
        "instance oracle settles single n, never the universal claim."
    ),
    "twin_prime_infinitude": (
        "Are there infinitely many twin primes? Open; an instance oracle "
        "checks single pairs, never the infinitude."
    ),
    "goldbach_general": (
        "Is every even integer > 2 a sum of two primes? Open; an instance "
        "oracle checks single n, never the universal claim."
    ),
    "riemann_hypothesis": (
        "Do all non-trivial zeros of the Riemann zeta function lie on "
        "Re(s)=1/2? Millennium Prize problem; no checkable proof object."
    ),
    "p_vs_np": "Does P = NP? Millennium Prize problem; no checkable proof object.",
}

#: The oracle-side four-word vocabulary mapped onto the invariant.  This is
#: the literal reconciliation table; :func:`three_state_of` wraps it with a
#: strict-unknown guard.
_VERIFIER_STATUS_TO_STATE: dict[str, ThreeState] = {
    "confirmed": ThreeState.GROUNDED,
    "refuted": ThreeState.GROUNDED,
    "inconclusive": ThreeState.UNAVAILABLE,
    "unavailable": ThreeState.UNAVAILABLE,
    "undecidable": ThreeState.UNDECIDABLE,
}


def three_state_of(verifier_status: str) -> ThreeState:
    """Map a verifier ledger status onto the unified :class:`ThreeState`.

    The ``inconclusive`` / ``unavailable`` -> :attr:`ThreeState.UNAVAILABLE`
    classification is by **cause**, and in the verifier family every such
    result is a *decidable instance not produced this run* -- never a
    problem that is undecidable in principle.  Per call site
    (``omni_mercury_engine.verifiers.registry``):

    * ``submit_goldbach`` ``inconclusive`` (registry.py: ``find_partition``
      returned ``None``) -- the candidate generator proposed no partition
      for this ``n`` (odd / <=2 input, or -- unreachably -- a counterexample).
      The instance is decidable by exhaustive search: UNAVAILABLE.
    * ``submit_collatz`` ``inconclusive`` (registry.py: ``compute_trajectory``
      returned ``None``) -- the bounded run hit ``max_steps`` (collatz.py:
      "non-termination is only semi-decidable").  A larger budget could
      decide this n: UNAVAILABLE.  (The *general* Collatz conjecture is the
      UNDECIDABLE case, reached only via :meth:`submit_undecidable`.)
    * ``submit_theorem`` ``unavailable`` (registry.py: ``verify_lean_proof``
      returned ``available=False``) -- no Lean toolchain on PATH.  The
      theorem (e.g. ``2+2=4``) is decidable once Lean is installed:
      UNAVAILABLE.

    Args:
        verifier_status: One of ``confirmed`` / ``refuted`` /
            ``inconclusive`` / ``unavailable`` / ``undecidable``.

    Returns:
        The corresponding :class:`ThreeState`.

    Raises:
        ValueError: If ``verifier_status`` is not a known status -- callers
            must extend the reconciliation table deliberately, never let an
            unrecognised status fall through to a wrong state.

    """
    try:
        return _VERIFIER_STATUS_TO_STATE[verifier_status]
    except KeyError:
        raise ValueError(
            f"unknown verifier status {verifier_status!r}; expected one of "
            f"{sorted(_VERIFIER_STATUS_TO_STATE)}"
        ) from None


__all__ = [
    "KNOWN_UNDECIDABLE_IN_GENERAL",
    "ThreeState",
    "three_state_of",
]
