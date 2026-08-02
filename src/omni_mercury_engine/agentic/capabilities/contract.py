# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Capability-as-contract envelope for Mercury's general-purpose capabilities.

Mercury's capability layer already enforces three safety invariants, but as
*scattered, hand-rolled* logic inside each method: ``WebResearcher`` returns a
typed transparent-negative instead of raising when the network fails; the
``GeneralAssistant`` withholds an uncited answer on a provenance-required topic;
the ``ExtractiveSynthesizer`` redacts operational content sentence by sentence.
Nothing declared those guarantees as a *contract* or proved they hold, so a
future refactor could silently drop one and no test would notice.

:func:`capability_contract` turns each guarantee into a declared, runtime-enforced
postcondition attached to the capability method. It does **not** invent new
behaviour -- it reuses the capability's own transparent-negative shapes -- and it never
weakens a result: enforcement can only *downgrade* an output toward refusal /
redaction, never upgrade one toward permit. The four invariants:

* :attr:`Invariant.FAIL_CLOSED` -- an unexpected exception must not escape as an
  unguarded error; it becomes the capability's typed transparent-negative (a
  ``FetchResult`` with ``error`` set, a refused ``ResearchReport``, an empty
  summary), audited before it is returned. A capability can never crash a caller
  into an unguarded path.
* :attr:`Invariant.CITE_OR_REFUSE` -- emitted content on a provenance-required
  (hazardous) topic must carry citations; an uncited emission is downgraded to a
  refusal. A plain-``ALLOW`` benign result legitimately needs no citation, so the
  invariant only bites when provenance is required (matching the existing
  ``ALLOW_PROVENANCE`` output-boundary rule).
* :attr:`Invariant.MONOTONE_HARM` -- output harm is monotonically bounded by the
  gate: every emitted span must pass the capability's sentence gate or already be
  the redaction notice. A regression that let gate-unsafe content through is
  caught and redacted, so adding harmful input can only *increase* redaction,
  never decrease it.
* :attr:`Invariant.GATED_BOUNDARY` -- the *only* precondition invariant. The real
  decision (surface, domain, request, payload) is run through the single
  fail-closed harm-uplift choke point in
  :mod:`~omni_mercury_engine.cognitive.decision_gate` **before** the body
  executes; a blocking disposition raises ``EthicalConstraintViolationError``
  out of the surface. Unlike the postcondition invariants this one deliberately
  *raises* rather than repairing: there is no safe partial result for a decision
  the harm policy refuses to make, and a repaired-to-empty return would be a
  fail-open path dressed as a transparent negative. It is enforced ahead of the
  ``FAIL_CLOSED`` guard so the refusal can never be swallowed into a sentinel.

Enforcement is itself fail-closed and fail-safe: a violation is logged and
durably audited via :func:`~omni_mercury_engine.cognitive.gate_audit.record_gate_decision`,
then repaired to the safe result. A *misconfiguration* (declaring an invariant
without the hooks it needs) raises :class:`ContractViolation` at decoration time
-- loudly, at import, never at serve time.

The decorator stamps :data:`CONTRACT_MARKER` on the wrapped function and registers
it, so a meta-test can assert the annotation is still present (deleting it fails
CI) and that every declared invariant holds under adversarial input.
"""

from __future__ import annotations

import functools
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, TypeVar, cast, runtime_checkable

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

#: Attribute stamped on a contracted function; value is ``frozenset[Invariant]``.
CONTRACT_MARKER = "__capability_contract__"


class Invariant(Enum):
    """A safety invariant a capability can be contracted to uphold at runtime."""

    FAIL_CLOSED = "fail_closed"
    CITE_OR_REFUSE = "cite_or_refuse"
    MONOTONE_HARM = "monotone_harm"
    GATED_BOUNDARY = "gated_boundary"


class ContractViolation(RuntimeError):
    """A capability contract is *misconfigured* (a decoration-time programmer error).

    Raised only at decoration time when a declared invariant is missing the hook
    it needs. Runtime invariant breaches are never raised -- they are fail-closed
    (audited and repaired to the safe result), because raising into the hot
    enforcement path would itself be a way to break the control.
    """


@runtime_checkable
class SupportsRefusal(Protocol):
    """Structural type of a capability result that can express a transparent refusal.

    The "type" half of the capability-as-contract envelope: any result carrying a
    boolean ``refused`` (e.g. :class:`~omni_mercury_engine.agentic.capabilities.assistant.ResearchReport`)
    structurally satisfies it, letting callers and static analysis recognise a
    valid fail-closed shape without importing the concrete class.
    """

    @property
    def refused(self) -> bool:  # pragma: no cover - structural protocol
        """True when the capability refused to produce a substantive result."""
        ...


def is_honest_negative(result: object) -> bool:
    """True when ``result`` structurally advertises a refusal (a :class:`SupportsRefusal`)."""
    return isinstance(result, SupportsRefusal) and bool(result.refused)


# Registry of every contracted capability, in decoration order. Enables an
# enumerating meta-test ("are all core capabilities still contracted?").
_REGISTRY: list[tuple[str, frozenset[Invariant]]] = []


def registered_contracts() -> tuple[tuple[str, frozenset[Invariant]], ...]:
    """Return ``(qualified_label, invariants)`` for every contracted capability."""
    return tuple(_REGISTRY)


def capability_contract(
    *invariants: Invariant,
    on_error: Callable[[BaseException, tuple[Any, ...], dict[str, Any]], Any] | None = None,
    emitted: Callable[[Any, Any], bool] | None = None,
    provenance_required: Callable[[Any, Any], bool] | None = None,
    cited: Callable[[Any, Any], bool] | None = None,
    refuse: Callable[[Any, Any], Any] | None = None,
    harm_residue: Callable[[Any, Any], Sequence[str]] | None = None,
    redact: Callable[[Any, Any], Any] | None = None,
    boundary_subject: Callable[[tuple[Any, ...], dict[str, Any]], Any] | None = None,
    label: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Contract a capability method to uphold ``invariants`` at runtime.

    Three of the four are postconditions checked on the return value;
    :attr:`Invariant.GATED_BOUNDARY` is a precondition checked before the body
    runs.

    The hooks let the decorator stay type-agnostic: each is supplied at the
    annotation site (which knows the concrete result type) and operates on the
    method's return value plus, where useful, the bound instance.

    Args:
        invariants: One or more :class:`Invariant` this capability must uphold.
        on_error: Required for :attr:`Invariant.FAIL_CLOSED`. Given
            ``(exc, args, kwargs)`` (``args[0]`` is the bound instance), returns
            the capability's typed transparent-negative for a caught exception.
        emitted: Required for :attr:`Invariant.CITE_OR_REFUSE`. ``(result, instance)
            -> bool``: whether the result emitted substantive content (vs a
            refusal/unavailable).
        provenance_required: Required for :attr:`Invariant.CITE_OR_REFUSE`.
            ``(result, instance) -> bool``: whether the topic mandates citations.
        cited: Required for :attr:`Invariant.CITE_OR_REFUSE`. ``(result, instance)
            -> bool``: whether the emitted result actually carries citations.
        refuse: Required for :attr:`Invariant.CITE_OR_REFUSE`. ``(result, instance)
            -> result``: downgrade an uncited emission to a refusal.
        harm_residue: Required for :attr:`Invariant.MONOTONE_HARM`.
            ``(result, instance) -> spans``: the gate-unsafe spans present in the
            output (empty when the output is fully bounded by the gate).
        redact: Required for :attr:`Invariant.MONOTONE_HARM`. ``(result, instance)
            -> result``: redact the residue.
        boundary_subject: Required for :attr:`Invariant.GATED_BOUNDARY`.
            ``(args, kwargs) -> DecisionSubject`` (``args[0]`` is the bound
            instance): build the *real* decision -- surface, domain, request,
            payload -- from the actual call. It must never fabricate a synthetic
            keyword string; that is the failure mode this invariant exists to
            prevent.
        label: Audit/registry label (defaults to the method's ``__qualname__``).

    Returns:
        A decorator that wraps the method, enforcing the invariants and stamping
        :data:`CONTRACT_MARKER`.

    Raises:
        ContractViolation: at decoration time, if a declared invariant is missing
            a required hook, or no invariant is declared.
    """
    inv_set = frozenset(invariants)
    if not inv_set:
        raise ContractViolation("capability_contract requires at least one Invariant")
    if Invariant.FAIL_CLOSED in inv_set and on_error is None:
        raise ContractViolation("FAIL_CLOSED requires an on_error sentinel")
    if Invariant.CITE_OR_REFUSE in inv_set and not all(
        (emitted, provenance_required, cited, refuse)
    ):
        raise ContractViolation(
            "CITE_OR_REFUSE requires emitted, provenance_required, cited and refuse hooks"
        )
    if Invariant.MONOTONE_HARM in inv_set and not all((harm_residue, redact)):
        raise ContractViolation("MONOTONE_HARM requires harm_residue and redact hooks")
    if Invariant.GATED_BOUNDARY in inv_set and boundary_subject is None:
        raise ContractViolation("GATED_BOUNDARY requires a boundary_subject hook")

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        contract_label: str = label or str(getattr(func, "__qualname__", "?"))

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            instance = args[0] if args else None
            if Invariant.GATED_BOUNDARY in inv_set:
                # Precondition, and deliberately OUTSIDE the FAIL_CLOSED guard:
                # a refused decision boundary must propagate as a refusal, not
                # be converted into a transparent-negative sentinel that reads
                # to a caller like "nothing found".
                _enforce_boundary(args, kwargs, contract_label)
            if Invariant.FAIL_CLOSED in inv_set:
                # Postcondition enforcement runs INSIDE the guard, so a raising
                # hook fails closed to the transparent-negative too -- the whole path
                # can never crash a caller into an unguarded result.
                try:
                    result = func(*args, **kwargs)
                    return _enforce_postconditions(result, instance, inv_set, contract_label)
                except Exception as exc:
                    logger.warning(
                        "capability_contract[%s]: fail-closed on %s: %s",
                        contract_label,
                        type(exc).__name__,
                        exc,
                    )
                    record_gate_decision(
                        decision="capability_fail_closed",
                        source=contract_label,
                        disposition="hard_refuse",
                        signals=("capability_contract", "fail_closed"),
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    assert on_error is not None  # guaranteed by decoration-time check
                    return cast("R", on_error(exc, args, kwargs))
            # Without FAIL_CLOSED the caller opted out of a sentinel, so a hook or
            # capability error propagates (there is no typed transparent-negative to
            # substitute); pair MONOTONE_HARM / CITE_OR_REFUSE with FAIL_CLOSED to
            # make enforcement itself fail-closed.
            return _enforce_postconditions(func(*args, **kwargs), instance, inv_set, contract_label)

        setattr(wrapper, CONTRACT_MARKER, inv_set)
        _REGISTRY.append((contract_label, inv_set))
        return wrapper

    def _enforce_boundary(
        args: tuple[Any, ...], kwargs: dict[str, Any], contract_label: str
    ) -> None:
        """Run the real decision through the single fail-closed harm-uplift gate.

        Imported lazily so ``contract`` stays importable without pulling the
        ethics stack (which imports this module's sibling audit sink).

        Raises:
            EthicalConstraintViolationError: when the gate blocks, or -- fail
                closed -- when building the subject itself raises. A hook that
                cannot describe the decision is not evidence that the decision
                is safe.
        """
        from omni_mercury_engine.cognitive.decision_gate import enforce_decision_boundary

        assert boundary_subject is not None  # guaranteed by decoration-time check
        try:
            subject = boundary_subject(args, kwargs)
        except Exception as exc:
            from omni_mercury_engine.cognitive.ethical_bounding import (
                EthicalConstraintViolationError,
            )

            logger.exception(
                "capability_contract[%s]: could not build the decision subject; failing closed",
                contract_label,
            )
            record_gate_decision(
                decision="refused",
                source=contract_label,
                disposition="hard_refuse",
                signals=("capability_contract", "gated_boundary"),
                reason=f"subject construction failed: {type(exc).__name__}: {exc}",
            )
            raise EthicalConstraintViolationError(
                action=contract_label,
                score=0.0,
                threshold=1.0,
                check="harm_uplift",
                details={"surface": contract_label, "fail_closed": True},
            ) from exc
        enforce_decision_boundary(subject)

    def _enforce_postconditions(
        result: R, instance: Any, active: frozenset[Invariant], contract_label: str
    ) -> R:
        if Invariant.CITE_OR_REFUSE in active:
            assert emitted is not None and provenance_required is not None
            assert cited is not None and refuse is not None
            if (
                emitted(result, instance)
                and provenance_required(result, instance)
                and not cited(result, instance)
            ):
                logger.warning(
                    "capability_contract[%s]: cite_or_refuse breach; refusing (fail-closed)",
                    contract_label,
                )
                record_gate_decision(
                    decision="capability_cite_or_refuse",
                    source=contract_label,
                    disposition="refuse_redact",
                    signals=("capability_contract", "cite_or_refuse"),
                    reason="emitted provenance-required content without citations; refused",
                )
                result = refuse(result, instance)
        if Invariant.MONOTONE_HARM in active:
            assert harm_residue is not None and redact is not None
            residue = harm_residue(result, instance)
            if residue:
                logger.warning(
                    "capability_contract[%s]: monotone_harm breach (%d span(s)); redacting",
                    contract_label,
                    len(residue),
                )
                record_gate_decision(
                    decision="capability_monotone_harm",
                    source=contract_label,
                    disposition="refuse_redact",
                    signals=("capability_contract", "monotone_harm"),
                    reason=f"{len(residue)} gate-unsafe span(s) in output; redacted",
                )
                result = redact(result, instance)
        return result

    return decorate


__all__ = [
    "CONTRACT_MARKER",
    "ContractViolation",
    "Invariant",
    "SupportsRefusal",
    "capability_contract",
    "is_honest_negative",
    "registered_contracts",
]
