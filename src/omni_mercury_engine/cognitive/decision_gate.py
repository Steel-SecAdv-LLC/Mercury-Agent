# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The single fail-closed choke point every public decision surface passes through.

Mercury used to enforce a *benevolence float* at its decision boundaries: a
keyword/context score that had to clear ``0.99`` before a detection was allowed
to run.  Two things were wrong with that control, and both are fixed here.

1. **It measured the wrong thing.**  The score was computed over a *fixed
   string* the engine synthesised for itself
   (``"anomaly_detection:{domain}:audit verify protect research evidence fair
   oversight monitor data care help support"``) — a keyword salad chosen so the
   gate would pass.  The caller's actual request never reached the scorer, so
   the number could not discriminate anything.  The control was theatre.
2. **A high pass-bar on a benign-data score is not a harm control.**  It
   rejected benign inputs whose vocabulary happened to be plain, and it let any
   input through whose vocabulary happened to be positive.

The enforced control is now the two-axis (hazard-domain × operational-intent)
**harm-uplift gate** documented in ``docs/HARM_POLICY.md`` and implemented by
:func:`~omni_mercury_engine.cognitive.ethical_bounding.assess_weapons_uplift`.
It is scored over the **real decision** — the surface being called, the caller's
domain hint, and the caller's actual request/payload — never over a synthetic
string.  Its polarity is *block on harm*: a benign decision is permitted because
no harm evidence was found, not because it scored highly on a positivity
lexicon.

Fail-closed, in three places
----------------------------

* :func:`assess_weapons_uplift` already fails closed internally: any internal
  error yields ``HARD_REFUSE`` rather than an exception or a silent ALLOW.
* :func:`enforce_decision_boundary` fails closed *around* it: if building the
  subject, running the gate, or auditing the verdict raises anything at all,
  the boundary raises
  :class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`
  with ``check="harm_uplift"``.  There is no path through this function that
  returns normally on an error.
* There is **no flag, environment variable, or keyword argument that disables
  the gate.**  Removing the call is the only way to bypass it, and the
  capability-contract registry (:data:`~omni_mercury_engine.agentic.capabilities.contract.CONTRACT_MARKER`)
  makes that removal a CI failure — see ``tests/pillars/test_non_maleficence.py``.

Benevolence is retained as an **advisory** signal only: when a scorer is
supplied the boundary computes it, attaches it to the verdict, and logs it.  It
never decides.

Determinism
-----------

:func:`enforce_decision_boundary` is a pure function of the subject: the same
decision produces the same :class:`BoundaryVerdict` on every surface.  That is
what lets ``tests/pillars/test_non_maleficence.py`` assert "identical decision →
identical verdict" across ``detect`` / ``detect_batch`` / ``detect_biometric`` /
``detect_security_threat`` / fleet dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
    HazardDomain,
    OperationalIntent,
    WeaponsDisposition,
    WeaponsRiskAssessment,
    assess_weapons_uplift,
    sanitize_domain,
)
from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: Upper bound on the free-text handed to the harm gate.  Bounds hot-path cost
#: and the size of an audited record; the cap is generous enough that a real
#: uplift request cannot hide past it (the gate's lexicons match early tokens).
MAX_SUBJECT_CHARS = 4096

#: Upper bound on the rendered summary of a single structured payload value.
_MAX_VALUE_CHARS = 512

#: Number of mapping keys rendered into a payload summary before eliding.
_MAX_KEYS = 24


def summarize_payload(payload: Any, *, _depth: int = 0) -> str:
    """Render ``payload`` as the factual description of what is being decided on.

    The point is to surface the caller's *real* input to the gate without
    materialising megabytes of telemetry:

    * arrays/tensors become ``ndarray(shape=..., dtype=...)`` — structure only,
      because a float matrix carries no lexical harm evidence;
    * strings are passed through verbatim (truncated), because a string payload
      *is* the request;
    * mappings and sequences are walked one level so a domain payload that
      carries text (``{"query": "...", "notes": "..."}``) reaches the gate
      instead of being reduced to ``dict``.

    Args:
        payload: Any caller-supplied input.
        _depth: Internal recursion guard.

    Returns:
        A bounded, human-readable description.  Never raises: an object whose
        ``repr`` explodes degrades to its type name, since a summariser that
        could raise would be a way to break the control it feeds.
    """
    try:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload[:_MAX_VALUE_CHARS]
        if isinstance(payload, (bool, int, float)):
            return str(payload)
        shape = getattr(payload, "shape", None)
        if shape is not None:
            dtype = getattr(payload, "dtype", None)
            return f"{type(payload).__name__}(shape={tuple(shape)}, dtype={dtype})"
        if _depth >= 2:
            return type(payload).__name__
        if isinstance(payload, dict):
            keys = list(payload)[:_MAX_KEYS]
            rendered = ", ".join(
                f"{k}={summarize_payload(payload[k], _depth=_depth + 1)}" for k in keys
            )
            elided = "" if len(payload) <= _MAX_KEYS else f", +{len(payload) - _MAX_KEYS} more"
            return f"{{{rendered}{elided}}}"
        if isinstance(payload, (list, tuple, set, frozenset)):
            items = list(payload)[:_MAX_KEYS]
            rendered = ", ".join(summarize_payload(v, _depth=_depth + 1) for v in items)
            return f"[{rendered}]"
        return type(payload).__name__
    except Exception:  # pragma: no cover - defensive; a summariser must not raise
        return type(payload).__name__


@dataclass(frozen=True)
class DecisionSubject:
    """The real decision a public surface is about to make.

    Every field is sourced from the actual call — there is deliberately no way
    to construct a subject out of canned positive keywords, which is what the
    superseded ``_enforce_ethics_at_boundary`` action string did.

    Attributes:
        surface: Fully-qualified name of the public surface (e.g.
            ``"OmniMercuryEngine.detect"``).  Audit provenance.
        operation: A short factual statement of what the surface does with the
            input (e.g. ``"analyse request payload for injection/XSS threats"``).
            This is a *description of the code path*, not a claim about the
            caller's intent, and it is fixed per surface so it cannot be tuned
            per call to move a verdict.
        domain: Caller-supplied domain hint, collapsed to the whitelisted
            alphabet by
            :func:`~omni_mercury_engine.cognitive.ethical_bounding.sanitize_domain`
            so a hostile hint cannot inject either harm or allow keywords.
        request: The caller's own free text, verbatim (a query, a payload, a
            file path).  Empty when the surface takes no text.
        payload: The caller's structured input; summarised by
            :func:`summarize_payload`.
        context: Extra structured provenance passed to the gate (e.g.
            ``licensed_context``) and to the audit record.
    """

    surface: str
    operation: str
    domain: str = "general"
    request: str = ""
    payload: Any = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Render the decision as the text the harm gate scores.

        Returns:
            ``"<surface>: <operation> | domain=... | input=... | request=..."``,
            truncated to :data:`MAX_SUBJECT_CHARS`.  The caller's own text is
            placed last so truncation drops trailing caller text rather than the
            surface provenance an auditor needs.
        """
        parts = [f"{self.surface}: {self.operation}", f"domain={sanitize_domain(self.domain)}"]
        summary = summarize_payload(self.payload)
        if summary:
            parts.append(f"input={summary}")
        if self.request:
            parts.append(f"request={self.request}")
        return " | ".join(parts)[:MAX_SUBJECT_CHARS]


@dataclass(frozen=True)
class BoundaryVerdict:
    """The permitted outcome of a boundary check.

    A blocking assessment never produces a verdict — it raises — so every
    instance of this class describes a *permitted* decision.

    Attributes:
        subject: The decision that was gated.
        assessment: The harm-uplift assessment that permitted it.
        benevolence: The advisory benevolence score, when a scorer was
            supplied; ``None`` otherwise.  Advisory means advisory: it is
            logged and attached, and it decides nothing.
        requires_provenance: ``True`` when the gate permitted the decision but
            requires any emitted answer to be source-attributed
            (``ALLOW_PROVENANCE``).  Non-blocking, enforced at the output
            boundary.
    """

    subject: DecisionSubject
    assessment: WeaponsRiskAssessment
    benevolence: float | None = None

    @property
    def requires_provenance(self) -> bool:
        """True when the permit is conditional on source attribution."""
        return self.assessment.disposition is WeaponsDisposition.ALLOW_PROVENANCE


def enforce_decision_boundary(
    subject: DecisionSubject,
    *,
    advisory_scorer: Any | None = None,
    harm_classifier: Any | None = None,
) -> BoundaryVerdict:
    """Run the enforced harm-uplift gate over a real decision; raise if it blocks.

    This is the choke point.  Every public decision surface calls it, with the
    same semantics, over the same assessment function, so a red-team decision
    that is blocked on one surface is blocked on all of them.

    Args:
        subject: The decision being made.  Built from the actual call.
        advisory_scorer: Optional
            :class:`~omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`
            (or the cached wrapper).  When supplied its benevolence float is
            computed, logged and attached to the verdict — **advisory only**.
            A scorer that raises is logged and ignored: an advisory signal must
            not be able to fail the decision, and must not be able to pass it.
        harm_classifier: Optional meaning-level classifier forwarded to
            :func:`assess_weapons_uplift`.  Consulted, never trusted: it can
            only raise a disposition.

    Returns:
        A :class:`BoundaryVerdict` describing the permitted decision.

    Raises:
        EthicalConstraintViolationError: with ``check="harm_uplift"`` when the
            gate's disposition blocks (ESCALATE / REFUSE_REDACT / HARD_REFUSE),
            and — fail-closed — when anything in this function raises
            unexpectedly.
    """
    try:
        action = subject.describe()
        assessment = assess_weapons_uplift(
            action, dict(subject.context), harm_classifier=harm_classifier
        )
    except Exception as exc:
        # Fail closed. Never return normally on an error path: an exception in
        # the gate must read as "refused", not as "allowed".
        logger.exception("decision boundary %s: gate evaluation failed; failing closed", subject)
        raise EthicalConstraintViolationError(
            action=f"{subject.surface}:{subject.operation}",
            score=0.0,
            threshold=1.0,
            check="harm_uplift",
            details={
                "surface": subject.surface,
                "domain": sanitize_domain(subject.domain),
                "error": f"{type(exc).__name__}: {exc}",
                "fail_closed": True,
            },
        ) from exc

    if assessment.blocks:
        record_gate_decision(
            decision="refused",
            source="decision_boundary",
            disposition=assessment.disposition.value,
            hazard_domain=assessment.hazard_domain.value,
            intent=assessment.intent_tier.value,
            signals=assessment.signals,
            reason="harm-uplift gate blocked the decision at a public boundary",
            query=action,
            extra={"surface": subject.surface, "domain": sanitize_domain(subject.domain)},
        )
        raise EthicalConstraintViolationError(
            action=action,
            score=float(assessment.confidence),
            threshold=1.0,
            check="harm_uplift",
            details={
                "surface": subject.surface,
                "domain": sanitize_domain(subject.domain),
                "hazard_domain": assessment.hazard_domain.value,
                "operational_intent": assessment.intent_tier.value,
                "disposition": assessment.disposition.value,
                "signals": list(assessment.signals),
            },
        )

    benevolence: float | None = None
    if advisory_scorer is not None:
        try:
            benevolence = float(
                advisory_scorer.score_action(action, dict(subject.context)).benevolence_score
            )
        except Exception as exc:  # advisory: never blocks, never passes
            logger.warning(
                "decision boundary %s: advisory benevolence unavailable (%s)",
                subject.surface,
                exc,
            )
        else:
            logger.debug(
                "decision boundary %s: advisory benevolence=%.4f (informational; "
                "the enforced control is the harm-uplift gate)",
                subject.surface,
                benevolence,
            )

    return BoundaryVerdict(subject=subject, assessment=assessment, benevolence=benevolence)


__all__ = [
    "MAX_SUBJECT_CHARS",
    "BoundaryVerdict",
    "DecisionSubject",
    "HazardDomain",
    "OperationalIntent",
    "enforce_decision_boundary",
    "summarize_payload",
]
