# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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

import dataclasses
import logging
from collections import deque
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.cognitive.ethical_bounding import (
    USE_SHIPPED_CLASSIFIER,
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

#: Size of one passage handed to the harm gate, and of an audited record.
#:
#: It is NOT a bound on what the gate reads. It used to be: the gate scored one
#: string cut at this length, on the stated ground that "a real uplift request
#: cannot hide past it". It could. Each of the pillar suite's own red-team
#: requests was permitted behind 4,640 characters of weather small-talk. The
#: gate now reads all of the caller's text, as overlapping passages of this
#: size (:meth:`DecisionSubject.passages`).
MAX_SUBJECT_CHARS = 4096

#: How much caller text one decision may carry. Past it the gate does not
#: guess: the decision is refused, fail-closed, like any other it cannot
#: evaluate. Reading is about 1.7 ms per thousand characters, so this bounds
#: one decision at roughly two seconds.
MAX_ASSESSED_CHARS = 1_048_576

#: How many values of a structured payload the gate will walk looking for
#: text before it refuses to decide. A numeric array is not walked at all
#: (its shape is its summary), so this is reached only by very large plain
#: Python containers.
MAX_ASSESSED_NODES = 5_000_000

#: Nesting the walk follows before it refuses to decide.
_MAX_TEXT_DEPTH = 64

#: Upper bound on the rendered summary of a single structured payload value.
_MAX_VALUE_CHARS = 512

#: Number of mapping keys rendered into a payload summary before eliding.
_MAX_KEYS = 24

#: Upper bound on the whole payload summary. The gate scores ``describe()``,
#: whose caller-controlled ``request`` must always survive the
#: :data:`MAX_SUBJECT_CHARS` cut. Without a total cap a structured payload can
#: render to tens of kilobytes and, placed before ``request``, evict it from the
#: scored string entirely -- a harm-uplift bypass (large benign payload +
#: trailing uplift ``request`` -> only the benign padding is scored). The cap
#: leaves at least ``MAX_SUBJECT_CHARS - _MAX_PAYLOAD_SUMMARY_CHARS`` for the
#: request, and ``describe()`` additionally scores the request FIRST.
_MAX_PAYLOAD_SUMMARY_CHARS = 2048

#: Recursion ceiling for :func:`summarize_payload`. Deep enough that a string
#: nested a few containers down (``{"a": {"b": {"c": "<uplift text>"}}}``) still
#: reaches the gate's lexicons instead of collapsing to the token ``dict`` -- the
#: old depth-2 ceiling hid exactly that. The total-character cap above, not the
#: depth, is what bounds cost, so this can be generous.
_MAX_PAYLOAD_DEPTH = 6


class UnassessableSubjectError(ValueError):
    """The decision carries more than the gate can read; it must be refused."""


def _object_text(payload: Any) -> list[Any] | None:
    """What an object that is not a plain container holds, or ``None`` if nothing is read.

    A float matrix has no words in it and is summarised by its shape. An array
    of strings or objects, the text columns of a data frame, and the fields of
    a dataclass or a pydantic model are the caller's words in another
    container. Summarised by shape or by type name they went unread: the
    reasoning backend hands the gate a ``ReasoningContext``, which it saw as
    the single word ``ReasoningContext`` while the evidence inside went on to
    a language model.

    An object of none of these kinds, or one of them that cannot be read the
    way its kind is read, is opaque and stays its type name, as it always
    was: this never raises, so that an odd object is not a way to have a
    decision refused.
    """
    try:
        if dataclasses.is_dataclass(payload) and not isinstance(payload, type):
            return [getattr(payload, f.name, None) for f in dataclasses.fields(payload)]
        # Looked up on the type: an instance that answers to any name at all (a
        # mock, a proxy) is not thereby a pydantic model.
        if callable(getattr(type(payload), "model_dump", None)):
            return [payload.model_dump()]
        if hasattr(type(payload), "dtypes") and hasattr(type(payload), "columns"):  # a frame
            names = list(payload.columns)
            values: list[Any] = [str(name) for name in names]
            for name in names:
                column = payload[name]
                if getattr(getattr(column, "dtype", None), "kind", "O") in "OUST":
                    values.extend(_as_list(column))
            return values
        kind = getattr(getattr(payload, "dtype", None), "kind", None)
        if not isinstance(kind, str):
            return None
        return [_as_list(payload)] if kind in "OUST" else []
    except Exception:
        return None


def _as_list(values: Any) -> list[Any]:
    """An array or a column as a Python list (numpy and pandas ``tolist``, polars ``to_list``)."""
    convert = getattr(values, "tolist", None) or values.to_list
    listed: list[Any] = convert()
    return listed


def payload_text(payload: Any) -> list[str]:
    """Every piece of text in ``payload``, however deep and however far along.

    :func:`summarize_payload` is a bounded description for the audit record:
    the first 24 keys, the first 512 characters of a string, six levels. The
    gate cannot decide on a description that leaves text out, so it reads this
    instead: every string value, every string key, bytes as text, and the
    string cells of arrays and data frames. Numbers carry no words and are
    skipped.

    Raises:
        UnassessableSubjectError: When the payload holds more text than
            :data:`MAX_ASSESSED_CHARS`, more values than
            :data:`MAX_ASSESSED_NODES`, or nests deeper than the walk follows.
    """
    found: list[str] = []
    chars = 0
    nodes = 0
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(payload, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_ASSESSED_NODES:
            raise UnassessableSubjectError(
                f"the payload holds more than {MAX_ASSESSED_NODES:,} values; "
                "pass numeric data as an array"
            )
        if value is None or isinstance(value, (bool, int, float, complex)):
            continue
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8", errors="replace")
        if isinstance(value, str):
            if value:
                chars += len(value)
                if chars > MAX_ASSESSED_CHARS:
                    raise UnassessableSubjectError(
                        f"the payload holds more than {MAX_ASSESSED_CHARS:,} characters of text"
                    )
                found.append(value)
            continue
        if depth >= _MAX_TEXT_DEPTH:
            raise UnassessableSubjectError(
                f"the payload nests deeper than {_MAX_TEXT_DEPTH} levels"
            )
        children: list[Any] | None
        if isinstance(value, _Mapping):
            children = [part for item in value.items() for part in item]
        elif isinstance(value, (list, tuple, set, frozenset, deque)):
            children = list(value)
        else:
            children = _object_text(value)
        if not children:
            continue
        if id(value) in seen:  # a container reached twice is read once
            continue
        seen.add(id(value))
        stack.extend((child, depth + 1) for child in reversed(children))
    return found


def _windows(text: str, size: int) -> list[str]:
    """``text`` as passages of ``size`` that overlap by half.

    Any stretch of text up to half a passage long lies whole inside at least
    one of them, so a request cannot be cut in two at a boundary.
    """
    stride = max(size // 2, 1)
    last_start = max(len(text) - (size - stride), 1)
    return [text[start : start + size] for start in range(0, last_start, stride)]


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
        if _depth >= _MAX_PAYLOAD_DEPTH:
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
            ``"<surface>: <operation> | domain=... | request=... | input=..."``,
            truncated to :data:`MAX_SUBJECT_CHARS`.

        The ``request`` -- the caller's own free text, and the highest-signal
        harm evidence -- is placed **before** the structured payload summary, and
        the summary is capped at :data:`_MAX_PAYLOAD_SUMMARY_CHARS`. Both matter:
        the caller controls both fields, and a payload that renders to tens of
        kilobytes would otherwise fill the whole :data:`MAX_SUBJECT_CHARS` budget
        and push a trailing uplift ``request`` off the end -- so the gate would
        score only the benign padding and permit the decision. Ordering the
        request first, and bounding the summary, guarantees the request is always
        scored.
        """
        parts = [f"{self.surface}: {self.operation}", f"domain={sanitize_domain(self.domain)}"]
        if self.request:
            parts.append(f"request={self.request}")
        summary = summarize_payload(self.payload)
        if summary:
            parts.append(f"input={summary[:_MAX_PAYLOAD_SUMMARY_CHARS]}")
        return " | ".join(parts)[:MAX_SUBJECT_CHARS]

    def passages(self) -> list[str]:
        """Everything the caller said, as the passages the harm gate scores.

        The first is :meth:`describe`. When that holds all of the caller's
        text it is the only one, and the gate decides exactly as it did. When
        it does not -- the request runs past :data:`MAX_SUBJECT_CHARS`, or the
        payload holds text its bounded summary leaves out -- the whole request
        and the whole text of the payload follow, each as overlapping passages
        under the same ``surface: operation | domain=`` heading.

        Raises:
            UnassessableSubjectError: When the caller's text exceeds what the
                gate will read (:data:`MAX_ASSESSED_CHARS`).
        """
        first = self.describe()
        passages = [first]
        texts = payload_text(self.payload)
        if len(self.request) + sum(map(len, texts)) > MAX_ASSESSED_CHARS:
            raise UnassessableSubjectError(
                f"the decision carries more than {MAX_ASSESSED_CHARS:,} characters of text"
            )
        heading = f"{self.surface}: {self.operation} | domain={sanitize_domain(self.domain)}"
        room = max(MAX_SUBJECT_CHARS - len(heading) - len(" | request="), 1024)
        if self.request and f"request={self.request}" not in first:
            passages.extend(f"{heading} | request={part}" for part in _windows(self.request, room))
        if texts and not (len(texts) <= _MAX_KEYS and all(text in first for text in texts)):
            passages.extend(
                f"{heading} | input={part}" for part in _windows(", ".join(texts), room)
            )
        return passages


#: The response ladder, least restrictive first (``WeaponsDisposition``'s own order).
_RESTRICTION: dict[WeaponsDisposition, int] = {
    disposition: rank for rank, disposition in enumerate(WeaponsDisposition)
}


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
    harm_classifier: Any | None = USE_SHIPPED_CLASSIFIER,
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
        harm_classifier: Meaning-level classifier forwarded to
            :func:`assess_weapons_uplift`.  Consulted, never trusted: it can
            only raise a disposition.  Defaults to Mercury's shipped offline
            classifier; pass ``None`` for a strictly lexical gate.  This
            forwarder must not default to ``None``, or the choke point every
            decision surface routes through would be the one place that opts
            out of meaning-level coverage.

    Returns:
        A :class:`BoundaryVerdict` describing the permitted decision.

    Raises:
        EthicalConstraintViolationError: with ``check="harm_uplift"`` when the
            gate's disposition blocks (ESCALATE / REFUSE_REDACT / HARD_REFUSE),
            and — fail-closed — when anything in this function raises
            unexpectedly.
    """
    try:
        # Every passage is scored. The first that blocks decides, and is what
        # the refusal and its audit record quote; otherwise the most
        # restrictive permit stands (a later passage that needs its answer
        # sourced makes the whole decision need it).
        action = subject.describe()
        assessment: WeaponsRiskAssessment | None = None
        for passage in subject.passages():
            scored = assess_weapons_uplift(
                passage, dict(subject.context), harm_classifier=harm_classifier
            )
            if scored.blocks:
                action, assessment = passage, scored
                break
            if (
                assessment is None
                or _RESTRICTION[scored.disposition] > _RESTRICTION[assessment.disposition]
            ):
                assessment = scored
        if assessment is None:  # pragma: no cover - passages() is never empty
            raise RuntimeError("no passage was scored")
    except Exception as exc:
        # Fail closed. Never return normally on an error path: an exception in
        # the gate must read as "refused", not as "allowed".
        #
        # Log provenance ONLY. The full subject carries ``request``, ``payload``
        # and ``context`` -- the caller's own data -- so logging it would spill
        # user input (credentials in a scanned payload, patient text, an
        # intercepted attacker body) into an unstructured error log with a
        # different retention and access policy than the audit trail, and would
        # emit up to MAX_SUBJECT_CHARS per failure. The full detail still
        # reaches the structured refusal below and the gate audit record.
        logger.exception(
            "decision boundary %s (domain=%s): gate evaluation failed; failing closed",
            subject.surface,
            sanitize_domain(subject.domain),
        )
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
    "MAX_ASSESSED_CHARS",
    "MAX_ASSESSED_NODES",
    "MAX_SUBJECT_CHARS",
    "BoundaryVerdict",
    "DecisionSubject",
    "HazardDomain",
    "OperationalIntent",
    "UnassessableSubjectError",
    "enforce_decision_boundary",
    "payload_text",
    "summarize_payload",
]
