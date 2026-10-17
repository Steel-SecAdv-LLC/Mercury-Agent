# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The gates every assessment surface passes before a library module runs.

Three layers, applied in this order by :func:`gate`:

1. **Operator policy** (human-subject capabilities only). The Concealed
   Information Test and contactless vital signs measure a person's body. They
   are off until the operator lists the lawful bases they permit in
   :data:`HUMAN_SUBJECT_POLICY_ENV`. Deploying the platform therefore never
   exposes physiological inference by default.
2. **The harm-uplift decision gate**,
   :func:`~omni_mercury_engine.cognitive.decision_gate.enforce_decision_boundary`,
   the choke point every other decision surface uses. It fails closed.
3. **The purpose screen** (human-subject capabilities only). The caller must
   state a purpose. It is refused when it describes covert or non-consensual
   sensing (:func:`covert_sensing_signal`), and when the benevolence scorer
   finds it impermissible (grave harm to a person: stalking, harassment,
   intimidation). Layer 2 alone does not cover this: it is scoped to weapons
   uplift and, measured, allows "covertly stalk my ex by tracking her heartbeat
   through the wall"; the scorer allows "secretly monitor my neighbour's heart
   rate through the wall without them knowing". Each of those is refused here.

Every refusal raises
:class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`
(``check`` names the layer) or :class:`CapabilityDisabledError`, and is
recorded with
:func:`~omni_mercury_engine.cognitive.gate_audit.record_gate_decision`. Every
admitted human-subject call is recorded too, with a fingerprint of its consent
or attestation, so the audit trail shows who authorised each measurement.

Consent (layer 4, in :mod:`.service`) is checked by the library itself: a
:class:`~omni_mercury_engine.truth_decipher.ConsentRecord` for the subject,
obtained before the recording, with the capability's scope, not withdrawn.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

#: Comma-separated lawful bases the operator permits for human-subject
#: capabilities: ``consent`` and/or ``vital_interests``. Unset or empty means
#: both capabilities are disabled.
HUMAN_SUBJECT_POLICY_ENV = "MERCURY_HUMAN_SUBJECT_CAPABILITIES"

#: Shortest purpose statement accepted for a human-subject call. Long enough
#: that the screen has a sentence to read, short enough not to burden a
#: genuine request ("sleep-lab respiration study, protocol 12").
MIN_PURPOSE_CHARS = 20
#: Longest purpose kept; the gate scores at most this much.
MAX_PURPOSE_CHARS = 2000


class LawfulBasis(StrEnum):
    """Why a human-subject measurement may be made.

    ``CONSENT``: the subject's informed consent, recorded as a
    :class:`~omni_mercury_engine.truth_decipher.ConsentRecord` (GDPR Art.
    9(2)(a)).

    ``VITAL_INTERESTS``: protecting the life of a person who cannot consent,
    such as a survivor trapped under rubble (GDPR Art. 9(2)(c)). It applies to
    contactless vital signs only, needs an operator attestation naming the
    incident, and must be enabled separately from consent.
    """

    CONSENT = "consent"
    VITAL_INTERESTS = "vital_interests"


class CapabilityDisabledError(PermissionError):
    """The operator has not enabled this capability or lawful basis."""


@dataclass(frozen=True)
class GateVerdict:
    """What an admitted call passed; attached to audit records and results."""

    surface: str
    human_subject: bool
    basis: str | None
    authorisation_fingerprint: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return the verdict as a JSON-compatible mapping."""
        return {
            "surface": self.surface,
            "human_subject": self.human_subject,
            "lawful_basis": self.basis,
            "authorisation_fingerprint": self.authorisation_fingerprint,
        }


def permitted_bases(environ: dict[str, str] | None = None) -> frozenset[LawfulBasis]:
    """Return the lawful bases the operator enabled (empty: capability off).

    Raises:
        CapabilityDisabledError: The variable names an unknown basis. A typo
            must not silently enable or disable a basis.
    """
    raw = (os.environ if environ is None else environ).get(HUMAN_SUBJECT_POLICY_ENV, "")
    bases: set[LawfulBasis] = set()
    for token in (t.strip().lower() for t in raw.split(",")):
        if not token:
            continue
        try:
            bases.add(LawfulBasis(token))
        except ValueError as exc:
            raise CapabilityDisabledError(
                f"{HUMAN_SUBJECT_POLICY_ENV} names unknown lawful basis {token!r}; "
                f"expected a comma-separated subset of {[b.value for b in LawfulBasis]}"
            ) from exc
    return frozenset(bases)


def require_basis_enabled(
    basis: LawfulBasis, capability: str, environ: dict[str, str] | None = None
) -> None:
    """Refuse unless the operator enabled ``basis`` for human-subject capabilities."""
    if basis not in permitted_bases(environ):
        raise CapabilityDisabledError(
            f"{capability} is disabled on this deployment for lawful basis "
            f"{basis.value!r}. The operator enables it by adding {basis.value!r} "
            f"to {HUMAN_SUBJECT_POLICY_ENV}."
        )


#: Covert or non-consensual sensing, stated in the purpose. Each alternative
#: names concealment from, or the absence of agreement by, the person sensed.
#: Topic words alone ("covert channel detection", "consent form") do not
#: match: concealment words need a sensing or person context within the
#: clause, and the "without ..." forms name knowledge, consent or permission.
_SENSING = (
    r"(?:monitor\w*|track\w*|watch\w*|measur\w*|record\w*|sens\w*|scan\w*|observ\w*"
    r"|listen\w*|surveil\w*|spy\w*|captur\w*|heart\w*|breath\w*|pulse|vital\w*)"
)
_COVERT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "concealment",
        re.compile(
            r"\b(?:secret(?:ly)?|covert(?:ly)?|surreptitious(?:ly)?|clandestine(?:ly)?"
            r"|undetected|stealth(?:ily)?|discreetly|hidden)\b[^.!?;]{0,60}?\b" + _SENSING,
            re.IGNORECASE,
        ),
    ),
    (
        "concealment",
        re.compile(
            r"\b" + _SENSING + r"\b[^.!?;]{0,60}?\b(?:secretly|covertly|surreptitiously"
            r"|clandestinely|undetected|without being (?:noticed|detected|seen))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "no_knowledge_or_consent",
        re.compile(
            r"\bwithout\s+(?:\w+\s+){0,3}?(?:knowing|knowledge|consent(?:ing)?|permission"
            r"|noticing|awareness|being aware|realising|realizing|agreement)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "no_knowledge_or_consent",
        re.compile(
            r"\b(?:unbeknownst|behind (?:their|her|his|my \w+'s) back"
            r"|(?:does|do|did|will|would) ?n[o']t (?:know|consent|agree)"
            r"|ha(?:s|ve) ?n[o']t (?:consented|agreed))\b",
            re.IGNORECASE,
        ),
    ),
    ("spying", re.compile(r"\bspy(?:ing)?\s+on\b", re.IGNORECASE)),
)


def covert_sensing_signal(purpose: str) -> str | None:
    """Return the label of the first covert-sensing pattern in ``purpose``, or ``None``."""
    for label, pattern in _COVERT_PATTERNS:
        if pattern.search(purpose):
            return label
    return None


def fingerprint(value: Any) -> str:
    """SHA-256 over the canonical JSON of ``value`` (truncated to 32 hex chars).

    Audit records carry this instead of the consent or attestation itself, so
    the trail can be matched to a record held elsewhere without copying the
    subject's pseudonym into every log line.
    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _refuse(
    *, surface: str, check: str, reason: str, query: str, extra: dict[str, Any] | None = None
) -> Exception:
    from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
    from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

    record_gate_decision(
        decision="refused",
        source="assessment_surface",
        disposition=check,
        reason=reason,
        query=query,
        extra={"surface": surface, **(extra or {})},
    )
    return EthicalConstraintViolationError(
        action=surface,
        score=0.0,
        threshold=1.0,
        check=check,
        details={"surface": surface, "reason": reason, **(extra or {})},
    )


def _normalise_purpose(purpose: object) -> str:
    if not isinstance(purpose, str):
        raise ValueError("purpose must be a string stating why the measurement is made")
    text = " ".join(purpose.split())
    if len(text) < MIN_PURPOSE_CHARS:
        raise ValueError(
            f"purpose must state why the measurement is made (at least "
            f"{MIN_PURPOSE_CHARS} characters)"
        )
    return text[:MAX_PURPOSE_CHARS]


def gate(
    *,
    surface: str,
    operation: str,
    domain: str = "general",
    request: str = "",
    payload: Any = None,
    human_subject: bool = False,
    purpose: object = None,
    basis: LawfulBasis | None = None,
    authorisation: Any = None,
    environ: dict[str, str] | None = None,
) -> GateVerdict:
    """Run the gates for one call; raise on refusal, return the verdict otherwise.

    Args:
        surface: Stable surface id, e.g. ``"api.assessment.vital_signs"``.
        operation: Fixed description of what the surface does.
        domain: Decision domain passed to the harm-uplift gate.
        request: Caller-supplied free text (scored first by the gate).
        payload: Caller-supplied structured input (summarised by the gate).
        human_subject: The capability measures a person (CIT, vital signs).
        purpose: The caller's stated purpose (required when ``human_subject``).
        basis: The lawful basis claimed (required when ``human_subject``).
        authorisation: The consent record or attestation, fingerprinted into
            the audit record of an admitted call.
        environ: Environment override for tests.

    Raises:
        CapabilityDisabledError: Operator policy does not permit the call.
        EthicalConstraintViolationError: ``check`` is ``"harm_uplift"`` (the
            decision gate), ``"covert_sensing"`` or ``"benevolence"`` (the
            purpose screen).
        ValueError: ``purpose`` or ``basis`` missing on a human-subject call.
    """
    from omni_mercury_engine.cognitive.decision_gate import (
        DecisionSubject,
        enforce_decision_boundary,
    )

    text = request
    if human_subject:
        if basis is None:
            raise ValueError("a human-subject call must state its lawful basis")
        require_basis_enabled(basis, operation, environ)
        stated = _normalise_purpose(purpose)
        text = f"{stated} {request}".strip()

    enforce_decision_boundary(
        DecisionSubject(
            surface=surface, operation=operation, domain=domain, request=text, payload=payload
        )
    )

    if not human_subject:
        return GateVerdict(surface, False, None, None)

    signal = covert_sensing_signal(stated)
    if signal is not None:
        raise _refuse(
            surface=surface,
            check="covert_sensing",
            reason=(
                f"the stated purpose describes covert or non-consensual sensing ({signal}); "
                "this capability measures only people who know and agree, or whose life "
                "it protects"
            ),
            query=stated,
            extra={"signal": signal},
        )

    from omni_mercury_engine.cognitive.ethical_bounding import (
        MINIMUM_BENEVOLENCE_FLOOR,
        BenevolenceScorer,
    )

    score = BenevolenceScorer(benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR).score_action(
        f"{operation}: {stated}", {}
    )
    if not score.is_permissible:
        raise _refuse(
            surface=surface,
            check="benevolence",
            reason="the stated purpose was assessed as impermissible harm to a person",
            query=stated,
            extra={"grave_harm": bool(getattr(score, "grave_harm", False))},
        )

    from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

    auth_fp = fingerprint(authorisation) if authorisation is not None else None
    assert basis is not None  # checked above
    record_gate_decision(
        decision="approved",
        source="assessment_surface",
        disposition="allow_log",
        reason=f"human-subject measurement under lawful basis {basis.value!r}",
        query=stated,
        extra={"surface": surface, "lawful_basis": basis.value, "authorisation": auth_fp},
    )
    return GateVerdict(surface, True, basis.value, auth_fp)
