# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One implementation behind the API, CLI and MCP assessment surfaces.

Every public function takes JSON-compatible input and a ``surface`` id
(``"api.assessment.verify_statements"`` and so on), runs the gates of
:mod:`.policy`, calls the library module, and returns JSON-compatible output.
Input is parsed strictly: an unknown key is an error, never silently ignored,
because a misspelt ``withdrawn_at`` or ``reliability`` would otherwise change
a result without anyone noticing.

Errors, and how the surfaces report them:

* :class:`InputError` (a ``ValueError``): malformed input. API 400.
* :class:`~omni_mercury_engine.assessment.policy.CapabilityDisabledError`,
  :class:`~omni_mercury_engine.truth_decipher.ConsentRequiredError` (both
  ``PermissionError``) and
  :class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`:
  refused. API 403.
* :class:`~omni_mercury_engine.security.adjudication.ExportRefused`: the
  ledger cannot support the request. API 409.
* ``ImportError``: an optional dependency (torch, AMA) is missing; the gate
  itself needs torch, so nothing runs ungated. API 503.

Data minimisation: the vital-signs surfaces return rates, quality figures and
abstention reasons. The respiration waveform, a physiological signal of its
own, is returned only when the caller asks for it.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from omni_mercury_engine.assessment.policy import (
    CapabilityDisabledError,
    LawfulBasis,
    gate,
    require_basis_enabled,
)

if TYPE_CHECKING:
    from omni_mercury_engine.biometric.rf_vital_signs import VitalSignEstimate, VitalSignTrack
    from omni_mercury_engine.security.adjudication import AdjudicationLedger
    from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine
    from omni_mercury_engine.truth_decipher import (
        Claim,
        ClaimKey,
        ConsentRecord,
        EvidenceItem,
        Statement,
        TimeInterval,
    )

#: Input limits. A request past one is refused, never truncated.
MAX_STATEMENTS = 1_000
MAX_EVIDENCE = 20_000
MAX_CIT_QUESTIONS = 200
MAX_REPORTS = 5_000
#: Total numeric values in one vital-signs request. 60 s of 100 Hz CSI over
#: 56 subcarriers and 3 antennas is about 1.0 million; longer recordings are
#: sent as several windows.
MAX_SIGNAL_VALUES = 2_000_000
#: Pseudonyms are opaque ids: letters, digits and ``. _ : -``. A real name
#: ("Jane Doe"), an e-mail address or a phone number with separators cannot
#: pass, which is the rule ``truth_decipher.ConsentRecord`` documents.
_PSEUDONYM = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

_FUSION_CHECKPOINT_ENV = "MERCURY_INTEL_FUSION_CHECKPOINT"
#: Learn per-discipline reliabilities from the ledger's training split.
_LEARN_RELIABILITY_ENV = "MERCURY_INTEL_LEARN_RELIABILITY"


class InputError(ValueError):
    """The request is malformed; nothing was run."""


# --------------------------------------------------------------------------- #
# Generic parsing.
# --------------------------------------------------------------------------- #
def _obj(
    value: Any, name: str, required: Collection[str], optional: Collection[str] = ()
) -> dict[str, Any]:
    """Return ``value`` as a dict with exactly ``required`` plus any of ``optional``."""
    if not isinstance(value, Mapping):
        raise InputError(f"{name} must be an object")
    keys = set(value)
    missing = set(required) - keys
    unknown = keys - set(required) - set(optional)
    if missing:
        raise InputError(f"{name} is missing {sorted(missing)}")
    if unknown:
        raise InputError(f"{name} has unknown keys {sorted(unknown)}")
    return dict(value)


def _list(value: Any, name: str, limit: int) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InputError(f"{name} must be a list")
    if len(value) > limit:
        raise InputError(f"{name} has {len(value)} entries; at most {limit} are accepted")
    return list(value)


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"{name} must be finite")
    return number


def _opt_num(value: Any, name: str) -> float | None:
    return None if value is None else _num(value, name)


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a non-empty string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise InputError(f"{name} must be true or false")
    return value


def _pseudonym(value: Any, name: str = "subject_pseudonym") -> str:
    text = _str(value, name)
    if not _PSEUDONYM.match(text):
        raise InputError(
            f"{name} must be an opaque pseudonymous id (letters, digits, '.', '_', ':', '-'; "
            "at most 64 characters), never a name or contact detail"
        )
    return text


_T = TypeVar("_T")


def _wrap(  # noqa: UP047 - PEP 695 type params require Python 3.12+
    call: Callable[..., _T], *args: Any, **kwargs: Any
) -> _T:
    """Run a library constructor, turning its validation errors into :class:`InputError`."""
    try:
        return call(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(str(exc)) from exc


def jsonable(value: Any) -> Any:
    """Convert results (dataclasses, numpy arrays, enums, NaN) to JSON builtins.

    Non-finite floats become ``None``: JSON has no NaN, and every NaN a
    library returns means "not computed" or "abstained".
    """
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(jsonable(k)): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((jsonable(v) for v in value), key=str)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# --------------------------------------------------------------------------- #
# Statement verification.
# --------------------------------------------------------------------------- #
def _interval(value: Any, name: str) -> TimeInterval | None:
    from omni_mercury_engine.truth_decipher import TimeInterval

    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _wrap(TimeInterval.at, _num(value, name))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return _wrap(TimeInterval, _num(value[0], f"{name}[0]"), _num(value[1], f"{name}[1]"))
    item = _obj(value, name, {"start", "end"})
    return _wrap(
        TimeInterval, _num(item["start"], f"{name}.start"), _num(item["end"], f"{name}.end")
    )


def _claim_key(value: Any, name: str) -> ClaimKey:
    from omni_mercury_engine.truth_decipher import ClaimKey

    item = _obj(value, name, {"subject", "predicate"})
    return _wrap(ClaimKey, item["subject"], item["predicate"])


def _claim(value: Any, name: str) -> Claim:
    from omni_mercury_engine.truth_decipher import Claim, Location

    item = _obj(value, name, {"subject", "predicate", "value"}, {"interval", "location"})
    location = None
    if item.get("location") is not None:
        loc = _obj(item["location"], f"{name}.location", {"lat", "lon"}, {"radius_km"})
        location = _wrap(
            Location,
            _num(loc["lat"], f"{name}.location.lat"),
            _num(loc["lon"], f"{name}.location.lon"),
            _num(loc.get("radius_km", 0.0), f"{name}.location.radius_km"),
        )
    return _wrap(
        Claim,
        item["subject"],
        item["predicate"],
        item["value"],
        interval=_interval(item.get("interval"), f"{name}.interval"),
        location=location,
    )


def _statement(value: Any, index: int) -> Statement:
    from omni_mercury_engine.truth_decipher import Detail, Statement, TemporalOrdering

    name = f"statements[{index}]"
    item = _obj(
        value,
        name,
        {"statement_id", "speaker", "made_at"},
        {"claims", "details", "disclosed_evidence_ids", "orderings"},
    )
    claims = tuple(
        _claim(c, f"{name}.claims[{i}]")
        for i, c in enumerate(_list(item.get("claims", []), f"{name}.claims", MAX_EVIDENCE))
    )
    details = []
    for i, d in enumerate(_list(item.get("details", []), f"{name}.details", MAX_EVIDENCE)):
        detail = _obj(d, f"{name}.details[{i}]", {"text", "verifiable"})
        details.append(
            _wrap(
                Detail,
                _str(detail["text"], f"{name}.details[{i}].text"),
                _bool(detail["verifiable"], f"{name}.details[{i}].verifiable"),
            )
        )
    orderings = []
    for i, o in enumerate(_list(item.get("orderings", []), f"{name}.orderings", MAX_EVIDENCE)):
        entry = _obj(o, f"{name}.orderings[{i}]", {"first", "second", "relations"})
        relations = entry["relations"]
        relations = [relations] if isinstance(relations, str) else _list(relations, "relations", 13)
        orderings.append(
            _wrap(
                TemporalOrdering,
                _claim_key(entry["first"], f"{name}.orderings[{i}].first"),
                _claim_key(entry["second"], f"{name}.orderings[{i}].second"),
                frozenset(relations),
            )
        )
    disclosed = _list(
        item.get("disclosed_evidence_ids", []), f"{name}.disclosed_evidence_ids", MAX_EVIDENCE
    )
    return _wrap(
        Statement,
        statement_id=item["statement_id"],
        speaker=item["speaker"],
        made_at=_num(item["made_at"], f"{name}.made_at"),
        claims=claims,
        details=tuple(details),
        disclosed_evidence_ids=frozenset(_str(e, "evidence id") for e in disclosed),
        orderings=tuple(orderings),
    )


def _evidence(value: Any, index: int) -> EvidenceItem:
    from omni_mercury_engine.truth_decipher import EvidenceItem

    name = f"evidence[{index}]"
    item = _obj(
        value,
        name,
        {"evidence_id"},
        {
            "target",
            "stance",
            "asserted",
            "target_statement_id",
            "reliability",
            "source_id",
            "independence_cluster",
            "time",
        },
    )
    return _wrap(
        EvidenceItem,
        evidence_id=item["evidence_id"],
        target=(
            _claim_key(item["target"], f"{name}.target") if item.get("target") is not None else None
        ),
        stance=item.get("stance"),
        asserted=(
            _claim(item["asserted"], f"{name}.asserted")
            if item.get("asserted") is not None
            else None
        ),
        target_statement_id=item.get("target_statement_id"),
        reliability=_opt_num(item.get("reliability"), f"{name}.reliability"),
        source_id=item.get("source_id"),
        independence_cluster=item.get("independence_cluster"),
        time=_interval(item.get("time"), f"{name}.time"),
    )


def _verifier_config(value: Any) -> Any:
    from omni_mercury_engine.truth_decipher import ValueTolerance, VerifierConfig

    if value is None:
        return VerifierConfig()
    item = _obj(
        value,
        "config",
        set(),
        {
            "prior_probability",
            "corroborated_at",
            "contradicted_at",
            "reliability_epsilon",
            "low_reliability_policy",
            "default_tolerance",
            "predicate_tolerances",
            "confidence_level",
        },
    )

    def tolerance(raw: Any, name: str) -> Any:
        tol = _obj(raw, name, set(), {"abs_tol", "rel_tol"})
        return _wrap(ValueTolerance, **{k: _num(v, f"{name}.{k}") for k, v in tol.items()})

    kwargs: dict[str, Any] = {}
    for key in (
        "prior_probability",
        "corroborated_at",
        "contradicted_at",
        "reliability_epsilon",
        "confidence_level",
    ):
        if key in item:
            kwargs[key] = _num(item[key], f"config.{key}")
    if "low_reliability_policy" in item:
        kwargs["low_reliability_policy"] = item["low_reliability_policy"]
    if "default_tolerance" in item:
        kwargs["default_tolerance"] = tolerance(
            item["default_tolerance"], "config.default_tolerance"
        )
    if "predicate_tolerances" in item:
        table = item["predicate_tolerances"]
        if not isinstance(table, Mapping):
            raise InputError("config.predicate_tolerances must map predicates to tolerances")
        kwargs["predicate_tolerances"] = {
            str(k): tolerance(v, f"config.predicate_tolerances.{k}") for k, v in table.items()
        }
    return _wrap(VerifierConfig, **kwargs)


def _statement_text(statements: Sequence[Statement]) -> str:
    parts: list[str] = []
    for statement in statements:
        parts.extend(f"{c.subject} {c.predicate} {c.value}" for c in statement.claims)
        parts.extend(d.text for d in statement.details)
    return " ; ".join(parts)


def verify_statements(payload: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    """Verify statements against evidence (``truth_decipher.StatementVerifier``).

    Input: ``statements`` (required), ``evidence``, ``config``,
    ``source_reliability`` (source id to reliability), and
    ``verifiability_reference`` (a context-specific reference distribution of
    verifiable-detail proportions).

    Returns:
        ``{"report": VerificationReport.as_dict(), "gate": ...}``. The report
        never labels a person or an account (see the module's framing).
    """
    from omni_mercury_engine.truth_decipher import StatementVerifier, TableSourceReliability

    item = _obj(
        payload,
        "request",
        {"statements"},
        {"evidence", "config", "source_reliability", "verifiability_reference"},
    )
    statements = [
        _statement(s, i)
        for i, s in enumerate(_list(item["statements"], "statements", MAX_STATEMENTS))
    ]
    evidence = [
        _evidence(e, i)
        for i, e in enumerate(_list(item.get("evidence", []), "evidence", MAX_EVIDENCE))
    ]
    table_raw = item.get("source_reliability") or {}
    if not isinstance(table_raw, Mapping):
        raise InputError("source_reliability must map source ids to reliabilities")
    table = {str(k): _num(v, f"source_reliability.{k}") for k, v in table_raw.items()}
    reference = item.get("verifiability_reference")
    ref = (
        None
        if reference is None
        else [
            _num(v, "verifiability_reference")
            for v in _list(reference, "verifiability_reference", 100_000)
        ]
    )
    config = _verifier_config(item.get("config"))
    verdict = gate(
        surface=surface,
        operation="verify statements against independent evidence; report corroboration, "
        "contradictions and consistency without judging any person",
        domain="intelligence",
        request=_statement_text(statements),
        payload={"n_statements": len(statements), "n_evidence": len(evidence)},
    )
    verifier = StatementVerifier(config, reliability_model=_wrap(TableSourceReliability, table))
    report = _wrap(verifier.verify, statements, evidence, verifiability_reference=ref)
    return {"report": report.as_dict(), "gate": verdict.as_dict()}


# --------------------------------------------------------------------------- #
# Consent.
# --------------------------------------------------------------------------- #
def _consent(value: Any) -> ConsentRecord:
    from omni_mercury_engine.truth_decipher import ConsentRecord

    item = _obj(
        value,
        "consent",
        {"subject_pseudonym", "obtained_at", "scope", "obtained_by"},
        {"withdrawn_at"},
    )
    scope = item["scope"]
    scope = [scope] if isinstance(scope, str) else _list(scope, "consent.scope", 32)
    return _wrap(
        ConsentRecord,
        subject_pseudonym=_pseudonym(item["subject_pseudonym"], "consent.subject_pseudonym"),
        obtained_at=_num(item["obtained_at"], "consent.obtained_at"),
        scope=frozenset(_str(s, "consent.scope entry") for s in scope),
        obtained_by=_str(item["obtained_by"], "consent.obtained_by"),
        withdrawn_at=_opt_num(item.get("withdrawn_at"), "consent.withdrawn_at"),
    )


def _consent_fingerprint_source(record: ConsentRecord) -> dict[str, Any]:
    return {
        "subject_pseudonym": record.subject_pseudonym,
        "obtained_at": record.obtained_at,
        "scope": sorted(record.scope),
        "obtained_by": record.obtained_by,
    }


# --------------------------------------------------------------------------- #
# Concealed Information Test.
# --------------------------------------------------------------------------- #
def _series(value: Any, name: str) -> Any:
    from omni_mercury_engine.truth_decipher import TimeSeries

    if value is None:
        return None
    item = _obj(value, name, {"times", "values"})
    times = _list(item["times"], f"{name}.times", MAX_SIGNAL_VALUES)
    values = _list(item["values"], f"{name}.values", MAX_SIGNAL_VALUES)
    return _wrap(
        TimeSeries, np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)
    )


def _windows(value: Any) -> Any:
    from omni_mercury_engine.truth_decipher import ResponseWindows

    if value is None:
        return None
    item = _obj(
        value, "windows", set(), {"scr", "hr_baseline", "hr_response", "rll", "allow_overlap"}
    )
    kwargs: dict[str, Any] = {}
    for key in ("scr", "hr_baseline", "hr_response", "rll"):
        if key in item:
            pair = _list(item[key], f"windows.{key}", 2)
            if len(pair) != 2:
                raise InputError(f"windows.{key} must be [start, end]")
            kwargs[key] = (_num(pair[0], f"windows.{key}[0]"), _num(pair[1], f"windows.{key}[1]"))
    if "allow_overlap" in item:
        kwargs["allow_overlap"] = _bool(item["allow_overlap"], "windows.allow_overlap")
    return _wrap(ResponseWindows, **kwargs)


def _cit_config(value: Any) -> Any:
    from omni_mercury_engine.truth_decipher import CITConfig

    if value is None:
        return CITConfig()
    item = _obj(
        value,
        "config",
        set(),
        {
            "alpha",
            "primary_test",
            "channel_weights",
            "max_exact_permutations",
            "monte_carlo_permutations",
            "seed",
        },
    )
    kwargs: dict[str, Any] = {}
    if "alpha" in item:
        kwargs["alpha"] = _num(item["alpha"], "config.alpha")
    if "primary_test" in item:
        kwargs["primary_test"] = item["primary_test"]
    if "channel_weights" in item:
        weights = item["channel_weights"]
        if not isinstance(weights, Mapping):
            raise InputError("config.channel_weights must map channel names to weights")
        kwargs["channel_weights"] = {
            k: _num(v, f"config.channel_weights.{k}") for k, v in weights.items()
        }
    for key in ("max_exact_permutations", "monte_carlo_permutations", "seed"):
        if key in item:
            if isinstance(item[key], bool) or not isinstance(item[key], int):
                raise InputError(f"config.{key} must be an integer")
            kwargs[key] = item[key]
    return _wrap(CITConfig, **kwargs)


def _cit_question(value: Any, index: int, series: Mapping[str, Any], windows: Any) -> Any:
    from omni_mercury_engine.truth_decipher import CITQuestion, question_from_time_series

    name = f"questions[{index}]"
    item = _obj(
        value,
        name,
        {"question_id", "probe", "irrelevants"},
        {"responses", "presentations", "probe_publicly_disclosed"},
    )
    irrelevants = tuple(
        _str(x, f"{name}.irrelevants")
        for x in _list(item["irrelevants"], f"{name}.irrelevants", 64)
    )
    disclosed = _bool(
        item.get("probe_publicly_disclosed", False), f"{name}.probe_publicly_disclosed"
    )
    if ("responses" in item) == ("presentations" in item):
        raise InputError(
            f"{name} needs exactly one of 'responses' (amplitudes) or 'presentations' (onsets)"
        )
    if "responses" in item:
        responses = item["responses"]
        if not isinstance(responses, Mapping):
            raise InputError(
                f"{name}.responses must map channel names to repetitions x items matrices"
            )
        return _wrap(
            CITQuestion,
            question_id=item["question_id"],
            probe=item["probe"],
            irrelevants=irrelevants,
            responses={k: _list(v, f"{name}.responses.{k}", 10_000) for k, v in responses.items()},
            probe_publicly_disclosed=disclosed,
        )
    presentations = []
    for i, p in enumerate(_list(item["presentations"], f"{name}.presentations", 10_000)):
        pair = _list(p, f"{name}.presentations[{i}]", 2)
        if len(pair) != 2:
            raise InputError(f"{name}.presentations[{i}] must be [item label, onset seconds]")
        presentations.append((_str(pair[0], "item label"), _num(pair[1], "onset")))
    if not series:
        raise InputError(f"{name} uses presentations but the request has no 'series'")
    try:
        return question_from_time_series(
            item["question_id"],
            item["probe"],
            irrelevants,
            presentations,
            windows=windows,
            probe_publicly_disclosed=disclosed,
            **series,
        )
    except (TypeError, ValueError) as exc:  # InsufficientDataError is a ValueError
        raise InputError(f"{name}: {exc}") from exc


def concealed_information_test(
    payload: Mapping[str, Any], *, surface: str, environ: dict[str, str] | None = None
) -> dict[str, Any]:
    """Score a Concealed Information Test session (``truth_decipher.ConcealedInformationTest``).

    Input: ``subject_pseudonym``, ``administered_at`` (seconds, consent's
    clock), ``purpose``, ``consent`` (a ConsentRecord whose scope includes
    ``"concealed_information_test"``), ``questions``; optional ``series``
    (session recordings: ``skin_conductance``/``heart_rate``/``respiration``
    as ``{times, values}``, used by questions given as ``presentations``),
    ``windows``, ``config``, ``probe_details_publicly_disclosed``.

    The CIT measures recognition of event details, not deception, and a
    finding is not evidence of guilt (stated in every result).

    Raises:
        CapabilityDisabledError: The operator has not enabled ``consent``.
        ConsentRequiredError: The consent record does not cover the session.
    """
    from omni_mercury_engine.truth_decipher import (
        ConcealedInformationTest,
        require_consent,
    )

    require_basis_enabled(LawfulBasis.CONSENT, "the Concealed Information Test", environ)
    item = _obj(
        payload,
        "request",
        {"subject_pseudonym", "administered_at", "purpose", "consent", "questions"},
        {"series", "windows", "config", "probe_details_publicly_disclosed"},
    )
    subject = _pseudonym(item["subject_pseudonym"])
    administered_at = _num(item["administered_at"], "administered_at")
    consent = _consent(item["consent"])
    require_consent(consent, subject, administered_at)
    series_raw = item.get("series") or {}
    series_obj = _obj(
        series_raw, "series", set(), {"skin_conductance", "heart_rate", "respiration"}
    )
    series = {k: _series(v, f"series.{k}") for k, v in series_obj.items() if v is not None}
    windows = _windows(item.get("windows"))
    questions = [
        _cit_question(q, i, series, windows)
        for i, q in enumerate(_list(item["questions"], "questions", MAX_CIT_QUESTIONS))
    ]
    config = _cit_config(item.get("config"))
    verdict = gate(
        surface=surface,
        operation="score a consented Concealed Information Test for recognition of event details",
        domain="human_subject",
        request=" ; ".join(f"{q.probe} {' '.join(q.irrelevants)}" for q in questions),
        payload={"n_questions": len(questions)},
        human_subject=True,
        purpose=item["purpose"],
        basis=LawfulBasis.CONSENT,
        authorisation=_consent_fingerprint_source(consent),
        environ=environ,
    )
    result = ConcealedInformationTest(config).run(
        questions,
        consent=consent,
        subject_pseudonym=subject,
        administered_at=administered_at,
        probe_details_publicly_disclosed=_bool(
            item.get("probe_details_publicly_disclosed", False), "probe_details_publicly_disclosed"
        ),
    )
    return {"result": result.as_dict(), "gate": verdict.as_dict()}


# --------------------------------------------------------------------------- #
# Contactless vital signs.
# --------------------------------------------------------------------------- #
_SIGNAL_FIELDS = (
    "csi_amplitude",
    "csi_phase",
    "subcarrier_indices",
    "radar_displacement_m",
    "radar_phase_rad",
    "timestamps_s",
)


def _signal(value: Any, name: str) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a numeric array: {exc}") from exc
    if array.dtype == object or array.ndim == 0:
        raise InputError(f"{name} must be a rectangular numeric array")
    return array


def _vital_config(value: Any) -> Any:
    from omni_mercury_engine.biometric.rf_vital_signs import VitalSignConfig

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InputError("config must be an object of VitalSignConfig fields")
    known = {f.name for f in fields(VitalSignConfig)}
    unknown = set(value) - known
    if unknown:
        raise InputError(f"config has unknown keys {sorted(unknown)}")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in value.items()}
    return _wrap(VitalSignConfig, **kwargs)


def _vital_result(result: Any, include_waveform: bool) -> dict[str, Any]:
    out = jsonable(result)
    if not include_waveform:
        for key in ("respiration_waveform", "sample_times_s", "motion_mask", "missing_mask"):
            out.pop(key, None)
    return dict(out)


def vital_signs(
    payload: Mapping[str, Any], *, surface: str, environ: dict[str, str] | None = None
) -> dict[str, Any]:
    """Estimate (or track) respiration and heart rate from radar or Wi-Fi CSI.

    Input: ``purpose``, ``lawful_basis`` (``"consent"`` or
    ``"vital_interests"``), ``fs``, ``recorded_at`` (seconds, consent's
    clock), exactly one modality (``csi_amplitude``/``csi_phase``, or
    ``radar_displacement_m``, or ``radar_phase_rad`` with ``wavelength_m``);
    optional ``mode`` (``"estimate"``, default, or ``"track"``),
    ``subcarrier_indices``, ``timestamps_s``, ``window_s``, ``step_s``,
    ``t0_s``, ``config``, ``include_waveform`` (default false).

    Under ``consent``: ``subject_pseudonym`` and a ``consent`` record whose
    scope includes ``"contactless_vital_signs"``, obtained before
    ``recorded_at`` and not withdrawn. Under ``vital_interests`` (locating
    people who cannot consent, such as survivors under rubble): an
    ``attestation`` ``{incident_id, attested_by}``.

    Raises:
        CapabilityDisabledError: The operator has not enabled the basis.
        ConsentRequiredError: The consent record does not cover the recording.
    """
    from omni_mercury_engine.biometric.rf_vital_signs import (
        VITAL_SIGNS_CONSENT_SCOPE,
        estimate_vital_signs,
        track_vital_signs,
    )
    from omni_mercury_engine.truth_decipher import require_consent

    item = _obj(
        payload,
        "request",
        {"purpose", "lawful_basis", "fs", "recorded_at"},
        {
            "mode",
            "subject_pseudonym",
            "consent",
            "attestation",
            "wavelength_m",
            "window_s",
            "step_s",
            "t0_s",
            "config",
            "include_waveform",
            *_SIGNAL_FIELDS,
        },
    )
    try:
        basis = LawfulBasis(item["lawful_basis"])
    except ValueError as exc:
        raise InputError(f"lawful_basis must be one of {[b.value for b in LawfulBasis]}") from exc
    require_basis_enabled(basis, "contactless vital-sign sensing", environ)
    recorded_at = _num(item["recorded_at"], "recorded_at")
    consent_confirmed: Any
    if basis is LawfulBasis.CONSENT:
        if "attestation" in item:
            raise InputError("an attestation applies to lawful_basis 'vital_interests' only")
        subject = _pseudonym(item.get("subject_pseudonym"))
        record = _consent(item.get("consent"))
        require_consent(record, subject, recorded_at, scope=VITAL_SIGNS_CONSENT_SCOPE)
        consent_confirmed = record
        authorisation: dict[str, Any] = _consent_fingerprint_source(record)
    else:
        if "consent" in item or "subject_pseudonym" in item:
            raise InputError(
                "lawful_basis 'vital_interests' covers people who cannot consent; send an "
                "attestation, not a consent record or subject pseudonym"
            )
        attestation = _obj(item.get("attestation"), "attestation", {"incident_id", "attested_by"})
        consent_confirmed = True
        authorisation = {
            "incident_id": _str(attestation["incident_id"], "attestation.incident_id"),
            "attested_by": _str(attestation["attested_by"], "attestation.attested_by"),
            "recorded_at": recorded_at,
        }
    mode = item.get("mode", "estimate")
    if mode not in ("estimate", "track"):
        raise InputError("mode must be 'estimate' or 'track'")
    signals = {name: _signal(item.get(name), name) for name in _SIGNAL_FIELDS}
    total = sum(int(a.size) for a in signals.values() if a is not None)
    if total > MAX_SIGNAL_VALUES:
        raise InputError(
            f"{total} signal values; at most {MAX_SIGNAL_VALUES} are accepted per request"
        )
    include_waveform = _bool(item.get("include_waveform", False), "include_waveform")
    config = _vital_config(item.get("config"))
    fs = _num(item["fs"], "fs")
    verdict = gate(
        surface=surface,
        operation="estimate respiration and heart rate from contactless radar or Wi-Fi CSI",
        domain="human_subject",
        request="",
        payload={"mode": mode, "fs": fs, "n_values": total},
        human_subject=True,
        purpose=item["purpose"],
        basis=basis,
        authorisation=authorisation,
        environ=environ,
    )
    common: dict[str, Any] = {
        "fs": fs,
        "consent_confirmed": consent_confirmed,
        "wavelength_m": _opt_num(item.get("wavelength_m"), "wavelength_m"),
        "config": config,
        **signals,
    }
    result: VitalSignEstimate | VitalSignTrack
    if mode == "estimate":
        result = _wrap(estimate_vital_signs, **common)
    else:
        result = _wrap(
            track_vital_signs,
            window_s=_num(item.get("window_s", 30.0), "window_s"),
            step_s=_num(item.get("step_s", 1.0), "step_s"),
            t0_s=_opt_num(item.get("t0_s"), "t0_s"),
            **common,
        )
    return {
        "mode": mode,
        "result": _vital_result(result, include_waveform),
        "gate": verdict.as_dict(),
    }


# --------------------------------------------------------------------------- #
# Counterintelligence screening and all-source fusion.
# --------------------------------------------------------------------------- #
_ENGINE_LOCK = threading.Lock()
_ENGINE: IntelligenceFusionEngine | None = None
_ENGINE_STATUS: dict[str, Any] = {}


def _reports(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise InputError("intel_reports must be a non-empty object of discipline -> report(s)")
    count = sum(len(v) if isinstance(v, list) else 1 for v in value.values())
    if count > MAX_REPORTS:
        raise InputError(f"{count} reports; at most {MAX_REPORTS} are accepted")
    return dict(value)


def _report_text(reports: Mapping[str, Any]) -> str:
    texts: list[str] = []
    for value in reports.values():
        for report in value if isinstance(value, list) else [value]:
            if isinstance(report, Mapping):
                for key in ("text", "summary", "description", "content"):
                    if isinstance(report.get(key), str):
                        texts.append(report[key])
                indicators = report.get("indicators")
                if isinstance(indicators, list):
                    texts.extend(str(i) for i in indicators if isinstance(i, str))
    return " ; ".join(texts)


def fusion_engine() -> IntelligenceFusionEngine:
    """The served all-source fusion engine, built once.

    With ``MERCURY_INTEL_LEARN_RELIABILITY=1`` the per-discipline
    reliabilities are first learned from the ledger's training split
    (installed only if cross-validation says they beat the declared table).
    When ``MERCURY_INTEL_FUSION_CHECKPOINT`` names a checkpoint, it is put
    through the merit gate against the adjudication ledger's held-out labels
    (``MERCURY_ADJUDICATION_PATH``): admitted only if significantly better
    than evidence scoring, with attempts counted in
    ``MERCURY_MERIT_ATTEMPT_LOG`` when that is set. Otherwise, or on refusal,
    the engine scores from evidence. :func:`fusion_status` reports which.
    """
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine

        engine = IntelligenceFusionEngine()
        checkpoint = os.environ.get(_FUSION_CHECKPOINT_ENV, "").strip()
        status: dict[str, Any] = {"decides_with": "evidence", "checkpoint": checkpoint or None}
        if os.environ.get(_LEARN_RELIABILITY_ENV, "").strip().lower() in ("1", "true", "on"):
            # Training split only: the held-out split stays unspent for the
            # checkpoint's merit test below.
            try:
                fit = engine.fit_reliability_scores(_ledger().training_cases())
                status["reliability"] = asdict(fit)
            except Exception as exc:  # keep the declared table; record why
                status["reliability"] = {
                    "installed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        if checkpoint:
            try:
                heldout = _ledger().export_heldout_for_merit_gate()
                merit = engine.load_trained_weights(checkpoint, heldout)
                status.update(decides_with="network", merit=asdict(merit))
            except Exception as exc:  # refusal keeps evidence scoring; record why
                status["refused"] = f"{type(exc).__name__}: {exc}"
        _ENGINE_STATUS.clear()
        _ENGINE_STATUS.update(status)
        _ENGINE = engine
        return engine


def fusion_status() -> dict[str, Any]:
    """Which path the served fusion engine decides with, and why."""
    fusion_engine()
    return dict(_ENGINE_STATUS)


def screen_intel_reports(payload: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    """Counterintelligence screens over a report batch (no score, no threat level).

    Input: ``intel_reports`` (discipline to report or list of reports),
    optional ``history`` (``outcomes``, ``baseline_rate_per_hour``,
    ``default_baseline_rate_per_hour``).

    Returns:
        ``{"assessment": CounterintelligenceAssessment.as_dict(), "gate": ...}``.
        Flags are leads for an analyst, not findings of deception.
    """
    from omni_mercury_engine.security.counterintelligence import CounterintelligenceAssessor

    item = _obj(payload, "request", {"intel_reports"}, {"history"})
    reports = _reports(item["intel_reports"])
    verdict = gate(
        surface=surface,
        operation="screen intelligence reports for circular reporting, coordination, "
        "fabrication and impossible travel",
        domain="intelligence",
        request=_report_text(reports),
        payload={"disciplines": sorted(reports)},
    )
    assessor = CounterintelligenceAssessor.from_engine(fusion_engine())
    try:
        assessment = assessor.assess(reports, item.get("history"))
    except (TypeError, ValueError) as exc:
        raise InputError(str(exc)) from exc
    return {"assessment": assessment.as_dict(), "gate": verdict.as_dict()}


def fuse_intelligence(payload: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    """All-source fusion with counterintelligence screening (``IntelligenceFusionEngine``).

    Input: ``intel_reports``, optional ``temporal_context`` (a list of
    earlier report mappings) and ``history`` (as :func:`screen_intel_reports`).

    Returns:
        ``{"result": IntelligenceFusionResult, "decides_with": ..., "gate": ...}``.
    """
    from omni_mercury_engine.core.exceptions import InputModalityError

    item = _obj(payload, "request", {"intel_reports"}, {"temporal_context", "history"})
    reports = _reports(item["intel_reports"])
    temporal = item.get("temporal_context")
    if temporal is not None:
        temporal = _list(temporal, "temporal_context", MAX_REPORTS)
    verdict = gate(
        surface=surface,
        operation="fuse screened multi-source intelligence reports into a threat assessment",
        domain="intelligence",
        request=_report_text(reports),
        payload={"disciplines": sorted(reports)},
    )
    engine = fusion_engine()
    try:
        result = engine.fuse_intelligence(
            reports, temporal, counterintelligence_history=item.get("history")
        )
    except (TypeError, ValueError, InputModalityError) as exc:
        raise InputError(str(exc)) from exc
    return {
        "result": jsonable(result),
        "decides_with": _ENGINE_STATUS.get("decides_with", "evidence"),
        "gate": verdict.as_dict(),
    }


# --------------------------------------------------------------------------- #
# Adjudication ledger.
# --------------------------------------------------------------------------- #
def _ledger() -> AdjudicationLedger:
    from omni_mercury_engine.security.adjudication import AdjudicationLedger
    from omni_mercury_engine.security.signing_keys import pinned_public_key_from_env

    return AdjudicationLedger(expected_public_key_hex=pinned_public_key_from_env())


def _signing_keypair() -> tuple[bytes, bytes] | None:
    from omni_mercury_engine.security.signing_keys import keypair_from_env

    return keypair_from_env()


def _when(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise InputError(f"{name} must be an ISO 8601 timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{name} {value!r} is not an ISO 8601 timestamp") from exc
    if parsed.utcoffset() is None:
        raise InputError(f"{name} must carry a timezone")
    return parsed.astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)


def adjudication_status() -> dict[str, Any]:
    """Chain validity, agreement and counts, and the work queues (when the chain verifies)."""
    from omni_mercury_engine.security.adjudication import ExportRefused

    ledger = _ledger()
    ok, reason = ledger.verify()
    out: dict[str, Any] = {"ledger": str(ledger.path), "valid": ok, "reason": reason}
    if not ledger.path.exists():
        out["stats"] = None
        out["pending"] = None
        return out
    out["stats"] = jsonable(ledger.stats())
    try:
        out["pending"] = ledger.pending()
    except ExportRefused as exc:
        out["pending"] = None
        out["pending_refused"] = str(exc)
    return out


def adjudication_verify(expected_head_digest: str | None = None) -> dict[str, Any]:
    """Verify the chain (and pinned signatures); optionally that a head is still in it."""
    ledger = _ledger()
    ok, reason = ledger.verify(expected_head_digest=expected_head_digest)
    return {"ledger": str(ledger.path), "valid": ok, "reason": reason}


def adjudication_rater_view(rater_id: str, case_id: str) -> dict[str, Any]:
    """What ``rater_id`` may see of ``case_id`` (blind until they have rated)."""
    try:
        view = _ledger().rater_view(_str(rater_id, "rater_id"), _str(case_id, "case_id"))
    except KeyError as exc:
        raise InputError(f"case {case_id!r} is not recorded") from exc
    return dict(jsonable(view))


def _append(write: Any, obj: Any) -> dict[str, Any]:
    try:
        record = write(obj, signing_keypair=_signing_keypair())
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    return {
        "index": record["index"],
        "record_digest": record["record_digest"],
        "kind": record["entry"]["kind"],
        "signed": "signature" in record,
    }


def adjudication_add_case(payload: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    """Record a case (its split is fixed by its id)."""
    from omni_mercury_engine.security.adjudication import Case

    item = _obj(
        payload,
        "case",
        {"case_id", "intel_reports", "report_source_ids", "latest_report_at"},
        {"created_at"},
    )
    reports = _reports(item["intel_reports"])
    gate(
        surface=surface,
        operation="record an intelligence case in the adjudication ledger",
        domain="intelligence",
        request=_report_text(reports),
        payload={"case_id": item["case_id"]},
    )
    case = _wrap(
        Case,
        case_id=item["case_id"],
        created_at=_when(item["created_at"], "created_at") if "created_at" in item else _now(),
        intel_reports=reports,
        report_source_ids=tuple(_list(item["report_source_ids"], "report_source_ids", MAX_REPORTS)),
        latest_report_at=_when(item["latest_report_at"], "latest_report_at"),
    )
    return _append(_ledger().add_case, case)


def adjudication_add_outcome(payload: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    """Record what happened in a case."""
    from omni_mercury_engine.security.adjudication import Outcome

    item = _obj(
        payload,
        "outcome",
        {"case_id", "observed_at", "description", "outcome_source_ids"},
        {"outcome_level", "outcome_scale"},
    )
    gate(
        surface=surface,
        operation="record a case outcome in the adjudication ledger",
        domain="intelligence",
        request=_str(item["description"], "description"),
        payload={"case_id": item["case_id"]},
    )
    outcome = _wrap(
        Outcome,
        case_id=item["case_id"],
        observed_at=_when(item["observed_at"], "observed_at"),
        description=item["description"],
        outcome_source_ids=tuple(
            _list(item["outcome_source_ids"], "outcome_source_ids", MAX_REPORTS)
        ),
        outcome_level=item.get("outcome_level"),
        outcome_scale=item.get("outcome_scale"),
    )
    return _append(_ledger().add_outcome, outcome)


def adjudication_add_rating(
    payload: Mapping[str, Any], *, rater_id: str, surface: str
) -> dict[str, Any]:
    """Record a blind rating by ``rater_id`` (the authenticated principal), stamped now.

    The rater id and time come from the surface, not the request: a caller can
    neither rate as somebody else nor backdate a rating.
    """
    from omni_mercury_engine.security.adjudication import Rating

    item = _obj(payload, "rating", {"case_id", "level", "rationale"})
    gate(
        surface=surface,
        operation="record a blind threat-level rating in the adjudication ledger",
        domain="intelligence",
        request=_str(item["rationale"], "rationale"),
        payload={"case_id": item["case_id"]},
    )
    rating = _wrap(
        Rating,
        _str(rater_id, "rater_id"),
        item["case_id"],
        item["level"],
        item["rationale"],
        _now(),
    )
    return _append(_ledger().add_rating, rating)


def adjudication_add_decision(
    payload: Mapping[str, Any], *, adjudicator_id: str, surface: str
) -> dict[str, Any]:
    """Record an adjudicator's decision by ``adjudicator_id``, stamped now."""
    from omni_mercury_engine.security.adjudication import AdjudicatorDecision

    item = _obj(payload, "decision", {"case_id", "level", "rationale"})
    gate(
        surface=surface,
        operation="record an adjudicator decision in the adjudication ledger",
        domain="intelligence",
        request=_str(item["rationale"], "rationale"),
        payload={"case_id": item["case_id"]},
    )
    decision = _wrap(
        AdjudicatorDecision,
        _str(adjudicator_id, "adjudicator_id"),
        item["case_id"],
        item["level"],
        item["rationale"],
        _now(),
    )
    return _append(_ledger().add_decision, decision)


def adjudication_export(split: str, out: str | None = None) -> dict[str, Any]:
    """Export one split's labels, signed when ``MERCURY_ADJUDICATION_SIGNING_KEY`` is set."""
    from omni_mercury_engine.security.adjudication import SPLITS

    if split not in SPLITS:
        raise InputError(f"split must be one of {list(SPLITS)}")
    return _ledger().export(split, out, signing_keypair=_signing_keypair())


def adjudication_repair() -> dict[str, Any]:
    """Drop a crash-torn last line (only that; any other damage is left for verify)."""
    ledger = _ledger()
    try:
        removed = ledger.repair_torn_tail()
    except ValueError as exc:
        return {"ledger": str(ledger.path), "repaired": False, "reason": str(exc)}
    return {"ledger": str(ledger.path), "repaired": True, "bytes_removed": removed}


__all__ = [
    "MAX_CIT_QUESTIONS",
    "MAX_EVIDENCE",
    "MAX_REPORTS",
    "MAX_SIGNAL_VALUES",
    "MAX_STATEMENTS",
    "CapabilityDisabledError",
    "InputError",
    "adjudication_add_case",
    "adjudication_add_decision",
    "adjudication_add_outcome",
    "adjudication_add_rating",
    "adjudication_export",
    "adjudication_rater_view",
    "adjudication_repair",
    "adjudication_status",
    "adjudication_verify",
    "concealed_information_test",
    "fuse_intelligence",
    "fusion_engine",
    "fusion_status",
    "jsonable",
    "screen_intel_reports",
    "verify_statements",
    "vital_signs",
]
