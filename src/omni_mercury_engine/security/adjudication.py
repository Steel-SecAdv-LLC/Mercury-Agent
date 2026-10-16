# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adjudicated, non-circular, held-out threat-level labels for the fusion merit gate.

:meth:`~omni_mercury_engine.security.intelligence_fusion.IntelligenceFusionEngine.load_trained_weights`
admits fusion weights only when they beat evidence scoring on at least
``MIN_HELDOUT_CASES`` labelled held-out cases (one-sided exact McNemar test).
No public, labelled, multi-INT dataset exists, so the labels have to be made
here, from what actually happened. This module is the auditable source of
those labels: an append-only ledger of cases, observed outcomes, blind
ratings and adjudicator decisions, from which the labelled ``(intel_reports,
threat level)`` pairs are derived and exported.

Records (every timestamp is timezone-aware and stored in UTC):

* :class:`Case` -- the payload fusion is scored on (``intel_reports``), the
  sources that produced it (``report_source_ids``) and when the latest of
  those reports was produced (``latest_report_at``). The ledger records the
  case's split with it at first write; it never changes.
* :class:`Outcome` -- what happened (``description``), when it could first
  have happened (``observed_at``), and the sources it is known from
  (``outcome_source_ids``). ``outcome_level`` (1-5) is set only together
  with ``outcome_scale``, the documented scale it was derived from (for
  example :data:`UCDP_FATALITY_SCALE`).
* :class:`Rating` -- one rater's threat level (1-5) for a case's *outcome*,
  with a rationale. Raters rate outcomes, not reports: :meth:`AdjudicationLedger.rater_view`
  shows the outcome description and never the ``intel_reports``, so a label
  cannot anchor on the evidence the fusion network and the evidence path both
  see. Ratings are blind: the view never shows a rater any other rating on a
  case (not even how many there are) until that rater has submitted.
* :class:`AdjudicatorDecision` -- a third party's resolution of a
  disagreement, by someone who did not rate the case.

Adjudication (:class:`Adjudication`, method recorded with every label):

* The first two ratings agree: that level (``consensus``).
* They differ by one level: an adjudicator decision, or a third blind rating
  whose majority settles it. No automatic rule is applied -- taking the
  higher (or lower) level would bias every label in one direction, and an
  integer scale has no midpoint.
* They differ by more than one level: a third blind rating is required; an
  adjudicator decision alone is refused.
* Three ratings with a majority: the majority (``majority_of_three``); three
  distinct ratings: an adjudicator decision is required.
* An adjudicator decision (``adjudicator_decision``) must lie within the
  range of the ratings it resolves.

Non-circularity. A label is exported only if the outcome's sources are
disjoint from the report sources (after :func:`normalize_source_id`) and the
outcome's ``observed_at`` is strictly after ``latest_report_at``: a case is
never labelled by the same publisher that supplied its inputs, nor by an
event its inputs could have described.

Agreement (Cohen 1968; Fleiss & Cohen 1973). For every pair of raters, the
quadratic-weighted Cohen's kappa is computed over the cases on which they gave
the *first two* ratings. Third ratings are excluded: they are requested only
on disagreements, so counting them would select on the outcome and bias kappa
low. Cases still awaiting adjudication *are* counted -- leaving disagreements
out would inflate kappa. The point estimate is computed in exact rational
arithmetic (``fractions.Fraction``), so the gate's boundary is not subject to
float rounding. A seeded percentile-bootstrap confidence interval (Efron &
Tibshirani 1993) resamples the pair's cases with replacement. Labels are
exportable only when every rater pair with at least ``min_pair_cases``
co-rated cases reaches ``kappa_threshold`` (and at least one pair does);
cases rated by a pair below that minimum are excluded as having unmeasured
agreement. A pair whose kappa is undefined (no expected disagreement, e.g.
every rating identical) fails closed.

Configurable defaults (:class:`AgreementPolicy`), each a convention:

* ``kappa_threshold = 0.61`` -- the lower bound of the "substantial" band of
  Landis & Koch (1977). Their bands are a convention, which the authors
  themselves call arbitrary benchmarks; quadratic weighting gives partial
  credit to near misses, so 0.61 on a quadratic kappa is a weaker bar than
  0.61 on an unweighted one.
* ``min_pair_cases = 30`` -- a Mercury convention matching
  ``MIN_HELDOUT_CASES``, so agreement and merit are measured on samples of the
  same minimum size; it is not derived from a power analysis.
* ``n_boot = 2000``, ``confidence = 0.95``, ``seed = 0`` -- Efron & Tibshirani
  (1993) advise on the order of a thousand or more replicates for percentile
  intervals, far more than for a standard error.
* ``gate_on = "point"`` -- the gate uses the point estimate; ``"ci_lower"``
  gates on the interval's lower bound instead (a stricter reading).

Split. A case is held out iff ``int(sha3_256(case_id).hexdigest(), 16) % 100
< heldout_pct``. The default ``heldout_pct = 25`` follows the "typical split
... 50% for training, and 25% each for validation and testing" of Hastie,
Tibshirani & Friedman (2009, sec. 7.2), who stress that there is no general
rule; a validation set can be carved from the training split. The rule is
fixed by a ledger's first case: a case recording a different percentage, or a
split its hash rule does not give, is refused.

Storage mirrors :class:`omni_mercury_engine.self_model.SelfModelLedger`:
JSON lines ``{schema, index, prev_digest, entry, record_digest,
signature?}``, a SHA3-256 hash chain over canonical JSON, optional ML-DSA-65
signatures through :class:`~omni_mercury_engine.security.crypto_api.MLDSAProvider`
(checked against a pinned key). :meth:`AdjudicationLedger.verify` fails
closed with the record index and reason; an append never extends a broken
chain. On top of the chain, every entry is re-checked against the protocol
on every read: a file written around the API (a foreign or forged writer who
re-sealed the chain) is refused for export, naming the record that breaks the
protocol, rather than trusted because its digests are consistent.

References:
    Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision
    for scaled disagreement or partial credit. Psychological Bulletin,
    70(4), 213-220.

    Fleiss, J. L., & Cohen, J. (1973). The equivalence of weighted kappa and
    the intraclass correlation coefficient as measures of reliability.
    Educational and Psychological Measurement, 33(3), 613-619.

    Landis, J. R., & Koch, G. G. (1977). The measurement of observer
    agreement for categorical data. Biometrics, 33(1), 159-174.

    Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    Chapman & Hall.

    Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
    Statistical Learning (2nd ed.). Springer.

    Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical
    packages. The American Statistician, 50(4), 361-365.

    Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced
    Event Dataset. Journal of Peace Research, 50(4), 523-532.

    Hogbladh, S. (2026). UCDP GED Codebook version 26.1. Department of Peace
    and Conflict Research, Uppsala University.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

LEDGER_SCHEMA = "mercury.adjudication.ledger/v1"
EXPORT_SCHEMA = "mercury.adjudication.export/v1"
GENESIS_DIGEST = "0" * 64

#: Threat levels: the values of ``intelligence_fusion.ThreatLevel`` (LOW=1 .. CRITICAL=5).
LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)
KINDS: tuple[str, ...] = ("case", "outcome", "rating", "decision")
SPLITS: tuple[str, ...] = ("heldout", "training")
METHODS: tuple[str, ...] = ("consensus", "majority_of_three", "adjudicator_decision")

#: Landis & Koch (1977) lower bound of "substantial" agreement (a convention).
DEFAULT_KAPPA_THRESHOLD = 0.61
#: Mercury convention: the same minimum sample size as ``MIN_HELDOUT_CASES``.
DEFAULT_MIN_PAIR_CASES = 30
#: Bootstrap replicates for the percentile interval (Efron & Tibshirani 1993).
DEFAULT_N_BOOT = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0
#: Held-out percentage (Hastie, Tibshirani & Friedman 2009, sec. 7.2: a typical 25% test share).
DEFAULT_HELDOUT_PCT = 25
GATE_STATISTICS: tuple[str, ...] = ("point", "ci_lower")

#: Name of the documented outcome scale :func:`ucdp_fatality_level` implements.
UCDP_FATALITY_SCALE = "mercury.ucdp_fatality_level/v1"
#: Source id every UCDP-derived outcome carries, so a report built from UCDP is circular.
UCDP_SOURCE_ID = "UCDP GED"
#: Upper bounds (inclusive) of fatalities per level; anything above the last is level 5.
_UCDP_FATALITY_BINS: tuple[tuple[int, int], ...] = ((0, 1), (4, 2), (24, 3), (99, 4))
#: UCDP dates are local calendar days (codebook 26.1, sec. 5.6); UTC+14 is the
#: earliest local time anywhere, so its midnight is the earliest instant a day starts.
_EARLIEST_ZONE = timezone(timedelta(hours=14))
_UCDP_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})(?: 00:00:00(?:\.0+)?)?")
_UCDP_COUNT = re.compile(r"\d+")


class ExportRefused(ValueError):
    """Labels cannot be exported: the ledger, its content or its agreement fails a check."""


# ---------------------------------------------------------------------------
# Canonical form and field validation
# ---------------------------------------------------------------------------


def _canonical(payload: Any) -> bytes:
    """Canonical JSON (the self-model ledger's form), refusing non-finite numbers."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _digest(payload: Any) -> str:
    return hashlib.sha3_256(_canonical(payload)).hexdigest()


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


def _loads(line: str) -> Any:
    """Parse one ledger line; NaN and infinities are not JSON and are refused."""
    return json.loads(line, parse_constant=_reject_constant)


def normalize_source_id(source_id: str) -> str:
    """Return the comparison form of a source id: NFKC, case-folded, whitespace collapsed.

    Args:
        source_id: A publisher or collector name, e.g. ``"Vanguard"``.

    Returns:
        The normalised id; ``" Reuters "`` and ``"REUTERS"`` both become ``"reuters"``.
    """
    return " ".join(unicodedata.normalize("NFKC", source_id).casefold().split())


def _ident(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _level(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in LEVELS:
        raise ValueError(f"{name} must be an integer threat level 1-5, got {value!r}")
    return value


def _aware(name: str, value: Any) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _parse_time(name: str, value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO 8601 timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} {value!r} is not an ISO 8601 timestamp")
    return _aware(name, parsed)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _source_ids(name: str, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty list of source ids")
    for value in values:
        if not isinstance(value, str) or not normalize_source_id(value):
            raise ValueError(f"{name} must contain only non-empty strings")
    return tuple(values)


def _reports(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("intel_reports must be a non-empty mapping of discipline -> report")
    for key, report in value.items():
        if not isinstance(key, str) or not isinstance(report, Mapping):
            raise ValueError("intel_reports must map discipline names to report mappings")
    try:
        copied: dict[str, Any] = json.loads(_canonical(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"intel_reports must be JSON with finite numbers: {exc}")
    return copied


def _fields(entry: Mapping[str, Any], kind: str, names: tuple[str, ...]) -> None:
    expected = {"kind", "digest", *names}
    if entry.get("kind") != kind or set(entry) != expected:
        raise ValueError(
            f"malformed {kind} entry: fields {sorted(entry)} are not {sorted(expected)}"
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """An intelligence case: the payload fusion is scored on, and where it came from.

    Attributes:
        case_id: Stable identifier; it also fixes the case's split.
        created_at: When the case was assembled (never before ``latest_report_at``).
        intel_reports: The ``intel_reports`` mapping fed to fusion (JSON, finite numbers).
        report_source_ids: Every source that contributed a report.
        latest_report_at: When the most recent of those reports was produced.
    """

    case_id: str
    created_at: datetime
    intel_reports: dict[str, Any]
    report_source_ids: tuple[str, ...]
    latest_report_at: datetime

    def __post_init__(self) -> None:
        """Validate and normalise the fields (timestamps to UTC, reports to a JSON copy)."""
        _ident("case_id", self.case_id)
        object.__setattr__(self, "created_at", _aware("created_at", self.created_at))
        object.__setattr__(
            self, "latest_report_at", _aware("latest_report_at", self.latest_report_at)
        )
        object.__setattr__(self, "intel_reports", _reports(self.intel_reports))
        object.__setattr__(
            self, "report_source_ids", _source_ids("report_source_ids", self.report_source_ids)
        )
        if self.created_at < self.latest_report_at:
            raise ValueError(
                "created_at precedes latest_report_at: a case cannot be assembled before "
                "its reports exist"
            )

    def to_entry(self, split: str, heldout_pct: int) -> dict[str, Any]:
        """Return the ledger entry for this case with its recorded split."""
        return {
            "kind": "case",
            "case_id": self.case_id,
            "created_at": _iso(self.created_at),
            "intel_reports": json.loads(_canonical(self.intel_reports)),
            "report_source_ids": list(self.report_source_ids),
            "latest_report_at": _iso(self.latest_report_at),
            "split": split,
            "heldout_pct": heldout_pct,
        }

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> tuple[Case, str, int]:
        """Parse a ledger entry into ``(case, recorded split, recorded heldout_pct)``.

        Raises:
            ValueError: The entry is not a well-formed case entry.
        """
        _fields(
            entry,
            "case",
            (
                "case_id",
                "created_at",
                "intel_reports",
                "report_source_ids",
                "latest_report_at",
                "split",
                "heldout_pct",
            ),
        )
        case = cls(
            case_id=entry["case_id"],
            created_at=_parse_time("created_at", entry["created_at"]),
            intel_reports=entry["intel_reports"],
            report_source_ids=entry["report_source_ids"],
            latest_report_at=_parse_time("latest_report_at", entry["latest_report_at"]),
        )
        return case, entry["split"], _pct(entry["heldout_pct"])


@dataclass(frozen=True)
class Outcome:
    """What happened in a case, when it could first have happened, and who reported it.

    Attributes:
        case_id: The case this outcome resolves.
        observed_at: The earliest instant the outcome could have occurred; it
            must be strictly after the case's ``latest_report_at`` for the
            case to be labelled.
        description: What happened, as raters will read it.
        outcome_source_ids: The sources the outcome is known from; they must be
            disjoint from the case's report sources.
        outcome_level: Threat level 1-5 from a documented outcome scale, if any.
        outcome_scale: The documented scale ``outcome_level`` was derived from.
    """

    case_id: str
    observed_at: datetime
    description: str
    outcome_source_ids: tuple[str, ...]
    outcome_level: int | None = None
    outcome_scale: str | None = None

    def __post_init__(self) -> None:
        """Validate the fields; a level needs the scale it came from and vice versa."""
        _ident("case_id", self.case_id)
        object.__setattr__(self, "observed_at", _aware("observed_at", self.observed_at))
        _text("description", self.description)
        object.__setattr__(
            self, "outcome_source_ids", _source_ids("outcome_source_ids", self.outcome_source_ids)
        )
        if (self.outcome_level is None) != (self.outcome_scale is None):
            raise ValueError(
                "outcome_level and outcome_scale go together: a level needs the documented "
                "scale it was derived from"
            )
        if self.outcome_level is not None:
            _level("outcome_level", self.outcome_level)
            _text("outcome_scale", self.outcome_scale)

    def to_entry(self) -> dict[str, Any]:
        """Return the ledger entry for this outcome."""
        return {
            "kind": "outcome",
            "case_id": self.case_id,
            "observed_at": _iso(self.observed_at),
            "description": self.description,
            "outcome_source_ids": list(self.outcome_source_ids),
            "outcome_level": self.outcome_level,
            "outcome_scale": self.outcome_scale,
        }

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> Outcome:
        """Parse a ledger entry.

        Raises:
            ValueError: The entry is not a well-formed outcome entry.
        """
        _fields(
            entry,
            "outcome",
            (
                "case_id",
                "observed_at",
                "description",
                "outcome_source_ids",
                "outcome_level",
                "outcome_scale",
            ),
        )
        return cls(
            case_id=entry["case_id"],
            observed_at=_parse_time("observed_at", entry["observed_at"]),
            description=entry["description"],
            outcome_source_ids=entry["outcome_source_ids"],
            outcome_level=entry["outcome_level"],
            outcome_scale=entry["outcome_scale"],
        )


@dataclass(frozen=True)
class Rating:
    """One rater's blind threat level for a case's outcome.

    Attributes:
        rater_id: Who rated.
        case_id: The case rated.
        level: Threat level 1-5.
        rationale: Why, in the rater's words.
        rated_at: When the rating was submitted (not before the outcome).
    """

    rater_id: str
    case_id: str
    level: int
    rationale: str
    rated_at: datetime

    def __post_init__(self) -> None:
        """Validate the fields."""
        _ident("rater_id", self.rater_id)
        _ident("case_id", self.case_id)
        _level("level", self.level)
        _text("rationale", self.rationale)
        object.__setattr__(self, "rated_at", _aware("rated_at", self.rated_at))

    def to_entry(self) -> dict[str, Any]:
        """Return the ledger entry for this rating."""
        return {
            "kind": "rating",
            "rater_id": self.rater_id,
            "case_id": self.case_id,
            "level": self.level,
            "rationale": self.rationale,
            "rated_at": _iso(self.rated_at),
        }

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> Rating:
        """Parse a ledger entry.

        Raises:
            ValueError: The entry is not a well-formed rating entry.
        """
        _fields(entry, "rating", ("rater_id", "case_id", "level", "rationale", "rated_at"))
        return cls(
            rater_id=entry["rater_id"],
            case_id=entry["case_id"],
            level=entry["level"],
            rationale=entry["rationale"],
            rated_at=_parse_time("rated_at", entry["rated_at"]),
        )


@dataclass(frozen=True)
class AdjudicatorDecision:
    """A third party's resolution of a disagreement between ratings.

    Attributes:
        adjudicator_id: Who decided; never one of the case's raters.
        case_id: The case decided.
        level: The final level, within the range of the ratings it resolves.
        rationale: Why.
        decided_at: When (not before any rating it resolves).
    """

    adjudicator_id: str
    case_id: str
    level: int
    rationale: str
    decided_at: datetime

    def __post_init__(self) -> None:
        """Validate the fields."""
        _ident("adjudicator_id", self.adjudicator_id)
        _ident("case_id", self.case_id)
        _level("level", self.level)
        _text("rationale", self.rationale)
        object.__setattr__(self, "decided_at", _aware("decided_at", self.decided_at))

    def to_entry(self) -> dict[str, Any]:
        """Return the ledger entry for this decision."""
        return {
            "kind": "decision",
            "adjudicator_id": self.adjudicator_id,
            "case_id": self.case_id,
            "level": self.level,
            "rationale": self.rationale,
            "decided_at": _iso(self.decided_at),
        }

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> AdjudicatorDecision:
        """Parse a ledger entry.

        Raises:
            ValueError: The entry is not a well-formed decision entry.
        """
        _fields(
            entry, "decision", ("adjudicator_id", "case_id", "level", "rationale", "decided_at")
        )
        return cls(
            adjudicator_id=entry["adjudicator_id"],
            case_id=entry["case_id"],
            level=entry["level"],
            rationale=entry["rationale"],
            decided_at=_parse_time("decided_at", entry["decided_at"]),
        )


@dataclass(frozen=True)
class Adjudication:
    """The final level of a case and the method that produced it.

    Attributes:
        case_id: The case.
        final_level: The label, 1-5.
        method: One of :data:`METHODS`.
        levels: The ratings, in submission order.
    """

    case_id: str
    final_level: int
    method: str
    levels: tuple[int, ...]


def resolve(levels: Sequence[int], decision: int | None = None) -> tuple[int | None, str]:
    """Apply the adjudication rules to a case's ratings.

    Args:
        levels: The ratings in submission order (two or three).
        decision: An adjudicator's level, if one was recorded.

    Returns:
        ``(final_level, method)`` when resolved, else ``(None, reason)``.
    """
    if len(levels) < 2:
        return None, f"{len(levels)} of the 2 required ratings recorded"
    first, second = levels[0], levels[1]
    if first == second:
        return first, "consensus"
    if len(levels) >= 3:
        if levels[2] in (first, second):
            return levels[2], "majority_of_three"
        if decision is not None:
            return decision, "adjudicator_decision"
        return None, "three ratings without a majority: an adjudicator decision is required"
    gap = abs(first - second)
    if gap == 1:
        if decision is not None:
            return decision, "adjudicator_decision"
        return None, "one-level disagreement awaiting an adjudicator decision or a third rating"
    return None, f"{gap}-level disagreement awaiting a third rating"


def _decision_allowed(levels: Sequence[int]) -> bool:
    if len(levels) == 2:
        return abs(levels[0] - levels[1]) == 1
    return len(levels) == 3 and len(set(levels)) == 3


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def _pct(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 99:
        raise ValueError(f"heldout_pct must be an integer in [1, 99], got {value!r}")
    return value


def split_for(case_id: str, heldout_pct: int = DEFAULT_HELDOUT_PCT) -> str:
    """Return the split a case id falls in under the ledger's hash rule.

    Args:
        case_id: The case identifier (UTF-8 encoded for hashing).
        heldout_pct: Percentage of the hash space held out, 1-99.

    Returns:
        ``"heldout"`` iff ``int(sha3_256(case_id).hexdigest(), 16) % 100 < heldout_pct``,
        else ``"training"``.
    """
    bucket = int(hashlib.sha3_256(case_id.encode("utf-8")).hexdigest(), 16) % 100
    return "heldout" if bucket < _pct(heldout_pct) else "training"


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def _check_pairs(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) != len(b):
        raise ValueError(f"rating sequences differ in length ({len(a)} vs {len(b)})")
    if not a:
        raise ValueError("kappa needs at least one co-rated case")
    for value in (*a, *b):
        _level("rating", value)
    return len(a)


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int]) -> Fraction | None:
    """Exact quadratic-weighted Cohen's kappa (Cohen 1968) of two raters on levels 1-5.

    ``kappa = 1 - n * sum(w * O) / sum(w * r c)`` with ``w[i][j] = (i - j)**2``,
    ``O`` the confusion counts and ``r``, ``c`` its row and column totals, in
    integer arithmetic.

    Args:
        a: Rater A's levels.
        b: Rater B's levels on the same cases, in the same order.

    Returns:
        The kappa as a fraction, or ``None`` when it is undefined (no expected
        disagreement, e.g. every rating the same level).

    Raises:
        ValueError: Lengths differ, there are no cases, or a level is not 1-5.
    """
    n = _check_pairs(a, b)
    size = len(LEVELS)
    counts = [[0] * size for _ in range(size)]
    for x, y in zip(a, b):
        counts[x - 1][y - 1] += 1
    rows = [sum(row) for row in counts]
    cols = [sum(counts[i][j] for i in range(size)) for j in range(size)]
    observed = sum((i - j) ** 2 * counts[i][j] for i in range(size) for j in range(size))
    expected = sum((i - j) ** 2 * rows[i] * cols[j] for i in range(size) for j in range(size))
    if expected == 0:
        return None
    return 1 - Fraction(n * observed, expected)


def bootstrap_kappa_interval(
    a: Sequence[int],
    b: Sequence[int],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, float | None, int]:
    """Seeded percentile-bootstrap interval for the quadratic-weighted kappa.

    Resampling protocol (reproducible for a given NumPy version; NumPy does
    not promise stream stability across versions): ``idx =
    numpy.random.default_rng(seed).integers(0, n, size=(n_boot, n))``; row
    ``k`` of ``idx`` is one resample of the co-rated cases. Resamples whose
    kappa is undefined are dropped and counted; the interval is the
    ``(1 - confidence) / 2`` and ``(1 + confidence) / 2`` quantiles of the rest
    (NumPy's default linear interpolation, Hyndman & Fan 1996 definition 7).

    Args:
        a: Rater A's levels.
        b: Rater B's levels on the same cases.
        n_boot: Number of resamples.
        confidence: Interval coverage in (0, 1).
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        ``(low, high, n_undefined)``; ``low`` and ``high`` are ``None`` when every
        resample is undefined.

    Raises:
        ValueError: Invalid ratings, ``n_boot < 1`` or ``confidence`` outside (0, 1).
    """
    n = _check_pairs(a, b)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    size = len(LEVELS)
    codes = (np.asarray(a, dtype=np.int64) - 1) * size + (np.asarray(b, dtype=np.int64) - 1)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    flat = (np.arange(n_boot, dtype=np.int64)[:, None] * size * size + codes[idx]).ravel()
    counts = (
        np.bincount(flat, minlength=n_boot * size * size)
        .reshape(n_boot, size, size)
        .astype(np.float64)
    )
    grid = np.arange(size, dtype=np.float64)
    weights = (grid[:, None] - grid[None, :]) ** 2
    observed = (counts * weights).sum(axis=(1, 2))
    expected = np.einsum("bi,ij,bj->b", counts.sum(axis=2), weights, counts.sum(axis=1))
    defined = expected > 0
    kappas = 1.0 - n * observed[defined] / expected[defined]
    undefined = n_boot - int(defined.sum())
    if kappas.size == 0:
        return None, None, undefined
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(kappas, [tail, 1.0 - tail])
    return float(low), float(high), undefined


@dataclass(frozen=True)
class AgreementPolicy:
    """Configurable agreement gate (see the module docstring for each default's source).

    Attributes:
        kappa_threshold: Minimum quadratic-weighted kappa for every gated pair.
        min_pair_cases: Co-rated cases a rater pair needs before it is gated.
        n_boot: Bootstrap resamples for the interval.
        confidence: Interval coverage.
        seed: Bootstrap seed.
        gate_on: ``"point"`` (the estimate) or ``"ci_lower"`` (the interval's lower bound).
    """

    kappa_threshold: float = DEFAULT_KAPPA_THRESHOLD
    min_pair_cases: int = DEFAULT_MIN_PAIR_CASES
    n_boot: int = DEFAULT_N_BOOT
    confidence: float = DEFAULT_CONFIDENCE
    seed: int = DEFAULT_SEED
    gate_on: str = "point"

    def __post_init__(self) -> None:
        """Reject settings under which the gate would mean nothing."""
        if not -1.0 <= self.kappa_threshold <= 1.0:
            raise ValueError("kappa_threshold must lie in [-1, 1]")
        if self.min_pair_cases < 2:
            raise ValueError("min_pair_cases must be at least 2")
        if self.n_boot < 1:
            raise ValueError("n_boot must be at least 1")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1")
        if self.gate_on not in GATE_STATISTICS:
            raise ValueError(f"gate_on must be one of {GATE_STATISTICS}")

    @property
    def exact_threshold(self) -> Fraction:
        """The threshold as the decimal it was written as (0.61 is exactly 61/100)."""
        return Fraction(str(self.kappa_threshold))


@dataclass(frozen=True)
class PairAgreement:
    """Agreement of one rater pair on the cases they rated first and second.

    Attributes:
        raters: The two rater ids, sorted.
        n: Co-rated cases.
        kappa: Exact quadratic-weighted kappa, ``None`` when undefined.
        ci_low: Bootstrap interval lower bound.
        ci_high: Bootstrap interval upper bound.
        n_undefined: Resamples whose kappa was undefined.
        gated: Whether the pair has enough cases to be held to the threshold.
        passed: Whether a gated pair reaches the threshold.
    """

    raters: tuple[str, str]
    n: int
    kappa: Fraction | None
    ci_low: float | None
    ci_high: float | None
    n_undefined: int
    gated: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary (kappa as a float and as an exact fraction)."""
        return {
            "raters": list(self.raters),
            "n": self.n,
            "kappa": None if self.kappa is None else float(self.kappa),
            "kappa_exact": None if self.kappa is None else str(self.kappa),
            "ci": [self.ci_low, self.ci_high],
            "n_undefined_resamples": self.n_undefined,
            "gated": self.gated,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class AgreementReport:
    """The agreement gate's verdict and every pair behind it."""

    policy: AgreementPolicy
    pairs: tuple[PairAgreement, ...]
    passed: bool
    reason: str

    def pair(self, raters: tuple[str, str]) -> PairAgreement:
        """Return the agreement measured for two raters, given in either order."""
        key = tuple(sorted(raters))
        return next(p for p in self.pairs if p.raters == key)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary."""
        return {
            "passed": self.passed,
            "reason": self.reason,
            "kappa_threshold": self.policy.kappa_threshold,
            "gate_on": self.policy.gate_on,
            "min_pair_cases": self.policy.min_pair_cases,
            "confidence": self.policy.confidence,
            "n_boot": self.policy.n_boot,
            "seed": self.policy.seed,
            "pairs": [p.to_dict() for p in self.pairs],
        }


def _pair_passes(kappa: Fraction | None, ci_low: float | None, policy: AgreementPolicy) -> bool:
    if policy.gate_on == "ci_lower":
        return ci_low is not None and ci_low >= policy.kappa_threshold
    return kappa is not None and kappa >= policy.exact_threshold


def assess_agreement(
    pairs: Mapping[tuple[str, str], tuple[Sequence[int], Sequence[int]]],
    policy: AgreementPolicy | None = None,
) -> AgreementReport:
    """Measure every rater pair and apply the agreement gate.

    Args:
        pairs: ``(rater_a, rater_b) -> (a's levels, b's levels)`` over the
            cases the two rated first and second.
        policy: The gate's settings; defaults to :class:`AgreementPolicy`.

    Returns:
        The report; it passes iff at least one pair has ``min_pair_cases``
        co-rated cases and every such pair reaches the threshold.
    """
    policy = policy or AgreementPolicy()
    measured: list[PairAgreement] = []
    for raters in sorted(pairs, key=lambda key: tuple(sorted(key))):
        a, b = pairs[raters]
        kappa = quadratic_weighted_kappa(a, b)
        low, high, undefined = bootstrap_kappa_interval(
            a, b, n_boot=policy.n_boot, confidence=policy.confidence, seed=policy.seed
        )
        gated = len(a) >= policy.min_pair_cases
        first, second = sorted(raters)
        measured.append(
            PairAgreement(
                raters=(first, second),
                n=len(a),
                kappa=kappa,
                ci_low=low,
                ci_high=high,
                n_undefined=undefined,
                gated=gated,
                passed=gated and _pair_passes(kappa, low, policy),
            )
        )
    gated_pairs = [p for p in measured if p.gated]
    statistic = "kappa" if policy.gate_on == "point" else "kappa CI lower bound"
    if not gated_pairs:
        return AgreementReport(
            policy,
            tuple(measured),
            False,
            f"no rater pair has co-rated at least {policy.min_pair_cases} cases",
        )
    failing = [p for p in gated_pairs if not p.passed]
    if failing:
        names = ", ".join(f"{p.raters[0]}/{p.raters[1]}" for p in failing)
        return AgreementReport(
            policy,
            tuple(measured),
            False,
            f"{statistic} below {policy.kappa_threshold} or undefined for {names}",
        )
    return AgreementReport(
        policy,
        tuple(measured),
        True,
        f"every gated rater pair reaches {statistic} >= {policy.kappa_threshold}",
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass
class _CaseState:
    case: Case
    split: str
    heldout_pct: int
    entry_digest: str
    outcome: Outcome | None = None
    ratings: list[Rating] = field(default_factory=list)
    decision: AdjudicatorDecision | None = None


class _Protocol:
    """Replays entries in ledger order, refusing any the API would have refused."""

    def __init__(self) -> None:
        self.cases: dict[str, _CaseState] = {}
        self.heldout_pct: int | None = None

    def _known(self, case_id: str, kind: str) -> _CaseState:
        state = self.cases.get(case_id)
        if state is None:
            raise ValueError(f"{kind} for case {case_id!r}, which is not recorded before it")
        return state

    def apply(self, entry: Mapping[str, Any]) -> None:
        """Check ``entry`` against the protocol and record it; raise ValueError on a breach."""
        kind = entry.get("kind")
        if kind == "case":
            self._case(entry)
        elif kind == "outcome":
            self._outcome(Outcome.from_entry(entry))
        elif kind == "rating":
            self._rating(Rating.from_entry(entry))
        elif kind == "decision":
            self._decision(AdjudicatorDecision.from_entry(entry))
        else:
            raise ValueError(f"unknown entry kind {kind!r}")

    def _case(self, entry: Mapping[str, Any]) -> None:
        case, split, pct = Case.from_entry(entry)
        if case.case_id in self.cases:
            raise ValueError(f"case {case.case_id!r} is already recorded")
        if self.heldout_pct is not None and pct != self.heldout_pct:
            raise ValueError(
                f"case {case.case_id!r} records heldout_pct {pct}, but this ledger's split "
                f"rule is heldout_pct {self.heldout_pct}"
            )
        expected = split_for(case.case_id, pct)
        if split != expected:
            raise ValueError(
                f"case {case.case_id!r} records split {split!r}, but its hash rule gives "
                f"{expected!r}"
            )
        self.heldout_pct = pct
        self.cases[case.case_id] = _CaseState(case, split, pct, str(entry["digest"]))

    def _outcome(self, outcome: Outcome) -> None:
        state = self._known(outcome.case_id, "outcome")
        if state.outcome is not None:
            raise ValueError(f"case {outcome.case_id!r} already has an outcome")
        state.outcome = outcome

    def _rating(self, rating: Rating) -> None:
        state = self._known(rating.case_id, "rating")
        if state.outcome is None:
            raise ValueError(
                f"rating of case {rating.case_id!r} before its outcome: raters rate outcomes"
            )
        if state.decision is not None:
            raise ValueError(f"rating of case {rating.case_id!r} after its adjudicator decision")
        if any(r.rater_id == rating.rater_id for r in state.ratings):
            raise ValueError(f"rater {rating.rater_id!r} already rated case {rating.case_id!r}")
        if rating.rated_at < state.outcome.observed_at:
            raise ValueError(
                f"rating of case {rating.case_id!r} is dated before its outcome was observed"
            )
        if len(state.ratings) >= 3:
            raise ValueError(f"case {rating.case_id!r} already has three ratings")
        if len(state.ratings) == 2 and state.ratings[0].level == state.ratings[1].level:
            raise ValueError(
                f"case {rating.case_id!r}: the first two ratings agree, so a third is not taken"
            )
        state.ratings.append(rating)

    def _decision(self, decision: AdjudicatorDecision) -> None:
        state = self._known(decision.case_id, "decision")
        if state.decision is not None:
            raise ValueError(f"case {decision.case_id!r} already has an adjudicator decision")
        levels = [r.level for r in state.ratings]
        if not _decision_allowed(levels):
            raise ValueError(
                f"case {decision.case_id!r}: ratings {levels} admit no adjudicator decision "
                "(agreement or a majority settles them; a gap above one level needs a "
                "third rating first)"
            )
        if decision.adjudicator_id in {r.rater_id for r in state.ratings}:
            raise ValueError(
                f"adjudicator {decision.adjudicator_id!r} rated case {decision.case_id!r}"
            )
        if not min(levels) <= decision.level <= max(levels):
            raise ValueError(
                f"case {decision.case_id!r}: decision {decision.level} lies outside the "
                f"ratings' range {min(levels)}-{max(levels)}"
            )
        if decision.decided_at < max(r.rated_at for r in state.ratings):
            raise ValueError(
                f"case {decision.case_id!r}: decision is dated before a rating it resolves"
            )
        state.decision = decision


def _replay(records: Sequence[Mapping[str, Any]]) -> tuple[_Protocol, list[str]]:
    protocol = _Protocol()
    problems: list[str] = []
    for position, record in enumerate(records):
        try:
            protocol.apply(record["entry"])
        except ValueError as exc:
            problems.append(f"record {position}: {exc}")
    return protocol, problems


@dataclass(frozen=True)
class CaseLabel:
    """One case's derived label, or the reasons it has none.

    Attributes:
        case_id: The case.
        split: ``"heldout"`` or ``"training"``, as recorded.
        adjudication: The final level and method, when the ratings resolve.
        reasons: Why the case is excluded; empty when it is exportable.
        intel_reports: The case payload.
        entry_digest: Digest of the case entry, for provenance.
    """

    case_id: str
    split: str
    adjudication: Adjudication | None
    reasons: tuple[str, ...]
    intel_reports: dict[str, Any]
    entry_digest: str

    @property
    def eligible(self) -> bool:
        """Whether the case is exported."""
        return not self.reasons and self.adjudication is not None

    @property
    def level(self) -> int | None:
        """The adjudicated label, when resolved."""
        return None if self.adjudication is None else self.adjudication.final_level

    @property
    def method(self) -> str | None:
        """How the label was reached, when resolved."""
        return None if self.adjudication is None else self.adjudication.method


def _label(state: _CaseState, agreement: AgreementReport, protocol_ok: bool) -> CaseLabel:
    case, outcome = state.case, state.outcome
    reasons: list[str] = []
    if outcome is None:
        reasons.append("no outcome recorded")
    else:
        shared = {normalize_source_id(s) for s in case.report_source_ids} & {
            normalize_source_id(s) for s in outcome.outcome_source_ids
        }
        if shared:
            reasons.append(f"circular: outcome sources {sorted(shared)} also supplied the reports")
        if not outcome.observed_at > case.latest_report_at:
            reasons.append(
                f"outcome at {_iso(outcome.observed_at)} is not after the latest report at "
                f"{_iso(case.latest_report_at)}"
            )
    decision = state.decision.level if state.decision is not None else None
    levels = tuple(r.level for r in state.ratings)
    level, method = resolve(levels, decision)
    adjudication = None
    if level is None:
        reasons.append(method)
    else:
        adjudication = Adjudication(case.case_id, level, method, levels)
        pair = agreement.pair((state.ratings[0].rater_id, state.ratings[1].rater_id))
        if not pair.gated:
            reasons.append(
                f"rater pair {pair.raters[0]}/{pair.raters[1]} co-rated {pair.n} < "
                f"{agreement.policy.min_pair_cases} cases: their agreement is unmeasured"
            )
    if not agreement.passed:
        reasons.append(f"agreement gate failed: {agreement.reason}")
    if not protocol_ok:
        reasons.append("the ledger's content breaks the protocol")
    return CaseLabel(
        case_id=case.case_id,
        split=state.split,
        adjudication=adjudication,
        reasons=tuple(reasons),
        intel_reports=case.intel_reports,
        entry_digest=state.entry_digest,
    )


def _copy(reports: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = json.loads(_canonical(reports))
    return copied


@dataclass(frozen=True)
class LabelReport:
    """Everything derived from one read of the ledger.

    Attributes:
        ledger_ok: Whether the hash chain (and pinned signatures) verify.
        ledger_reason: :meth:`AdjudicationLedger.verify`'s reason.
        protocol_problems: Records the API would have refused, with their index.
        agreement: The agreement gate, when the chain verifies.
        cases: Every case's label or exclusion, in ledger order.
        head_digest: The last record's digest (the state these labels come from).
        outcome_levels: Each case's documented-scale ``outcome_level``, if any.
        prospective: Case ids assembled before their outcome could occur.
        heldout_pct: The split rule's percentage, fixed by the first case.
    """

    ledger_ok: bool
    ledger_reason: str
    protocol_problems: tuple[str, ...]
    agreement: AgreementReport | None
    cases: tuple[CaseLabel, ...]
    head_digest: str
    outcome_levels: Mapping[str, int] = field(default_factory=dict)
    prospective: frozenset[str] = frozenset()
    heldout_pct: int | None = None

    @property
    def exportable(self) -> bool:
        """Whether labels may leave the ledger at all (chain and content both sound)."""
        return self.ledger_ok and not self.protocol_problems

    def require_exportable(self) -> None:
        """Raise unless the chain verifies and every record keeps to the protocol.

        Raises:
            ExportRefused: Naming the broken record.
        """
        if not self.ledger_ok:
            raise ExportRefused(f"adjudication ledger fails verification: {self.ledger_reason}")
        if self.protocol_problems:
            raise ExportRefused(
                "adjudication ledger content breaks the protocol: " + self.protocol_problems[0]
            )

    def labels(self, split: str) -> list[tuple[dict[str, Any], int]]:
        """Return the eligible ``(intel_reports, level)`` pairs of one split, in ledger order."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return [
            (_copy(c.intel_reports), c.adjudication.final_level)
            for c in self.cases
            if c.split == split and c.eligible and c.adjudication is not None
        ]

    def exclusions(self) -> list[dict[str, Any]]:
        """Return every excluded case with its split and reasons."""
        return [
            {"case_id": c.case_id, "split": c.split, "reasons": list(c.reasons)}
            for c in self.cases
            if not c.eligible
        ]


def _min_heldout_cases() -> int:
    """``MIN_HELDOUT_CASES`` from the fusion module (imported lazily: it pulls in torch)."""
    from omni_mercury_engine.security.intelligence_fusion import MIN_HELDOUT_CASES

    return MIN_HELDOUT_CASES


def default_ledger_path() -> Path:
    """Return ``$MERCURY_ADJUDICATION_PATH`` or ``~/.mercury/adjudication.jsonl``."""
    configured = os.environ.get("MERCURY_ADJUDICATION_PATH")
    return Path(configured) if configured else Path.home() / ".mercury" / "adjudication.jsonl"


class AdjudicationLedger:
    """Append-only, hash-chained (optionally ML-DSA-65-signed) adjudication record.

    Each line is a JSON record ``{schema, index, prev_digest, entry,
    record_digest, signature?}``: ``entry`` carries its own ``digest`` (SHA3-256
    of the canonical entry without it), ``record_digest`` is the SHA3-256 of
    the canonical record without its digest and signature, and ``prev_digest``
    is the previous ``record_digest`` (:data:`GENESIS_DIGEST` first) -- the
    conventions of :class:`~omni_mercury_engine.self_model.SelfModelLedger`.

    The ledger assumes a single writer at a time: concurrent appends produce
    a broken chain, which :meth:`verify` then reports (it never silently
    merges).
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        heldout_pct: int = DEFAULT_HELDOUT_PCT,
        policy: AgreementPolicy | None = None,
        expected_public_key_hex: str | None = None,
    ) -> None:
        """Open (without creating) a ledger.

        Args:
            path: Ledger file; defaults to :func:`default_ledger_path`.
            heldout_pct: Split rule for new cases (an existing ledger's first
                case fixes it; a different value is refused on append).
            policy: Agreement gate settings.
            expected_public_key_hex: Pinned ML-DSA-65 key (hex, either case);
                when set, every record must be signed by it, on read and on append.
        """
        self.path = Path(path) if path is not None else default_ledger_path()
        self.heldout_pct = _pct(heldout_pct)
        self.policy = policy or AgreementPolicy()
        self.expected_public_key_hex = (
            expected_public_key_hex.strip().lower() if expected_public_key_hex is not None else None
        )

    # -- reading ------------------------------------------------------------

    def _lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        return [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def records(self) -> list[dict[str, Any]]:
        """Return every record in order (empty when the ledger does not exist yet)."""
        return [_loads(line) for line in self._lines()]

    def verify(self, expected_public_key_hex: str | None = None) -> tuple[bool, str]:
        """Check the whole chain, and signatures against a pinned key when one is set.

        Fails closed: an unparseable line, a malformed record or signature, or a
        signature that cannot be checked is a broken record, reported with its
        position, never an exception.

        Args:
            expected_public_key_hex: Pinned ML-DSA-65 key (hex, either case,
                surrounding whitespace ignored); defaults to the ledger's pinned key.

        Returns:
            ``(ok, reason)`` naming the first broken record.
        """
        key = (
            expected_public_key_hex
            if expected_public_key_hex is not None
            else self.expected_public_key_hex
        )
        pinned = key.strip().lower() if key is not None else None
        prev = GENESIS_DIGEST
        for position, line in enumerate(self._lines()):
            try:
                record = _loads(line)
            except ValueError:
                return False, f"record {position}: not valid JSON"
            if not isinstance(record, dict) or not isinstance(record.get("entry"), dict):
                return False, f"record {position}: not a ledger record"
            if record.get("schema") != LEDGER_SCHEMA:
                return False, f"record {position}: unsupported schema {record.get('schema')!r}"
            if record.get("index") != position:
                return False, f"record {position}: index {record.get('index')} out of order"
            if record.get("prev_digest") != prev:
                return False, f"record {position}: does not chain to its predecessor"
            unsigned = {k: v for k, v in record.items() if k not in ("record_digest", "signature")}
            digest = _digest(unsigned)
            if digest != record.get("record_digest"):
                return False, f"record {position}: contents altered after recording"
            entry = dict(record["entry"])
            claimed = entry.pop("digest", None)
            if _digest(entry) != claimed:
                return False, f"record {position}: entry digest mismatch"
            if pinned is not None:
                problem = _signature_problem(record.get("signature"), digest, pinned)
                if problem is not None:
                    return False, f"record {position}: {problem}"
            prev = digest
        return True, "valid"

    def head_digest(self) -> str:
        """Return the last record's digest (:data:`GENESIS_DIGEST` when empty)."""
        records = self.records()
        return str(records[-1]["record_digest"]) if records else GENESIS_DIGEST

    def report(self) -> LabelReport:
        """Derive every case's label, exclusions and the agreement gate from the ledger."""
        ok, reason = self.verify()
        if not ok:
            return LabelReport(False, reason, (), None, (), GENESIS_DIGEST)
        records = self.records()
        protocol, problems = _replay(records)
        rated: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
        for state in protocol.cases.values():
            if len(state.ratings) < 2:
                continue
            first, second = state.ratings[0], state.ratings[1]
            ordered = sorted((first, second), key=lambda r: r.rater_id)
            key = (ordered[0].rater_id, ordered[1].rater_id)
            a, b = rated.setdefault(key, ([], []))
            a.append(ordered[0].level)
            b.append(ordered[1].level)
        agreement = assess_agreement(rated, self.policy)
        labels = tuple(_label(state, agreement, not problems) for state in protocol.cases.values())
        levels = {
            cid: s.outcome.outcome_level
            for cid, s in protocol.cases.items()
            if s.outcome is not None and s.outcome.outcome_level is not None
        }
        prospective = frozenset(
            cid
            for cid, s in protocol.cases.items()
            if s.outcome is not None and s.case.created_at < s.outcome.observed_at
        )
        head = str(records[-1]["record_digest"]) if records else GENESIS_DIGEST
        return LabelReport(
            True,
            reason,
            tuple(problems),
            agreement,
            labels,
            head,
            levels,
            prospective,
            protocol.heldout_pct,
        )

    def heldout_cases(self) -> list[tuple[dict[str, Any], int]]:
        """Return the eligible held-out ``(intel_reports, level)`` pairs.

        Raises:
            ExportRefused: The chain does not verify or a record breaks the protocol.
        """
        report = self.report()
        report.require_exportable()
        return report.labels("heldout")

    def training_cases(self) -> list[tuple[dict[str, Any], int]]:
        """Return the eligible training ``(intel_reports, level)`` pairs.

        Raises:
            ExportRefused: The chain does not verify or a record breaks the protocol.
        """
        report = self.report()
        report.require_exportable()
        return report.labels("training")

    def exclusions(self) -> list[dict[str, Any]]:
        """Return every excluded case with its split and the reasons it is excluded."""
        return self.report().exclusions()

    def export_heldout_for_merit_gate(self) -> list[tuple[dict[str, Any], int]]:
        """Return held-out cases for ``IntelligenceFusionEngine.load_trained_weights``.

        Raises:
            ExportRefused: The ledger is unsound, the agreement gate fails, or
                fewer than ``MIN_HELDOUT_CASES`` held-out cases are eligible.
        """
        return self._export("heldout", self.report())

    def _export(self, split: str, report: LabelReport) -> list[tuple[dict[str, Any], int]]:
        report.require_exportable()
        if report.agreement is None or not report.agreement.passed:
            reason = report.agreement.reason if report.agreement is not None else "unmeasured"
            raise ExportRefused(f"agreement gate failed: {reason}")
        cases = report.labels(split)
        if split == "heldout":
            minimum = _min_heldout_cases()
            if len(cases) < minimum:
                raise ExportRefused(
                    f"{len(cases)} eligible held-out cases; the merit gate needs at least "
                    f"{minimum}"
                )
        return cases

    def export(self, split: str, out: str | Path) -> dict[str, Any]:
        """Write one split's labels to a JSON file (see :func:`load_export`).

        Args:
            split: ``"heldout"`` or ``"training"``.
            out: Destination file.

        Returns:
            The document written.

        Raises:
            ExportRefused: As :meth:`export_heldout_for_merit_gate` (the case
                minimum applies to the held-out split only).
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        report = self.report()
        self._export(split, report)
        agreement = report.agreement
        document = {
            "schema": EXPORT_SCHEMA,
            "split": split,
            "ledger": str(self.path),
            "ledger_head_digest": report.head_digest,
            "agreement": agreement.to_dict() if agreement is not None else None,
            "cases": [
                {
                    "case_id": c.case_id,
                    "intel_reports": _copy(c.intel_reports),
                    "label": c.level,
                    "method": c.method,
                    "case_entry_digest": c.entry_digest,
                }
                for c in report.cases
                if c.split == split and c.eligible
            ],
        }
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return document

    def stats(self) -> dict[str, Any]:
        """Summarise the ledger: validity, agreement, counts per split and level, exclusions."""
        report = self.report()
        summary: dict[str, Any] = {
            "ledger": str(self.path),
            "valid": report.ledger_ok,
            "reason": report.ledger_reason,
            "records": len(self._lines()),
            "protocol_problems": list(report.protocol_problems),
            "head_digest": report.head_digest,
        }
        if not report.ledger_ok or report.agreement is None:
            return summary
        splits: dict[str, Any] = {}
        for split in SPLITS:
            members = [c for c in report.cases if c.split == split]
            eligible = [c for c in members if c.eligible]
            per_level = Counter(c.level for c in eligible)
            splits[split] = {
                "cases": len(members),
                "eligible": len(eligible),
                "excluded": len(members) - len(eligible),
                "eligible_per_level": {str(level): per_level[level] for level in LEVELS},
            }
        eligible_all = [c for c in report.cases if c.eligible]
        methods = Counter(c.method for c in eligible_all)
        scaled = [
            (c.adjudication.final_level, report.outcome_levels[c.case_id])
            for c in eligible_all
            if c.adjudication is not None and c.case_id in report.outcome_levels
        ]
        scale_kappa = (
            quadratic_weighted_kappa([a for a, _ in scaled], [b for _, b in scaled])
            if scaled
            else None
        )
        minimum = _min_heldout_cases()
        summary.update(
            {
                "split_rule": {
                    "rule": "int(sha3_256(case_id).hexdigest(), 16) % 100 < heldout_pct",
                    "heldout_pct": report.heldout_pct,
                },
                "agreement": report.agreement.to_dict(),
                "splits": splits,
                "methods": {method: methods[method] for method in METHODS},
                "prospective_eligible": sum(c.case_id in report.prospective for c in eligible_all),
                "outcome_scale_agreement": {
                    "n": len(scaled),
                    "kappa": None if scale_kappa is None else float(scale_kappa),
                },
                "min_heldout_cases": minimum,
                "heldout_ready": report.exportable
                and report.agreement.passed
                and splits["heldout"]["eligible"] >= minimum,
                "exclusions": report.exclusions(),
            }
        )
        return summary

    def rater_view(self, rater_id: str, case_id: str) -> dict[str, Any]:
        """What a rater may see of a case: its outcome, and others' ratings only after their own.

        The view never includes the case's ``intel_reports`` or the outcome's
        documented-scale level, and hides every other rating -- and how many
        there are -- until ``rater_id`` has submitted a rating for the case.

        Args:
            rater_id: The rater asking.
            case_id: The case.

        Returns:
            ``{case_id, outcome, own_rating, other_ratings}``; ``outcome`` is
            ``None`` before one is recorded and ``other_ratings`` is ``None``
            until the rater has rated.

        Raises:
            ExportRefused: The ledger does not verify or breaks the protocol.
            KeyError: The case is not recorded.
        """
        protocol = self._sound_protocol()
        state = protocol.cases[case_id]
        own = next((r for r in state.ratings if r.rater_id == rater_id), None)
        outcome = state.outcome
        return {
            "case_id": case_id,
            "outcome": (
                None
                if outcome is None
                else {"observed_at": _iso(outcome.observed_at), "description": outcome.description}
            ),
            "own_rating": None if own is None else _public_rating(own),
            "other_ratings": (
                None
                if own is None
                else [_public_rating(r) for r in state.ratings if r.rater_id != rater_id]
            ),
        }

    def pending(self) -> dict[str, list[str]]:
        """Return case ids awaiting an outcome, ratings, a third rating or a decision."""
        protocol = self._sound_protocol()
        queues: dict[str, list[str]] = {
            "awaiting_outcome": [],
            "awaiting_ratings": [],
            "awaiting_third_rating": [],
            "awaiting_decision": [],
        }
        for case_id, state in protocol.cases.items():
            levels = [r.level for r in state.ratings]
            decision = state.decision.level if state.decision is not None else None
            if state.outcome is None:
                queues["awaiting_outcome"].append(case_id)
            elif len(levels) < 2:
                queues["awaiting_ratings"].append(case_id)
            elif resolve(levels, decision)[0] is not None:
                continue
            else:
                if len(levels) == 2:
                    queues["awaiting_third_rating"].append(case_id)
                if _decision_allowed(levels):
                    queues["awaiting_decision"].append(case_id)
        return queues

    def _sound_protocol(self) -> _Protocol:
        ok, reason = self.verify()
        if not ok:
            raise ExportRefused(f"adjudication ledger fails verification: {reason}")
        protocol, problems = _replay(self.records())
        if problems:
            raise ExportRefused("adjudication ledger content breaks the protocol: " + problems[0])
        return protocol

    # -- writing ------------------------------------------------------------

    def add_case(
        self, case: Case, signing_keypair: tuple[bytes, bytes] | None = None
    ) -> dict[str, Any]:
        """Record a case with the split its id hashes to under the ledger's rule.

        Returns:
            The record written.

        Raises:
            ValueError: The ledger is broken, the case id is taken, or the
                ledger's split rule differs from ``heldout_pct``.
        """
        entry = case.to_entry(split_for(case.case_id, self.heldout_pct), self.heldout_pct)
        return self._append(entry, signing_keypair)

    def add_outcome(
        self, outcome: Outcome, signing_keypair: tuple[bytes, bytes] | None = None
    ) -> dict[str, Any]:
        """Record a case's outcome.

        An outcome whose sources overlap the reports', or that is not after the
        latest report, is recorded (it is a fact) but its case is never exported.

        Returns:
            The record written.

        Raises:
            ValueError: The ledger is broken, the case is unknown or already has one.
        """
        return self._append(outcome.to_entry(), signing_keypair)

    def add_rating(
        self, rating: Rating, signing_keypair: tuple[bytes, bytes] | None = None
    ) -> dict[str, Any]:
        """Record a blind rating.

        Returns:
            The record written.

        Raises:
            ValueError: The ledger is broken, the case has no outcome yet, the
                rater already rated it, it is already resolved by agreement or
                decision, or the rating predates the outcome.
        """
        return self._append(rating.to_entry(), signing_keypair)

    def add_decision(
        self, decision: AdjudicatorDecision, signing_keypair: tuple[bytes, bytes] | None = None
    ) -> dict[str, Any]:
        """Record an adjudicator decision on a disagreement.

        Returns:
            The record written.

        Raises:
            ValueError: The ledger is broken, or the ratings admit no decision,
                the adjudicator rated the case, the level lies outside the
                ratings' range, or the decision predates a rating.
        """
        return self._append(decision.to_entry(), signing_keypair)

    def extend(
        self,
        items: Sequence[Case | Outcome | Rating | AdjudicatorDecision],
        signing_keypair: tuple[bytes, bytes] | None = None,
    ) -> list[dict[str, Any]]:
        """Record several items in order, all or nothing (one verification for the batch).

        Args:
            items: Cases, outcomes, ratings and decisions, in the order they
                are to be recorded.
            signing_keypair: Optional ML-DSA-65 ``(public_key, secret_key)``.

        Returns:
            The records written.

        Raises:
            ValueError: As the ``add_*`` methods; when any item is refused,
                nothing is written.
        """
        entries = [
            (
                item.to_entry(split_for(item.case_id, self.heldout_pct), self.heldout_pct)
                if isinstance(item, Case)
                else item.to_entry()
            )
            for item in items
        ]
        return self._append_entries(entries, signing_keypair)

    def _append(
        self, entry: dict[str, Any], signing_keypair: tuple[bytes, bytes] | None
    ) -> dict[str, Any]:
        return self._append_entries([entry], signing_keypair)[0]

    def _append_entries(
        self, entries: Sequence[dict[str, Any]], signing_keypair: tuple[bytes, bytes] | None
    ) -> list[dict[str, Any]]:
        ok, reason = self.verify()
        if not ok:
            raise ValueError(f"refusing to extend a broken adjudication ledger: {reason}")
        records = self.records()
        protocol, problems = _replay(records)
        if problems:
            raise ValueError(
                f"refusing to extend an adjudication ledger whose content breaks the protocol: "
                f"{problems[0]}"
            )
        if self.expected_public_key_hex is not None and (
            signing_keypair is None
            or signing_keypair[0].hex().lower() != self.expected_public_key_hex
        ):
            raise ValueError("this ledger is pinned to a key: records must be signed by it")
        signer = None
        if signing_keypair is not None:
            from omni_mercury_engine.security.crypto_api import MLDSAProvider

            signer = MLDSAProvider()
        prev = records[-1]["record_digest"] if records else GENESIS_DIGEST
        written: list[dict[str, Any]] = []
        for offset, entry in enumerate(entries):
            sealed = {**entry, "digest": _digest(entry)}
            protocol.apply(sealed)
            record: dict[str, Any] = {
                "schema": LEDGER_SCHEMA,
                "index": len(records) + offset,
                "prev_digest": prev,
                "entry": sealed,
            }
            record["record_digest"] = _digest(record)
            if signer is not None and signing_keypair is not None:
                public_key, secret_key = signing_keypair
                record["signature"] = {
                    "algorithm": "ML-DSA-65",
                    "public_key_hex": public_key.hex(),
                    "value_hex": signer.sign(record["record_digest"].encode(), secret_key).hex(),
                }
            prev = record["record_digest"]
            written.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in written:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return written


def _public_rating(rating: Rating) -> dict[str, Any]:
    return {
        "rater_id": rating.rater_id,
        "level": rating.level,
        "rationale": rating.rationale,
        "rated_at": _iso(rating.rated_at),
    }


def _signature_problem(signature: Any, digest: str, pinned: str) -> str | None:
    """Return why a record's signature fails against the pinned key, or ``None``."""
    if (
        not isinstance(signature, dict)
        or str(signature.get("public_key_hex", "")).strip().lower() != pinned
    ):
        return "not signed by the pinned key"
    try:
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        valid = MLDSAProvider().verify(
            digest.encode(), bytes.fromhex(signature["value_hex"]), bytes.fromhex(pinned)
        )
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return f"signature could not be checked: {exc}"
    if not valid:
        return "signature does not verify"
    return None


# ---------------------------------------------------------------------------
# Export files
# ---------------------------------------------------------------------------


def load_export(path: str | Path) -> list[tuple[dict[str, Any], int]]:
    """Read an export written by :meth:`AdjudicationLedger.export`.

    Args:
        path: The export file.

    Returns:
        ``(intel_reports, label)`` pairs, ready for ``load_trained_weights``.

    Raises:
        ValueError: The file is not an adjudication export, or a case is malformed.
    """
    document = _loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"{path} is not an adjudication export ({EXPORT_SCHEMA})")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path} has no case list")
    pairs: list[tuple[dict[str, Any], int]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"{path}: a case is not an object")
        pairs.append((_reports(case.get("intel_reports")), _level("label", case.get("label"))))
    return pairs


# ---------------------------------------------------------------------------
# UCDP GED outcomes
# ---------------------------------------------------------------------------


def ucdp_fatality_level(best: int) -> int:
    """Map a UCDP GED event's best fatality estimate to a threat level.

    Bins (a Mercury convention, not a UCDP scale): 0 -> 1, 1-4 -> 2, 5-24 -> 3,
    25-99 -> 4, 100 or more -> 5 -- roughly a factor of five per level. The
    25 cut-point echoes UCDP's 25 battle-related deaths per calendar year for
    an active conflict, but applied to a single event it carries no UCDP
    meaning.

    Args:
        best: The ``best`` field (best estimate of total fatalities), >= 0.

    Returns:
        The threat level 1-5.

    Raises:
        ValueError: ``best`` is not a non-negative integer.
    """
    if isinstance(best, bool) or not isinstance(best, int) or best < 0:
        raise ValueError(f"best must be a non-negative integer fatality count, got {best!r}")
    for ceiling, level in _UCDP_FATALITY_BINS:
        if best <= ceiling:
            return level
    return 5


def _ucdp_field(row: Mapping[str, str], name: str) -> str:
    value = row.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f"UCDP GED row has no {name!r}")
    return str(value).strip()


def _ucdp_date(row: Mapping[str, str], name: str) -> date:
    raw = _ucdp_field(row, name)
    match = _UCDP_DATE.fullmatch(raw)
    if match is None:
        raise ValueError(f"UCDP GED {name} {raw!r} is not YYYY-MM-DD (optionally at 00:00:00)")
    return date.fromisoformat(match.group(1))


def _ucdp_count(row: Mapping[str, str], name: str) -> int:
    raw = _ucdp_field(row, name)
    if _UCDP_COUNT.fullmatch(raw) is None:
        raise ValueError(f"UCDP GED {name} {raw!r} is not a non-negative integer")
    return int(raw)


def outcome_from_ucdp_row(row: Mapping[str, str], *, case_id: str | None = None) -> Outcome:
    """Build an :class:`Outcome` from one row of the UCDP GED event CSV.

    Field names and meanings were checked against the UCDP GED Codebook
    version 26.1 (Hogbladh 2026) and the header of ``GEDEvent_v26_1.csv``
    (downloaded from https://ucdp.uu.se/downloads/ on 2026-10-16):

    * ``id`` -- unique numeric event id, the one to rely on across versions
      (``relid`` can change); the default ``case_id`` is ``"ucdp-ged-<id>"``,
      so the case's split is fixed by UCDP's id, not chosen by whoever builds
      the case.
    * ``date_start`` / ``date_end`` -- earliest and last possible date of the
      event. Dates are local calendar days (codebook sec. 5.6) and the CSV
      writes them as ``YYYY-MM-DD 00:00:00.000``. ``observed_at`` is the
      earliest instant ``date_start`` begins anywhere (00:00 at UTC+14), so
      a report written on the event's first local day can never count as
      preceding it.
    * ``best`` (with ``low``/``high``) -- best, lowest and highest estimate of
      total fatalities; ``best`` sets ``outcome_level`` through
      :func:`ucdp_fatality_level`.
    * ``source_office`` -- the publishing organisations, ``;``-separated.
      They become ``outcome_source_ids`` (with :data:`UCDP_SOURCE_ID`) so a
      case whose reports came from the same publisher is circular. Older
      events leave it empty (97,496 of the 417,968 rows of GED 26.1); such a
      row is refused, since non-circularity cannot be checked.

    UCDP datasets are free of charge under CC BY 4.0 (stated on the downloads
    page); cite Sundberg & Melander (2013) and the codebook version used.

    Args:
        row: A ``csv.DictReader`` row of the GED event file.
        case_id: The case this outcome resolves; defaults to ``"ucdp-ged-<id>"``.

    Returns:
        The outcome, with ``outcome_scale`` :data:`UCDP_FATALITY_SCALE`.

    Raises:
        ValueError: A required field is missing or malformed, ``date_start`` is
            after ``date_end``, or no publisher is named.
    """
    event_id = _ucdp_count(row, "id")
    start = _ucdp_date(row, "date_start")
    end = _ucdp_date(row, "date_end")
    if start > end:
        raise ValueError(f"UCDP GED event {event_id}: date_start {start} is after date_end {end}")
    best = _ucdp_count(row, "best")
    offices = [o.strip() for o in str(row.get("source_office") or "").split(";") if o.strip()]
    if not offices:
        raise ValueError(
            f"UCDP GED event {event_id} names no source_office: the publishers behind the "
            "outcome are unknown, so non-circularity cannot be checked"
        )
    extras = [
        f"{label} {str(row.get(name)).strip()}"
        for name, label in (("low", "low"), ("high", "high"))
        if row.get(name) is not None and str(row.get(name)).strip()
    ]
    context = [
        str(row.get(name)).strip()
        for name in ("conflict_name", "country", "where_description")
        if row.get(name) is not None and str(row.get(name)).strip()
    ]
    description = (
        f"UCDP GED event {event_id}"
        + (f" ({'; '.join(context)})" if context else "")
        + f": best estimate {best} fatalities"
        + (f" ({', '.join(extras)})" if extras else "")
        + f", between {start.isoformat()} and {end.isoformat()}"
    )
    return Outcome(
        case_id=case_id if case_id is not None else f"ucdp-ged-{event_id}",
        observed_at=datetime.combine(start, time(0), tzinfo=_EARLIEST_ZONE).astimezone(UTC),
        description=description,
        outcome_source_ids=(UCDP_SOURCE_ID, *dict.fromkeys(offices)),
        outcome_level=ucdp_fatality_level(best),
        outcome_scale=UCDP_FATALITY_SCALE,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``verify`` the chain, print ``stats``, or ``export`` one split's labels."""
    import argparse

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", type=Path, default=None, help="ledger path")
    common.add_argument("--pinned-key", default=None, help="pinned ML-DSA-65 public key (hex)")
    common.add_argument("--kappa-threshold", type=float, default=DEFAULT_KAPPA_THRESHOLD)
    common.add_argument("--min-pair-cases", type=int, default=DEFAULT_MIN_PAIR_CASES)
    common.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    common.add_argument("--seed", type=int, default=DEFAULT_SEED)
    common.add_argument("--gate-on", choices=GATE_STATISTICS, default="point")
    parser = argparse.ArgumentParser(description="Mercury's adjudicated intelligence-case labels")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", parents=[common], help="verify the ledger chain")
    commands.add_parser("stats", parents=[common], help="agreement, counts and exclusions")
    export = commands.add_parser("export", parents=[common], help="export one split's labels")
    export.add_argument("--split", choices=SPLITS, required=True)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ledger = AdjudicationLedger(
        args.ledger,
        policy=AgreementPolicy(
            kappa_threshold=args.kappa_threshold,
            min_pair_cases=args.min_pair_cases,
            n_boot=args.n_boot,
            seed=args.seed,
            gate_on=args.gate_on,
        ),
        expected_public_key_hex=args.pinned_key,
    )
    if args.command == "verify":
        ok, reason = ledger.verify()
        print(json.dumps({"ledger": str(ledger.path), "valid": ok, "reason": reason}))
        return 0 if ok else 1
    if args.command == "stats":
        summary = ledger.stats()
        print(json.dumps(summary, indent=2))
        return 0 if summary["valid"] else 1
    try:
        document = ledger.export(args.split, args.out)
    except ExportRefused as exc:
        print(json.dumps({"exported": False, "split": args.split, "reason": str(exc)}))
        return 1
    print(
        json.dumps(
            {
                "exported": True,
                "split": args.split,
                "out": str(args.out),
                "cases": len(document["cases"]),
                "ledger_head_digest": document["ledger_head_digest"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_HELDOUT_PCT",
    "DEFAULT_KAPPA_THRESHOLD",
    "DEFAULT_MIN_PAIR_CASES",
    "DEFAULT_N_BOOT",
    "DEFAULT_SEED",
    "EXPORT_SCHEMA",
    "GENESIS_DIGEST",
    "LEDGER_SCHEMA",
    "LEVELS",
    "METHODS",
    "SPLITS",
    "UCDP_FATALITY_SCALE",
    "UCDP_SOURCE_ID",
    "Adjudication",
    "AdjudicationLedger",
    "AdjudicatorDecision",
    "AgreementPolicy",
    "AgreementReport",
    "Case",
    "CaseLabel",
    "ExportRefused",
    "LabelReport",
    "Outcome",
    "PairAgreement",
    "Rating",
    "assess_agreement",
    "bootstrap_kappa_interval",
    "default_ledger_path",
    "load_export",
    "normalize_source_id",
    "outcome_from_ucdp_row",
    "quadratic_weighted_kappa",
    "resolve",
    "split_for",
    "ucdp_fatality_level",
]
