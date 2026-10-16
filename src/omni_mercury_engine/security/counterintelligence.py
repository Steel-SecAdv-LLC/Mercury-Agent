# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Defensive counterintelligence screens for the all-source fusion pipeline.

The fusion engine (``security/intelligence_fusion.py``) weights each
discipline's report by a fixed reliability prior and scales its confidence by
how many disciplines report. Both steps assume the reports are honest and
independent. This module tests those assumptions on the reports it is given
and records, with evidence, where they fail. It never synthesises a report, a
score or a threat level. When the inputs do not support a test, it abstains
and records why.

Components. Each cites its method, and every default is declared and
configurable on :class:`CounterintelligenceConfig`.

1. **Source reliability.** A Beta-Bernoulli posterior per source. The prior
   mean comes from the source's NATO Admiralty Code reliability grade
   (AJP-2.1; STANAG 2511): A completely reliable, B usually reliable, C fairly
   reliable, D not usually reliable, E unreliable, F reliability cannot be
   judged. The doctrine gives words, not numbers, so the grade-to-mean table
   :data:`ADMIRALTY_RELIABILITY_PRIOR_MEAN` is this module's declared choice:
   each descriptor sits at the midpoint of an ICD 203 (ODNI, 2015) likelihood
   band. F and a missing grade give the uninformative Beta(1, 1). Adjudicated
   outcomes update the posterior. Optional exponential forgetting follows the
   Beta reputation system (Jøsang & Ismail, 2002). The fusion weight
   logit(mean) is the optimal weight for independent binary experts (Nitzan &
   Paroush, 1982). It is clipped to ±ln 99, the top of the ICD 203 "almost
   certain" band.
2. **Independence and circular reporting.** A provenance graph is built from
   declared citations (``derived_from``, ``citations``, or the sources of an
   ``intel/provenance.py`` ``Provenance`` record) and from near-duplicate text.
   Near duplicates are found with word-shingle MinHash (Broder, 1997) under a
   seeded Carter-Wegman (1979) universal hash family. Cycles, corroboration
   loops and shared upstream origins are flagged. Reports are grouped into
   independence clusters, and corroboration counts clusters, not reports.
3. **Coordinated reporting.** The scan statistic is the largest number of
   distinct sources reporting one claim inside a sliding window (Glaz, Naus &
   Wallenstein, 2001). Under a Poisson null, its p-value is bounded above by the
   expected number of reports that open such a window. The Mecke equation
   (Last & Penrose, 2017, ch. 4) gives that expectation exactly from Poisson
   tail probabilities. Benjamini-Hochberg (1995) controls the false discovery
   rate across claims.
4. **Fabrication screens on numeric content.** The Benford first-digit test
   uses chi-square and Nigrini's MAD with his conformity bands (Nigrini, 2012).
   Terminal-digit uniformity uses chi-square (Preece, 1981; Mosimann, Wiseman &
   Edelman, 1995). Excess rounding to 0 or 5 uses an exact binomial test. Each
   screen runs only when its applicability conditions hold.
5. **Impossible travel.** Consecutive fixes of one entity or source are
   flagged when the implied great-circle speed exceeds a declared maximum
   after both location-uncertainty radii are subtracted.
6. **Training-data poisoning.** Spectral signatures (Tran, Li & Madry,
   NeurIPS 2018) give a stand-alone screen for learned representations.

:class:`CounterintelligenceAssessor` runs components 1-5 over a fusion-engine
report mapping. It returns a frozen :class:`CounterintelligenceAssessment`
whose ``fusion_weight_multipliers`` the engine can apply to its reliability
weights.

Limitations. Read these before acting on an output.

* None of these screens is validated on operational intelligence reporting.
  They are statistical and structural tests whose assumptions are stated next
  to each one. A flag is a lead for an analyst, not a finding of deception.
* The grade-to-prior table, the penalty factors, and the window, threshold and
  shingle defaults are declared choices, not calibrated estimates.
* Provenance analysis sees only the citations and texts it is given. An
  undeclared, paraphrased copy evades it.
* The scan p-value is conservative (an upper bound) and assumes a known
  baseline rate. A mis-specified rate invalidates it.
* Benford, terminal-digit and rounding departures have innocent causes:
  bounded quantities, estimation, and unit conversion. Excess rounding in
  particular is what honest estimates look like.

Tests: ``tests/security/test_counterintelligence.py``.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import itertools
import math
import numbers
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from omni_mercury_engine.evaluation.event_coincidence import benjamini_hochberg
from omni_mercury_engine.utils.geo import haversine_km

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

# --------------------------------------------------------------------------- #
# Declared tables and constants.
# --------------------------------------------------------------------------- #

#: Admiralty Code source-reliability grades (AJP-2.1; STANAG 2511).
ADMIRALTY_GRADES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

#: The doctrinal descriptor of each source-reliability grade.
ADMIRALTY_RELIABILITY_DESCRIPTORS: Mapping[str, str] = MappingProxyType(
    {
        "A": "completely reliable",
        "B": "usually reliable",
        "C": "fairly reliable",
        "D": "not usually reliable",
        "E": "unreliable",
        "F": "reliability cannot be judged",
    }
)

#: Declared grade-to-prior-mean table. This is a choice made by this module, not
#: doctrine: AJP-2.1 and STANAG 2511 attach no numbers to the grades, and Irwin &
#: Mandel (2019, Intelligence and National Security 34(4)) show that analysts read
#: the descriptors inconsistently. Each descriptor is placed at the midpoint of the
#: ICD 203 (ODNI, 2015) likelihood band it most plainly matches: A "almost
#: certain" (95-99%), B "very likely" (80-95%), C "likely" (55-80%), D "unlikely"
#: (20-45%), E "very unlikely" (5-20%). F has no mean: it is the uninformative
#: Beta(1, 1). Override it with ``CounterintelligenceConfig.grade_prior_mean``.
ADMIRALTY_RELIABILITY_PRIOR_MEAN: Mapping[str, float | None] = MappingProxyType(
    {"A": 0.97, "B": 0.875, "C": 0.675, "D": 0.325, "E": 0.125, "F": None}
)

#: Admiralty Code information-credibility grades (AJP-2.1). They are a separate
#: axis from source reliability and are never folded into a source's posterior.
INFORMATION_CREDIBILITY_DESCRIPTORS: Mapping[int, str] = MappingProxyType(
    {
        1: "confirmed by other sources",
        2: "probably true",
        3: "possibly true",
        4: "doubtful",
        5: "improbable",
        6: "truth cannot be judged",
    }
)

#: Benford's first-digit law, P(d) = log10(1 + 1/d) for d = 1..9 (Benford, 1938).
BENFORD_FIRST_DIGIT_PROBABILITIES: tuple[float, ...] = tuple(
    math.log10(1.0 + 1.0 / d) for d in range(1, 10)
)

#: Nigrini (2012) first-digit MAD conformity ranges. Each upper bound is
#: inclusive. A MAD above the last bound is ``nonconformity``.
NIGRINI_FIRST_DIGIT_MAD_BOUNDS: tuple[tuple[float, str], ...] = (
    (0.006, "close conformity"),
    (0.012, "acceptable conformity"),
    (0.015, "marginally acceptable conformity"),
)
NIGRINI_NONCONFORMITY = "nonconformity"

#: Smallest prime above 2**32. With 32-bit base hashes ``x`` and coefficients
#: ``a, b < 2**32``, ``a*x + b < 2**64``, so the universal hash
#: ``(a*x + b) mod p`` (Carter & Wegman, 1979) is computed in uint64 without
#: overflow. For ``a != 0`` it is a bijection on ``[0, p)``.
MINHASH_PRIME = 4_294_967_311

#: Declared default speed bound for impossible travel, in km/h. A long-haul jet
#: cruising at Mach 0.85 in the ISO 2533 standard atmosphere above the tropopause
#: (216.65 K, speed of sound 1,062 km/h) has a true airspeed of about 903 km/h.
#: About 300 km/h is added for a strong jet-stream tailwind, so that legitimate
#: airline travel is not flagged. Use a lower bound (for example 150 km/h) for
#: entities restricted to the surface.
DEFAULT_MAX_SPEED_KMH = 1200.0

#: ±ln(99) = logit(0.99): the outer edge of ICD 203's "almost certain" band.
DEFAULT_LOG_ODDS_CLIP = math.log(99.0)

_ROUNDED_DIGITS = (0, 5)
_ROUNDED_SHARE = 0.2  # P(terminal digit in {0, 5}) under uniform terminal digits
_DEFAULT_GROUP = "__assessment__"
_SECONDS_PER_HOUR = 3600.0
_PRECISION_TOLERANCE = 1e-6  # in units of the last recorded digit
_EXACT_INTEGER_LIMIT = 2.0**53  # float64 integers are exact below this
_SVD_RELATIVE_TOLERANCE = 1e-9
_TOKEN_RE = re.compile(r"\w+")


class FlagKind(StrEnum):
    """Kinds of counterintelligence finding."""

    CIRCULAR_REPORTING = "circular_reporting"
    SHARED_ORIGIN = "shared_origin"
    COORDINATED_REPORTING = "coordinated_reporting"
    FABRICATION = "fabrication"
    IMPOSSIBLE_TRAVEL = "impossible_travel"
    CONFLICTING_CLAIMS = "conflicting_claims"


_FLAG_ORDER = {kind: index for index, kind in enumerate(FlagKind)}


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _as_float(value: object) -> float | None:
    """Return a finite float for a real number (not a bool), else ``None``."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _as_id(value: object) -> str | None:
    """Return an identifier string for a str or integer value, else ``None``."""
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, numbers.Integral):
        return str(int(value))
    return None


def parse_timestamp(value: object) -> float | None:
    """Convert a report timestamp to POSIX seconds.

    Accepts epoch seconds (int or float) or an ISO-8601 string or ``datetime``
    with a UTC offset. A timezone-naive time is refused (``None``) rather than
    guessed, because a wrong zone can fabricate an impossible-travel or burst
    finding.

    Args:
        value: The raw ``reported_at`` value.

    Returns:
        POSIX seconds, or ``None`` when the value is missing, unparsable or naive.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return _as_float(value)
    if moment.tzinfo is None or moment.utcoffset() is None:
        return None
    return moment.timestamp()


# --------------------------------------------------------------------------- #
# Configuration.
# --------------------------------------------------------------------------- #


def _validate_grade_table(table: Mapping[str, float | None]) -> None:
    """Check a grade-to-prior-mean table covers A-F with F uninformative."""
    _require(set(table) == set(ADMIRALTY_GRADES), "grade table must have exactly the keys A-F")
    _require(table["F"] is None, "grade F (cannot be judged) must map to None: Beta(1, 1)")
    for grade in ADMIRALTY_GRADES[:-1]:
        mean = table[grade]
        _require(
            mean is not None and 0.0 < mean < 1.0,
            f"grade {grade} prior mean must lie strictly between 0 and 1, got {mean!r}",
        )


@dataclass(frozen=True)
class CounterintelligenceConfig:
    """Declared, configurable parameters. None of the defaults is calibrated.

    Attributes:
        grade_prior_mean: Admiralty grade to Beta prior mean. See
            :data:`ADMIRALTY_RELIABILITY_PRIOR_MEAN` for the rationale.
        prior_strength: Pseudo-count ``alpha + beta`` of a graded prior.
            Default 2 equals the weight of the uniform Beta(1, 1) used for F,
            so a grade moves the prior's location but not its weight, and two
            adjudications move the posterior as far as the grade did.
        forgetting_factor: Per-outcome discount ``lambda`` in (0, 1] applied to
            earlier evidence (Jøsang & Ismail, 2002). Default 1 keeps all
            evidence at full weight.
        credible_level: Mass of the equal-tailed credible interval. The
            default 0.95 is the conventional level.
        log_odds_clip: Bound on ``|logit(mean)|``. The default is ln 99, the
            edge of ICD 203's "almost certain" band.
        discipline_prior_strength: Pseudo-count given to the engine's
            per-discipline prior when it is pooled with source evidence.
            Default 2, for the same reason as ``prior_strength``.
        shingle_size: Words per shingle. The default 3 is a declared choice for
            short reports of tens of words, where Broder's 10-word shingles
            would leave too few shingles. One changed word alters at most 3.
        minhash_permutations: MinHash signature length ``k``. The estimator's
            standard error is ``sqrt(J(1-J)/k)``, at most 0.031 at ``k = 256``.
        near_duplicate_threshold: Estimated Jaccard resemblance at or above
            which two texts are near duplicates. The default 0.5 means that a
            majority of the shingle union is shared. It is not tuned on
            intelligence text.
        seed: Seed for the MinHash hash family.
        coordination_window_hours: Scan window length. The default of 1 hour is
            declared and deliberately coarse for analyst-paced reporting.
            Coordination studies estimate the window from their own data
            (Giglietto et al., 2020).
        coordination_min_sources: Fewest distinct sources a window must hold to
            be flagged. The default 3 is the smallest burst beyond a pair.
        observation_hours: Monitored period ``T`` for the Poisson null. With
            ``None``, the span of the batch's timestamps is used, at least one
            window. A data-derived ``T`` is a declared approximation.
        fdr_alpha: Benjamini-Hochberg false-discovery level (conventional 0.05).
        benford_min_n: Fewest positive values for the Benford test. The default
            110 is the smallest ``n`` with every expected digit count at least
            5 (Cochran, 1954), since ``110 * P(9) = 5.04``.
        benford_min_orders_of_magnitude: Minimum ``log10(max/min)``. Benford
            behaviour needs data spread over several orders of magnitude
            (Fewster, 2009).
        terminal_digit_min_n: Fewest values for the terminal-digit chi-square.
            The default 50 gives an expected count of 5 per digit (Cochran, 1954).
        rounding_min_n: Fewest values for the rounding test. The default 25
            gives an expected count of 5 rounded values.
        digit_decimals: Decimal places at which values are recorded. The
            terminal digit is the last of these.
        terminal_min_units: Values below this many units of the last recorded
            place are excluded from the digit tests. A one-digit number's last
            digit is also its first, and it is not uniform (Preece, 1981).
        fabrication_unit: ``"report"`` screens each report's values. ``"source"``
            pools the values of every report from one source.
        max_speed_kmh: Impossible-travel speed bound
            (:data:`DEFAULT_MAX_SPEED_KMH`).
        circular_factor: Weight multiplier for members of a citation cycle.
        coordinated_factor: Weight multiplier for reports in a flagged campaign.
        fabrication_factor: Weight multiplier for fabrication-flagged reports.
        impossible_travel_factor: Weight multiplier for both reports of a
            violating pair.
        claim_conflict_tolerance: Largest spread of stated ``threat_score``
            values inside one independence cluster (and corroboration group)
            that is not flagged as CONFLICTING_CLAIMS. Default 0: reports that
            cannot count independently but state different scores are always
            surfaced, because only the most reliable of them carries weight.

    The four factors default to 0.5, which halves the weight. This is a declared,
    uncalibrated choice. Set a factor to 0 to exclude flagged reports, or to 1
    to report without down-weighting.
    """

    # A factory, not the proxy itself: before Python 3.12 a mappingproxy is
    # unhashable, so dataclasses reject it as a mutable default.
    grade_prior_mean: Mapping[str, float | None] = field(
        default_factory=lambda: ADMIRALTY_RELIABILITY_PRIOR_MEAN
    )
    prior_strength: float = 2.0
    forgetting_factor: float = 1.0
    credible_level: float = 0.95
    log_odds_clip: float = DEFAULT_LOG_ODDS_CLIP
    discipline_prior_strength: float = 2.0
    shingle_size: int = 3
    minhash_permutations: int = 256
    near_duplicate_threshold: float = 0.5
    seed: int = 0
    coordination_window_hours: float = 1.0
    coordination_min_sources: int = 3
    observation_hours: float | None = None
    fdr_alpha: float = 0.05
    benford_min_n: int = 110
    benford_min_orders_of_magnitude: float = 2.0
    terminal_digit_min_n: int = 50
    rounding_min_n: int = 25
    digit_decimals: int = 0
    terminal_min_units: int = 10
    fabrication_unit: Literal["report", "source"] = "report"
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH
    circular_factor: float = 0.5
    coordinated_factor: float = 0.5
    fabrication_factor: float = 0.5
    impossible_travel_factor: float = 0.5
    claim_conflict_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Validate every parameter.

        Raises:
            ValueError: If any parameter is outside its documented range.
        """
        _validate_grade_table(self.grade_prior_mean)
        _require(self.prior_strength > 0.0, "prior_strength must be positive")
        _require(0.0 < self.forgetting_factor <= 1.0, "forgetting_factor must lie in (0, 1]")
        _require(0.0 < self.credible_level < 1.0, "credible_level must lie in (0, 1)")
        _require(self.log_odds_clip > 0.0, "log_odds_clip must be positive")
        _require(self.discipline_prior_strength > 0.0, "discipline_prior_strength must be positive")
        _require(self.shingle_size >= 1, "shingle_size must be at least 1")
        _require(self.minhash_permutations >= 1, "minhash_permutations must be at least 1")
        _require(
            0.0 < self.near_duplicate_threshold <= 1.0,
            "near_duplicate_threshold must lie in (0, 1]",
        )
        _require(self.coordination_window_hours > 0.0, "coordination_window_hours must be positive")
        _require(self.coordination_min_sources >= 2, "coordination_min_sources must be at least 2")
        _require(
            self.observation_hours is None or self.observation_hours > 0.0,
            "observation_hours must be positive when given",
        )
        _require(0.0 < self.fdr_alpha < 1.0, "fdr_alpha must lie in (0, 1)")
        _require(self.benford_min_n >= 1, "benford_min_n must be at least 1")
        _require(
            self.benford_min_orders_of_magnitude >= 0.0,
            "benford_min_orders_of_magnitude must be non-negative",
        )
        _require(self.terminal_digit_min_n >= 1, "terminal_digit_min_n must be at least 1")
        _require(self.rounding_min_n >= 1, "rounding_min_n must be at least 1")
        _require(self.digit_decimals >= 0, "digit_decimals must be non-negative")
        _require(self.terminal_min_units >= 1, "terminal_min_units must be at least 1")
        _require(
            self.fabrication_unit in ("report", "source"),
            "fabrication_unit must be 'report' or 'source'",
        )
        _require(self.max_speed_kmh > 0.0, "max_speed_kmh must be positive")
        for name in (
            "circular_factor",
            "coordinated_factor",
            "fabrication_factor",
            "impossible_travel_factor",
        ):
            value = getattr(self, name)
            _require(0.0 <= value <= 1.0, f"{name} must lie in [0, 1], got {value!r}")
        _require(
            self.claim_conflict_tolerance >= 0.0,
            "claim_conflict_tolerance must be non-negative",
        )


_DEFAULT_CONFIG = CounterintelligenceConfig()


# --------------------------------------------------------------------------- #
# 1. Source reliability: Beta-Bernoulli posteriors.
# --------------------------------------------------------------------------- #

#: Largest evidential weight one adjudicated outcome may carry, in Bernoulli
#: observations. Fitted, not chosen: :func:`fit_outcome_weight_cap` on 8,401
#: PolitiFact rulings of 1,072 speakers (LIAR; Wang, 2017), each predicted
#: from the same speaker's earlier rulings under this module's default
#: Beta(1, 1) prior (``prior_strength`` 2). Counting every outcome ``w`` times
#: predicts worse than ignoring the track record once ``w`` exceeds 3.07
#: (bootstrap 95% CI over speakers 1.58-5.39); the cap is that interval's
#: lower bound rounded down to 0.5. The previously declared cap of 10 allowed
#: weights at which a track record was measurably harmful (log loss 0.759 vs
#: 0.693 with no history). The loss-minimising weight is 0.33: an ordinary
#: outcome is worth about a third of an independent observation. LIAR is a
#: proxy for intelligence sources; refit on your own outcomes with
#: ``benchmarks/fit_outcome_weight_cap.py --histories``
#: (``benchmarks/results/outcome_weight_fit.json``). Without a bound, one
#: outcome of weight 1e308 drove a source's posterior mean -- and the fusion
#: engine's adjusted reliability -- to 1.0 in a single step.
MAX_OUTCOME_WEIGHT = 1.5


@dataclass(frozen=True)
class SourceOutcome:
    """One adjudicated outcome of a source's reporting.

    Attributes:
        confirmed: True if the report was later confirmed, False if refuted.
        weight: Evidential weight of the adjudication, a finite number in
            ``[0, MAX_OUTCOME_WEIGHT]`` (default 1; 0 is a no-op).
    """

    confirmed: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the outcome.

        Raises:
            TypeError: If ``confirmed`` is not a bool.
            ValueError: If ``weight`` is not a finite number in
                ``[0, MAX_OUTCOME_WEIGHT]``.
        """
        if not isinstance(self.confirmed, bool | np.bool_):
            raise TypeError(f"confirmed must be a bool, got {type(self.confirmed).__name__}")
        weight = _as_float(self.weight)
        _require(
            weight is not None and 0.0 <= weight <= MAX_OUTCOME_WEIGHT,
            f"weight must be finite and in [0, {MAX_OUTCOME_WEIGHT:g}], got {self.weight!r}",
        )


def admiralty_prior(
    grade: str | None,
    *,
    prior_strength: float = 2.0,
    table: Mapping[str, float | None] = ADMIRALTY_RELIABILITY_PRIOR_MEAN,
) -> tuple[float, float]:
    """Return the Beta prior ``(alpha, beta)`` for an Admiralty reliability grade.

    Args:
        grade: ``"A"``-``"F"`` (case-insensitive), or ``None`` when ungraded.
        prior_strength: Pseudo-count ``alpha + beta`` of a graded prior.
        table: Grade-to-prior-mean table (declared, see
            :data:`ADMIRALTY_RELIABILITY_PRIOR_MEAN`).

    Returns:
        ``(prior_strength * m, prior_strength * (1 - m))`` for a graded source
        with table mean ``m``. F and ``None`` give the uninformative ``(1, 1)``.

    Raises:
        ValueError: If ``grade`` is not one of A-F, or ``prior_strength <= 0``.
    """
    _require(prior_strength > 0.0, "prior_strength must be positive")
    if grade is None:
        return 1.0, 1.0
    key = grade.strip().upper()
    _require(key in ADMIRALTY_GRADES, f"unknown Admiralty reliability grade {grade!r}")
    mean = table[key]
    if mean is None:
        return 1.0, 1.0
    return prior_strength * mean, prior_strength * (1.0 - mean)


@dataclass(frozen=True)
class SourcePosterior:
    """A source's Beta posterior reliability.

    Attributes:
        source_id: The source.
        grade: Admiralty grade the prior came from (``None`` when ungraded).
        prior_alpha: Prior ``alpha``.
        prior_beta: Prior ``beta``.
        confirmed_weight: Discounted weight of confirmed outcomes.
        refuted_weight: Discounted weight of refuted outcomes.
        alpha: Posterior ``alpha = prior_alpha + confirmed_weight``.
        beta: Posterior ``beta = prior_beta + refuted_weight``.
        mean: Posterior mean ``alpha / (alpha + beta)``.
        credible_interval: Equal-tailed interval at ``credible_level``.
        credible_level: Interval mass.
        effective_sample_size: ``alpha + beta`` (prior pseudo-counts plus
            discounted evidence).
        log_odds_weight: ``logit(mean) = ln(alpha / beta)``, clipped to
            ``±log_odds_clip``.
        informative_prior: Whether the prior came from an A-E grade.
    """

    source_id: str
    grade: str | None
    prior_alpha: float
    prior_beta: float
    confirmed_weight: float
    refuted_weight: float
    alpha: float
    beta: float
    mean: float
    credible_interval: tuple[float, float]
    credible_level: float
    effective_sample_size: float
    log_odds_weight: float
    informative_prior: bool

    @property
    def evidence_weight(self) -> float:
        """Total discounted weight of adjudicated outcomes."""
        return self.confirmed_weight + self.refuted_weight

    @property
    def informative_counts(self) -> tuple[float, float]:
        """Pseudo-counts this source adds when pooled into a discipline.

        A graded source adds its whole posterior. An ungraded source (or grade F)
        adds only its evidence, because its Beta(1, 1) carries no
        source-specific information.
        """
        if self.informative_prior:
            return self.alpha, self.beta
        return self.confirmed_weight, self.refuted_weight


def source_posterior(
    source_id: str,
    grade: str | None = None,
    outcomes: Sequence[SourceOutcome] = (),
    *,
    config: CounterintelligenceConfig | None = None,
) -> SourcePosterior:
    """Update a source's Admiralty-seeded Beta prior with adjudicated outcomes.

    Outcomes are applied in the order given, which should be chronological. With
    forgetting factor ``lambda``, each existing evidence count is multiplied by
    ``lambda`` before the next outcome is added (Jøsang & Ismail, 2002). The
    prior is not discounted.

    Args:
        source_id: The source identifier.
        grade: Admiralty reliability grade A-F, or ``None``.
        outcomes: Adjudicated outcomes, oldest first.
        config: Parameters (defaults when ``None``).

    Returns:
        The :class:`SourcePosterior`.
    """
    cfg = config or _DEFAULT_CONFIG
    prior_alpha, prior_beta = admiralty_prior(
        grade, prior_strength=cfg.prior_strength, table=cfg.grade_prior_mean
    )
    confirmed = 0.0
    refuted = 0.0
    for outcome in outcomes:
        confirmed *= cfg.forgetting_factor
        refuted *= cfg.forgetting_factor
        if outcome.confirmed:
            confirmed += float(outcome.weight)
        else:
            refuted += float(outcome.weight)
    alpha = prior_alpha + confirmed
    beta = prior_beta + refuted
    tail = (1.0 - cfg.credible_level) / 2.0
    low, high = stats.beta.ppf([tail, 1.0 - tail], alpha, beta)
    normalised = None if grade is None else grade.strip().upper()
    return SourcePosterior(
        source_id=source_id,
        grade=normalised,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        confirmed_weight=confirmed,
        refuted_weight=refuted,
        alpha=alpha,
        beta=beta,
        mean=alpha / (alpha + beta),
        credible_interval=(float(low), float(high)),
        credible_level=cfg.credible_level,
        effective_sample_size=alpha + beta,
        log_odds_weight=float(
            np.clip(math.log(alpha / beta), -cfg.log_odds_clip, cfg.log_odds_clip)
        ),
        informative_prior=normalised is not None and cfg.grade_prior_mean[normalised] is not None,
    )


@dataclass(frozen=True)
class OutcomeWeightFit:
    """Prequential fit of the evidential weight of adjudicated outcomes.

    Attributes:
        n_sources: Sources with at least two outcomes (the only ones that
            contribute a prediction made from their own history).
        n_outcomes: Outcomes predicted.
        prior_strength: Beta prior pseudo-count ``alpha + beta`` used.
        weights: The weight grid evaluated.
        log_loss: Mean prequential log loss at each grid weight.
        optimal_weight: The grid weight with the lowest loss: how many
            Bernoulli observations one ordinary outcome is worth.
        break_even_weight: The largest weight at which a track record still
            predicts no worse than no track record (``log_loss(w) <=
            log_loss(0)``). Above it, counting outcomes makes predictions worse.
        break_even_ci: Percentile bootstrap interval of ``break_even_weight``
            over sources (``None`` when ``n_boot`` is 0).
        confidence: Level of ``break_even_ci``.
    """

    n_sources: int
    n_outcomes: int
    prior_strength: float
    weights: tuple[float, ...]
    log_loss: tuple[float, ...]
    optimal_weight: float
    break_even_weight: float
    break_even_ci: tuple[float, float] | None
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return dict(_jsonable(self))


def _prequential_terms(
    histories: Sequence[npt.NDArray[np.float64]],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.intp]
]:
    """Flatten histories into (outcome, confirmed-before, seen-before, source index) arrays."""
    outcome = np.concatenate(histories)
    confirmed = np.concatenate([np.concatenate(([0.0], np.cumsum(h)[:-1])) for h in histories])
    seen = np.concatenate([np.arange(h.size, dtype=np.float64) for h in histories])
    source = np.concatenate([np.full(h.size, i, dtype=np.intp) for i, h in enumerate(histories)])
    return outcome, confirmed, seen, source


def _per_source_loss(
    terms: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.intp],
    ],
    weight: float,
    prior_strength: float,
    n_sources: int,
) -> npt.NDArray[np.float64]:
    """Summed prequential log loss of each source at one weight."""
    outcome, confirmed, seen, source = terms
    p = (prior_strength / 2.0 + weight * confirmed) / (prior_strength + weight * seen)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    loss = -np.where(outcome > 0.5, np.log(p), np.log1p(-p))
    return np.bincount(source, weights=loss, minlength=n_sources)


def _last_crossing(grid: npt.NDArray[np.float64], excess: npt.NDArray[np.float64]) -> float:
    """Largest grid point region with ``excess <= 0``, linearly interpolated at its edge."""
    if excess[-1] <= 0.0:
        return float(grid[-1])
    for i in range(grid.size - 1, 0, -1):
        if excess[i - 1] <= 0.0 < excess[i]:
            fraction = -excess[i - 1] / (excess[i] - excess[i - 1])
            return float(grid[i - 1] + fraction * (grid[i] - grid[i - 1]))
    return 0.0


def fit_outcome_weight_cap(
    histories: Mapping[str, Sequence[bool | SourceOutcome]],
    *,
    prior_strength: float = 2.0,
    weights: Sequence[float] | None = None,
    n_boot: int = 400,
    confidence: float = 0.95,
    seed: int = 0,
    upper: float = 100.0,
) -> OutcomeWeightFit:
    """Fit how much evidential weight an adjudicated outcome deserves, on real outcomes.

    Each source's outcomes, oldest first, are predicted one at a time from
    that source's earlier outcomes alone under the Beta-Bernoulli update of
    :func:`source_posterior` with every outcome counted ``w`` times. The mean
    log loss of those predictions (Dawid's prequential principle) is computed
    over a grid of ``w``. Two numbers come out:

    * the loss-minimising weight, what one ordinary outcome is worth;
    * the break-even weight, beyond which counting outcomes predicts worse
      than ignoring a source's track record. :data:`MAX_OUTCOME_WEIGHT` is the
      lower bound of its bootstrap interval: no admissible weight makes the
      track record harmful even at the pessimistic end.

    Sources resample with replacement in the bootstrap, keeping each source's
    sequence intact (outcomes within a source are not independent).

    Args:
        histories: Source id to its outcomes, oldest first (bools or
            :class:`SourceOutcome`; an outcome's own weight is ignored here,
            since the weight is what is being fitted).
        prior_strength: Beta prior pseudo-count, as ``CounterintelligenceConfig``.
        weights: Grid for ``log_loss``; a default geometric grid when ``None``.
        n_boot: Bootstrap resamples (0 skips the interval).
        confidence: Interval level.
        seed: Bootstrap seed.
        upper: Largest weight searched for the break-even point.

    Returns:
        The :class:`OutcomeWeightFit`.

    Raises:
        ValueError: Fewer than two sources with at least two outcomes.
    """
    sequences: list[npt.NDArray[np.float64]] = []
    for outcomes in histories.values():
        values = [float(o.confirmed if isinstance(o, SourceOutcome) else bool(o)) for o in outcomes]
        if len(values) >= 2:
            sequences.append(np.asarray(values, dtype=np.float64))
    _require(len(sequences) >= 2, "need at least two sources with two or more outcomes")
    _require(prior_strength > 0.0, "prior_strength must be positive")
    n = len(sequences)
    terms = _prequential_terms(sequences)
    counts = np.bincount(terms[3], minlength=n).astype(np.float64)

    def mean_loss(weight: float) -> float:
        return float(_per_source_loss(terms, weight, prior_strength, n).sum() / counts.sum())

    grid = tuple(
        float(w)
        for w in (
            weights
            if weights is not None
            else (0.0, *np.geomspace(0.05, upper, 25).round(4).tolist())
        )
    )
    losses = tuple(mean_loss(w) for w in grid)

    # Break-even: per-source losses on a fine grid once; each bootstrap draw
    # is then a weighted sum over sources (multiplicities of the resample).
    search = np.geomspace(1e-3, upper, 512)
    table = np.stack([_per_source_loss(terms, float(w), prior_strength, n) for w in search], axis=1)
    baseline = _per_source_loss(terms, 0.0, prior_strength, n)

    def break_even(multiplicity: npt.NDArray[np.float64]) -> float:
        total = float(multiplicity @ counts)
        excess = (multiplicity @ table - multiplicity @ baseline) / total
        return _last_crossing(search, excess)

    point = break_even(np.ones(n))
    interval: tuple[float, float] | None = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        draws = [
            break_even(np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64))
            for _ in range(n_boot)
        ]
        tail = (1.0 - confidence) / 2.0 * 100.0
        low, high = np.percentile(draws, [tail, 100.0 - tail])
        interval = (float(low), float(high))
    return OutcomeWeightFit(
        n_sources=len(sequences),
        n_outcomes=int(sum(s.size for s in sequences)),
        prior_strength=prior_strength,
        weights=grid,
        log_loss=losses,
        optimal_weight=grid[int(np.argmin(losses))],
        break_even_weight=point,
        break_even_ci=interval,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# Report normalisation.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeoFix:
    """A reported location with its uncertainty radius.

    Attributes:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_km: Uncertainty radius in km (0 when not stated).
    """

    lat: float
    lon: float
    radius_km: float = 0.0


@dataclass(frozen=True)
class ReportRecord:
    """One report, normalised from the fusion engine's input mapping.

    Attributes:
        report_id: Report identifier. Defaults to the discipline key, or to
            ``"<discipline>[i]"`` for the i-th report in a list. Default ids
            are reserved: a declared id equal to another report's default id
            is renamed (see :func:`normalize_reports`), so a report can never
            take over another discipline's identity by declaring its key.
        discipline: The discipline key the report arrived under.
        source_id: Reporting source, when stated.
        reported_at: POSIX seconds, when stated with a timezone.
        text: Report text, when stated.
        derived_from: Declared upstream report or document ids.
        claim_id: Claim the report bears on. Without one, the report is
            evidence for the batch's overall assessment.
        entity_id: Entity whose location ``location`` gives (impossible
            travel). Without one, the source's own track is used.
        information_credibility: Admiralty information-credibility grade 1-6.
        source_grade: Admiralty source-reliability grade A-F.
        location: Reported location.
        numeric_values: Finite numeric values for the fabrication screens.
        provenance_adequate: The repository's ``Provenance.is_adequate()``
            verdict for an attached provenance record (``None`` if absent).
        threat_score: The report's stated ``threat_score`` when it is a
            finite number in ``[0, 1]``. Read only to detect conflicting
            claims inside an independence cluster; never used to score.
        same_as: Ids of reports that declared the same report id. They are
            the same report presented twice, merged into one cluster without
            an edge, so neither copy is presumed to be the original.
    """

    report_id: str
    discipline: str
    source_id: str | None = None
    reported_at: float | None = None
    text: str | None = None
    derived_from: tuple[str, ...] = ()
    claim_id: str | None = None
    entity_id: str | None = None
    information_credibility: int | None = None
    source_grade: str | None = None
    location: GeoFix | None = None
    numeric_values: tuple[float, ...] = ()
    provenance_adequate: bool | None = None
    threat_score: float | None = None
    same_as: tuple[str, ...] = ()

    @property
    def source_key(self) -> str:
        """The source id, or a per-report anonymous key when none is stated."""
        return self.source_id if self.source_id is not None else f"anonymous:{self.report_id}"

    @property
    def group(self) -> str:
        """The corroboration group: the claim id, or the whole assessment."""
        return self.claim_id if self.claim_id is not None else _DEFAULT_GROUP


def _id_list(value: object) -> tuple[list[str], int]:
    """Parse a list of ids, returning the ids and the number of rejected items."""
    if value is None:
        return [], 0
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple):
        return [], 1
    ids = [_as_id(item) for item in value]
    return [item for item in ids if item is not None], sum(item is None for item in ids)


def _provenance_citations(value: object) -> tuple[tuple[str, ...], bool | None]:
    """Read citations and adequacy from a ``Provenance`` record or its ``as_dict``.

    The adequacy verdict is ``Provenance.is_adequate()`` from
    ``intel/provenance.py``: an attributed origin with at least one citation.
    The provenance module is imported only when a mapping must be converted,
    because its import chain is heavy.
    """
    if isinstance(value, Mapping):
        from omni_mercury_engine.intel.provenance import Provenance, ProvenanceOrigin

        sources = value.get("sources", ())
        if not isinstance(sources, list | tuple):
            return (), None
        try:
            record = Provenance(
                origin=ProvenanceOrigin(str(value.get("origin"))),
                sources=tuple(str(s) for s in sources),
                verified=bool(value.get("verified", False)),
            )
        except ValueError:
            return (), None
    elif hasattr(value, "sources") and callable(getattr(value, "is_adequate", None)):
        record = value  # type: ignore[assignment]
    else:
        return (), None
    cited = tuple(s.strip() for s in record.sources if isinstance(s, str) and s.strip())
    return cited, bool(record.is_adequate())


def _location(value: object) -> GeoFix | None:
    """Parse a ``{lat, lon, radius_km}`` mapping."""
    if not isinstance(value, Mapping):
        return None
    lat = _as_float(value.get("lat"))
    lon = _as_float(value.get("lon"))
    radius = _as_float(value.get("radius_km", 0.0))
    if lat is None or lon is None or radius is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and radius >= 0.0):
        return None
    return GeoFix(lat=lat, lon=lon, radius_km=radius)


def _normalise_one(
    discipline: str, default_id: str, raw: Mapping[str, Any]
) -> tuple[ReportRecord, list[str]]:
    """Normalise one report mapping, noting every field that was dropped."""
    notes: list[str] = []
    report_id = _as_id(raw.get("report_id")) or default_id
    label = f"report {report_id}"

    def note(name: str, why: str) -> None:
        notes.append(f"{label}: {name} ignored ({why})")

    reported_at = parse_timestamp(raw.get("reported_at"))
    if raw.get("reported_at") is not None and reported_at is None:
        note("reported_at", "not epoch seconds or a timezone-aware ISO-8601 time")
    text = raw.get("text")
    if text is not None and not isinstance(text, str):
        note("text", "not a string")
        text = None
    cited, rejected = _id_list(raw.get("derived_from"))
    more, rejected_more = _id_list(raw.get("citations"))
    if rejected + rejected_more:
        note("derived_from/citations", f"{rejected + rejected_more} non-id entries")
    prov_cited, adequate = _provenance_citations(raw.get("provenance"))
    if raw.get("provenance") is not None and adequate is None:
        note("provenance", "not a Provenance record or its as_dict() mapping")
    elif adequate is False:
        notes.append(
            f"{label}: provenance fails Provenance.is_adequate() "
            "(unattributed origin or no citation)"
        )
    derived = tuple(dict.fromkeys([*cited, *more, *prov_cited]))
    credibility_raw = raw.get("information_credibility")
    credibility: int | None = None
    if credibility_raw is not None:
        as_number = _as_float(credibility_raw)
        if as_number is not None and as_number.is_integer() and 1 <= as_number <= 6:
            credibility = int(as_number)
        else:
            note("information_credibility", "not an integer grade 1-6")
    grade_raw = raw.get("source_grade")
    grade: str | None = None
    if grade_raw is not None:
        if isinstance(grade_raw, str) and grade_raw.strip().upper() in ADMIRALTY_GRADES:
            grade = grade_raw.strip().upper()
        else:
            note("source_grade", "not an Admiralty grade A-F")
    location = _location(raw.get("location"))
    if raw.get("location") is not None and location is None:
        note("location", "needs finite lat in [-90, 90], lon in [-180, 180], radius_km >= 0")
    values_raw = raw.get("numeric_values")
    numeric: tuple[float, ...] = ()
    if values_raw is not None:
        if isinstance(values_raw, list | tuple | np.ndarray):
            parsed = [_as_float(v) for v in values_raw]
            numeric = tuple(v for v in parsed if v is not None)
            dropped = len(parsed) - len(numeric)
            if dropped:
                note("numeric_values", f"{dropped} non-finite or non-numeric entries dropped")
        else:
            note("numeric_values", "not a list of numbers")
    stated_score = _as_float(raw.get("threat_score"))
    record = ReportRecord(
        report_id=report_id,
        discipline=discipline,
        source_id=_as_id(raw.get("source_id")),
        reported_at=reported_at,
        text=text,
        derived_from=derived,
        claim_id=_as_id(raw.get("claim_id")),
        entity_id=_as_id(raw.get("entity_id")),
        information_credibility=credibility,
        source_grade=grade,
        location=location,
        numeric_values=numeric,
        provenance_adequate=adequate,
        threat_score=(
            stated_score if stated_score is not None and 0.0 <= stated_score <= 1.0 else None
        ),
    )
    return record, notes


def _fresh_id(base: str, discipline: str, taken: set[str]) -> str:
    """Return ``"<base>@<discipline>"``, suffixed ``#n`` until it is not in ``taken``."""
    renamed = f"{base}@{discipline}"
    suffix = 2
    while renamed in taken:
        renamed = f"{base}@{discipline}#{suffix}"
        suffix += 1
    return renamed


def normalize_reports(
    intel_reports: Mapping[str, Any],
) -> tuple[tuple[ReportRecord, ...], tuple[str, ...]]:
    """Normalise a fusion-engine report mapping into :class:`ReportRecord` values.

    A discipline's value may be one report mapping, which is the engine's
    format, or a list of them. Disciplines are read in sorted key order, so
    the result does not depend on dict insertion order. The input is never
    mutated.

    Identity rules:

    * A report without a declared ``report_id`` gets a default id (the
      discipline key, or ``"<discipline>[i]"``). Default ids are reserved: a
      declared id equal to *another* report's default id is renamed
      ``"<id>@<discipline>"`` and is not treated as that report. Otherwise
      ``{"communications": {"report_id": "signals"}}`` would turn the genuine
      signals report into a "copy" of the impostor and zero its weight.
    * A declared id seen twice is the same report presented twice. The second
      copy is renamed ``"<id>@<discipline>"`` and listed in ``same_as`` of the
      first's id. The pair forms one independence cluster *without* a
      citation edge, so which copy carries the weight is decided by
      reliability (:func:`analyse_independence`), not by key order.

    Args:
        intel_reports: Mapping of discipline key to report mapping(s).

    Returns:
        ``(records, notes)``, where ``notes`` explains every ignored field.

    Raises:
        TypeError: If ``intel_reports`` is not a mapping.
    """
    if not isinstance(intel_reports, Mapping):
        raise TypeError(
            f"intel_reports must be a mapping of discipline to report, got "
            f"{type(intel_reports).__name__}"
        )
    records: list[ReportRecord] = []
    notes: list[str] = []
    pending: list[tuple[str, str, Mapping[str, Any]]] = []
    for key in sorted(intel_reports, key=str):
        discipline = str(key)
        value = intel_reports[key]
        items: list[tuple[str, object]]
        if isinstance(value, Mapping):
            items = [(discipline, value)]
        elif isinstance(value, list | tuple):
            items = [(f"{discipline}[{i}]", item) for i, item in enumerate(value)]
        else:
            notes.append(f"discipline {discipline}: value is not a report mapping; ignored")
            continue
        for default_id, raw in items:
            if not isinstance(raw, Mapping):
                notes.append(f"{default_id}: not a report mapping; ignored")
                continue
            pending.append((discipline, default_id, raw))
    reserved = {default_id for _, default_id, _ in pending}
    seen: set[str] = set()
    for discipline, default_id, raw in pending:
        record, record_notes = _normalise_one(discipline, default_id, raw)
        notes.extend(record_notes)
        declared = record.report_id
        if declared != default_id and declared in reserved:
            renamed = _fresh_id(declared, discipline, seen | reserved)
            record = replace(record, report_id=renamed)
            notes.append(
                f"report_id {declared} declared under {discipline} is another report's "
                f"default id; renamed {renamed}, not treated as that report"
            )
        elif declared in seen:
            renamed = _fresh_id(declared, discipline, seen | reserved)
            record = replace(record, report_id=renamed, same_as=(declared,))
            notes.append(
                f"report {declared} appears again under {discipline}; the copy is "
                f"{renamed}, treated as the same report as {declared}"
            )
        seen.add(record.report_id)
        records.append(record)
    return tuple(records), tuple(notes)


# --------------------------------------------------------------------------- #
# 2a. Near-duplicate text: word shingles and MinHash.
# --------------------------------------------------------------------------- #


def word_shingles(text: str, size: int = 3) -> frozenset[str]:
    """Return the set of ``size``-word shingles of ``text``.

    Tokens are maximal runs of word characters after Unicode case folding. A
    text with fewer than ``size`` tokens has no shingles. It is then not
    compared, because two identical one-word texts are not evidence of copying.

    Args:
        text: The text.
        size: Words per shingle.

    Returns:
        The shingles, each the space-joined tokens.

    Raises:
        ValueError: If ``size < 1``.
    """
    _require(size >= 1, "shingle size must be at least 1")
    tokens = _TOKEN_RE.findall(text.casefold())
    return frozenset(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float | None:
    """Exact Jaccard similarity ``|A & B| / |A | B|``, or ``None`` if both are empty."""
    set_a, set_b = frozenset(a), frozenset(b)
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def _base_hash(shingle: str) -> int:
    """Unsalted 32-bit BLAKE2b (RFC 7693) hash, stable across processes."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


class MinHasher:
    """MinHash signatures under a seeded universal hash family.

    Permutation ``i`` is ``h_i(x) = (a_i * x + b_i) mod p``, with ``p`` equal to
    :data:`MINHASH_PRIME`, applied to a 32-bit base hash ``x`` of each shingle.
    The fraction of signature positions on which two sets agree estimates
    their Jaccard similarity (Broder, 1997). The estimate is unbiased only
    under min-wise independence, which a linear family satisfies only
    approximately. The tests measure the error against exact Jaccard.
    """

    def __init__(self, num_permutations: int = 256, seed: int = 0) -> None:
        """Draw the hash family.

        Args:
            num_permutations: Signature length ``k``.
            seed: Seed for ``numpy.random.default_rng``.

        Raises:
            ValueError: If ``num_permutations < 1``.
        """
        _require(num_permutations >= 1, "num_permutations must be at least 1")
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, 2**32, size=num_permutations, dtype=np.uint64)
        self._b = rng.integers(0, 2**32, size=num_permutations, dtype=np.uint64)

    @property
    def num_permutations(self) -> int:
        """Signature length."""
        return int(self._a.shape[0])

    def signature(self, shingles: Iterable[str]) -> npt.NDArray[np.uint64]:
        """Return the MinHash signature of a non-empty shingle set.

        Args:
            shingles: The shingles.

        Returns:
            A uint64 array of length :attr:`num_permutations`.

        Raises:
            ValueError: If the set is empty.
        """
        items = sorted(frozenset(shingles))
        _require(bool(items), "an empty shingle set has no MinHash signature")
        x = np.fromiter((_base_hash(s) for s in items), dtype=np.uint64, count=len(items))
        hashed = (self._a[:, None] * x[None, :] + self._b[:, None]) % np.uint64(MINHASH_PRIME)
        return np.asarray(hashed.min(axis=1), dtype=np.uint64)

    @staticmethod
    def estimate(sig_a: npt.NDArray[np.uint64], sig_b: npt.NDArray[np.uint64]) -> float:
        """Estimated Jaccard similarity: the share of agreeing signature positions.

        Args:
            sig_a: First signature.
            sig_b: Second signature, from the same :class:`MinHasher`.

        Returns:
            The estimate in ``[0, 1]``.

        Raises:
            ValueError: If the signatures differ in shape.
        """
        _require(sig_a.shape == sig_b.shape, "signatures must come from the same MinHasher")
        return float(np.mean(sig_a == sig_b))


@dataclass(frozen=True)
class NearDuplicatePair:
    """Two reports whose texts are near duplicates.

    Attributes:
        report_a: The earlier report when ``directed``. Otherwise the smaller id.
        report_b: The later report when ``directed``. Otherwise the larger id.
        jaccard_estimate: MinHash estimate of the shingle Jaccard similarity.
        directed: True when both times are known and differ. ``report_b`` is
            then presumed derived from ``report_a``.
    """

    report_a: str
    report_b: str
    jaccard_estimate: float
    directed: bool


def find_near_duplicates(
    records: Sequence[ReportRecord], config: CounterintelligenceConfig | None = None
) -> tuple[NearDuplicatePair, ...]:
    """Find report pairs whose MinHash Jaccard estimate reaches the threshold.

    Every pair of shingled texts is compared. The cost is ``O(n^2 k)``, which
    suits batches up to a few thousand reports. LSH banding would scale
    further and is not implemented.

    Args:
        records: Normalised reports. Those without text or shingles are skipped.
        config: Parameters (defaults when ``None``).

    Returns:
        The pairs, sorted by ``(report_a, report_b)``.
    """
    cfg = config or _DEFAULT_CONFIG
    shingled = [
        (record, word_shingles(record.text, cfg.shingle_size))
        for record in records
        if record.text is not None
    ]
    shingled = [(record, shingles) for record, shingles in shingled if shingles]
    if len(shingled) < 2:
        return ()
    hasher = MinHasher(cfg.minhash_permutations, cfg.seed)
    signatures = np.stack([hasher.signature(shingles) for _, shingles in shingled])
    pairs: list[NearDuplicatePair] = []
    for i in range(len(shingled) - 1):
        estimates = np.mean(signatures[i + 1 :] == signatures[i], axis=1)
        for offset in np.flatnonzero(estimates >= cfg.near_duplicate_threshold):
            first = shingled[i][0]
            second = shingled[i + 1 + int(offset)][0]
            estimate = float(estimates[offset])
            t_first, t_second = first.reported_at, second.reported_at
            if t_first is not None and t_second is not None and t_first != t_second:
                earlier, later = (first, second) if t_first < t_second else (second, first)
                pairs.append(NearDuplicatePair(earlier.report_id, later.report_id, estimate, True))
            else:
                low, high = sorted((first.report_id, second.report_id))
                pairs.append(NearDuplicatePair(low, high, estimate, False))
    return tuple(sorted(pairs, key=lambda p: (p.report_a, p.report_b)))


# --------------------------------------------------------------------------- #
# 2b. Provenance graph, circularity, shared origins and independence clusters.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CounterintelligenceFlag:
    """One finding, with the evidence that produced it.

    Attributes:
        kind: What was found.
        report_ids: Reports the finding concerns.
        description: One-line explanation.
        evidence: Paths, statistics and thresholds behind the finding.
        p_value: Raw p-value for statistical findings, ``None`` for structural
            and physical ones, which rest on no sampling model.
        q_value: Benjamini-Hochberg adjusted p-value, when one applies.
    """

    kind: FlagKind
    report_ids: tuple[str, ...]
    description: str
    evidence: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    p_value: float | None = None
    q_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        result: dict[str, Any] = _jsonable(self)
        return result


@dataclass(frozen=True)
class IndependenceCluster:
    """Reports that cannot count as independent of one another.

    Attributes:
        cluster_id: ``"cluster:<representative>"``.
        representative: The member that carries the cluster's weight: an
            origin report (no upstream edge) if there is one, then the most
            reliable, then the earliest, then the smallest id. Reliability
            ranks before time because ``reported_at`` is self-reported: an
            impostor reusing a genuine report's id or source with an earlier
            timestamp must not displace a more reliable report.
        report_ids: Members, sorted.
        origins: Upstream origin tokens shared through the cluster.
        source_ids: Distinct stated sources among the members.
        disciplines: Disciplines the members arrived under.
    """

    cluster_id: str
    representative: str
    report_ids: tuple[str, ...]
    origins: tuple[str, ...]
    source_ids: tuple[str, ...]
    disciplines: tuple[str, ...]


@dataclass(frozen=True)
class IndependenceAnalysis:
    """Provenance graph analysis of a report batch.

    Attributes:
        clusters: Independence clusters, sorted by id.
        cluster_of: Report id to cluster id.
        origins_of: Report id to its origin tokens.
        near_duplicates: Near-duplicate text pairs.
        edges: ``(from, to, kind)`` provenance edges. ``kind`` is
            ``"citation"`` or ``"near_duplicate"``, and edges point upstream.
        cycles: Report ids of each citation cycle (strongly connected component).
        distinct_origins: Number of distinct origin tokens. It is an upper
            bound on :attr:`effective_independent_sources`.
        flags: Circular-reporting and shared-origin findings.
    """

    clusters: tuple[IndependenceCluster, ...]
    cluster_of: Mapping[str, str]
    origins_of: Mapping[str, tuple[str, ...]]
    near_duplicates: tuple[NearDuplicatePair, ...]
    edges: tuple[tuple[str, str, str], ...]
    cycles: tuple[tuple[str, ...], ...]
    distinct_origins: int
    flags: tuple[CounterintelligenceFlag, ...]

    @property
    def effective_independent_sources(self) -> int:
        """Number of independence clusters, which is what corroboration counts."""
        return len(self.clusters)


def _strongly_connected_components(
    nodes: Sequence[str], succ: Mapping[str, Sequence[str]]
) -> list[list[str]]:
    """Tarjan's SCC algorithm, iterative.

    Components are emitted in reverse topological order: each appears after
    every component it can reach.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(succ.get(root, ())))]
        while work:
            node, successors = work[-1]
            descended = False
            for nxt in successors:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(succ.get(nxt, ()))))
                    descended = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return components


def _bfs_paths(start: str, succ: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Shortest upstream path from ``start`` to every node it reaches (excluding itself)."""
    parent: dict[str, str] = {}
    frontier = [start]
    visited = {start}
    while frontier:
        nxt_frontier: list[str] = []
        for node in frontier:
            for nxt in succ.get(node, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = node
                    nxt_frontier.append(nxt)
        frontier = nxt_frontier
    paths: dict[str, list[str]] = {}
    for target in parent:
        path = [target]
        while path[-1] != start:
            path.append(parent[path[-1]])
        paths[target] = path[::-1]
    return paths


class _UnionFind:
    """Union-find whose root is always the smallest key (deterministic)."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self._parent.setdefault(key, key)
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]  # path halving
            key = self._parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            low, high = sorted((root_a, root_b))
            self._parent[high] = low


def analyse_independence(
    records: Sequence[ReportRecord],
    config: CounterintelligenceConfig | None = None,
    reliability: Mapping[str, float | None] | None = None,
) -> IndependenceAnalysis:
    """Build the provenance graph and derive independence clusters and flags.

    Edges point upstream: a report points to each id it cites, and a later
    near duplicate points to the earlier one. Ids not in the batch are external
    origins. The *origins* of a report are the sink components it reaches, plus
    its own source, because a source cannot corroborate itself. Reports whose
    origin sets overlap, directly or transitively, form one independence
    cluster. Undirected near duplicates (equal or unknown times) are merged.

    The function emits three findings:

    * a ``citation_cycle`` (CIRCULAR_REPORTING) for every strongly connected
      component, which is reporting with no external origin;
    * a ``corroboration_loop`` (CIRCULAR_REPORTING) when a report's chain
      reaches another report in the same corroboration group, because it is
      not independent corroboration of that report;
    * SHARED_ORIGIN when reports from two or more distinct sources in one
      group trace, through at least one edge, to the same upstream origin;
    * CONFLICTING_CLAIMS when members of one cluster in the same group state
      ``threat_score`` values further apart than
      ``claim_conflict_tolerance``. Only the representative carries weight,
      so the disagreement is surfaced rather than silently dropped.

    Reports in each other's ``same_as`` (one declared id presented twice) are
    merged without an edge, so neither is presumed to be the original.

    Args:
        records: Normalised reports.
        config: Parameters (defaults when ``None``).
        reliability: Report id to reliability in ``[0, 1]`` used to choose
            each cluster's representative. Missing or ``None`` ranks below
            every stated value. When omitted, all reports tie and the
            representative is the earliest origin, then the smallest id.

    Returns:
        The :class:`IndependenceAnalysis`.
    """
    cfg = config or _DEFAULT_CONFIG
    by_id = {record.report_id: record for record in records}
    near = find_near_duplicates(records, cfg)

    succ_sets: dict[str, set[str]] = {record.report_id: set() for record in records}
    edge_kind: dict[tuple[str, str], str] = {}
    for record in records:
        for cited in record.derived_from:
            succ_sets[record.report_id].add(cited)
            succ_sets.setdefault(cited, set())
            edge_kind[(record.report_id, cited)] = "citation"
    for pair in near:
        if pair.directed:
            succ_sets[pair.report_b].add(pair.report_a)
            edge_kind.setdefault((pair.report_b, pair.report_a), "near_duplicate")
    succ = {node: sorted(targets) for node, targets in succ_sets.items()}
    nodes = sorted(succ)

    components = _strongly_connected_components(nodes, succ)
    comp_of = {node: ci for ci, comp in enumerate(components) for node in comp}
    cyclic = [len(comp) > 1 or comp[0] in succ_sets[comp[0]] for comp in components]
    comp_succ = [
        sorted({comp_of[t] for n in comp for t in succ[n] if comp_of[t] != ci})
        for ci, comp in enumerate(components)
    ]

    def token(ci: int) -> str:
        comp = components[ci]
        if cyclic[ci]:
            return f"cycle:{comp[0]}"
        node = comp[0]
        record = by_id.get(node)
        if record is None:
            return f"external:{node}"
        return f"source:{record.source_id}" if record.source_id else f"report:{node}"

    sinks: list[frozenset[int]] = []
    upstream: list[frozenset[int]] = []  # sinks reached through at least one edge
    for ci in range(len(components)):  # reverse topological: successors come first
        if not comp_succ[ci]:
            sinks.append(frozenset({ci}))
            upstream.append(frozenset({ci}) if cyclic[ci] else frozenset())
        else:
            reached = frozenset().union(*(sinks[d] for d in comp_succ[ci]))
            sinks.append(reached)
            upstream.append(reached)

    origins_of: dict[str, tuple[str, ...]] = {}
    upstream_of: dict[str, frozenset[str]] = {}
    union_find = _UnionFind()
    for record in records:
        ci = comp_of[record.report_id]
        origin_tokens = {token(s) for s in sinks[ci]}
        if record.source_id is not None:
            origin_tokens.add(f"source:{record.source_id}")
        origins_of[record.report_id] = tuple(sorted(origin_tokens))
        upstream_of[record.report_id] = frozenset(token(s) for s in upstream[ci])
        for origin in origin_tokens:
            union_find.union(f"R|{record.report_id}", f"T|{origin}")
    for pair in near:
        if not pair.directed:
            union_find.union(f"R|{pair.report_a}", f"R|{pair.report_b}")
    for record in records:
        for twin in record.same_as:
            if twin in by_id:
                union_find.union(f"R|{record.report_id}", f"R|{twin}")

    members: dict[str, list[ReportRecord]] = defaultdict(list)
    for record in records:
        members[union_find.find(f"R|{record.report_id}")].append(record)

    stated_reliability = reliability or {}

    def representative_key(record: ReportRecord) -> tuple[int, float, float, str]:
        is_origin = not succ[record.report_id]
        rel = stated_reliability.get(record.report_id)
        rank = -rel if rel is not None and math.isfinite(rel) else 1.0  # unknown ranks last
        time = record.reported_at if record.reported_at is not None else math.inf
        return (0 if is_origin else 1, rank, time, record.report_id)

    clusters: list[IndependenceCluster] = []
    cluster_of: dict[str, str] = {}
    for group in members.values():
        representative = min(group, key=representative_key)
        cluster_id = f"cluster:{representative.report_id}"
        for record in group:
            cluster_of[record.report_id] = cluster_id
        clusters.append(
            IndependenceCluster(
                cluster_id=cluster_id,
                representative=representative.report_id,
                report_ids=tuple(sorted(r.report_id for r in group)),
                origins=tuple(sorted({o for r in group for o in origins_of[r.report_id]})),
                source_ids=tuple(sorted({r.source_id for r in group if r.source_id})),
                disciplines=tuple(sorted({r.discipline for r in group})),
            )
        )
    clusters.sort(key=lambda c: c.cluster_id)

    flags: list[CounterintelligenceFlag] = []
    for cluster in clusters:
        by_claim: dict[str, list[ReportRecord]] = defaultdict(list)
        for rid in cluster.report_ids:
            if by_id[rid].threat_score is not None:
                by_claim[by_id[rid].group].append(by_id[rid])
        for group_key in sorted(by_claim):
            stated = sorted(by_claim[group_key], key=lambda r: r.report_id)
            scores = [r.threat_score for r in stated if r.threat_score is not None]
            if len(stated) < 2 or max(scores) - min(scores) <= cfg.claim_conflict_tolerance:
                continue
            flags.append(
                CounterintelligenceFlag(
                    kind=FlagKind.CONFLICTING_CLAIMS,
                    report_ids=tuple(r.report_id for r in stated),
                    description=(
                        f"{len(stated)} reports in {cluster.cluster_id} cannot count "
                        f"independently but state threat scores {min(scores):g}-"
                        f"{max(scores):g}; only {cluster.representative} carries weight"
                    ),
                    evidence=MappingProxyType(
                        {
                            "cluster": cluster.cluster_id,
                            "group": group_key,
                            "representative": cluster.representative,
                            "threat_scores": MappingProxyType(
                                {r.report_id: r.threat_score for r in stated}
                            ),
                            "reliability": MappingProxyType(
                                {r.report_id: stated_reliability.get(r.report_id) for r in stated}
                            ),
                        }
                    ),
                )
            )
    cycles: list[tuple[str, ...]] = []
    for ci, comp in enumerate(components):
        if not cyclic[ci]:
            continue
        cycles.append(tuple(comp))
        start = comp[0]
        inside = {n: [t for t in succ[n] if t in comp] for n in comp}
        if start in succ_sets[start]:
            witness = [start, start]
        else:
            first = next(t for t in inside[start] if t != start)
            witness = [start, *_bfs_paths(first, inside)[start]]
        flags.append(
            CounterintelligenceFlag(
                kind=FlagKind.CIRCULAR_REPORTING,
                report_ids=tuple(comp),
                description=(f"citation cycle through {len(comp)} report(s); no external origin"),
                evidence=MappingProxyType(
                    {
                        "pattern": "citation_cycle",
                        "cycle": tuple(witness),
                        "edge_kinds": tuple(
                            edge_kind[(a, b)] for a, b in itertools.pairwise(witness)
                        ),
                    }
                ),
            )
        )

    for record in records:
        paths = _bfs_paths(record.report_id, succ)
        for target in sorted(paths):
            other = by_id.get(target)
            if other is None or other.group != record.group:
                continue
            if comp_of[target] == comp_of[record.report_id] and cyclic[comp_of[target]]:
                continue  # already reported as a citation cycle
            path = paths[target]
            flags.append(
                CounterintelligenceFlag(
                    kind=FlagKind.CIRCULAR_REPORTING,
                    report_ids=(record.report_id, target),
                    description=(
                        f"report {record.report_id} corroborates {target} but derives from it"
                    ),
                    evidence=MappingProxyType(
                        {
                            "pattern": "corroboration_loop",
                            "path": tuple(path),
                            "edge_kinds": tuple(
                                edge_kind[(a, b)] for a, b in itertools.pairwise(path)
                            ),
                        }
                    ),
                )
            )

    by_group: dict[str, list[ReportRecord]] = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)
    for group_key in sorted(by_group):
        reach: dict[str, list[ReportRecord]] = defaultdict(list)
        for record in by_group[group_key]:
            for origin in upstream_of[record.report_id]:
                if not origin.startswith("cycle:"):
                    reach[origin].append(record)
        for origin in sorted(reach):
            reporters = sorted(reach[origin], key=lambda r: r.report_id)
            source_keys = sorted({r.source_key for r in reporters})
            if len(source_keys) < 2:
                continue
            flags.append(
                CounterintelligenceFlag(
                    kind=FlagKind.SHARED_ORIGIN,
                    report_ids=tuple(r.report_id for r in reporters),
                    description=(
                        f"{len(source_keys)} sources presented as corroborating trace to "
                        f"the same upstream origin {origin}"
                    ),
                    evidence=MappingProxyType(
                        {
                            "origin": origin,
                            "group": group_key,
                            "source_keys": tuple(source_keys),
                        }
                    ),
                )
            )

    edges = tuple(sorted((a, b, kind) for (a, b), kind in edge_kind.items()))
    all_origins = {o for tokens in origins_of.values() for o in tokens}
    return IndependenceAnalysis(
        clusters=tuple(clusters),
        cluster_of=MappingProxyType(dict(sorted(cluster_of.items()))),
        origins_of=MappingProxyType(dict(sorted(origins_of.items()))),
        near_duplicates=near,
        edges=edges,
        cycles=tuple(sorted(cycles)),
        distinct_origins=len(all_origins),
        flags=tuple(flags),
    )


# --------------------------------------------------------------------------- #
# 3. Coordinated reporting: scan statistic with a Poisson null.
# --------------------------------------------------------------------------- #


def benjamini_hochberg_adjusted(p_values: Sequence[float]) -> tuple[float, ...]:
    """Benjamini-Hochberg (1995) adjusted p-values (q-values), in input order.

    ``q_(i) = min_{j >= i} min(1, m * p_(j) / j)`` over the ascending order.
    A hypothesis is rejected at FDR level ``alpha`` exactly when ``q <= alpha``.
    The decision itself is taken from the repository's
    ``evaluation.event_coincidence.benjamini_hochberg``.

    Args:
        p_values: Raw p-values in ``[0, 1]``.

    Returns:
        The adjusted p-values.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return ()
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return tuple(float(q) for q in adjusted)


@dataclass(frozen=True)
class ScanWindow:
    """The window holding the most distinct sources.

    Attributes:
        count: Distinct sources in the window, which is the scan statistic.
        start: Window start, in POSIX seconds (``None`` if there are no events).
        end: Start plus the window length.
        report_ids: Reports inside the window.
        source_keys: Distinct sources inside the window.
    """

    count: int
    start: float | None
    end: float | None
    report_ids: tuple[str, ...]
    source_keys: tuple[str, ...]


def scan_window(events: Sequence[tuple[float, str, str]], window_seconds: float) -> ScanWindow:
    """Maximum number of distinct sources in any closed window ``[t, t + w]``.

    A window holding the maximum can always be slid to start at one of its
    events, so only windows anchored at events are examined. Ties go to the
    earliest window.

    Args:
        events: ``(time_seconds, report_id, source_key)`` triples.
        window_seconds: Window length ``w > 0``.

    Returns:
        The :class:`ScanWindow`.

    Raises:
        ValueError: If ``window_seconds <= 0``.
    """
    _require(window_seconds > 0.0, "window_seconds must be positive")
    ordered = sorted(events)
    if not ordered:
        return ScanWindow(0, None, None, (), ())
    times = [event[0] for event in ordered]
    best = ScanWindow(0, None, None, (), ())
    for i, (start, _, _) in enumerate(ordered):
        stop = bisect.bisect_right(times, start + window_seconds)
        inside = ordered[i:stop]
        keys = tuple(sorted({event[2] for event in inside}))
        if len(keys) > best.count:
            best = ScanWindow(
                count=len(keys),
                start=start,
                end=start + window_seconds,
                report_ids=tuple(sorted(event[1] for event in inside)),
                source_keys=keys,
            )
    return best


def scan_statistic_pvalue(
    k: int, rate_per_hour: float, window_hours: float, observation_hours: float
) -> float:
    """Upper bound on ``P(S_w >= k)`` for a Poisson process of known rate.

    ``S_w`` is the most events in any window of length ``w`` within ``[0, T]``.
    If ``S_w >= k``, the first event of such a window has at least ``k - 1``
    further events within ``w``. Let ``M`` count events with that property. By
    the Mecke equation (Last & Penrose, 2017, ch. 4),
    ``E[M] = lambda T P(X >= k-1) - (k-1) P(X >= k)``, where
    ``X ~ Poisson(lambda w)``. The second term is the edge correction for
    windows truncated at ``T``. Markov's inequality then gives
    ``P(S_w >= k) <= P(M >= 1) <= E[M]``. The bound is conservative (a valid
    p-value), and it is tight in the tail where decisions are made. Naus's
    (1982) approximation is tighter and is not implemented.

    Args:
        k: Observed scan statistic.
        rate_per_hour: Baseline rate ``lambda`` under the null (> 0).
        window_hours: Window length ``w`` (> 0).
        observation_hours: Monitored period ``T``, raised to ``w`` if shorter.

    Returns:
        ``min(1, E[M])``. A count of 0 gives 1.

    Raises:
        ValueError: On a non-positive rate, window or period, or a negative ``k``.
    """
    _require(k >= 0, "k must be non-negative")
    _require(rate_per_hour > 0.0, "rate_per_hour must be positive")
    _require(window_hours > 0.0, "window_hours must be positive")
    _require(observation_hours > 0.0, "observation_hours must be positive")
    if k == 0:
        return 1.0
    period = max(observation_hours, window_hours)
    psi = rate_per_hour * window_hours
    m = k - 1
    expected = rate_per_hour * period * float(stats.poisson.sf(m - 1, psi)) - m * float(
        stats.poisson.sf(m, psi)
    )
    return float(min(1.0, max(0.0, expected)))


@dataclass(frozen=True)
class CoordinationTest:
    """A scan test of one claim for a coordinated burst.

    Attributes:
        claim: Claim key: the ``claim_id``, or ``"content:<id>"`` for a
            near-duplicate text component without one.
        applicable: Whether the test ran.
        reason: Why it ran or abstained.
        report_ids: Reports bearing on the claim.
        window: The maximal window.
        baseline_rate_per_hour: Null rate ``lambda``.
        window_hours: Window length ``w``.
        observation_hours: Period ``T`` used.
        p_value: Upper-bound p-value (:func:`scan_statistic_pvalue`).
        q_value: Benjamini-Hochberg adjusted p-value across tested claims.
        flagged: ``q <= fdr_alpha`` and at least ``coordination_min_sources``.
        near_duplicate_pairs: Near-duplicate pairs from distinct sources inside
            the window. This is content evidence and is not used in the p-value.
    """

    claim: str
    applicable: bool
    reason: str
    report_ids: tuple[str, ...]
    window: ScanWindow | None = None
    baseline_rate_per_hour: float | None = None
    window_hours: float | None = None
    observation_hours: float | None = None
    p_value: float | None = None
    q_value: float | None = None
    flagged: bool = False
    near_duplicate_pairs: tuple[tuple[str, str], ...] = ()


def detect_coordinated_reporting(
    records: Sequence[ReportRecord],
    near_duplicates: Sequence[NearDuplicatePair],
    baseline_rate_per_hour: Mapping[str, float],
    default_baseline_rate_per_hour: float | None = None,
    config: CounterintelligenceConfig | None = None,
) -> tuple[CoordinationTest, ...]:
    """Scan each claim for a burst of distinct sources against a Poisson baseline.

    Reports with a ``claim_id`` are grouped by it. Reports without one but with
    near-duplicate texts are grouped by text component, keyed
    ``"content:<smallest id>"``. A claim is tested only when it has timed
    reports and a baseline rate, either its own or the default. Benjamini-
    Hochberg is applied across all tested claims.

    Args:
        records: Normalised reports.
        near_duplicates: Near-duplicate pairs (from :func:`find_near_duplicates`).
        baseline_rate_per_hour: Null report rate per claim key.
        default_baseline_rate_per_hour: Rate for claims without their own.
        config: Parameters (defaults when ``None``).

    Returns:
        One :class:`CoordinationTest` per claim, sorted by claim key.
    """
    cfg = config or _DEFAULT_CONFIG
    groups: dict[str, list[ReportRecord]] = defaultdict(list)
    content = _UnionFind()
    unclaimed = {r.report_id: r for r in records if r.claim_id is None}
    linked: set[str] = set()
    for pair in near_duplicates:
        if pair.report_a in unclaimed and pair.report_b in unclaimed:
            content.union(pair.report_a, pair.report_b)
            linked.update((pair.report_a, pair.report_b))
    for record in records:
        if record.claim_id is not None:
            groups[record.claim_id].append(record)
        elif record.report_id in linked:
            groups[f"content:{content.find(record.report_id)}"].append(record)

    timed_all = [r.reported_at for r in records if r.reported_at is not None]
    window_hours = cfg.coordination_window_hours
    if cfg.observation_hours is not None:
        period = cfg.observation_hours
    elif timed_all:
        period = max(window_hours, (max(timed_all) - min(timed_all)) / _SECONDS_PER_HOUR)
    else:
        period = window_hours

    pending: list[CoordinationTest] = []
    for claim in sorted(groups):
        group = groups[claim]
        report_ids = tuple(sorted(r.report_id for r in group))
        events = [
            (r.reported_at, r.report_id, r.source_key) for r in group if r.reported_at is not None
        ]
        rate = baseline_rate_per_hour.get(claim, default_baseline_rate_per_hour)
        if not events:
            pending.append(
                CoordinationTest(claim, False, "no report carries a usable reported_at", report_ids)
            )
            continue
        if rate is None:
            pending.append(
                CoordinationTest(claim, False, "no baseline rate for this claim", report_ids)
            )
            continue
        window = scan_window(events, window_hours * _SECONDS_PER_HOUR)
        in_window = set(window.report_ids)
        source_of = {r.report_id: r.source_key for r in group}
        pairs = tuple(
            (p.report_a, p.report_b)
            for p in near_duplicates
            if p.report_a in in_window
            and p.report_b in in_window
            and source_of[p.report_a] != source_of[p.report_b]
        )
        pending.append(
            CoordinationTest(
                claim=claim,
                applicable=True,
                reason="tested",
                report_ids=report_ids,
                window=window,
                baseline_rate_per_hour=rate,
                window_hours=window_hours,
                observation_hours=period,
                p_value=scan_statistic_pvalue(window.count, rate, window_hours, period),
                near_duplicate_pairs=pairs,
            )
        )

    tested = [(i, t.p_value) for i, t in enumerate(pending) if t.p_value is not None]
    p_values = [p for _, p in tested]
    q_values = benjamini_hochberg_adjusted(p_values)
    reject = benjamini_hochberg(p_values, cfg.fdr_alpha)
    for position, (i, _) in enumerate(tested):
        test = pending[i]
        count = test.window.count if test.window is not None else 0
        pending[i] = replace(
            test,
            q_value=q_values[position],
            flagged=bool(reject[position]) and count >= cfg.coordination_min_sources,
        )
    return tuple(pending)


# --------------------------------------------------------------------------- #
# 4. Fabrication screens on numeric content.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScreenResult:
    """Outcome of one numeric fabrication screen.

    Attributes:
        test: ``"benford_first_digit"``, ``"terminal_digit"`` or ``"excess_rounding"``.
        applicable: Whether the screen's conditions held and it ran.
        reason: Why it ran or abstained.
        n: Values the statistic was computed on.
        statistic: Chi-square statistic, or for rounding the share of 0/5 digits.
        p_value: Upper-tail p-value.
        observed: Digit counts (1-9 for Benford, 0-9 for terminal digits).
        expected: Expected counts under the null.
        mad: Nigrini's mean absolute deviation (Benford only).
        conformity: Nigrini's conformity band (Benford only).
        excluded: Values excluded before testing (non-positive, too short, ...).
    """

    test: str
    applicable: bool
    reason: str
    n: int = 0
    statistic: float | None = None
    p_value: float | None = None
    observed: tuple[int, ...] = ()
    expected: tuple[float, ...] = ()
    mad: float | None = None
    conformity: str | None = None
    excluded: int = 0


def _finite_array(values: Iterable[float]) -> npt.NDArray[np.float64]:
    """Return the finite entries of ``values`` as float64."""
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    return np.asarray(arr[np.isfinite(arr)], dtype=np.float64)


def first_significant_digit(value: float) -> int:
    """First significant digit of a positive finite number.

    The digit is read from the shortest decimal string that round-trips
    (``repr``), so ``0.3`` gives 3. Its binary value, 0.29999..., would give 2.

    Args:
        value: A positive finite number.

    Returns:
        The digit, 1-9.

    Raises:
        ValueError: If ``value`` is not positive and finite.
    """
    number = float(value)
    _require(math.isfinite(number) and number > 0.0, "value must be positive and finite")
    return int(next(ch for ch in repr(number) if ch in "123456789"))


def nigrini_conformity(mad: float) -> str:
    """Nigrini (2012) first-digit conformity band for a MAD (upper bounds inclusive)."""
    for bound, label in NIGRINI_FIRST_DIGIT_MAD_BOUNDS:
        if mad <= bound:
            return label
    return NIGRINI_NONCONFORMITY


def benford_first_digit_test(
    values: Iterable[float], *, min_n: int = 110, min_orders_of_magnitude: float = 2.0
) -> ScreenResult:
    """Benford first-digit test: chi-square (8 df) and Nigrini's MAD.

    Only positive values are used. Non-positive and non-finite values are
    counted in ``excluded``. The test abstains unless at least ``min_n``
    positive values span at least ``min_orders_of_magnitude`` orders of
    magnitude. Nigrini (2012) prefers the MAD to chi-square on large data,
    because chi-square then rejects for immaterial deviations. Both are
    reported.

    Args:
        values: The numbers.
        min_n: Minimum count (default 110: every expected count is at least 5).
        min_orders_of_magnitude: Minimum ``log10(max / min)``.

    Returns:
        The :class:`ScreenResult`.
    """
    raw = np.asarray(list(values), dtype=np.float64).ravel()
    positive = raw[np.isfinite(raw) & (raw > 0.0)]
    excluded = int(raw.size - positive.size)
    name = "benford_first_digit"
    n = int(positive.size)
    if n < min_n:
        return ScreenResult(
            name,
            False,
            f"{n} positive values; at least {min_n} are needed",
            n=n,
            excluded=excluded,
        )
    span = float(np.log10(positive.max() / positive.min()))
    if span < min_orders_of_magnitude:
        return ScreenResult(
            name,
            False,
            f"values span {span:.4g} orders of magnitude; at least "
            f"{min_orders_of_magnitude:g} are needed",
            n=n,
            excluded=excluded,
        )
    digits = np.array([first_significant_digit(v) for v in positive], dtype=np.int64)
    observed = np.bincount(digits, minlength=10)[1:10]
    probabilities = np.asarray(BENFORD_FIRST_DIGIT_PROBABILITIES)
    expected = n * probabilities
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    mad = float(np.mean(np.abs(observed / n - probabilities)))
    return ScreenResult(
        name,
        True,
        "tested",
        n=n,
        statistic=chi2,
        p_value=float(stats.chi2.sf(chi2, df=8)),
        observed=tuple(int(c) for c in observed),
        expected=tuple(float(e) for e in expected),
        mad=mad,
        conformity=nigrini_conformity(mad),
        excluded=excluded,
    )


def terminal_digits(
    values: Iterable[float], *, decimals: int = 0, min_units: int = 10
) -> tuple[npt.NDArray[np.int64], int, str | None]:
    """Last recorded digits of ``values`` recorded at ``decimals`` places.

    Args:
        values: The numbers. Only finite entries are used.
        decimals: Decimal places of the recording precision.
        min_units: Values below this many units of the last place are excluded
            (one-digit values have no separate terminal digit).

    Returns:
        ``(digits, n_excluded, refusal)``. ``refusal`` is a reason string when
        the values are not all recorded at that precision, or are too large for
        an exact float64 digit. ``digits`` is then empty.
    """
    finite = _finite_array(values)
    empty = np.zeros(0, dtype=np.int64)
    scaled = np.abs(finite) * 10.0**decimals
    if np.any(scaled >= _EXACT_INTEGER_LIMIT):
        return empty, 0, "values too large for an exact float64 terminal digit"
    nearest = np.rint(scaled)
    if np.any(np.abs(scaled - nearest) > _PRECISION_TOLERANCE):
        return (
            empty,
            0,
            f"values are not all recorded at {decimals} decimal place(s); "
            "terminal digits are undefined",
        )
    keep = nearest >= min_units
    digits = np.mod(nearest[keep], 10).astype(np.int64)
    return digits, int(np.count_nonzero(~keep)), None


def terminal_digit_test(
    values: Iterable[float], *, decimals: int = 0, min_units: int = 10, min_n: int = 50
) -> ScreenResult:
    """Chi-square test (9 df) that terminal digits are uniform on 0-9.

    For a quantity measured with spread much larger than its last recorded
    unit, the terminal digit is close to uniform (Preece, 1981). Invented
    numbers tend to depart from that (Mosimann, Wiseman & Edelman, 1995).

    Args:
        values: The numbers.
        decimals: Decimal places of the recording precision.
        min_units: See :func:`terminal_digits`.
        min_n: Minimum count (default 50: expected count 5 per digit).

    Returns:
        The :class:`ScreenResult`.
    """
    name = "terminal_digit"
    digits, excluded, refusal = terminal_digits(values, decimals=decimals, min_units=min_units)
    if refusal is not None:
        return ScreenResult(name, False, refusal)
    n = int(digits.size)
    if n < min_n:
        return ScreenResult(
            name, False, f"{n} usable values; at least {min_n} are needed", n=n, excluded=excluded
        )
    observed = np.bincount(digits, minlength=10)
    expected = np.full(10, n / 10.0)
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    return ScreenResult(
        name,
        True,
        "tested",
        n=n,
        statistic=chi2,
        p_value=float(stats.chi2.sf(chi2, df=9)),
        observed=tuple(int(c) for c in observed),
        expected=tuple(float(e) for e in expected),
        excluded=excluded,
    )


def excess_rounding_test(
    values: Iterable[float], *, decimals: int = 0, min_units: int = 10, min_n: int = 25
) -> ScreenResult:
    """Exact one-sided binomial test for an excess of terminal digits 0 and 5.

    Under uniform terminal digits ``P(0 or 5) = 0.2``. An excess is what
    estimated or invented numbers look like, so the test cannot tell the two
    apart.

    Args:
        values: The numbers.
        decimals: Decimal places of the recording precision.
        min_units: See :func:`terminal_digits`.
        min_n: Minimum count (default 25: expected count 5 of rounded values).

    Returns:
        The :class:`ScreenResult`. ``statistic`` is the observed share.
    """
    name = "excess_rounding"
    digits, excluded, refusal = terminal_digits(values, decimals=decimals, min_units=min_units)
    if refusal is not None:
        return ScreenResult(name, False, refusal)
    n = int(digits.size)
    if n < min_n:
        return ScreenResult(
            name, False, f"{n} usable values; at least {min_n} are needed", n=n, excluded=excluded
        )
    rounded = int(np.count_nonzero(np.isin(digits, _ROUNDED_DIGITS)))
    result = stats.binomtest(rounded, n, _ROUNDED_SHARE, alternative="greater")
    return ScreenResult(
        name,
        True,
        "tested",
        n=n,
        statistic=rounded / n,
        p_value=float(result.pvalue),
        observed=(rounded, n - rounded),
        expected=(n * _ROUNDED_SHARE, n * (1.0 - _ROUNDED_SHARE)),
        excluded=excluded,
    )


@dataclass(frozen=True)
class FabricationScreen:
    """The three numeric screens for one unit (a report, or a pooled source).

    Attributes:
        unit: ``"report:<id>"`` or ``"source:<id>"``.
        report_ids: Reports whose values were screened.
        results: Benford, terminal-digit and rounding results, in that order.
    """

    unit: str
    report_ids: tuple[str, ...]
    results: tuple[ScreenResult, ...]


def screen_fabrication(
    records: Sequence[ReportRecord], config: CounterintelligenceConfig | None = None
) -> tuple[FabricationScreen, ...]:
    """Run the numeric screens per report, or per source when configured.

    Args:
        records: Normalised reports. Those without numeric values are skipped.
        config: Parameters (defaults when ``None``).

    Returns:
        One :class:`FabricationScreen` per unit, sorted by unit.
    """
    cfg = config or _DEFAULT_CONFIG
    units: dict[str, list[ReportRecord]] = defaultdict(list)
    for record in records:
        if not record.numeric_values:
            continue
        if cfg.fabrication_unit == "source" and record.source_id is not None:
            units[f"source:{record.source_id}"].append(record)
        else:
            units[f"report:{record.report_id}"].append(record)
    screens: list[FabricationScreen] = []
    for unit in sorted(units):
        group = units[unit]
        values = [v for record in group for v in record.numeric_values]
        screens.append(
            FabricationScreen(
                unit=unit,
                report_ids=tuple(sorted(r.report_id for r in group)),
                results=(
                    benford_first_digit_test(
                        values,
                        min_n=cfg.benford_min_n,
                        min_orders_of_magnitude=cfg.benford_min_orders_of_magnitude,
                    ),
                    terminal_digit_test(
                        values,
                        decimals=cfg.digit_decimals,
                        min_units=cfg.terminal_min_units,
                        min_n=cfg.terminal_digit_min_n,
                    ),
                    excess_rounding_test(
                        values,
                        decimals=cfg.digit_decimals,
                        min_units=cfg.terminal_min_units,
                        min_n=cfg.rounding_min_n,
                    ),
                ),
            )
        )
    return tuple(screens)


def _fabrication_flags(
    screens: Sequence[FabricationScreen], alpha: float
) -> tuple[CounterintelligenceFlag, ...]:
    """BH across every applicable (unit, screen) p-value, then flag units.

    A Benford rejection counts only with Nigrini nonconformity
    (MAD > 0.015): statistical significance alone can reflect a deviation
    too small to matter (Nigrini, 2012).
    """
    tested = [
        (si, result, result.p_value)
        for si, screen in enumerate(screens)
        for result in screen.results
        if result.applicable and result.p_value is not None
    ]
    p_values = [p for _, _, p in tested]
    q_values = benjamini_hochberg_adjusted(p_values)
    reject = benjamini_hochberg(p_values, alpha)
    fired: dict[int, list[tuple[ScreenResult, float, float]]] = defaultdict(list)
    for position, (si, result, p) in enumerate(tested):
        if not reject[position]:
            continue
        if result.test == "benford_first_digit" and result.conformity != NIGRINI_NONCONFORMITY:
            continue
        fired[si].append((result, p, q_values[position]))
    flags: list[CounterintelligenceFlag] = []
    for si in sorted(fired):
        screen = screens[si]
        hits = fired[si]
        strongest = min(hits, key=lambda hit: (hit[1], hit[0].test))
        flags.append(
            CounterintelligenceFlag(
                kind=FlagKind.FABRICATION,
                report_ids=screen.report_ids,
                description=(
                    f"{screen.unit}: numeric content fails "
                    + ", ".join(result.test for result, _, _ in hits)
                ),
                evidence=MappingProxyType(
                    {
                        "unit": screen.unit,
                        "screens": tuple(
                            MappingProxyType(
                                {
                                    "test": result.test,
                                    "n": result.n,
                                    "statistic": result.statistic,
                                    "p_value": p,
                                    "q_value": q,
                                    "mad": result.mad,
                                    "conformity": result.conformity,
                                }
                            )
                            for result, p, q in hits
                        ),
                    }
                ),
                p_value=strongest[1],
                q_value=strongest[2],
            )
        )
    return tuple(flags)


# --------------------------------------------------------------------------- #
# 5. Impossible travel.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrackPoint:
    """A timed location fix for one entity.

    Attributes:
        report_id: Report the fix came from.
        entity: Entity or source the fix locates.
        time_s: POSIX seconds.
        lat: Latitude, decimal degrees.
        lon: Longitude, decimal degrees.
        radius_km: Location uncertainty radius, km.
    """

    report_id: str
    entity: str
    time_s: float
    lat: float
    lon: float
    radius_km: float = 0.0


@dataclass(frozen=True)
class TravelViolation:
    """Two consecutive fixes that imply a speed above the bound.

    Attributes:
        entity: Entity or source.
        report_a: Earlier fix.
        report_b: Later fix.
        time_a: Earlier time, POSIX seconds.
        time_b: Later time, POSIX seconds.
        distance_km: Great-circle distance between the fix centres.
        effective_distance_km: Distance minus both radii, floored at 0.
        elapsed_hours: ``(time_b - time_a) / 3600``.
        implied_speed_kmh: ``effective_distance_km / elapsed_hours``. It is
            infinite when no time elapsed and the uncertainty discs do not
            overlap.
        max_speed_kmh: The bound that was exceeded.
    """

    entity: str
    report_a: str
    report_b: str
    time_a: float
    time_b: float
    distance_km: float
    effective_distance_km: float
    elapsed_hours: float
    implied_speed_kmh: float
    max_speed_kmh: float


def impossible_travel(
    points: Sequence[TrackPoint], *, max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH
) -> tuple[TravelViolation, ...]:
    """Flag consecutive fixes per entity whose implied speed exceeds the bound.

    Fixes are ordered by ``(time, report_id)`` within each entity. Distances use
    the repository's haversine kernel (``utils.geo.haversine_km``, IUGG mean
    radius). Both uncertainty radii are subtracted before the speed is computed.
    Only consecutive pairs are checked, so a large middle radius can mask a
    violation between non-adjacent fixes. Timestamp error is not modelled.

    Args:
        points: Fixes, in any order.
        max_speed_kmh: Speed bound. A pair is flagged when strictly above it.

    Returns:
        Violations, sorted by entity and time.

    Raises:
        ValueError: If ``max_speed_kmh <= 0``.
    """
    _require(max_speed_kmh > 0.0, "max_speed_kmh must be positive")
    tracks: dict[str, list[TrackPoint]] = defaultdict(list)
    for point in points:
        tracks[point.entity].append(point)
    violations: list[TravelViolation] = []
    for entity in sorted(tracks):
        track = sorted(tracks[entity], key=lambda p: (p.time_s, p.report_id))
        for a, b in itertools.pairwise(track):
            distance = haversine_km(a.lat, a.lon, b.lat, b.lon)
            effective = max(0.0, distance - a.radius_km - b.radius_km)
            hours = (b.time_s - a.time_s) / _SECONDS_PER_HOUR
            if hours > 0.0:
                speed = effective / hours
            else:
                speed = math.inf if effective > 0.0 else 0.0
            if speed > max_speed_kmh:
                violations.append(
                    TravelViolation(
                        entity=entity,
                        report_a=a.report_id,
                        report_b=b.report_id,
                        time_a=a.time_s,
                        time_b=b.time_s,
                        distance_km=distance,
                        effective_distance_km=effective,
                        elapsed_hours=hours,
                        implied_speed_kmh=speed,
                        max_speed_kmh=max_speed_kmh,
                    )
                )
    return tuple(violations)


def track_points(records: Sequence[ReportRecord]) -> tuple[TrackPoint, ...]:
    """Extract fixes from reports that carry a time, a location and an entity or source.

    Args:
        records: Normalised reports.

    Returns:
        One :class:`TrackPoint` per usable report. The entity is ``entity_id``,
        else ``source_id``.
    """
    points: list[TrackPoint] = []
    for record in records:
        entity = record.entity_id or record.source_id
        if entity is None or record.location is None or record.reported_at is None:
            continue
        points.append(
            TrackPoint(
                report_id=record.report_id,
                entity=entity,
                time_s=record.reported_at,
                lat=record.location.lat,
                lon=record.location.lon,
                radius_km=record.location.radius_km,
            )
        )
    return tuple(points)


# --------------------------------------------------------------------------- #
# 6. Training-data poisoning: spectral signatures.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpectralClassResult:
    """Spectral-signature screen of one class.

    Attributes:
        label: The class label.
        n: Samples in the class.
        n_flagged: Samples flagged (``ceil(removal_fraction * n)``).
        applicable: False when the class has fewer than 2 samples or no variance.
        reason: Why the class was or was not screened.
        top_singular_value: Largest singular value of the centred class matrix.
    """

    label: int | str
    n: int
    n_flagged: int
    applicable: bool
    reason: str
    top_singular_value: float


@dataclass(frozen=True, eq=False)
class SpectralSignatureResult:
    """Outlier scores and flags from spectral signatures (read-only arrays).

    Attributes:
        scores: Per-sample squared projection on its class's top singular
            vector. Every score is returned for audit.
        flagged: Per-sample flag (the top ``removal_fraction`` of each class).
        epsilon: Assumed poisoning rate.
        removal_fraction: ``min(1, removal_factor * epsilon)``.
        per_class: Per-class summaries, sorted by label.
    """

    scores: npt.NDArray[np.float64]
    flagged: npt.NDArray[np.bool_]
    epsilon: float
    removal_fraction: float
    per_class: tuple[SpectralClassResult, ...]


def spectral_signature_scores(
    representations: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    epsilon: float,
    removal_factor: float = 1.5,
) -> SpectralSignatureResult:
    """Spectral-signature poisoning screen (Tran, Li & Madry, NeurIPS 2018).

    For each class, the representations are centred, and the top right singular
    vector ``v`` of the centred matrix is taken. Each sample is scored
    ``((r_i - mean) . v) ** 2``. The top ``removal_factor * epsilon`` fraction
    of each class is flagged, rounded up. The default factor 1.5 is the paper's
    Algorithm 1. Every class gets the same fraction, so a clean class still has
    samples flagged. The screen detects a backdoor only when the poisoned
    points separate along one direction, as the paper's analysis requires.

    Args:
        representations: ``(n, d)`` learned representations, for example
            penultimate-layer activations.
        labels: ``(n,)`` class labels (integers or strings).
        epsilon: Assumed poisoning rate, in ``(0, 1)``. There is no default:
            the caller must state the assumption.
        removal_factor: Over-removal factor (paper default 1.5).

    Returns:
        The :class:`SpectralSignatureResult`.

    Raises:
        ValueError: On malformed shapes, non-finite values, or out-of-range
            ``epsilon`` / ``removal_factor``.
    """
    features = np.asarray(representations, dtype=np.float64)
    classes = np.asarray(labels)
    _require(features.ndim == 2, "representations must be a 2-D (n, d) array")
    _require(classes.ndim == 1, "labels must be a 1-D array")
    _require(features.shape[0] == classes.shape[0], "representations and labels differ in length")
    _require(bool(np.all(np.isfinite(features))), "representations must be finite")
    _require(0.0 < epsilon < 1.0, "epsilon must lie in (0, 1)")
    _require(removal_factor > 0.0, "removal_factor must be positive")
    fraction = min(1.0, removal_factor * epsilon)
    scores = np.zeros(features.shape[0], dtype=np.float64)
    flagged = np.zeros(features.shape[0], dtype=bool)
    per_class: list[SpectralClassResult] = []
    for label in np.unique(classes):
        idx = np.flatnonzero(classes == label)
        label_value: int | str = label.item() if hasattr(label, "item") else label
        if not isinstance(label_value, int | str):
            label_value = str(label_value)
        if idx.size < 2:
            per_class.append(
                SpectralClassResult(label_value, int(idx.size), 0, False, "fewer than 2", 0.0)
            )
            continue
        block = features[idx]
        centred = block - block.mean(axis=0)
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        top = float(singular[0])
        if top <= _SVD_RELATIVE_TOLERANCE * max(1.0, float(np.linalg.norm(block))):
            per_class.append(
                SpectralClassResult(label_value, int(idx.size), 0, False, "no variance", top)
            )
            continue
        class_scores = (centred @ vt[0]) ** 2
        scores[idx] = class_scores
        # Subtract a tolerance before ceil: 1.5 * 0.05 * 200 is 15.000000000000002.
        k = min(int(idx.size), math.ceil(fraction * idx.size - 1e-9))
        order = np.lexsort((idx, -class_scores))
        flagged[idx[order[:k]]] = True
        per_class.append(SpectralClassResult(label_value, int(idx.size), k, True, "screened", top))
    scores.setflags(write=False)
    flagged.setflags(write=False)
    return SpectralSignatureResult(
        scores=scores,
        flagged=flagged,
        epsilon=epsilon,
        removal_fraction=fraction,
        per_class=tuple(per_class),
    )


# --------------------------------------------------------------------------- #
# 7. Orchestration.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssessmentHistory:
    """Information from outside the batch.

    Attributes:
        outcomes: Source id to adjudicated outcomes, oldest first.
        baseline_rate_per_hour: Claim key to the null report rate.
        default_baseline_rate_per_hour: Null rate for claims without their own.
    """

    outcomes: Mapping[str, tuple[SourceOutcome, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    baseline_rate_per_hour: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    default_baseline_rate_per_hour: float | None = None

    @classmethod
    def coerce(cls, value: object) -> tuple[AssessmentHistory, tuple[str, ...]]:
        """Build a history from an instance, a mapping with the same keys, or ``None``.

        Outcomes may be :class:`SourceOutcome`, bools, or mappings with
        ``confirmed`` and optional ``weight``. Invalid entries -- including a
        weight that is not finite or lies outside ``[0, MAX_OUTCOME_WEIGHT]``
        -- are dropped with a note, never guessed or clipped.

        Args:
            value: The history, in any accepted form.

        Returns:
            ``(history, notes)``.

        Raises:
            TypeError: If ``value`` is neither ``None``, a history nor a mapping.
        """
        if value is None:
            return cls(), ()
        if isinstance(value, AssessmentHistory):
            return value, ()
        if not isinstance(value, Mapping):
            raise TypeError(f"history must be an AssessmentHistory or mapping, got {value!r}")
        notes: list[str] = []
        outcomes: dict[str, tuple[SourceOutcome, ...]] = {}
        raw_outcomes = value.get("outcomes") or {}
        if not isinstance(raw_outcomes, Mapping):
            notes.append("history: outcomes ignored (not a mapping of source to outcomes)")
            raw_outcomes = {}
        for source in sorted(raw_outcomes, key=str):
            parsed: list[SourceOutcome] = []
            items = raw_outcomes[source]
            if not isinstance(items, list | tuple):
                notes.append(f"history: outcomes for source {source} ignored (not a list)")
                continue
            for item in items:
                try:
                    if isinstance(item, SourceOutcome):
                        parsed.append(item)
                    elif isinstance(item, bool | np.bool_):
                        parsed.append(SourceOutcome(bool(item)))
                    elif isinstance(item, Mapping):
                        parsed.append(SourceOutcome(item["confirmed"], item.get("weight", 1.0)))
                    else:
                        raise TypeError(type(item).__name__)
                except (KeyError, TypeError, ValueError) as exc:
                    notes.append(f"history: outcome for source {source} dropped ({exc})")
            outcomes[str(source)] = tuple(parsed)
        rates: dict[str, float] = {}
        raw_rates = value.get("baseline_rate_per_hour") or {}
        if not isinstance(raw_rates, Mapping):
            notes.append("history: baseline_rate_per_hour ignored (not a mapping)")
            raw_rates = {}
        for claim, rate in raw_rates.items():
            number = _as_float(rate)
            if number is None or number <= 0.0:
                notes.append(f"history: baseline rate for claim {claim} dropped (not > 0)")
            else:
                rates[str(claim)] = number
        default_raw = value.get("default_baseline_rate_per_hour")
        default = _as_float(default_raw)
        if default_raw is not None and (default is None or default <= 0.0):
            notes.append("history: default baseline rate dropped (not > 0)")
            default = None
        return (
            cls(
                outcomes=MappingProxyType(outcomes),
                baseline_rate_per_hour=MappingProxyType(rates),
                default_baseline_rate_per_hour=default,
            ),
            tuple(notes),
        )


@dataclass(frozen=True)
class CounterintelligenceAssessment:
    """Result of :meth:`CounterintelligenceAssessor.assess`.

    Attributes:
        adjusted_reliability: Discipline to pooled reliability. ``None`` when
            there is neither an engine prior nor source information.
        source_posteriors: Source id to :class:`SourcePosterior`.
        independence: Provenance analysis and independence clusters.
        flags: Every finding, ordered by kind then reports.
        fusion_weight_multipliers: Discipline to a multiplier in ``[0, 1]``.
            It is the largest multiplier among the discipline's reports.
        report_weight_multipliers: Report id to a multiplier in ``[0, 1]``. A
            report's multiplier is 1 for its cluster's representative and 0 for
            other members, times each applicable configured factor (circular,
            coordinated, fabrication, impossible travel).
        discipline_clusters: Discipline to the clusters of its reports.
        coordination_tests: Every claim's scan test, including abstentions.
        fabrication_screens: Every unit's numeric screens, including abstentions.
        travel_violations: Impossible-travel pairs.
        reports: The normalised reports.
        reasons: Every note, abstention and weight rule applied, as text.
    """

    adjusted_reliability: dict[str, float | None]
    source_posteriors: dict[str, SourcePosterior]
    independence: IndependenceAnalysis
    flags: tuple[CounterintelligenceFlag, ...]
    fusion_weight_multipliers: dict[str, float]
    report_weight_multipliers: dict[str, float]
    discipline_clusters: dict[str, tuple[str, ...]]
    coordination_tests: tuple[CoordinationTest, ...]
    fabrication_screens: tuple[FabricationScreen, ...]
    travel_violations: tuple[TravelViolation, ...]
    reports: tuple[ReportRecord, ...]
    reasons: tuple[str, ...]

    @property
    def clusters(self) -> tuple[IndependenceCluster, ...]:
        """The independence clusters."""
        return self.independence.clusters

    @property
    def effective_independent_sources(self) -> int:
        """Number of independence clusters."""
        return self.independence.effective_independent_sources

    def flags_of(self, kind: FlagKind) -> tuple[CounterintelligenceFlag, ...]:
        """Return the flags of one kind."""
        return tuple(flag for flag in self.flags if flag.kind is kind)

    def independent_corroboration(self, disciplines: Iterable[str] | None = None) -> int:
        """Count the distinct clusters behind ``disciplines`` that keep a non-zero weight.

        The fusion engine's corroboration factor should count this, not the
        number of disciplines.

        Args:
            disciplines: Disciplines to count over (all when ``None``).

        Returns:
            Distinct clusters with at least one report in ``disciplines`` and a
            positive representative multiplier.
        """
        wanted = set(self.discipline_clusters if disciplines is None else disciplines)
        live = {
            cluster.cluster_id
            for cluster in self.independence.clusters
            if self.report_weight_multipliers.get(cluster.representative, 0.0) > 0.0
        }
        counted = {
            cluster_id
            for discipline, cluster_ids in self.discipline_clusters.items()
            if discipline in wanted
            for cluster_id in cluster_ids
            if cluster_id in live
        }
        return len(counted)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping, for example to attach to a fusion result."""
        result: dict[str, Any] = _jsonable(self)
        return result


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, mappings, tuples, enums and arrays to JSON types."""
    if isinstance(value, StrEnum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


class CounterintelligenceAssessor:
    """Assess a fusion-engine report batch for deception and dependence.

    The assessor is stateless apart from its configuration and the engine's
    discipline priors. :meth:`assess` is a pure function of its arguments: it
    never mutates them, and it returns the same result for the same input.
    """

    def __init__(
        self,
        config: CounterintelligenceConfig | None = None,
        discipline_priors: Mapping[str, float] | None = None,
    ) -> None:
        """Create an assessor.

        Args:
            config: Parameters (defaults when ``None``).
            discipline_priors: Discipline to prior reliability in ``[0, 1]``,
                normally ``IntelligenceFusionEngine.int_reliability_scores``.

        Raises:
            ValueError: If a prior is not a finite number in ``[0, 1]``.
        """
        self.config = config or _DEFAULT_CONFIG
        priors: dict[str, float] = {}
        for discipline, value in (discipline_priors or {}).items():
            number = _as_float(value)
            if number is None or not 0.0 <= number <= 1.0:
                raise ValueError(
                    f"prior for {discipline!r} must be a number in [0, 1], got {value!r}"
                )
            priors[str(discipline)] = number
        self.discipline_priors: Mapping[str, float] = MappingProxyType(priors)

    @classmethod
    def from_engine(
        cls, engine: object, config: CounterintelligenceConfig | None = None
    ) -> CounterintelligenceAssessor:
        """Create an assessor seeded with a fusion engine's ``int_reliability_scores``.

        Args:
            engine: An ``IntelligenceFusionEngine``, or any object with that attribute.
            config: Parameters (defaults when ``None``).

        Returns:
            The assessor.
        """
        return cls(config=config, discipline_priors=getattr(engine, "int_reliability_scores", {}))

    def _posteriors(
        self, records: Sequence[ReportRecord], history: AssessmentHistory, reasons: list[str]
    ) -> dict[str, SourcePosterior]:
        """One posterior per stated source. Conflicting grades resolve to the least reliable."""
        grades: dict[str, set[str]] = defaultdict(set)
        for record in records:
            if record.source_id is not None:
                grades.setdefault(record.source_id, set())
                if record.source_grade is not None:
                    grades[record.source_id].add(record.source_grade)
        table = self.config.grade_prior_mean
        posteriors: dict[str, SourcePosterior] = {}
        for source in sorted(grades):
            stated = grades[source]
            judged = sorted(g for g in stated if table[g] is not None)
            grade: str | None
            if judged:
                grade = min(judged, key=lambda g: (table[g], g))
                if len(stated) > 1:
                    reasons.append(
                        f"source {source}: conflicting grades {sorted(stated)}; the least "
                        f"reliable judged grade {grade} is used"
                    )
            else:
                grade = "F" if "F" in stated else None
            posteriors[source] = source_posterior(
                source, grade, history.outcomes.get(source, ()), config=self.config
            )
        return posteriors

    def _adjusted_reliability(
        self,
        records: Sequence[ReportRecord],
        posteriors: Mapping[str, SourcePosterior],
        reasons: list[str],
    ) -> dict[str, float | None]:
        """Pool each discipline's engine prior with its sources' Beta pseudo-counts.

        The prior counts as ``discipline_prior_strength`` pseudo-observations at
        its mean. Each distinct source adds its
        :attr:`SourcePosterior.informative_counts`. The result is the pooled
        Beta mean. With no source information the prior is returned unchanged.
        """
        sources: dict[str, set[str]] = defaultdict(set)
        for record in records:
            sources.setdefault(record.discipline, set())
            if record.source_id is not None:
                sources[record.discipline].add(record.source_id)
        strength = self.config.discipline_prior_strength
        adjusted: dict[str, float | None] = {}
        for discipline in sorted(sources):
            prior = self.discipline_priors.get(discipline)
            alpha = beta = 0.0
            for source in sorted(sources[discipline]):
                add_alpha, add_beta = posteriors[source].informative_counts
                alpha += add_alpha
                beta += add_beta
            if alpha + beta == 0.0:
                adjusted[discipline] = prior
                if prior is None:
                    reasons.append(
                        f"discipline {discipline}: no engine prior and no source "
                        "information; reliability not adjusted"
                    )
                continue
            if prior is not None:
                alpha += strength * prior
                beta += strength * (1.0 - prior)
            adjusted[discipline] = alpha / (alpha + beta)
        return adjusted

    def assess(
        self, intel_reports: Mapping[str, Any], history: object = None
    ) -> CounterintelligenceAssessment:
        """Assess a report batch.

        Args:
            intel_reports: The fusion engine's input. It maps each discipline
                key to a report mapping, or to a list of report mappings. Every
                field this module reads is optional.
            history: ``None``, an :class:`AssessmentHistory`, or a mapping with
                its keys (``outcomes``, ``baseline_rate_per_hour``,
                ``default_baseline_rate_per_hour``).

        Returns:
            The frozen :class:`CounterintelligenceAssessment`.
        """
        cfg = self.config
        records, notes = normalize_reports(intel_reports)
        past, history_notes = AssessmentHistory.coerce(history)
        reasons: list[str] = [*notes, *history_notes]
        if not records:
            reasons.append("no reports to assess")

        posteriors = self._posteriors(records, past, reasons)
        adjusted = self._adjusted_reliability(records, posteriors, reasons)
        # Representatives are chosen by the reliability the fusion engine
        # weights with, never by self-reported time alone.
        independence = analyse_independence(
            records, cfg, {r.report_id: adjusted.get(r.discipline) for r in records}
        )
        coordination = detect_coordinated_reporting(
            records,
            independence.near_duplicates,
            past.baseline_rate_per_hour,
            past.default_baseline_rate_per_hour,
            cfg,
        )
        screens = screen_fabrication(records, cfg)
        violations = impossible_travel(track_points(records), max_speed_kmh=cfg.max_speed_kmh)

        flags: list[CounterintelligenceFlag] = list(independence.flags)
        for test in coordination:
            if not test.applicable:
                reasons.append(f"coordination: claim {test.claim} not tested ({test.reason})")
                continue
            if not test.flagged or test.window is None:
                continue
            flags.append(
                CounterintelligenceFlag(
                    kind=FlagKind.COORDINATED_REPORTING,
                    report_ids=test.window.report_ids,
                    description=(
                        f"claim {test.claim}: {test.window.count} distinct sources within "
                        f"{test.window_hours:g} h against a baseline of "
                        f"{test.baseline_rate_per_hour:g} reports/h"
                    ),
                    evidence=MappingProxyType(
                        {
                            "claim": test.claim,
                            "window_start": test.window.start,
                            "window_end": test.window.end,
                            "source_keys": test.window.source_keys,
                            "baseline_rate_per_hour": test.baseline_rate_per_hour,
                            "observation_hours": test.observation_hours,
                            "near_duplicate_pairs": test.near_duplicate_pairs,
                        }
                    ),
                    p_value=test.p_value,
                    q_value=test.q_value,
                )
            )
        for screen in screens:
            for result in screen.results:
                if not result.applicable:
                    reasons.append(f"{result.test}: {screen.unit} not tested ({result.reason})")
        flags.extend(_fabrication_flags(screens, cfg.fdr_alpha))
        for violation in violations:
            flags.append(
                CounterintelligenceFlag(
                    kind=FlagKind.IMPOSSIBLE_TRAVEL,
                    report_ids=(violation.report_a, violation.report_b),
                    description=(
                        f"{violation.entity}: {violation.effective_distance_km:.1f} km beyond "
                        f"the uncertainty radii in {violation.elapsed_hours:.3g} h exceeds "
                        f"{violation.max_speed_kmh:g} km/h"
                    ),
                    evidence=MappingProxyType(_jsonable(violation)),
                )
            )
        flags.sort(key=lambda f: (_FLAG_ORDER[f.kind], f.report_ids, f.description))

        cycle_members = {rid for cycle in independence.cycles for rid in cycle}
        penalised: dict[str, list[tuple[str, float]]] = defaultdict(list)
        factor_of = {
            FlagKind.COORDINATED_REPORTING: ("coordinated", cfg.coordinated_factor),
            FlagKind.FABRICATION: ("fabrication", cfg.fabrication_factor),
            FlagKind.IMPOSSIBLE_TRAVEL: ("impossible travel", cfg.impossible_travel_factor),
        }
        for rid in sorted(cycle_members):
            penalised[rid].append(("citation cycle", cfg.circular_factor))
        for flag in flags:
            if flag.kind in factor_of:
                for rid in flag.report_ids:
                    if factor_of[flag.kind] not in penalised[rid]:
                        penalised[rid].append(factor_of[flag.kind])

        representative_of = {c.cluster_id: c.representative for c in independence.clusters}
        report_multipliers: dict[str, float] = {}
        for record in records:
            rid = record.report_id
            cluster_id = independence.cluster_of[rid]
            representative = representative_of[cluster_id]
            multiplier = 1.0 if representative == rid else 0.0
            if multiplier == 0.0:
                reasons.append(
                    f"report {rid}: weight 0, not independent of {representative}, which "
                    f"represents {cluster_id}"
                )
            for label, factor in penalised.get(rid, []):
                multiplier *= factor
                reasons.append(f"report {rid}: weight x{factor:g} ({label} flag)")
            report_multipliers[rid] = multiplier

        discipline_multipliers: dict[str, float] = {}
        discipline_clusters: dict[str, set[str]] = defaultdict(set)
        for record in records:
            disc = record.discipline
            discipline_multipliers[disc] = max(
                discipline_multipliers.get(disc, 0.0), report_multipliers[record.report_id]
            )
            discipline_clusters[disc].add(independence.cluster_of[record.report_id])

        clusters_by_group: dict[str, set[str]] = defaultdict(set)
        for record in records:
            clusters_by_group[record.group].add(independence.cluster_of[record.report_id])
        for record in records:
            if record.information_credibility == 1:
                own = independence.cluster_of[record.report_id]
                if not clusters_by_group[record.group] - {own}:
                    reasons.append(
                        f"report {record.report_id}: credibility 1 (confirmed by other "
                        "sources) has no independent corroboration in this batch"
                    )
        by_id = {record.report_id: record for record in records}
        for a, b, kind in independence.edges:
            if kind != "citation" or a not in by_id or b not in by_id:
                continue
            t_a, t_b = by_id[a].reported_at, by_id[b].reported_at
            if t_a is not None and t_b is not None and t_b > t_a:
                reasons.append(f"report {a} cites {b}, which is timestamped later")

        return CounterintelligenceAssessment(
            adjusted_reliability=adjusted,
            source_posteriors=posteriors,
            independence=independence,
            flags=tuple(flags),
            fusion_weight_multipliers=dict(sorted(discipline_multipliers.items())),
            report_weight_multipliers=dict(sorted(report_multipliers.items())),
            discipline_clusters={
                disc: tuple(sorted(ids)) for disc, ids in sorted(discipline_clusters.items())
            },
            coordination_tests=coordination,
            fabrication_screens=screens,
            travel_violations=violations,
            reports=records,
            reasons=tuple(reasons),
        )


__all__ = [
    "ADMIRALTY_GRADES",
    "ADMIRALTY_RELIABILITY_DESCRIPTORS",
    "ADMIRALTY_RELIABILITY_PRIOR_MEAN",
    "BENFORD_FIRST_DIGIT_PROBABILITIES",
    "DEFAULT_LOG_ODDS_CLIP",
    "DEFAULT_MAX_SPEED_KMH",
    "INFORMATION_CREDIBILITY_DESCRIPTORS",
    "MAX_OUTCOME_WEIGHT",
    "MINHASH_PRIME",
    "NIGRINI_FIRST_DIGIT_MAD_BOUNDS",
    "NIGRINI_NONCONFORMITY",
    "AssessmentHistory",
    "CoordinationTest",
    "CounterintelligenceAssessment",
    "CounterintelligenceAssessor",
    "CounterintelligenceConfig",
    "CounterintelligenceFlag",
    "FabricationScreen",
    "FlagKind",
    "GeoFix",
    "IndependenceAnalysis",
    "IndependenceCluster",
    "MinHasher",
    "NearDuplicatePair",
    "OutcomeWeightFit",
    "ReportRecord",
    "ScanWindow",
    "ScreenResult",
    "SourceOutcome",
    "SourcePosterior",
    "SpectralClassResult",
    "SpectralSignatureResult",
    "TrackPoint",
    "TravelViolation",
    "admiralty_prior",
    "analyse_independence",
    "benford_first_digit_test",
    "benjamini_hochberg_adjusted",
    "detect_coordinated_reporting",
    "excess_rounding_test",
    "find_near_duplicates",
    "first_significant_digit",
    "fit_outcome_weight_cap",
    "impossible_travel",
    "jaccard_similarity",
    "nigrini_conformity",
    "normalize_reports",
    "parse_timestamp",
    "scan_statistic_pvalue",
    "scan_window",
    "screen_fabrication",
    "source_posterior",
    "spectral_signature_scores",
    "terminal_digit_test",
    "terminal_digits",
    "track_points",
    "word_shingles",
]
