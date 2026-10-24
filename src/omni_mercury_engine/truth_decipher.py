# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evidence-grounded statement verification and a consent-gated Concealed Information Test.

Scientific framing. Comparison-question polygraph testing lacks adequate
validity as a way to tell whether a person's account is accurate (National
Research Council, 2003). This module performs no such classification. Every
output reports an evidence-grounded finding with a probability or a p-value,
names what it measures, and abstains when the evidence is insufficient. No
output labels a person or an account.

The methods, and the source for each:

1. Corroboration against independent evidence. Each claim gets a posterior
   probability from Bayesian log-odds: the prior log-odds plus, for each
   independence cluster of evidence, the log likelihood ratio of the
   cluster's strongest item (Good, 1950; Kass & Raftery, 1995). For an item
   of reliability ``r`` the likelihood model is
   ``P(supporting report | claim holds) = r`` and
   ``P(supporting report | claim does not hold) = 1 - r``. A supporting item
   therefore has ``log LR = log(r / (1 - r))`` and a refuting item the
   negative of that. Reliabilities are clipped to ``[eps, 1 - eps]``
   (Cromwell's rule; Lindley, 1985), so that no single source can settle a
   claim. Only the strongest item in a cluster counts. Items that share an
   origin are not conditionally independent, and summing them would count one
   observation more than once (Schum, 1994).
2. Contradictions. Claims that share a subject and predicate and refer to
   overlapping times are compared by value, using an absolute and relative
   numeric tolerance or categorical equality. Claimed orderings of events are
   checked against evidence times with Allen's interval algebra (Allen, 1983).
   Two located observations of one subject are checked for impossible travel
   with the haversine great-circle distance.
3. Consistency over time. The module lists contradictions within and between
   statements, and the changes between a speaker's successive statements. The
   Strategic Use of Evidence (SUE) indicator compares the statement-evidence
   inconsistency rate for evidence withheld before a statement with the rate
   for evidence disclosed before it. It also counts later statements that
   change to match newly disclosed evidence (Hartwig et al., 2006; Granhag &
   Hartwig, 2015). Counts come with Wilson score intervals (Wilson, 1927;
   Brown, Cai & DasGupta, 2001) and no verdict.
4. Verifiability. The proportion of details annotated as verifiable, with a
   Wilson interval (the Verifiability Approach; Nahari, Vrij & Fisher, 2014).
   A z-score and a percentile are given only against a reference distribution
   that the caller supplies, because norms are specific to each context.
5. Concealed Information Test (CIT) (Lykken, 1959; meta-analyses by
   Ben-Shakhar & Elaad, 2003, and Meijer et al., 2014). The CIT detects
   recognition of event details, not the accuracy of any statement. Responses
   are standardised within subject per channel and question (Ben-Shakhar,
   1985), signed so that a larger value means a stronger response, and
   combined across channels. Two tests are run: an exact rank test and an
   item-relabelling permutation test (Fisher, 1935; Phipson & Smyth, 2010).
   The test runs only with a consent record and abstains when the probe
   details were publicly disclosed. A finding of recognition is not evidence
   of guilt.

Limitations. Reliabilities, priors, tolerances, the maximum travel speed, the
CIT channel weights and the time-series windows are inputs or declared
defaults. None of them is a calibrated value. Independence clusters are only
as good as the caller's declarations, and undeclared dependence between items
inflates corroboration. Wilson intervals treat each counted pair as an
independent trial, but pairs that share a statement or an evidence item are
not independent, so those intervals are too narrow. The SUE and
Verifiability Approach effects are group-level research findings. This module
reports their measures but does not validate them for an individual case. No
output has been validated on field data in this repository.

``StatementVerifier`` is the entry point. ``TruthDecipherFramework`` remains
as a name alias for it. The earlier five-phase anomaly framework of that name
was removed because its outputs were fabricated, and its API is not restored.
Source reliability, independence clusters and travel feasibility are
``typing.Protocol`` interfaces, so another module can supply its own models.
Corroboration statuses map onto :class:`intel.verifier_loop.ClaimStatus`, and
evidence items can carry an :class:`intel.provenance.Provenance` record that
corroboration merges weakest-wins. Both of those modules are imported lazily,
so importing this one needs only numpy and scipy.

References:
    Allen, J. F. (1983). Maintaining knowledge about temporal intervals.
    Communications of the ACM, 26(11), 832-843.
    Ben-Shakhar, G. (1985). Standardization within individuals: A simple
    method to neutralize individual differences in skin conductance.
    Psychophysiology, 22(3), 292-299.
    Ben-Shakhar, G., & Elaad, E. (2003). The validity of psychophysiological
    detection of information with the Guilty Knowledge Test: A meta-analytic
    review. Journal of Applied Psychology, 88(1), 131-151.
    Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for
    a binomial proportion. Statistical Science, 16(2), 101-133.
    Dawson, M. E., Schell, A. M., & Filion, D. L. (2017). The electrodermal
    system. In J. T. Cacioppo, L. G. Tassinary & G. G. Berntson (Eds.),
    Handbook of Psychophysiology (4th ed.). Cambridge University Press.
    Fisher, R. A. (1935). The Design of Experiments. Oliver & Boyd.
    Gamer, M., Verschuere, B., Crombez, G., & Vossel, G. (2008). Combining
    physiological measures in the detection of concealed information.
    Physiology & Behavior, 95(3), 333-340.
    Good, I. J. (1950). Probability and the Weighing of Evidence. Griffin.
    Granhag, P. A., & Hartwig, M. (2015). The Strategic Use of Evidence
    technique: A conceptual overview. In P. A. Granhag, A. Vrij &
    B. Verschuere (Eds.), Detecting Deception: Current Challenges and
    Cognitive Approaches. Wiley.
    Hartwig, M., Granhag, P. A., Stromwall, L. A., & Kronkvist, O. (2006).
    Strategic use of evidence during police interviews. Law and Human
    Behavior, 30(5), 603-619.
    ISO 2533:1975. Standard Atmosphere.
    Kass, R. E., & Raftery, A. E. (1995). Bayes factors. Journal of the
    American Statistical Association, 90(430), 773-795.
    Lindley, D. V. (1985). Making Decisions (2nd ed.). Wiley.
    Lykken, D. T. (1959). The GSR in the detection of guilt. Journal of
    Applied Psychology, 43(6), 385-388.
    Meijer, E. H., klein Selle, N., Elber, L., & Ben-Shakhar, G. (2014).
    Memory detection with the Concealed Information Test: A meta analysis of
    skin conductance, respiration, heart rate, and P300 data.
    Psychophysiology, 51(9), 879-904.
    Moritz, H. (2000). Geodetic Reference System 1980. Journal of Geodesy,
    74(1), 128-133.
    Nahari, G., Vrij, A., & Fisher, R. P. (2014). Exploiting liars' verbal
    strategies by examining the verifiability of details. Legal and
    Criminological Psychology, 19(2), 227-239.
    National Research Council (2003). The Polygraph and Lie Detection. The
    National Academies Press.
    Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be
    zero. Statistical Applications in Genetics and Molecular Biology, 9(1),
    Article 39.
    Schum, D. A. (1994). The Evidential Foundations of Probabilistic
    Reasoning. Wiley.
    Timm, H. W. (1982). Analyzing deception from respiration patterns.
    Journal of Police Science and Administration, 10(1), 47-51.
    Vilain, M. B. (1982). A system for reasoning about time. Proceedings of
    AAAI-82, 197-201.
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. Journal of the American Statistical Association,
    22(158), 209-212.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from itertools import combinations, pairwise, product
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from omni_mercury_engine.intel.provenance import Provenance
    from omni_mercury_engine.intel.verifier_loop import ClaimStatus, ClaimVerdict

# --------------------------------------------------------------------------- #
# Declared constants. Each one is cited or is a configurable default with its
# rationale stated next to it.
# --------------------------------------------------------------------------- #

#: CIT finding when the primary test's p-value is at or below alpha.
FINDING_INDICATED = "recognition of the probe details indicated"
#: CIT finding when the primary test's p-value is above alpha.
FINDING_NOT_INDICATED = "recognition of the probe details not indicated"

#: Scope a :class:`ConsentRecord` must include before a CIT is scored.
CIT_CONSENT_SCOPE = "concealed_information_test"

#: IUGG mean Earth radius R1 in kilometres (Moritz, 2000: 6 371 008.7714 m).
EARTH_MEAN_RADIUS_KM = 6371.0088

#: Declared default maximum travel speed (km/h) for the impossible-travel check.
#: Rationale: a commercial jet airliner cruises near Mach 0.85. The ISO 2533
#: standard atmosphere gives a speed of sound of 295.07 m/s at the 11 km
#: tropopause, so that cruise speed is about 903 km/h. A 1000 km/h limit
#: therefore flags only observation pairs that no commercial flight could
#: connect. It ignores airport and ground transit time, so it under-flags. Lower
#: it when air travel is ruled out.
DEFAULT_MAX_SPEED_KMH = 1000.0

#: Declared default prior probability that a claim holds (indifference). The
#: posterior depends on it, so callers with a justified base rate should set it.
DEFAULT_PRIOR_PROBABILITY = 0.5

#: Declared default posterior thresholds for the corroboration status. With the
#: default prior of 0.5, a posterior of 0.95 requires a combined Bayes factor
#: of 19 (2 ln B = 5.9). That is the upper end of the "positive" band in Kass &
#: Raftery (1995). The thresholds are symmetric.
DEFAULT_CORROBORATED_AT = 0.95
DEFAULT_CONTRADICTED_AT = 0.05

#: Declared default reliability clip (Cromwell's rule; Lindley, 1985). A
#: reliability of exactly 0 or 1 would give an infinite likelihood ratio that
#: no contrary evidence could outweigh. Clipping to [0.001, 0.999] caps one
#: cluster's |log LR| at ln(999) = 6.91.
DEFAULT_RELIABILITY_EPSILON = 1e-3

#: Declared default two-sided confidence level for Wilson intervals.
DEFAULT_CONFIDENCE = 0.95

#: Declared default CIT significance level (the conventional 5% level).
DEFAULT_ALPHA = 0.05

#: Declared default CIT channel weights: equal. Gamer et al. (2008) show that
#: weights fitted on a calibration sample can outperform any single channel, so
#: a caller who has such a sample should supply fitted weights.
DEFAULT_CHANNEL_WEIGHT = 1.0

#: Declared default skin-conductance scoring window, in seconds after onset.
#: Dawson, Schell & Filion (2017) describe a specific skin-conductance
#: response as beginning 1-3 s after the stimulus. The window opens at 1 s, the
#: shortest of those latencies, and closes at 5 s to allow the rise to its
#: peak. The 5 s end is a declared default, not a cited value.
DEFAULT_SCR_WINDOW = (1.0, 5.0)
#: Declared default heart-rate baseline window, in seconds relative to onset.
DEFAULT_HR_BASELINE_WINDOW = (-3.0, 0.0)
#: Declared default heart-rate response window, in seconds relative to onset.
DEFAULT_HR_RESPONSE_WINDOW = (0.0, 10.0)
#: Declared default respiration line length window, in seconds relative to onset.
DEFAULT_RLL_WINDOW = (0.0, 10.0)
# The three windows above are chosen to fit an inter-stimulus interval of at
# least 10 s. They do not come from a published protocol, so set them to your
# protocol's scoring windows.

#: Declared default bound on exact permutation enumeration. Above it a seeded
#: Monte Carlo approximation is used.
DEFAULT_MAX_EXACT_PERMUTATIONS = 1_000_000
#: Declared default number of Monte Carlo relabellings.
DEFAULT_MONTE_CARLO_PERMUTATIONS = 10_000

#: Relative numerical floor below which a channel's within-question standard
#: deviation is treated as zero (the channel carries no information). This is
#: a floating-point guard, not a scientific threshold.
_ZERO_VARIANCE_RTOL = 1e-12

#: Text attached to every :class:`VerificationReport`.
SCIENTIFIC_FRAMING: tuple[str, ...] = (
    "This report aggregates evidence findings about statements; it does not "
    "classify any person or account.",
    "Comparison-question polygraph testing lacks adequate validity for judging "
    "the accuracy of accounts (National Research Council, 2003); no such "
    "physiological judgement is made here.",
    "Methods: claims checked against independent evidence (Bayesian log-odds over "
    "independence clusters); statement-evidence inconsistency under the Strategic "
    "Use of Evidence technique (Hartwig et al., 2006; Granhag & Hartwig, 2015); the "
    "proportion of verifiable details under the Verifiability Approach (Nahari, "
    "Vrij & Fisher, 2014).",
    "When evidence is insufficient the finding abstains: status 'unresolved' or an "
    "estimate of None.",
)

#: Limitations attached to every :class:`VerificationReport`.
VERIFIER_LIMITATIONS: tuple[str, ...] = (
    "Reliabilities, the prior, value tolerances and the maximum travel speed are "
    "inputs or declared defaults, not calibrated values.",
    "Independence clusters are taken from the caller; undeclared dependence between "
    "evidence items inflates corroboration.",
    "A contradiction may reflect error, ambiguity, imprecise wording or new knowledge; "
    "it does not establish intent.",
    "Wilson intervals treat each counted pair as an independent trial; pairs sharing "
    "a statement or evidence item are not, so the intervals are too narrow.",
    "The SUE and Verifiability Approach effects are group-level research findings; "
    "these measures are not validated for individual cases, and none of the outputs "
    "has been validated on field data in this repository.",
)

CORROBORATION_MEASURES = (
    "Posterior probability that the claim holds given the supplied evidence, under "
    "a symmetric reliability likelihood model and the stated prior; it measures "
    "evidential support, not the speaker's intent."
)
CONTRADICTION_MEASURES = (
    "Logical incompatibility between two claims, or between claims and evidence "
    "(values at overlapping times, Allen interval relations, or required travel "
    "speed); it does not establish intent."
)
SUE_MEASURES = (
    "Statement-evidence inconsistency rate for evidence withheld before each "
    "statement versus evidence disclosed before it (Strategic Use of Evidence; "
    "Hartwig et al., 2006); descriptive counts with Wilson intervals."
)
SUE_NOTE = (
    "In SUE research, higher inconsistency with withheld than with disclosed evidence "
    "is associated with concealment at group level (Hartwig et al., 2006); it is "
    "reported here as counts, not as a classification of the speaker."
)
ADAPTATION_MEASURES = (
    "Of the claims contradicted by evidence newly disclosed between two successive "
    "statements, the share whose later version the same evidence supports; "
    "descriptive count with a Wilson interval."
)
VERIFIABILITY_MEASURES = (
    "Proportion of details annotated as verifiable (Verifiability Approach; Nahari, "
    "Vrij & Fisher, 2014); the annotation is a human coding judgement."
)
VERIFIABILITY_NO_NORM_NOTE = (
    "No reference distribution supplied: verifiability norms are context-specific, "
    "so no z-score or percentile is given."
)
CIT_MEASURES = (
    "Whether physiological responses to the probe item exceed those to the "
    "irrelevant items more than chance predicts, i.e. recognition of the probe "
    "details (Lykken, 1959; Ben-Shakhar & Elaad, 2003; Meijer et al., 2014). It "
    "does not measure whether any statement is accurate."
)
CIT_ASSUMPTIONS: tuple[str, ...] = (
    "The probe details are known only to people involved in the event and to the "
    "examiners, and were not disclosed through media, prior questioning or the test "
    "itself.",
    "To an examinee without knowledge of the event the probe is indistinguishable "
    "from the irrelevant items (equally plausible items, randomised order).",
    "A knowledgeable examinee encoded the probe details and still remembers them.",
    "Under the null hypothesis the items of a question are exchangeable and the "
    "questions are independent of one another.",
)
NOT_EVIDENCE_OF_GUILT = (
    "A finding of recognition is not evidence of guilt: recognition can arise from "
    "innocent exposure to the details, and the absence of recognition does not "
    "establish non-involvement."
)
CIT_LIMITATIONS: tuple[str, ...] = (
    "Most validity data come from laboratory mock-crime studies; validity under "
    "field conditions is less established (Ben-Shakhar & Elaad, 2003; Meijer et "
    "al., 2014).",
    "Countermeasures (deliberate responses to irrelevant items) can reduce the "
    "difference between probe and irrelevant responses.",
    "Channel weights and time-series scoring windows are declared defaults, not "
    "calibrated values.",
    "Measurement error in contactless (radio-frequency) heart-rate and respiration "
    "estimates is not modelled.",
)


class ConsentRequiredError(PermissionError):
    """A Concealed Information Test was requested without a valid consent record."""


class InsufficientDataError(ValueError):
    """A time-series window holds too few samples to measure a response."""


# --------------------------------------------------------------------------- #
# Small numeric helpers.
# --------------------------------------------------------------------------- #
def _finite(value: float, name: str) -> float:
    """Return ``value`` as a finite float, or raise."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, not bool")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return out


def _probability(value: float, name: str) -> float:
    """Return ``value`` as a float in ``[0, 1]``, or raise."""
    out = _finite(value, name)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}")
    return out


def _non_empty(value: str, name: str) -> str:
    """Return ``value`` if it is a non-blank string, or raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _expit(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    """Log-odds of a probability in ``(0, 1)``."""
    return math.log(p) - math.log1p(-p)


def _jsonable(obj: object) -> object:
    """Convert a result object tree into JSON-compatible builtins."""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, frozenset | set):
        return sorted((_jsonable(x) for x in obj), key=str)
    if isinstance(obj, tuple | list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# Proportions with Wilson score intervals.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proportion:
    """A count-based proportion with a Wilson score interval.

    Attributes:
        successes: Number of counted events.
        trials: Number of trials.
        estimate: ``successes / trials``; ``None`` (abstain) when ``trials == 0``.
        lower: Lower Wilson bound; ``None`` when ``trials == 0``.
        upper: Upper Wilson bound; ``None`` when ``trials == 0``.
        confidence: Two-sided confidence level of the interval.
    """

    successes: int
    trials: int
    estimate: float | None
    lower: float | None
    upper: float | None
    confidence: float


def wilson_interval(
    successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE
) -> Proportion:
    """Wilson score interval for a binomial proportion (Wilson, 1927).

    The Wilson interval has close to nominal coverage for small ``n`` and for
    proportions near 0 or 1, where the Wald interval does not (Brown, Cai &
    DasGupta, 2001).

    Args:
        successes: Number of events, ``0 <= successes <= trials``.
        trials: Number of trials (``0`` yields an abstaining proportion).
        confidence: Two-sided confidence level in ``(0, 1)``.

    Returns:
        The :class:`Proportion`.

    Raises:
        ValueError: On counts out of range or a confidence outside ``(0, 1)``.
    """
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("counts must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"need 0 <= successes <= trials; got {successes}/{trials}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
    if trials == 0:
        return Proportion(successes, trials, None, None, None, confidence)
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    n = float(trials)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return Proportion(
        successes, trials, p, max(0.0, centre - half), min(1.0, centre + half), confidence
    )


# --------------------------------------------------------------------------- #
# Time: intervals and Allen's interval algebra.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeInterval:
    """A closed time interval ``[start, end]`` in seconds on a common clock.

    Use one clock for every input, for example POSIX epoch seconds. A point in
    time is the degenerate interval ``start == end``.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        """Validate and normalise the bounds."""
        start = _finite(self.start, "start")
        end = _finite(self.end, "end")
        if start > end:
            raise ValueError(f"interval start {start} is after end {end}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @classmethod
    def at(cls, instant: float) -> TimeInterval:
        """Return the degenerate interval for one instant."""
        return cls(instant, instant)


class AllenRelation(StrEnum):
    """Allen's (1983) thirteen relations between two intervals ``a`` and ``b``."""

    BEFORE = "before"
    MEETS = "meets"
    OVERLAPS = "overlaps"
    STARTS = "starts"
    DURING = "during"
    FINISHES = "finishes"
    EQUALS = "equals"
    FINISHED_BY = "finished_by"
    CONTAINS = "contains"
    STARTED_BY = "started_by"
    OVERLAPPED_BY = "overlapped_by"
    MET_BY = "met_by"
    AFTER = "after"


_CONVERSE: Mapping[AllenRelation, AllenRelation] = MappingProxyType(
    {
        AllenRelation.BEFORE: AllenRelation.AFTER,
        AllenRelation.MEETS: AllenRelation.MET_BY,
        AllenRelation.OVERLAPS: AllenRelation.OVERLAPPED_BY,
        AllenRelation.STARTS: AllenRelation.STARTED_BY,
        AllenRelation.DURING: AllenRelation.CONTAINS,
        AllenRelation.FINISHES: AllenRelation.FINISHED_BY,
        AllenRelation.EQUALS: AllenRelation.EQUALS,
        AllenRelation.FINISHED_BY: AllenRelation.FINISHES,
        AllenRelation.CONTAINS: AllenRelation.DURING,
        AllenRelation.STARTED_BY: AllenRelation.STARTS,
        AllenRelation.OVERLAPPED_BY: AllenRelation.OVERLAPS,
        AllenRelation.MET_BY: AllenRelation.MEETS,
        AllenRelation.AFTER: AllenRelation.BEFORE,
    }
)

#: Relations under which two intervals share at least one instant of
#: activity. ``meets`` and ``met_by`` only touch at a boundary, which is
#: treated as a hand-over, not as overlap.
CONCURRENT_RELATIONS: frozenset[AllenRelation] = frozenset(
    {
        AllenRelation.OVERLAPS,
        AllenRelation.OVERLAPPED_BY,
        AllenRelation.STARTS,
        AllenRelation.STARTED_BY,
        AllenRelation.DURING,
        AllenRelation.CONTAINS,
        AllenRelation.FINISHES,
        AllenRelation.FINISHED_BY,
        AllenRelation.EQUALS,
    }
)


def allen_relation(a: TimeInterval, b: TimeInterval) -> AllenRelation:
    """Return the Allen relation of ``a`` to ``b`` (Allen, 1983).

    For proper intervals (``start < end``) the thirteen relations are jointly
    exhaustive and pairwise disjoint. Degenerate intervals (time points) use
    the same endpoint comparisons, checked in an order that yields Vilain's
    (1982) point-interval relations. A point at an interval's start
    ``starts`` it, a point at its end ``finishes`` it, and two equal points are
    ``equals``. ``meets`` and ``met_by`` occur only between proper intervals.

    Args:
        a: The first interval.
        b: The second interval.

    Returns:
        The relation ``a R b``.
    """
    if a.end < b.start:
        return AllenRelation.BEFORE
    if b.end < a.start:
        return AllenRelation.AFTER
    if a.start == b.start:
        if a.end == b.end:
            return AllenRelation.EQUALS
        return AllenRelation.STARTS if a.end < b.end else AllenRelation.STARTED_BY
    if a.end == b.end:
        return AllenRelation.FINISHES if a.start > b.start else AllenRelation.FINISHED_BY
    if a.end == b.start:
        return AllenRelation.MEETS
    if b.end == a.start:
        return AllenRelation.MET_BY
    if a.start > b.start and a.end < b.end:
        return AllenRelation.DURING
    if a.start < b.start and a.end > b.end:
        return AllenRelation.CONTAINS
    if a.start < b.start:
        return AllenRelation.OVERLAPS
    return AllenRelation.OVERLAPPED_BY


def converse(relation: AllenRelation) -> AllenRelation:
    """Return the converse relation ``S``, such that ``a R b`` holds iff ``b S a``."""
    return _CONVERSE[relation]


def converse_relations(relations: Iterable[AllenRelation]) -> frozenset[AllenRelation]:
    """Return the set of converses of ``relations``."""
    return frozenset(_CONVERSE[r] for r in relations)


#: Relations of a claim's interval to an evidence interval under which the
#: evidence covers the whole claimed interval. Only such evidence can support
#: a claim that something held over that interval: one instant inside a long
#: interval says nothing about the rest of it, so its likelihood ratio would be
#: overstated. Overlap is still enough to refute.
COVERED_RELATIONS: frozenset[AllenRelation] = frozenset(
    {AllenRelation.EQUALS, AllenRelation.STARTS, AllenRelation.DURING, AllenRelation.FINISHES}
)


def interval_covered(claim: TimeInterval, evidence: TimeInterval) -> bool:
    """Whether ``evidence`` spans the whole of ``claim`` (see :data:`COVERED_RELATIONS`)."""
    return allen_relation(claim, evidence) in COVERED_RELATIONS


def intervals_concurrent(a: TimeInterval | None, b: TimeInterval | None) -> bool:
    """Whether two intervals share time (``True`` when either is unknown).

    A claim without an interval is treated as timeless, so it is concurrent
    with every other claim about the same subject and predicate.
    """
    if a is None or b is None:
        return True
    return allen_relation(a, b) in CONCURRENT_RELATIONS


# --------------------------------------------------------------------------- #
# Space: locations, haversine distance and the travel-feasibility interface.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Location:
    """A circular location: centre in decimal degrees and an uncertainty radius in km."""

    lat: float
    lon: float
    radius_km: float = 0.0

    def __post_init__(self) -> None:
        """Validate the coordinates and radius."""
        lat = _finite(self.lat, "lat")
        lon = _finite(self.lon, "lon")
        radius = _finite(self.radius_km, "radius_km")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"lat must be in [-90, 90]; got {lat}")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"lon must be in [-180, 180]; got {lon}")
        if radius < 0.0:
            raise ValueError(f"radius_km must be >= 0; got {radius}")
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "radius_km", radius)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km on a sphere of radius :data:`EARTH_MEAN_RADIUS_KM`.

    This is the ``atan2`` form of the haversine formula, which is well
    conditioned for nearby and for antipodal points. A spherical Earth model
    errs by up to about 0.5% against the ellipsoid.

    Args:
        lat1: Latitude of the first point, in decimal degrees.
        lon1: Longitude of the first point, in decimal degrees.
        lat2: Latitude of the second point, in decimal degrees.
        lon2: Longitude of the second point, in decimal degrees.

    Returns:
        The distance in kilometres.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    h = min(1.0, max(0.0, h))
    return 2.0 * EARTH_MEAN_RADIUS_KM * math.atan2(math.sqrt(h), math.sqrt(1.0 - h))


@dataclass(frozen=True)
class Reference:
    """Points at the claim, ordering or evidence item that a finding is about.

    Attributes:
        kind: ``"claim"``, ``"ordering"`` or ``"evidence"``.
        identifier: The statement id (claims, orderings) or evidence id.
        index: Position of the claim or ordering in its statement.
        speaker: The statement's speaker, when the reference is to a statement.
    """

    kind: Literal["claim", "ordering", "evidence"]
    identifier: str
    index: int | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class LocatedObservation:
    """One subject at one place over one time interval."""

    subject: str
    location: Location
    interval: TimeInterval
    reference: Reference


@dataclass(frozen=True)
class TravelAssessment:
    """Result of an impossible-travel check between two located observations.

    Attributes:
        distance_km: Great-circle distance between the two centres.
        effective_distance_km: Distance minus both uncertainty radii (>= 0).
        gap_seconds: Time between the observations (0 when they overlap).
        required_speed_kmh: Speed needed to connect them (``inf`` if impossible
            at any speed because they overlap in time but not in space).
        max_speed_kmh: The speed limit applied.
        feasible: ``required_speed_kmh <= max_speed_kmh``.
    """

    distance_km: float
    effective_distance_km: float
    gap_seconds: float
    required_speed_kmh: float
    max_speed_kmh: float
    feasible: bool


@runtime_checkable
class TravelFeasibilityCheck(Protocol):
    """Decides whether one subject could have been at both observations.

    A different implementation, such as an impossible-travel model with
    transport networks, can replace :class:`HaversineTravelCheck`.
    """

    def assess(self, first: LocatedObservation, second: LocatedObservation) -> TravelAssessment:
        """Assess travel feasibility between two observations of one subject."""
        ...


@dataclass(frozen=True)
class HaversineTravelCheck:
    """Default travel check: haversine distance and a single speed limit."""

    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH

    def __post_init__(self) -> None:
        """Validate the speed limit."""
        speed = _finite(self.max_speed_kmh, "max_speed_kmh")
        if speed <= 0.0:
            raise ValueError(f"max_speed_kmh must be > 0; got {speed}")
        object.__setattr__(self, "max_speed_kmh", speed)

    def assess(self, first: LocatedObservation, second: LocatedObservation) -> TravelAssessment:
        """Return the speed needed to be at both observations, and whether it is feasible.

        The effective distance subtracts both uncertainty radii. The time gap
        is zero when the intervals are concurrent (or meet). Otherwise it is
        the time from the end of the earlier interval to the start of the later
        one.

        Args:
            first: One observation.
            second: The other observation.

        Returns:
            The :class:`TravelAssessment`.
        """
        a, b = first.location, second.location
        distance = haversine_km(a.lat, a.lon, b.lat, b.lon)
        effective = max(0.0, distance - a.radius_km - b.radius_km)
        relation = allen_relation(first.interval, second.interval)
        if relation in CONCURRENT_RELATIONS:
            gap = 0.0
        elif relation in (AllenRelation.BEFORE, AllenRelation.MEETS):
            gap = second.interval.start - first.interval.end
        else:
            gap = first.interval.start - second.interval.end
        if effective == 0.0:
            speed = 0.0
        elif gap == 0.0:
            speed = math.inf
        else:
            speed = effective / (gap / 3600.0)
        return TravelAssessment(
            distance_km=distance,
            effective_distance_km=effective,
            gap_seconds=gap,
            required_speed_kmh=speed,
            max_speed_kmh=self.max_speed_kmh,
            feasible=speed <= self.max_speed_kmh,
        )


# --------------------------------------------------------------------------- #
# Data model: claims, statements and evidence.
# --------------------------------------------------------------------------- #
#: A claim value: categorical (str, bool) or numeric (int, float).
ClaimValue = str | int | float | bool


def _check_value(value: object) -> ClaimValue:
    """Validate a claim value's type (and finiteness for floats)."""
    if isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"claim value must be finite; got {value!r}")
        return value
    raise TypeError(f"claim value must be str, int, float or bool; got {type(value).__name__}")


@dataclass(frozen=True, order=True)
class ClaimKey:
    """The proposition a claim is about: a subject and a predicate."""

    subject: str
    predicate: str

    def __post_init__(self) -> None:
        """Validate that both parts are non-empty."""
        _non_empty(self.subject, "subject")
        _non_empty(self.predicate, "predicate")

    def __str__(self) -> str:
        """Render as ``subject.predicate``."""
        return f"{self.subject}.{self.predicate}"


@dataclass(frozen=True)
class Claim:
    """One checkable assertion: ``subject``'s ``predicate`` has ``value``.

    Attributes:
        subject: Who or what the claim is about.
        predicate: The property asserted (for example ``"location"``).
        value: The asserted value.
        interval: When the claim applies; ``None`` means timeless.
        location: Where the subject was, when the claim places it.
        statement_id: The statement the claim belongs to. A
            :class:`Statement` fills this in.
    """

    subject: str
    predicate: str
    value: ClaimValue
    interval: TimeInterval | None = None
    location: Location | None = None
    statement_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the claim."""
        _non_empty(self.subject, "subject")
        _non_empty(self.predicate, "predicate")
        _check_value(self.value)

    @property
    def key(self) -> ClaimKey:
        """The claim's subject and predicate."""
        return ClaimKey(self.subject, self.predicate)


@dataclass(frozen=True)
class Detail:
    """One detail of an account, annotated as verifiable or not (Nahari et al., 2014)."""

    text: str
    verifiable: bool

    def __post_init__(self) -> None:
        """Validate the annotation."""
        if not isinstance(self.verifiable, bool):
            raise TypeError("verifiable must be a bool")


@dataclass(frozen=True)
class TemporalOrdering:
    """A claimed ordering: event ``first`` stands in one of ``relations`` to ``second``.

    Events are identified by claim keys. For example,
    ``ClaimKey("X", "left_home")`` ``BEFORE`` ``ClaimKey("X", "alarm")``.
    """

    first: ClaimKey
    second: ClaimKey
    relations: frozenset[AllenRelation]

    def __post_init__(self) -> None:
        """Validate and normalise the relation set."""
        if self.first == self.second:
            raise ValueError("an ordering needs two different events")
        if isinstance(self.relations, AllenRelation | str):
            rels = frozenset({AllenRelation(self.relations)})
        else:
            rels = frozenset(AllenRelation(r) for r in self.relations)
        if not rels:
            raise ValueError("an ordering needs at least one relation")
        object.__setattr__(self, "relations", rels)


@dataclass(frozen=True)
class Statement:
    """One account given by a speaker at a time.

    Attributes:
        statement_id: Unique identifier.
        speaker: Who gave the statement.
        made_at: When it was given (seconds, same clock as intervals).
        claims: Checkable claims extracted from the statement.
        details: Details annotated for verifiability.
        disclosed_evidence_ids: Evidence shown to the speaker before this
            statement. Disclosure is cumulative per speaker: evidence disclosed
            before an earlier statement counts as disclosed before every later
            one.
        orderings: Claimed temporal orderings of events.
    """

    statement_id: str
    speaker: str
    made_at: float
    claims: tuple[Claim, ...] = ()
    details: tuple[Detail, ...] = ()
    disclosed_evidence_ids: frozenset[str] = frozenset()
    orderings: tuple[TemporalOrdering, ...] = ()

    def __post_init__(self) -> None:
        """Validate, freeze collections and stamp claims with the statement id."""
        _non_empty(self.statement_id, "statement_id")
        _non_empty(self.speaker, "speaker")
        object.__setattr__(self, "made_at", _finite(self.made_at, "made_at"))
        if isinstance(self.disclosed_evidence_ids, str):
            raise TypeError("disclosed_evidence_ids must be a collection of ids, not a str")
        stamped: list[Claim] = []
        for claim in tuple(self.claims):
            if not isinstance(claim, Claim):
                raise TypeError("claims must be Claim instances")
            if claim.statement_id is None:
                claim = replace(claim, statement_id=self.statement_id)
            elif claim.statement_id != self.statement_id:
                raise ValueError(
                    f"claim statement_id {claim.statement_id!r} != {self.statement_id!r}"
                )
            stamped.append(claim)
        object.__setattr__(self, "claims", tuple(stamped))
        object.__setattr__(self, "details", tuple(self.details))
        object.__setattr__(self, "orderings", tuple(self.orderings))
        object.__setattr__(self, "disclosed_evidence_ids", frozenset(self.disclosed_evidence_ids))


class Stance(StrEnum):
    """Whether an evidence item supports or refutes the claim it targets."""

    SUPPORTS = "supports"
    REFUTES = "refutes"


@dataclass(frozen=True)
class EvidenceItem:
    """One item of evidence, given in one of two forms.

    Target form: ``target`` names a claim key and ``stance`` says whether
    the item supports or refutes it. ``target_statement_id`` can restrict the
    item to the claims of one statement.

    Asserted form: ``asserted`` is the item's own claim. It supports a
    statement claim with the same key, at a concurrent time, when the values
    are compatible and the locations, if both are given, overlap. Otherwise
    it refutes that claim.

    Attributes:
        evidence_id: Unique identifier.
        target: The claim key the item bears on (target form).
        stance: Supports or refutes (target form).
        asserted: The item's own claim (asserted form).
        target_statement_id: Restrict a target-form item to one statement.
        reliability: ``r`` in ``[0, 1]``. When omitted, the reliability model
            looks it up by ``source_id``.
        source_id: Originating source (used for reliability lookup and as the
            default independence cluster).
        independence_cluster: Declared cluster of mutually dependent items.
        time: When the evidenced event happened (target form). In asserted
            form the asserted claim's interval is used.
        provenance: Optional :class:`intel.provenance.Provenance` record.
    """

    evidence_id: str
    target: ClaimKey | None = None
    stance: Stance | None = None
    asserted: Claim | None = None
    target_statement_id: str | None = None
    reliability: float | None = None
    source_id: str | None = None
    independence_cluster: str | None = None
    time: TimeInterval | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one form is used."""
        _non_empty(self.evidence_id, "evidence_id")
        if (self.target is None) == (self.asserted is None):
            raise ValueError("give exactly one of target (with stance) or asserted")
        if self.target is not None:
            if self.stance is None:
                raise ValueError("a target-form evidence item needs a stance")
            object.__setattr__(self, "stance", Stance(self.stance))
        elif self.stance is not None or self.target_statement_id is not None:
            raise ValueError("stance and target_statement_id apply only to target form")
        if self.reliability is not None:
            object.__setattr__(self, "reliability", _probability(self.reliability, "reliability"))
        if (
            self.asserted is not None
            and self.asserted.interval is not None
            and self.time is not None
            and self.time != self.asserted.interval
        ):
            raise ValueError("time conflicts with the asserted claim's interval")

    @property
    def key(self) -> ClaimKey:
        """The claim key this item bears on."""
        if self.asserted is not None:
            return self.asserted.key
        assert self.target is not None  # guaranteed by __post_init__
        return self.target

    @property
    def effective_time(self) -> TimeInterval | None:
        """The asserted claim's interval if set, else ``time``."""
        if self.asserted is not None and self.asserted.interval is not None:
            return self.asserted.interval
        return self.time


# --------------------------------------------------------------------------- #
# Value compatibility.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValueTolerance:
    """Numeric tolerance: compatible iff ``|a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)``.

    This is the rule of :func:`math.isclose`. The default, ``rel_tol=1e-9`` and
    ``abs_tol=0``, is that function's own default. It treats any difference
    beyond floating-point noise as incompatible. No domain-neutral tolerance
    exists for quantities that people report, so set per-predicate tolerances
    in :class:`VerifierConfig`.
    """

    abs_tol: float = 0.0
    rel_tol: float = 1e-9

    def __post_init__(self) -> None:
        """Validate that both tolerances are finite and non-negative."""
        for name in ("abs_tol", "rel_tol"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0; got {value}")
            object.__setattr__(self, name, value)


def _is_number(value: object) -> bool:
    """Whether ``value`` is numeric (bool is categorical)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _normalise_text(text: str) -> str:
    """Collapse whitespace and casefold for categorical comparison."""
    return " ".join(text.split()).casefold()


def values_compatible(a: ClaimValue, b: ClaimValue, tolerance: ValueTolerance) -> bool:
    """Whether two claim values can both hold.

    Two numbers are compared with ``tolerance``. Two strings are compared for
    equality after collapsing whitespace and casefolding. Any other pair must
    have the same type and be equal, so ``True`` is incompatible with ``1`` and
    with ``"true"``.

    Args:
        a: The first value.
        b: The second value.
        tolerance: The numeric tolerance.

    Returns:
        ``True`` when the values are compatible.
    """
    if _is_number(a) and _is_number(b):
        return math.isclose(
            float(a), float(b), rel_tol=tolerance.rel_tol, abs_tol=tolerance.abs_tol
        )
    if isinstance(a, str) and isinstance(b, str):
        return _normalise_text(a) == _normalise_text(b)
    return type(a) is type(b) and a == b


def _locations_compatible(a: Location | None, b: Location | None) -> bool:
    """Whether two location circles overlap (``True`` when either is unknown)."""
    if a is None or b is None:
        return True
    return haversine_km(a.lat, a.lon, b.lat, b.lon) <= a.radius_km + b.radius_km


# --------------------------------------------------------------------------- #
# Pluggable source-reliability and independence-cluster models.
# --------------------------------------------------------------------------- #
@runtime_checkable
class SourceReliabilityModel(Protocol):
    """Supplies the reliability ``r`` of an evidence item, or ``None`` to abstain."""

    def reliability(self, item: EvidenceItem) -> float | None:
        """Return ``r`` in ``[0, 1]``, or ``None`` when no reliability is known."""
        ...


@runtime_checkable
class IndependenceClusterModel(Protocol):
    """Assigns each evidence item to a cluster of mutually dependent items."""

    def cluster_of(self, item: EvidenceItem) -> str:
        """Return the identifier of the item's independence cluster."""
        ...


@dataclass(frozen=True)
class TableSourceReliability:
    """Default reliability model: the item's own reliability, else a per-source table.

    When neither is available the model returns ``None``, and the item is
    excluded from corroboration with the reason recorded. No reliability is
    guessed.
    """

    table: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze the table."""
        frozen = {
            _non_empty(k, "source id"): _probability(v, f"reliability of {k!r}")
            for k, v in self.table.items()
        }
        object.__setattr__(self, "table", MappingProxyType(frozen))

    def reliability(self, item: EvidenceItem) -> float | None:
        """Return the item's reliability, its source's tabled value, or ``None``."""
        if item.reliability is not None:
            return item.reliability
        if item.source_id is not None:
            return self.table.get(item.source_id)
        return None


@dataclass(frozen=True)
class DeclaredIndependenceClusters:
    """Default cluster model: declared cluster, else source, else the item itself.

    An item with neither a declared cluster nor a source is treated as
    independent of all others. That assumption can only be checked by the
    caller.
    """

    def cluster_of(self, item: EvidenceItem) -> str:
        """Return ``cluster:<id>``, ``source:<id>`` or ``item:<evidence_id>``."""
        if item.independence_cluster:
            return f"cluster:{item.independence_cluster}"
        if item.source_id:
            return f"source:{item.source_id}"
        return f"item:{item.evidence_id}"


# --------------------------------------------------------------------------- #
# Corroboration.
# --------------------------------------------------------------------------- #
class CorroborationStatus(StrEnum):
    """Evidence-based status of a claim."""

    CORROBORATED = "corroborated"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"

    def to_claim_status(self) -> ClaimStatus:
        """Map onto :class:`intel.verifier_loop.ClaimStatus` (imported lazily).

        ``corroborated`` maps to ``CONFIRMED``, ``contradicted`` to ``REFUTED``
        and ``unresolved`` to ``UNAVAILABLE``. Corroboration is probabilistic,
        not an oracle, so a consumer should use the verifier loop's soft mode
        for these verdicts.
        """
        from omni_mercury_engine.intel.verifier_loop import ClaimStatus

        return {
            CorroborationStatus.CORROBORATED: ClaimStatus.CONFIRMED,
            CorroborationStatus.CONTRADICTED: ClaimStatus.REFUTED,
            CorroborationStatus.UNRESOLVED: ClaimStatus.UNAVAILABLE,
        }[self]


LowReliabilityPolicy = Literal["ignore", "invert"]


def log_likelihood_ratio(
    reliability: float,
    stance: Stance,
    *,
    epsilon: float = DEFAULT_RELIABILITY_EPSILON,
    low_reliability_policy: LowReliabilityPolicy = "ignore",
) -> float | None:
    """Log likelihood ratio of one evidence item under the symmetric reliability model.

    The model is ``P(supporting report | claim holds) = r`` and
    ``P(supporting report | claim does not hold) = 1 - r``. A refuting report
    is a report that the claim does not hold, with the same reliability. So
    ``log LR = +/- log(r / (1 - r))`` after ``r`` is clipped to
    ``[epsilon, 1 - epsilon]``.

    If ``r == 0.5`` the item is uninformative (``LR = 1``) and ``None`` is
    returned. If ``r < 0.5`` the model says the source reports the opposite of
    the truth more often than not, which usually means the reliability is
    mis-specified. Under the default policy ``"ignore"`` such an item is
    treated as uninformative (``None``). Under ``"invert"`` the formula is
    applied as is, so the item counts against what it reports.

    Args:
        reliability: ``r`` in ``[0, 1]``.
        stance: Whether the item supports or refutes the claim.
        epsilon: Clip in ``(0, 0.5)``.
        low_reliability_policy: ``"ignore"`` or ``"invert"``.

    Returns:
        The natural-log likelihood ratio, or ``None`` if the item is uninformative.

    Raises:
        ValueError: On an out-of-range reliability, epsilon or policy.
    """
    r = _probability(reliability, "reliability")
    if not 0.0 < epsilon < 0.5:
        raise ValueError(f"epsilon must be in (0, 0.5); got {epsilon!r}")
    if low_reliability_policy not in ("ignore", "invert"):
        raise ValueError(f"unknown low_reliability_policy {low_reliability_policy!r}")
    r = min(1.0 - epsilon, max(epsilon, r))
    if r == 0.5:
        return None
    if r < 0.5 and low_reliability_policy == "ignore":
        return None
    llr = _logit(r)
    return llr if Stance(stance) is Stance.SUPPORTS else -llr


@dataclass(frozen=True)
class ItemWeight:
    """One evidence item's weight within its cluster."""

    evidence_id: str
    cluster_id: str
    stance: Stance
    reliability: float
    reliability_used: float
    log_lr: float
    provenance: Provenance | None = None


@dataclass(frozen=True)
class ClusterContribution:
    """What one independence cluster contributes to a claim's log-odds.

    The cluster contributes the log LR of the item with the largest
    ``|log LR|``. Ties in the same direction go to the smallest evidence id.
    If the largest magnitude is shared by a supporting and a refuting item,
    the cluster contributes 0, ``evidence_id`` is ``None`` and ``conflicting``
    is set. ``conflicting`` is also set whenever the cluster holds informative
    items in both directions.
    """

    cluster_id: str
    evidence_id: str | None
    stance: Stance | None
    log_lr: float
    n_items: int
    conflicting: bool
    members: tuple[ItemWeight, ...]
    provenance: Provenance | None = None


@dataclass(frozen=True)
class ExcludedEvidence:
    """An evidence item that bears on a claim but was not counted, and why."""

    evidence_id: str
    reason: str


@dataclass(frozen=True)
class ClaimCorroboration:
    """Corroboration of one claim by the supplied evidence.

    Attributes:
        statement_id: The claim's statement.
        claim_index: Position of the claim in its statement.
        key: The claim's subject and predicate.
        value: The claimed value.
        prior: Prior probability.
        prior_log_odds: Its log-odds.
        log_likelihood_ratio: Sum of cluster contributions.
        posterior_log_odds: Prior log-odds plus the sum.
        posterior: Posterior probability that the claim holds (equals the
            prior when no informative evidence exists).
        status: Corroborated, contradicted or unresolved.
        contributions: One entry per independence cluster.
        excluded: Bearing items that were not counted, with reasons.
        insufficient_evidence: ``True`` when no cluster was informative (the
            status is then an abstention).
        measures: What the posterior measures.
    """

    statement_id: str
    claim_index: int
    key: ClaimKey
    value: ClaimValue
    prior: float
    prior_log_odds: float
    log_likelihood_ratio: float
    posterior_log_odds: float
    posterior: float
    status: CorroborationStatus
    contributions: tuple[ClusterContribution, ...]
    excluded: tuple[ExcludedEvidence, ...]
    insufficient_evidence: bool
    measures: str = CORROBORATION_MEASURES

    @property
    def provenance(self) -> Provenance | None:
        """Weakest-wins merge of the counted items' provenance.

        ``None`` when nothing was counted or any counted item lacks a
        provenance record. Corroboration then cannot be attributed, and an
        output boundary that requires provenance withholds it.
        """
        counted = [c.provenance for c in self.contributions if c.evidence_id is not None]
        present = [p for p in counted if p is not None]
        if not present or len(present) != len(counted):
            return None
        merged = present[0]
        for other in present[1:]:
            merged = merged.merge(other)
        return merged

    def as_claim_verdict(self) -> ClaimVerdict:
        """Express this result as an :class:`intel.verifier_loop.ClaimVerdict`."""
        from omni_mercury_engine.intel.verifier_loop import ClaimVerdict

        n_counted = sum(1 for c in self.contributions if c.log_lr != 0.0)
        return ClaimVerdict(
            kind="evidence_corroboration",
            claim_text=f"{self.key} = {self.value!r}",
            status=self.status.to_claim_status(),
            reason=(
                f"posterior {self.posterior:.6g} (prior {self.prior:.6g}) from "
                f"{n_counted} informative independence cluster(s)"
            ),
            checker="bayesian_log_odds_corroboration",
        )


def _cluster_contribution(cluster_id: str, members: list[ItemWeight]) -> ClusterContribution:
    """Apply the strongest-item rule to one cluster."""
    ordered = sorted(members, key=lambda w: (-abs(w.log_lr), w.evidence_id))
    top_magnitude = abs(ordered[0].log_lr)
    top = [w for w in ordered if abs(w.log_lr) == top_magnitude]
    conflicting = any(w.log_lr > 0.0 for w in ordered) and any(w.log_lr < 0.0 for w in ordered)
    tied_opposite = any(w.log_lr > 0.0 for w in top) and any(w.log_lr < 0.0 for w in top)
    if tied_opposite:
        return ClusterContribution(
            cluster_id, None, None, 0.0, len(ordered), True, tuple(ordered), None
        )
    chosen = top[0]
    return ClusterContribution(
        cluster_id,
        chosen.evidence_id,
        chosen.stance,
        chosen.log_lr,
        len(ordered),
        conflicting,
        tuple(ordered),
        chosen.provenance,
    )


_AMBIGUOUS_REASON = (
    "ambiguous target: claims with this key disagree; give target_statement_id or an "
    "asserted claim"
)
_NO_RELIABILITY_REASON = "no reliability available (none on the item and none for its source)"
_UNINFORMATIVE_REASON = "uninformative: reliability is 0.5 after clipping"
_LOW_RELIABILITY_REASON = (
    "reliability below 0.5 is treated as uninformative under the 'ignore' policy"
)
_UNTIMED_REASON = "evidence has no time, so it cannot bear on a claim about a specific interval"
_PARTIAL_COVER_REASON = (
    "evidence covers only part of the claimed interval: it can refute the claim, not support it"
)


#: Absolute tolerance, in log-odds, for the inclusive posterior thresholds, so
#: a Bayes factor of exactly 19 reaches 0.95 despite floating-point rounding.
_THRESHOLD_LOG_ODDS_TOLERANCE = 1e-12


# --------------------------------------------------------------------------- #
# Contradictions.
# --------------------------------------------------------------------------- #
class ContradictionKind(StrEnum):
    """What kind of incompatibility a contradiction is."""

    VALUE = "value"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class ContradictionScope(StrEnum):
    """Where the two sides of a contradiction come from."""

    WITHIN_STATEMENT = "within_statement"
    SAME_SPEAKER = "same_speaker"
    BETWEEN_SPEAKERS = "between_speakers"
    CLAIM_VS_EVIDENCE = "claim_vs_evidence"


@dataclass(frozen=True)
class Contradiction:
    """A logical incompatibility between claims, or between claims and evidence.

    Attributes:
        kind: Value, temporal or spatial.
        scope: Within a statement, across one speaker's statements, between
            speakers, or claim against evidence.
        keys: The claim keys involved.
        references: The claims, orderings and evidence items involved.
        description: A plain-language account of the incompatibility.
        values: The incompatible values (value contradictions).
        observed_relation: The Allen relation found (temporal contradictions).
        claimed_relations: The relations claimed (temporal contradictions).
        travel: The travel assessment (spatial contradictions).
        measures: What a contradiction measures.
    """

    kind: ContradictionKind
    scope: ContradictionScope
    keys: tuple[ClaimKey, ...]
    references: tuple[Reference, ...]
    description: str
    values: tuple[ClaimValue, ...] = ()
    observed_relation: AllenRelation | None = None
    claimed_relations: frozenset[AllenRelation] = frozenset()
    travel: TravelAssessment | None = None
    measures: str = CONTRADICTION_MEASURES


def _scope(*refs: Reference) -> ContradictionScope:
    """Classify a contradiction's scope from its references."""
    if any(r.kind == "evidence" for r in refs):
        return ContradictionScope.CLAIM_VS_EVIDENCE
    if len({r.identifier for r in refs}) == 1:
        return ContradictionScope.WITHIN_STATEMENT
    if len({r.speaker for r in refs}) == 1:
        return ContradictionScope.SAME_SPEAKER
    return ContradictionScope.BETWEEN_SPEAKERS


@dataclass(frozen=True)
class _ClaimSlot:
    """A claim together with its statement and position."""

    statement: Statement
    index: int
    claim: Claim

    @property
    def reference(self) -> Reference:
        """Reference to this claim."""
        return Reference("claim", self.statement.statement_id, self.index, self.statement.speaker)


def _evidence_reference(item: EvidenceItem) -> Reference:
    """Reference to an evidence item."""
    return Reference("evidence", item.evidence_id)


# --------------------------------------------------------------------------- #
# Consistency, SUE and adaptation.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClaimChange:
    """A difference between a speaker's successive statements for one claim key.

    Attributes:
        kind: ``"changed"`` (incompatible values at overlapping times),
            ``"added"`` or ``"dropped"``.
        key: The claim key.
        from_statement: The earlier statement.
        to_statement: The later statement.
        previous_values: Values in the earlier statement.
        new_values: Values in the later statement.
        matches_newly_disclosed: Evidence disclosed between the two statements
            that refuted the earlier version and supports the later one.
    """

    kind: Literal["changed", "added", "dropped"]
    key: ClaimKey
    from_statement: str
    to_statement: str
    previous_values: tuple[ClaimValue, ...]
    new_values: tuple[ClaimValue, ...]
    matches_newly_disclosed: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdaptationEvent:
    """A later statement changed to match evidence disclosed after the earlier one."""

    evidence_id: str
    key: ClaimKey
    from_statement: str
    to_statement: str


@dataclass(frozen=True)
class SUEIndicator:
    """Statement-evidence inconsistency for withheld versus disclosed evidence.

    Each counted unit is a (claim, evidence item) pair where the item bears on
    the claim. ``withheld`` counts pairs whose item had not been disclosed to
    the speaker before the statement. ``disclosed`` counts pairs whose item had
    been. ``successes`` are the pairs where the item refutes the claim. The
    counts use every bearing item whatever its reliability, because they
    describe what the statement says relative to the evidence held, not how
    strongly that evidence corroborates it.
    """

    withheld: Proportion
    disclosed: Proportion
    rate_difference: float | None
    measures: str = SUE_MEASURES
    note: str = SUE_NOTE


@dataclass(frozen=True)
class AdaptationIndicator:
    """How often a later statement changed to match newly disclosed evidence.

    The trials (opportunities) are the claims of an earlier statement that
    evidence disclosed before the next statement refutes. The successes are
    those where the next statement has a claim with the same key that the
    same evidence supports.
    """

    proportion: Proportion
    events: tuple[AdaptationEvent, ...]
    measures: str = ADAPTATION_MEASURES


@dataclass(frozen=True)
class SpeakerConsistency:
    """Consistency findings for one speaker's statements, in chronological order."""

    speaker: str
    statement_ids: tuple[str, ...]
    within_statement: tuple[Contradiction, ...]
    between_statements: tuple[Contradiction, ...]
    changes: tuple[ClaimChange, ...]
    sue: SUEIndicator
    adaptation: AdaptationIndicator


# --------------------------------------------------------------------------- #
# Verifiability Approach.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerifiabilityResult:
    """Verifiable-detail proportion of one statement.

    Attributes:
        statement_id: The statement.
        speaker: Its speaker.
        n_details: Annotated details.
        n_verifiable: Details annotated verifiable.
        proportion: ``n_verifiable / n_details`` with a Wilson interval
            (``estimate`` is ``None`` when there are no details).
        reference_n: Size of the supplied reference distribution (0 if none).
        z_score: ``(p - mean) / sd`` against the reference (``sd`` with
            ``ddof=1``), when computable.
        percentile: Mean-rank percentile of ``p`` in the reference (ties count
            half), when a reference is supplied.
        note: Why a comparison was or was not made.
        measures: What the proportion measures.
    """

    statement_id: str
    speaker: str
    n_details: int
    n_verifiable: int
    proportion: Proportion
    reference_n: int
    z_score: float | None
    percentile: float | None
    note: str
    measures: str = VERIFIABILITY_MEASURES


def assess_verifiability(
    statement: Statement,
    reference: Sequence[float] | None = None,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> VerifiabilityResult:
    """Proportion of verifiable details, optionally compared with a reference distribution.

    Args:
        statement: The statement whose details are annotated.
        reference: Verifiable-detail proportions from a comparison sample in
            the same context. Without one no norm is applied, because norms
            are context-specific.
        confidence: Wilson interval level.

    Returns:
        The :class:`VerifiabilityResult`.

    Raises:
        ValueError: If a reference value is not a finite proportion.
    """
    n = len(statement.details)
    k = sum(1 for d in statement.details if d.verifiable)
    proportion = wilson_interval(k, n, confidence)
    if reference is None:
        return VerifiabilityResult(
            statement.statement_id,
            statement.speaker,
            n,
            k,
            proportion,
            0,
            None,
            None,
            VERIFIABILITY_NO_NORM_NOTE,
        )
    ref = np.asarray([_probability(v, "reference proportion") for v in reference], dtype=float)
    if proportion.estimate is None:
        note = "No annotated details: abstained, no comparison with the reference."
        return VerifiabilityResult(
            statement.statement_id, statement.speaker, n, k, proportion, ref.size, None, None, note
        )
    p = proportion.estimate
    percentile: float | None = None
    z_score: float | None = None
    notes: list[str] = []
    if ref.size == 0:
        notes.append("Reference distribution is empty: no comparison.")
    else:
        below = int(np.count_nonzero(ref < p))
        equal = int(np.count_nonzero(ref == p))
        percentile = 100.0 * (below + 0.5 * equal) / ref.size
        if ref.size >= 2 and float(np.std(ref, ddof=1)) > 0.0:
            z_score = (p - float(np.mean(ref))) / float(np.std(ref, ddof=1))
        else:
            notes.append("z-score needs at least two reference values with non-zero spread.")
        notes.append(
            "Compared with the supplied reference; it is valid only if it comes from the same "
            "context and coding scheme."
        )
    return VerifiabilityResult(
        statement.statement_id,
        statement.speaker,
        n,
        k,
        proportion,
        int(ref.size),
        z_score,
        percentile,
        " ".join(notes),
    )


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VerifierConfig:
    """Settings for :class:`StatementVerifier`. Every default is declared above.

    Attributes:
        prior_probability: Prior that a claim holds, in ``(0, 1)``.
        corroborated_at: Posterior at or above which a claim is corroborated.
        contradicted_at: Posterior at or below which a claim is contradicted.
        reliability_epsilon: Reliability clip ``eps`` in ``(0, 0.5)``.
        low_reliability_policy: Handling of ``r < 0.5`` (see
            :func:`log_likelihood_ratio`).
        default_tolerance: Numeric tolerance for predicates not listed below.
        predicate_tolerances: Per-predicate numeric tolerances.
        confidence_level: Wilson interval level.
    """

    prior_probability: float = DEFAULT_PRIOR_PROBABILITY
    corroborated_at: float = DEFAULT_CORROBORATED_AT
    contradicted_at: float = DEFAULT_CONTRADICTED_AT
    reliability_epsilon: float = DEFAULT_RELIABILITY_EPSILON
    low_reliability_policy: LowReliabilityPolicy = "ignore"
    default_tolerance: ValueTolerance = field(default_factory=ValueTolerance)
    predicate_tolerances: Mapping[str, ValueTolerance] = field(default_factory=dict)
    confidence_level: float = DEFAULT_CONFIDENCE

    def __post_init__(self) -> None:
        """Validate the settings and freeze the tolerance table."""
        prior = _finite(self.prior_probability, "prior_probability")
        if not 0.0 < prior < 1.0:
            raise ValueError("prior_probability must be in (0, 1) (Cromwell's rule)")
        hi = _finite(self.corroborated_at, "corroborated_at")
        lo = _finite(self.contradicted_at, "contradicted_at")
        if not 0.0 < lo < hi < 1.0:
            raise ValueError("need 0 < contradicted_at < corroborated_at < 1")
        if not 0.0 < self.reliability_epsilon < 0.5:
            raise ValueError("reliability_epsilon must be in (0, 0.5)")
        if self.low_reliability_policy not in ("ignore", "invert"):
            raise ValueError(f"unknown low_reliability_policy {self.low_reliability_policy!r}")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        tolerances = dict(self.predicate_tolerances)
        for predicate, tol in tolerances.items():
            if not isinstance(tol, ValueTolerance):
                raise TypeError(f"tolerance for {predicate!r} must be a ValueTolerance")
        object.__setattr__(self, "predicate_tolerances", MappingProxyType(tolerances))

    def tolerance_for(self, predicate: str) -> ValueTolerance:
        """Return the numeric tolerance for ``predicate``."""
        return self.predicate_tolerances.get(predicate, self.default_tolerance)


@dataclass(frozen=True)
class ReportSummary:
    """Aggregate counts of the evidence findings; it adds no judgement of its own."""

    n_statements: int
    n_claims: int
    n_evidence: int
    n_corroborated: int
    n_contradicted: int
    n_unresolved: int
    n_insufficient_evidence: int
    contradictions_by_kind: Mapping[str, int]
    contradictions_by_scope: Mapping[str, int]
    findings: tuple[str, ...]


@dataclass(frozen=True)
class VerificationReport:
    """The frozen result of :meth:`StatementVerifier.verify`.

    Attributes:
        corroboration: Per-claim corroboration, in statement then claim order.
        contradictions: Every contradiction found.
        consistency: Per-speaker consistency, SUE and adaptation indicators.
        verifiability: Per-statement verifiable-detail proportions.
        summary: Aggregate counts.
        settings: The configuration used.
        framing: The scientific framing of the report.
        limitations: What the report does not establish.
    """

    corroboration: tuple[ClaimCorroboration, ...]
    contradictions: tuple[Contradiction, ...]
    consistency: tuple[SpeakerConsistency, ...]
    verifiability: tuple[VerifiabilityResult, ...]
    summary: ReportSummary
    settings: VerifierConfig
    framing: tuple[str, ...] = SCIENTIFIC_FRAMING
    limitations: tuple[str, ...] = VERIFIER_LIMITATIONS

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping (infinite floats rendered as strings)."""
        out = _jsonable(self)
        assert isinstance(out, dict)  # a dataclass always converts to a dict
        return out

    def to_json(self) -> str:
        """Return the report as JSON text with sorted keys."""
        return json.dumps(self.as_dict(), sort_keys=True)


class StatementVerifier:
    """Verifies statements against evidence and reports evidence-grounded findings.

    The verifier never mutates its inputs, has no randomness, and returns the
    same report for the same inputs whatever their order.
    """

    def __init__(
        self,
        config: VerifierConfig | None = None,
        *,
        reliability_model: SourceReliabilityModel | None = None,
        cluster_model: IndependenceClusterModel | None = None,
        travel_check: TravelFeasibilityCheck | None = None,
    ) -> None:
        """Create a verifier.

        Args:
            config: Settings (declared defaults when omitted).
            reliability_model: Source of item reliabilities (default
                :class:`TableSourceReliability` with an empty table).
            cluster_model: Independence-cluster assignment (default
                :class:`DeclaredIndependenceClusters`).
            travel_check: Impossible-travel check (default
                :class:`HaversineTravelCheck`).
        """
        self.config = config or VerifierConfig()
        self.reliability_model: SourceReliabilityModel = (
            reliability_model or TableSourceReliability()
        )
        self.cluster_model: IndependenceClusterModel = (
            cluster_model or DeclaredIndependenceClusters()
        )
        self.travel_check: TravelFeasibilityCheck = travel_check or HaversineTravelCheck()

    # ------------------------------------------------------------- bearing
    def _bearing(self, item: EvidenceItem, slot: _ClaimSlot) -> Stance | None:
        """Whether ``item`` supports or refutes ``slot``'s claim, or ``None`` if it does not bear."""
        claim = slot.claim
        if item.key != claim.key:
            return None
        if not intervals_concurrent(item.effective_time, claim.interval):
            return None
        if item.asserted is None:
            if (
                item.target_statement_id is not None
                and item.target_statement_id != slot.statement.statement_id
            ):
                return None
            return item.stance
        tolerance = self.config.tolerance_for(claim.predicate)
        compatible = values_compatible(
            item.asserted.value, claim.value, tolerance
        ) and _locations_compatible(item.asserted.location, claim.location)
        return Stance.SUPPORTS if compatible else Stance.REFUTES

    def _ambiguous_items(
        self, evidence: Sequence[EvidenceItem], slots: Sequence[_ClaimSlot]
    ) -> frozenset[str]:
        """Target-form items without a statement id whose candidate claims disagree."""
        ambiguous: set[str] = set()
        for item in evidence:
            if item.asserted is not None or item.target_statement_id is not None:
                continue
            candidates = [s for s in slots if self._bearing(item, s) is not None]
            tolerance = self.config.tolerance_for(item.key.predicate)
            for a, b in combinations(candidates, 2):
                if not values_compatible(
                    a.claim.value, b.claim.value, tolerance
                ) or not _locations_compatible(a.claim.location, b.claim.location):
                    ambiguous.add(item.evidence_id)
                    break
        return frozenset(ambiguous)

    # ------------------------------------------------------- corroboration
    def _corroborate(
        self,
        slot: _ClaimSlot,
        evidence: Sequence[EvidenceItem],
        ambiguous: frozenset[str],
    ) -> ClaimCorroboration:
        """Bayesian log-odds corroboration of one claim."""
        cfg = self.config
        excluded: list[ExcludedEvidence] = []
        clusters: dict[str, list[ItemWeight]] = defaultdict(list)
        for item in evidence:
            stance = self._bearing(item, slot)
            if stance is None:
                continue
            if item.evidence_id in ambiguous:
                excluded.append(ExcludedEvidence(item.evidence_id, _AMBIGUOUS_REASON))
                continue
            interval = slot.claim.interval
            if interval is not None:
                when = item.effective_time
                if when is None:
                    excluded.append(ExcludedEvidence(item.evidence_id, _UNTIMED_REASON))
                    continue
                if stance is Stance.SUPPORTS and not interval_covered(interval, when):
                    excluded.append(ExcludedEvidence(item.evidence_id, _PARTIAL_COVER_REASON))
                    continue
            raw = self.reliability_model.reliability(item)
            if raw is None:
                excluded.append(ExcludedEvidence(item.evidence_id, _NO_RELIABILITY_REASON))
                continue
            reliability = _probability(raw, f"reliability of {item.evidence_id!r}")
            used = min(1.0 - cfg.reliability_epsilon, max(cfg.reliability_epsilon, reliability))
            llr = log_likelihood_ratio(
                reliability,
                stance,
                epsilon=cfg.reliability_epsilon,
                low_reliability_policy=cfg.low_reliability_policy,
            )
            if llr is None:
                reason = _UNINFORMATIVE_REASON if used == 0.5 else _LOW_RELIABILITY_REASON
                excluded.append(ExcludedEvidence(item.evidence_id, reason))
                continue
            cluster_id = self.cluster_model.cluster_of(item)
            clusters[cluster_id].append(
                ItemWeight(
                    item.evidence_id,
                    cluster_id,
                    stance,
                    reliability,
                    used,
                    llr,
                    item.provenance,
                )
            )
        contributions = tuple(_cluster_contribution(cid, clusters[cid]) for cid in sorted(clusters))
        total = math.fsum(c.log_lr for c in contributions)
        prior_lo = _logit(cfg.prior_probability)
        posterior_lo = prior_lo + total
        posterior = _expit(posterior_lo)
        insufficient = all(c.log_lr == 0.0 for c in contributions)
        tol = _THRESHOLD_LOG_ODDS_TOLERANCE
        upper, lower = _logit(cfg.corroborated_at), _logit(cfg.contradicted_at)
        # A prior already past a threshold cannot carry a claim across it on
        # its own: the evidence must then reach the threshold by itself (as it
        # would have to from indifference), not merely point the right way.
        reaches_upper = posterior_lo >= upper - tol and (prior_lo < upper or total >= upper - tol)
        reaches_lower = posterior_lo <= lower + tol and (prior_lo > lower or total <= lower + tol)
        if insufficient:
            status = CorroborationStatus.UNRESOLVED
        elif reaches_upper and total > 0.0:
            status = CorroborationStatus.CORROBORATED
        elif reaches_lower and total < 0.0:
            status = CorroborationStatus.CONTRADICTED
        else:
            status = CorroborationStatus.UNRESOLVED
        return ClaimCorroboration(
            statement_id=slot.statement.statement_id,
            claim_index=slot.index,
            key=slot.claim.key,
            value=slot.claim.value,
            prior=cfg.prior_probability,
            prior_log_odds=prior_lo,
            log_likelihood_ratio=total,
            posterior_log_odds=posterior_lo,
            posterior=posterior,
            status=status,
            contributions=contributions,
            excluded=tuple(excluded),
            insufficient_evidence=insufficient,
        )

    # ------------------------------------------------------ contradictions
    def _value_contradictions(
        self, slots: Sequence[_ClaimSlot], evidence: Sequence[EvidenceItem]
    ) -> list[Contradiction]:
        """Incompatible values at overlapping times: claim-claim and claim-asserted evidence."""
        out: list[Contradiction] = []
        for a, b in combinations(slots, 2):
            if a.claim.key != b.claim.key:
                continue
            if not intervals_concurrent(a.claim.interval, b.claim.interval):
                continue
            tolerance = self.config.tolerance_for(a.claim.predicate)
            if values_compatible(a.claim.value, b.claim.value, tolerance):
                continue
            out.append(
                Contradiction(
                    kind=ContradictionKind.VALUE,
                    scope=_scope(a.reference, b.reference),
                    keys=(a.claim.key,),
                    references=(a.reference, b.reference),
                    description=(
                        f"{a.claim.key}: {a.claim.value!r} and {b.claim.value!r} "
                        "claimed for overlapping times"
                    ),
                    values=(a.claim.value, b.claim.value),
                )
            )
        for slot in slots:
            for item in evidence:
                asserted = item.asserted
                if asserted is None or asserted.key != slot.claim.key:
                    continue
                if not intervals_concurrent(item.effective_time, slot.claim.interval):
                    continue
                tolerance = self.config.tolerance_for(slot.claim.predicate)
                if values_compatible(slot.claim.value, asserted.value, tolerance):
                    continue
                ref = _evidence_reference(item)
                out.append(
                    Contradiction(
                        kind=ContradictionKind.VALUE,
                        scope=ContradictionScope.CLAIM_VS_EVIDENCE,
                        keys=(slot.claim.key,),
                        references=(slot.reference, ref),
                        description=(
                            f"{slot.claim.key}: claimed {slot.claim.value!r}; evidence "
                            f"{item.evidence_id!r} asserts {asserted.value!r} for an "
                            "overlapping time"
                        ),
                        values=(slot.claim.value, asserted.value),
                    )
                )
        return out

    def _spatial_contradictions(
        self, slots: Sequence[_ClaimSlot], evidence: Sequence[EvidenceItem]
    ) -> list[Contradiction]:
        """Impossible travel between located, timed observations of one subject."""
        observations: list[tuple[LocatedObservation, ClaimKey]] = []
        for slot in slots:
            c = slot.claim
            if c.location is not None and c.interval is not None:
                observations.append(
                    (LocatedObservation(c.subject, c.location, c.interval, slot.reference), c.key)
                )
        for item in evidence:
            asserted = item.asserted
            when = item.effective_time
            if asserted is not None and asserted.location is not None and when is not None:
                observations.append(
                    (
                        LocatedObservation(
                            asserted.subject, asserted.location, when, _evidence_reference(item)
                        ),
                        asserted.key,
                    )
                )
        out: list[Contradiction] = []
        for (a, key_a), (b, key_b) in combinations(observations, 2):
            if a.subject != b.subject:
                continue
            if a.reference.kind == "evidence" and b.reference.kind == "evidence":
                continue
            assessment = self.travel_check.assess(a, b)
            if assessment.feasible:
                continue
            out.append(
                Contradiction(
                    kind=ContradictionKind.SPATIAL,
                    scope=_scope(a.reference, b.reference),
                    keys=(key_a, key_b),
                    references=(a.reference, b.reference),
                    description=(
                        f"{a.subject} placed {assessment.effective_distance_km:.3f} km apart "
                        f"(beyond location radii) with {assessment.gap_seconds:.0f} s between "
                        f"observations; required speed {assessment.required_speed_kmh:.6g} km/h "
                        f"exceeds {assessment.max_speed_kmh:.6g} km/h"
                    ),
                    travel=assessment,
                )
            )
        return out

    def _event_times(
        self,
        key: ClaimKey,
        statement: Statement,
        evidence: Sequence[EvidenceItem],
        ambiguous: frozenset[str] = frozenset(),
    ) -> list[tuple[EvidenceItem, TimeInterval]]:
        """Evidence times for an event key, from items that report the event as the statement does.

        Target-form items whose target is ambiguous are skipped, as they are in
        corroboration: which claim they report on is unknown.
        """
        claims = [c for c in statement.claims if c.key == key]
        tolerance = self.config.tolerance_for(key.predicate)
        out: list[tuple[EvidenceItem, TimeInterval]] = []
        for item in evidence:
            when = item.effective_time
            if item.key != key or when is None or item.evidence_id in ambiguous:
                continue
            if item.asserted is None:
                if item.stance is not Stance.SUPPORTS:
                    continue
                if item.target_statement_id not in (None, statement.statement_id):
                    continue
            elif any(
                not values_compatible(item.asserted.value, c.value, tolerance) for c in claims
            ):
                continue
            out.append((item, when))
        return out

    def _temporal_contradictions(
        self,
        statements: Sequence[Statement],
        evidence: Sequence[EvidenceItem],
        ambiguous: frozenset[str] = frozenset(),
    ) -> list[Contradiction]:
        """Claimed orderings against evidence times, own claimed intervals and other orderings."""
        out: list[Contradiction] = []
        entries: list[tuple[Statement, int, TemporalOrdering, Reference]] = [
            (s, j, o, Reference("ordering", s.statement_id, j, s.speaker))
            for s in statements
            for j, o in enumerate(s.orderings)
        ]
        for statement, _j, ordering, oref in entries:
            first_times = self._event_times(ordering.first, statement, evidence, ambiguous)
            second_times = self._event_times(ordering.second, statement, evidence, ambiguous)
            for (ia, ta), (ib, tb) in product(first_times, second_times):
                if ia.evidence_id == ib.evidence_id:
                    continue
                relation = allen_relation(ta, tb)
                if relation in ordering.relations:
                    continue
                out.append(
                    Contradiction(
                        kind=ContradictionKind.TEMPORAL,
                        scope=ContradictionScope.CLAIM_VS_EVIDENCE,
                        keys=(ordering.first, ordering.second),
                        references=(oref, _evidence_reference(ia), _evidence_reference(ib)),
                        description=(
                            f"claimed {ordering.first} {_relations_text(ordering.relations)} "
                            f"{ordering.second}; evidence times give '{relation.value}'"
                        ),
                        observed_relation=relation,
                        claimed_relations=ordering.relations,
                    )
                )
            firsts = [
                (i, c)
                for i, c in enumerate(statement.claims)
                if c.key == ordering.first and c.interval is not None
            ]
            seconds = [
                (i, c)
                for i, c in enumerate(statement.claims)
                if c.key == ordering.second and c.interval is not None
            ]
            for (i, ca), (m, cb) in product(firsts, seconds):
                assert ca.interval is not None and cb.interval is not None
                relation = allen_relation(ca.interval, cb.interval)
                if relation in ordering.relations:
                    continue
                refs = (
                    oref,
                    Reference("claim", statement.statement_id, i, statement.speaker),
                    Reference("claim", statement.statement_id, m, statement.speaker),
                )
                out.append(
                    Contradiction(
                        kind=ContradictionKind.TEMPORAL,
                        scope=ContradictionScope.WITHIN_STATEMENT,
                        keys=(ordering.first, ordering.second),
                        references=refs,
                        description=(
                            f"claimed {ordering.first} {_relations_text(ordering.relations)} "
                            f"{ordering.second}; the statement's own claimed times give "
                            f"'{relation.value}'"
                        ),
                        observed_relation=relation,
                        claimed_relations=ordering.relations,
                    )
                )
        for (_s1, _j1, o1, r1), (_s2, _j2, o2, r2) in combinations(entries, 2):
            if (o1.first, o1.second) == (o2.first, o2.second):
                aligned = o2.relations
            elif (o1.first, o1.second) == (o2.second, o2.first):
                aligned = converse_relations(o2.relations)
            else:
                continue
            if o1.relations & aligned:
                continue
            out.append(
                Contradiction(
                    kind=ContradictionKind.TEMPORAL,
                    scope=_scope(r1, r2),
                    keys=(o1.first, o1.second),
                    references=(r1, r2),
                    description=(
                        f"{o1.first} {_relations_text(o1.relations)} {o1.second} versus "
                        f"{o1.first} {_relations_text(aligned)} {o1.second}: no common relation"
                    ),
                    claimed_relations=o1.relations | aligned,
                )
            )
        return out

    # --------------------------------------------------------- consistency
    def _speaker_consistency(
        self,
        speaker: str,
        statements: Sequence[Statement],
        slots_by_statement: Mapping[str, Sequence[_ClaimSlot]],
        evidence: Sequence[EvidenceItem],
        ambiguous: frozenset[str],
        contradictions: Sequence[Contradiction],
    ) -> SpeakerConsistency:
        """Consistency, SUE and adaptation for one speaker."""
        cfg = self.config
        ids = tuple(s.statement_id for s in statements)
        id_set = set(ids)

        def _mine(c: Contradiction) -> bool:
            return all(r.identifier in id_set for r in c.references)

        within = tuple(
            c for c in contradictions if c.scope is ContradictionScope.WITHIN_STATEMENT and _mine(c)
        )
        between = tuple(
            c for c in contradictions if c.scope is ContradictionScope.SAME_SPEAKER and _mine(c)
        )
        usable = [e for e in evidence if e.evidence_id not in ambiguous]

        cumulative: list[frozenset[str]] = []
        seen: set[str] = set()
        for s in statements:
            seen |= s.disclosed_evidence_ids
            cumulative.append(frozenset(seen))

        withheld_n = withheld_k = disclosed_n = disclosed_k = 0
        for s, disclosed in zip(statements, cumulative):
            for slot in slots_by_statement[s.statement_id]:
                for item in usable:
                    stance = self._bearing(item, slot)
                    if stance is None:
                        continue
                    refutes = stance is Stance.REFUTES
                    if item.evidence_id in disclosed:
                        disclosed_n += 1
                        disclosed_k += refutes
                    else:
                        withheld_n += 1
                        withheld_k += refutes
        withheld = wilson_interval(withheld_k, withheld_n, cfg.confidence_level)
        disclosed_p = wilson_interval(disclosed_k, disclosed_n, cfg.confidence_level)
        difference = (
            withheld.estimate - disclosed_p.estimate
            if withheld.estimate is not None and disclosed_p.estimate is not None
            else None
        )
        sue = SUEIndicator(withheld, disclosed_p, difference)

        changes: list[ClaimChange] = []
        events: list[AdaptationEvent] = []
        opportunities = 0
        for pos in range(len(statements) - 1):
            prev, nxt = statements[pos], statements[pos + 1]
            newly = cumulative[pos + 1] - cumulative[pos]
            prev_slots = slots_by_statement[prev.statement_id]
            next_slots = slots_by_statement[nxt.statement_id]
            adapted_by_key: dict[ClaimKey, list[str]] = defaultdict(list)
            for item in usable:
                if item.evidence_id not in newly:
                    continue
                for slot in prev_slots:
                    if self._bearing(item, slot) is not Stance.REFUTES:
                        continue
                    opportunities += 1
                    if any(
                        n.claim.key == slot.claim.key and self._bearing(item, n) is Stance.SUPPORTS
                        for n in next_slots
                    ):
                        events.append(
                            AdaptationEvent(
                                item.evidence_id,
                                slot.claim.key,
                                prev.statement_id,
                                nxt.statement_id,
                            )
                        )
                        if item.evidence_id not in adapted_by_key[slot.claim.key]:
                            adapted_by_key[slot.claim.key].append(item.evidence_id)
            changes.extend(_claim_changes(prev, nxt, adapted_by_key, cfg))
        adaptation = AdaptationIndicator(
            wilson_interval(len(events), opportunities, cfg.confidence_level), tuple(events)
        )
        return SpeakerConsistency(
            speaker=speaker,
            statement_ids=ids,
            within_statement=within,
            between_statements=between,
            changes=tuple(changes),
            sue=sue,
            adaptation=adaptation,
        )

    # -------------------------------------------------------------- verify
    def verify(
        self,
        statements: Iterable[Statement],
        evidence: Iterable[EvidenceItem] = (),
        *,
        verifiability_reference: Sequence[float] | None = None,
    ) -> VerificationReport:
        """Verify ``statements`` against ``evidence``.

        Args:
            statements: The statements (any order; ids must be unique).
            evidence: The evidence items (any order; ids must be unique).
            verifiability_reference: Optional reference distribution of
                verifiable-detail proportions from the same context.

        Returns:
            A frozen :class:`VerificationReport`.

        Raises:
            ValueError: On duplicate statement or evidence ids.
        """
        ordered = sorted(statements, key=lambda s: (s.made_at, s.statement_id))
        items = sorted(evidence, key=lambda e: e.evidence_id)
        _require_unique([s.statement_id for s in ordered], "statement_id")
        _require_unique([e.evidence_id for e in items], "evidence_id")
        slots_by_statement: dict[str, tuple[_ClaimSlot, ...]] = {
            s.statement_id: tuple(_ClaimSlot(s, i, c) for i, c in enumerate(s.claims))
            for s in ordered
        }
        slots = [slot for s in ordered for slot in slots_by_statement[s.statement_id]]
        ambiguous = self._ambiguous_items(items, slots)

        corroboration = tuple(self._corroborate(slot, items, ambiguous) for slot in slots)
        contradictions = tuple(
            self._value_contradictions(slots, items)
            + self._temporal_contradictions(ordered, items, ambiguous)
            + self._spatial_contradictions(slots, items)
        )
        speakers = sorted({s.speaker for s in ordered})
        consistency = tuple(
            self._speaker_consistency(
                speaker,
                [s for s in ordered if s.speaker == speaker],
                slots_by_statement,
                items,
                ambiguous,
                contradictions,
            )
            for speaker in speakers
        )
        verifiability = tuple(
            assess_verifiability(
                s, verifiability_reference, confidence=self.config.confidence_level
            )
            for s in ordered
        )
        summary = _summarise(ordered, items, corroboration, contradictions, self.config)
        return VerificationReport(
            corroboration=corroboration,
            contradictions=contradictions,
            consistency=consistency,
            verifiability=verifiability,
            summary=summary,
            settings=self.config,
        )


#: Name alias kept for the removed module's top-level class. Only the name is
#: kept: the old ``decipher_truth`` API produced fabricated outputs and is gone.
TruthDecipherFramework = StatementVerifier


def _require_unique(ids: Sequence[str], name: str) -> None:
    """Raise if ``ids`` contains duplicates."""
    seen: set[str] = set()
    for identifier in ids:
        if identifier in seen:
            raise ValueError(f"duplicate {name} {identifier!r}")
        seen.add(identifier)


def _relations_text(relations: frozenset[AllenRelation]) -> str:
    """Render a relation set deterministically."""
    names = sorted(r.value for r in relations)
    return names[0] if len(names) == 1 else "{" + ", ".join(names) + "}"


def _claim_changes(
    prev: Statement,
    nxt: Statement,
    adapted_by_key: Mapping[ClaimKey, Sequence[str]],
    cfg: VerifierConfig,
) -> list[ClaimChange]:
    """Changed, added and dropped claim keys between two successive statements."""
    out: list[ClaimChange] = []
    keys = sorted({c.key for c in prev.claims} | {c.key for c in nxt.claims})
    for key in keys:
        old = [c for c in prev.claims if c.key == key]
        new = [c for c in nxt.claims if c.key == key]
        old_values = tuple(c.value for c in old)
        new_values = tuple(c.value for c in new)
        if not new:
            kind: Literal["changed", "added", "dropped"] = "dropped"
        elif not old:
            kind = "added"
        else:
            tolerance = cfg.tolerance_for(key.predicate)
            changed = any(
                intervals_concurrent(a.interval, b.interval)
                and not values_compatible(a.value, b.value, tolerance)
                for a in old
                for b in new
            )
            if not changed:
                continue
            kind = "changed"
        out.append(
            ClaimChange(
                kind=kind,
                key=key,
                from_statement=prev.statement_id,
                to_statement=nxt.statement_id,
                previous_values=old_values,
                new_values=new_values,
                matches_newly_disclosed=tuple(adapted_by_key.get(key, ())),
            )
        )
    return out


def _summarise(
    statements: Sequence[Statement],
    evidence: Sequence[EvidenceItem],
    corroboration: Sequence[ClaimCorroboration],
    contradictions: Sequence[Contradiction],
    cfg: VerifierConfig,
) -> ReportSummary:
    """Aggregate counts and plain-language findings."""
    by_status = dict.fromkeys(CorroborationStatus, 0)
    for c in corroboration:
        by_status[c.status] += 1
    insufficient = sum(1 for c in corroboration if c.insufficient_evidence)
    by_kind = {kind.value: 0 for kind in ContradictionKind}
    by_scope = {scope.value: 0 for scope in ContradictionScope}
    for contradiction in contradictions:
        by_kind[contradiction.kind.value] += 1
        by_scope[contradiction.scope.value] += 1
    n_claims = len(corroboration)
    findings = (
        f"{n_claims} claim(s) assessed against {len(evidence)} evidence item(s): "
        f"{by_status[CorroborationStatus.CORROBORATED]} corroborated, "
        f"{by_status[CorroborationStatus.CONTRADICTED]} contradicted, "
        f"{by_status[CorroborationStatus.UNRESOLVED]} unresolved (corroborated at posterior "
        f">= {cfg.corroborated_at:g}, contradicted at <= {cfg.contradicted_at:g}, prior "
        f"{cfg.prior_probability:g}).",
        f"{insufficient} claim(s) had no informative evidence; their status is an abstention.",
        f"{by_kind['value']} value, {by_kind['temporal']} temporal and {by_kind['spatial']} "
        "spatial contradiction(s) found.",
    )
    return ReportSummary(
        n_statements=len(statements),
        n_claims=n_claims,
        n_evidence=len(evidence),
        n_corroborated=by_status[CorroborationStatus.CORROBORATED],
        n_contradicted=by_status[CorroborationStatus.CONTRADICTED],
        n_unresolved=by_status[CorroborationStatus.UNRESOLVED],
        n_insufficient_evidence=insufficient,
        contradictions_by_kind=MappingProxyType(by_kind),
        contradictions_by_scope=MappingProxyType(by_scope),
        findings=findings,
    )


# --------------------------------------------------------------------------- #
# Concealed Information Test.
# --------------------------------------------------------------------------- #
class Channel(StrEnum):
    """Physiological channels scored in the CIT."""

    SKIN_CONDUCTANCE = "skin_conductance"
    HEART_RATE = "heart_rate"
    RESPIRATION_LINE_LENGTH = "respiration_line_length"


#: Sign that turns each channel's amplitude into "larger means a stronger
#: response". A larger skin-conductance response amplitude is a stronger
#: response (+1). Heart-rate deceleration (a negative post-minus-baseline
#: change) and respiration line length suppression (a shorter line) are
#: responses in the negative direction (-1). These are the directions that the
#: CIT meta-analysis reports for probes (Meijer et al., 2014).
CHANNEL_SIGN: Mapping[Channel, int] = MappingProxyType(
    {
        Channel.SKIN_CONDUCTANCE: 1,
        Channel.HEART_RATE: -1,
        Channel.RESPIRATION_LINE_LENGTH: -1,
    }
)


@dataclass(frozen=True)
class ConsentRecord:
    """Informed consent for a Concealed Information Test.

    Attributes:
        subject_pseudonym: A pseudonymous subject id. The caller must not use
            a real name or contact detail.
        obtained_at: When consent was obtained (seconds, same clock as
            ``administered_at``).
        scope: What the consent covers. It must include
            :data:`CIT_CONSENT_SCOPE`.
        obtained_by: Who obtained it (a role or pseudonymous staff id).
        withdrawn_at: When the subject withdrew consent, if they have. A
            withdrawn record never authorises scoring again (GDPR Art. 7(3):
            withdrawal must be as easy as giving consent); data scored before
            the withdrawal stays scored.
    """

    subject_pseudonym: str
    obtained_at: float
    scope: frozenset[str]
    obtained_by: str
    withdrawn_at: float | None = None

    def __post_init__(self) -> None:
        """Validate the record and freeze the scope."""
        _non_empty(self.subject_pseudonym, "subject_pseudonym")
        _non_empty(self.obtained_by, "obtained_by")
        object.__setattr__(self, "obtained_at", _finite(self.obtained_at, "obtained_at"))
        if self.withdrawn_at is not None:
            withdrawn = _finite(self.withdrawn_at, "withdrawn_at")
            if withdrawn < self.obtained_at:
                raise ValueError("withdrawn_at must not precede obtained_at")
            object.__setattr__(self, "withdrawn_at", withdrawn)
        scope = frozenset({self.scope}) if isinstance(self.scope, str) else frozenset(self.scope)
        if not scope:
            raise ValueError("consent scope must not be empty")
        for entry in scope:
            _non_empty(entry, "consent scope entry")
        object.__setattr__(self, "scope", scope)


def require_consent(
    consent: ConsentRecord | None,
    subject_pseudonym: str,
    administered_at: float,
    *,
    scope: str = CIT_CONSENT_SCOPE,
) -> ConsentRecord:
    """Check that a consent record covers this administration.

    Args:
        consent: The consent record.
        subject_pseudonym: Pseudonymous id of the examinee.
        administered_at: When the test (or recording) was administered.
        scope: The scope the record must include; the CIT's by default.
            Other consent-gated entry points of the package pass their own
            (for example ``rf_vital_signs.VITAL_SIGNS_CONSENT_SCOPE``).

    Returns:
        The validated record.

    Raises:
        ConsentRequiredError: If the record is missing, of the wrong type,
            for another subject, does not include ``scope``, was obtained
            after the administration, or has been withdrawn.
    """
    if consent is None:
        raise ConsentRequiredError(f"a ConsentRecord with scope {scope!r} is required")
    if not isinstance(consent, ConsentRecord):
        raise ConsentRequiredError("consent must be a ConsentRecord")
    if consent.subject_pseudonym != subject_pseudonym:
        raise ConsentRequiredError("the consent record is for a different subject")
    if scope not in consent.scope:
        raise ConsentRequiredError(f"the consent scope does not include {scope!r}")
    if consent.obtained_at > _finite(administered_at, "administered_at"):
        raise ConsentRequiredError("consent was obtained after the test was administered")
    if consent.withdrawn_at is not None:
        raise ConsentRequiredError("the subject has withdrawn consent")
    return consent


@dataclass(frozen=True)
class CITQuestion:
    """One CIT question: a probe, ``k - 1`` irrelevant items and response amplitudes.

    Attributes:
        question_id: Unique identifier.
        probe: The event-relevant item.
        irrelevants: Items that an examinee without knowledge of the event
            cannot tell apart from the probe.
        responses: For each channel, a ``repetitions x k`` matrix of raw
            response amplitudes. Columns are in the order
            ``(probe, *irrelevants)``, whatever the presentation order was.
            Skin conductance is the response amplitude, heart rate is the
            post-minus-baseline change in bpm, and respiration is the line
            length (see :data:`CHANNEL_SIGN`).
        probe_publicly_disclosed: The probe detail is public. The question is
            then excluded, because an examinee without involvement would
            recognise it.
    """

    question_id: str
    probe: str
    irrelevants: tuple[str, ...]
    responses: Mapping[Channel, tuple[tuple[float, ...], ...]]
    probe_publicly_disclosed: bool = False

    def __post_init__(self) -> None:
        """Validate shapes and values, and freeze the responses."""
        _non_empty(self.question_id, "question_id")
        _non_empty(self.probe, "probe")
        if isinstance(self.irrelevants, str):
            raise TypeError("irrelevants must be a sequence of item labels, not a str")
        irrelevants = tuple(self.irrelevants)
        items = (self.probe, *irrelevants)
        if len(irrelevants) < 1:
            raise ValueError("a CIT question needs at least one irrelevant item")
        if len(set(items)) != len(items):
            raise ValueError("item labels must be unique within a question")
        k = len(items)
        frozen: dict[Channel, tuple[tuple[float, ...], ...]] = {}
        repetitions: int | None = None
        for raw_channel, rows in self.responses.items():
            # Channel is a StrEnum, so a mapping cannot hold one channel twice
            # (Channel.HEART_RATE == "heart_rate" and they hash alike).
            channel = Channel(raw_channel)
            arr = np.asarray(rows, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != k:
                raise ValueError(
                    f"{channel.value}: responses must be a repetitions x {k} matrix; "
                    f"got shape {arr.shape}"
                )
            if not bool(np.all(np.isfinite(arr))):
                raise ValueError(f"{channel.value}: responses must be finite")
            if repetitions is None:
                repetitions = int(arr.shape[0])
            elif arr.shape[0] != repetitions:
                raise ValueError("every channel must have the same number of repetitions")
            frozen[channel] = tuple(tuple(float(v) for v in row) for row in arr)
        if not frozen:
            raise ValueError("a CIT question needs at least one channel")
        ordered = {ch: frozen[ch] for ch in Channel if ch in frozen}
        object.__setattr__(self, "irrelevants", irrelevants)
        object.__setattr__(self, "responses", MappingProxyType(ordered))
        if not isinstance(self.probe_publicly_disclosed, bool):
            raise TypeError("probe_publicly_disclosed must be a bool")

    @property
    def items(self) -> tuple[str, ...]:
        """Item labels in column order ``(probe, *irrelevants)``."""
        return (self.probe, *self.irrelevants)

    @property
    def n_repetitions(self) -> int:
        """Number of repetitions."""
        return len(next(iter(self.responses.values())))


PrimaryTest = Literal["rank", "permutation"]


@dataclass(frozen=True)
class CITConfig:
    """Settings for :class:`ConcealedInformationTest`.

    Attributes:
        alpha: Significance level for the finding.
        primary_test: The prespecified test that decides the finding. Only one
            test decides, so alpha holds. The default ``"rank"`` is exact and
            distribution-free. ``"permutation"`` uses response magnitudes.
        channel_weights: Non-negative weight per channel (equal by default).
            A channel without a weight, or with weight 0, is not used.
        max_exact_permutations: Enumerate every relabelling up to this count.
        monte_carlo_permutations: Relabellings drawn above that count.
        seed: Seed for the Monte Carlo relabellings.
    """

    alpha: float = DEFAULT_ALPHA
    primary_test: PrimaryTest = "rank"
    channel_weights: Mapping[Channel, float] = field(
        default_factory=lambda: dict.fromkeys(Channel, DEFAULT_CHANNEL_WEIGHT)
    )
    max_exact_permutations: int = DEFAULT_MAX_EXACT_PERMUTATIONS
    monte_carlo_permutations: int = DEFAULT_MONTE_CARLO_PERMUTATIONS
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the settings and freeze the weights."""
        if not 0.0 < _finite(self.alpha, "alpha") < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.primary_test not in ("rank", "permutation"):
            raise ValueError(f"unknown primary_test {self.primary_test!r}")
        weights: dict[Channel, float] = {}
        for raw_channel, raw_weight in self.channel_weights.items():
            weight = _finite(raw_weight, f"weight of {raw_channel!r}")
            if weight < 0.0:
                raise ValueError("channel weights must be >= 0")
            weights[Channel(raw_channel)] = weight
        if not any(w > 0.0 for w in weights.values()):
            raise ValueError("at least one channel weight must be > 0")
        if self.max_exact_permutations < 1 or self.monte_carlo_permutations < 1:
            raise ValueError("permutation counts must be >= 1")
        object.__setattr__(self, "channel_weights", MappingProxyType(weights))


@dataclass(frozen=True)
class CITQuestionResult:
    """Scoring of one CIT question.

    Attributes:
        question_id: The question.
        probe: The probe label.
        n_items: ``k``.
        n_repetitions: Repetitions.
        channels_used: Channels that entered the combined score.
        channels_dropped: Channels left out, with reasons.
        item_scores: Combined standardised score per item, averaged over
            repetitions, in ``(probe, *irrelevants)`` order.
        probe_rank: Probe rank among the ``k`` items (1 is the strongest
            response). Ties count against the probe.
        standardised_difference: Probe score minus the mean irrelevant score,
            in within-subject z units.
    """

    question_id: str
    probe: str
    n_items: int
    n_repetitions: int
    channels_used: tuple[Channel, ...]
    channels_dropped: tuple[tuple[Channel, str], ...]
    item_scores: tuple[float, ...]
    probe_rank: int
    standardised_difference: float


@dataclass(frozen=True)
class ExcludedQuestion:
    """A question that was not scored, and why."""

    question_id: str
    reason: str


@dataclass(frozen=True)
class CITResult:
    """Result of a Concealed Information Test.

    Attributes:
        finding: :data:`FINDING_INDICATED` or :data:`FINDING_NOT_INDICATED`,
            decided by the primary test at ``alpha``. ``None`` when abstained.
        abstained: Whether the test abstained.
        abstention_reason: Why it abstained.
        alpha: Significance level.
        primary_test: The test that decided the finding.
        rank_sum: Sum of probe ranks over the scored questions.
        rank_test_p: Exact one-sided p-value ``P(T <= rank_sum)`` under H0.
        permutation_p: Item-relabelling permutation p-value.
        permutation_method: ``"exact"`` or ``"monte_carlo"``.
        permutations_evaluated: Relabellings enumerated or drawn.
        effect_size: Mean standardised probe-minus-irrelevant difference
            across questions (within-subject z units).
        mean_probe_rank: Mean probe rank.
        expected_mean_rank_h0: Its expectation under H0, ``mean((k + 1) / 2)``.
        lykken_score: Descriptive Lykken (1959) score: 2 per question where
            the probe ranks first and 1 where it ranks second. No cut-off is
            applied.
        lykken_max: ``2 x`` the number of scored questions.
        questions: Per-question scoring.
        excluded_questions: Questions not scored.
        subject_pseudonym: The consenting examinee's pseudonym.
        consent_scope: The consent scope.
        measures: What the test measures.
        assumptions: The guilty-knowledge assumptions the result relies on.
        not_evidence_of_guilt: Why recognition is not evidence of guilt.
        limitations: Known limitations.
    """

    finding: str | None
    abstained: bool
    abstention_reason: str | None
    alpha: float
    primary_test: PrimaryTest
    rank_sum: int | None
    rank_test_p: float | None
    permutation_p: float | None
    permutation_method: Literal["exact", "monte_carlo"] | None
    permutations_evaluated: int | None
    effect_size: float | None
    mean_probe_rank: float | None
    expected_mean_rank_h0: float | None
    lykken_score: int | None
    lykken_max: int | None
    questions: tuple[CITQuestionResult, ...]
    excluded_questions: tuple[ExcludedQuestion, ...]
    subject_pseudonym: str
    consent_scope: frozenset[str]
    measures: str = CIT_MEASURES
    assumptions: tuple[str, ...] = CIT_ASSUMPTIONS
    not_evidence_of_guilt: str = NOT_EVIDENCE_OF_GUILT
    limitations: tuple[str, ...] = CIT_LIMITATIONS

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping."""
        out = _jsonable(self)
        assert isinstance(out, dict)  # a dataclass always converts to a dict
        return out


def rank_sum_distribution(n_items: Sequence[int]) -> tuple[int, ...]:
    """Exact null distribution of a sum of independent discrete-uniform ranks.

    Under H0 the probe rank in question ``q`` is uniform on
    ``{1, ..., k_q}``, independently across questions. The rank sum
    ``T = sum R_q`` then has ``prod k_q`` equally likely rank vectors. This
    returns the number of vectors for each value of ``T``, from ``T = q`` to
    ``T = sum k_q``, by convolving the uniform counts in exact integer
    arithmetic.

    Args:
        n_items: ``k_q`` for each question (each ``>= 2``).

    Returns:
        Counts indexed by ``T - q``.

    Raises:
        ValueError: If no questions are given or some ``k_q < 2``.
    """
    ks = [int(k) for k in n_items]
    if not ks or any(k < 2 for k in ks):
        raise ValueError("need at least one question, each with k >= 2 items")
    counts = [1]
    for k in ks:
        new = [0] * (len(counts) + k - 1)
        for offset, count in enumerate(counts):
            for r in range(k):
                new[offset + r] += count
        counts = new
    return tuple(counts)


def exact_rank_sum_p_value(rank_sum: int, n_items: Sequence[int]) -> float:
    """One-sided exact p-value ``P(T <= rank_sum)`` of the CIT rank-sum test.

    A recognised probe is expected to give the strongest response (rank 1),
    so small rank sums are evidence against H0.

    Args:
        rank_sum: Observed sum of probe ranks.
        n_items: ``k_q`` for each question.

    Returns:
        The exact p-value (the ratio of integers, correctly rounded).
    """
    ks = [int(k) for k in n_items]
    counts = rank_sum_distribution(ks)
    # The support starts at T = q (every probe ranked first).
    upto = int(rank_sum) - len(ks)
    if upto < 0:
        return 0.0
    return sum(counts[: upto + 1]) / math.prod(ks)


def _standardised_scores(
    question: CITQuestion, weights: Mapping[Channel, float]
) -> tuple[npt.NDArray[np.float64] | None, tuple[Channel, ...], tuple[tuple[Channel, str], ...]]:
    """Ben-Shakhar within-question z-scores, signed and weighted, averaged over repetitions."""
    combined: npt.NDArray[np.float64] | None = None
    weight_sum = 0.0
    used: list[Channel] = []
    dropped: list[tuple[Channel, str]] = []
    for channel, rows in question.responses.items():
        weight = weights.get(channel, 0.0)
        if weight <= 0.0:
            dropped.append((channel, "no positive weight configured"))
            continue
        x = np.asarray(rows, dtype=np.float64)
        sd = float(np.std(x))
        scale = max(1.0, float(np.max(np.abs(x))))
        if sd <= _ZERO_VARIANCE_RTOL * scale:
            dropped.append((channel, "zero variance within the question"))
            continue
        z = (x - float(np.mean(x))) / sd
        term = weight * CHANNEL_SIGN[channel] * z
        combined = term if combined is None else combined + term
        weight_sum += weight
        used.append(channel)
    if combined is None:
        return None, tuple(used), tuple(dropped)
    item_scores = np.asarray(np.mean(combined / weight_sum, axis=0), dtype=np.float64)
    return item_scores, tuple(used), tuple(dropped)


def _relabel_differences(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``d_j = s_j - mean(s_others)`` for every item ``j`` taken as the probe."""
    k = scores.size
    total = float(np.sum(scores))
    return np.asarray(scores - (total - scores) / (k - 1), dtype=np.float64)


def _permutation_p_value(
    tables: Sequence[npt.NDArray[np.float64]], config: CITConfig
) -> tuple[float, Literal["exact", "monte_carlo"], int]:
    """Item-relabelling permutation p-value for the summed probe differences.

    Under H0 the probe label is exchangeable among a question's items, and
    questions are relabelled independently. So every assignment of the probe
    label, one item per question, is equally likely. The statistic is the sum
    over questions of the probe-minus-mean-irrelevant difference. The p-value
    is the share of relabellings whose statistic is at least the observed
    one. All ``prod k_q`` relabellings are enumerated when that count is at
    most ``max_exact_permutations``. Otherwise ``monte_carlo_permutations``
    relabellings are drawn with ``seed`` and ``(1 + count) / (1 + B)`` is
    reported, which is never zero (Phipson & Smyth, 2010).
    """
    observed = 0.0
    for table in tables:
        observed += float(table[0])
    tie_tol = 1e-12 * max(1.0, abs(observed))
    total = math.prod(t.size for t in tables)
    if total <= config.max_exact_permutations:
        sums = np.zeros(1, dtype=np.float64)
        for table in tables:
            sums = (sums[:, None] + table[None, :]).ravel()
        count = int(np.count_nonzero(sums >= observed - tie_tol))
        return count / total, "exact", total
    rng = np.random.default_rng(config.seed)
    draws = np.zeros(config.monte_carlo_permutations, dtype=np.float64)
    for table in tables:
        draws += table[rng.integers(0, table.size, size=config.monte_carlo_permutations)]
    count = int(np.count_nonzero(draws >= observed - tie_tol))
    return (
        (count + 1) / (config.monte_carlo_permutations + 1),
        "monte_carlo",
        config.monte_carlo_permutations,
    )


class ConcealedInformationTest:
    """Consent-gated Concealed Information Test scoring.

    The test measures recognition of probe details, not the accuracy of any
    statement, and its result is not evidence of guilt. See the module
    docstring for the method and the guilty-knowledge assumptions.
    """

    def __init__(self, config: CITConfig | None = None) -> None:
        """Create a scorer with ``config`` (declared defaults when omitted)."""
        self.config = config or CITConfig()

    def _abstain(
        self,
        reason: str,
        consent: ConsentRecord,
        excluded: tuple[ExcludedQuestion, ...] = (),
    ) -> CITResult:
        """Build an abstaining result."""
        return CITResult(
            finding=None,
            abstained=True,
            abstention_reason=reason,
            alpha=self.config.alpha,
            primary_test=self.config.primary_test,
            rank_sum=None,
            rank_test_p=None,
            permutation_p=None,
            permutation_method=None,
            permutations_evaluated=None,
            effect_size=None,
            mean_probe_rank=None,
            expected_mean_rank_h0=None,
            lykken_score=None,
            lykken_max=None,
            questions=(),
            excluded_questions=excluded,
            subject_pseudonym=consent.subject_pseudonym,
            consent_scope=consent.scope,
        )

    def run(
        self,
        questions: Iterable[CITQuestion],
        *,
        consent: ConsentRecord | None,
        subject_pseudonym: str,
        administered_at: float,
        probe_details_publicly_disclosed: bool = False,
    ) -> CITResult:
        """Score a CIT session.

        Consent is checked before any response is scored. Building questions
        (:func:`question_from_time_series`, :func:`derive_response_amplitudes`)
        only reduces recordings to amplitudes; recording and holding those
        data must itself be covered by the consent the caller obtained.

        Args:
            questions: The session's questions (unique ids).
            consent: The examinee's consent record.
            subject_pseudonym: Pseudonymous id of the examinee.
            administered_at: When the session took place.
            probe_details_publicly_disclosed: The probe details of the whole
                session are public, so the test abstains.

        Returns:
            The :class:`CITResult`.

        Raises:
            ConsentRequiredError: Without a valid consent record.
            ValueError: On duplicate question ids.
        """
        record = require_consent(consent, subject_pseudonym, administered_at)
        qs = list(questions)
        _require_unique([q.question_id for q in qs], "question_id")
        if probe_details_publicly_disclosed:
            return self._abstain(
                "probe details were publicly disclosed: an examinee without involvement "
                "could recognise them",
                record,
            )
        excluded: list[ExcludedQuestion] = []
        scored: list[CITQuestionResult] = []
        tables: list[npt.NDArray[np.float64]] = []
        for question in qs:
            if question.probe_publicly_disclosed:
                excluded.append(
                    ExcludedQuestion(question.question_id, "probe detail publicly disclosed")
                )
                continue
            scores, used, dropped = _standardised_scores(question, self.config.channel_weights)
            if scores is None:
                excluded.append(
                    ExcludedQuestion(question.question_id, "no channel carried information")
                )
                continue
            differences = _relabel_differences(scores)
            probe_score = float(scores[0])
            rank = 1 + int(np.count_nonzero(scores[1:] >= probe_score))
            scored.append(
                CITQuestionResult(
                    question_id=question.question_id,
                    probe=question.probe,
                    n_items=int(scores.size),
                    n_repetitions=question.n_repetitions,
                    channels_used=used,
                    channels_dropped=dropped,
                    item_scores=tuple(float(s) for s in scores),
                    probe_rank=rank,
                    standardised_difference=float(differences[0]),
                )
            )
            tables.append(differences)
        if not scored:
            return self._abstain("no question could be scored", record, tuple(excluded))
        ks = [q.n_items for q in scored]
        rank_sum = sum(q.probe_rank for q in scored)
        rank_p = exact_rank_sum_p_value(rank_sum, ks)
        perm_p, method, n_perm = _permutation_p_value(tables, self.config)
        primary_p = rank_p if self.config.primary_test == "rank" else perm_p
        finding = FINDING_INDICATED if primary_p <= self.config.alpha else FINDING_NOT_INDICATED
        n_q = len(scored)
        return CITResult(
            finding=finding,
            abstained=False,
            abstention_reason=None,
            alpha=self.config.alpha,
            primary_test=self.config.primary_test,
            rank_sum=rank_sum,
            rank_test_p=rank_p,
            permutation_p=perm_p,
            permutation_method=method,
            permutations_evaluated=n_perm,
            effect_size=math.fsum(q.standardised_difference for q in scored) / n_q,
            mean_probe_rank=rank_sum / n_q,
            expected_mean_rank_h0=math.fsum((k + 1) / 2.0 for k in ks) / n_q,
            lykken_score=sum(
                2 if q.probe_rank == 1 else 1 if q.probe_rank == 2 else 0 for q in scored
            ),
            lykken_max=2 * n_q,
            questions=tuple(scored),
            excluded_questions=tuple(excluded),
            subject_pseudonym=record.subject_pseudonym,
            consent_scope=record.scope,
        )


# --------------------------------------------------------------------------- #
# Response amplitudes from physiological time series.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class TimeSeries:
    """A sampled physiological signal (copied and made read-only on construction).

    Attributes:
        times: Sample times in seconds, strictly increasing.
        values: Sample values: skin conductance (microsiemens), heart rate
            (bpm) or a respiration displacement signal (arbitrary units).
    """

    times: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate, copy and freeze the arrays."""
        t = np.array(self.times, dtype=np.float64, copy=True)
        v = np.array(self.values, dtype=np.float64, copy=True)
        if t.ndim != 1 or v.shape != t.shape or t.size == 0:
            raise ValueError("times and values must be equal-length, non-empty 1-D arrays")
        if not bool(np.all(np.isfinite(t))) or not bool(np.all(np.isfinite(v))):
            raise ValueError("times and values must be finite")
        if t.size > 1 and bool(np.any(np.diff(t) <= 0.0)):
            raise ValueError("times must be strictly increasing")
        t.flags.writeable = False
        v.flags.writeable = False
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "values", v)

    def window(self, start: float, end: float) -> npt.NDArray[np.float64]:
        """Values with ``start <= t < end`` (half-open window)."""
        mask = (self.times >= start) & (self.times < end)
        return np.asarray(self.values[mask], dtype=np.float64)


def _check_window(window: tuple[float, float], name: str) -> tuple[float, float]:
    """Validate a ``(start, end)`` offset window."""
    start, end = (_finite(window[0], f"{name} start"), _finite(window[1], f"{name} end"))
    if start >= end:
        raise ValueError(f"{name} must have start < end; got {window}")
    return start, end


@dataclass(frozen=True)
class ResponseWindows:
    """Scoring windows, in seconds relative to stimulus onset (declared defaults).

    Attributes:
        scr: Skin-conductance window for the largest trough-to-peak rise.
        hr_baseline: Pre-stimulus heart-rate baseline window.
        hr_response: Post-stimulus heart-rate window.
        rll: Respiration line length window.
        allow_overlap: Permit a response window to run past the next onset.
            Off by default, because the next stimulus contaminates the response.
    """

    scr: tuple[float, float] = DEFAULT_SCR_WINDOW
    hr_baseline: tuple[float, float] = DEFAULT_HR_BASELINE_WINDOW
    hr_response: tuple[float, float] = DEFAULT_HR_RESPONSE_WINDOW
    rll: tuple[float, float] = DEFAULT_RLL_WINDOW
    allow_overlap: bool = False

    def __post_init__(self) -> None:
        """Validate the windows."""
        for name in ("scr", "hr_baseline", "hr_response", "rll"):
            object.__setattr__(self, name, _check_window(getattr(self, name), name))
        if self.hr_baseline[1] > self.hr_response[0]:
            raise ValueError("the heart-rate baseline must end before the response window starts")


def skin_conductance_amplitude(
    series: TimeSeries, onset: float, window: tuple[float, float] = DEFAULT_SCR_WINDOW
) -> float:
    """Largest trough-to-peak rise of skin conductance within the scoring window.

    The amplitude is ``max over i <= j of (v_j - v_i)`` over the samples in
    ``[onset + window[0], onset + window[1])``. It is ``0`` when the signal
    only falls.

    Args:
        series: Skin conductance samples.
        onset: Stimulus onset time.
        window: Offset window in seconds.

    Returns:
        The amplitude, in the signal's units.

    Raises:
        InsufficientDataError: If the window holds fewer than two samples.
    """
    start, end = _check_window(window, "scr window")
    seg = series.window(onset + start, onset + end)
    if seg.size < 2:
        raise InsufficientDataError(f"skin conductance window at onset {onset} has < 2 samples")
    return float(np.max(seg - np.minimum.accumulate(seg)))


def heart_rate_change(
    series: TimeSeries,
    onset: float,
    baseline: tuple[float, float] = DEFAULT_HR_BASELINE_WINDOW,
    response: tuple[float, float] = DEFAULT_HR_RESPONSE_WINDOW,
) -> float:
    """Mean post-stimulus heart rate minus the mean pre-stimulus baseline (bpm).

    A negative value is a deceleration, which :data:`CHANNEL_SIGN` scores as a
    stronger response. Heart-rate series from contactless (radio-frequency)
    sensing enter here.

    Args:
        series: Heart-rate samples in bpm.
        onset: Stimulus onset time.
        baseline: Baseline offset window.
        response: Response offset window.

    Returns:
        The change in bpm.

    Raises:
        InsufficientDataError: If either window holds no samples.
    """
    b0, b1 = _check_window(baseline, "hr baseline")
    r0, r1 = _check_window(response, "hr response")
    pre = series.window(onset + b0, onset + b1)
    post = series.window(onset + r0, onset + r1)
    if pre.size == 0 or post.size == 0:
        raise InsufficientDataError(f"heart-rate window at onset {onset} has no samples")
    return float(np.mean(post)) - float(np.mean(pre))


def respiration_line_length(
    series: TimeSeries, onset: float, window: tuple[float, float] = DEFAULT_RLL_WINDOW
) -> float:
    """Respiration line length: total variation ``sum |v[i+1] - v[i]|`` within the window.

    This is the unit-free form of Timm's (1982) line length. Scaling time
    would change the Euclidean tracing length, but it does not change the
    total variation, and within-question standardisation removes the
    amplitude scale. A shorter line (suppressed breathing) is a response in
    the negative direction.

    Args:
        series: Respiration displacement samples.
        onset: Stimulus onset time.
        window: Offset window in seconds.

    Returns:
        The line length, in the signal's units.

    Raises:
        InsufficientDataError: If the window holds fewer than two samples.
    """
    start, end = _check_window(window, "rll window")
    seg = series.window(onset + start, onset + end)
    if seg.size < 2:
        raise InsufficientDataError(f"respiration window at onset {onset} has < 2 samples")
    return float(np.sum(np.abs(np.diff(seg))))


def derive_response_amplitudes(
    onsets: Sequence[float],
    *,
    skin_conductance: TimeSeries | None = None,
    heart_rate: TimeSeries | None = None,
    respiration: TimeSeries | None = None,
    windows: ResponseWindows | None = None,
) -> Mapping[Channel, tuple[float, ...]]:
    """Response amplitude per channel for each stimulus onset.

    Args:
        onsets: Stimulus onset times (seconds).
        skin_conductance: Skin conductance series, if recorded.
        heart_rate: Heart-rate series (bpm), if recorded.
        respiration: Respiration series, if recorded.
        windows: Scoring windows (declared defaults when omitted).

    Returns:
        For each supplied channel, one amplitude per onset in input order.

    Raises:
        ValueError: If no series is supplied.
        InsufficientDataError: If a window lacks samples.
    """
    w = windows or ResponseWindows()
    times = [_finite(t, "onset") for t in onsets]
    out: dict[Channel, tuple[float, ...]] = {}
    if skin_conductance is not None:
        out[Channel.SKIN_CONDUCTANCE] = tuple(
            skin_conductance_amplitude(skin_conductance, t, w.scr) for t in times
        )
    if heart_rate is not None:
        out[Channel.HEART_RATE] = tuple(
            heart_rate_change(heart_rate, t, w.hr_baseline, w.hr_response) for t in times
        )
    if respiration is not None:
        out[Channel.RESPIRATION_LINE_LENGTH] = tuple(
            respiration_line_length(respiration, t, w.rll) for t in times
        )
    if not out:
        raise ValueError("supply at least one physiological series")
    return MappingProxyType(out)


def question_from_time_series(
    question_id: str,
    probe: str,
    irrelevants: Sequence[str],
    presentations: Sequence[tuple[str, float]],
    *,
    skin_conductance: TimeSeries | None = None,
    heart_rate: TimeSeries | None = None,
    respiration: TimeSeries | None = None,
    windows: ResponseWindows | None = None,
    probe_publicly_disclosed: bool = False,
) -> CITQuestion:
    """Build a :class:`CITQuestion` from physiological series and stimulus presentations.

    Each item must be presented the same number of times. Repetition ``r`` of
    an item is its ``r``-th presentation in time order.

    Args:
        question_id: Question identifier.
        probe: Probe label.
        irrelevants: Irrelevant item labels.
        presentations: ``(item label, onset time)`` pairs, in any order.
        skin_conductance: Skin conductance series, if recorded.
        heart_rate: Heart-rate series (bpm), if recorded.
        respiration: Respiration series, if recorded.
        windows: Scoring windows (declared defaults when omitted).
        probe_publicly_disclosed: The probe detail is public.

    Returns:
        The question, with amplitudes in ``(probe, *irrelevants)`` column order.

    Raises:
        ValueError: On unknown labels, unequal repetition counts, or (unless
            ``windows.allow_overlap``) response windows that run past the next
            onset.
        InsufficientDataError: If a window lacks samples.
    """
    w = windows or ResponseWindows()
    items = (probe, *tuple(irrelevants))
    ordered = sorted(((_finite(t, "onset"), label) for label, t in presentations))
    if not ordered:
        raise ValueError("no presentations given")
    unknown = {label for _, label in ordered} - set(items)
    if unknown:
        raise ValueError(f"presentations name unknown items {sorted(unknown)}")
    ends: list[float] = []
    starts: list[float] = [0.0]
    if skin_conductance is not None:
        ends.append(w.scr[1])
        starts.append(w.scr[0])
    if heart_rate is not None:
        ends.append(w.hr_response[1])
        starts.append(w.hr_baseline[0])
    if respiration is not None:
        ends.append(w.rll[1])
        starts.append(w.rll[0])
    if ends and not w.allow_overlap:
        # The next item's earliest window (a pre-onset baseline starts before
        # its onset) must begin after this item's latest window ends.
        reach = max(ends) - min(starts)
        for (t0, _), (t1, _) in pairwise(ordered):
            if t1 - t0 < reach:
                raise ValueError(
                    f"onsets {t0} and {t1} are {t1 - t0} s apart; response windows reach "
                    f"{reach} s (set allow_overlap to accept contamination)"
                )
    amplitudes = derive_response_amplitudes(
        [t for t, _ in ordered],
        skin_conductance=skin_conductance,
        heart_rate=heart_rate,
        respiration=respiration,
        windows=w,
    )
    positions: dict[str, list[int]] = {label: [] for label in items}
    for idx, (_, label) in enumerate(ordered):
        positions[label].append(idx)
    counts = {len(v) for v in positions.values()}
    if len(counts) != 1:
        raise ValueError("every item must be presented the same number of times")
    repetitions = counts.pop()
    responses = {
        channel: tuple(
            tuple(values[positions[label][r]] for label in items) for r in range(repetitions)
        )
        for channel, values in amplitudes.items()
    }
    return CITQuestion(
        question_id=question_id,
        probe=probe,
        irrelevants=tuple(irrelevants),
        responses=responses,
        probe_publicly_disclosed=probe_publicly_disclosed,
    )


# --------------------------------------------------------------------------- #
# CIT design power analysis.
# --------------------------------------------------------------------------- #
#: Shortest inter-stimulus interval the default response windows allow when
#: all three channels are recorded: the latest window end (10 s) minus the
#: earliest window start (the 3 s pre-onset heart-rate baseline). Session
#: length is ``presentations * MIN_ISI_S``.
MIN_ISI_S = DEFAULT_HR_RESPONSE_WINDOW[1] - DEFAULT_HR_BASELINE_WINDOW[0]


@dataclass(frozen=True)
class CITDesign:
    """A CIT session design: how many questions, items and repetitions.

    Attributes:
        questions: Questions per session.
        items: Items per question, the probe included.
        repetitions: Presentations of each item.
    """

    questions: int
    items: int
    repetitions: int

    def __post_init__(self) -> None:
        """Validate the counts."""
        if self.questions < 1 or self.items < 2 or self.repetitions < 1:
            raise ValueError("need questions >= 1, items >= 2 and repetitions >= 1")

    @property
    def presentations(self) -> int:
        """Stimulus presentations per session."""
        return self.questions * self.items * self.repetitions

    @property
    def minutes(self) -> float:
        """Session length at :data:`MIN_ISI_S` between presentations."""
        return self.presentations * MIN_ISI_S / 60.0


@dataclass(frozen=True)
class CITPower:
    """Simulated rejection rate of a design under one alternative."""

    design: CITDesign
    primary_test: str
    effect_sd: Mapping[Channel, float]
    alpha: float
    power: Proportion


def cit_power(
    design: CITDesign,
    effect_sd: Mapping[Channel, float] | Mapping[str, float],
    *,
    primary_test: PrimaryTest = "rank",
    alpha: float = DEFAULT_ALPHA,
    sessions: int = 1000,
    seed: int = 0,
    config: CITConfig | None = None,
) -> CITPower:
    """Simulate the power of a CIT design with :class:`ConcealedInformationTest` itself.

    Every amplitude is standard normal within subject and channel; the probe
    is shifted by ``effect_sd[channel]`` standard deviations on every
    repetition, in the direction :data:`CHANNEL_SIGN` scores as a response.
    All three channels are recorded; a channel missing from ``effect_sd``
    carries no effect (pure noise). The rejection rate at ``alpha`` comes with
    a Wilson interval. ``effect_sd`` all zero gives the type-I error.

    Args:
        design: The session design.
        effect_sd: Probe shift per channel, in within-subject SDs.
        primary_test: ``"rank"`` or ``"permutation"``.
        alpha: Significance level.
        sessions: Simulated sessions.
        seed: Seed for the amplitudes and the permutation test.
        config: Scoring configuration; ``alpha``, ``primary_test`` and
            ``seed`` above override its fields.

    Returns:
        The :class:`CITPower`.
    """
    effects = {Channel(k): _finite(v, f"effect of {k!r}") for k, v in effect_sd.items()}
    base = config or CITConfig()
    cit = ConcealedInformationTest(replace(base, alpha=alpha, primary_test=primary_test, seed=seed))
    consent = ConsentRecord("power-simulation", 0.0, frozenset({CIT_CONSENT_SCOPE}), "simulation")
    rng = np.random.default_rng(seed)
    labels = tuple(f"irrelevant-{i}" for i in range(1, design.items))
    rejected = 0
    for s in range(sessions):
        questions = []
        for q in range(design.questions):
            responses: dict[Channel, tuple[tuple[float, ...], ...]] = {}
            for channel in Channel:
                amplitudes = rng.standard_normal((design.repetitions, design.items))
                amplitudes[:, 0] += CHANNEL_SIGN[channel] * effects.get(channel, 0.0)
                responses[channel] = tuple(tuple(float(v) for v in row) for row in amplitudes)
            questions.append(CITQuestion(f"q{q}", "probe", labels, responses))
        result = cit.run(
            questions,
            consent=consent,
            subject_pseudonym="power-simulation",
            administered_at=s + 1.0,
        )
        p = result.rank_test_p if primary_test == "rank" else result.permutation_p
        rejected += int(p is not None and p <= alpha)
    return CITPower(
        design=design,
        primary_test=primary_test,
        effect_sd=MappingProxyType(effects),
        alpha=alpha,
        power=wilson_interval(rejected, sessions, 0.95),
    )


def plan_cit_design(
    effect_sd: Mapping[Channel, float] | Mapping[str, float],
    *,
    target_power: float = 0.8,
    primary_test: PrimaryTest = "rank",
    alpha: float = DEFAULT_ALPHA,
    questions: Sequence[int] = (4, 6, 8, 10, 12),
    items: Sequence[int] = (5, 6),
    repetitions: Sequence[int] = (2, 3, 4, 5),
    sessions: int = 1000,
    seed: int = 0,
) -> tuple[CITPower | None, tuple[CITPower, ...]]:
    """The shortest design whose power's Wilson lower bound reaches ``target_power``.

    Designs are tried in order of session length (presentations); the first
    whose 95% Wilson lower bound on simulated power is at least
    ``target_power`` is returned, so the guarantee holds with the simulation's
    own uncertainty taken into account. Every design tried is returned too.

    Args:
        effect_sd: Assumed probe shift per channel (see :func:`cit_power`).
        target_power: Required power.
        primary_test: The test the session will be scored with.
        alpha: Significance level.
        questions: Candidate questions per session.
        items: Candidate items per question.
        repetitions: Candidate repetitions.
        sessions: Simulated sessions per design.
        seed: Simulation seed.

    Returns:
        ``(chosen, tried)``; ``chosen`` is ``None`` when no candidate reaches
        the target.
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    candidates = sorted(
        (CITDesign(q, k, r) for q in questions for k in items for r in repetitions),
        key=lambda d: (d.presentations, d.repetitions, d.items),
    )
    tried: list[CITPower] = []
    for design in candidates:
        measured = cit_power(
            design, effect_sd, primary_test=primary_test, alpha=alpha, sessions=sessions, seed=seed
        )
        tried.append(measured)
        lower = measured.power.lower
        if lower is not None and lower >= target_power:
            return measured, tuple(tried)
    return None, tuple(tried)


__all__ = [
    "CHANNEL_SIGN",
    "CIT_ASSUMPTIONS",
    "CIT_CONSENT_SCOPE",
    "CIT_LIMITATIONS",
    "CIT_MEASURES",
    "CONCURRENT_RELATIONS",
    "COVERED_RELATIONS",
    "DEFAULT_ALPHA",
    "DEFAULT_CONTRADICTED_AT",
    "DEFAULT_CORROBORATED_AT",
    "DEFAULT_MAX_SPEED_KMH",
    "DEFAULT_PRIOR_PROBABILITY",
    "DEFAULT_RELIABILITY_EPSILON",
    "EARTH_MEAN_RADIUS_KM",
    "FINDING_INDICATED",
    "FINDING_NOT_INDICATED",
    "MIN_ISI_S",
    "NOT_EVIDENCE_OF_GUILT",
    "SCIENTIFIC_FRAMING",
    "VERIFIER_LIMITATIONS",
    "AdaptationEvent",
    "AdaptationIndicator",
    "AllenRelation",
    "CITConfig",
    "CITDesign",
    "CITPower",
    "CITQuestion",
    "CITQuestionResult",
    "CITResult",
    "Channel",
    "Claim",
    "ClaimChange",
    "ClaimCorroboration",
    "ClaimKey",
    "ClaimValue",
    "ClusterContribution",
    "ConcealedInformationTest",
    "ConsentRecord",
    "ConsentRequiredError",
    "Contradiction",
    "ContradictionKind",
    "ContradictionScope",
    "CorroborationStatus",
    "DeclaredIndependenceClusters",
    "Detail",
    "EvidenceItem",
    "ExcludedEvidence",
    "ExcludedQuestion",
    "HaversineTravelCheck",
    "IndependenceClusterModel",
    "InsufficientDataError",
    "ItemWeight",
    "LocatedObservation",
    "Location",
    "Proportion",
    "Reference",
    "ReportSummary",
    "ResponseWindows",
    "SUEIndicator",
    "SourceReliabilityModel",
    "SpeakerConsistency",
    "Stance",
    "Statement",
    "StatementVerifier",
    "TableSourceReliability",
    "TemporalOrdering",
    "TimeInterval",
    "TimeSeries",
    "TravelAssessment",
    "TravelFeasibilityCheck",
    "TruthDecipherFramework",
    "ValueTolerance",
    "VerifiabilityResult",
    "VerificationReport",
    "VerifierConfig",
    "allen_relation",
    "assess_verifiability",
    "cit_power",
    "converse",
    "converse_relations",
    "derive_response_amplitudes",
    "exact_rank_sum_p_value",
    "haversine_km",
    "heart_rate_change",
    "interval_covered",
    "intervals_concurrent",
    "log_likelihood_ratio",
    "plan_cit_design",
    "question_from_time_series",
    "rank_sum_distribution",
    "require_consent",
    "respiration_line_length",
    "skin_conductance_amplitude",
    "values_compatible",
    "wilson_interval",
]
