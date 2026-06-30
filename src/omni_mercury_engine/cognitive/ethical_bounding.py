# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ethical Bounding and Benevolence Scoring.

Implements Phase 6 of the neuro-symbolic evolution:
- Hardcoded utility function for scoring actions on good/evil metrics
- Benevolence threshold enforcement (>=0.99 required)
- Empathy modules for human-centric choices
- Value preservation for positive outcomes
- Audit mechanisms for alignment verification

Research Sources:
- AI Safety (Amodei et al., 2016)
- Value Alignment (Russell, 2019)
- Ethical AI (Floridi & Cowls, 2019)
- Gini Coefficient for Equity (Gini, 1912)

Integration:
    This module provides ethical bounding that integrates with
    the autonomous agent to ensure all actions meet benevolence
    requirements before execution.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard ethical floor — callers cannot configure the benevolence threshold
# below this value, regardless of domain or operational mode.
# ---------------------------------------------------------------------------
MINIMUM_BENEVOLENCE_FLOOR: float = 0.70


@dataclass(frozen=True)
class BenevolenceCalibration:
    """Calibration knobs for the benevolence scorer.

    These are the parameters meant to be *fit on labeled decisions* (via
    ``tools/benevolence_calibration_report.py`` / ``benevolence_certifier.py``)
    rather than hand-set. They are gathered here, version-pinned, and frozen so a
    change is explicit and invalidates the benevolence cache (bump
    ``ETHICAL.RULESET_VERSION``). The component weights must sum to 1.
    """

    w_harm: float = 0.30
    w_benefit: float = 0.25
    w_equity: float = 0.20
    w_principles: float = 0.15
    w_long_term: float = 0.10
    # Strength of the severity x irreversibility damping (multiplicative, <=1).
    severity_gamma: float = 0.5
    # Char-trigram cosine above which a word counts as a semantic match of a
    # harm/benefit keyword (catches morphological variants the substring scan
    # misses: "injuries" -> "injury", "manipulative" -> "manipulate").
    semantic_match_threshold: float = 0.6


BENEVOLENCE_CALIBRATION = BenevolenceCalibration()


def _det_hash(s: str) -> int:
    """Deterministic (process-independent) string hash.

    Python's built-in ``hash`` is salted per process (PYTHONHASHSEED), which
    would make benevolence scores non-reproducible and break the cache /
    certifier. This polynomial rolling hash is stable across runs.
    """
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


@functools.lru_cache(maxsize=4096)
def _char_trigram_vector(word: str, dim: int = 512) -> tuple[float, ...]:
    """Deterministic char-trigram term-frequency vector for a word (padded)."""
    v = [0.0] * dim
    w = f"  {word.lower()} "
    for i in range(len(w) - 2):
        v[_det_hash(w[i : i + 3]) % dim] += 1.0
    return tuple(v)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    av = np.asarray(a)
    bv = np.asarray(b)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


# ---------------------------------------------------------------------------
# Euphemism / paraphrase lexicon for lethal-and-grave harm intent.
#
# Char-trigram matching (below) catches *morphology* ("injuries" ~ "injury"),
# NOT *meaning*: an adversary who writes "make them disappear" or "put him down"
# instead of "kill" sails straight past it. This curated lexicon closes that gap
# deterministically, with no model dependency, by matching the finite set of
# euphemisms real harmful intent actually uses.
#
# Design constraint -- DO NOT false-positive on Mercury's own defensive/security
# language. Bare verbs like "neutralize", "eliminate", "terminate", "kill" (as in
# "kill the process" / "neutralize the threat") are legitimate defensive idioms,
# so they are deliberately EXCLUDED here. Only entries that are unambiguous about
# harm *to a person* are included: standalone lethal idioms, and verb phrases
# pinned to a human object via :data:`_EUPHEMISM_OBJECT`. The lexicon is auditable
# and version-pinned by ``ETHICAL.RULESET_VERSION``.
# ---------------------------------------------------------------------------

# Standalone idioms whose meaning is lethal/grave harm regardless of object.
_EUPHEMISM_STANDALONE: tuple[str, ...] = (
    "make them disappear",
    "make him disappear",
    "make her disappear",
    "make it look like an accident",
    "wet work",
    "dispose of the body",
    "sleep with the fishes",
    "six feet under",
    "put a hit on",
    "take out a hit",
    "contract killing",
    "snuff out",
    "off the witness",
    "silence the witness",
    "silence the journalist",
    "ethnic cleansing",
    "final solution",
)

# Human objects an attacker pins a euphemism to. Deliberately HIGH-PRECISION:
# caregiving-ambiguous objects ("the kids", "the family") and generic nouns are
# excluded so benign phrasings ("take care of the kids", "take out the trash")
# never match. Only objects that make the verb phrase unambiguously about
# harming a *specific* person are listed.
_EUPHEMISM_OBJECT: tuple[str, ...] = (
    "him",
    "her",
    "them",
    "the witness",
    "the witnesses",
    "the target",
    "the targets",
    "the journalist",
    "the prisoners",
    "the hostages",
    "everyone",
    "these people",
    "those people",
)
_OBJ = "|".join(re.escape(o) for o in _EUPHEMISM_OBJECT)

# Separable particle verbs ("take X out" / "take out X", "put X down"): the
# object may sit before OR after the particle. "deal with" / "take care of" are
# intentionally absent -- they are too ambiguous (benevolent in "take care of the
# patient") to carry harm evidence without a model.
_PARTICLE_VERBS = "put|take|knock|bump|finish"
_PARTICLES = "down|out|off"
# Non-separable prepositional verbs ("get rid of X").
_PREP_VERBS = "get rid of|do away with|dispose of"

_EUPHEMISM_PATTERN = re.compile(
    r"\b(?:(?:" + _PARTICLE_VERBS + r")\s+(?:(?:" + _OBJ + r")\s+(?:" + _PARTICLES + r")"
    r"|(?:" + _PARTICLES + r")\s+(?:" + _OBJ + r"))"
    r"|(?:" + _PREP_VERBS + r")\s+(?:" + _OBJ + r"))\b"
)


def _euphemism_harm_present(text: str) -> bool:
    """True when ``text`` contains a curated lethal/grave-harm euphemism.

    Deterministic, model-free paraphrase detection: it matches the standalone
    idioms above and human-object-pinned (separable or prepositional) verb
    phrases. This is genuine *meaning-level* harm evidence -- the layer the
    char-trigram matcher cannot provide -- tuned for high precision (it must not
    over-block Mercury's own defensive language), and fail-closed (it only ever
    ADDS harm).
    """
    t = text.lower()
    if any(phrase in t for phrase in _EUPHEMISM_STANDALONE):
        return True
    return bool(_EUPHEMISM_PATTERN.search(t))


def _semantic_match_count(words: tuple[str, ...], keywords: list[str], threshold: float) -> int:
    """Count keywords with a strong char-trigram cosine to some word in ``words``.

    NOTE: this is *morphological/lexical* matching, not semantic-by-meaning. It is
    a numpy/stdlib-only stand-in that catches morphological variants and
    near-spellings of the harm/benefit keywords the exact-substring scan misses
    ("injuries" ~ "injury", "manipulative" ~ "manipulate"), with no model
    dependency or non-determinism. It does NOT catch synonyms or euphemism
    ("put him down" for "kill") -- that meaning-level evidence comes from the
    curated :func:`_euphemism_harm_present` lexicon and the optional pluggable
    harm classifier, not from here.
    """
    if not words or not keywords:
        return 0
    # Ignore very short words: 3-letter stopwords ("for", "the") share a
    # trigram prefix with longer keywords ("force") and would false-match. True
    # morphological variants of the (>=4-char) keywords are themselves >=4 chars.
    long_words = [w for w in words if len(w) >= 4]
    if not long_words:
        return 0
    count = 0
    for kw in keywords:
        kv = _char_trigram_vector(kw)
        if any(_cosine(_char_trigram_vector(w), kv) >= threshold for w in long_words):
            count += 1
    return count


# Reversibility lexicons: an irreversible action is far harder to justify than a
# reversible one. Used (with context overrides, fail-closed) by the severity x
# irreversibility damping in the benevolence calculation.
_IRREVERSIBLE_KEYWORDS = (
    "destroy",
    "permanent",
    "death",
    "irreversible",
    "delete",
    "wipe",
    "kill",
    "erase",
    "fatal",
)
_REVERSIBLE_KEYWORDS = (
    "temporary",
    "undo",
    "restore",
    "rollback",
    "reversible",
    "recover",
    "revert",
)

# ---------------------------------------------------------------------------
# σ_Immutable Wave B Vector 2-6 closure: canonical domain sanitiser.
#
# Every public decision boundary (engine.detect_with_fusion[_calibrated],
# CognitiveOrchestrator.analyze, NeuroSymbolicHub.predict, voice.py
# narrative entry points, the federated aggregator) accepts a caller-
# supplied ``domain`` hint that ends up interpolated into the action
# string scored by :class:`BenevolenceScorer`.  An attacker who controls
# that hint could inject harm-keyword substrings (``damage``, ``track``,
# ``expose``, ``destroy``, …) and either (a) trip a false negative on a
# legitimate request or (b) inject positive keywords (``audit``,
# ``protect``) that bias the scorer toward false approval.
#
# ``sanitize_domain`` collapses an arbitrary caller value to a fixed
# whitelist of known-safe canonical labels — the union of
# :class:`~omni_mercury_engine.cognitive.ipb_engine.EnvironmentDomain`
# members and the ``"general"`` fallback sentinel.  Anything else
# (a non-string, an enum, a typo, an injection payload, ``None``) is
# replaced with ``"general"``.  The function is imported by every
# boundary so the whitelist is one source-of-truth, not five copies
# that can drift independently.
#
# The IPB-engine import is deferred (lazy local import inside the
# function) so that ``cognitive.ethical_bounding`` keeps its current
# zero-cost import contract for callers that never touch the
# orchestrator.
# ---------------------------------------------------------------------------

# Cached so the whitelist is built exactly once per process.
_SAFE_DOMAIN_LABELS: frozenset[str] | None = None
_DEFAULT_DOMAIN: str = "general"


def _build_safe_domain_labels() -> frozenset[str]:
    """Return the cached union of ``EnvironmentDomain`` values + ``"general"``."""
    global _SAFE_DOMAIN_LABELS
    if _SAFE_DOMAIN_LABELS is None:
        # Deferred import to keep this module importable without
        # eagerly pulling the IPB engine (which lives in ``cognitive``
        # and would otherwise create an import cycle when the engine
        # imports ``ethical_bounding`` at top level).
        from omni_mercury_engine.cognitive.ipb_engine import EnvironmentDomain

        _SAFE_DOMAIN_LABELS = frozenset(
            {member.value for member in EnvironmentDomain} | {_DEFAULT_DOMAIN}
        )
    return _SAFE_DOMAIN_LABELS


def sanitize_domain(raw_domain: Any) -> str:
    """Collapse an arbitrary caller-supplied ``domain`` to a safe label.

    Args:
        raw_domain: The value the caller passed.  Any type is accepted
            (str, EnvironmentDomain enum, dict, None, …) so the caller
            never has to defensively coerce before passing through.

    Returns:
        A string drawn from the union of ``EnvironmentDomain`` member
        values plus the ``"general"`` fallback sentinel.  Anything
        else — including ``None``, non-string types, unknown labels,
        or strings carrying harm/safety keyword payloads — is replaced
        with ``"general"``.
    """
    # Enum carriers (e.g. ``EnvironmentDomain.CYBER``) expose their
    # canonical label under ``.value``; everything else is forced
    # through ``isinstance(str)`` so non-string types cannot reach
    # the membership test below (an unhashable list/dict would raise
    # ``TypeError`` from ``in``).
    if hasattr(raw_domain, "value"):
        raw_domain = raw_domain.value
    if not isinstance(raw_domain, str):
        return _DEFAULT_DOMAIN
    return raw_domain if raw_domain in _build_safe_domain_labels() else _DEFAULT_DOMAIN


class EthicalConstraintViolationError(RuntimeError):
    """Raised when a hard ethical constraint is violated and execution must halt.

    Unlike the advisory :meth:`BenevolenceScorer.score_action` path (which
    returns ``is_permissible=False`` and leaves enforcement to the caller),
    this exception propagates up the call stack so that impermissible actions
    **cannot** be silently ignored.

    The same class is re-exported from :mod:`omni_mercury_engine.ethical` as
    ``EthicalViolation`` — that is the canonical name to use at new decision
    boundaries.  ``EthicalConstraintViolationError`` is retained so existing
    catch sites and tests keep working.

    Attributes:
        action: The action that triggered the violation.
        score: The computed benevolence score.
        threshold: The minimum required benevolence score.
        check: Identifier for which gate raised — ``"benevolence"`` for the
            scorer-based check, ``"sigma_immutable"`` for GOSNN's neural
            ethical gate, ``"gosnn_unavailable"`` when the gate could not be
            evaluated and the boundary failed closed.
        details: Optional structured context (e.g., domain, requesting
            component, raw scalar vector summary) for downstream auditing.
        analysis_time_ms: Optional wall-clock time for the analysis run that
            triggered the violation, captured by the orchestrator before the
            exception was raised.  ``None`` when the raiser did not measure it.
    """

    def __init__(
        self,
        action: str,
        score: float,
        threshold: float,
        analysis_time_ms: float | None = None,
        *,
        check: str = "benevolence",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the instance."""
        self.action = action
        self.score = score
        self.threshold = threshold
        self.check = check
        self.details = details or {}
        self.analysis_time_ms = analysis_time_ms
        super().__init__(
            f"Ethical constraint '{check}' violated for action '{action}': "
            f"score={score:.4f} < threshold={threshold:.4f}. "
            "Execution blocked at the decision boundary."
        )


class EthicalPrinciple(Enum):
    """Core ethical principles."""

    COMPASSION = "compassion"
    EVIDENCE = "evidence"
    JUSTICE = "justice"
    ALTRUISM = "altruism"
    CONTROL = "control"
    CHARACTER = "character"
    COMPETENCE = "competence"
    COMMITMENT = "commitment"


class HarmCategory(Enum):
    """Categories of potential harm."""

    PHYSICAL = "physical"
    PSYCHOLOGICAL = "psychological"
    FINANCIAL = "financial"
    PRIVACY = "privacy"
    AUTONOMY = "autonomy"
    DIGNITY = "dignity"
    ENVIRONMENTAL = "environmental"
    SOCIETAL = "societal"


class BenefitCategory(Enum):
    """Categories of potential benefit."""

    SAFETY = "safety"
    WELLBEING = "wellbeing"
    KNOWLEDGE = "knowledge"
    EFFICIENCY = "efficiency"
    EQUITY = "equity"
    SUSTAINABILITY = "sustainability"
    EMPOWERMENT = "empowerment"
    HUMANITARIAN = "humanitarian"


@dataclass
class EthicalScore:
    """Comprehensive ethical evaluation score."""

    score_id: str
    action: str
    benevolence_score: float
    harm_score: float
    benefit_score: float
    equity_score: float
    long_term_score: float
    is_permissible: bool
    principle_scores: dict[str, float]
    harm_breakdown: dict[str, float]
    benefit_breakdown: dict[str, float]
    explanation: str
    recommendations: list[str]
    timestamp: float = field(default_factory=time.time)
    # Severity (max weighted harm) and reversibility in [0, 1]; together they
    # form the multiplicative damping in the benevolence calculation. Defaults
    # (0.0 / 1.0) make a high-severity-irreversible action the only thing the
    # damping bites, and keep older positional constructors working.
    severity_score: float = 0.0
    reversibility_score: float = 1.0


@dataclass
class EmpathyAssessment:
    """Assessment of human-centric impact."""

    assessment_id: str
    affected_parties: list[str]
    impact_scores: dict[str, float]
    vulnerability_factors: list[str]
    mitigation_suggestions: list[str]
    overall_empathy_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValuePreservation:
    """Value preservation analysis."""

    preservation_id: str
    values_at_risk: list[str]
    preservation_score: float
    default_to_positive: bool
    safeguards_needed: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlignmentAudit:
    """Audit record for alignment verification."""

    audit_id: str
    action: str
    ethical_score: EthicalScore
    empathy_assessment: EmpathyAssessment | None
    value_preservation: ValuePreservation | None
    passed: bool
    failure_reasons: list[str]
    timestamp: float = field(default_factory=time.time)


class HarmReducer:
    """Evaluates and minimizes potential harm from actions.

    Uses weighted scoring across harm categories to ensure actions minimize negative impacts.
    """

    HARM_WEIGHTS = {
        HarmCategory.PHYSICAL: 1.0,
        HarmCategory.PSYCHOLOGICAL: 0.9,
        HarmCategory.FINANCIAL: 0.7,
        HarmCategory.PRIVACY: 0.8,
        HarmCategory.AUTONOMY: 0.85,
        HarmCategory.DIGNITY: 0.9,
        HarmCategory.ENVIRONMENTAL: 0.75,
        HarmCategory.SOCIETAL: 0.8,
    }

    def __init__(self, harm_classifier: Any | None = None) -> None:
        """Initialize harm reducer.

        Args:
            harm_classifier: Optional ``Callable[[str], float]`` returning a harm
                probability in ``[0, 1]`` for a piece of text. This is the
                meaning-level extension point -- a deployment can plug in a real
                *semantic* classifier (e.g. one backed by Mercury's own local
                Ollama reasoning backend, see
                :func:`omni_mercury_engine.reasoning.backends.reasoning_harm_classifier`)
                without this module taking a model dependency. It is fail-safe and
                can only RAISE harm (combined by ``max``); an exception or a
                lower score never lowers the deterministic lexical harm. Default
                ``None`` keeps the scorer fully deterministic and model-free.
        """
        self._evaluation_counter = 0
        self._harm_classifier = harm_classifier

    def evaluate_harm(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Evaluate potential harm from an action.

        Layers of evidence, fail-closed (each can only RAISE harm):
        1. exact-substring keyword scan, per harm category;
        2. char-trigram morphological match (catches spelling/inflection);
        3. curated euphemism/paraphrase lexicon (meaning-level: "put him down");
        4. an optional pluggable harm classifier (e.g. a local-model semantic
           classifier), combined by ``max``.

        Args:
            action: Action to evaluate
            context: Context for the action

        Returns:
            Tuple of (overall_harm_score, category_breakdown)
        """
        breakdown = {}

        for category in HarmCategory:
            harm_level = self._assess_category_harm(action, context, category)
            breakdown[category.value] = harm_level

        combined = action.lower() + " " + str(context).lower()

        # Euphemism/paraphrase evidence is unambiguous harm-to-person intent ->
        # raise the gravest (PHYSICAL) category to near-max. Meaning-level, not
        # spelling-level; fail-closed (max, never lowers).
        if _euphemism_harm_present(combined):
            breakdown[HarmCategory.PHYSICAL.value] = max(
                breakdown[HarmCategory.PHYSICAL.value], 0.9
            )

        weighted_sum = sum(breakdown[cat.value] * self.HARM_WEIGHTS[cat] for cat in HarmCategory)
        max_weighted = sum(self.HARM_WEIGHTS.values())
        overall_harm = weighted_sum / max_weighted

        # Optional semantic classifier: can only RAISE the overall harm. If it
        # errors we keep the deterministic lexical harm (never lower it for a
        # classifier failure), so an unavailable model is not a safety regression.
        if self._harm_classifier is not None:
            try:
                score = float(self._harm_classifier(combined))
                overall_harm = max(overall_harm, min(max(score, 0.0), 1.0))
            except Exception as exc:
                logger.warning(
                    "harm_classifier failed (%s); keeping deterministic lexical harm", exc
                )

        return overall_harm, breakdown

    def _assess_category_harm(
        self,
        action: str,
        context: dict[str, Any],
        category: HarmCategory,
    ) -> float:
        """Assess harm level for a specific category."""
        harm_keywords = {
            HarmCategory.PHYSICAL: ["injury", "damage", "hurt", "harm", "violence"],
            HarmCategory.PSYCHOLOGICAL: ["stress", "anxiety", "fear", "trauma", "distress"],
            HarmCategory.FINANCIAL: ["loss", "cost", "expense", "debt", "bankruptcy"],
            HarmCategory.PRIVACY: ["expose", "leak", "reveal", "track", "surveil"],
            HarmCategory.AUTONOMY: ["force", "coerce", "manipulate", "control", "restrict"],
            HarmCategory.DIGNITY: ["humiliate", "degrade", "demean", "disrespect"],
            HarmCategory.ENVIRONMENTAL: ["pollute", "destroy", "deplete", "waste"],
            HarmCategory.SOCIETAL: ["divide", "discriminate", "exclude", "marginalize"],
        }

        action_lower = action.lower()
        context_str = str(context).lower()
        combined = action_lower + " " + context_str

        keywords = harm_keywords.get(category, [])
        substring_matches = sum(1 for kw in keywords if kw in combined)

        # Semantic term: catch morphological variants ("injuries", "harmful",
        # "manipulative") that the exact-substring scan misses. Combined with
        # max() so the semantic path can only ADD evidence of harm, never lower
        # it -- fail-closed even if the similarity is noisy.
        words = tuple(re.findall(r"[a-z]+", combined))
        semantic_matches = _semantic_match_count(
            words, keywords, BENEVOLENCE_CALIBRATION.semantic_match_threshold
        )
        matches = max(substring_matches, semantic_matches)

        harm_level = min(1.0, matches * 0.25)

        if context.get("potential_harm"):
            harm_level = min(1.0, harm_level + 0.3)

        return harm_level


class BenefitMaximizer:
    """Evaluates and maximizes potential benefits from actions.

    Uses weighted scoring across benefit categories to ensure actions maximize positive impacts.
    """

    BENEFIT_WEIGHTS = {
        BenefitCategory.SAFETY: 1.0,
        BenefitCategory.WELLBEING: 0.95,
        BenefitCategory.KNOWLEDGE: 0.7,
        BenefitCategory.EFFICIENCY: 0.6,
        BenefitCategory.EQUITY: 0.85,
        BenefitCategory.SUSTAINABILITY: 0.8,
        BenefitCategory.EMPOWERMENT: 0.75,
        BenefitCategory.HUMANITARIAN: 1.0,
    }

    def __init__(self) -> None:
        """Initialize benefit maximizer."""
        self._evaluation_counter = 0

    def evaluate_benefit(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Evaluate potential benefit from an action.

        Args:
            action: Action to evaluate
            context: Context for the action

        Returns:
            Tuple of (overall_benefit_score, category_breakdown)
        """
        breakdown = {}

        for category in BenefitCategory:
            benefit_level = self._assess_category_benefit(action, context, category)
            breakdown[category.value] = benefit_level

        weighted_sum = sum(
            breakdown[cat.value] * self.BENEFIT_WEIGHTS[cat] for cat in BenefitCategory
        )
        max_weighted = sum(self.BENEFIT_WEIGHTS.values())
        overall_benefit = weighted_sum / max_weighted

        return overall_benefit, breakdown

    def _assess_category_benefit(
        self,
        action: str,
        context: dict[str, Any],
        category: BenefitCategory,
    ) -> float:
        """Assess benefit level for a specific category."""
        benefit_keywords = {
            BenefitCategory.SAFETY: ["protect", "secure", "safe", "prevent", "guard"],
            BenefitCategory.WELLBEING: ["health", "wellness", "care", "support", "help"],
            BenefitCategory.KNOWLEDGE: ["learn", "discover", "research", "educate", "inform"],
            BenefitCategory.EFFICIENCY: ["optimize", "improve", "streamline", "automate"],
            BenefitCategory.EQUITY: ["fair", "equal", "inclusive", "accessible", "just"],
            BenefitCategory.SUSTAINABILITY: ["sustain", "renew", "conserve", "preserve"],
            BenefitCategory.EMPOWERMENT: ["enable", "empower", "assist", "facilitate"],
            BenefitCategory.HUMANITARIAN: ["humanitarian", "rescue", "aid", "relief", "crisis"],
        }

        action_lower = action.lower()
        context_str = str(context).lower()
        combined = action_lower + " " + context_str

        keywords = benefit_keywords.get(category, [])
        matches = sum(1 for kw in keywords if kw in combined)

        benefit_level = min(1.0, matches * 0.25)

        if context.get("humanitarian"):
            benefit_level = min(1.0, benefit_level + 0.4)

        return benefit_level


class EquityCalculator:
    """Calculates equity metrics using Gini-like coefficients.

    Ensures actions promote fairness and reduce inequality.
    """

    def __init__(self) -> None:
        """Initialize equity calculator."""
        pass

    def calculate_gini(self, values: list[float]) -> float:
        """Calculate Gini coefficient for a distribution.

        Args:
            values: List of values representing distribution

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if not values or len(values) < 2:
            return 0.0

        values = sorted(values)
        n = len(values)
        total = sum(values)

        if total == 0:
            return 0.0

        cumulative: float = 0
        gini_sum: float = 0
        for i, v in enumerate(values):
            cumulative += v
            gini_sum += (2 * (i + 1) - n - 1) * v

        gini = gini_sum / (n * total)
        return max(0.0, min(1.0, gini))

    def evaluate_equity(
        self,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """Evaluate equity impact of an action.

        Args:
            action: Action to evaluate
            context: Context for the action

        Returns:
            Equity score (higher = more equitable)
        """
        base_equity = 0.7

        equity_positive = ["fair", "equal", "inclusive", "accessible", "diverse"]
        equity_negative = ["discriminate", "exclude", "bias", "unfair", "privilege"]

        combined = (action + " " + str(context)).lower()

        for word in equity_positive:
            if word in combined:
                base_equity += 0.1

        for word in equity_negative:
            if word in combined:
                base_equity -= 0.15

        if "distribution" in context:
            dist = context["distribution"]
            if isinstance(dist, list) and len(dist) > 1:
                gini = self.calculate_gini(dist)
                base_equity -= gini * 0.3

        return max(0.0, min(1.0, base_equity))


class EmpathyModule:
    """Empathy module for human-centric decision making.

    Considers impact on affected parties and vulnerable populations.
    """

    def __init__(self) -> None:
        """Initialize empathy module."""
        self._assessment_counter = 0

    def assess_empathy(
        self,
        action: str,
        context: dict[str, Any],
    ) -> EmpathyAssessment:
        """Assess human-centric impact of an action.

        Args:
            action: Action to assess
            context: Context for the action

        Returns:
            EmpathyAssessment with detailed analysis
        """
        self._assessment_counter += 1
        assessment_id = f"empathy_{self._assessment_counter:06d}"

        affected_parties = self._identify_affected_parties(context)
        impact_scores = self._calculate_impact_scores(action, context, affected_parties)
        vulnerability_factors = self._identify_vulnerabilities(context)
        mitigation_suggestions = self._generate_mitigations(vulnerability_factors)

        overall_score = self._calculate_overall_empathy(impact_scores, vulnerability_factors)

        return EmpathyAssessment(
            assessment_id=assessment_id,
            affected_parties=affected_parties,
            impact_scores=impact_scores,
            vulnerability_factors=vulnerability_factors,
            mitigation_suggestions=mitigation_suggestions,
            overall_empathy_score=overall_score,
        )

    def _identify_affected_parties(self, context: dict[str, Any]) -> list[str]:
        """Identify parties affected by the action."""
        parties = ["general_public"]

        if context.get("users"):
            parties.append("direct_users")
        if context.get("stakeholders"):
            parties.extend(context["stakeholders"])
        if context.get("vulnerable_groups"):
            parties.extend(context["vulnerable_groups"])

        return list(set(parties))

    def _calculate_impact_scores(
        self,
        action: str,
        context: dict[str, Any],
        parties: list[str],
    ) -> dict[str, float]:
        """Calculate impact scores for each affected party."""
        scores = {}

        for party in parties:
            base_score = 0.7

            if "vulnerable" in party.lower():
                base_score -= 0.1
            if "humanitarian" in action.lower():
                base_score += 0.2

            scores[party] = max(0.0, min(1.0, base_score))

        return scores

    def _identify_vulnerabilities(self, context: dict[str, Any]) -> list[str]:
        """Identify vulnerability factors."""
        vulnerabilities = []

        if context.get("children_involved"):
            vulnerabilities.append("children_at_risk")
        if context.get("elderly_involved"):
            vulnerabilities.append("elderly_at_risk")
        if context.get("medical_context"):
            vulnerabilities.append("health_sensitive")
        if context.get("financial_hardship"):
            vulnerabilities.append("economic_vulnerability")

        return vulnerabilities

    def _generate_mitigations(self, vulnerabilities: list[str]) -> list[str]:
        """Generate mitigation suggestions for vulnerabilities."""
        mitigations = []

        mitigation_map = {
            "children_at_risk": "Implement additional safeguards for minors",
            "elderly_at_risk": "Ensure accessibility and clear communication",
            "health_sensitive": "Consult medical ethics guidelines",
            "economic_vulnerability": "Consider financial impact mitigation",
        }

        for vuln in vulnerabilities:
            if vuln in mitigation_map:
                mitigations.append(mitigation_map[vuln])

        return mitigations

    def _calculate_overall_empathy(
        self,
        impact_scores: dict[str, float],
        vulnerabilities: list[str],
    ) -> float:
        """Calculate overall empathy score."""
        if not impact_scores:
            return 0.7

        avg_impact = sum(impact_scores.values()) / len(impact_scores)

        vulnerability_penalty = len(vulnerabilities) * 0.05

        return max(0.0, min(1.0, avg_impact - vulnerability_penalty))


class ValuePreserver:
    """Value preservation module for maintaining positive outcomes.

    Ensures actions default to positive outcomes and preserve important values.
    """

    CORE_VALUES = [
        "human_dignity",
        "autonomy",
        "privacy",
        "safety",
        "fairness",
        "transparency",
        "accountability",
        "beneficence",
    ]

    def __init__(self) -> None:
        """Initialize value preserver."""
        self._preservation_counter = 0

    def analyze_preservation(
        self,
        action: str,
        context: dict[str, Any],
    ) -> ValuePreservation:
        """Analyze value preservation for an action.

        Args:
            action: Action to analyze
            context: Context for the action

        Returns:
            ValuePreservation analysis
        """
        self._preservation_counter += 1
        preservation_id = f"preserve_{self._preservation_counter:06d}"

        values_at_risk = self._identify_values_at_risk(action, context)
        preservation_score = self._calculate_preservation_score(values_at_risk)
        default_to_positive = preservation_score >= 0.7
        safeguards = self._recommend_safeguards(values_at_risk)

        return ValuePreservation(
            preservation_id=preservation_id,
            values_at_risk=values_at_risk,
            preservation_score=preservation_score,
            default_to_positive=default_to_positive,
            safeguards_needed=safeguards,
        )

    def _identify_values_at_risk(
        self,
        action: str,
        context: dict[str, Any],
    ) -> list[str]:
        """Identify values potentially at risk."""
        at_risk = []

        risk_indicators = {
            "human_dignity": ["degrade", "humiliate", "demean"],
            "autonomy": ["force", "coerce", "manipulate"],
            "privacy": ["expose", "track", "surveil", "collect"],
            "safety": ["danger", "risk", "harm", "threat"],
            "fairness": ["bias", "discriminate", "unfair"],
            "transparency": ["hide", "obscure", "deceive"],
            "accountability": ["anonymous", "untraceable"],
            "beneficence": ["harm", "damage", "hurt"],
        }

        combined = (action + " " + str(context)).lower()

        for value, indicators in risk_indicators.items():
            for indicator in indicators:
                if indicator in combined:
                    at_risk.append(value)
                    break

        return at_risk

    def _calculate_preservation_score(self, values_at_risk: list[str]) -> float:
        """Calculate preservation score based on values at risk."""
        if not values_at_risk:
            return 1.0

        risk_ratio = len(values_at_risk) / len(self.CORE_VALUES)
        return max(0.0, 1.0 - risk_ratio)

    def _recommend_safeguards(self, values_at_risk: list[str]) -> list[str]:
        """Recommend safeguards for at-risk values."""
        safeguards = []

        safeguard_map = {
            "human_dignity": "Ensure respectful treatment in all interactions",
            "autonomy": "Provide opt-out mechanisms and informed consent",
            "privacy": "Implement data minimization and encryption",
            "safety": "Add safety checks and human oversight",
            "fairness": "Conduct bias audits and fairness testing",
            "transparency": "Document decision-making processes",
            "accountability": "Maintain audit trails and responsibility chains",
            "beneficence": "Verify positive outcome likelihood",
        }

        for value in values_at_risk:
            if value in safeguard_map:
                safeguards.append(safeguard_map[value])

        return safeguards


class BenevolenceScorer:
    """Main benevolence scoring engine.

    Combines harm reduction, benefit maximization, equity, empathy, and value preservation into a
    unified score.
    """

    BENEVOLENCE_THRESHOLD = 0.99

    def __init__(
        self, benevolence_threshold: float = 0.99, *, harm_classifier: Any | None = None
    ) -> None:
        """Initialize benevolence scorer.

        Args:
            benevolence_threshold: Minimum score for action approval.  Must be
                at or above ``MINIMUM_BENEVOLENCE_FLOOR`` (0.70).  Values below
                this absolute floor are clamped with a warning, and any later
                assignment to :attr:`benevolence_threshold` is also clamped
                via the property setter — the floor cannot be lowered after
                construction.
            harm_classifier: Optional ``Callable[[str], float]`` forwarded to the
                :class:`HarmReducer` -- a meaning-level harm classifier (e.g. one
                backed by Mercury's local Ollama reasoning backend) that can only
                RAISE harm. Default ``None`` keeps scoring deterministic and
                model-free.
        """
        # Use the property setter so the floor is enforced consistently
        # whether the value is set in __init__ or reassigned later.
        self.benevolence_threshold = benevolence_threshold

        self.harm_reducer = HarmReducer(harm_classifier=harm_classifier)
        self.benefit_maximizer = BenefitMaximizer()
        self.equity_calculator = EquityCalculator()
        self.empathy_module = EmpathyModule()
        self.value_preserver = ValuePreserver()

        self._score_counter = 0
        self._audit_counter = 0

        self.audit_history: list[AlignmentAudit] = []

        # Log the clamped value (self.benevolence_threshold via the property
        # getter), not the raw constructor argument — otherwise an operator
        # debugging a below-floor request would see the value they tried to
        # set instead of the value the gate is actually using.
        logger.info("BenevolenceScorer initialized with threshold %s", self.benevolence_threshold)

    @property
    def benevolence_threshold(self) -> float:
        """Approval threshold, always at or above ``MINIMUM_BENEVOLENCE_FLOOR``."""
        return self._benevolence_threshold

    @benevolence_threshold.setter
    def benevolence_threshold(self, value: float) -> None:
        """Clamp every assignment to the absolute floor.

        Storing the threshold as a property instead of a plain attribute
        ensures the ``MINIMUM_BENEVOLENCE_FLOOR`` guarantee survives later
        mutation (``scorer.benevolence_threshold = 0.0`` no longer bypasses
        the gate — the assignment is silently raised to the floor with a
        warning).
        """
        if value < MINIMUM_BENEVOLENCE_FLOOR:
            logger.warning(
                "benevolence_threshold=%.4f is below the absolute minimum "
                "floor of %.4f — clamping to floor.",
                value,
                MINIMUM_BENEVOLENCE_FLOOR,
            )
            value = MINIMUM_BENEVOLENCE_FLOOR
        self._benevolence_threshold = value

    def score_action(
        self,
        action: str,
        context: dict[str, Any],
    ) -> EthicalScore:
        """Score an action for benevolence.

        Args:
            action: Action to score
            context: Context for the action

        Returns:
            EthicalScore with comprehensive evaluation
        """
        self._score_counter += 1
        score_id = f"ethical_{self._score_counter:06d}"

        harm_score, harm_breakdown = self.harm_reducer.evaluate_harm(action, context)
        benefit_score, benefit_breakdown = self.benefit_maximizer.evaluate_benefit(action, context)
        equity_score = self.equity_calculator.evaluate_equity(action, context)

        principle_scores = self._evaluate_principles(action, context)

        long_term_score = self._evaluate_long_term(action, context, benefit_score, harm_score)

        severity_score = self._assess_severity(harm_breakdown, context)
        reversibility_score = self._assess_reversibility(action, context)

        benevolence_score = self._calculate_benevolence(
            harm_score=harm_score,
            benefit_score=benefit_score,
            equity_score=equity_score,
            principle_scores=principle_scores,
            long_term_score=long_term_score,
            severity=severity_score,
            reversibility=reversibility_score,
        )

        is_permissible = benevolence_score >= self.benevolence_threshold

        explanation = self._generate_explanation(
            action, benevolence_score, harm_score, benefit_score, is_permissible
        )
        recommendations = self._generate_recommendations(
            harm_breakdown, benefit_breakdown, is_permissible
        )

        return EthicalScore(
            score_id=score_id,
            action=action,
            benevolence_score=benevolence_score,
            harm_score=harm_score,
            benefit_score=benefit_score,
            equity_score=equity_score,
            long_term_score=long_term_score,
            is_permissible=is_permissible,
            principle_scores=principle_scores,
            harm_breakdown=harm_breakdown,
            benefit_breakdown=benefit_breakdown,
            explanation=explanation,
            recommendations=recommendations,
            severity_score=severity_score,
            reversibility_score=reversibility_score,
        )

    def enforce(
        self,
        action: str,
        context: dict[str, Any],
    ) -> EthicalScore:
        """Score an action and raise on violation — the *mandatory* gate.

        Unlike :meth:`score_action`, which returns the result regardless of
        permissibility, ``enforce`` raises
        :class:`EthicalConstraintViolationError` when the benevolence score
        falls below the configured threshold so that impermissible actions
        **cannot** be silently ignored upstream.

        Args:
            action: Action to evaluate.
            context: Contextual metadata for the action.

        Returns:
            EthicalScore when the action is permissible.

        Raises:
            EthicalConstraintViolationError: When ``benevolence_score < threshold``.
        """
        result = self.score_action(action, context)
        if not result.is_permissible:
            raise EthicalConstraintViolationError(
                action=action,
                score=result.benevolence_score,
                threshold=self.benevolence_threshold,
            )
        return result

    def _evaluate_principles(
        self,
        action: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate action against ethical principles."""
        scores = {}

        for principle in EthicalPrinciple:
            scores[principle.value] = self._score_principle(action, context, principle)

        return scores

    def _score_principle(
        self,
        action: str,
        context: dict[str, Any],
        principle: EthicalPrinciple,
    ) -> float:
        """Score action against a specific principle."""
        base_score = 0.8

        principle_keywords = {
            EthicalPrinciple.COMPASSION: ["care", "help", "support", "empathy"],
            EthicalPrinciple.EVIDENCE: ["data", "research", "verify", "prove"],
            EthicalPrinciple.JUSTICE: ["fair", "just", "equal", "rights"],
            EthicalPrinciple.ALTRUISM: ["selfless", "benefit", "humanitarian", "aid"],
            EthicalPrinciple.CONTROL: ["oversight", "review", "approve", "monitor"],
            EthicalPrinciple.CHARACTER: ["integrity", "honest", "ethical", "moral"],
            EthicalPrinciple.COMPETENCE: ["capable", "skilled", "qualified", "expert"],
            EthicalPrinciple.COMMITMENT: ["dedicated", "persistent", "reliable", "consistent"],
        }

        combined = (action + " " + str(context)).lower()
        keywords = principle_keywords.get(principle, [])

        for kw in keywords:
            if kw in combined:
                base_score += 0.05

        return min(1.0, base_score)

    def _evaluate_long_term(
        self,
        action: str,
        context: dict[str, Any],
        benefit_score: float,
        harm_score: float,
    ) -> float:
        """Evaluate long-term societal impact."""
        base_score = 0.7

        base_score += benefit_score * 0.2
        base_score -= harm_score * 0.3

        if context.get("sustainable"):
            base_score += 0.1
        if context.get("short_term_only"):
            base_score -= 0.1

        return max(0.0, min(1.0, base_score))

    @staticmethod
    def _assess_severity(harm_breakdown: dict[str, float], context: dict[str, Any]) -> float:
        """Severity in [0, 1]: worst per-category harm, raised by context.

        A caller-supplied ``context['severity']`` can only INCREASE severity
        (MAX), never decrease it -- fail-closed.
        """
        sev = max(harm_breakdown.values()) if harm_breakdown else 0.0
        ctx = context.get("severity")
        if isinstance(ctx, (int, float)) and not isinstance(ctx, bool):
            sev = max(sev, float(np.clip(ctx, 0.0, 1.0)))
        return float(np.clip(sev, 0.0, 1.0))

    @staticmethod
    def _assess_reversibility(action: str, context: dict[str, Any]) -> float:
        """Reversibility in [0, 1] (1 = fully reversible).

        Derived from reversible/irreversible lexicons over the action+context
        text; a caller-supplied ``context['reversibility']`` can only DECREASE
        reversibility (MIN), never increase it -- fail-closed (a caller may
        assert an action is *less* reversible, never more).
        """
        text = (action + " " + str(context)).lower()
        irreversible = any(kw in text for kw in _IRREVERSIBLE_KEYWORDS)
        reversible = any(kw in text for kw in _REVERSIBLE_KEYWORDS)
        # Assume reversible (1.0) unless there is positive EVIDENCE of
        # irreversibility. Damping benevolence merely because reversibility is
        # *unknown* would false-reject legitimate actions (the failure mode the
        # ethics-gate hardening is meant to avoid); the damping bites only when
        # an irreversibility signal is actually present.
        if irreversible and not reversible:
            base = 0.1
        else:
            base = 1.0
        ctx = context.get("reversibility")
        if isinstance(ctx, (int, float)) and not isinstance(ctx, bool):
            base = min(base, float(np.clip(ctx, 0.0, 1.0)))
        return float(np.clip(base, 0.0, 1.0))

    def _calculate_benevolence(
        self,
        harm_score: float,
        benefit_score: float,
        equity_score: float,
        principle_scores: dict[str, float],
        long_term_score: float,
        severity: float = 0.0,
        reversibility: float = 1.0,
    ) -> float:
        """Calculate overall benevolence score.

        Weighted sum of (1-harm), benefit, equity, principles, long-term using
        the calibratable :data:`BENEVOLENCE_CALIBRATION` weights, then a
        multiplicative severity x irreversibility damping so a high-severity,
        irreversible action cannot be "averaged away" by positive keywords. The
        damping multiplier is ``1 - severity*(1-reversibility)*gamma`` (always
        <= 1, so it only ever LOWERS the score -- fail-closed). Defaults
        (severity=0, reversibility=1) leave the score unchanged for legacy
        callers.
        """
        cal = BENEVOLENCE_CALIBRATION
        harm_component = (1 - harm_score) * cal.w_harm
        benefit_component = benefit_score * cal.w_benefit
        equity_component = equity_score * cal.w_equity

        principles_avg = sum(principle_scores.values()) / len(principle_scores)
        principles_component = principles_avg * cal.w_principles

        long_term_component = long_term_score * cal.w_long_term

        weighted_sum = (
            harm_component
            + benefit_component
            + equity_component
            + principles_component
            + long_term_component
        )

        damping = 1.0 - severity * (1.0 - reversibility) * cal.severity_gamma
        benevolence = weighted_sum * max(0.0, damping)

        return max(0.0, min(1.0, benevolence))

    def _generate_explanation(
        self,
        action: str,
        benevolence_score: float,
        harm_score: float,
        benefit_score: float,
        is_permissible: bool,
    ) -> str:
        """Generate explanation for the ethical score."""
        status = "APPROVED" if is_permissible else "BLOCKED"

        return (
            f"Action '{action}' scored {benevolence_score:.2%} benevolence ({status}). "
            f"Harm potential: {harm_score:.0%}, Benefit potential: {benefit_score:.0%}. "
            f"Threshold: {self.benevolence_threshold:.0%}."
        )

    def _generate_recommendations(
        self,
        harm_breakdown: dict[str, float],
        benefit_breakdown: dict[str, float],
        is_permissible: bool,
    ) -> list[str]:
        """Generate recommendations for improving ethical score."""
        recommendations = []

        if not is_permissible:
            recommendations.append("Action does not meet benevolence threshold")

        high_harm_categories = [cat for cat, score in harm_breakdown.items() if score > 0.3]
        for cat in high_harm_categories:
            recommendations.append(f"Reduce {cat} harm potential")

        low_benefit_categories = [cat for cat, score in benefit_breakdown.items() if score < 0.3]
        if low_benefit_categories:
            recommendations.append("Consider ways to increase positive impact")

        return recommendations

    def full_audit(
        self,
        action: str,
        context: dict[str, Any],
    ) -> AlignmentAudit:
        """Perform full alignment audit on an action.

        Args:
            action: Action to audit
            context: Context for the action

        Returns:
            AlignmentAudit with comprehensive analysis
        """
        self._audit_counter += 1
        audit_id = f"audit_{self._audit_counter:06d}"

        ethical_score = self.score_action(action, context)
        empathy_assessment = self.empathy_module.assess_empathy(action, context)
        value_preservation = self.value_preserver.analyze_preservation(action, context)

        failure_reasons = []

        if not ethical_score.is_permissible:
            failure_reasons.append(
                f"Benevolence score {ethical_score.benevolence_score:.2%} below threshold"
            )

        if empathy_assessment.overall_empathy_score < 0.7:
            failure_reasons.append(
                f"Empathy score {empathy_assessment.overall_empathy_score:.2%} too low"
            )

        if not value_preservation.default_to_positive:
            failure_reasons.append("Action does not default to positive outcomes")

        passed = len(failure_reasons) == 0

        audit = AlignmentAudit(
            audit_id=audit_id,
            action=action,
            ethical_score=ethical_score,
            empathy_assessment=empathy_assessment,
            value_preservation=value_preservation,
            passed=passed,
            failure_reasons=failure_reasons,
        )

        self.audit_history.append(audit)

        return audit

    def is_action_permissible(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[bool, float, str]:
        """Quick check if action is permissible.

        Args:
            action: Action to check
            context: Context for the action

        Returns:
            Tuple of (is_permissible, benevolence_score, explanation)
        """
        score = self.score_action(action, context)
        return score.is_permissible, score.benevolence_score, score.explanation

    def get_statistics(self) -> dict[str, Any]:
        """Get scorer statistics."""
        passed_audits = sum(1 for a in self.audit_history if a.passed)

        return {
            "scores_generated": self._score_counter,
            "audits_performed": self._audit_counter,
            "audits_passed": passed_audits,
            "pass_rate": passed_audits / self._audit_counter if self._audit_counter > 0 else 0,
            "benevolence_threshold": self.benevolence_threshold,
        }

    def get_audit_history(self, limit: int = 100) -> list[AlignmentAudit]:
        """Get recent audit history."""
        return self.audit_history[-limit:]
