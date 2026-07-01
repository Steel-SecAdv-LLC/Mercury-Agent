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
from dataclasses import (
    dataclass,
    field,
    fields as dataclass_fields,
)
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.harm_normalization import (
    MULTILINGUAL_HAZARD_TERMS,
    MULTILINGUAL_OFFENSIVE_CUES,
    normalized_haystack,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard ethical floor — callers cannot configure the benevolence threshold
# below this value, regardless of domain or operational mode.
# ---------------------------------------------------------------------------
MINIMUM_BENEVOLENCE_FLOOR: float = 0.70


@dataclass(frozen=True)
class BenevolenceCalibration:
    """Calibration knobs for the benevolence scorer.

    These parameters are meant to be *fit on labeled decisions* rather than
    hand-set. The defaults below are honest fallbacks; a fitted set is produced
    by ``scripts/fit_weapons_gate_calibration.py`` from the labeled corpus in
    ``benchmarks/weapons_gate_corpus.jsonl`` and written to
    ``configs/weapons_gate_calibration.json``, which :meth:`load_default` reads at
    import time. When that file is absent the defaults apply and
    :attr:`is_fitted` is ``False`` -- so the code never *claims* a measurement it
    does not have. The parameters are gathered here, version-pinned, and frozen so
    a change is explicit and invalidates the benevolence cache (bump
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
    # -- Weapons/mass-casualty uplift confidence model (see assess_weapons_uplift)
    # The Axis-B offensive confidence is a logistic over evidence:
    #   z = conf_bias + conf_w_offensive*n_off - conf_w_allow*n_allow
    #       + conf_w_weight*hazard_weight + conf_w_classifier*classifier_boost
    #   confidence = sigmoid(z)
    # fit by scripts/fit_weapons_gate_calibration.py (maximum-likelihood logistic
    # on the labeled corpus). Monotone in offensive evidence, the hazard weight,
    # and the classifier signal; decreasing in allow-signal -- so the gray-zone
    # ESCALATE/REFUSE_REDACT split tracks measured evidence, not a raw ratio.
    conf_bias: float = -1.0
    conf_w_offensive: float = 1.1
    conf_w_allow: float = 1.3
    conf_w_weight: float = 0.6
    conf_w_classifier: float = 3.0
    # Below this Axis-B confidence, a B6 (production/synthesis) verdict is
    # treated as the genuine gray zone and ESCALATEd (slowed + audited) rather
    # than REFUSE_REDACTed outright; an asserted licensed-professional context
    # has the same effect. B7-B10 ignore this entirely (always HARD_REFUSE).
    weapons_b6_escalate_confidence: float = 0.6
    # Harm-score floors a weapons-uplift disposition raises the deterministic
    # lexical harm to (max-only, fail-closed -- never lowers it). HARD_REFUSE
    # (B7-B10) floors highest; REFUSE_REDACT (non-gray B6) next; ESCALATE
    # (gray-zone B6) just above the general-capability HARM_REFUSAL_THRESHOLD
    # so the harm-score gate and the disposition gate agree.
    weapons_hard_refuse_harm_floor: float = 0.95
    weapons_refuse_redact_harm_floor: float = 0.75
    weapons_escalate_harm_floor: float = 0.55
    # Provenance of the active parameters: "default" (hand-set fallbacks) or the
    # calibration-file path the fitted values were loaded from. Never fabricated.
    source: str = "default"

    @property
    def is_fitted(self) -> bool:
        """True only when the active parameters were loaded from a fitted file."""
        return self.source != "default"

    @classmethod
    def load_default(cls) -> BenevolenceCalibration:
        """Load fitted parameters from ``configs/weapons_gate_calibration.json``.

        Falls back to the hand-set defaults (``source="default"``) when the file
        is absent or unreadable -- fail-safe and honest: an unfit deployment is
        labelled ``default``, never silently presented as calibrated. Only known
        float fields are accepted from the file; unknown keys are ignored.
        """
        import json
        from pathlib import Path

        # Repo layout: .../src/omni_mercury_engine/cognitive/ethical_bounding.py
        # -> repo root is four parents up; configs/ sits at the root.
        candidates = [
            Path(__file__).resolve().parents[3] / "configs" / "weapons_gate_calibration.json",
        ]
        float_fields = {f.name for f in dataclass_fields(cls) if f.name not in ("source",)}
        for path in candidates:
            try:
                if not path.is_file():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                params = data.get("parameters", data)
                overrides = {k: float(v) for k, v in params.items() if k in float_fields}
                if not overrides:
                    continue
                return cls(**overrides, source=str(path))
            except Exception:  # pragma: no cover - fail-safe to defaults
                logger.warning("failed to load weapons-gate calibration from %s", path)
        return cls()


BENEVOLENCE_CALIBRATION = BenevolenceCalibration.load_default()


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
# Weapons / mass-casualty uplift gate -- two-axis (hazard domain x operational
# intent) assessment. See docs/HARM_POLICY.md for the full rationale.
#
# Root-cause fix for a structural gap a scalar "does this text mention a
# hazardous topic" check cannot resolve: a gate loose enough to catch a real
# attack false-positives on every profession that works *inside* the same
# hazard vocabulary every day (toxicology, pathology, virology,
# critical-infrastructure safety, licensed demolition/blasting, atmospheric
# modeling, CBRN emergency response, ...); a gate tight enough to spare them
# misses real attacks. The two axes resolve the conflict:
#
#   Axis A (HazardDomain) -- a deliberately high-recall, no-precision-
#   required *router*. A hit only decides whether to run the real (Axis B)
#   assessment; it is NEVER by itself a reason to refuse. Over-triggering
#   here is fine and expected.
#
#   Axis B (OperationalIntent) -- the actual gate. It asks "does fulfilling
#   this materially help someone build, produce, acquire, weaponize,
#   disseminate, enhance, or deploy a weapon, cause mass casualties, or
#   defeat a safety/screening control" -- not "does this mention a
#   dangerous substance/method." Mechanism, detection, defense/treatment,
#   response, policy, and licensed professional practice default to ALLOW
#   (some logged); only the narrow production/weaponization/acquisition-
#   evasion/offensive-enhancement/targeting intersection is refused.
#
# This is deliberately *more permissive* than a keyword blocklist for every
# legitimate profession Mercury serves, and strictly *stronger* on the thing
# that actually matters. All lexicons below are taxonomy-level category
# terms (the same vocabulary used in public CBRN/non-proliferation policy
# literature) -- never operational specifics -- consistent with the
# high-recall/no-precision-required role Axis A is meant to play.
# ---------------------------------------------------------------------------


class HazardDomain(Enum):
    """Axis A: the hazard vocabulary a piece of text falls in (routing only)."""

    NONE = "none"
    CHEMICAL = "chemical"
    BIOLOGICAL = "biological"
    RADIOLOGICAL = "radiological"
    NUCLEAR = "nuclear"
    HIGH_YIELD_EXPLOSIVE = "high_yield_explosive"
    CONVENTIONAL_WEAPON = "conventional_weapon"
    CYBER_OFFENSIVE = "cyber_offensive"
    OTHER_MASS_HARM = "other_mass_harm"


class OperationalIntent(Enum):
    """Axis B: actionability toward harm, low -> high (the real gate)."""

    MECHANISM = "mechanism"  # B0 - science / "how does X work"
    DETECTION = "detection"  # B1 - detection / diagnosis / identification
    DEFENSE = "defense"  # B2 - defense / mitigation / treatment / decontamination
    RESPONSE = "response"  # B3 - response / triage / consequence & dispersion modeling
    POLICY = "policy"  # B4 - policy / compliance / history / safety standards
    LICENSED_PRACTICE = "licensed_practice"  # B5 - licensed professional practice
    PRODUCTION = "production"  # B6 - production/synthesis route toward a weapon
    WEAPONIZATION = "weaponization"  # B7 - weaponization / delivery / dissemination design
    ACQUISITION = "acquisition"  # B8 - acquisition / diversion / screening-evasion
    ENHANCEMENT = "enhancement"  # B9 - offensive pathogen/agent enhancement
    TARGETING = "targeting"  # B10 - targeting / attack planning to maximize casualties


# Ordering used to pick the single highest (most actionable) matched tier.
_INTENT_ORDER: tuple[OperationalIntent, ...] = (
    OperationalIntent.MECHANISM,
    OperationalIntent.DETECTION,
    OperationalIntent.DEFENSE,
    OperationalIntent.RESPONSE,
    OperationalIntent.POLICY,
    OperationalIntent.LICENSED_PRACTICE,
    OperationalIntent.PRODUCTION,
    OperationalIntent.WEAPONIZATION,
    OperationalIntent.ACQUISITION,
    OperationalIntent.ENHANCEMENT,
    OperationalIntent.TARGETING,
)
# Intents whose *phrasing is weapons-specific on its own* -- "weaponize",
# "increase transmissibility/lethality", "immune escape", "maximize
# casualties", "attack plan targeting a crowd". Only these justify the
# no-hazard-domain safety net in assess_weapons_uplift, because they cannot
# describe a benign activity. The generic production verbs
# (make/cook/build/synthesize) and the acquisition-evasion verbs
# (smuggle/evade detection) are deliberately EXCLUDED: they are meaningful
# only once Axis A has routed a hazard domain, and firing the safety net on
# them alone would false-positive on benign text ("how do I cook dinner",
# "how does antivirus evade detection").
_INHERENTLY_OFFENSIVE_INTENTS = frozenset(
    {
        OperationalIntent.WEAPONIZATION,
        OperationalIntent.ENHANCEMENT,
        OperationalIntent.TARGETING,
    }
)


class WeaponsDisposition(Enum):
    """Calibrated response ladder for a weapons/mass-casualty verdict.

    Not binary, so professionals are not crippled by a gate tuned for genuine
    mass-casualty uplift. In increasing order of restriction:

    * ``ALLOW`` -- permit, no special handling.
    * ``ALLOW_LOG`` -- permit, log (licensed-practice framing).
    * ``ALLOW_PROVENANCE`` -- permit, but the emitted answer MUST be
      source-attributed. Assigned to an otherwise-allowable query that sits in a
      high-severity hazard domain (defensive/mechanistic CBRN etc.): the answer
      is available, but only sourced/cited -- never uncited synthesis on a
      hazardous topic. Enforced at the output boundary (GeneralAssistant), which
      withholds when no citable source is available. Non-blocking.
    * ``ESCALATE`` -- gray zone; defer to a human-in-the-loop reviewer.
    * ``REFUSE_REDACT`` -- refuse the operational part; redact at sentence level.
    * ``HARD_REFUSE`` -- refuse outright.
    """

    ALLOW = "allow"
    ALLOW_LOG = "allow_log"
    ALLOW_PROVENANCE = "allow_provenance"
    ESCALATE = "escalate"
    REFUSE_REDACT = "refuse_redact"
    HARD_REFUSE = "hard_refuse"


@dataclass(frozen=True)
class WeaponsRiskAssessment:
    """Result of the two-axis weapons/mass-casualty uplift assessment.

    Attributes:
        hazard_domain: Axis A -- the routing hazard vocabulary detected.
        hazard_weight: Per-domain severity weight in ``[0, 1]`` (``0.0`` for
            :attr:`HazardDomain.NONE`).
        intent_tier: Axis B -- the highest-actionability intent matched.
        confidence: In ``[0, 1]``; how one-sidedly the evidence favors
            offensive intent over allow-signal (mechanism/detection/
            defense/...) intent. Low confidence in the B6 gray zone favors
            :attr:`WeaponsDisposition.ESCALATE` over
            :attr:`WeaponsDisposition.REFUSE_REDACT`.
        disposition: The calibrated response per the decision rule.
        signals: Human-readable matched-pattern labels, for audit logging.
    """

    hazard_domain: HazardDomain = HazardDomain.NONE
    hazard_weight: float = 0.0
    intent_tier: OperationalIntent = OperationalIntent.MECHANISM
    confidence: float = 0.0
    disposition: WeaponsDisposition = WeaponsDisposition.ALLOW
    signals: tuple[str, ...] = ()

    @property
    def blocks(self) -> bool:
        """True when this assessment must prevent the action outright."""
        return self.disposition in (
            WeaponsDisposition.ESCALATE,
            WeaponsDisposition.REFUSE_REDACT,
            WeaponsDisposition.HARD_REFUSE,
        )


# Axis-A routing lexicon: deliberately high recall, taxonomy-level category
# terms only. A hit here only decides whether Axis B runs -- it is never, by
# itself, a reason to refuse.
_HAZARD_DOMAIN_KEYWORDS: dict[HazardDomain, tuple[str, ...]] = {
    HazardDomain.CHEMICAL: (
        "nerve agent",
        "nerve toxin",
        "chemical warfare agent",
        "chemical weapon",
        "blister agent",
        "choking agent",
        "blood agent",
        "organophosphate",
        "chlorine gas",
        "phosgene",
        "sarin",
        "tabun",
        "soman",
        "vx nerve",
        "mustard gas",
        "novichok",
        "cyanide gas",
    ),
    HazardDomain.BIOLOGICAL: (
        "biological weapon",
        "bioweapon",
        "biological agent",
        "select agent",
        "pathogen",
        "gain-of-function",
        "gain of function",
        "pandemic",
        "virulence",
        "transmissibility",
        "anthrax",
        "botulinum toxin",
        "ricin",
        "smallpox",
        "plague bacterium",
        "biohazard",
        "bioterrorism",
    ),
    HazardDomain.RADIOLOGICAL: (
        "dirty bomb",
        "radiological dispersal device",
        "radioactive source",
        "radiological weapon",
        "cesium-137",
        "cobalt-60",
        "orphan source",
    ),
    HazardDomain.NUCLEAR: (
        "nuclear weapon",
        "nuclear device",
        "fissile material",
        "enriched uranium",
        "highly enriched uranium",
        "weapons-grade uranium",
        "weapons grade uranium",
        "uranium enrichment",
        "enriching uranium",
        # Morphological/word-order variants of the enrichment vocabulary above.
        # The lexicon is high-recall routing only (a hit merely subjects the
        # query to Axis-B intent analysis; benign/mechanistic enrichment
        # discussion resolves to ALLOW/ALLOW_PROVENANCE, never a refusal), so
        # covering the bare-verb form "enrich uranium" -- as in "how to enrich
        # uranium to weapons-grade", which the -ing/-ment forms miss -- is
        # FP-safe and closes a nuclear production false-negative.
        "enrich uranium",
        "enrich plutonium",
        "reprocess plutonium",
        "reprocessing plutonium",
        "weapons-grade plutonium",
        "weapons grade plutonium",
        "nuclear warhead",
        "gas centrifuge cascade",
        "critical mass assembly",
    ),
    HazardDomain.HIGH_YIELD_EXPLOSIVE: (
        "improvised explosive device",
        "high explosive charge",
        "detonator",
        "explosive precursor",
        "explosive properties",
        "commercial explosive",
        "blasting",
        "pipe bomb",
        "car bomb",
        "vehicle-borne explosive",
        "tatp",
        "petn",
        "rdx",
        "semtex",
        "plastic explosive",
    ),
    HazardDomain.CONVENTIONAL_WEAPON: (
        "untraceable firearm",
        "ghost gun",
        "illegal firearm conversion",
        "automatic weapon conversion",
        "silencer fabrication",
    ),
    HazardDomain.CYBER_OFFENSIVE: (
        "ransomware deployment",
        "zero-day exploit",
        "critical infrastructure attack",
        "scada attack",
        "botnet for attack",
        "destructive malware",
    ),
    HazardDomain.OTHER_MASS_HARM: (
        "weapon of mass destruction",
        "mass casualty weapon",
        "mass-casualty attack",
    ),
}
_HAZARD_DOMAIN_WEIGHTS: dict[HazardDomain, float] = {
    HazardDomain.NONE: 0.0,
    HazardDomain.CHEMICAL: 0.9,
    HazardDomain.BIOLOGICAL: 1.0,
    HazardDomain.RADIOLOGICAL: 0.95,
    HazardDomain.NUCLEAR: 1.0,
    HazardDomain.HIGH_YIELD_EXPLOSIVE: 0.85,
    HazardDomain.CONVENTIONAL_WEAPON: 0.6,
    HazardDomain.CYBER_OFFENSIVE: 0.65,
    HazardDomain.OTHER_MASS_HARM: 0.9,
}

# Per-domain weight at/above which a domain is "high severity" (mass-casualty
# CBRN + high-yield explosive + mass-harm). Used to route an otherwise-allowable
# query to ALLOW_PROVENANCE (source-attribution required) rather than plain ALLOW.
HIGH_SEVERITY_WEIGHT: float = 0.85

# Merge the multilingual/transliterated taxonomy terms (native script + Latin
# transliteration across widely spoken languages) into the Axis-A router. Axis A
# is high-recall/no-precision -- a hit only decides whether Axis B runs, never a
# refusal on its own -- so broad multilingual coverage is safe and desired. The
# obfuscation-normalized haystack (see assess_weapons_uplift) additionally lets
# these match through leetspeak, homoglyph spoofing, and separator insertion.
_HAZARD_DOMAIN_KEYWORDS = {
    domain: keywords + MULTILINGUAL_HAZARD_TERMS.get(domain.value, ())
    for domain, keywords in _HAZARD_DOMAIN_KEYWORDS.items()
}

# Hazard-context anchors: generic (not agent-specific) nouns that mark a query
# as being *about* a weapon/agent even when the specific agent is absent from
# the Axis-A lexicon. Paired with a production/acquisition verb below, they close
# the out-of-lexicon false-negative (Task #4: "synthesis route for <novel agent>"
# where <novel agent> is not itself a listed keyword). Deliberately weapons-
# anchored so a benign "how do I make bread" carries none of them.
_HAZARD_CONTEXT_ANCHORS: frozenset[str] = frozenset(
    {
        "nerve agent",
        "chemical agent",
        "biological agent",
        "toxin",
        "toxic agent",
        "poison gas",
        "poisonous gas",
        "lethal gas",
        "deadly gas",
        "chemical weapon",
        "biological weapon",
        "bioweapon",
        "pathogen",
        "select agent",
        "explosive",
        "explosive charge",
        "warhead",
        "propellant",
        "fissile",
        "radiological",
        "weapon of mass destruction",
        "mass casualty",
        "mass-casualty",
        "chemical warfare",
        "biological warfare",
    }
)

# Axis-B intent patterns: (compiled regex, tier, audit label). Offensive
# patterns match *actionable verbs/phrases* ("how to produce", "evade
# detection", "maximize casualties"), never the hazard noun alone -- Axis A
# already covers that. Allow patterns match the mechanism/detection/defense/
# response/policy/licensed-practice frame the professions in
# docs/HARM_POLICY.md work inside every day.
_OFFENSIVE_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], OperationalIntent, str], ...] = (
    (
        re.compile(
            r"\b(?:how to|how (?:do|can|would|should) (?:i|we|one|you)|steps? to|"
            r"guide (?:to|for|on)|instructions? (?:to|for|on)|recipe (?:to|for)|"
            r"tutorial (?:to|for|on)|help me|teach me to|(?:want|trying|need) to|"
            r"best way to)\s+(?:make|produce|synthesi[sz]e|manufacture|fabricate|"
            r"build|construct|assemble|formulate|create|generate)\b"
        ),
        OperationalIntent.PRODUCTION,
        "production_howto",
    ),
    (
        re.compile(
            r"\bsynthesis route\b|\bsynthesis steps?\b|\bproduction process for\b|"
            r"\bprecursors? (?:for|to make)\b|"
            # indirect / gerund production framings ("process of making",
            # "go about producing", "fabrication of", "cooking up", "enriching
            # uranium to weapons grade"). Verb stems catch make/making/produced/...
            r"\b(?:process|method|procedure|steps?) (?:of|for) (?:mak|produc|"
            r"synthesi[sz]|manufactur|build|fabricat|construct|assembl|creat)\w*|"
            r"\b(?:go about|trying to|want to|need to|help me|teach me to) (?:mak|"
            r"produc|synthesi[sz]|manufactur|build|fabricat|construct|assembl|creat|"
            r"generat)\w*|\bfabrication of\b|\bcooking up\b|"
            r"\brecipe for (?:a |an )?homemade\b|"
            # Weapons-DIRECTED nuclear enrichment/reprocessing. An enrichment or
            # reprocessing verb *directed* (to/for/into) at an explicit weapons
            # marker (weapons-grade / warhead / a nuclear weapon-bomb-device) is
            # production intent regardless of word order or morphology, so
            # "enrich uranium to weapons-grade" and "enriching uranium to 90%
            # U-235 for a warhead" both gate. The directional preposition is what
            # keeps this FP-safe: a definitional/policy question ("what percent
            # enrichment IS weapons-grade uranium") uses a copula, not to/for/into,
            # and a civilian "enrich uranium for reactor fuel" carries no weapons
            # marker -- both stay mechanism (ALLOW_PROVENANCE), not a refusal.
            # ``.`` never crosses the newline between normalization variants.
            r"\b(?:enrich|reprocess)\w*\b.{0,40}?\b(?:to|for|into)\b.{0,25}?"
            r"\b(?:weapons?[- ]grade|warhead|(?:a |an )?nuclear (?:weapon|device|bomb))\b"
        ),
        OperationalIntent.PRODUCTION,
        "production_route",
    ),
    (
        re.compile(
            r"\bweaponi[sz](?:e|es|ed|ing|ation)\b|\bdelivery mechanism\b|"
            r"\bdissemination device\b|\baerosoli[sz](?:e|es|ed|ing)\b|"
            r"\bdetonation mechanism\b|"
            r"\barm(?:ing)? the device\b|\bwarhead design\b|"
            # "turn X into a weapon/bomb" and "device that disperses/releases X"
            r"\bturn(?:s|ing)?\b.{0,30}?\binto a (?:weapon|bioweapon|bomb|"
            r"dispersal device)\b|"
            r"\bdevice that (?:disperses|releases|aerosoli[sz]es|spreads|"
            r"disseminates)\b"
        ),
        OperationalIntent.WEAPONIZATION,
        "weaponization_design",
    ),
    (
        re.compile(
            r"\b(?:evade|avoid|escape)\s+detection\b|\bwithout (?:being )?detect(?:ed|ion)\b|"
            r"\bbypass screening\b|\bcircumvent (?:export controls?|safeguards?|customs)\b|"
            r"\bsmuggl(?:e|ing)\b|\bblack market\b|\bundetectable\b|\bavoid (?:customs|inspection)\b|"
            r"\bdivert(?:ing)? .*(?:material|precursor)\b"
        ),
        OperationalIntent.ACQUISITION,
        "acquisition_evasion",
    ),
    (
        re.compile(
            r"\bincrease (?:the )?(?:transmissibility|lethality|virulence)\b|"
            r"\benhance .*(?:pathogen|virus|toxin)\b|\bimmune escape\b|"
            r"\bengineer .*(?:more (?:lethal|contagious)|immune evasion)\b"
        ),
        OperationalIntent.ENHANCEMENT,
        "offensive_enhancement",
    ),
    (
        re.compile(
            r"\bmaximi[sz]e casualties\b|\battack plan\b|\btarget(?:ing)? "
            r"(?:a crowd|civilians|a stadium|a school|a gathering)\b|\bkill as many\b|"
            r"\bmass[- ]casualty target\b|\bantipersonnel\b|"
            r"\b(?:against|to (?:kill|harm|maim|injure)) (?:people|persons|civilians|humans)\b"
        ),
        OperationalIntent.TARGETING,
        "targeting_planning",
    ),
)
_ALLOW_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], OperationalIntent, str], ...] = (
    (
        re.compile(r"\bhow does .* work\b|\bmechanism of action\b|\bchemistry of\b|\bphysics of\b"),
        OperationalIntent.MECHANISM,
        "mechanism",
    ),
    (
        re.compile(
            r"\bdetect(?:s|ed|ion|ing|or|ors)?\b|\bidentif(?:y|ication)\b|"
            r"\bdiagnos(?:is|e|tic|tics)\b|\bscreen(?:ing)? for\b|\btest(?:ing)? for\b|"
            r"\btest kit\b|\bassay\b|\bsensor\b|\bbiosensor\b|\balarm\b|\bmonitor(?:ing)?\b|"
            r"\brecogni[sz]e symptoms\b|\bsequencing for diagnosis\b|\bsurveillance\b|"
            r"\bepidemiology\b"
        ),
        OperationalIntent.DETECTION,
        "detection",
    ),
    (
        re.compile(
            r"\btreat(?:ment|ing)?\b|\bantidote\b|\bdecontaminat(?:e|ion)\b|"
            r"\bprotective equipment\b|\bppe\b|\bvaccin(?:e|ation)\b|\btherapy\b|"
            r"\bdosing\b|\boxime\b|\batropine\b|\bcasualty care\b|\bcountermeasure\b|"
            r"\bprophylaxis\b|\bprotective (?:gear|suit|mask|clothing)\b|\brespirator\b|"
            r"\bfilter(?:ing)?\b|\bmedical countermeasure\b"
        ),
        OperationalIntent.DEFENSE,
        "defense_treatment",
    ),
    (
        re.compile(
            r"\btriage\b|\bconsequence modeling\b|\bdispersion model(?:ing)?\b|"
            r"\bplume model(?:ing)?\b|\bevacuation plan(?:ning)?\b|\bemergency response\b|"
            r"\bcasualty management\b"
        ),
        OperationalIntent.RESPONSE,
        "response_modeling",
    ),
    (
        re.compile(
            r"\bpolicy\b|\bregulation\b|\bcompliance\b|\btreaty\b|\bhistory of\b|"
            r"\bsafety standards?\b|\bnon-proliferation\b|\bexport control law\b"
        ),
        OperationalIntent.POLICY,
        "policy_history",
    ),
    (
        re.compile(
            r"\blicensed\b|\bcertified\b|\bunder permit\b|\bregulatory approval\b|"
            r"\bcommercial (?:blasting|demolition)\b|\bcontrolled demolition\b|"
            r"\blicensing (?:standard|requirement)\b"
        ),
        OperationalIntent.LICENSED_PRACTICE,
        "licensed_practice",
    ),
)


# Multilingual Axis-B offensive cues compiled to (substring, intent, label).
# The cue tier name (a plain string in the language table so that module needs
# no enum import) is resolved to an OperationalIntent here.
_MULTILINGUAL_OFFENSIVE: tuple[tuple[str, OperationalIntent, str], ...] = tuple(
    (cue, OperationalIntent[tier_name], label)
    for cue, tier_name, label in MULTILINGUAL_OFFENSIVE_CUES
)


def _match_hazard_domain(haystack: str) -> HazardDomain:
    """Axis A: cheap high-recall routing match; returns the first domain hit."""
    for domain, keywords in _HAZARD_DOMAIN_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return domain
    return HazardDomain.NONE


def _match_multilingual_offensive(haystack: str) -> list[tuple[OperationalIntent, str]]:
    """Substring pass for the multilingual offensive-intent cues."""
    return [(tier, label) for cue, tier, label in _MULTILINGUAL_OFFENSIVE if cue in haystack]


def _match_intent_patterns(
    haystack: str, patterns: tuple[tuple[re.Pattern[str], OperationalIntent, str], ...]
) -> list[tuple[OperationalIntent, str]]:
    return [(tier, label) for pattern, tier, label in patterns if pattern.search(haystack)]


def _offensive_confidence(
    n_offensive: int,
    n_allow: int,
    hazard_weight: float,
    classifier_boost: float,
    cal: BenevolenceCalibration = BENEVOLENCE_CALIBRATION,
) -> float:
    """Calibrated logistic confidence that evidence favors offensive intent.

    ``sigmoid(bias + w_off*n_off - w_allow*n_allow + w_weight*weight
    + w_classifier*boost)``. Monotone increasing in offensive evidence, the
    hazard weight, and the classifier signal; decreasing in allow-signal. The
    coefficients are fit on the labeled corpus (see
    :meth:`BenevolenceCalibration.load_default`); the defaults are honest
    fallbacks. Returns a value in ``(0, 1)``.
    """
    z = (
        cal.conf_bias
        + cal.conf_w_offensive * float(n_offensive)
        - cal.conf_w_allow * float(n_allow)
        + cal.conf_w_weight * float(hazard_weight)
        + cal.conf_w_classifier * float(classifier_boost)
    )
    # Numerically stable logistic (avoids overflow warnings on large |z|).
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))


@dataclass(frozen=True)
class _GateEvidence:
    """Routed Axis-A/Axis-B evidence for a query (pre-disposition).

    ``domain is HazardDomain.NONE`` with empty ``offensive`` means a clean ALLOW
    (no hazard vocabulary and no gating offensive intent).
    """

    domain: HazardDomain
    weight: float
    offensive: list[tuple[OperationalIntent, str]]
    allowed: list[tuple[OperationalIntent, str]]


def _gate_evidence(text: str, context: dict[str, Any] | None) -> _GateEvidence:
    """Route a query through Axis A + Axis B and return the matched evidence.

    Shared by :func:`assess_weapons_uplift` (which maps evidence -> disposition)
    and :func:`compute_gate_features` (which exposes the evidence to the
    calibration fitter), so the two can never diverge.
    """
    ctx = context or {}
    # Obfuscation-resistant, multilingual normalization: the matchers run over
    # base + homoglyph/leet-folded + separator-collapsed variants at once, so
    # "n3rv3 ag3nt", Cyrillic-glyph spoofing, and "s a r i n" are all seen.
    haystack = normalized_haystack(f"{text}\n{ctx}")
    domain = _match_hazard_domain(haystack)
    offensive = _match_intent_patterns(
        haystack, _OFFENSIVE_INTENT_PATTERNS
    ) + _match_multilingual_offensive(haystack)

    # Safety net (no Axis-A hazard domain routed). Two ways an offensive request
    # still gates: (a) an *inherently weapons-specific* phrase; or (b) a
    # production/acquisition verb ANCHORED to generic hazard-context vocabulary
    # (Task #4 out-of-lexicon false-negative). The bare verb alone is NOT enough
    # (it would false-positive on "how do I cook dinner"). Either way the query
    # is routed to OTHER_MASS_HARM so Axis B gates it.
    if domain is HazardDomain.NONE:
        anchored = any(anchor in haystack for anchor in _HAZARD_CONTEXT_ANCHORS)
        gated_intents = set(_INHERENTLY_OFFENSIVE_INTENTS)
        if anchored:
            gated_intents |= {OperationalIntent.PRODUCTION, OperationalIntent.ACQUISITION}
        keep = [(tier, label) for tier, label in offensive if tier in gated_intents]
        if not keep:
            return _GateEvidence(HazardDomain.NONE, 0.0, [], [])
        domain = HazardDomain.OTHER_MASS_HARM
        offensive = keep

    allowed = _match_intent_patterns(haystack, _ALLOW_INTENT_PATTERNS)
    weight = _HAZARD_DOMAIN_WEIGHTS[domain]

    # Defensive-production carve-out (protect professionals). *Making* a detector,
    # sensor, vaccine, antidote, countermeasure, or protective equipment is
    # defensive work, not weapons production -- but the production verb ("make",
    # "process of making", "create") fires on it. When the only offensive tier is
    # B6 PRODUCTION *and* a detection / defense / response allow-signal is present,
    # the production is defensively framed: drop the PRODUCTION matches so the
    # query resolves to the ALLOW ladder. B7-B10 (weaponization / acquisition /
    # enhancement / targeting) are inherently offensive and are NEVER carved out
    # by a defensive object. The residual (an attacker appending a defensive noun)
    # is carried by the reasoning-backed classifier + escalation + audit -- the
    # deliberate bias is toward not strangling defensive CBRN work.
    _DEFENSIVE_ALLOW = {
        OperationalIntent.DETECTION,
        OperationalIntent.DEFENSE,
        OperationalIntent.RESPONSE,
    }
    if any(t is OperationalIntent.PRODUCTION for t, _ in offensive) and (
        _DEFENSIVE_ALLOW & {t for t, _ in allowed}
    ):
        offensive = [(t, lbl) for t, lbl in offensive if t is not OperationalIntent.PRODUCTION]

    return _GateEvidence(domain, weight, offensive, allowed)


def compute_gate_features(
    text: str,
    context: dict[str, Any] | None = None,
    *,
    harm_classifier: Any | None = None,
) -> tuple[int, int, float, float]:
    """Return ``(n_offensive_labels, n_allow_labels, hazard_weight, classifier_boost)``.

    The exact feature vector the confidence logistic consumes, exposed so
    ``scripts/fit_weapons_gate_calibration.py`` fits the SAME features the gate
    scores at runtime. ``classifier_boost`` is ``0.0`` unless a ``harm_classifier``
    is supplied and returns a value.
    """
    ev = _gate_evidence(text, context)
    n_off = len({label for _, label in ev.offensive})
    n_allow = len({label for _, label in ev.allowed})
    boost = 0.0
    if harm_classifier is not None:
        try:
            boost = max(0.0, min(1.0, float(harm_classifier(text))))
        except Exception:  # pragma: no cover - fail-open
            boost = 0.0
    return n_off, n_allow, ev.weight, boost


def assess_weapons_uplift(
    text: str,
    context: dict[str, Any] | None = None,
    *,
    harm_classifier: Any | None = None,
) -> WeaponsRiskAssessment:
    """Two-axis (hazard-domain x operational-intent) weapons/mass-casualty gate.

    Answers "does fulfilling this materially help someone build, produce,
    acquire, weaponize, disseminate, enhance, or deploy a weapon, cause mass
    casualties, or defeat a safety/screening control" -- not "does this
    mention a hazardous topic." See ``docs/HARM_POLICY.md`` for the full
    rationale and response ladder.

    Args:
        text: The action/query/content to assess.
        context: Optional context; ``context["licensed_context"]`` (truthy)
            marks caller-asserted licensed-professional context, which can
            widen a non-gray B6 verdict to ESCALATE -- it can never downgrade
            a B7-B10 HARD_REFUSE, nor upgrade a REFUSE_REDACT/HARD_REFUSE to
            an ALLOW variant.
        harm_classifier: Optional ``Callable[[str], float]`` -- Mercury's
            existing pluggable meaning-level classifier hook (e.g.
            :func:`~omni_mercury_engine.reasoning.backends.reasoning_harm_classifier`).
            Consulted, not trusted: it can only raise offensive confidence in
            the B6 gray zone, never lower a disposition already earned by
            lexical evidence; its own failure is treated as "no additional
            evidence," not a crash.

    Returns:
        A :class:`WeaponsRiskAssessment`. Fail-closed: any internal error
        (malformed input, a regex/logic bug) yields ``HARD_REFUSE`` rather
        than propagating an exception or silently defaulting to ALLOW.
    """
    ctx = context or {}
    try:
        ev = _gate_evidence(text, ctx)
        domain, weight, offensive, allowed = ev.domain, ev.weight, ev.offensive, ev.allowed

        if domain is HazardDomain.NONE and not offensive:
            return WeaponsRiskAssessment()  # clean ALLOW / MECHANISM / NONE defaults

        if not offensive:
            # Hazard vocabulary present but no offensive-actionability signal at
            # all: permit. Licensed-practice framing is ALLOW_LOG; an otherwise-
            # allowable query that sits in a HIGH-severity hazard domain (weight
            # >= HIGH_SEVERITY_WEIGHT: CBRN / high-yield-explosive / mass-harm) is
            # ALLOW_PROVENANCE -- answerable, but only source-attributed, never
            # uncited synthesis on a hazardous topic (enforced at the output
            # boundary). Everything else is plain ALLOW.
            tier = max(
                (t for t, _ in allowed),
                key=_INTENT_ORDER.index,
                default=OperationalIntent.MECHANISM,
            )
            if tier is OperationalIntent.LICENSED_PRACTICE:
                disposition = WeaponsDisposition.ALLOW_LOG
            elif weight >= HIGH_SEVERITY_WEIGHT:
                disposition = WeaponsDisposition.ALLOW_PROVENANCE
            else:
                disposition = WeaponsDisposition.ALLOW
            return WeaponsRiskAssessment(
                domain, weight, tier, 0.0, disposition, tuple(label for _, label in allowed)
            )

        tier = max((t for t, _ in offensive), key=_INTENT_ORDER.index)
        offensive_labels = tuple(sorted({label for _, label in offensive}))
        allow_labels = tuple(sorted({label for _, label in allowed}))

        # Confidence: how one-sidedly the evidence favors offensive intent, via
        # the calibrated logistic in BENEVOLENCE_CALIBRATION (fit on the labeled
        # corpus, not a raw match ratio). Mixed evidence (e.g. a defensive-
        # treatment query that also brushes a production-adjacent phrase) reads
        # as genuinely ambiguous, which only matters for the B6 gray zone below
        # -- B7-B10 refuse regardless. The optional meaning-level classifier is
        # consulted here and can only RAISE confidence, never lower a disposition
        # already earned by lexical evidence.
        classifier_boost = 0.0
        if harm_classifier is not None:
            try:
                classifier_boost = max(0.0, min(1.0, float(harm_classifier(text))))
            except Exception as exc:
                logger.warning(
                    "weapons-uplift harm_classifier failed (%s); no confidence boost applied", exc
                )
        confidence = _offensive_confidence(
            len(offensive_labels), len(allow_labels), weight, classifier_boost
        )

        licensed_context = bool(ctx.get("licensed_context")) or "licensed_practice" in allow_labels
        signals = offensive_labels + allow_labels

        if tier is not OperationalIntent.PRODUCTION:
            # B7-B10: hard refuse. No gray zone, no partial, no escalation --
            # this is the ~6-category intersection that does not overlap any
            # legitimate profession.
            return WeaponsRiskAssessment(
                domain, weight, tier, confidence, WeaponsDisposition.HARD_REFUSE, signals
            )

        # B6 (PRODUCTION): the one genuine gray zone. Low confidence or an
        # asserted licensed-professional context slows and audits a real
        # engineer instead of denying them outright; otherwise refuse (the
        # defensive/mechanistic remainder stays answerable via the output-
        # gate's REFUSE_REDACT sentence-level redaction, not a blanket denial).
        if confidence < BENEVOLENCE_CALIBRATION.weapons_b6_escalate_confidence or licensed_context:
            disposition = WeaponsDisposition.ESCALATE
        else:
            disposition = WeaponsDisposition.REFUSE_REDACT
        return WeaponsRiskAssessment(domain, weight, tier, confidence, disposition, signals)
    except Exception:
        logger.exception("assess_weapons_uplift failed; failing closed to HARD_REFUSE")
        return WeaponsRiskAssessment(
            HazardDomain.OTHER_MASS_HARM,
            1.0,
            OperationalIntent.TARGETING,
            0.0,
            WeaponsDisposition.HARD_REFUSE,
            ("assessment_error",),
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
    # Two-axis weapons/mass-casualty uplift verdict (see assess_weapons_uplift).
    # Defaults are the "no hazard vocabulary detected" case so older positional
    # constructors and callers that never touch weapons content are unaffected.
    hazard_domain: str = "none"
    operational_intent: str = "mechanism"
    weapons_disposition: str = "allow"


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
        # Populated by evaluate_harm(); read back by BenevolenceScorer.score_action
        # so the two-axis weapons verdict rides along with the harm computation
        # instead of being assessed twice. Single-caller-per-instance-at-a-time
        # assumption, consistent with the counters above.
        self.last_weapons_assessment: WeaponsRiskAssessment = WeaponsRiskAssessment()

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

        # Two-axis weapons/mass-casualty uplift assessment (Axis A hazard
        # routing + Axis B operational-intent gate). A blocking disposition
        # raises PHYSICAL (direct casualty risk) and SOCIETAL (mass-harm
        # blast radius) to a floor -- fail-closed (max-only), never lowering
        # the lexical harm already computed above. This is the ONE harm
        # policy: every caller of HarmReducer/BenevolenceScorer -- including
        # GeneralAssistant's general-capability gate -- inherits this signal
        # through the ordinary harm_score/EthicalScore path rather than a
        # second, divergent check.
        self.last_weapons_assessment = assess_weapons_uplift(
            action, context, harm_classifier=self._harm_classifier
        )
        # A blocking weapons verdict floors both the relevant categories (so the
        # breakdown and the severity damping reflect it) AND the overall harm
        # (below). Flooring only 2 of 8 categories would otherwise be diluted by
        # the weighted average to well under 0.5 -- e.g. a HARD_REFUSE with two
        # categories at 0.95 averages to ~0.26 -- so the scalar harm_score would
        # not track the verdict. Both are max-only / fail-closed (never lower).
        weapons_overall_floor = 0.0
        if self.last_weapons_assessment.disposition is WeaponsDisposition.HARD_REFUSE:
            floor = BENEVOLENCE_CALIBRATION.weapons_hard_refuse_harm_floor
            breakdown[HarmCategory.PHYSICAL.value] = max(
                breakdown[HarmCategory.PHYSICAL.value], floor
            )
            breakdown[HarmCategory.SOCIETAL.value] = max(
                breakdown[HarmCategory.SOCIETAL.value], floor
            )
            weapons_overall_floor = floor
        elif self.last_weapons_assessment.disposition is WeaponsDisposition.REFUSE_REDACT:
            floor = BENEVOLENCE_CALIBRATION.weapons_refuse_redact_harm_floor
            breakdown[HarmCategory.PHYSICAL.value] = max(
                breakdown[HarmCategory.PHYSICAL.value], floor
            )
            breakdown[HarmCategory.SOCIETAL.value] = max(
                breakdown[HarmCategory.SOCIETAL.value], floor
            )
            weapons_overall_floor = floor
        elif self.last_weapons_assessment.disposition is WeaponsDisposition.ESCALATE:
            floor = BENEVOLENCE_CALIBRATION.weapons_escalate_harm_floor
            breakdown[HarmCategory.SOCIETAL.value] = max(
                breakdown[HarmCategory.SOCIETAL.value], floor
            )
            weapons_overall_floor = floor

        weighted_sum = sum(breakdown[cat.value] * self.HARM_WEIGHTS[cat] for cat in HarmCategory)
        max_weighted = sum(self.HARM_WEIGHTS.values())
        overall_harm = max(weighted_sum / max_weighted, weapons_overall_floor)

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

        # Weapons/mass-casualty uplift hard veto: a blocking Axis-B
        # disposition (ESCALATE/REFUSE_REDACT/HARD_REFUSE) forces
        # is_permissible False regardless of the benevolence float --
        # monotone, can only revoke permission, never grant it. This is what
        # makes the general-capability layer inherit the SAME hard gate as
        # detect/analyze/predict instead of a separate, weaker check: one
        # harm policy, computed once, consulted everywhere.
        weapons = self.harm_reducer.last_weapons_assessment
        if weapons.blocks:
            is_permissible = False

        explanation = self._generate_explanation(
            action, benevolence_score, harm_score, benefit_score, is_permissible
        )
        recommendations = self._generate_recommendations(
            harm_breakdown, benefit_breakdown, is_permissible
        )
        if weapons.blocks:
            recommendations.insert(
                0,
                f"weapons-uplift gate: {weapons.disposition.value} "
                f"(hazard={weapons.hazard_domain.value}, intent={weapons.intent_tier.value})",
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
            hazard_domain=weapons.hazard_domain.value,
            operational_intent=weapons.intent_tier.value,
            weapons_disposition=weapons.disposition.value,
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
