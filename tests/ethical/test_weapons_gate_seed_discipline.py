# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Held-out discipline for the weapons-gate adversarial slice.

The axes are not disjoint: ``build_adversarial_corpus`` re-emits each
``out_of_lexicon`` seed into ``obfuscation`` (leet-mangled) and ``paraphrase``
(once per framing), so fixing one seed moves three axes mechanically. Pinned
here because cross-axis movement was once read as evidence of generalisation
(retracted in ``docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md``).

Pins: the seed structure and 4.59x amplification, so seed-level metrics are
available next to row-level ones; and a ratchet on lexicon terms occurring in
the slice, split into terms added after measuring it and pre-existing hazard
vocabulary.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from weapons_gate_adversarial import (
    _FRAMINGS,
    _OUT_OF_LEXICON_OFFENSIVE,
    AdversarialRow,
    build_adversarial_corpus,
)

#: Measured structure of the slice. Pinned so a change to the generator that
#: alters the amplification or the axis overlap has to be acknowledged.
MEASURED_SEEDS = 29
MEASURED_OFFENSIVE_ROWS = 133
MEASURED_MULTI_AXIS_SEEDS = 16
MEASURED_MAX_AXES_PER_SEED = 3
#: Rows whose seed is one of the six ``out_of_lexicon`` seeds, by axis.
MEASURED_OOL_FOOTPRINT = {"out_of_lexicon": 6, "obfuscation": 6, "paraphrase": 18}

#: Terms added to the lexicon **after** the held-out slice was measured, which
#: occur in that slice. This is the methodology concern: the published claim
#: was "neither taken from the evaluation corpus", and these contradict it.
#: Withdrawn in ``docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md`` and recorded here.
#: The set is a ratchet -- it may shrink, never grow without a deliberate edit.
ADDED_AFTER_HELD_OUT_MEASUREMENT: dict[str, int] = {
    "munition": 4,
    "binary": 4,
    "fourth-generation agent": 5,
    # Only occurrence is a hard-benign row the gate must ALLOW.
    "organophosphate nerve agent": 1,
}

#: Hazard vocabulary pre-dating the slice, which unavoidably occurs in a
#: hazard corpus. Not a leak; recorded so the ratchet can exclude it.
#: Values are ``(total rows matched, of which benign)``.
PRE_EXISTING_HAZARD_VOCABULARY: dict[str, tuple[int, int]] = {
    "nerve agent": (11, 2),
    "toxin": (10, 1),
    "explosive": (9, 1),
    "explosive charge": (8, 0),
    "chemical weapon": (1, 1),
    "mass-casualty": (1, 1),
    "biological weapon": (1, 1),
    "pathogen": (1, 1),
}


#: Gate vocabulary pre-dating the slice, across the two surfaces the ratchet
#: originally missed: the Axis-A domain routing lexicon
#: (``_HAZARD_DOMAIN_KEYWORDS``) and the literals of the Axis-B intent
#: patterns. Mechanically derived, so it is pinned whole rather than curated:
#: the connective words a regex alternation contributes ("that", "would",
#: "ation") sit here beside real hazard vocabulary ("sarin", "ricin"), and
#: that is deliberate. The control is that the set cannot change without the
#: diff showing it, so a reviewer seeing one new line has one term to justify.
#: Deciding per term which words "count" would be a judgment that rots.
PRE_EXISTING_GATE_VOCABULARY: frozenset[str] = frozenset(
    {
        "action",
        "aerosoli",
        "against",
        "anthrax",
        "antidote",
        "assembl",
        "ation",
        "attack",
        "bomb",
        "botulinum toxin",
        "build",
        "care",
        "casualties",
        "casualty",
        "chemistry",
        "chlorine gas",
        "control",
        "cooking",
        "crowd",
        "decontaminat",
        "decontamination",
        "design",
        "detect",
        "detection",
        "detector",
        "detectors",
        "dirty bomb",
        "does",
        "enrich",
        "enriched uranium",
        "enriching uranium",
        "epidemiology",
        "equipment",
        "evacuation",
        "generat",
        "grade",
        "guide",
        "help",
        "highly enriched uranium",
        "history",
        "ication",
        "into",
        "kill",
        "lethal",
        "licensed",
        "make",
        "management",
        "manufactur",
        "manufacture",
        "many",
        "mass",
        "material",
        "mechanism",
        "ment",
        "method",
        "monitor",
        "mustard gas",
        "need",
        "ning",
        "nuclear",
        "organophosphate",
        "people",
        "pipe bomb",
        "plan",
        "plastic explosive",
        "policy",
        "procedure",
        "process",
        "produc",
        "producing",
        "proliferation",
        "prophylaxis",
        "protective",
        "radioactive source",
        "recipe",
        "recogni",
        "requirement",
        "ricin",
        "safeguards",
        "safety",
        "sarin",
        "should",
        "soman",
        "stadium",
        "steps",
        "synthesi",
        "synthesis",
        "teach",
        "that",
        "treat",
        "treaty",
        "triage",
        "turn",
        "vaccin",
        "weapon",
        "weaponi",
        "weapons",
        "work",
        "would",
    }
)


def _gate_terms() -> set[str]:
    """Every term the gate can match on, across both axes.

    Axis A contributes its hazard-context anchors and its domain routing
    lexicon verbatim. Axis B contributes the literals of its intent patterns,
    extracted with ``harm_normalization``'s own regexes so there is one
    definition of "literal" rather than a second that can drift from it.
    """
    from omni_mercury_engine.cognitive import ethical_bounding as eb
    from omni_mercury_engine.cognitive.harm_normalization import (
        _REGEX_ESCAPE_RE,
        _REGEX_LITERAL_WORD_RE,
    )

    terms: set[str] = set(eb._HAZARD_CONTEXT_ANCHORS)
    terms.update(t for keywords in eb._HAZARD_DOMAIN_KEYWORDS.values() for t in keywords)
    sources = (
        [pat.pattern for pat, _tier, _label in eb._OFFENSIVE_INTENT_PATTERNS]
        + [pat.pattern for pat, _tier, _label in eb._ALLOW_INTENT_PATTERNS]
        + [eb._DEFENSIVE_PRODUCTION_RE.pattern]
    )
    for source in sources:
        terms.update(_REGEX_LITERAL_WORD_RE.findall(_REGEX_ESCAPE_RE.sub(" ", source.lower())))
    return terms


@pytest.fixture(scope="module")
def corpus() -> list[AdversarialRow]:
    rows: list[AdversarialRow] = build_adversarial_corpus()
    return rows


@pytest.fixture(scope="module")
def offensive(corpus: list[AdversarialRow]) -> list[AdversarialRow]:
    return [r for r in corpus if r.label == "offensive"]


def test_every_row_records_its_seed(corpus: list[AdversarialRow]) -> None:
    """Seed provenance is present, so the overlap cannot go unnoticed again."""
    for row in corpus:
        assert row.seed, f"row has no seed provenance: {row.text[:60]!r}"
    # A seed row is its own seed.
    for seed in _OUT_OF_LEXICON_OFFENSIVE:
        matches = [r for r in corpus if r.text == seed.strip()]
        assert matches and matches[0].seed == seed.strip()


def test_the_slice_is_amplified_and_the_factor_is_pinned(offensive: list[AdversarialRow]) -> None:
    """Row-level recall is not seed-level recall, by a factor of 4.59."""
    seeds = {r.seed for r in offensive}
    assert len(offensive) == MEASURED_OFFENSIVE_ROWS
    assert len(seeds) == MEASURED_SEEDS
    amplification = len(offensive) / len(seeds)
    assert amplification == pytest.approx(4.59, abs=0.05), (
        f"amplification moved to {amplification:.2f}x; row-level metrics weight "
        "seeds unequally and the published figures assume 4.59x"
    )


def test_the_axes_are_not_disjoint(offensive: list[AdversarialRow]) -> None:
    """One seed reaches several axes, so cross-axis movement is not independent."""
    axes_per_seed: dict[str, set[str]] = defaultdict(set)
    for row in offensive:
        axes_per_seed[row.seed].add(row.axis)
    multi = {s: a for s, a in axes_per_seed.items() if len(a) > 1}

    assert len(multi) == MEASURED_MULTI_AXIS_SEEDS, (
        f"{len(multi)} seeds span more than one axis (pinned "
        f"{MEASURED_MULTI_AXIS_SEEDS}); the axes' independence changed"
    )
    assert max(len(a) for a in axes_per_seed.values()) == MEASURED_MAX_AXES_PER_SEED
    assert multi, (
        "the axes are now disjoint -- if that is deliberate, the retraction in "
        "docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md should be revisited, because "
        "cross-axis movement would then carry real information"
    )


def test_the_out_of_lexicon_seeds_footprint_is_pinned(offensive: list[AdversarialRow]) -> None:
    """Why fixing 2 of 6 out-of-lexicon seeds moved three axes by +2/+6/+2."""
    ool = {s.strip() for s in _OUT_OF_LEXICON_OFFENSIVE}
    footprint = Counter(r.axis for r in offensive if r.seed in ool)
    assert (
        dict(footprint) == MEASURED_OOL_FOOTPRINT
    ), f"out-of-lexicon seed footprint moved to {dict(footprint)}"
    assert sum(footprint.values()) == 30, "6 seeds reach 30 of the 133 attack rows"

    # The memorisation arithmetic the doc now records: k seeds fixed predicts
    # +k out_of_lexicon, +k*len(_FRAMINGS) paraphrase, +k obfuscation.
    k = 2
    assert len(_FRAMINGS) == 3
    predicted = {"out_of_lexicon": k, "paraphrase": k * len(_FRAMINGS), "obfuscation": k}
    assert predicted == {"out_of_lexicon": 2, "paraphrase": 6, "obfuscation": 2}, (
        "the published deltas (+2/+6/+2) are exactly this prediction; if the "
        "framing count changes, the retraction's arithmetic needs restating"
    )


def test_seed_level_and_row_level_recall_are_both_computable(
    offensive: list[AdversarialRow],
) -> None:
    """The grouping seed-level recall needs is available from the corpus.

    Asserts no recall value; that is pinned by
    ``test_weapons_gate_adversarial_eval.py``.
    """
    by_seed: dict[str, list[AdversarialRow]] = defaultdict(list)
    for row in offensive:
        by_seed[row.seed].append(row)
    assert len(by_seed) == MEASURED_SEEDS
    # The most-amplified seed, which row-level metrics weight most heavily.
    worst = max(by_seed.values(), key=len)
    assert len(worst) >= 5, (
        "expected at least one heavily-amplified seed; row-level recall "
        "over-weights it relative to a single-variant seed"
    )


class TestLexiconCorpusOverlap:
    """Terms lifted from the held-out corpus: recorded, ratcheted, not hidden."""

    def test_the_known_overlap_still_matches_the_recorded_counts(
        self, corpus: list[AdversarialRow]
    ) -> None:
        """Each recorded term still appears in the corpus the recorded number of times."""
        texts = [r.text.lower() for r in corpus]
        for term, pinned in ADDED_AFTER_HELD_OUT_MEASUREMENT.items():
            hits = sum(1 for t in texts if term in t)
            assert hits == pinned, (
                f"{term!r} now matches {hits} corpus rows (recorded {pinned}); "
                "update the pin deliberately and say why in the doc"
            )

    def test_the_benign_overlap_is_called_out(self, corpus: list[AdversarialRow]) -> None:
        """A hazard anchor taken from a row the gate must ALLOW."""
        term = "organophosphate nerve agent"
        benign_hits = [r for r in corpus if term in r.text.lower() and r.label == "benign"]
        assert (
            len(benign_hits) == 1
        ), f"expected exactly the one recorded hard-benign row containing {term!r}"
        assert benign_hits[0].expected == "allow"
        assert benign_hits[0].axis == "hard_benign"

    def test_pre_existing_hazard_vocabulary_overlap_is_pinned(
        self, corpus: list[AdversarialRow]
    ) -> None:
        """Expected overlap, pinned because the ratchet works by difference."""
        texts = [(r.text.lower(), r.label) for r in corpus]
        for term, (pinned_total, pinned_benign) in PRE_EXISTING_HAZARD_VOCABULARY.items():
            total = sum(1 for t, _ in texts if term in t)
            benign = sum(1 for t, label in texts if term in t and label == "benign")
            assert (total, benign) == (pinned_total, pinned_benign), (
                f"{term!r} now matches {total} rows ({benign} benign); "
                f"recorded ({pinned_total}, {pinned_benign})"
            )

    def test_no_undeclared_gate_term_overlaps_the_held_out_slice(
        self, corpus: list[AdversarialRow]
    ) -> None:
        """The ratchet: any gate term overlapping the slice must be declared.

        Deleting a term to pass this would be a safety regression, so the
        control is disclosure, not removal. It prevents a term being lifted
        from the slice unannounced.

        **Scope.** This scans all three decision-bearing surfaces. It first
        covered only ``_HAZARD_CONTEXT_ANCHORS``, the weakest of them -- an
        anchor hit only decides whether Axis B runs, and is never by itself a
        refusal. Measured, that was 9 of the 110 terms overlapping the slice:
        the Axis-A routing lexicon contributes 21 more and the Axis-B pattern
        literals 86. A term lifted into either of those contaminates the
        recall figure exactly as an anchor would, while passing the ratchet.
        A control scoped to one surface is no control on the others.
        """
        texts = [r.text.lower() for r in corpus]
        declared = (
            set(ADDED_AFTER_HELD_OUT_MEASUREMENT)
            | set(PRE_EXISTING_HAZARD_VOCABULARY)
            | set(PRE_EXISTING_GATE_VOCABULARY)
        )
        undeclared = {t for t in _gate_terms() if any(t.lower() in x for x in texts)} - declared
        assert not undeclared, (
            "gate terms occur in the held-out corpus but are declared in no "
            f"bucket: {sorted(undeclared)}. If a term was added in response to "
            "measuring this slice, put it in ADDED_AFTER_HELD_OUT_MEASUREMENT "
            "and say so where the recall figures are published -- the slice is "
            "no longer held out for it. If it pre-dates the slice, put it in "
            "PRE_EXISTING_HAZARD_VOCABULARY (hazard-context anchors, with "
            "counts) or PRE_EXISTING_GATE_VOCABULARY (Axis-A keywords and "
            "Axis-B pattern literals)."
        )

    def test_the_ratchet_scans_every_decision_bearing_surface(self) -> None:
        """The scan's input set is pinned, because narrowing it is the bug.

        The original ratchet read one lexicon and missed two, and nothing
        caught it: a control that scans too little still passes. This pins
        what :func:`_gate_terms` draws on, so dropping a surface fails here
        instead of silently shrinking the ratchet's reach.
        """
        from omni_mercury_engine.cognitive import ethical_bounding as eb

        terms = _gate_terms()
        anchors = set(eb._HAZARD_CONTEXT_ANCHORS)
        domain = {t for keywords in eb._HAZARD_DOMAIN_KEYWORDS.values() for t in keywords}
        assert anchors <= terms, "Axis-A hazard-context anchors dropped from the scan"
        assert domain <= terms, "Axis-A domain routing lexicon dropped from the scan"
        assert terms - anchors - domain, "Axis-B pattern literals dropped from the scan"
        assert {"synthesi", "weaponi"} & terms, (
            "the Axis-B literal harvest returned nothing recognisable; "
            "re-derive it against harm_normalization's extraction"
        )
