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

    def test_no_undeclared_anchor_overlaps_the_held_out_slice(
        self, corpus: list[AdversarialRow]
    ) -> None:
        """The ratchet: an anchor overlapping the slice must be in one bucket.

        Deleting an anchor to pass this would be a safety regression, so the
        control is disclosure, not removal. It prevents a term being lifted
        from the slice unannounced.
        """
        from omni_mercury_engine.cognitive.ethical_bounding import (
            _HAZARD_CONTEXT_ANCHORS,
        )

        texts = [r.text.lower() for r in corpus]
        declared = set(ADDED_AFTER_HELD_OUT_MEASUREMENT) | set(PRE_EXISTING_HAZARD_VOCABULARY)
        overlapping = {
            anchor for anchor in _HAZARD_CONTEXT_ANCHORS if any(anchor.lower() in t for t in texts)
        }
        undeclared = overlapping - declared
        assert not undeclared, (
            "hazard-context anchors occur in the held-out corpus but are "
            f"declared in neither bucket: {sorted(undeclared)}. If a term was "
            "added in response to measuring this slice, put it in "
            "ADDED_AFTER_HELD_OUT_MEASUREMENT and say so where the recall "
            "figures are published -- the slice is no longer held out for it. "
            "If it is ordinary hazard vocabulary that pre-dates the slice, put "
            "it in PRE_EXISTING_HAZARD_VOCABULARY."
        )
