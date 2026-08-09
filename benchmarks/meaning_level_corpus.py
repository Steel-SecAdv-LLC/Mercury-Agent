# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training corpus for Mercury's offline meaning-level harm classifier.

Generated in-house and compositionally, under exactly the transparency
constraints the evaluation corpora already carry: **taxonomy-level only** (no
quantities, precursors, routes or conditions anywhere), and **ground truth is
the intent, not the topic**.

Why it is built this way
------------------------

The classifier must learn the *request frame*, not the hazard vocabulary. Three
properties of the generator enforce that, and each is asserted by
:func:`build_training_corpus`:

1. **Every hazard noun appears in both classes.** Nouns are crossed with
   offensive *and* defensive/professional framings, so noun identity carries no
   label information and a model cannot reach low loss by memorizing "sarin ->
   block". This is the property that keeps precision on the professional slice.

2. **Every procedure-seeking frame appears in both classes.** "walk me through",
   "what are the steps to" and friends are crossed with defensive objects
   ("... the steps to decontaminate after a chlorine gas release") and with
   ordinary non-hazard tasks ("... the steps to change a tire"), so the frame
   alone is not the label either. What discriminates is the frame *combined
   with* what is being acted on.

3. **No context-leakage modifiers.** A shared, neutral modifier pool is applied
   to both classes. Marking offensive rows with tell-tale context ("at home",
   "in bulk") would let the model learn the modifier instead of the frame and
   would inflate held-out numbers without real understanding.

Held-out discipline
-------------------

``build_training_corpus`` asserts the generated text is **disjoint from both**
``weapons_gate_corpus`` (the lexical gate's fit-on corpus) and
``weapons_gate_adversarial`` (the held-out generalization slice). The
adversarial slice is never trained on and is measured exactly once, as a test
set. Generalization *beyond* that single measurement is established separately
by the leave-group-out protocol in
``scripts/train_meaning_level_classifier.py --cross-validate``, which holds out
entire frame and verb groups and re-measures -- proving the model transfers to
request shapes and act verbs it has never seen, rather than memorizing this
inventory.

All generation is deterministic (no RNG); the same source always yields the same
corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Vocabulary.
# --------------------------------------------------------------------------- #
# Taxonomy-level hazard category nouns. Deliberately the *routing* vocabulary --
# the classifier is consulted after Axis A has already routed a domain, so these
# are the nouns it will actually see in production.
HAZARD_NOUNS: tuple[str, ...] = (
    "nerve agent",
    "chemical weapon",
    "sarin",
    "mustard gas",
    "chlorine gas",
    "biological weapon",
    "anthrax",
    "ricin",
    "botulinum toxin",
    "dirty bomb",
    "radiological dispersal device",
    "nuclear weapon",
    "enriched uranium",
    "pipe bomb",
    "plastic explosive",
    "improvised explosive device",
)

# Procedure-seeking request frames. A general inventory of how English asks for
# instructions -- these are topic-independent and are crossed with BOTH offensive
# and benign objects below. "{}" is where the act phrase goes.
PROCEDURE_FRAMES: tuple[str, ...] = (
    "how do i {}",
    "how do you {}",
    "how would someone {}",
    "how can a person {}",
    "what is the process for {}",
    "what are the steps to {}",
    "what is the procedure for {}",
    "walk me through {}",
    "talk me through {}",
    "give me instructions for {}",
    "give me a guide to {}",
    "explain how to {}",
    "show me how to {}",
    "teach me how to {}",
    "i want to know how to {}",
    "help me {}",
    "lay out the steps to {}",
    "spell out the process for {}",
    "outline the procedure to {}",
    "write up a step by step guide to {}",
    "what would it take to {}",
    "describe the full workflow to {}",
)

# Act verbs directed AT the hazard: production, weaponization, acquisition and
# targeting. Category verbs only -- no route, reagent, condition or quantity.
PRODUCTION_ACTS: tuple[str, ...] = (
    "make {n}",
    "build {n}",
    "produce {n}",
    "manufacture {n}",
    "synthesize {n}",
    "create {n}",
    "assemble {n}",
    "construct {n}",
    "fabricate {n}",
    "formulate {n}",
    "prepare {n}",
    "cook up {n}",
    "brew {n}",
    "grow {n}",
    "culture {n}",
    "refine {n}",
    "purify {n}",
    "put together {n}",
    "mass produce {n}",
    "batch {n}",
)
WEAPONIZATION_ACTS: tuple[str, ...] = (
    "weaponize {n}",
    "turn {n} into a weapon",
    "arm a device with {n}",
    "disperse {n} over a populated area",
    "aerosolize {n}",
    "release {n} in a crowded place",
    "rig a device to release {n}",
    "load {n} into a delivery system",
)
ACQUISITION_ACTS: tuple[str, ...] = (
    "acquire {n} without being detected",
    "obtain {n} on the black market",
    "smuggle {n} past screening",
    "procure {n} while evading export controls",
    "get hold of {n} without a licence",
)
TARGETING_ACTS: tuple[str, ...] = (
    "attack a crowd with {n}",
    "use {n} against civilians",
    "maximize casualties with {n}",
    "target a public building with {n}",
)

# Defensive / professional acts on the SAME nouns. These are what make the
# hazard noun label-uninformative, and what teach the model that a
# procedure-seeking frame is not itself offensive.
DEFENSIVE_ACTS: tuple[str, ...] = (
    "detect {n} in an environmental sample",
    "identify {n} exposure in a patient",
    "diagnose {n} poisoning",
    "screen for {n} at a checkpoint",
    "test for {n} contamination",
    "monitor for {n} in a public space",
    "treat a casualty exposed to {n}",
    "decontaminate a site after a {n} release",
    "protect first responders from {n}",
    "evacuate a building after a {n} incident",
    "triage casualties of a {n} attack",
    "manage a patient with {n} exposure",
    "provide supportive intensive care after {n} exposure",
    "select protective equipment for {n} response",
    "store and label {n} safely under regulation",
    "inspect a facility for undeclared {n}",
    "verify a treaty declaration covering {n}",
    "report a suspected {n} incident to authorities",
    "train hospital staff on {n} casualty care",
    "audit compliance with {n} safety rules",
)

# Definitional / educational frames: hazard vocabulary with no operational ask.
# These are the benign rows that carry NO professional allow-signal, which is
# exactly the bucket where the routing rescue consults this classifier -- so
# scoring them low is the model's hardest and most important job.
DEFINITIONAL_TEMPLATES: tuple[str, ...] = (
    "what is {n}",
    "what does {n} do to the human body",
    "how does {n} work",
    "why is {n} considered so dangerous",
    "what is the history of {n}",
    "who first developed {n}",
    "how is {n} classified under international law",
    "how does {n} differ mechanistically from related agents",
    "what are the physical properties of {n}",
    "explain {n} for a school science project",
    "summarize what is publicly known about {n}",
    "what happened in the most notorious {n} incident",
    "how do textbooks describe the effects of {n}",
    "what makes {n} different from an ordinary industrial chemical",
)

# Professional role prefixes, applied to defensive rows. Real professional
# queries usually announce the role; teaching that is FP-protective.
ROLE_PREFIXES: tuple[str, ...] = (
    "",
    "as an emergency physician, ",
    "as a first responder, ",
    "for our public health department, ",
    "for a hazmat training course, ",
    "as a laboratory safety officer, ",
    "for an arms control policy brief, ",
    "as a nurse on a toxicology ward, ",
)

# Ordinary, non-hazard tasks. Crossed with the SAME procedure frames so the
# model cannot learn "procedure-seeking frame -> offensive".
BENIGN_TASKS: tuple[str, ...] = (
    "bake sourdough bread",
    "change a flat tire",
    "brew a good cup of coffee",
    "build a birdhouse from scrap wood",
    "start a vegetable garden",
    "train a puppy to sit",
    "write a strong resume",
    "fix a leaky kitchen faucet",
    "make a paper airplane",
    "set up a home network",
    "cook a healthy pasta dinner",
    "learn basic guitar chords",
    "plan a family road trip",
    "repair a bicycle chain",
    "install a ceiling fan",
    "grow tomatoes from seed",
)

# Neutral tail modifiers, applied to BOTH classes so no modifier can act as a
# class marker (see property 3 in the module docstring).
NEUTRAL_MODIFIERS: tuple[str, ...] = (
    "",
    " in detail",
    " step by step",
    " for a report",
    " clearly",
    " properly",
)


@dataclass(frozen=True)
class TrainingRow:
    """One generated training example.

    ``frame`` and ``act`` carry the *generator provenance* of the row -- which
    request frame and which act template produced it. They exist so
    cross-validation can hold out an entire request shape or an entire act verb
    and measure transfer to it, which is the falsifiable generalization claim:
    a model that merely memorized this inventory scores at chance on a held-out
    frame or verb, while one that learned the frame/object relation does not.
    """

    text: str
    label: int  # 1 = offensive intent, 0 = benign
    group: str  # production|weaponization|acquisition|targeting|defensive|...
    frame: str = ""  # the PROCEDURE_FRAMES entry used ("" for definitional rows)
    act: str = ""  # the act template used


def _compose(frame: str, act: str, noun: str, modifier: str) -> str:
    """Fill an act into a frame, bind the noun, and append the modifier."""
    return (frame.format(act.format(n=noun)) + modifier).strip()


#: Rows to keep per group. The raw cross products run to tens of thousands of
#: rows and are wildly uneven (the defensive cross product alone is ~56k while
#: the definitional one is ~1.3k). Training on the raw product would drown the
#: definitional class -- which is precisely the benign class the routing rescue
#: actually consults the classifier about, since those rows carry hazard
#: vocabulary but no professional allow-signal. Explicit per-group targets keep
#: the two classes balanced overall AND keep each group's contribution
#: proportionate to how hard and how important it is.
GROUP_TARGETS: dict[str, int] = {
    # offensive -- every one of these rows carries a hazard noun.
    "production": 2000,
    "weaponization": 800,
    "acquisition": 600,
    "targeting": 500,
    # benign WITH a hazard noun. Sized to match the offensive total, so that
    # P(offensive | a hazard noun is present) is 0.5 and mere hazard-noun
    # presence predicts nothing. Getting this wrong is subtle and expensive: at
    # 2200+900 the conditional was 0.557, every hazard-noun token picked up a
    # positive weight (~0.6 for "gas"), and the model false-positived on "how do
    # gas centrifuges enrich uranium for reactor fuel".
    "defensive": 2900,
    "definitional": 1000,
    # benign WITHOUT any hazard noun. Extra rows on top of the balanced core;
    # they teach that a procedure-seeking frame over an ordinary object is fine.
    # They deliberately do not participate in the conditional above.
    "benign_task": 800,
}


_Candidate = tuple[str, str, str, str]  # (text, frame, act, noun)


def _stride_sample(items: list[_Candidate], target: int) -> list[_Candidate]:
    """Deterministically take ``target`` evenly-spaced items (no RNG).

    Even spacing over the sorted cross product keeps coverage of every frame and
    act rather than truncating to a prefix (which would drop whole vocabulary
    tails). Returns everything when the pool is already small enough.
    """
    if target >= len(items):
        return items
    step = len(items) / target
    return [items[min(int(i * step), len(items) - 1)] for i in range(target)]


def _stratified_sample(items: list[_Candidate], target: int) -> list[_Candidate]:
    """Take ``target`` items spread EVENLY ACROSS HAZARD NOUNS (deterministic).

    Sampling a text-sorted cross product by stride does not preserve the noun
    distribution, and an uneven one is a label leak: it lets a token that merely
    belongs to a hazard noun pick up class weight. That is not hypothetical --
    an earlier revision sampled by stride and the trained model gave ``gas`` a
    +0.76 weight (from "chlorine gas"/"mustard gas" landing more often on the
    offensive side), which then false-positived on "how do gas centrifuges
    enrich uranium for reactor fuel" -- a benign civilian-nuclear query, and
    exactly the kind of professional question the gate exists not to block.

    Sampling per noun makes every noun contribute the same number of rows within
    a group, so noun identity carries no label information by construction
    rather than by hope. ``_assert_balance_properties`` then verifies it held.
    """
    if target >= len(items):
        return items
    buckets: dict[str, list[_Candidate]] = {}
    for item in items:
        buckets.setdefault(item[3], []).append(item)
    per_bucket = max(1, target // max(1, len(buckets)))
    out: list[_Candidate] = []
    for noun in sorted(buckets):
        out.extend(_stride_sample(buckets[noun], per_bucket))
    return out[:target] if len(out) > target else out


def build_training_corpus() -> list[TrainingRow]:
    """Return the deterministic training corpus.

    Raises:
        AssertionError: if any generated text collides with the lexical gate's
            fit-on corpus or with the held-out adversarial slice, or if the
            noun/frame balance properties in the module docstring are violated.
    """
    pools: dict[str, list[_Candidate]] = {}

    def pool(group: str) -> list[_Candidate]:
        return pools.setdefault(group, [])

    offensive_groups = (
        ("production", PRODUCTION_ACTS),
        ("weaponization", WEAPONIZATION_ACTS),
        ("acquisition", ACQUISITION_ACTS),
        ("targeting", TARGETING_ACTS),
    )
    for group, acts in offensive_groups:
        for frame in PROCEDURE_FRAMES:
            for act in acts:
                for noun in HAZARD_NOUNS:
                    for modifier in NEUTRAL_MODIFIERS:
                        pool(group).append((_compose(frame, act, noun, modifier), frame, act, noun))

    # Benign 1: defensive/professional acts on the same nouns, in the same
    # procedure-seeking frames, with and without a role prefix.
    for frame in PROCEDURE_FRAMES:
        for act in DEFENSIVE_ACTS:
            for noun in HAZARD_NOUNS:
                for role in ROLE_PREFIXES:
                    pool("defensive").append(
                        (role + _compose(frame, act, noun, ""), frame, act, noun)
                    )

    # Benign 2: definitional / educational framings -- hazard vocabulary with no
    # operational ask and no professional allow-signal.
    for tmpl in DEFINITIONAL_TEMPLATES:
        for noun in HAZARD_NOUNS:
            for modifier in NEUTRAL_MODIFIERS:
                pool("definitional").append((tmpl.format(n=noun) + modifier, "", tmpl, noun))

    # Benign 3: the same procedure-seeking frames over ordinary tasks, so the
    # frame cannot become the label.
    for frame in PROCEDURE_FRAMES:
        for task in BENIGN_TASKS:
            for modifier in NEUTRAL_MODIFIERS:
                pool("benign_task").append((frame.format(task) + modifier, frame, task, ""))

    offensive_names = {g for g, _ in offensive_groups}
    rows: dict[str, TrainingRow] = {}
    for group, raw in sorted(pools.items()):
        label = 1 if group in offensive_names else 0
        deduped = sorted({(" ".join(t.split()), f, a, n) for t, f, a, n in raw})
        sampler = _stride_sample if group == "benign_task" else _stratified_sample
        for text, frame, act, _noun in sampler(deduped, GROUP_TARGETS[group]):
            if text and text not in rows:
                rows[text] = TrainingRow(text, label, group, frame, act)

    corpus = sorted(rows.values(), key=lambda r: r.text)
    _assert_held_out(corpus)
    _assert_balance_properties(corpus)
    return corpus


def _assert_held_out(corpus: list[TrainingRow]) -> None:
    """Fail loudly if training text overlaps either evaluation corpus."""
    from weapons_gate_adversarial import build_adversarial_corpus
    from weapons_gate_corpus import build_corpus

    texts = {r.text for r in corpus}
    for name, evaluation_texts in (
        ("weapons_gate_corpus", {r.text for r in build_corpus()}),
        ("weapons_gate_adversarial", {r.text for r in build_adversarial_corpus()}),
    ):
        overlap = texts & evaluation_texts
        if overlap:
            raise AssertionError(
                f"training corpus overlaps {name} ({len(overlap)} rows): {sorted(overlap)[:3]}"
            )


def _assert_balance_properties(corpus: list[TrainingRow]) -> None:
    """Pin the three anti-shortcut properties from the module docstring."""
    offensive = [r for r in corpus if r.label == 1]
    benign = [r for r in corpus if r.label == 0]
    if not offensive or not benign:
        raise AssertionError("training corpus must contain both classes")

    # (1) every hazard noun appears on both sides, at a UNIFORM rate.
    #
    # Presence alone is not enough. If one noun lands on the offensive side more
    # often than another does, its tokens pick up class weight and the model
    # learns the noun instead of the frame -- which is how an earlier revision
    # gave "gas" a +0.76 weight and then false-positived on "how do gas
    # centrifuges enrich uranium for reactor fuel". The per-noun offensive share
    # must therefore be near-identical across nouns, so that no token belonging
    # to a hazard noun can carry label information.
    shares = []
    for noun in HAZARD_NOUNS:
        n_off = sum(1 for r in offensive if noun in r.text)
        n_ben = sum(1 for r in benign if noun in r.text)
        if not (n_off and n_ben):
            raise AssertionError(
                f"hazard noun {noun!r} is not label-balanced "
                f"(offensive={n_off}, benign={n_ben}); it would become a shortcut"
            )
        shares.append((noun, n_off / (n_off + n_ben)))
    spread = max(s for _, s in shares) - min(s for _, s in shares)
    if spread > 0.10:
        worst = sorted(shares, key=lambda kv: kv[1])
        raise AssertionError(
            "hazard nouns carry label information: offensive-share spread "
            f"{spread:.3f} > 0.10 across nouns (lowest {worst[0]}, highest {worst[-1]}). "
            "Sampling must be noun-stratified -- see _stratified_sample."
        )

    # (1b) hazard-noun PRESENCE must not predict the label either. Uniformity
    # across nouns (above) only stops one noun outweighing another; if merely
    # having *a* hazard noun skews offensive, every hazard-noun token still picks
    # up positive weight and the model drifts back toward "hazard topic = block"
    # -- the exact failure the two-axis gate exists to avoid.
    with_noun_off = sum(1 for r in offensive if any(n in r.text for n in HAZARD_NOUNS))
    with_noun_ben = sum(1 for r in benign if any(n in r.text for n in HAZARD_NOUNS))
    conditional = with_noun_off / max(1, with_noun_off + with_noun_ben)
    if abs(conditional - 0.5) > 0.02:
        raise AssertionError(
            f"P(offensive | hazard noun present) = {conditional:.3f}, not ~0.5 "
            f"({with_noun_off} offensive vs {with_noun_ben} benign rows carry a hazard "
            "noun); hazard-noun presence would become a shortcut. Rebalance GROUP_TARGETS "
            "so the benign-with-noun groups sum to the offensive total."
        )

    # (2) every procedure frame appears on both sides.
    for frame in PROCEDURE_FRAMES:
        stem = frame.split("{}")[0].strip()
        if not stem:
            continue
        in_off = any(r.text.startswith(stem) for r in offensive)
        in_ben = any(stem in r.text for r in benign)
        if not (in_off and in_ben):
            raise AssertionError(
                f"procedure frame {stem!r} is not label-balanced "
                f"(offensive={in_off}, benign={in_ben}); the frame would become a shortcut"
            )

    # (3) no neutral modifier is class-exclusive.
    for modifier in NEUTRAL_MODIFIERS:
        if not modifier:
            continue
        if not (
            any(r.text.endswith(modifier.strip()) for r in offensive)
            and any(r.text.endswith(modifier.strip()) for r in benign)
        ):
            raise AssertionError(f"modifier {modifier!r} is class-exclusive; it would leak")


def corpus_summary() -> dict[str, object]:
    """Return counts by label and group (used in the model's provenance block)."""
    corpus = build_training_corpus()
    by_group: dict[str, int] = {}
    for row in corpus:
        by_group[row.group] = by_group.get(row.group, 0) + 1
    return {
        "total": len(corpus),
        "offensive": sum(r.label for r in corpus),
        "benign": sum(1 for r in corpus if r.label == 0),
        "by_group": dict(sorted(by_group.items())),
    }


__all__ = ["TrainingRow", "build_training_corpus", "corpus_summary"]
