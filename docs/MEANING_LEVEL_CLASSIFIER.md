# Mercury's Offline Meaning-Level Harm Classifier

Applies to Mercury Agent **v2.1.x**. Added 2026-08-04.

## The gap this closes

The weapons/mass-casualty gate (`docs/HARM_POLICY.md`) routes on two lexical
axes: Axis A matches a hazard-domain vocabulary, Axis B matches actionable-intent
patterns. For everything the lexicons cannot decide, the gate consults a
*meaning-level* `harm_classifier` hook.

Until now the only thing that could serve that hook was a **generative model** —
Ollama, or a cloud backend. Everywhere no model was running, the hook returned
`0.0` and the gate ran lexical-only. That is CI, every air-gapped deployment,
and every default install. Measured on the held-out adversarial slice, the
lexical-only posture missed **99 of 133** offensive requests (FN-rate 0.744).

The reason is structural, not a missing lexicon entry. Axis B detects production
intent as `REQUEST_MARKER + ENUMERATED_VERB`. The request-marker set is a small
closed class, but the act-verb set is an **open class** — `brew`, `cook up`,
`culture`, `refine`, `put together`, `rig`, `parts list and assembly order` — and
no enumeration ever finishes it. Growing the lexicon would be memorization, and
the project has deliberately refused to make the gate pass that way.

This module supplies the missing layer *intrinsically*: a trained classifier that
learns the **request frame** rather than the verb list, ships inside the package,
and needs no model server.

## Result

Held-out adversarial slice (`benchmarks/weapons_gate_adversarial.py`, 163 rows,
133 offensive / 30 hard-benign professional). The slice is never trained on.

| posture | FN | FN-rate | recall | precision | FP |
|---|---|---|---|---|---|
| lexical-only (previous default) | 99 / 133 | 0.744 | 0.256 | 1.00 | 0 |
| **shipped classifier (new default)** | 35 / 133 | **0.263** | **0.737** | **1.00** | **0** |

Per axis, recall rises: paraphrase 0.190 → 0.672, conjunction 0.298 → 0.872,
obfuscation 0.364 → 0.727.

The classifier's own separation on the slice is **AUROC 0.9970**. Its score
distribution has real margin at the gate's 0.600 escalation threshold:

| pool | n | max score | FP at the gate |
|---|---|---|---|
| base-corpus benign | 180 | 0.409 | 0 |
| hard-benign professional | 30 | 0.280 | 0 |
| extra civilian/professional probes | 10 | 0.541 | 0 |

Reproduce:

```bash
PYTHONPATH=src:benchmarks python benchmarks/eval_weapons_gate_adversarial.py --posture default
PYTHONPATH=src:benchmarks python scripts/train_meaning_level_classifier.py --dry-run
PYTHONPATH=src:benchmarks python scripts/train_meaning_level_classifier.py --cross-validate --dry-run
```

## What it is

A binary logistic regression over binary lexical-frame features:

* `w:<token>` — unigram presence
* `b:<t1>_<t2>` — adjacent-pair presence, which is what encodes a frame
  (`walk_me`, `me_through`, `how_to`, `process_for`)
* `p:<prefix>` — the first five characters of a longer token, giving
  inflectional transfer so `refines`/`refining`/`refine` reach one weight

Text is de-obfuscated by Mercury's own `harm_normalization.canonical_normalize`
*before* tokenizing, so leetspeak, homoglyph spoofing and per-character spacing
are folded away and the model never has to learn them.

Properties that make it safe to run at every decision boundary:

* **Stdlib only.** No numpy, no model server, no network call, no new dependency.
* **Deterministic.** Weights start at zero, the optimizer is full-batch gradient
  descent, and the corpus generator uses no RNG. There is nothing to seed; the
  same source always yields byte-identical weights.
* **Fast.** ~18–25 µs per query. Adds ~40 µs to a hazard-routed gate call and
  ~1 µs to a benign one (the rescue path is not reached for most benign text).
* **Auditable.** The artifact is plain JSON mapping human-readable feature
  strings to weights (1931 features, 57 KiB). `MeaningLevelModel.explain()`
  returns the top contributing features for any input, so "why did it score
  that" is answerable without retraining.
* **Fail-open.** A missing, corrupt, empty, or version-mismatched artifact
  yields `0.0` — exactly the pre-existing lexical-only behaviour. A broken model
  is a loss of the improvement, never a safety regression.

## Contract: consulted, never trusted

Unchanged from the existing design, and the reason a small linear model is
sufficient here rather than alarming:

* The gate combines the classifier by `max`. It can only ever **raise** a
  disposition, never lower one earned by lexical evidence.
* The routing rescue raises to **ESCALATE** — human-in-the-loop review — never
  to an automatic refusal on classifier-alone evidence.
* It is consulted only when a hazard domain has **already routed** and Axis B
  found neither offensive intent nor any professional allow-signal. A
  defensively-framed query never reaches it. On fully-benign, domain-NONE text
  it is not consulted at all.

`MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER=1` turns it off for a strictly lexical,
model-free posture.

## Training corpus and the anti-shortcut properties

`benchmarks/meaning_level_corpus.py` generates 8576 rows compositionally, under
the same transparency constraints as the evaluation corpora: taxonomy-level only,
ground truth is intent rather than topic.

The corpus is built to make the *shortcuts* unavailable, and each property is
asserted at build time rather than hoped for:

1. **Hazard-noun presence must not predict the label.**
   `P(offensive | a hazard noun is present) = 0.500 ± 0.02`, enforced by sizing
   the benign-with-noun groups to match the offensive total.
2. **No single hazard noun may outweigh another.** Sampling is noun-stratified,
   and the offensive-share spread across nouns must stay under 0.10.
3. **Every procedure-seeking frame appears in both classes.** `walk me through`
   is crossed with defensive objects (`... the steps to decontaminate after a
   chlorine gas release`) and with ordinary tasks (`... the steps to change a
   tire`), so the request shape alone is not the label.
4. **No context-leakage modifiers.** A shared neutral modifier pool is applied to
   both classes; marking offensive rows with tell-tale context (`at home`,
   `in bulk`) would let the model learn the modifier instead of the frame.

These are not theoretical. An earlier revision sampled by text-stride rather than
by noun, which left `gas` with a **+0.76** weight (from `chlorine gas`/`mustard
gas` landing more often on the offensive side). That model then false-positived
on *"how do gas centrifuges enrich uranium for reactor fuel"* — a benign
civilian-nuclear question, and exactly the kind of professional query the gate
exists not to block. Properties 1 and 2 were added in response, and the assertion
now fails the build rather than shipping the shortcut.

`build_training_corpus()` also asserts the generated text is **disjoint from
both** `weapons_gate_corpus` (the lexical gate's fit-on corpus) and
`weapons_gate_adversarial` (the held-out slice).

## Generalization, measured independently of the held-out slice

In-sample fit is 1.00 on everything and means nothing for a compositional
corpus — a random split would share frames and verbs with hundreds of training
rows. The real protocol holds out **entire request frames** and **entire act
verbs**, so test rows use a shape or a verb the model has never seen in any row.
That is the property the classifier actually needs, since the act-verb class is
open and cannot be enumerated.

Measured 2026-08-04 on the shipped corpus (`PENDING_CV_TABLE`):

Read honestly: transfer to **unseen request frames is near-perfect**, which is
the load-bearing case, since frames are a closed class the corpus covers well.
Transfer to **unseen act verbs keeps recall high (0.976) but costs precision
(0.730)**. That is a real limitation and worth stating plainly: with a quarter of
the act vocabulary — *including the defensive verbs* — held out simultaneously,
the model falls back on the frame and over-predicts offensive.

Two things bound the practical impact, neither of which excuses it:

* A 4-fold split removes 25% of all verbs at once, which is far harsher than
  meeting one novel verb at a time. The `leave-act-out-fine` (10-fold)
  configuration is recorded in the artifact's provenance block for comparison.
* In the deployed gate the classifier is consulted *only* when Axis B found no
  professional allow-signal, so the professional framings the lexical patterns
  already cover never reach it. The measured false-positive count on the real
  hard-benign professional slice is **0**.

The practical mitigation is that the defensive/professional act inventory in the
training corpus is what buys precision, so it should grow alongside the offensive
one. A future revision that adds offensive verbs without adding professional ones
would show up as a precision drop in this table.

## Known residual

The dominant remaining bypass is **uniform single-space character insertion**
(`m a k e a b o m b`). Word segmentation there is genuinely ambiguous — the
string has no unique reading — so `harm_normalization._rejoin_spaced_words`
declines to guess rather than fabricate one. Axis A still routes such a query via
the collapsed variant, so it reaches the classifier, but the classifier sees
single-character tokens and has little to work with. Wide-gap spacing (how a
human actually writes a spaced-out phrase) is fully recovered.

The residual 35 held-out false negatives are dominated by **out-of-lexicon agent
names** — `VX`, `fourth-generation agent`, `binary chemical munition` — where
Axis A does not route at all, so the rescue is never consulted by design (with no
hazard evidence the classifier would be the sole signal, which is the least safe
place to trust it). Closing that requires Axis-A routing coverage, not a better
classifier.

## Retraining

```bash
PYTHONPATH=src:benchmarks python scripts/train_meaning_level_classifier.py
```

Rewrites `src/omni_mercury_engine/cognitive/meaning_level_weights.json`. The
artifact records its own provenance: corpus composition, hyperparameters, feature
counts, fit metrics and the cross-validation block.

Bump `meaning_level.FEATURE_VERSION` whenever the feature extractor changes.
`load_model` refuses a version-mismatched artifact rather than scoring with
features the model was never fit on.

## Gates

`tests/ethical/test_weapons_gate_adversarial_eval.py`:

* `test_meaning_level_coverage_is_always_available` — the shipped model must
  load. This lane previously **skipped** whenever no operator model was
  configured, so the meaning-level budget was never measured on a normal PR.
* `test_meaning_level_fn_budget` — held-out FN-rate ≤ 0.35 with FP 0.
* `test_shipped_classifier_beats_the_lexical_floor` — the model must measurably
  cut false negatives (≤ 0.30 FN-rate), not merely be present. A weights file
  that loaded but had no discriminative power would otherwise satisfy the
  availability assertion while changing nothing.

`tests/ethical/test_meaning_level_classifier.py` pins fail-open behaviour at
every failure mode, determinism, the raise-only contract, the default wiring at
each boundary, and the corpus disjointness and balance properties.
